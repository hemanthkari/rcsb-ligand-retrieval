"""
Build a TARGET-DISJOINT train/val/test split (audit finding E-3 fix).

The shipped "canonical" split is a naive random permutation, so near-identical
same-target complexes (the paper's own Table 3 shows a cos-0.9997 cluster, and
2.19% of joined embeddings are exact duplicates) land on both sides. Any model
scored on it — especially a retrieval-augmented one — is measuring memorization.

This clusters complexes by PROTEIN SEQUENCE with MMseqs2 at ~30% identity and
assigns WHOLE clusters to train/val/test, so no target family crosses the split
boundary. Entries with no resolvable sequence become singleton clusters.

Outputs under <retrieval_dir>/splits/:
    train_target_disjoint.npy   int64 metadata indices
    val_target_disjoint.npy     int64 metadata indices
    test_target_disjoint.npy    int64 metadata indices
    target_disjoint_info.json   params + counts + a disjointness assertion result

Requires MMseqs2 on PATH (conda: `conda install -c bioconda mmseqs2`).

Usage:
    python scripts/build_target_disjoint_split.py --retrieval_dir data/retrieval_index \
        --min_seq_id 0.3 --coverage 0.8 --val_frac 0.05 --test_frac 0.05 --seed 42
    python scripts/build_target_disjoint_split.py --self_test   # logic check, no data/MMseqs2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def read_sequences(metadata_path: Path):
    """Return (indices, seqs): protein_seq per metadata row ('' if missing)."""
    idx, seqs = [], []
    with open(metadata_path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            s = row.get("protein_seq") or ""
            idx.append(i)
            seqs.append(s.strip())
    return idx, seqs


def run_mmseqs_cluster(seqs, min_seq_id: float, coverage: float, mmseqs: str = "mmseqs") -> dict:
    """Cluster the non-empty sequences with MMseqs2 easy-cluster.

    Returns cluster_of_index: {metadata_index -> cluster_id (int)}. Entries with an
    empty sequence get their OWN singleton cluster (can't leak; nothing to align).
    """
    if shutil.which(mmseqs) is None:
        raise SystemExit(
            "MMseqs2 not found on PATH (looked for %r). Install: conda install -c bioconda mmseqs2"
            % mmseqs
        )
    tmp = Path(tempfile.mkdtemp(prefix="mmseqs_"))
    try:
        fasta = tmp / "in.fasta"
        with open(fasta, "w") as fh:
            for i, s in enumerate(seqs):
                if s:
                    fh.write(">idx_%d\n%s\n" % (i, s))
        out_prefix = tmp / "clu"
        cmd = [
            mmseqs,
            "easy-cluster",
            str(fasta),
            str(out_prefix),
            str(tmp / "work"),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
        ]
        print("[mmseqs] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        # <prefix>_cluster.tsv: rows "<rep_id>\t<member_id>"
        rep_to_cid: dict = {}
        cluster_of_index: dict = {}
        with open(str(out_prefix) + "_cluster.tsv") as fh:
            for line in fh:
                rep, member = line.rstrip("\n").split("\t")
                cid = rep_to_cid.setdefault(rep, len(rep_to_cid))
                cluster_of_index[int(member.split("_")[1])] = cid
        # sequenceless entries -> unique singleton clusters
        next_cid = len(rep_to_cid)
        for i, s in enumerate(seqs):
            if not s and i not in cluster_of_index:
                cluster_of_index[i] = next_cid
                next_cid += 1
        return cluster_of_index
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def assign_clusters_to_splits(cluster_of_index: dict, val_frac: float, test_frac: float, seed: int):
    """Assign WHOLE clusters to train/val/test. Returns (train, val, test) index lists.

    Guarantees no cluster crosses a split boundary (asserted). Fractions are
    approximate (whole-cluster granularity).
    """
    # group indices by cluster
    members: dict = {}
    for i, c in cluster_of_index.items():
        members.setdefault(c, []).append(int(i))
    clusters = list(members.items())  # (cid, [idx...])
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(clusters))
    clusters = [clusters[k] for k in order]

    n_total = sum(len(m) for _, m in clusters)
    val_target = val_frac * n_total
    test_target = test_frac * n_total
    train, val, test = [], [], []
    acc_val = acc_test = 0
    for _cid, m in clusters:
        if acc_val < val_target:
            val.extend(m)
            acc_val += len(m)
        elif acc_test < test_target:
            test.extend(m)
            acc_test += len(m)
        else:
            train.extend(m)

    # hard disjointness assertion: no cluster id appears in more than one split
    def cids(ixs):
        return {cluster_of_index[i] for i in ixs}

    ctr, cv, cte = cids(train), cids(val), cids(test)
    assert not (ctr & cv) and not (ctr & cte) and not (cv & cte), "cluster crosses split boundary!"
    assert len(train) + len(val) + len(test) == n_total
    assert len(set(train) | set(val) | set(test)) == n_total, "index appears twice"
    return sorted(train), sorted(val), sorted(test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_dir", type=Path)
    ap.add_argument("--min_seq_id", type=float, default=0.3)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--self_test", action="store_true", help="Run assignment-logic check and exit.")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.retrieval_dir is None:
        sys.exit("--retrieval_dir is required (or pass --self_test)")

    meta_path = args.retrieval_dir / "metadata.jsonl"
    idx, seqs = read_sequences(meta_path)
    n_seq = sum(1 for s in seqs if s)
    print("Loaded %d metadata rows (%d with a sequence)" % (len(idx), n_seq), flush=True)

    cluster_of_index = run_mmseqs_cluster(seqs, args.min_seq_id, args.coverage, args.mmseqs)
    n_clusters = len(set(cluster_of_index.values()))
    print("MMseqs2: %d clusters over %d entries" % (n_clusters, len(cluster_of_index)), flush=True)

    train, val, test = assign_clusters_to_splits(
        cluster_of_index, args.val_frac, args.test_frac, args.seed
    )
    print("Split: %d train / %d val / %d test" % (len(train), len(val), len(test)), flush=True)

    out = args.retrieval_dir / "splits"
    out.mkdir(exist_ok=True)
    np.save(out / "train_target_disjoint.npy", np.array(train, dtype=np.int64))
    np.save(out / "val_target_disjoint.npy", np.array(val, dtype=np.int64))
    np.save(out / "test_target_disjoint.npy", np.array(test, dtype=np.int64))
    info = {
        "method": "MMseqs2 easy-cluster, whole-cluster assignment (target-disjoint)",
        "min_seq_id": args.min_seq_id,
        "coverage": args.coverage,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "n_total": len(cluster_of_index),
        "n_clusters": n_clusters,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "disjoint_asserted": True,
    }
    (out / "target_disjoint_info.json").write_text(json.dumps(info, indent=2))
    print("Wrote splits + target_disjoint_info.json to %s" % out)


def _self_test() -> None:
    """Synthetic clusters -> verify disjointness, coverage, approx fractions."""
    rng = np.random.RandomState(1)
    # 500 entries in 60 clusters of uneven size
    cluster_of_index = {}
    i = 0
    for c in range(60):
        size = int(rng.randint(1, 20))
        for _ in range(size):
            cluster_of_index[i] = c
            i += 1
    n = len(cluster_of_index)
    train, val, test = assign_clusters_to_splits(cluster_of_index, 0.1, 0.1, seed=42)
    # coverage + disjoint index sets
    assert len(train) + len(val) + len(test) == n
    assert (
        not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    )

    # cluster disjointness (the whole point)
    def cl(ixs):
        return {cluster_of_index[j] for j in ixs}

    assert not (cl(train) & cl(val)) and not (cl(train) & cl(test)) and not (cl(val) & cl(test))
    print(
        "build_target_disjoint_split self-test: OK  "
        "(n=%d -> %d train / %d val / %d test; clusters fully disjoint; "
        "val~%.0f%% test~%.0f%%)"
        % (n, len(train), len(val), len(test), 100 * len(val) / n, 100 * len(test) / n)
    )


if __name__ == "__main__":
    main()
