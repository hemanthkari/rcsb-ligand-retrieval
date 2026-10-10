"""
Per-chain re-embedding of pocket representations for the v0.2 release.

WHY THIS EXISTS
---------------
The v0.1 pocket embeddings were produced by concatenating every polymer chain of
model 1 into a single ESM-2 input, with no separator tokens, and pooling the
final-layer representations at the pocket residue positions. Two controlled
experiments established that this materially degrades the representation:

  * re-embedding the pocket-bearing chain alone and comparing with the released
    embedding gives median cosine 0.814, against a random-pair floor of 0.706 in
    the same space -- and the median entry has ALL of its pocket residues inside
    one chain, so this is not explained by dropped interface residues;
  * same-UniProt hit@1 measured on 4,000 stratified complexes is 0.588 with the
    concatenated input versus 0.620 per-chain (+3.25 pp), and 0.539 versus 0.582
    (+4.23 pp) restricted to multi-chain entries.

WHAT THIS DOES DIFFERENTLY FROM THE CONTROL EXPERIMENT
------------------------------------------------------
The control embedded only the *pocket-bearing* chain and discarded pocket
residues located in other chains, which affects the 21.1% of pockets that span
chains. It won anyway. This script does the strictly better thing:

    every chain that contributes at least one pocket residue is encoded
    SEPARATELY, and each pocket residue is pooled from ITS OWN chain's
    representation.

That removes the fabricated inter-chain adjacencies while retaining every pocket
residue, so it should exceed the +3.25 pp measured for the control variant.

CORRECTNESS GATE
----------------
For every pocket residue we recompute its position within its own chain and
assert that the residue name at that position matches the name recorded in
`pocket_residue_map.parquet` (itself verified against the released coordinates
to 99.99% exactness). Any entry failing the gate is skipped and counted -- never
silently mis-pooled, because a wrong index yields a plausible-looking but wrong
embedding.

ROW ALIGNMENT
-------------
Output rows are written at the SAME offsets as the released
`pockets/pocket_embeddings.npy`, so every downstream artifact that indexes it
(metadata offsets, splits, manifests) keeps working unchanged.

Note on coverage: because the v0.1 1,022-residue cap applied to the SUM over
chains, every entry that survived it necessarily has all individual chains under
the cap. Per-chain encoding therefore excludes nothing that was previously
included. (It would also admit large multimers that the sum-cap rejected; that
is a coverage extension, deliberately out of scope here so that v0.2 row
alignment is preserved.)

USAGE (Colab)
-------------
    !python scripts/reembed_per_chain.py \
        --retrieval_dir /content/snapshot \
        --residue_map /content/drive/MyDrive/fluxion_reextract/pocket_residue_map.parquet \
        --out_dir /content/drive/MyDrive/fluxion_v02 \
        --shard_dir /content/drive/MyDrive/fluxion_v02/embed_shards \
        --cif_dir /content/drive/MyDrive/pdb_mmcif        # optional local mirror

    python scripts/reembed_per_chain.py --self_test       # no data, no GPU
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

MAX_SEQ_LEN = 1022
POCKET_RADIUS = 8.0

# Must match rcsb_ligand_retrieval.retrieval.embed_pockets._THREE_TO_ONE exactly.
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


# ---------------------------------------------------------------------------
# Per-chain sequence construction
# ---------------------------------------------------------------------------


def per_chain_sequences(structure):
    """Return {chain_name: (sequence, [(seqid, icode, resname3), ...])}.

    Mirrors the residue filter used by the deposited `_extract_pocket` --- first
    model only, polymer residues only, resolvable through the 22-code table ---
    but keeps each chain separate instead of concatenating them.
    """
    import gemmi

    out = {}
    for model in structure:
        for chain in model:
            chars, meta = [], []
            for residue in chain:
                if residue.entity_type != gemmi.EntityType.Polymer:
                    continue
                one = THREE_TO_ONE.get(residue.name.upper())
                if one is None:
                    continue
                chars.append(one)
                meta.append(
                    (int(residue.seqid.num), str(residue.seqid.icode or "").strip(), residue.name)
                )
            if chars:
                out[chain.name] = ("".join(chars), meta)
        break  # first model only
    return out


def map_pocket_to_chains(provenance, chains):
    """Resolve each pocket residue to (chain, index within that chain's sequence).

    `provenance` is the list of dicts stored in the residue map. Returns
    (assignments, n_unresolved) where assignments is a list of (chain, idx)
    aligned with `provenance`; unresolved entries appear as None.
    """
    lookup = {}
    for cname, (_seq, meta) in chains.items():
        for i, (num, icode, resname) in enumerate(meta):
            lookup[(cname, num, icode)] = (i, resname)

    assignments, unresolved = [], 0
    for p in provenance:
        key = (p["chain"], int(p["seqid"]), str(p.get("icode") or "").strip())
        hit = lookup.get(key)
        if hit is None:
            assignments.append(None)
            unresolved += 1
            continue
        idx, resname = hit
        # CORRECTNESS GATE: the residue at the recovered index must be the
        # residue the verified map says it is.
        if resname.upper() != str(p["resname"]).upper():
            assignments.append(None)
            unresolved += 1
            continue
        assignments.append((p["chain"], idx))
    return assignments, unresolved


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class Embedder:
    """ESM-2 wrapper. fp16 on GPU to match the original build."""

    def __init__(self, model_name="esm2_t33_650M_UR50D", device=None):
        import torch
        import esm

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device).eval()
        if self.device.startswith("cuda"):
            self.model = self.model.half()
        self.bc = alphabet.get_batch_converter()
        self.layer = self.model.num_layers
        self.dim = self.model.embed_dim

    def chain_reprs(self, seqs):
        """seqs: list of (name, sequence). Returns {name: (L, D) float32 array}."""
        torch = self.torch
        _, _, toks = self.bc(seqs)
        with torch.no_grad():
            out = self.model(toks.to(self.device), repr_layers=[self.layer], return_contacts=False)
        reps = out["representations"][self.layer]
        res = {}
        for b, (name, s) in enumerate(seqs):
            # +1 skips the BOS token; take exactly len(s) positions
            res[name] = reps[b, 1 : len(s) + 1].float().cpu().numpy().astype(np.float32)
        return res


def embed_entry(embedder, chains, assignments, batch_cap=4):
    """Pool each pocket residue from its OWN chain's representation."""
    needed = sorted({c for c in (a[0] for a in assignments if a)})
    if not needed:
        return None
    seqs = [(c, chains[c][0]) for c in needed if len(chains[c][0]) <= MAX_SEQ_LEN]
    if len(seqs) != len(needed):
        return None  # a required chain exceeds the context window
    reprs = {}
    for i in range(0, len(seqs), batch_cap):
        reprs.update(embedder.chain_reprs(seqs[i : i + batch_cap]))
    vecs = [reprs[c][idx] for (c, idx) in (a for a in assignments if a)]
    if not vecs:
        return None
    return np.stack(vecs).mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# mmCIF access
# ---------------------------------------------------------------------------


def load_structure(pdb_id, cif_dir=None):
    import gemmi
    import requests

    pid = pdb_id.lower()
    raw = None
    if cif_dir:
        for cand in (
            Path(cif_dir) / (pid + ".cif.gz"),
            Path(cif_dir) / pid[1:3] / (pid + ".cif.gz"),
        ):
            if cand.exists():
                raw = cand.read_bytes()
                break
    if raw is None:
        r = requests.get("https://files.rcsb.org/download/%s.cif.gz" % pid.upper(), timeout=60)
        if r.status_code != 200:
            return None
        raw = r.content
    tmp = Path("/tmp") if Path("/tmp").exists() else Path(".")
    p = tmp / ("_reembed_%s.cif.gz" % pid)
    p.write_bytes(raw)
    try:
        st = gemmi.read_structure(str(p))
        st.setup_entities()
        return st
    except Exception:
        return None
    finally:
        try:
            p.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args):
    import pandas as pd

    rd = Path(args.retrieval_dir)
    old = np.load(rd / "pockets" / "pocket_embeddings.npy", mmap_mode="r")
    manifest = [json.loads(l) for l in open(rd / "pockets" / "pocket_manifest.jsonl")]
    pk_to_off = {m["pocket_key"]: int(m["offset"]) for m in manifest}

    rmap = pd.read_parquet(args.residue_map)
    print("residue map rows : %d" % len(rmap))
    print("old embeddings   : %s" % (old.shape,))

    shard_dir = Path(args.shard_dir or (Path(args.out_dir) / "embed_shards"))
    shard_dir.mkdir(parents=True, exist_ok=True)

    done = {}
    for sh in sorted(shard_dir.glob("emb_*.npz")):
        z = np.load(sh, allow_pickle=True)
        for k, v in zip(z["keys"], z["vecs"]):
            done[str(k)] = v
    print("resumed          : %d already embedded" % len(done))

    todo = [r for r in rmap.itertuples() if r.pocket_key not in done]
    if args.limit:
        todo = todo[: args.limit]
    print("to process       : %d" % len(todo))
    if not todo:
        print("nothing to do")
    else:
        embedder = Embedder(device=args.device)
        print("embedder ready   : dim=%d device=%s" % (embedder.dim, embedder.device))

    stats = {"ok": 0, "no_cif": 0, "unresolved": 0, "too_long": 0, "no_pocket": 0}
    buf_k, buf_v, shard_n, t0 = [], [], len(list(shard_dir.glob("emb_*.npz"))), time.time()

    for n, r in enumerate(todo, 1):
        st = load_structure(r.pdb_id, args.cif_dir)
        if st is None:
            stats["no_cif"] += 1
            continue
        chains = per_chain_sequences(st)
        prov = json.loads(r.pocket_res_provenance)
        assignments, unres = map_pocket_to_chains(prov, chains)
        if unres or not any(assignments):
            stats["unresolved"] += 1
            continue
        vec = embed_entry(embedder, chains, assignments)
        if vec is None:
            stats["too_long"] += 1
            continue
        buf_k.append(r.pocket_key)
        buf_v.append(vec)
        stats["ok"] += 1

        if n % 200 == 0:
            rate = n / max(1e-9, time.time() - t0)
            print(
                "  %6d/%d  %.1f/s  eta %.0f min  %s"
                % (n, len(todo), rate, (len(todo) - n) / rate / 60, stats),
                flush=True,
            )
        if len(buf_k) >= args.checkpoint_every:
            np.savez_compressed(
                shard_dir / ("emb_%04d.npz" % shard_n),
                keys=np.array(buf_k, dtype=object),
                vecs=np.stack(buf_v),
            )
            shard_n += 1
            buf_k, buf_v = [], []

    if buf_k:
        np.savez_compressed(
            shard_dir / ("emb_%04d.npz" % shard_n),
            keys=np.array(buf_k, dtype=object),
            vecs=np.stack(buf_v),
        )

    # ---- assemble, preserving v0.1 row alignment -------------------------
    done = {}
    for sh in sorted(shard_dir.glob("emb_*.npz")):
        z = np.load(sh, allow_pickle=True)
        for k, v in zip(z["keys"], z["vecs"]):
            done[str(k)] = v

    out = np.array(old, dtype=np.float32)  # start from v0.1
    replaced, missing = 0, []
    for pk, vec in done.items():
        off = pk_to_off.get(pk)
        if off is None:
            continue
        out[off] = vec
        replaced += 1
    for m in manifest:
        if m["pocket_key"] not in done:
            missing.append(m["pocket_key"])

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "pocket_embeddings.npy", out)
    (outdir / "reembed_report.json").write_text(
        json.dumps(
            {
                "n_rows": int(out.shape[0]),
                "n_replaced_per_chain": replaced,
                "n_retained_from_v01": len(missing),
                "retained_keys_sample": missing[:50],
                "stats": stats,
                "note": (
                    "Rows not replaced retain their v0.1 (concatenated-input) value. "
                    "Report this count; do not describe the release as fully "
                    "per-chain unless it is zero."
                ),
            },
            indent=2,
        )
    )
    print("\nwrote %s" % (outdir / "pocket_embeddings.npy"))
    print("replaced %d rows; %d retained from v0.1; stats=%s" % (replaced, len(missing), stats))
    if missing:
        print("WARNING: %d rows are still the v0.1 concatenated embedding." % len(missing))


# ---------------------------------------------------------------------------


def _self_test():
    """Exercise the chain-splitting and pocket-mapping logic with no data."""

    class R:
        def __init__(self, num, icode, name):
            self.seqid = type("S", (), {"num": num, "icode": icode})()
            self.name = name
            self.entity_type = "POLY"

    # stand-in for gemmi structures: mimic the iteration protocol
    class Chain(list):
        def __init__(self, name, residues):
            super().__init__(residues)
            self.name = name

    chains_in = {
        "A": [R(1, "", "ALA"), R(2, "", "SEP"), R(3, "", "GLY")],
        "B": [R(10, "", "VAL"), R(11, "A", "LEU")],
    }

    # replicate per_chain_sequences' filter without importing gemmi
    built = {}
    for cname, residues in chains_in.items():
        chars, meta = [], []
        for r in residues:
            one = THREE_TO_ONE.get(r.name.upper())
            if one is None:
                continue  # SEP dropped, consumes no position
            chars.append(one)
            meta.append((int(r.seqid.num), str(r.seqid.icode or "").strip(), r.name))
        built[cname] = ("".join(chars), meta)

    assert built["A"][0] == "AG", built["A"][0]  # SEP silently dropped
    assert built["B"][0] == "VL"

    prov = [
        {"chain": "A", "seqid": 3, "icode": "", "resname": "GLY"},
        {"chain": "B", "seqid": 11, "icode": "A", "resname": "LEU"},
    ]
    assign, unres = map_pocket_to_chains(prov, built)
    assert unres == 0, unres
    assert assign == [("A", 1), ("B", 1)], assign  # GLY is index 1 after SEP drop

    # gate must reject a provenance/structure disagreement
    bad = [{"chain": "A", "seqid": 3, "icode": "", "resname": "TRP"}]
    _, unres_bad = map_pocket_to_chains(bad, built)
    assert unres_bad == 1

    # unknown residue -> unresolved, not a silent wrong index
    miss = [{"chain": "A", "seqid": 99, "icode": "", "resname": "GLY"}]
    _, unres_miss = map_pocket_to_chains(miss, built)
    assert unres_miss == 1

    print(
        "reembed_per_chain self-test: OK  "
        "(non-standard residues consume no position; pocket residues resolve to "
        "per-chain indices; correctness gate rejects mismatched and missing residues)"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--retrieval_dir")
    ap.add_argument("--residue_map")
    ap.add_argument("--out_dir")
    ap.add_argument("--shard_dir", default=None)
    ap.add_argument("--cif_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint_every", type=int, default=500)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    missing = [k for k in ("retrieval_dir", "residue_map", "out_dir") if not getattr(args, k)]
    if missing:
        sys.exit("missing required arguments: %s (or pass --self_test)" % missing)
    run(args)


if __name__ == "__main__":
    main()
