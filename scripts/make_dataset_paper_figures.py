"""Regenerate every figure in paper/nature_dataset_paper.{tex,md}.

Figure 1 (pipeline schematic) is data-free and runs anywhere:

    python scripts/make_dataset_paper_figures.py --fig 1 --out paper/figures

Figures 2-6 need the released dataset. Easiest path is Colab with the HF
snapshot mirrored to a writable directory:

    from huggingface_hub import snapshot_download
    snap = snapshot_download(
        "HemanthKari/rcsb-ligand-retrieval-v0.1", repo_type="dataset")

    !python scripts/make_dataset_paper_figures.py --fig 2 \
        --retrieval_dir {snap} --out paper/figures \
        --pocket_map /content/drive/MyDrive/FluxionDock/pocket_residue_map.parquet
    !python scripts/make_dataset_paper_figures.py --fig 3 \
        --retrieval_dir {snap} --out paper/figures --n_eval 0 --seed 42
    !python scripts/make_dataset_paper_figures.py --fig 4 \
        --retrieval_dir {snap} --out paper/figures --seed 42
    !python scripts/make_dataset_paper_figures.py --fig 5 \
        --retrieval_dir {snap} --out paper/figures --seed 42
    !python scripts/make_dataset_paper_figures.py --fig 6 \
        --retrieval_dir {snap} --out paper/figures --seed 42

    # or everything in one process (SIFTS + matrices loaded once per figure):
    !python scripts/make_dataset_paper_figures.py --fig all \
        --retrieval_dir {snap} --out paper/figures --n_eval 0 --seed 42

--n_eval 0 (or any value >= the index size) evaluates the FULL joined index.
Seed 42 matches every other evaluation in the paper and is printed into every
figure footer, so a rendered PDF always states the seed it was made with.

Provenance notes that the captions depend on
--------------------------------------------
* Figure 2c plots the TRUE ESM-2 input length taken from
  ``pocket_residue_map.parquet`` (column ``esm_input_length``), recovered by
  notebook 09. It is NOT ``len(metadata.protein_seq)``: that field is the
  longest single canonical chain, whereas the ESM-2 input is the concatenation
  of every polymer chain in the entry (48.85% of entries are multi-chain), and
  it is the concatenated length that the 50-1,022 residue filter was applied
  to. Both quantities are drawn in panel (c) so the difference is explicit.
* Figure 3 separates EXACT search (chunked exact inner product over the
  L2-normalised joined rows, mathematically the same quantity FAISS
  ``IndexFlatIP`` returns) from the SHIPPED ``retrieval.faiss`` HNSW graph.
  The published Figure 3 numbers were exact-search numbers; users of the
  released artifact get the HNSW numbers. Both are drawn.
* No figure in this file makes a calibration claim. Cosine magnitudes in this
  corpus are anisotropy-dominated (Fig 3c); only the ORDERING is meaningful.

Memory: the largest working set is Figure 5's alpha sweep, which streams the
pocket and ligand matrices in row chunks and never materialises the 3,328-d
concatenation for the whole corpus. Peak RSS is roughly 2 GB.

Style: matplotlib only, Agg backend, every figure written as both .pdf and
.png at 300 dpi. Headline numbers are printed to stdout so captions can be
locked to the rendered figure.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


_LE = "≤"  # ≤
_GE = "≥"  # ≥
_AA = "Å"  # Å
_RA = "→"  # →

# Shared palette. Blue = the shipped/joint artifact, green = pocket modality,
# pink = ligand modality, grey = null/random baseline, red = a caveat.
C_JOINT = "#3949ab"
C_POCKET = "#2e7d32"
C_LIGAND = "#c2185b"
C_RANDOM = "#9e9e9e"
C_ALERT = "#d32f2f"
C_ANN = "#ef6c00"
C_CENTRED = "#00838f"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _thousands(value):
    """Comma-grouped integer, without f-strings."""
    return format(int(value), ",d")


def _read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pick(rows, keys):
    for k in keys:
        if k in rows[0]:
            return k
    raise KeyError("none of %s in %s" % (keys, sorted(rows[0])))


def _panel(ax, letter, dx=-0.12, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=15, weight="bold", va="bottom")


def _panel_pts(ax, letter, dx=-38, dy=10):
    """Panel letter placed in offset points, so it cannot fall off the canvas
    when the left margin is dominated by long tick labels."""
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=15,
        weight="bold",
        va="bottom",
        ha="left",
        annotation_clip=False,
    )


def _footer(fig, text):
    """Provenance line under the panels.

    Wrapped, because it is the one place a figure states which artefact each
    panel was measured on: an unwrapped line runs off the right edge and the
    reader loses the end of it, which is exactly where the panel-by-panel
    attribution lives. Width is derived from the figure so this holds for
    every figure size in this file.
    """
    import textwrap

    width = max(60, int(fig.get_figwidth() * 15.5))
    fig.text(
        0.005,
        0.004,
        chr(10).join(textwrap.wrap(text, width)),
        fontsize=6.5,
        color="#546e7a",
        ha="left",
        va="bottom",
    )


def _save(fig, out_dir, stem):
    pdf = os.path.join(out_dir, stem + ".pdf")
    png = os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("wrote %s.{pdf,png}  (300 dpi)" % stem)


def _headline(title):
    print("")
    print("=" * 74)
    print("HEADLINE NUMBERS -- %s" % title)
    print("=" * 74)


def _stat_line(name, arr):
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        print("  %-38s UNAVAILABLE (no values)" % name)
        return
    print(
        "  %-38s median %.4f  IQR [%.4f, %.4f]  min %.4f  max %.4f"
        % (
            name,
            float(np.median(a)),
            float(np.percentile(a, 25)),
            float(np.percentile(a, 75)),
            float(a.min()),
            float(a.max()),
        )
    )


# ---------------------------------------------------------------------------
# Core artifact loading
# ---------------------------------------------------------------------------


def _load_core(retrieval_dir, matrices=True):
    """metadata + memory-mapped embedding matrices + per-entry offsets."""
    import numpy as np

    meta_path = os.path.join(retrieval_dir, "metadata.jsonl")
    if not os.path.exists(meta_path):
        raise SystemExit("metadata.jsonl not found under --retrieval_dir %s" % retrieval_dir)
    meta = _read_jsonl(meta_path)
    core = {"meta": meta, "n": len(meta), "dir": retrieval_dir}
    if not matrices:
        return core

    joined = np.load(os.path.join(retrieval_dir, "joined_embeddings.npy"), mmap_mode="r")
    pocket = np.load(os.path.join(retrieval_dir, "pockets", "pocket_embeddings.npy"), mmap_mode="r")
    ligand = np.load(os.path.join(retrieval_dir, "ligands", "ligand_embeddings.npy"), mmap_mode="r")
    core["joined"] = joined
    core["pocket"] = pocket
    core["ligand"] = ligand
    core["p_off"] = np.array([m["pocket_emb_offset"] for m in meta], dtype=np.int64)
    core["l_off"] = np.array([m["ligand_emb_offset"] for m in meta], dtype=np.int64)
    print(
        "loaded: metadata %s rows | joined %s | pocket %s | ligand %s"
        % (_thousands(core["n"]), joined.shape, pocket.shape, ligand.shape)
    )
    return core


def _load_splits(retrieval_dir):
    import numpy as np

    tr = np.load(os.path.join(retrieval_dir, "splits", "train_indices.npy"))
    va = np.load(os.path.join(retrieval_dir, "splits", "val_indices.npy"))
    return tr.astype(np.int64), va.astype(np.int64)


def _gather(mat, rows, chunk=8192):
    """Materialise mat[rows] as float32 without a giant intermediate."""
    import numpy as np

    out = np.empty((len(rows), mat.shape[1]), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        e = min(s + chunk, len(rows))
        out[s:e] = np.asarray(mat[rows[s:e]], dtype=np.float32)
    return out


def _chunked_stats(mat, rows=None, chunk=8192):
    """Streaming (mean vector, per-row L2 norm) over mat, optionally gathered.

    Used instead of materialising the 1,280-d / 2,048-d matrices when only
    norms and the corpus mean direction are needed.
    """
    import numpy as np

    n = mat.shape[0] if rows is None else len(rows)
    acc = np.zeros(mat.shape[1], dtype=np.float64)
    norms = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = np.asarray(mat[s:e] if rows is None else mat[rows[s:e]], dtype=np.float64)
        acc += block.sum(axis=0)
        norms[s:e] = np.linalg.norm(block, axis=1)
    return (acc / float(n)).astype(np.float32), norms


def _topk_chunked(Q, mat, k, rows=None, exclude=None, center=None, chunk=8192, qbatch=2048):
    """Exact top-k inner-product search, streamed over the corpus.

    Q          (nq, d) float32; already centred/normalised if `center` is used.
    mat        (M, d) array or memmap.
    rows       optional index array selecting the searchable subset of `mat`;
               returned ids are positions in that subset.
    exclude    optional (nq,) array of subset positions to mask (self-match).
    center     optional (d,) vector subtracted from every corpus row before
               re-normalising, for the mean-centred variant.

    Returns (D, I) sorted descending, exactly like faiss.IndexFlatIP.search on
    L2-normalised rows (float accumulation order differs, so exact ties may be
    resolved differently; the values agree to float32 precision).
    """
    import numpy as np

    nq = Q.shape[0]
    n_corpus = mat.shape[0] if rows is None else len(rows)
    k = int(min(k, n_corpus))
    best_d = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)

    for s in range(0, n_corpus, chunk):
        e = min(s + chunk, n_corpus)
        block = np.asarray(mat[s:e] if rows is None else mat[rows[s:e]], dtype=np.float32)
        if center is not None:
            block = block - center
            block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
        blockT = np.ascontiguousarray(block.T)
        del block
        for qs in range(0, nq, qbatch):
            qe = min(qs + qbatch, nq)
            sims = Q[qs:qe] @ blockT
            if exclude is not None:
                ex = exclude[qs:qe]
                m = (ex >= s) & (ex < e)
                hit = np.flatnonzero(m)
                if hit.size:
                    sims[hit, ex[hit] - s] = -np.inf
            kk = int(min(k, sims.shape[1]))
            part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            cand_d = np.take_along_axis(sims, part, axis=1)
            cand_i = (part + s).astype(np.int64)
            all_d = np.concatenate([best_d[qs:qe], cand_d], axis=1)
            all_i = np.concatenate([best_i[qs:qe], cand_i], axis=1)
            order = np.argsort(-all_d, axis=1, kind="stable")[:, :k]
            best_d[qs:qe] = np.take_along_axis(all_d, order, axis=1)
            best_i[qs:qe] = np.take_along_axis(all_i, order, axis=1)
        del blockT
    return best_d, best_i


# ---------------------------------------------------------------------------
# SIFTS labels (UniProt / Pfam / EC)
# ---------------------------------------------------------------------------

_SIFTS_BASE = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/"
_SIFTS_SPEC = {
    "uniprot": ("pdb_chain_uniprot.csv.gz", "SP_PRIMARY"),
    "pfam": ("pdb_chain_pfam.csv.gz", "PFAM_ID"),
    "ec": ("pdb_chain_enzyme.csv.gz", "EC_NUMBER"),
}


def _ensure_sifts(sifts_dir, which, allow_download=True):
    fname, _ = _SIFTS_SPEC[which]
    path = os.path.join(sifts_dir, fname)
    if os.path.exists(path):
        return path
    if not allow_download:
        raise SystemExit(
            "%s missing from --sifts_dir %s and --no_download was given" % (fname, sifts_dir)
        )
    os.makedirs(sifts_dir, exist_ok=True)
    url = _SIFTS_BASE + fname
    # Loud, because this silently changed a published figure once. SIFTS is
    # a LIVE service: the snapshot fetched today is not the snapshot the
    # deposit ships, UniProt mappings move between them, and the resulting
    # figure is then not reproducible from the deposited data. Pass
    # --sifts_dir <release>/provenance/sifts to use the deposited snapshot.
    print("")
    print("  !! DOWNLOADING A LIVE SIFTS SNAPSHOT: %s" % fname)
    print("  !! This is NOT the snapshot in the deposit. Figures built from it")
    print("  !! will not reproduce from the released data. Use --sifts_dir")
    print("  !! <release>/provenance/sifts --no_download for a release figure.")
    print("")
    urllib.request.urlretrieve(url, path)
    return path


def _load_sifts_labels(sifts_dir, meta, which, allow_download=True):
    """Per-entry label array (empty string where the PDB id is unmapped)."""
    import numpy as np
    import pandas as pd

    fname, col = _SIFTS_SPEC[which]
    path = _ensure_sifts(sifts_dir, which, allow_download=allow_download)
    df = pd.read_csv(path, compression="gzip", skiprows=1)
    df["PDB"] = df["PDB"].astype(str).str.lower()
    mapping = df.groupby("PDB")[col].first().to_dict()
    if which == "ec":
        labels = [str(mapping.get(m["pdb_id"].lower(), "")).split(".")[0] for m in meta]
        labels = ["" if v in ("", "nan", "None", "?") else v for v in labels]
    else:
        labels = [str(mapping.get(m["pdb_id"].lower(), "")) for m in meta]
        labels = ["" if v in ("nan", "None") else v for v in labels]
    arr = np.array(labels, dtype=object)
    n_res = int((arr != "").sum())
    print(
        "SIFTS %-8s resolved %s / %s entries (%.1f%%)"
        % (which, _thousands(n_res), _thousands(len(arr)), 100.0 * n_res / max(1, len(arr)))
    )
    return arr


# Best-effort Pfam short names. The accession is always rendered as well, so a
# missing or stale name never makes the figure wrong. Overridden by the
# InterPro API when it is reachable (see _pfam_names).
_PFAM_FALLBACK_NAMES = {
    "PF00069": "Protein kinase domain",
    "PF07714": "Protein tyrosine/Ser-Thr kinase",
    "PF00089": "Trypsin",
    "PF00194": "Carbonic anhydrase",
    "PF00186": "Dihydrofolate reductase",
    "PF00144": "Beta-lactamase",
    "PF00501": "AMP-binding enzyme",
    "PF00248": "Aldo/keto reductase",
    "PF00107": "Zinc-binding dehydrogenase",
    "PF00135": "Carboxylesterase family B",
    "PF00156": "Phosphoribosyl transferase",
    "PF00246": "Zinc carboxypeptidase",
    "PF00753": "Metallo-beta-lactamase fold",
    "PF00560": "Leucine-rich repeat",
    "PF00450": "Serine carboxypeptidase",
    "PF00483": "Nucleotidyl transferase",
    "PF01048": "Phosphorylase superfamily",
    "PF00300": "Histidine phosphatase",
    "PF00378": "Enoyl-CoA hydratase/isomerase",
    "PF02898": "PF02898",
}


def _pfam_names(accessions, cache_path=None, fetch=True, timeout=6.0):
    """accession -> short name, InterPro first, built-in table as fallback."""
    names = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                names.update(json.load(fh))
        except (OSError, ValueError):
            names = {}
    missing = [a for a in accessions if a not in names]
    if fetch and missing:
        n_failed = 0
        for acc in missing:
            url = "https://www.ebi.ac.uk/interpro/api/entry/pfam/%s/" % acc
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                meta_block = payload.get("metadata", {})
                nm = meta_block.get("name", {})
                label = nm.get("name") or nm.get("short") or ""
                if label:
                    names[acc] = label
                else:
                    n_failed += 1
            except (OSError, ValueError):
                n_failed += 1
        if n_failed:
            print(
                "  note: %d of %d Pfam short names could not be fetched from InterPro; "
                "falling back to the built-in table / bare accession" % (n_failed, len(missing))
            )
        if cache_path:
            try:
                with open(cache_path, "w") as fh:
                    json.dump(names, fh, indent=1, sort_keys=True)
            except OSError as exc:
                print("  note: could not write the Pfam name cache (%s)" % exc)
    for acc in accessions:
        if acc not in names:
            names[acc] = _PFAM_FALLBACK_NAMES.get(acc, acc)
    return names


# ---------------------------------------------------------------------------
# Figure 1: pipeline schematic (data-free)
# ---------------------------------------------------------------------------


def _box(ax, x, y, w, h, text, *, fc="#ffffff", ec="#1f2933", lw=1.2, fontsize=10, weight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, wrap=True)


def _arrow(ax, x1, y1, x2, y2, *, color="#1f2933", lw=1.2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2),
    )


def make_figure_1(out_dir):
    """Pipeline schematic. Independent of any data files."""
    fig, ax = plt.subplots(figsize=(8.6, 9.8))
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.4, 11.2)
    ax.axis("off")

    # Source
    _box(ax, 5.0, 10.4, 4.2, 0.7, "RCSB Protein Data Bank (mmCIF)", fc="#e3f2fd", weight="bold")
    _arrow(ax, 5.0, 10.0, 5.0, 9.45)

    # License & quality filter (single wide box, comma-separated bullets).
    # Use hyphen-minus (not en-dash) inside figure boxes: matplotlib's PDF
    # backend can render U+2013 too thin, looking like a space in print.
    # The 50-1,022 bound is on the SUMMED polymer length and selects which
    # ENTRIES are admitted. It is not the ESM-2 input length: from v0.2 the
    # 1,022-token cap applies per chain, and Fig 2c plots esm_input_length.
    filter_text = (
        "License & quality filter\n"
        "X-ray, "
        + _LE
        + " 2.5 "
        + _AA
        + ", ligand 10-70 heavy atoms,"
        + chr(10)
        + "summed polymer length 50-1,022 residues (entry selection)"
    )
    _box(ax, 5.0, 9.05, 8.4, 1.05, filter_text, fc="#fff8e1")
    _arrow(ax, 5.0, 8.6, 5.0, 8.05)

    # Filtered manifest
    _box(
        ax,
        5.0,
        7.65,
        4.6,
        0.65,
        "Filtered manifest (n = 84,396 entries)",
        fc="#ffffff",
        weight="bold",
    )
    _arrow(ax, 3.2, 7.30, 2.4, 6.75)
    _arrow(ax, 6.8, 7.30, 7.6, 6.75)

    # ---- Left branch: pocket -------------------------------------------------
    _box(
        ax,
        2.4,
        6.30,
        4.4,
        0.85,
        "Pocket extraction\nresidues within 8 " + _AA + " of any\nligand heavy atom",
        fc="#e8f5e9",
    )
    _arrow(ax, 2.4, 5.85, 2.4, 5.30)

    _box(
        ax,
        2.4,
        4.85,
        4.4,
        0.95,
        "ESM-2 t33 650M over each\npocket-bearing chain separately\n"
        "mean-pool over pocket residues\n1,280-d embedding per complex",
        fc="#c8e6c9",
        fontsize=9,
    )
    _arrow(ax, 2.4, 4.35, 4.2, 3.55)

    # ---- Right branch: ligand ------------------------------------------------
    _box(ax, 7.6, 6.30, 4.4, 0.85, "Ligand extraction\ncoords + SMILES from CCD", fc="#fce4ec")
    _arrow(ax, 7.6, 5.85, 7.6, 5.30)

    _box(
        ax,
        7.6,
        4.85,
        4.4,
        0.95,
        "RDKit ECFP4 (radius 2)\n2,048-bit Morgan fingerprint\nper CCD chemical component",
        fc="#f8bbd0",
        fontsize=9,
    )
    _arrow(ax, 7.6, 4.35, 5.8, 3.55)

    # ---- Indices ------------------------------------------------------------
    # v0.2 ships the two single-modality indices as the primary artifacts.
    # The 3,328-d concatenation is retained only to reproduce the preview,
    # so it is drawn as a secondary branch, not as the pipeline's output.
    _box(
        ax,
        2.4,
        3.10,
        4.4,
        0.95,
        "indices/pocket.faiss\n1,280-d, L2-normalised\nFAISS HNSW M = 32, efC = 200",
        fc="#d1c4e9",
        weight="bold",
        fontsize=9,
    )
    _box(
        ax,
        7.6,
        3.10,
        4.4,
        0.95,
        "indices/ligand.faiss\n2,048-d, L2-normalised\nFAISS HNSW M = 32, efC = 200",
        fc="#d1c4e9",
        weight="bold",
        fontsize=9,
    )
    _arrow(ax, 2.4, 2.60, 4.2, 2.10)
    _arrow(ax, 7.6, 2.60, 5.8, 2.10)

    _box(
        ax,
        5.0,
        1.65,
        6.6,
        0.95,
        "also retained: retrieval.faiss, built over the\nconcatenated embedding matrices (3,328-d), kept for continuity",
        fc="#ede7f6",
        fontsize=9,
    )
    _arrow(ax, 5.0, 1.15, 5.0, 0.65)

    _box(
        ax,
        5.0,
        0.30,
        7.8,
        0.85,
        "Released dataset\n"
        "n = 72,199 (joined index)\n"
        "72,485 pocket and 72,388 ligand embeddings",
        fc="#1f2933",
        ec="#1f2933",
        weight="bold",
        fontsize=10,
    )
    ax.texts[-1].set_color("#ffffff")

    fig.tight_layout(pad=0.5)
    _save(fig, out_dir, "fig1_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: per-complex distributions
# ---------------------------------------------------------------------------


def _find_pocket_map(retrieval_dir, out_dir, explicit=None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    if retrieval_dir:
        candidates.append(os.path.join(retrieval_dir, "pocket_residue_map.parquet"))
    candidates.append(os.path.join(out_dir, "pocket_residue_map.parquet"))
    candidates.append("pocket_residue_map.parquet")
    candidates.append(os.path.join("data", "pocket_residue_map.parquet"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise SystemExit(
        "pocket_residue_map.parquet not found (looked in: %s).\n"
        "Figure 2c REQUIRES it: it carries esm_input_length, the true ESM-2 "
        "input length recovered by notebook 09. Do not substitute "
        "len(metadata.protein_seq) - that is the longest single chain, a "
        "different quantity, and plotting it against the 50/1,022 bounds is "
        "the bug this rewrite fixes. Pass --pocket_map /path/to/file."
        % ", ".join([c for c in candidates if c])
    )


def make_figure_2(out_dir, retrieval_dir, pocket_map=None):
    import numpy as np
    import pandas as pd

    pocket_manifest = _read_jsonl(os.path.join(retrieval_dir, "pockets", "pocket_manifest.jsonl"))
    ligand_manifest = _read_jsonl(os.path.join(retrieval_dir, "ligands", "ligand_manifest.jsonl"))
    raft_path = os.path.join(retrieval_dir, "raft_pdb_filtered.jsonl")
    raft = _read_jsonl(raft_path) if os.path.exists(raft_path) else []

    metadata_path = os.path.join(retrieval_dir, "metadata.jsonl")
    metadata = _read_jsonl(metadata_path) if os.path.exists(metadata_path) else []

    # Panel (c): the TRUE ESM-2 input length. pocket_residue_map.parquet is the
    # notebook-09 re-extraction; esm_input_length is len() of the exact string
    # that was tokenised, i.e. every polymer chain of the entry concatenated in
    # chain_order, minus the non-standard residues the extractor dropped. This
    # is the quantity the 50-1,022 filter bounds apply to. It is NOT
    # len(metadata.protein_seq), which is the longest single canonical chain
    # and is drawn as a second, lighter histogram for contrast.
    map_path = _find_pocket_map(retrieval_dir, out_dir, pocket_map)
    pmap = pd.read_parquet(map_path, columns=["pocket_key", "esm_input_length"])
    esm_len = pmap["esm_input_length"].to_numpy(dtype=np.int64)
    print("pocket_residue_map: %s -> %s rows" % (map_path, _thousands(len(esm_len))))

    heavy_key = _pick(ligand_manifest, ["ligand_heavy_atoms", "heavy_atoms"])
    res_key = _pick(raft, ["resolution", "resolution_A"]) if raft else None
    pocket_key = _pick(pocket_manifest, ["pocket_n_residues", "pocket_residues"])

    ligand_heavy = np.array(
        [r[heavy_key] for r in ligand_manifest if r.get(heavy_key) is not None], dtype=np.float64
    )
    resolutions = (
        np.array([r[res_key] for r in raft if r.get(res_key) is not None], dtype=np.float64)
        if raft
        else np.array([])
    )
    longest_chain = np.array(
        [
            len(r["protein_seq"])
            for r in metadata
            if isinstance(r.get("protein_seq"), str) and r["protein_seq"]
        ],
        dtype=np.float64,
    )
    pocket_residues = np.array(
        [r[pocket_key] for r in pocket_manifest if r.get(pocket_key) is not None], dtype=np.float64
    )

    n_over = int((esm_len > 1022).sum())
    n_under = int((esm_len < 50).sum())

    _headline("Figure 2 (dataset characterisation)")
    _stat_line("ligand heavy atoms", ligand_heavy)
    if resolutions.size:
        _stat_line("resolution (A)", resolutions)
    _stat_line("ESM-2 input length (TRUE, panel c)", esm_len)
    _stat_line("longest single chain (contrast)", longest_chain)
    _stat_line("pocket residues within 8 A", pocket_residues)
    print(
        "  ESM-2 input length: %s entries > 1,022 (%.2f%%), %s entries < 50 (%.2f%%)"
        % (
            _thousands(n_over),
            100.0 * n_over / max(1, len(esm_len)),
            _thousands(n_under),
            100.0 * n_under / max(1, len(esm_len)),
        )
    )
    if longest_chain.size:
        print(
            "  median ESM-2 input %.0f vs median longest chain %.0f residues "
            "(ratio %.2fx) - the two are NOT the same quantity"
            % (
                float(np.median(esm_len)),
                float(np.median(longest_chain)),
                float(np.median(esm_len)) / max(1.0, float(np.median(longest_chain))),
            )
        )
    else:
        print(
            "  WARNING: no metadata.protein_seq values in this snapshot, so the "
            "longest-single-chain contrast in panel (c) is omitted. Panel (c)'s "
            "primary series (esm_input_length) is unaffected."
        )

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    fig.suptitle("Distribution of dataset characteristics", fontsize=13, weight="bold")

    simple_panels = [
        ("a", "Ligand heavy atoms", ligand_heavy, (10, 70), "Heavy-atom count", axes[0, 0]),
        (
            "b",
            "Crystal resolution (" + _AA + ")",
            resolutions,
            (None, 2.5),
            "Resolution (" + _AA + ")",
            axes[0, 1],
        ),
        (
            "d",
            "Pocket residues within 8 " + _AA,
            pocket_residues,
            (None, None),
            "Residues",
            axes[1, 1],
        ),
    ]
    for label, title, data, (lo, hi), xlabel, ax in simple_panels:
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, "artifact unavailable", ha="center", va="center")
            ax.axis("off")
            continue
        ax.hist(data, bins=40, color=C_JOINT, alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Complexes")
        if lo is not None:
            ax.axvline(lo, color=C_ALERT, linestyle="--", linewidth=1)
        if hi is not None:
            ax.axvline(hi, color=C_ALERT, linestyle="--", linewidth=1)
        _panel(ax, label, dx=0.02, dy=1.04)
        ax.text(
            0.98,
            0.95,
            "n = " + _thousands(len(data)),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cfd8dc"),
        )

    # ---- panel (c): the corrected quantity ---------------------------------
    ax = axes[1, 0]
    hi_cut = float(max(1100.0, np.percentile(esm_len, 99.9)))
    bins = np.linspace(0, hi_cut, 60)
    ax.hist(
        esm_len,
        bins=bins,
        color=C_JOINT,
        alpha=0.85,
        edgecolor="white",
        label="ESM-2 input length (n = " + _thousands(len(esm_len)) + ")",
    )
    if longest_chain.size:
        ax.hist(
            longest_chain,
            bins=bins,
            histtype="step",
            color=C_LIGAND,
            linewidth=1.4,
            label="longest single chain (n = " + _thousands(len(longest_chain)) + ")",
        )
    ax.axvline(50, color=C_ALERT, linestyle="--", linewidth=1)
    ax.axvline(1022, color=C_ALERT, linestyle="--", linewidth=1)
    ax.set_title("ESM-2 input length (residues)")
    ax.set_xlabel("Residues in the tokenised sequence")
    ax.set_ylabel("Complexes")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    _panel(ax, "c", dx=0.02, dy=1.04)
    ax.text(
        0.98,
        0.62,
        "50 / 1,022 = entry-selection bounds\non summed polymer length,\nnot on this quantity",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=C_ALERT,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _footer(
        fig,
        "Panel c: esm_input_length from pocket_residue_map.parquet (per-chain re-embedding), "
        "not len(metadata.protein_seq). 39 entries fall below 50. Bins fixed, no sampling.",
    )
    _save(fig, out_dir, "fig2_dataset_characterisation")


# ---------------------------------------------------------------------------
# Figure 3: technical validation (exact vs shipped HNSW, anisotropy floor)
# ---------------------------------------------------------------------------


def make_figure_3(
    out_dir,
    retrieval_dir,
    n_eval=1000,
    seed=42,
    ks=(1, 5, 10, 50, 100),
    ef_list=(16, 64, 128, 256),
    skip_hnsw=False,
    n_ann=2000,
    n_rand=20000,
):
    import numpy as np

    core = _load_core(retrieval_dir)
    joined = core["joined"]
    pocket = core["pocket"]
    n_keep = joined.shape[0]

    rng = np.random.default_rng(seed)
    if n_eval is None or n_eval <= 0 or n_eval >= n_keep:
        query_idx = np.arange(n_keep, dtype=np.int64)
        n_eval = n_keep
        print("evaluating on the full %s-row joined index" % _thousands(n_keep))
    else:
        query_idx = np.sort(rng.choice(n_keep, size=n_eval, replace=False)).astype(np.int64)
        print("evaluating on a random sample of n=%s (seed=%d)" % (_thousands(n_eval), seed))
    queries = _gather(joined, query_idx)

    max_k = max(ks)
    print("exact search (chunked inner product, self excluded)...")
    exact_d, exact_i = _topk_chunked(queries, joined, max_k, exclude=query_idx)
    exact_d = np.asarray(exact_d, dtype=np.float64)

    med_exact = [float(np.median(exact_d[:, k - 1])) for k in ks]
    q1_exact = [float(np.percentile(exact_d[:, k - 1], 25)) for k in ks]
    q3_exact = [float(np.percentile(exact_d[:, k - 1], 75)) for k in ks]
    top1 = exact_d[:, 0]

    # ---- random-pair baselines: joint space, raw and mean-centred ----------
    n_rand = int(min(n_rand, n_keep))
    ia = rng.integers(0, n_keep, size=n_rand)
    ib = rng.integers(0, n_keep, size=n_rand)
    Ja = _gather(joined, ia)
    Jb = _gather(joined, ib)
    rand_raw = (Ja.astype(np.float64) * Jb.astype(np.float64)).sum(axis=1) / (
        np.linalg.norm(Ja, axis=1) * np.linalg.norm(Jb, axis=1) + 1e-12
    )

    print("computing corpus mean vectors (streamed)...")
    j_mean, j_norms = _chunked_stats(joined)
    Jac = Ja - j_mean
    Jbc = Jb - j_mean
    rand_centred = (Jac.astype(np.float64) * Jbc.astype(np.float64)).sum(axis=1) / (
        np.linalg.norm(Jac, axis=1) * np.linalg.norm(Jbc, axis=1) + 1e-12
    )
    aniso_joint = float(np.linalg.norm(j_mean) / max(1e-12, j_norms.mean()))
    del Ja, Jb, Jac, Jbc

    # ---- the same experiment in POCKET space, where the floor is largest ---
    p_off = core["p_off"]
    p_mean, p_norms = _chunked_stats(pocket, rows=p_off)
    Pa = _gather(pocket, p_off[ia])
    Pb = _gather(pocket, p_off[ib])
    prand_raw = (Pa.astype(np.float64) * Pb.astype(np.float64)).sum(axis=1) / (
        np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-12
    )
    Pac = Pa - p_mean
    Pbc = Pb - p_mean
    prand_centred = (Pac.astype(np.float64) * Pbc.astype(np.float64)).sum(axis=1) / (
        np.linalg.norm(Pac, axis=1) * np.linalg.norm(Pbc, axis=1) + 1e-12
    )
    aniso_pocket = float(np.linalg.norm(p_mean) / max(1e-12, p_norms.mean()))
    del Pa, Pb, Pac, Pbc

    # ---- the SHIPPED HNSW index -------------------------------------------
    hnsw_med = None
    recall = {}
    hnsw_note = "shipped HNSW index not evaluated"
    if not skip_hnsw:
        try:
            import faiss

            hnsw_path = os.path.join(retrieval_dir, "retrieval.faiss")
            hnsw = faiss.read_index(hnsw_path)
            ann_idx = query_idx if len(query_idx) <= n_ann else query_idx[:n_ann]
            Qa = queries if len(query_idx) <= n_ann else queries[:n_ann]
            exact_ids_sub = exact_i[: len(ann_idx)]

            default_ef = 128 if 128 in ef_list else ef_list[-1]
            for ef in ef_list:
                # Do NOT swallow this: if efSearch cannot be set, every point on
                # panel (d) is measured at the same actual efSearch and the
                # x-axis of the recall curve is a fiction.
                hnsw.hnsw.efSearch = int(ef)
                if int(hnsw.hnsw.efSearch) != int(ef):
                    raise RuntimeError(
                        "efSearch did not take: asked for %d, index reports %d"
                        % (int(ef), int(hnsw.hnsw.efSearch))
                    )
                D, I = hnsw.search(np.ascontiguousarray(Qa), max_k + 1)
                # drop self-matches row-wise
                keep_d = np.empty((len(ann_idx), max_k), dtype=np.float64)
                keep_i = np.empty((len(ann_idx), max_k), dtype=np.int64)
                for r in range(len(ann_idx)):
                    mask = (I[r] != ann_idx[r]) & (I[r] >= 0)
                    dd = D[r][mask][:max_k]
                    ii = I[r][mask][:max_k]
                    if len(dd) < max_k:
                        pad = max_k - len(dd)
                        dd = np.concatenate([dd, np.full(pad, np.nan)])
                        ii = np.concatenate([ii, np.full(pad, -1, dtype=np.int64)])
                    keep_d[r] = dd
                    keep_i[r] = ii
                kr = int(min(10, max_k))
                r1 = float(np.mean(keep_i[:, 0] == exact_ids_sub[:, 0]))
                r10 = float(
                    np.mean(
                        [
                            len(set(keep_i[r, :kr].tolist()) & set(exact_ids_sub[r, :kr].tolist()))
                            / float(kr)
                            for r in range(len(ann_idx))
                        ]
                    )
                )
                recall[int(ef)] = (r1, r10)
                if int(ef) == int(default_ef):
                    hnsw_med = [float(np.nanmedian(keep_d[:, k - 1])) for k in ks]
            hnsw_note = "shipped HNSW, efSearch = %d" % default_ef
            print("HNSW evaluated on %s queries" % _thousands(len(ann_idx)))
        except Exception as exc:  # pragma: no cover - environment dependent
            print("WARNING: could not evaluate the shipped HNSW index (%s)" % exc)
            hnsw_note = "shipped HNSW index unavailable in this environment"

    # ---- headline numbers --------------------------------------------------
    _headline("Figure 3 (technical validation)")
    print("  search backend for panels a/b : EXACT chunked inner product")
    print("  shipped artifact              : retrieval.faiss (HNSW, M=32, efC=200)")
    print("  n_eval = %s, seed = %d" % (_thousands(n_eval), seed))
    print("  exact top-k medians at k = %s -> %s" % (list(ks), [round(v, 4) for v in med_exact]))
    if hnsw_med is not None:
        print("  HNSW  top-k medians at k = %s -> %s" % (list(ks), [round(v, 4) for v in hnsw_med]))
    _stat_line("exact top-1 cosine", top1)
    _stat_line("random pair, joint space (raw)", rand_raw)
    _stat_line("random pair, joint space (centred)", rand_centred)
    _stat_line("random pair, pocket space (raw)", prand_raw)
    _stat_line("random pair, pocket space (centred)", prand_centred)
    print(
        "  anisotropy ||mean|| / mean||row||   joint %.3f   pocket %.3f"
        % (aniso_joint, aniso_pocket)
    )
    for ef in sorted(recall):
        print(
            "  HNSW efSearch=%-4d recall@1 %.4f  recall@10 %.4f"
            % (ef, recall[ef][0], recall[ef][1])
        )
    hit1 = float((exact_d[:, 0] >= 0.999).mean())
    hit5 = float((exact_d[:, :5].max(axis=1) >= 0.999).mean())
    hit10 = float((exact_d[:, :10].max(axis=1) >= 0.999).mean())
    print(
        "  near-identical rate at >=0.999 cosine: @1 %.3f  @5 %.3f  @10 %.3f" % (hit1, hit5, hit10)
    )

    # ---- draw --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) top-k cosine, exact vs shipped HNSW
    ax_a.fill_between(ks, q1_exact, q3_exact, color=C_JOINT, alpha=0.20, label="exact IQR")
    ax_a.plot(ks, med_exact, "o-", color=C_JOINT, lw=2, ms=7, label="exact search (median)")
    if hnsw_med is not None:
        ax_a.plot(
            ks,
            hnsw_med,
            "s--",
            color=C_ANN,
            lw=1.8,
            ms=6,
            label=hnsw_note + " (median)",
        )
    ax_a.set_xscale("log")
    ax_a.set_xticks(list(ks))
    ax_a.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_a.set_xlabel("Retrieval rank k")
    ax_a.set_ylabel("Cosine similarity")
    ax_a.set_title("Top-k cosine, self excluded")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="lower left", fontsize=8)
    _panel(ax_a, "a")

    # (b) top-1 vs random, joint space, raw and mean-centred
    bins = np.linspace(-0.4, 1.0, 90)
    ax_b.hist(
        top1,
        bins=bins,
        color=C_JOINT,
        alpha=0.75,
        label="query " + _RA + " top-1, exact (median %.3f)" % float(np.median(top1)),
    )
    ax_b.hist(
        rand_raw,
        bins=bins,
        color=C_RANDOM,
        alpha=0.75,
        label="query " + _RA + " random, raw (median %.3f)" % float(np.median(rand_raw)),
    )
    ax_b.hist(
        rand_centred,
        bins=bins,
        histtype="step",
        color=C_CENTRED,
        lw=1.6,
        label="query "
        + _RA
        + " random, mean-centred (median %.3f)" % float(np.median(rand_centred)),
    )
    ax_b.axvline(0.0, color="#455a64", lw=0.8, ls=":")
    ax_b.set_xlabel("Cosine similarity")
    ax_b.set_ylabel("Pairs")
    ax_b.set_title("Joint space: retrieval vs. random baseline")
    ax_b.legend(loc="upper left", fontsize=7.5)
    _panel(ax_b, "b")

    # (c) the anisotropy floor in pocket space
    bins_c = np.linspace(-0.4, 1.0, 90)
    ax_c.hist(
        prand_raw,
        bins=bins_c,
        color=C_POCKET,
        alpha=0.70,
        label="random pocket pair, raw (median %.3f)" % float(np.median(prand_raw)),
    )
    ax_c.hist(
        prand_centred,
        bins=bins_c,
        color=C_CENTRED,
        alpha=0.70,
        label="random pocket pair, mean-centred (median %.3f)" % float(np.median(prand_centred)),
    )
    ax_c.axvline(0.0, color="#455a64", lw=0.8, ls=":")
    ax_c.set_xlabel("Cosine similarity")
    ax_c.set_ylabel("Pairs")
    ax_c.set_title("Anisotropy floor (pocket space)")
    ax_c.legend(loc="upper left", fontsize=7.5)
    ax_c.text(
        0.98,
        0.55,
        "||corpus mean|| = %.2fx a typical row norm.\n"
        "Absolute cosine magnitudes are an offset;\n"
        "only the ORDERING is meaningful." % aniso_pocket,
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=C_ALERT,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec=C_ALERT, lw=0.8),
    )
    _panel(ax_c, "c")

    # (d) ANN recall of the shipped index
    if recall:
        efs = sorted(recall)
        r1s = [recall[e][0] for e in efs]
        r10s = [recall[e][1] for e in efs]
        ax_d.plot(efs, r1s, "o-", color=C_ANN, lw=2, ms=7, label="recall@1 vs exact")
        ax_d.plot(efs, r10s, "s--", color="#6d4c41", lw=1.8, ms=6, label="recall@10 vs exact")
        ax_d.set_xscale("log")
        ax_d.set_xticks(efs)
        ax_d.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax_d.set_ylim(min(0.9, min(r1s) - 0.02), 1.005)
        ax_d.set_xlabel("HNSW efSearch")
        ax_d.set_ylabel("Recall vs. exact search")
        ax_d.set_title("Shipped HNSW index vs. exact search")
        ax_d.grid(True, alpha=0.3)
        ax_d.legend(loc="lower right", fontsize=8)
        for e, v in zip(efs, r1s):
            ax_d.annotate(
                "%.3f" % v,
                (e, v),
                textcoords="offset points",
                xytext=(0, 7),
                fontsize=7,
                ha="center",
            )
    else:
        ax_d.text(
            0.5,
            0.5,
            "shipped HNSW index not evaluated\n(faiss or retrieval.faiss unavailable)",
            ha="center",
            va="center",
            fontsize=9,
            color=C_ALERT,
        )
        ax_d.axis("off")
    _panel(ax_d, "d")

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    _footer(
        fig,
        "Panels a, b: exact chunked inner-product search over the carried-over joined rows "
        "(pocket block = superseded all-chain encoding); panel a also plots the shipped "
        "HNSW median. Panel c: released per-chain pocket matrix. Panel d: shipped "
        "retrieval.faiss HNSW graph. n_eval = %s, random pairs n = %s, "
        "numpy default_rng seed %d." % (_thousands(n_eval), _thousands(n_rand), seed),
    )
    _save(fig, out_dir, "fig3_technical_validation")

    # Sidecar, so the prose can be checked against the measurement.
    # Figure 3 was the only figure without one, and it is the figure whose
    # numbers have drifted from the text most often - including the caption
    # mismatch Reviewer 3 opened this round with.
    summary = {
        "seed": seed,
        "n_eval": int(n_eval),
        "n_rand": int(n_rand),
        "topk": {str(k): m for k, m in zip(ks, med_exact)},
        "exact_top1_median": float(np.median(top1)),
        "joint_random_raw_median": float(np.median(rand_raw)),
        "joint_random_centred_median": float(np.median(rand_centred)),
        "pocket_random_raw_median": float(np.median(prand_raw)),
        "pocket_random_centred_median": float(np.median(prand_centred)),
        "anisotropy_joint": aniso_joint,
        "anisotropy_pocket": aniso_pocket,
        "hnsw_recall": recall,
    }
    with open(os.path.join(out_dir, "fig3_technical_validation.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote fig3_technical_validation.json (measured values)")


# ---------------------------------------------------------------------------
# Figure 4: per-family stratified retrieval
# ---------------------------------------------------------------------------


def _hit_at_k_matrix(labels, qidx, joined, k=10, center=None):
    """Boolean (nq, k) same-label matrix for the top-k non-self neighbours."""
    import numpy as np

    Q = _gather(joined, qidx)
    if center is not None:
        Q = Q - center
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    _, I = _topk_chunked(Q, joined, k, exclude=qidx, center=center)
    out = np.zeros((len(qidx), k), dtype=bool)
    for r, qi in enumerate(qidx):
        lq = labels[qi]
        for c in range(k):
            j = I[r, c]
            out[r, c] = bool(j >= 0 and lq and labels[j] == lq)
    return out


def _random_baseline_at_k(counts, n_total, k):
    """P(at least one of k uniform draws without replacement shares the label)."""
    p = 1.0
    same = max(0, counts - 1)
    pool = n_total - 1
    for j in range(k):
        num = pool - same - j
        den = pool - j
        if den <= 0:
            return 1.0
        p *= max(0.0, num) / float(den)
    return 1.0 - p


def make_figure_4(
    out_dir,
    retrieval_dir,
    seed=42,
    sifts_dir=".",
    allow_download=True,
    min_family=20,
    max_families=12,
    max_queries_per_family=300,
):
    """Per-family stratified same-UniProt retrieval.

    The defaults (top 12 Pfam families, >= 20 members, <= 300 queries each,
    seed 42) reproduce the family SET used by notebook 08 cell 13, which is
    where the per-family numbers quoted in the response letter come from.
    Widening --max_families changes which family is the weakest and will
    therefore contradict the letter unless the letter is updated too.
    """
    import numpy as np

    core = _load_core(retrieval_dir)
    meta, joined, n = core["meta"], core["joined"], core["n"]
    rng = np.random.default_rng(seed)

    u_lab = _load_sifts_labels(sifts_dir, meta, "uniprot", allow_download)
    p_lab = _load_sifts_labels(sifts_dir, meta, "pfam", allow_download)

    uni_counts = collections.Counter([u for u in u_lab if u])
    fam_counts = collections.Counter([p for p in p_lab if p])
    families = [f for f, c in fam_counts.most_common() if c >= min_family][:max_families]
    if not families:
        raise SystemExit("no Pfam family with >= %d members; lower --min_family" % min_family)

    rows = []
    for fam in families:
        members = np.flatnonzero(p_lab == fam).astype(np.int64)
        if len(members) > max_queries_per_family:
            qq = np.sort(rng.choice(members, max_queries_per_family, replace=False))
        else:
            qq = np.sort(members)
        qq = np.array([i for i in qq if u_lab[i]], dtype=np.int64)
        if len(qq) < 10:
            continue
        hits = _hit_at_k_matrix(u_lab, qq, joined, k=10)
        h1 = float(hits[:, 0].mean())
        h10 = float(hits.any(axis=1).mean())
        # size-adjusted random baseline for the metric ACTUALLY measured
        # (same-UniProt), averaged over the query set's own UniProt priors.
        b1 = float(np.mean([_random_baseline_at_k(uni_counts[u_lab[i]], n, 1) for i in qq]))
        b10 = float(np.mean([_random_baseline_at_k(uni_counts[u_lab[i]], n, 10) for i in qq]))
        # the family-size proxy the audit notebook used, reported for continuity
        b1_fam = (len(members) - 1) / float(n - 1)
        rows.append(
            {
                "pfam": fam,
                "n_family": int(len(members)),
                "n_queried": int(len(qq)),
                "hit1": h1,
                "hit10": h10,
                "base1": b1,
                "base10": b10,
                "base1_family_proxy": b1_fam,
            }
        )
        print(
            "  %-9s n=%-6s queried=%-5s hit@1 %.3f  hit@10 %.3f  baseline@1 %.5f"
            % (fam, _thousands(len(members)), _thousands(len(qq)), h1, h10, b1)
        )

    if not rows:
        raise SystemExit(
            "no Pfam family survived the query filter: every candidate family had "
            "fewer than 10 members with a resolvable SIFTS UniProt accession. "
            "Check --sifts_dir (%s) and lower --min_family (currently %d)."
            % (sifts_dir, min_family)
        )

    rows.sort(key=lambda r: r["hit1"])
    names = _pfam_names(
        [r["pfam"] for r in rows],
        cache_path=os.path.join(out_dir, "pfam_names_cache.json"),
        fetch=allow_download,
    )

    _headline("Figure 4 (per-family stratified retrieval)")
    print("  metric: fraction of top-k non-self neighbours sharing the query's SIFTS UniProt")
    print(
        "  stratification: SIFTS Pfam, families with >= %d members, top %d by size"
        % (min_family, max_families)
    )
    print("  seed %d, <= %d queries per family, exact search" % (seed, max_queries_per_family))
    worst = rows[0]
    best = rows[-1]
    print(
        "  WORST family: %s (%s)  n=%s  hit@1 %.3f  hit@10 %.3f"
        % (
            worst["pfam"],
            names[worst["pfam"]],
            _thousands(worst["n_family"]),
            worst["hit1"],
            worst["hit10"],
        )
    )
    print(
        "  BEST  family: %s (%s)  n=%s  hit@1 %.3f  hit@10 %.3f"
        % (
            best["pfam"],
            names[best["pfam"]],
            _thousands(best["n_family"]),
            best["hit1"],
            best["hit10"],
        )
    )
    if worst["hit1"] > 0:
        spread = "%.1fx" % (best["hit1"] / worst["hit1"])
    else:
        spread = "undefined, weakest family is exactly 0"
    print(
        "  spread across families: hit@1 %.3f - %.3f (%s), pooled mean %.3f"
        % (
            worst["hit1"],
            best["hit1"],
            spread,
            float(np.mean([r["hit1"] for r in rows])),
        )
    )
    ratios = [r["hit1"] / r["base1"] for r in rows if r["base1"] > 0]
    if ratios and min(ratios) > 1.0:
        print("  every family beats its size-adjusted random baseline by >= %.0fx" % min(ratios))
    elif ratios:
        print(
            "  WARNING: the weakest family is only %.2fx its size-adjusted random "
            "baseline - do NOT write 'every family beats chance' in the caption" % min(ratios)
        )

    labels = []
    for r in rows:
        nm = names[r["pfam"]]
        if nm == r["pfam"]:
            labels.append("%s  (n=%s)" % (r["pfam"], _thousands(r["n_family"])))
        else:
            labels.append("%s %s  (n=%s)" % (r["pfam"], nm[:30], _thousands(r["n_family"])))

    y = np.arange(len(rows))
    fig, (ax1, ax10) = plt.subplots(1, 2, figsize=(12.6, 0.42 * len(rows) + 3.0), sharey=True)

    baseline_handle = None
    for ax, key, bkey, title, colour in (
        (ax1, "hit1", "base1", "hit@1 (same UniProt)\nsorted, weakest family at the top", C_JOINT),
        (ax10, "hit10", "base10", "hit@10 (same UniProt)\nsame family order as panel a", C_POCKET),
    ):
        vals = [r[key] for r in rows]
        cols = [C_ALERT if i < 3 else colour for i in range(len(rows))]
        ax.barh(y, vals, color=cols, alpha=0.88, height=0.68, edgecolor="white")
        base = [r[bkey] for r in rows]
        baseline_handle = ax.scatter(
            base,
            y,
            marker="|",
            s=170,
            linewidths=2.0,
            color="#212121",
            zorder=5,
            label="size-adjusted random baseline (same-UniProt prior)",
        )
        for i, v in enumerate(vals):
            ax.text(min(v + 0.015, 0.99), i, "%.3f" % v, va="center", fontsize=7.5)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Fraction of queries")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()  # weakest families at the top
    _panel_pts(ax1, "a", dx=-150, dy=6)
    _panel_pts(ax10, "b", dx=-30, dy=6)
    fig.suptitle(
        "Retrieval accuracy is not uniform across protein families", fontsize=13, weight="bold"
    )
    if baseline_handle is not None:
        fig.legend(
            handles=[baseline_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.028),
            fontsize=8,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.065, 1, 0.945))
    _footer(
        fig,
        "Exact search over the joined index; queries drawn per family with numpy default_rng "
        "seed %d, <= %d per family. Baseline = P(a uniform draw shares the query's UniProt), "
        "averaged over the family's queries. Red = the three weakest families by hit@1."
        % (seed, max_queries_per_family),
    )
    _save(fig, out_dir, "fig4_stratified_retrieval")

    with open(os.path.join(out_dir, "fig4_stratified_retrieval.json"), "w") as fh:
        json.dump({"seed": seed, "rows": rows, "names": names}, fh, indent=2)
    print("wrote fig4_stratified_retrieval.json (per-family table)")


# ---------------------------------------------------------------------------
# Figure 5: embedding geometry
# ---------------------------------------------------------------------------


def _alpha_sweep(core, qidx, alphas, chunk=8192):
    """Exact top-1 under u = [sqrt(a) p_hat ; sqrt(1-a) l_hat] for each alpha.

    cos(u_i, u_j) = a * cos_pocket + (1-a) * cos_ligand exactly, so one pass
    over the corpus produces every alpha at once and the 3,328-d
    concatenation is never materialised.
    """
    import numpy as np

    pocket, ligand = core["pocket"], core["ligand"]
    p_off, l_off = core["p_off"], core["l_off"]
    n = core["n"]
    nq = len(qidx)

    Qp = _gather(pocket, p_off[qidx])
    Qp /= np.linalg.norm(Qp, axis=1, keepdims=True) + 1e-12
    Ql = _gather(ligand, l_off[qidx])
    Ql /= np.linalg.norm(Ql, axis=1, keepdims=True) + 1e-12

    best_v = {a: np.full(nq, -np.inf, dtype=np.float32) for a in alphas}
    best_j = {a: np.full(nq, -1, dtype=np.int64) for a in alphas}

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Bp = _gather(pocket, p_off[s:e])
        Bp /= np.linalg.norm(Bp, axis=1, keepdims=True) + 1e-12
        Bl = _gather(ligand, l_off[s:e])
        Bl /= np.linalg.norm(Bl, axis=1, keepdims=True) + 1e-12
        Cp = Qp @ Bp.T
        Cl = Ql @ Bl.T
        del Bp, Bl
        m = (qidx >= s) & (qidx < e)
        hit = np.flatnonzero(m)
        for a in alphas:
            # cos(u_i, u_j) = a * cos_pocket + (1 - a) * cos_ligand, exactly
            S = a * Cp + (1.0 - a) * Cl
            if hit.size:
                S[hit, qidx[hit] - s] = -np.inf
            loc = np.argmax(S, axis=1)
            val = S[np.arange(nq), loc]
            upd = val > best_v[a]
            best_v[a][upd] = val[upd]
            best_j[a][upd] = (loc + s)[upd]
            del S
        del Cp, Cl
    return best_v, best_j


def make_figure_5(
    out_dir,
    retrieval_dir,
    seed=42,
    sifts_dir=".",
    allow_download=True,
    n_sweep=3000,
    alphas=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
    n_nearcurve=20000,
):
    import numpy as np

    core = _load_core(retrieval_dir)
    meta, joined, n = core["meta"], core["joined"], core["n"]
    rng = np.random.default_rng(seed)

    # ---- (a)(b) norms and w_p, streamed -----------------------------------
    print("streaming pocket / ligand norms...")
    _p_mean, pn = _chunked_stats(core["pocket"], rows=core["p_off"])
    _l_mean, ln = _chunked_stats(core["ligand"], rows=core["l_off"])
    w_p = pn**2 / (pn**2 + ln**2 + 1e-12)

    # ---- (c) alpha sweep ---------------------------------------------------
    u_lab = _load_sifts_labels(sifts_dir, meta, "uniprot", allow_download)
    ccd = np.array([m["ligand_resname"] for m in meta], dtype=object)
    qidx = np.sort(rng.choice(n, size=int(min(n_sweep, n)), replace=False)).astype(np.int64)
    print("alpha sweep over %d values, %s queries..." % (len(alphas), _thousands(len(qidx))))
    _bv, bj = _alpha_sweep(core, qidx, list(alphas))

    uni_ok = np.array([bool(u_lab[i]) for i in qidx])
    sweep_uni, sweep_ccd = [], []
    for a in alphas:
        nb = bj[a]
        same_u = np.array([u_lab[i] == u_lab[j] for i, j in zip(qidx, nb)])
        same_c = np.array([ccd[i] == ccd[j] for i, j in zip(qidx, nb)])
        sweep_uni.append(float(same_u[uni_ok].mean()))
        sweep_ccd.append(float(same_c.mean()))

    # the shipped concatenation, same query set, exact search
    Qj = _gather(joined, qidx)
    _, Ij = _topk_chunked(Qj, joined, 1, exclude=qidx)
    nb_ship = Ij[:, 0]
    ship_uni = float(np.array([u_lab[i] == u_lab[j] for i, j in zip(qidx, nb_ship)])[uni_ok].mean())
    ship_ccd = float(np.mean([ccd[i] == ccd[j] for i, j in zip(qidx, nb_ship)]))
    del Qj

    # crossover of the two curves (linear interpolation on the alpha grid)
    a_arr = np.array(alphas, dtype=np.float64)
    d = np.array(sweep_uni) - np.array(sweep_ccd)
    cross = None
    for i in range(len(a_arr) - 1):
        if d[i] == 0.0:
            cross = float(a_arr[i])
            break
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            cross = float(a_arr[i] + t * (a_arr[i + 1] - a_arr[i]))
            break

    # ---- (d) near-duplicate / leakage curve --------------------------------
    # A byte-identical pair does NOT score exactly 1.0: the inner product is a
    # float32 accumulation over 3,328 terms and lands within ~1e-6 of 1. Testing
    # `>= t` at t = 1.0 therefore drops roughly half of the exact duplicates and
    # makes the curve's endpoint fall BELOW the measured exact-duplicate rate,
    # which is impossible. Compare with a tolerance so t = 1.0 means
    # "numerically identical", matching the "cosine = 1.00" row of the response
    # letter's leakage table.
    _COS_TOL = 1e-5
    thresholds = np.linspace(0.80, 1.0, 41)
    leak_curve, dup_curve = None, None
    exact_dup_val = None
    try:
        train_ix, val_ix = _load_splits(retrieval_dir)
        Qv = _gather(joined, val_ix)
        Dv, _ = _topk_chunked(Qv, joined, 1, rows=train_ix)
        leak_curve = np.array([float((Dv[:, 0] >= t - _COS_TOL).mean()) for t in thresholds])
        del Qv
        # exact duplicates: md5 of the float32 rows
        tr_hash = set()
        for s in range(0, len(train_ix), 8192):
            blk = _gather(joined, train_ix[s : s + 8192])
            for r in range(blk.shape[0]):
                tr_hash.add(hashlib.md5(blk[r].tobytes()).hexdigest())
        n_dup = 0
        for s in range(0, len(val_ix), 8192):
            blk = _gather(joined, val_ix[s : s + 8192])
            for r in range(blk.shape[0]):
                if hashlib.md5(blk[r].tobytes()).hexdigest() in tr_hash:
                    n_dup += 1
        exact_dup_val = n_dup / float(len(val_ix))
        del tr_hash
    except (OSError, ValueError, KeyError) as exc:
        print("WARNING: split leakage curve unavailable (%s)" % exc)

    qs = np.sort(rng.choice(n, size=int(min(n_nearcurve, n)), replace=False)).astype(np.int64)
    Qc = _gather(joined, qs)
    Dc, _ = _topk_chunked(Qc, joined, 1, exclude=qs)
    dup_curve = np.array([float((Dc[:, 0] >= t - _COS_TOL).mean()) for t in thresholds])
    del Qc

    # ---- headline numbers --------------------------------------------------
    _headline("Figure 5 (embedding geometry)")
    _stat_line("||pocket|| (ESM-2 mean-pooled)", pn)
    _stat_line("||ligand|| (ECFP4, = sqrt(on-bits))", ln)
    _stat_line("w_p = ||p||^2 / (||p||^2 + ||l||^2)", w_p)
    print("  w_p std %.4f -> the implicit modality weighting is NOT constant" % float(w_p.std()))
    print(
        "  alpha (1 = pocket only, 0 = ligand only), n_query = %s, seed %d"
        % (_thousands(len(qidx)), seed)
    )
    for a, su, sc in zip(alphas, sweep_uni, sweep_ccd):
        print("    alpha=%.3f  same-UniProt hit@1 %.4f   same-CCD top-1 %.4f" % (a, su, sc))
    print(
        "  shipped concatenation: same-UniProt hit@1 %.4f   same-CCD top-1 %.4f"
        % (ship_uni, ship_ccd)
    )
    if cross is not None:
        print("  curves cross at alpha ~ %.3f" % cross)
    if leak_curve is not None:
        for t_want in (0.90, 0.95, 0.99):
            i = int(np.argmin(np.abs(thresholds - t_want)))
            print(
                "  frac of val with a train neighbour >= %.2f cosine: %.4f"
                % (thresholds[i], leak_curve[i])
            )
    if exact_dup_val is not None:
        print("  frac of val that is an EXACT duplicate of a train row: %.4f" % exact_dup_val)

    # ---- draw --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) norms
    lo = float(min(pn.min(), ln.min()))
    hi = float(max(np.percentile(pn, 99.9), np.percentile(ln, 99.9)))
    binsn = np.linspace(lo, hi, 80)
    ax_a.hist(
        pn,
        bins=binsn,
        color=C_POCKET,
        alpha=0.62,
        label="||pocket|| (median %.2f)" % float(np.median(pn)),
    )
    ax_a.hist(
        ln,
        bins=binsn,
        color=C_LIGAND,
        alpha=0.62,
        label="||ligand|| (median %.2f)" % float(np.median(ln)),
    )
    ax_a.axvline(float(np.median(pn)), color=C_POCKET, ls="--", lw=1)
    ax_a.axvline(float(np.median(ln)), color=C_LIGAND, ls="--", lw=1)
    ax_a.set_xlabel("L2 norm of the raw modality vector")
    ax_a.set_ylabel("Complexes")
    ax_a.set_title("The two modalities carry comparable norm")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.text(
        0.02,
        0.95,
        "||ligand|| = sqrt(Morgan on-bits)\nmedian %d bits" % int(round(float(np.median(ln)) ** 2)),
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#455a64",
    )
    _panel(ax_a, "a")

    # (b) w_p
    ax_b.hist(w_p, bins=np.linspace(0, 1, 90), color=C_JOINT, alpha=0.85, edgecolor="white")
    med_w = float(np.median(w_p))
    q1w, q3w = float(np.percentile(w_p, 25)), float(np.percentile(w_p, 75))
    ax_b.axvspan(q1w, q3w, color=C_JOINT, alpha=0.15, label="IQR [%.3f, %.3f]" % (q1w, q3w))
    ax_b.axvline(med_w, color=C_ALERT, ls="--", lw=1.4, label="median %.3f" % med_w)
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("$w_p$ = pocket share of the joint squared norm")
    ax_b.set_ylabel("Complexes")
    ax_b.set_title("The implicit modality weight is per-entry, not fixed")
    ax_b.legend(loc="upper right", fontsize=8)
    ax_b.text(
        0.02,
        0.95,
        "range %.3f - %.3f\nstd %.3f\nEvery pair is compared under a\n"
        "different implicit weighting." % (float(w_p.min()), float(w_p.max()), float(w_p.std())),
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=C_ALERT,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec=C_ALERT, lw=0.8),
    )
    _panel(ax_b, "b")

    # (c) alpha sweep -- the two-modality argument
    ax_c.plot(
        alphas,
        sweep_uni,
        "o-",
        color=C_POCKET,
        lw=2,
        ms=6,
        label="same-UniProt hit@1 (target identity)",
    )
    ax_c.plot(
        alphas,
        sweep_ccd,
        "s-",
        color=C_LIGAND,
        lw=2,
        ms=6,
        label="same-CCD top-1 (ligand identity)",
    )
    ax_c.axhline(ship_uni, color=C_POCKET, ls=":", lw=1.2)
    ax_c.axhline(ship_ccd, color=C_LIGAND, ls=":", lw=1.2)
    ax_c.annotate(
        "shipped concatenation: %.3f" % ship_uni,
        xy=(0.03, ship_uni),
        xytext=(0.03, ship_uni + 0.035),
        fontsize=7.5,
        color=C_POCKET,
    )
    ax_c.annotate(
        "shipped concatenation: %.3f" % ship_ccd,
        xy=(0.03, ship_ccd),
        xytext=(0.03, ship_ccd - 0.06),
        fontsize=7.5,
        color=C_LIGAND,
    )
    if cross is not None:
        ax_c.axvline(cross, color="#455a64", ls="--", lw=1)
        ax_c.text(
            cross,
            0.02,
            " crossover " + r"$\alpha \approx$" + " %.2f" % cross,
            fontsize=7.5,
            color="#455a64",
            rotation=90,
            va="bottom",
        )
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(0, 1.0)
    ax_c.set_xlabel(r"$\alpha$   (0 = ligand only,  1 = pocket only)")
    ax_c.set_ylabel("Top-1 agreement")
    ax_c.set_title("No single mixture is best for both questions")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="center right", fontsize=8)
    _panel(ax_c, "c")

    # (d) near-duplicate / leakage
    if leak_curve is not None:
        ax_d.plot(
            thresholds,
            leak_curve,
            "-",
            color=C_ALERT,
            lw=2.2,
            label="val entry with a TRAIN neighbour " + _GE + " t",
        )
        i99 = int(np.argmin(np.abs(thresholds - 0.99)))
        ax_d.plot([0.99], [leak_curve[i99]], "o", color=C_ALERT, ms=7)
        ax_d.annotate(
            "%.3f of val has a train\nneighbour at cosine " % leak_curve[i99] + _GE + " 0.99",
            xy=(0.99, leak_curve[i99]),
            textcoords="offset points",
            xytext=(-96, 26),
            fontsize=7.5,
            color=C_ALERT,
            arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=0.9),
        )
    if exact_dup_val is not None:
        ax_d.axhline(
            exact_dup_val,
            color="#6d4c41",
            ls="--",
            lw=1.2,
            label="exact duplicate of a train row (%.3f)" % exact_dup_val,
        )
    ax_d.plot(
        thresholds,
        dup_curve,
        "-",
        color=C_RANDOM,
        lw=2.0,
        label="any entry with a corpus neighbour " + _GE + " t",
    )
    ax_d.set_xlim(0.80, 1.0)
    ax_d.set_ylim(0, 1.0)
    ax_d.set_xlabel("Cosine threshold t")
    ax_d.set_ylabel("Fraction of queries")
    ax_d.set_title("Near-duplicate structure of the released random split")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="upper right", fontsize=7.5)
    _panel(ax_d, "d")

    fig.suptitle(
        "Geometry of the released embeddings: two modalities, not one vector",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    _footer(
        fig,
        "Panels a-b: all %s indexed complexes, no sampling. Panel c: %s queries, exact search, "
        "seed %d. Panel d: val vs train exact search, corpus curve on %s sampled queries, "
        "same seed; 1e-5 float32 tolerance on t, so t = 1.00 reads as identical."
        % (_thousands(n), _thousands(len(qidx)), seed, _thousands(len(qs))),
    )
    _save(fig, out_dir, "fig5_embedding_geometry")

    summary = {
        "seed": seed,
        "alphas": list(alphas),
        "same_uniprot_hit1": sweep_uni,
        "same_ccd_top1": sweep_ccd,
        "shipped_concat_same_uniprot_hit1": ship_uni,
        "shipped_concat_same_ccd_top1": ship_ccd,
        "crossover_alpha": cross,
        "w_p_median": med_w,
        "w_p_iqr": [q1w, q3w],
        "w_p_range": [float(w_p.min()), float(w_p.max())],
        "pocket_norm_median": float(np.median(pn)),
        "ligand_norm_median": float(np.median(ln)),
        "val_exact_duplicate_fraction": exact_dup_val,
        "threshold_float32_tolerance": _COS_TOL,
        "thresholds": thresholds.tolist(),
        "val_leak_curve": None if leak_curve is None else leak_curve.tolist(),
        "corpus_nearduplicate_curve": dup_curve.tolist(),
    }
    with open(os.path.join(out_dir, "fig5_embedding_geometry.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote fig5_embedding_geometry.json (sweep + curves)")


# ---------------------------------------------------------------------------
# Figure 6: UMAP coloured by EC class + ambient-space separation metrics
# ---------------------------------------------------------------------------

_EC_NAMES = {
    "1": "1 Oxidoreductases",
    "2": "2 Transferases",
    "3": "3 Hydrolases",
    "4": "4 Lyases",
    "5": "5 Isomerases",
    "6": "6 Ligases",
    "7": "7 Translocases",
}
_EC_COLORS = {
    "1": "#1f77b4",
    "2": "#d62728",
    "3": "#2ca02c",
    "4": "#9467bd",
    "5": "#ff7f0e",
    "6": "#8c564b",
    "7": "#17becf",
}
_EC_NONE = "none"
_EC_NONE_COLOR = "#c7cdd1"


def _pca2(X, seed=42, iters=30):
    """Deterministic top-2 principal directions via orthogonal iteration."""
    import numpy as np

    Xc = X - X.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(Xc.shape[1], 2)).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    for _ in range(iters):
        Z = Xc.T @ (Xc @ Q)
        Q, _ = np.linalg.qr(Z)
    return Xc @ Q


def _knn_purity(labels, qidx, joined, k=10, center=None):
    hits = _hit_at_k_matrix(labels, qidx, joined, k=k, center=center)
    return float(hits.mean())


def _pair_cosines(joined, ia, ib, center=None):
    import numpy as np

    A = _gather(joined, ia)
    B = _gather(joined, ib)
    if center is not None:
        A = A - center
        B = B - center
    num = (A.astype(np.float64) * B.astype(np.float64)).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den


def make_figure_6(
    out_dir,
    retrieval_dir,
    seed=42,
    sifts_dir=".",
    allow_download=True,
    n_sample=20000,
    n_purity=3000,
    n_pairs=20000,
    n_neighbors=30,
    min_dist=0.1,
    ec_only=False,
):
    """EC-class structure of the joined space: projection + ambient metrics.

    Defaults match notebook 10 cell 10, whose generated caption the manuscript
    uses verbatim: 20,000 points, cosine metric, n_neighbors=30, min_dist=0.1,
    random_state=42. Changing them desynchronises the figure from that caption.

    The projected sample is drawn from the WHOLE index, not from the
    EC-annotated subset. Roughly 40 % of the corpus carries no SIFTS EC
    annotation, and dropping it would make the figure look tidier than the data
    is; the unannotated points are drawn in grey and counted in the legend.
    --ec_only restores the annotated-only view for a supplementary panel.
    """
    import numpy as np

    core = _load_core(retrieval_dir)
    meta, joined, n = core["meta"], core["joined"], core["n"]
    rng = np.random.default_rng(seed)

    e_lab = _load_sifts_labels(sifts_dir, meta, "ec", allow_download)
    labelled = np.flatnonzero(np.array([lab in _EC_NAMES for lab in e_lab])).astype(np.int64)
    if len(labelled) == 0:
        raise SystemExit("no EC-classified entries; check the SIFTS enzyme file")

    pool = labelled if ec_only else np.arange(n, dtype=np.int64)
    sample = (
        np.sort(rng.choice(pool, size=int(min(n_sample, len(pool))), replace=False))
        if len(pool) > n_sample
        else pool
    ).astype(np.int64)
    X = _gather(joined, sample)
    y = np.array(
        [e_lab[i] if e_lab[i] in _EC_NAMES else _EC_NONE for i in sample],
        dtype=object,
    )

    # ---- ambient-space metrics (the actual evidence; UMAP is decoration) ---
    print("computing ambient-space separation metrics...")
    j_mean, _ = _chunked_stats(joined)
    pq = (
        np.sort(rng.choice(labelled, size=int(min(n_purity, len(labelled))), replace=False))
        if len(labelled) > n_purity
        else labelled
    ).astype(np.int64)
    purity_raw = _knn_purity(e_lab, pq, joined, k=10)
    purity_centred = _knn_purity(e_lab, pq, joined, k=10, center=j_mean)

    # Within/between pair statistics are about EC classes, so they are computed
    # over the EC-annotated members of the sample only.
    lab_sample = sample[y != _EC_NONE]
    lab_y = y[y != _EC_NONE]
    by_class = collections.defaultdict(list)
    for pos, i in enumerate(lab_sample):
        by_class[lab_y[pos]].append(int(i))
    within_a, within_b = [], []
    for _cls, members in by_class.items():
        if len(members) < 2:
            continue
        take = int(max(1, n_pairs * len(members) // max(1, len(lab_sample))))
        arr = np.array(members, dtype=np.int64)
        within_a.append(rng.choice(arr, size=take))
        within_b.append(rng.choice(arr, size=take))
    if not within_a:
        raise SystemExit("no EC class in the sample has >= 2 members; raise --umap_sample")
    within_a = np.concatenate(within_a)
    within_b = np.concatenate(within_b)
    keep = within_a != within_b
    within_a, within_b = within_a[keep], within_b[keep]

    between_a = rng.choice(lab_sample, size=n_pairs)
    between_b = rng.choice(lab_sample, size=n_pairs)
    lab_of = dict(zip(lab_sample.tolist(), lab_y.tolist()))
    keep = np.array([lab_of[int(a)] != lab_of[int(b)] for a, b in zip(between_a, between_b)])
    between_a, between_b = between_a[keep], between_b[keep]

    w_raw = _pair_cosines(joined, within_a, within_b)
    b_raw = _pair_cosines(joined, between_a, between_b)
    w_cen = _pair_cosines(joined, within_a, within_b, center=j_mean)
    b_cen = _pair_cosines(joined, between_a, between_b, center=j_mean)

    # Chance purity. Neighbours are drawn from the WHOLE index, so the chance
    # that a uniform draw shares an EC-annotated query's class is
    #   sum_c (N_c / N_labelled) * (N_c / n) = sum_c N_c^2 / (N_labelled * n),
    # NOT sum_c (N_c / N_labelled)^2. Using the latter inflates chance by
    # n / N_labelled (a factor of ~1.7 at 60 % EC coverage) and can make a
    # perfectly informative embedding look worse than chance.
    ec_counts = collections.Counter([lab for lab in e_lab if lab in _EC_NAMES])
    n_lab_total = int(sum(ec_counts.values()))
    chance = float(sum([c * c for c in ec_counts.values()]) / (float(n_lab_total) * float(n)))

    # Silhouette by EC class in the ambient space, raw and mean-centred. This is
    # the statistic notebook 10's locked caption quotes; the figure must carry it.
    sil_raw, sil_cen = None, None
    try:
        from sklearn.metrics import silhouette_score

        sil_n = int(min(10000, len(lab_sample)))
        sil_ix = np.sort(rng.choice(lab_sample, size=sil_n, replace=False))
        sil_lab = np.array([e_lab[i] for i in sil_ix], dtype=object)
        Xs = _gather(joined, sil_ix)
        Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
        Xc = _gather(joined, sil_ix) - j_mean
        Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        sil_raw = float(silhouette_score(Xs, sil_lab, metric="cosine"))
        sil_cen = float(silhouette_score(Xc, sil_lab, metric="cosine"))
        del Xs, Xc
    except ImportError:
        print("WARNING: scikit-learn unavailable; silhouette by EC class not computed")

    _headline("Figure 6 (UMAP by EC class) -- AMBIENT-SPACE metrics for the caption")
    print(
        "  EC-classified entries in the index : %s of %s (%.1f%%)"
        % (_thousands(len(labelled)), _thousands(n), 100.0 * len(labelled) / n)
    )
    print(
        "  projected sample                   : %s of %s (%s, seed %d)"
        % (
            _thousands(len(sample)),
            _thousands(len(pool)),
            "EC-annotated only" if ec_only else "whole index, unannotated shown in grey",
            seed,
        )
    )
    print("  kNN-10 purity by EC class, RAW     : %.4f" % purity_raw)
    print("  kNN-10 purity by EC class, CENTRED : %.4f" % purity_centred)
    print("  chance purity (uniform corpus draw): %.4f" % chance)
    print(
        "  purity over chance                 : %.2fx raw, %.2fx centred"
        % (purity_raw / max(1e-9, chance), purity_centred / max(1e-9, chance))
    )
    if sil_raw is not None:
        print("  silhouette by EC class, RAW        : %.4f" % sil_raw)
        print("  silhouette by EC class, CENTRED    : %.4f" % sil_cen)
    print("  mean cosine within EC class,  raw  : %.4f" % float(w_raw.mean()))
    print("  mean cosine between EC classes,raw : %.4f" % float(b_raw.mean()))
    print("  raw separation (within - between)  : %.4f" % float(w_raw.mean() - b_raw.mean()))
    print("  mean cosine within EC class, centred  : %.4f" % float(w_cen.mean()))
    print("  mean cosine between EC classes,centred: %.4f" % float(b_cen.mean()))
    print("  centred separation (within - between) : %.4f" % float(w_cen.mean() - b_cen.mean()))
    print("  NOTE the raw within/between gap is small in absolute terms because")
    print("       both sit on the same anisotropy offset; the centred gap and the")
    print("       purity-over-chance ratio are the honest separation statistics.")

    # ---- projection --------------------------------------------------------
    method = "UMAP"
    try:
        import umap  # noqa: WPS433

        print(
            "running UMAP (n_neighbors=%d, min_dist=%.2f, metric=cosine, random_state=%d)..."
            % (n_neighbors, min_dist, seed)
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="cosine",
            random_state=seed,
        )
        emb = reducer.fit_transform(X)
    except Exception as exc:
        print("WARNING: umap-learn unavailable or failed (%s); falling back to PCA" % exc)
        method = "PCA (umap-learn unavailable)"
        emb = _pca2(X, seed=seed)

    # ---- draw --------------------------------------------------------------
    fig, (ax_u, ax_m) = plt.subplots(
        1, 2, figsize=(12.4, 6.4), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # Grey (unannotated) is drawn FIRST so the annotated classes are not buried
    # under it, but it stays in the legend with its count so the figure never
    # implies the corpus is more fully annotated than it is.
    order = sorted(_EC_NAMES.keys(), key=lambda c: -ec_counts.get(c, 0))
    for cls in [_EC_NONE] + order:
        m = y == cls
        if not m.any():
            continue
        ax_u.scatter(
            emb[m, 0],
            emb[m, 1],
            s=2.0,
            alpha=0.30 if cls == _EC_NONE else 0.45,
            linewidths=0,
            color=_EC_NONE_COLOR if cls == _EC_NONE else _EC_COLORS[cls],
            label="%s (n=%s)"
            % (
                "no SIFTS EC annotation" if cls == _EC_NONE else _EC_NAMES[cls],
                _thousands(int(m.sum())),
            ),
            rasterized=True,
        )
    ax_u.set_xlabel(method + " 1")
    ax_u.set_ylabel(method + " 2")
    ax_u.set_title("Joined embeddings coloured by EC top-level class")
    ax_u.set_xticks([])
    ax_u.set_yticks([])
    leg = ax_u.legend(loc="upper right", fontsize=7.5, markerscale=4, framealpha=0.92)
    leg.set_title("EC class", prop={"size": 8})
    _panel(ax_u, "a", dx=-0.03)
    ax_u.text(
        0.01,
        0.01,
        "The projection is illustrative. The separation claim rests on the\n"
        "ambient-space statistics in panel b, computed on the full 3,328-d vectors.",
        transform=ax_u.transAxes,
        fontsize=7,
        color="#455a64",
        va="bottom",
    )

    # metrics panel: within vs between cosine, raw and centred
    xs = np.arange(2)
    width = 0.34
    ax_m.bar(
        xs - width / 2,
        [float(w_raw.mean()), float(w_cen.mean())],
        width,
        color=C_POCKET,
        alpha=0.85,
        label="within EC class",
    )
    ax_m.bar(
        xs + width / 2,
        [float(b_raw.mean()), float(b_cen.mean())],
        width,
        color=C_RANDOM,
        alpha=0.85,
        label="between EC classes",
    )
    for xi, (wv, bv) in enumerate([(w_raw.mean(), b_raw.mean()), (w_cen.mean(), b_cen.mean())]):
        ax_m.text(
            xi - width / 2,
            float(wv),
            "%.3f" % float(wv),
            ha="center",
            va="bottom" if wv >= 0 else "top",
            fontsize=8,
        )
        ax_m.text(
            xi + width / 2,
            float(bv),
            "%.3f" % float(bv),
            ha="center",
            va="bottom" if bv >= 0 else "top",
            fontsize=8,
        )
    ax_m.axhline(0, color="#455a64", lw=0.8)
    ax_m.set_xticks(xs)
    ax_m.set_xticklabels(["raw space", "mean-centred space"])
    ax_m.set_ylabel("Mean pairwise cosine")
    ax_m.set_title("Ambient-space separation by EC class")
    ax_m.legend(loc="upper right", fontsize=8)
    metrics_text = (
        "kNN-10 purity by EC class\n"
        "  raw      %.3f\n"
        "  centred  %.3f\n"
        "  chance   %.3f  (%.2fx / %.2fx over chance)"
        % (
            purity_raw,
            purity_centred,
            chance,
            purity_raw / max(1e-9, chance),
            purity_centred / max(1e-9, chance),
        )
    )
    if sil_raw is not None:
        metrics_text += "\nsilhouette  raw %.3f   centred %.3f" % (sil_raw, sil_cen)
    ax_m.text(
        0.02,
        0.02,
        metrics_text,
        transform=ax_m.transAxes,
        fontsize=8,
        va="bottom",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#eceff1", ec="#b0bec5"),
    )
    _panel(ax_m, "b", dx=-0.14)

    fig.suptitle("Enzyme-class structure of the joined embedding space", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    scope = "EC-classified complexes" if ec_only else "complexes sampled from the whole index"
    if method == "UMAP":
        proj_note = (
            "UMAP on %s %s (cosine metric, n_neighbors=%d, min_dist=%.2f, random_state=%d)."
            % (_thousands(len(sample)), scope, n_neighbors, min_dist, seed)
        )
    else:
        proj_note = "%s on %s %s (deterministic orthogonal iteration, seed %d)." % (
            method,
            _thousands(len(sample)),
            scope,
            seed,
        )
    _footer(
        fig,
        proj_note + " Purity on %s queries, pair statistics on ~%s pairs per group, "
        "numpy default_rng seed %d." % (_thousands(len(pq)), _thousands(len(within_a)), seed),
    )
    _save(fig, out_dir, "fig6_umap_ec_class")

    with open(os.path.join(out_dir, "fig6_umap_ec_class.json"), "w") as fh:
        json.dump(
            {
                "seed": seed,
                "projection": method,
                "projection_params": {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "metric": "cosine",
                    "random_state": seed,
                },
                "n_projected": int(len(sample)),
                "projected_pool": "ec_annotated_only" if ec_only else "whole_index",
                "n_unannotated_in_projection": int((y == _EC_NONE).sum()),
                "knn10_purity_ec_raw": purity_raw,
                "knn10_purity_ec_centred": purity_centred,
                "chance_purity": chance,
                "silhouette_ec_raw": sil_raw,
                "silhouette_ec_centred": sil_cen,
                "mean_cos_within_raw": float(w_raw.mean()),
                "mean_cos_between_raw": float(b_raw.mean()),
                "mean_cos_within_centred": float(w_cen.mean()),
                "mean_cos_between_centred": float(b_cen.mean()),
                "ec_counts_full_corpus": {k: int(v) for k, v in ec_counts.items()},
            },
            fh,
            indent=2,
        )
    print("wrote fig6_umap_ec_class.json (ambient-space metrics)")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Regenerate the Scientific Data figures (1-6).")
    parser.add_argument("--fig", required=True, help="figure number 1-6, or 'all'")
    parser.add_argument("--out", default="paper/figures")
    parser.add_argument("--retrieval_dir", help="HF snapshot dir; required for figs 2-6")
    parser.add_argument("--pocket_map", help="pocket_residue_map.parquet (fig 2c)")
    parser.add_argument("--sifts_dir", default=".", help="dir holding the SIFTS csv.gz files")
    parser.add_argument(
        "--no_download", action="store_true", help="never fetch SIFTS / Pfam names over the network"
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1000,
        help="fig 3 queries; 0 or >= index size means the full index",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_hnsw", action="store_true", help="fig 3 without retrieval.faiss")
    parser.add_argument(
        "--min_family",
        type=int,
        default=20,
        help="fig 4 minimum family size (20 = notebook 08 cell 13)",
    )
    parser.add_argument(
        "--max_families",
        type=int,
        default=12,
        help="fig 4 families to plot (12 = notebook 08 cell 13)",
    )
    parser.add_argument("--n_sweep", type=int, default=3000, help="fig 5 alpha-sweep queries")
    parser.add_argument("--umap_sample", type=int, default=20000, help="fig 6 points projected")
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument(
        "--ec_only",
        action="store_true",
        help="fig 6: project only EC-annotated entries (default projects the whole "
        "index and greys out the ~40%% with no SIFTS EC annotation)",
    )
    args = parser.parse_args()

    if args.fig.lower() == "all":
        figs = [1, 2, 3, 4, 5, 6]
    else:
        try:
            figs = [int(args.fig)]
        except ValueError:
            raise SystemExit("--fig must be 1-6 or 'all'")
        if figs[0] not in (1, 2, 3, 4, 5, 6):
            raise SystemExit("--fig must be 1-6 or 'all'")

    os.makedirs(args.out, exist_ok=True)
    allow_download = not args.no_download

    for f in figs:
        if f != 1 and not args.retrieval_dir:
            raise SystemExit("--retrieval_dir is required for fig %d" % f)
        print("")
        print("### Figure %d " % f + "#" * 58)
        if f == 1:
            make_figure_1(args.out)
        elif f == 2:
            make_figure_2(args.out, args.retrieval_dir, pocket_map=args.pocket_map)
        elif f == 3:
            make_figure_3(
                args.out,
                args.retrieval_dir,
                n_eval=args.n_eval,
                seed=args.seed,
                skip_hnsw=args.skip_hnsw,
            )
        elif f == 4:
            make_figure_4(
                args.out,
                args.retrieval_dir,
                seed=args.seed,
                sifts_dir=args.sifts_dir,
                allow_download=allow_download,
                min_family=args.min_family,
                max_families=args.max_families,
            )
        elif f == 5:
            make_figure_5(
                args.out,
                args.retrieval_dir,
                seed=args.seed,
                sifts_dir=args.sifts_dir,
                allow_download=allow_download,
                n_sweep=args.n_sweep,
            )
        elif f == 6:
            make_figure_6(
                args.out,
                args.retrieval_dir,
                seed=args.seed,
                sifts_dir=args.sifts_dir,
                allow_download=allow_download,
                n_sample=args.umap_sample,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                ec_only=args.ec_only,
            )


if __name__ == "__main__":
    main()
