# Sci Data revision — remaining reviewer analyses (encoders, overlap, diversity, panel, UMAP)

The reviewer asks that notebooks 08 and 09 did **not** measure. Everything
here runs off the released HuggingFace snapshot plus
`pocket_residue_map.parquet` + `esm_input_sequences.fasta.gz` produced by
notebook 09.

| § | Reviewer item | What is measured |
|---|---|---|
| 1 | **R1.3** | ESM-2 35M/150M/650M ladder vs **ProtT5-XL-UniRef50** (different model family) vs a **20-d amino-acid-composition control** (and a BLOSUM62-weighted variant), all pooling over *exactly the same* pocket residues |
| 2 | **R2.3** | PDB-ID overlap against PDBbind, BioLiP2, PLINDER, Binding MOAD |
| 3 | editor: "diversity" | unique UniProt / Pfam / EC, redundancy curve, Gini, top-N concentration |
| 4 | **R1.2** | 10-family qualitative retrieval panel + 2 deliberate negative cases |
| 5 | **R1.4** | silhouette by family (raw **and** mean-centred, cosine) + UMAP coloured by EC class |

### The one result that matters most

§1's composition control is not a footnote. A 20-dimensional bag of pocket
residues costs nothing to compute and needs no GPU. If it matches ESM-2 on
same-family retrieval, the honest claim in the manuscript is *"pocket
identity is largely recoverable from residue composition"*, not *"ESM-2
captures pocket chemistry"*. Cell 5 prints that comparison as a headline
verdict before anything else.

### Runtime

| Part | Needs | Time |
|---|---|---|
| §1 ESM-2 ladder (5,000 seqs x 3 models) | GPU | ~25 min |
| §1 ProtT5-XL | GPU >= 15 GB, or fp16 T4 | ~20 min (+ 5 GB download) |
| §1 composition controls | CPU | seconds |
| §2 overlap | network | ~10 min |
| §3, §4 | CPU | ~5 min |
| §5 silhouette + UMAP | CPU | ~20 min |

**Run Cell 0 and Cell 1 first.** After that the sections are independent — if
one fails, skip it and continue; Cell 12 writes whatever `RESULTS` holds.

---

## Cell 0 — Setup: dependencies, Drive, HF snapshot, notebook-09 artifacts

```python
!pip -q install huggingface_hub faiss-cpu numpy pandas scipy pyarrow requests h5py \
    scikit-learn umap-learn matplotlib fair-esm transformers sentencepiece fsspec aiohttp

import os, io, re, csv, sys, gzip, json, time, tarfile, hashlib
import collections
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from huggingface_hub import snapshot_download

REPO = "HemanthKari/rcsb-ligand-retrieval-v0.1"
LOCAL = Path(snapshot_download(REPO, repo_type="dataset"))
print("dataset at:", LOCAL)

# notebook 09 wrote its artifacts to DRIVE_DIR. Look there first, then in the
# session, then in the CWD. If you ran 09 elsewhere, set MAP_DIR by hand.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/fluxion_reextract")
except Exception:
    DRIVE_DIR = Path("./fluxion_reextract")

MAP_DIR = None
for cand in [DRIVE_DIR, Path("/content"), Path(".")]:
    if (cand / "pocket_residue_map.parquet").exists():
        MAP_DIR = cand
        break

OUT_DIR = DRIVE_DIR if DRIVE_DIR.exists() else Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("residue map dir:", MAP_DIR)
print("output dir     :", OUT_DIR)
```

---

## Cell 1 — Load artifacts, SIFTS labels, define helpers, init RESULTS

```python
# ---- checkpointing ------------------------------------------------------
# This notebook runs for HOURS (ProtT5-XL + the ESM-2 ladder). Without a
# flush after every check, one disconnect loses the entire run.
CKPT_JSON = OUT_DIR / "audit_results_2.json"
CKPT_MD   = OUT_DIR / "audit_results_2.md"

RESULTS = {}
if CKPT_JSON.exists():
    try:
        RESULTS = json.loads(CKPT_JSON.read_text(encoding="utf-8"))
        print("RESUMED %d section(s) from %s" % (len(RESULTS), CKPT_JSON))
        print("  already done:", ", ".join(sorted(RESULTS)) or "(none)")
    except Exception as _e:
        print("checkpoint unreadable (%s) - starting fresh" % type(_e).__name__)


def _flush_results():
    """Persist RESULTS after EVERY update, via temp-file replace."""
    try:
        _tmp = CKPT_JSON.with_suffix(".tmp")
        _tmp.write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
        _tmp.replace(CKPT_JSON)
        _md = ["# Sci Data revision - remaining analyses", ""]
        for _s, _kv in RESULTS.items():
            _md.append("## %s" % _s)
            for _k, _v in _kv.items():
                _md.append("- **%s**: `%s`" % (_k, _v))
            _md.append("")
        CKPT_MD.write_text(chr(10).join(_md), encoding="utf-8")
    except Exception as _e:
        print("  !! CHECKPOINT WRITE FAILED: %s: %s" % (type(_e).__name__, _e))


def done(section):
    """True if a previous run already produced this section."""
    return bool(RESULTS.get(section))


def result(section, **kv):
    """Record + print a labelled result block, then checkpoint to disk."""
    RESULTS.setdefault(section, {}).update(kv)
    print("\n" + "=" * 72)
    print("RESULT  [%s]" % section)
    print("=" * 72)
    for k, v in kv.items():
        print("  %-42s %s" % (k, v))
    _flush_results()

def print_table(headers, rows):
    """Fixed-width table to stdout; returns the same table as markdown text."""
    widths = [len(h) for h in headers]
    srows = []
    for r in rows:
        sr = ["" if x is None else str(x) for x in r]
        srows.append(sr)
        for i, cell in enumerate(sr):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for sr in srows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(sr)))
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    for sr in srows:
        md.append("| " + " | ".join(sr) + " |")
    return "\n".join(md)

# ---- core artifacts -------------------------------------------------------
# Pocket/ligand rows are reached through the metadata offset fields, never
# through a shared loop counter: pocket_embeddings.npy carries rows that the
# joined index dropped (zero-norm ligands), so the two are NOT row-aligned.
meta = [json.loads(l) for l in open(LOCAL / "metadata.jsonl")]

# ---------------------------------------------------------------------------
# EMBEDDING VERSION SWITCH (mirrors notebook 08)
# "auto" detects the per-chain re-embedded array from notebook 11. The shipped
# joined_embeddings.npy was built from the v0.1 pocket half and is STALE once the
# pocket embeddings change, so under v0.2 we rebuild the concatenation here.
# ---------------------------------------------------------------------------
V02_POCKET = "auto"
# Falling back to v0.1 must be an EXPLICIT choice, never a silent default.
ALLOW_V01_FALLBACK = False
_CANDIDATES = [
    "/content/drive/MyDrive/fluxion_v02/pocket_embeddings.npy",
    "/content/drive/MyDrive/fluxion_reextract/pocket_embeddings.npy",
    "/content/fluxion_v02/pocket_embeddings.npy",
    "./fluxion_v02/pocket_embeddings.npy",
]
if V02_POCKET == "auto":
    if not any(Path(c).exists() for c in _CANDIDATES):
        try:
            from google.colab import drive as _drv
            _drv.mount("/content/drive")
        except Exception:
            pass
    V02_POCKET = next((c for c in _CANDIDATES if Path(c).exists()), None)
    if not V02_POCKET and not ALLOW_V01_FALLBACK:
        raise SystemExit(chr(10).join([
            "", "=" * 72,
            "STOPPED: no v0.2 pocket embeddings found - Drive is not mounted.",
            "=" * 72,
            "Continuing would silently measure v0.1 and waste a multi-hour run.",
            "",
            "  1. Run this in a cell and approve the popup:",
            "",
            "         from google.colab import drive",
            "         drive.mount('/content/drive')",
            "",
            "  2. Re-run this cell.",
            "",
            "Paths searched:",
        ] + ["  - " + c for c in _CANDIDATES] + [
            "",
            "To measure v0.1 ON PURPOSE, set ALLOW_V01_FALLBACK = True above.",
            "=" * 72,
        ]))

ligand = np.load(LOCAL / "ligands" / "ligand_embeddings.npy", mmap_mode="r")
N = len(meta)
p_off = np.array([m["pocket_emb_offset"] for m in meta], dtype=np.int64)
l_off = np.array([m["ligand_emb_offset"] for m in meta], dtype=np.int64)

if V02_POCKET:
    pocket = np.load(V02_POCKET, mmap_mode="r")
    _P0 = np.asarray(pocket, dtype=np.float32)[p_off]
    _L0 = np.asarray(ligand, dtype=np.float32)[l_off]
    _J0 = np.hstack([_P0, _L0]).astype(np.float32)
    _J0 /= (np.linalg.norm(_J0, axis=1, keepdims=True) + 1e-12)
    joined = _J0
    EMB_VERSION = "v0.2 (per-chain)"
    del _P0, _L0, _J0
else:
    pocket = np.load(LOCAL / "pockets" / "pocket_embeddings.npy", mmap_mode="r")
    joined = np.load(LOCAL / "joined_embeddings.npy", mmap_mode="r")
    EMB_VERSION = "v0.1 (concatenated-input)"

print("")
print("#" * 72)
print("#  EMBEDDINGS IN USE : " + EMB_VERSION.ljust(47) + "#")
if not V02_POCKET:
    print("#  " + "WARNING: shipped v0.1 embeddings. If you meant v0.2, stop.".ljust(67) + "#")
print("#" * 72)
print("")
RESULTS["0_provenance"] = {"embedding_version": EMB_VERSION,
                           "v02_pocket_path": str(V02_POCKET)}
pk2row = {m["pocket_key"]: i for i, m in enumerate(meta)}

J = np.ascontiguousarray(np.asarray(joined, dtype=np.float32))   # ~1 GB
print("metadata rows:", N, " joined:", J.shape)

import faiss
FLAT = faiss.IndexFlatIP(J.shape[1]); FLAT.add(J)

SEED = 42                           # every sample in this notebook uses seed 42

def stable_seed(text, base=SEED):
    """Deterministic per-label seed. Python's built-in hash() is salted per
    process (PYTHONHASHSEED), so using it here would make 'seed 42' a lie
    across runs. md5 is stable forever."""
    d = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8]
    return base + (int(d, 16) % 1000003)

# corpus mean vector = the anisotropy offset. Accumulate in float64: a float32
# sum over 72k rows loses digits exactly where the centred metrics need them.
MU = J.mean(axis=0, dtype=np.float64).astype(np.float32)

# ---- SIFTS labels (identical loader to notebook 08 Cell 13) ---------------
!wget -q -nc https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz
!wget -q -nc https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_pfam.csv.gz
!wget -q -nc https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_enzyme.csv.gz

def load_sifts(fn, col):
    df = pd.read_csv(fn, compression="gzip", skiprows=1)
    df["PDB"] = df["PDB"].str.lower()
    return df.groupby("PDB")[col].first().to_dict()

uni_map  = load_sifts("pdb_chain_uniprot.csv.gz", "SP_PRIMARY")
pfam_map = load_sifts("pdb_chain_pfam.csv.gz", "PFAM_ID")
ec_map   = load_sifts("pdb_chain_enzyme.csv.gz", "EC_NUMBER")

def _clean(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("", "nan", "none", "?") else s

pdb_lower = [m["pdb_id"].lower() for m in meta]
# Defined HERE, not in §2: §3 needs it too and the sections are independent.
OURS = set(pdb_lower)
U_LAB = np.array([_clean(uni_map.get(p, "")) for p in pdb_lower])
P_LAB = np.array([_clean(pfam_map.get(p, "")) for p in pdb_lower])
EC_FULL = np.array([_clean(ec_map.get(p, "")) for p in pdb_lower])
EC_LAB = np.array([(e.split(".")[0] if e else "") for e in EC_FULL])

print("labelled: uniprot %d  pfam %d  ec %d (of %d)"
      % ((U_LAB != "").sum(), (P_LAB != "").sum(), (EC_LAB != "").sum(), N))

# ---- notebook-09 residue map + ESM input sequences ------------------------
HAVE_MAP = False
rmap = None
seq_by_key = {}
if MAP_DIR is not None:
    rmap = pd.read_parquet(MAP_DIR / "pocket_residue_map.parquet")
    fa = MAP_DIR / "esm_input_sequences.fasta.gz"
    if fa.exists():
        cur = None
        with gzip.open(fa, "rt") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    cur = line[1:].strip()
                elif cur is not None:
                    seq_by_key[cur] = seq_by_key.get(cur, "") + line.strip()
    HAVE_MAP = len(rmap) > 0 and len(seq_by_key) > 0
    print("residue map rows: %d   fasta sequences: %d" % (len(rmap), len(seq_by_key)))

if not HAVE_MAP:
    print("\n" + "!" * 72)
    print("pocket_residue_map.parquet / esm_input_sequences.fasta.gz NOT FOUND.")
    print("Section 1 (encoder comparison) will be SKIPPED. Sections 2-5 still run.")
    print("Re-run notebook 09, or set MAP_DIR in Cell 0 to the folder holding them.")
    print("!" * 72)

result("0_inputs",
    n_metadata_rows=N,
    joined_shape=str(J.shape),
    have_residue_map=HAVE_MAP,
    n_residue_map_rows=(0 if rmap is None else int(len(rmap))),
    n_esm_input_sequences=len(seq_by_key),
    n_with_uniprot=int((U_LAB != "").sum()),
    n_with_pfam=int((P_LAB != "").sum()),
    n_with_ec=int((EC_LAB != "").sum()),
)
```

---

## Cell 2 — §1 setup: the controlled 5,000-complex stratified subsample

Every encoder in §1 sees **the same complexes** and pools over **the same
residue indices**, so the only thing that varies is the encoder. Sampling is
sqrt-proportional over Pfam families (so the head does not swamp the tail) and
every family contributes at least 2 members, which is what makes same-family
hit@1 answerable *within the subsample*.

```python
SUBSAMPLE_N = 5000
MAX_SEQ_LEN = 1022          # the pipeline's own cap; keeps ESM-2 inputs legal
PER_FAM_MIN = 2             # invariant: a family is in with >= 2 members, or out
PER_FAM_CAP = 8             # stop the biggest families from eating the budget

if HAVE_MAP:
    m = rmap.copy()
    m["pdb_lower"] = m.pdb_id.str.lower()
    m["uniprot"] = [_clean(uni_map.get(p, "")) for p in m.pdb_lower]
    m["pfam"] = [_clean(pfam_map.get(p, "")) for p in m.pdb_lower]
    m["seq"] = [seq_by_key.get(k, "") for k in m.pocket_key]
    m["in_index"] = [k in pk2row for k in m.pocket_key]

    pool = m[(m.in_index) & (m.seq.str.len() > 0)
             & (m.seq.str.len() <= MAX_SEQ_LEN) & (m.pfam != "")].copy()
    pool["seq_len"] = pool.seq.str.len()
    pool = pool[pool.n_pocket_residues > 0]
    print("eligible pool: %d entries (%d Pfam families)"
          % (len(pool), pool.pfam.nunique()))

    fam_sizes = pool.pfam.value_counts()
    fam_sizes = fam_sizes[fam_sizes >= 2]        # hit@1 must be answerable
    pool = pool[pool.pfam.isin(fam_sizes.index)]

    # ---- allocation ------------------------------------------------------
    # sqrt-proportional so the head does not swamp the tail, floored at
    # PER_FAM_MIN and capped at PER_FAM_CAP.
    fams_all = fam_sizes.index.tolist()
    sizes_all = fam_sizes.values.astype(np.int64)
    w = np.sqrt(sizes_all.astype(np.float64))
    alloc = np.round(SUBSAMPLE_N * w / w.sum()).astype(np.int64)
    alloc = np.clip(alloc, PER_FAM_MIN, np.minimum(sizes_all, PER_FAM_CAP))

    # The floor means the minimum spend is PER_FAM_MIN * n_families, which for a
    # corpus with thousands of Pfam families exceeds the budget. When it does we
    # DROP WHOLE FAMILIES, chosen with a seeded shuffle rather than by size, so
    # the retained families stay an unbiased sample instead of a head-biased one.
    # A family is never left with one member: that would be an unanswerable query
    # masquerading as a miss.
    n_families_dropped = 0
    if alloc.sum() > SUBSAMPLE_N:
        ordr = np.random.default_rng(SEED).permutation(len(fams_all))
        keep_mask = np.zeros(len(fams_all), dtype=bool)
        spent = 0
        for fi in ordr:
            if spent + alloc[fi] > SUBSAMPLE_N:
                continue
            keep_mask[fi] = True
            spent += int(alloc[fi])
        n_families_dropped = int((~keep_mask).sum())
        fams_all = [f for f, k in zip(fams_all, keep_mask) if k]
        alloc = alloc[keep_mask]
        sizes_all = sizes_all[keep_mask]
    else:
        # under budget: top up the largest families, respecting BOTH their true
        # size and PER_FAM_CAP (the cap is reported as an invariant below, so it
        # has to hold on this branch too)
        room = SUBSAMPLE_N - int(alloc.sum())
        for fi in np.argsort(-sizes_all):
            if room <= 0:
                break
            headroom = int(min(sizes_all[fi], PER_FAM_CAP) - alloc[fi])
            if headroom <= 0:
                continue
            add = int(min(room, headroom))
            alloc[fi] += add
            room -= add

    plan = dict(zip(fams_all, alloc.tolist()))
    members_by_fam = collections.defaultdict(list)
    for ix, fam in zip(pool.index.to_numpy(), pool.pfam.to_numpy()):
        members_by_fam[fam].append(ix)

    picked = []
    for fam in fams_all:                          # deterministic order
        members = np.array(sorted(members_by_fam[fam]))
        k = int(min(plan[fam], len(members)))
        rs = np.random.default_rng(stable_seed(fam))
        picked.extend(rs.choice(members, size=k, replace=False).tolist())
    picked = sorted(set(picked))

    SUB = pool.loc[picked].reset_index(drop=True)
    SUB_SEQS = SUB.seq.tolist()
    SUB_IDXS = [json.loads(s) for s in SUB.pocket_res_indices]
    SUB_UNI = SUB.uniprot.to_numpy()
    SUB_PFAM = SUB.pfam.to_numpy()
    SUB_GLOBAL = np.array([pk2row[k] for k in SUB.pocket_key], dtype=np.int64)

    # sanity: pooling indices must be non-empty and inside their own sequence
    n_empty = sum(1 for ix in SUB_IDXS if not ix)
    assert n_empty == 0, "%d subsample rows have no pocket residues" % n_empty
    bad = sum(1 for s, ix in zip(SUB_SEQS, SUB_IDXS) if max(ix) >= len(s))
    assert bad == 0, "%d subsample rows have residue indices past the sequence end" % bad

    ufam = collections.Counter(SUB_PFAM)
    uuni = collections.Counter(SUB_UNI)
    assert min(ufam.values()) >= PER_FAM_MIN, \
        "invariant broken: a family entered the subsample with < %d members" % PER_FAM_MIN
    result("1a_encoder_subsample",
        n_complexes=len(SUB),
        n_pfam_families=len(ufam),
        n_families_with_ge2_in_subsample=sum(1 for v in ufam.values() if v >= 2),
        n_families_dropped_for_budget=n_families_dropped,
        per_family_min=PER_FAM_MIN,
        per_family_cap=PER_FAM_CAP,
        n_uniprot_targets=len(uuni),
        n_uniprot_with_ge2_in_subsample=sum(1 for k, v in uuni.items() if k and v >= 2),
        median_seq_len=int(np.median(SUB.seq_len)),
        max_seq_len=int(SUB.seq_len.max()),
        median_pocket_residues=int(np.median(SUB.n_pocket_residues)),
        largest_family_share=round(float(max(ufam.values())) / len(SUB), 4),
        NOTE="All encoders below embed exactly these sequences and mean-pool over "
             "exactly these residue indices. The only variable is the encoder.",
    )
else:
    SUB = None
    print("skipping §1 subsample construction - no residue map")
```

---

## Cell 3 — §1 encoders: ESM-2 ladder, ProtT5-XL, composition, BLOSUM62

`pocket_res_indices` are 0-based positions in the ESM input string.
**ESM-2 prepends BOS**, so token position = index + 1.
**ProtT5 prepends nothing** (it appends `</s>` only), so token position =
index. That off-by-one is the single easiest way to silently invalidate this
comparison, so the ProtT5 path asserts the alignment on its first batch.

```python
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = (DEVICE == "cuda")
print("device:", DEVICE, " fp16:", USE_FP16)

TOKEN_BUDGET = 8192          # batch = as many seqs as fit in this many tokens

def length_batches(seqs, budget=TOKEN_BUDGET, max_bs=16):
    """Length-sorted batches, so padding is minimal and comparable across models."""
    order = np.argsort([len(s) for s in seqs], kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(seqs[i])
        newmax = max(cur_max, L)
        if cur and ((len(cur) + 1) * newmax > budget or len(cur) >= max_bs):
            batches.append(cur); cur, cur_max = [], 0
            newmax = L
        cur.append(int(i)); cur_max = newmax
    if cur:
        batches.append(cur)
    return batches

# ---------------------------------------------------------------- ESM-2 ----
def embed_esm2(model_name, seqs, idx_lists, verbose_every=40):
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    # try/finally: if this model OOMs mid-ladder, its weights MUST leave the GPU
    # or the next (larger) model in the ladder OOMs too and the whole comparison
    # is lost to a cascade rather than to one failure.
    try:
        model = model.to(DEVICE).eval()
        if USE_FP16:
            model = model.half()
        bc = alphabet.get_batch_converter()
        layer = model.num_layers

        out = [None] * len(seqs)
        batches = length_batches(seqs)
        t0 = time.time()
        for bn, batch in enumerate(batches, 1):
            data = [("s%d" % i, seqs[i]) for i in batch]
            _, _, toks = bc(data)
            with torch.no_grad():
                rep = model(toks.to(DEVICE), repr_layers=[layer],
                            return_contacts=False)["representations"][layer]
            if bool(torch.isnan(rep).any()):
                raise RuntimeError(
                    "%s produced NaNs on batch %d/%d (fp16 overflow?) - a NaN "
                    "encoder would score as 'worse than composition'"
                    % (model_name, bn, len(batches)))
            for row, i in enumerate(batch):
                pos = torch.tensor([j + 1 for j in idx_lists[i]], device=rep.device)
                out[i] = rep[row].index_select(0, pos).mean(0).float().cpu().numpy()
            if bn % verbose_every == 0:
                print("   %s  batch %d/%d  %.0fs" % (model_name, bn, len(batches),
                                                     time.time() - t0), flush=True)
        return np.stack(out).astype(np.float32)
    finally:
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# --------------------------------------------------------------- ProtT5 ----
def embed_prott5(model_name, seqs, idx_lists, fp16=None, verbose_every=40):
    from transformers import T5Tokenizer, T5EncoderModel
    if fp16 is None:
        fp16 = USE_FP16
    try:
        tok = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    except TypeError:
        # older transformers do not accept `legacy`
        tok = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    try:
        model = model.to(DEVICE).eval()
        if fp16:
            model = model.half()

        out = [None] * len(seqs)
        batches = length_batches(seqs, budget=4096, max_bs=8)
        announced = False
        t0 = time.time()
        for bn, batch in enumerate(batches, 1):
            prepped = [" ".join(list(re.sub(r"[UZOB]", "X", seqs[i]))) for i in batch]
            enc = tok.batch_encode_plus(prepped, add_special_tokens=True,
                                        padding="longest", return_tensors="pt")
            ids = enc["input_ids"].to(DEVICE)
            att = enc["attention_mask"].to(DEVICE)
            # ProtT5 must emit exactly one token per residue plus a trailing </s>.
            # Checked on EVERY batch, not just the first: batches are length-sorted,
            # so batch 1 holds the shortest sequences and is the least likely place
            # for an odd residue character to tokenise into two pieces.
            got = att.sum(1).cpu().numpy().tolist()
            want = [len(seqs[i]) + 1 for i in batch]
            assert got == want, ("ProtT5 tokenisation is not 1-token-per-residue "
                                 "on batch %d (got %s, expected %s) - pooling "
                                 "offsets would be wrong" % (bn, got[:4], want[:4]))
            if not announced:
                RESULTS.setdefault("1b_encoder_sanity", {})["prott5_offset_check"] = (
                    "PASS on all %d batches: attention_mask length == len(seq)+1, "
                    "residue i -> token i (no BOS)" % len(batches))
                print("   ProtT5 offset check PASSED (residue i -> token i, no BOS)")
                announced = True
            with torch.no_grad():
                rep = model(input_ids=ids, attention_mask=att).last_hidden_state
            # T5 in fp16 is a known overflow risk, and it does NOT necessarily
            # show up on the shortest batch. Fail loudly on whichever batch first
            # goes non-finite rather than silently reporting a NaN encoder as
            # "worse than the composition control"; Cell 5 retries in fp32.
            if bool(torch.isnan(rep).any()):
                raise RuntimeError(
                    "ProtT5 produced NaNs on batch %d/%d (fp16=%s) - rerun with "
                    "fp16=False" % (bn, len(batches), fp16))
            for row, i in enumerate(batch):
                pos = torch.tensor(idx_lists[i], device=rep.device)
                out[i] = rep[row].index_select(0, pos).mean(0).float().cpu().numpy()
            if bn % verbose_every == 0:
                print("   ProtT5  batch %d/%d  %.0fs" % (bn, len(batches),
                                                         time.time() - t0), flush=True)
        return np.stack(out).astype(np.float32)
    finally:
        # free the 3B-parameter encoder before the fp32 retry, or the retry OOMs
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# ------------------------------------------------- composition controls ----
AA20 = "ARNDCQEGHILKMFPSTWYV"
AA_IX = {c: i for i, c in enumerate(AA20)}

# BLOSUM62, rows/cols in AA20 order. Hard-coded so the notebook has no
# Biopython dependency and the control is byte-reproducible.
BLOSUM62 = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
], dtype=np.float32)
assert BLOSUM62.shape == (20, 20)
assert np.abs(BLOSUM62 - BLOSUM62.T).max() == 0, "BLOSUM62 must be symmetric"

def composition_counts(seqs, idx_lists):
    C = np.zeros((len(seqs), 20), dtype=np.float32)
    for i, (s, idxs) in enumerate(zip(seqs, idx_lists)):
        for j in idxs:
            k = AA_IX.get(s[j])
            if k is not None:
                C[i, k] += 1.0
    return C

def embed_composition(seqs, idx_lists):
    """20-d amino-acid FRACTION over the pocket residues. No model, no GPU."""
    C = composition_counts(seqs, idx_lists)
    return C / np.maximum(C.sum(1, keepdims=True), 1.0)

def embed_blosum(seqs, idx_lists):
    """Same bag of residues, but each residue contributes its BLOSUM62 row, so
    chemically similar residues share support. Still 20-d, still no model."""
    C = composition_counts(seqs, idx_lists)
    B = C @ BLOSUM62
    return B / np.maximum(C.sum(1, keepdims=True), 1.0)

print("encoder functions ready")
```

---

## Cell 4 — §1 evaluation harness: hit@1 / hit@10 / same-family ROC-AUC

```python
from sklearn.metrics import roc_auc_score

def _codes(labels):
    codes, _ = pd.factorize(pd.Series(labels))
    codes = codes.astype(np.int64).copy()
    codes[np.array([l == "" for l in labels])] = -1
    return codes

def hit_rates(A, labels, k=10):
    """Exact within-subsample search. Only queries whose label has >= 2 members
    IN THE SUBSAMPLE are scored - anything else is unanswerable, not a miss."""
    n = A.shape[0]
    ii = faiss.IndexFlatIP(A.shape[1]); ii.add(A)
    _, I = ii.search(A, k + 1)
    cnt = collections.Counter(l for l in labels if l)
    h1, hk = [], []
    for qi in range(n):
        lq = labels[qi]
        if not lq or cnt[lq] < 2:
            continue
        nb = [int(j) for j in I[qi] if int(j) != qi][:k]
        match = [labels[j] == lq for j in nb]
        if not match:
            continue
        h1.append(bool(match[0]))
        hk.append(bool(any(match)))
    return (round(float(np.mean(h1)), 4) if h1 else None,
            round(float(np.mean(hk)), 4) if hk else None,
            len(h1))

def pair_auc(A, labels, max_pos=200000, n_neg=400000, max_members=40, chunk=50000):
    """ROC-AUC of cosine similarity as a same-family discriminator.

    Positive pairs are ENUMERATED exhaustively rather than hoped for in a
    uniform pair sample. With ~2,000 mostly-small families the same-family rate
    among random pairs is ~1e-4, so a 1M-pair uniform sample yields a couple of
    hundred positives and the AUC standard error (~0.04) is larger than the
    differences between encoders we are trying to measure. AUC is a ranking
    statistic and is invariant to the positive/negative ratio, so enumerating
    positives and sampling negatives is unbiased and far more precise.

    Chunked so the gather never materialises n_pairs x d.
    """
    code = _codes(labels)
    ok = np.flatnonzero(code >= 0)
    if len(ok) < 100:
        return None, 0, 0
    rr = np.random.default_rng(SEED)

    by_code = collections.defaultdict(list)
    for i in ok:
        by_code[int(code[i])].append(int(i))

    pos_a, pos_b = [], []
    for c in sorted(by_code.keys()):
        idxs = by_code[c]
        if len(idxs) < 2:
            continue
        if len(idxs) > max_members:      # bound C(m,2) for huge families
            idxs = sorted(rr.choice(np.array(idxs), size=max_members,
                                    replace=False).tolist())
        for x in range(len(idxs)):
            for y_ in range(x + 1, len(idxs)):
                pos_a.append(idxs[x]); pos_b.append(idxs[y_])
    if len(pos_a) < 50:
        return None, len(pos_a), 0

    pos_a = np.array(pos_a, dtype=np.int64)
    pos_b = np.array(pos_b, dtype=np.int64)
    if len(pos_a) > max_pos:
        sel = rr.choice(len(pos_a), size=max_pos, replace=False)
        pos_a, pos_b = pos_a[sel], pos_b[sel]

    na = rr.choice(ok, size=n_neg * 2)
    nb = rr.choice(ok, size=n_neg * 2)
    keepn = code[na] != code[nb]         # cross-family only
    na, nb = na[keepn][:n_neg], nb[keepn][:n_neg]
    if len(na) < 50:
        return None, int(len(pos_a)), 0

    ia = np.concatenate([pos_a, na])
    ib = np.concatenate([pos_b, nb])
    y = np.concatenate([np.ones(len(pos_a), dtype=np.int8),
                        np.zeros(len(na), dtype=np.int8)])
    s = np.empty(len(ia), dtype=np.float32)
    for st in range(0, len(ia), chunk):
        en = min(st + chunk, len(ia))
        s[st:en] = (A[ia[st:en]] * A[ib[st:en]]).sum(1)
    return round(float(roc_auc_score(y, s)), 4), int(y.sum()), int(len(y))

def l2norm(E):
    A = np.ascontiguousarray(np.asarray(E, dtype=np.float32))
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

def eval_encoder(name, E, dim_note=""):
    A = l2norm(E)
    p1, p10, np_ans = hit_rates(A, SUB_PFAM)
    u1, u10, nu_ans = hit_rates(A, SUB_UNI)
    auc, n_pos, n_pairs = pair_auc(A, SUB_PFAM)
    row = dict(encoder=name, dim=int(A.shape[1]),
               hit1_same_pfam=p1, hit10_same_pfam=p10,
               hit1_same_uniprot=u1, hit10_same_uniprot=u10,
               auc_same_pfam=auc,
               n_answerable_pfam=np_ans, n_answerable_uniprot=nu_ans,
               n_pairs_scored=n_pairs, n_positive_pairs=n_pos,
               auc_note="balanced pairs: all same-family pairs vs sampled "
                        "cross-family pairs; AUC is ratio-invariant",
               note=dim_note)
    print("  %-24s d=%-5d pfam@1 %-7s pfam@10 %-7s uni@1 %-7s uni@10 %-7s AUC %-7s"
          % (name, A.shape[1], p1, p10, u1, u10, auc))
    return row

print("evaluation harness ready")
```

---

## Cell 5 — §1 **R1.3: run every encoder and report** *(composition control is the headline)*

```python
ENCODER_ROWS = []
ENCODER_ERRORS = {}

RUN_ESM_LADDER = True
RUN_PROTT5 = True          # set False if the runtime cannot hold a 3B model

if HAVE_MAP:
    # ---- 0) the cheap controls first, so they are on the record even if a GPU
    #         model OOMs halfway through --------------------------------------
    ENCODER_ROWS.append(eval_encoder(
        "aa_composition_20d", embed_composition(SUB_SEQS, SUB_IDXS),
        "CONTROL: amino-acid fractions over the pocket residues. No model."))
    ENCODER_ROWS.append(eval_encoder(
        "aa_blosum62_20d", embed_blosum(SUB_SEQS, SUB_IDXS),
        "CONTROL: BLOSUM62-weighted bag of the same residues. No model."))

    # ---- 1) the shipped embedding, restricted to the same subsample ---------
    try:
        # via the metadata offset field (SUB_GLOBAL -> p_off), the same route
        # notebooks 08/09 use. pocket_embeddings.npy is NOT row-aligned to the
        # index, so a positional lookup here would be silently wrong.
        E_ship = np.asarray(pocket[p_off[SUB_GLOBAL]], dtype=np.float32)
        ENCODER_ROWS.append(eval_encoder(
            "SHIPPED_esm2_650M_fp16", E_ship,
            "the released pocket_embeddings.npy rows for these same complexes"))
    except Exception as e:
        ENCODER_ERRORS["SHIPPED_esm2_650M_fp16"] = "%s: %s" % (type(e).__name__, e)

    # ---- 2) the ESM-2 ladder ------------------------------------------------
    if RUN_ESM_LADDER:
        for tag, mname in [("esm2_t12_35M", "esm2_t12_35M_UR50D"),
                           ("esm2_t30_150M", "esm2_t30_150M_UR50D"),
                           ("esm2_t33_650M", "esm2_t33_650M_UR50D")]:
            try:
                print("\n-- embedding with %s --" % mname)
                E = embed_esm2(mname, SUB_SEQS, SUB_IDXS)
                ENCODER_ROWS.append(eval_encoder(tag, E, "ESM-2 ladder"))
                del E
            except Exception as e:
                ENCODER_ERRORS[tag] = "%s: %s" % (type(e).__name__, str(e)[:200])
                print("   FAILED:", ENCODER_ERRORS[tag])

    # ---- 3) a DIFFERENT model family ---------------------------------------
    if RUN_PROTT5:
        E = None
        for attempt_fp16 in ([True, False] if USE_FP16 else [False]):
            try:
                print("\n-- embedding with ProtT5-XL-UniRef50 (different family), "
                      "fp16=%s --" % attempt_fp16)
                E = embed_prott5("Rostlab/prot_t5_xl_uniref50", SUB_SEQS, SUB_IDXS,
                                 fp16=attempt_fp16)
                break
            except Exception as e:
                ENCODER_ERRORS["prott5_xl_uniref50_fp16=%s" % attempt_fp16] = (
                    "%s: %s" % (type(e).__name__, str(e)[:200]))
                print("   FAILED (fp16=%s): %s" % (attempt_fp16, str(e)[:200]))
        if E is not None:
            ENCODER_ROWS.append(eval_encoder(
                "prott5_xl_uniref50", E,
                "DIFFERENT MODEL FAMILY: T5 encoder, UniRef50, not the ESM line"))
            del E

    # ---- report -------------------------------------------------------------
    hdr = ["encoder", "dim", "pfam@1", "pfam@10", "uni@1", "uni@10", "AUC(same-Pfam)"]
    rows = [[r["encoder"], r["dim"], r["hit1_same_pfam"], r["hit10_same_pfam"],
             r["hit1_same_uniprot"], r["hit10_same_uniprot"], r["auc_same_pfam"]]
            for r in ENCODER_ROWS]
    print("\n" + "=" * 72)
    print("ENCODER COMPARISON  (n=%d complexes, identical residues pooled)" % len(SUB))
    print("=" * 72)
    enc_md = print_table(hdr, rows)

    # ---- THE HEADLINE: does a 20-d bag of residues tie the language models? --
    lm_rows = [r for r in ENCODER_ROWS if r["encoder"].startswith(("esm2", "prott5", "SHIPPED"))]
    ctrl_rows = [r for r in ENCODER_ROWS if r["encoder"].startswith("aa_")]

    def _best(rows_, field):
        vals = [(r[field], r["encoder"]) for r in rows_ if r[field] is not None]
        return max(vals) if vals else (None, None)

    best_lm_p1, best_lm_name = _best(lm_rows, "hit1_same_pfam")
    best_ct_p1, best_ct_name = _best(ctrl_rows, "hit1_same_pfam")
    best_lm_auc, best_lm_auc_name = _best(lm_rows, "auc_same_pfam")
    best_ct_auc, best_ct_auc_name = _best(ctrl_rows, "auc_same_pfam")

    gap_p1 = (None if (best_lm_p1 is None or best_ct_p1 is None)
              else round(best_lm_p1 - best_ct_p1, 4))
    gap_auc = (None if (best_lm_auc is None or best_ct_auc is None)
               else round(best_lm_auc - best_ct_auc, 4))

    if gap_p1 is None:
        verdict = "INCONCLUSIVE - not enough encoders completed"
    elif gap_p1 <= 0.0:
        verdict = ("COMPOSITION CONTROL WINS. A 20-d bag of pocket residues beats "
                   "every protein language model tested. The retrieval signal "
                   "attributed to ESM-2 is residue composition. This MUST be "
                   "reported in the main text, not the supplement.")
    elif gap_p1 < 0.05:
        verdict = ("COMPOSITION CONTROL TIES (gap %.3f hit@1). The language model "
                   "adds little over a 20-d bag of residues. Report the control "
                   "alongside the headline retrieval numbers." % gap_p1)
    else:
        verdict = ("LANGUAGE MODEL WINS by %.3f hit@1 over the composition control - "
                   "the embedding carries information beyond residue composition. "
                   "Report the control as the floor it is." % gap_p1)

    result("1c_encoder_comparison",
        n_complexes=len(SUB),
        encoders_evaluated=[r["encoder"] for r in ENCODER_ROWS],
        encoders_failed=ENCODER_ERRORS,
        table_markdown=enc_md,
        per_encoder=ENCODER_ROWS,
        best_language_model=best_lm_name,
        best_language_model_hit1_pfam=best_lm_p1,
        best_composition_control=best_ct_name,
        best_composition_control_hit1_pfam=best_ct_p1,
        gap_lm_minus_control_hit1_pfam=gap_p1,
        best_language_model_auc=best_lm_auc,
        best_control_auc=best_ct_auc,
        gap_lm_minus_control_auc=gap_auc,
        HEADLINE=verdict,
        NOTE="Every encoder mean-pools over the SAME residue indices from "
             "pocket_residue_map.parquet, on the SAME %d complexes, with an "
             "EXACT within-subsample index. Absolute rates are NOT comparable to "
             "the full-corpus numbers in the paper (%d candidates here, not %d) "
             "- only the columns are comparable to each other."
             % (len(SUB), len(SUB), N),
    )
    print("\n" + "*" * 72)
    print("HEADLINE (R1.3): " + verdict)
    print("*" * 72)
else:
    print("skipped - no residue map")
```

---

## Cell 6 — §2 **R2.3: PDB-ID overlap with PDBbind / BioLiP2 / PLINDER / Binding MOAD**

Each source is fetched from a public endpoint. A source that cannot be
retrieved is recorded as `NOT_REACHABLE` **with the reason**, and is excluded
from the "in none of them" denominator — it is never silently dropped.

To include a source manually, upload its file and add a path to
`MANUAL_ID_FILES` before running. Any text file works: the parser takes the
first PDB-like token on each line.

```python
MANUAL_ID_FILES = {
    # "PDBbind": "/content/INDEX_general_PL_data.2020",
    # "BioLiP2": "/content/BioLiP.txt",
}

# no real PDB ID starts with 0, so [1-9] drops a class of false positives
PDBID_RE = re.compile(r"^[1-9][A-Za-z0-9]{3}$")
SOURCES = {}          # OURS is defined in Cell 1 so §3 works without §2

def register(name, status, ids=None, detail=""):
    SOURCES[name] = dict(status=status, ids=(ids if ids else set()), detail=detail)
    print("  %-14s %-14s n_ids=%-8s %s"
          % (name, status, len(ids) if ids else 0, detail))

def http_bytes(url, timeout=180, max_mb=1500):
    """GET with a size cap. Returns (bytes, None) or (None, reason)."""
    try:
        r = requests.get(url, timeout=timeout, stream=True,
                         headers={"User-Agent": "rcsb-ligand-retrieval-audit/1.0"})
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:120])
    if r.status_code != 200:
        return None, "HTTP %d" % r.status_code
    buf, total = [], 0
    for chunk in r.iter_content(1 << 20):
        buf.append(chunk); total += len(chunk)
        if total > max_mb * (1 << 20):
            return None, "exceeded %d MB cap" % max_mb
    raw = b"".join(buf)
    head = raw[:400].lower()
    if b"<html" in head or b"<!doctype html" in head:
        return None, "endpoint returned an HTML page (registration wall?)"
    return raw, None

def ids_from_text(text, field=None):
    """Extract 4-character PDB codes. If `field` is given, look only at that
    comma/tab/space-separated column; else take the first matching token."""
    out = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,\t ]+", line)
        cand = []
        if field is not None and len(parts) > field:
            cand = [parts[field]]
        else:
            cand = parts[:6]
        for tok in cand:
            tok = tok.strip().strip('"').strip("'")
            if PDBID_RE.match(tok):
                out.add(tok.lower())
                break
    return out

print("fetching reference dataset ID lists ...")

# ---------------------------------------------------------------- PDBbind --
# PDBbind-CN gates its index files behind a registration form; there is no
# stable anonymous endpoint. Try anyway, then fall back to a manual upload.
pdbbind_urls = [
    "http://www.pdbbind.org.cn/download/PDBbind_v2020_plain_text_index.tar.gz",
    "http://www.pdbbind.org.cn/download/PDBbind_v2019_plain_text_index.tar.gz",
    "http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz",
]
_done = False
_why = []
for u in pdbbind_urls:
    raw, err = http_bytes(u, max_mb=400)
    if raw is None:
        _why.append("%s -> %s" % (u, err))
        continue
    try:
        txt = ""
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
            for mem in tf.getmembers():
                if mem.isfile() and "INDEX" in mem.name.upper():
                    txt += tf.extractfile(mem).read().decode("utf-8", "ignore") + "\n"
        got = ids_from_text(txt, field=0)
        if got:
            register("PDBbind", "FETCHED", got, "from %s" % u)
            _done = True
            break
        _why.append("%s -> downloaded but no INDEX member yielded PDB IDs" % u)
    except Exception as e:
        _why.append("%s -> %s: %s" % (u, type(e).__name__, str(e)[:100]))
if not _done:
    register("PDBbind", "NOT_REACHABLE", set(),
             "PDBbind-CN requires registration; no anonymous index endpoint "
             "responded. Upload INDEX_general_PL_data.* and add it to "
             "MANUAL_ID_FILES to include this source. Per-URL: %s" % "; ".join(_why))

# ---------------------------------------------------------------- BioLiP2 --
biolip_urls = [
    "https://zhanggroup.org/BioLiP/download/BioLiP.txt.gz",
    "https://zhanggroup.org/BioLiP/download/BioLiP_nr.txt.gz",
    "https://zhanggroup.org/BioLiP/weekly/BioLiP.txt.gz",
]
_done = False
_why = []
for u in biolip_urls:
    raw, err = http_bytes(u, max_mb=800)
    if raw is None:
        _why.append("%s -> %s" % (u, err))
        continue
    try:
        txt = gzip.decompress(raw).decode("utf-8", "ignore")
    except Exception:
        txt = raw.decode("utf-8", "ignore")      # some mirrors serve it plain
    got = ids_from_text(txt, field=0)
    if got:
        register("BioLiP2", "FETCHED", got, "from %s" % u)
        _done = True
        break
    _why.append("%s -> fetched %d bytes but no column-0 PDB IDs" % (u, len(raw)))
if not _done:
    register("BioLiP2", "NOT_REACHABLE", set(),
             "no zhanggroup.org BioLiP download URL responded with a parseable "
             "table. Per-URL: %s" % "; ".join(_why))

# ---------------------------------------------------------------- PLINDER --
# Public GCS bucket. Read ONLY the PDB-ID column via parquet column pruning +
# HTTP range requests, so the multi-GB annotation table is not downloaded.
plinder_urls = [
    "https://storage.googleapis.com/plinder/2024-06/v2/index/annotation_table.parquet",
    "https://storage.googleapis.com/plinder/2024-06/v2/index/annotation_table_nonredundant.parquet",
]
_done = False
plinder_err = "not attempted"
for u in plinder_urls:
    try:
        import fsspec
        import pyarrow.parquet as pq
        fs = fsspec.filesystem("http")
        with fs.open(u) as fh:
            pf = pq.ParquetFile(fh)
            names = list(pf.schema_arrow.names)
            col = None
            for want in ["entry_pdb_id", "system_pdb_id", "pdb_id", "system_id"]:
                if want in names:
                    col = want
                    break
            if col is None:
                raise RuntimeError("no pdb-id-like column in %s" % names[:12])
            tb = pf.read(columns=[col])     # column pruning + HTTP range reads
        vals = tb.column(col).to_pylist()
        got = set()
        for v in vals:
            if v is None:
                continue
            s = str(v).split("__")[0].strip().lower()
            if PDBID_RE.match(s):
                got.add(s)
        if got:
            register("PLINDER", "FETCHED", got, "column '%s' from %s" % (col, u))
            _done = True
            break
    except Exception as e:
        plinder_err = "%s: %s" % (type(e).__name__, str(e)[:140])
        continue
if not _done:
    register("PLINDER", "NOT_REACHABLE", set(),
             "could not range-read the public PLINDER annotation table "
             "(tried %d URLs; last error: %s)" % (len(plinder_urls), plinder_err))

# ----------------------------------------------------------- Binding MOAD --
moad_urls = [
    "https://bindingmoad.org/files/csv/every.csv",
    "http://bindingmoad.org/files/csv/every.csv",
    "https://bindingmoad.org/files/csv/every_part_a.csv",
]
_done = False
_why = []
for u in moad_urls:
    raw, err = http_bytes(u, max_mb=400)
    if raw is None:
        _why.append("%s -> %s" % (u, err))
        continue
    txt = raw.decode("utf-8", "ignore")
    got = set()
    for row in csv.reader(io.StringIO(txt)):
        for cell in row[:6]:
            cell = cell.strip()
            if PDBID_RE.match(cell):
                got.add(cell.lower())
                break
    if got:
        register("BindingMOAD", "FETCHED", got, "from %s" % u)
        _done = True
        break
    _why.append("%s -> fetched %d bytes but no PDB-like cell in the first 6 columns"
                % (u, len(raw)))
if not _done:
    register("BindingMOAD", "NOT_REACHABLE", set(),
             "no bindingmoad.org every.csv endpoint responded. Per-URL: %s"
             % "; ".join(_why))

# ---------------------------------------------------- manual overrides -----
for name, path in MANUAL_ID_FILES.items():
    p = Path(path)
    if not p.exists():
        print("  MANUAL %s: file not found at %s" % (name, path))
        continue
    txt = (gzip.open(p, "rt").read() if p.suffix == ".gz"
           else p.read_text(errors="ignore"))
    got = ids_from_text(txt, field=0)
    register(name, "FETCHED_MANUAL", got, "user-supplied file %s" % path)

# ------------------------------------------------------------ the numbers --
fetched = {k: v for k, v in SOURCES.items() if v["status"].startswith("FETCHED")}
missing = {k: v["detail"] for k, v in SOURCES.items() if not v["status"].startswith("FETCHED")}

per_source, union = {}, set()
for name, v in fetched.items():
    inter = OURS & v["ids"]
    union |= inter
    per_source[name] = dict(
        n_source_ids=len(v["ids"]),
        n_overlapping=len(inter),
        pct_of_our_index_present=round(100.0 * len(inter) / len(OURS), 2),
        pct_of_source_covered_by_us=round(100.0 * len(inter) / max(1, len(v["ids"])), 2),
        source=v["detail"],
    )

hdr = ["source", "status", "n_source_ids", "n_overlap", "% of our index"]
rows = []
for name in ["PDBbind", "BioLiP2", "PLINDER", "BindingMOAD"] + \
            [k for k in SOURCES if k not in ("PDBbind", "BioLiP2", "PLINDER", "BindingMOAD")]:
    if name not in SOURCES:
        continue
    v = SOURCES[name]
    if name in per_source:
        ps = per_source[name]
        rows.append([name, v["status"], ps["n_source_ids"], ps["n_overlapping"],
                     "%.2f%%" % ps["pct_of_our_index_present"]])
    else:
        rows.append([name, v["status"], 0, 0, "n/a"])
overlap_md = print_table(hdr, rows)

result("2_dataset_overlap",
    n_our_unique_pdb_ids=len(OURS),
    n_our_index_rows=N,
    sources_fetched=sorted(fetched.keys()),
    sources_NOT_REACHABLE=missing,
    per_source=per_source,
    n_in_at_least_one_fetched_source=len(union),
    pct_in_at_least_one_fetched_source=round(100.0 * len(union) / len(OURS), 2),
    n_in_NONE_of_the_fetched_sources=len(OURS) - len(union),
    pct_in_NONE_of_the_fetched_sources=round(100.0 * (len(OURS) - len(union)) / len(OURS), 2),
    table_markdown=overlap_md,
    DENOMINATOR_CAVEAT=("The 'in NONE' figure is computed over the FETCHED sources "
                        "only: %s. Unreachable sources (%s) would only DECREASE it. "
                        "State this qualification in the response letter."
                        % (sorted(fetched.keys()) or "none",
                           sorted(missing.keys()) or "none")),
    NOTE="Overlap is at PDB-entry granularity. A shared PDB ID does not mean a "
         "shared complex: the reference sets and this index may select different "
         "ligands from the same entry.",
)
```

---

## Cell 7 — §3 Diversity: unique targets, redundancy curve, Gini, top-N concentration

```python
import matplotlib.pyplot as plt

def gini(counts):
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return None
    cum = np.arange(1, n + 1)
    return round(float((2.0 * (cum * x).sum()) / (n * x.sum()) - (n + 1.0) / n), 4)

def concentration(counts_desc, tot):
    out = {}
    for k in [10, 100, 1000]:
        out["top%d_share" % k] = round(float(counts_desc[:k].sum()) / tot, 4)
    return out

div = {}
curves = {}
for label_name, labels in [("uniprot", U_LAB), ("pfam", P_LAB), ("ec_full", EC_FULL)]:
    lab = [x for x in labels if x]
    c = collections.Counter(lab)
    counts_desc = np.array(sorted(c.values(), reverse=True), dtype=np.float64)
    curves[label_name] = counts_desc
    d = dict(
        n_entries_labelled=len(lab),
        pct_of_index_labelled=round(100.0 * len(lab) / N, 2),
        n_unique=len(c),
        max_entries_per_target=int(counts_desc[0]) if len(counts_desc) else 0,
        median_entries_per_target=float(np.median(counts_desc)) if len(counts_desc) else None,
        n_singletons=int((counts_desc == 1).sum()),
        pct_targets_that_are_singletons=round(100.0 * float((counts_desc == 1).sum())
                                              / max(1, len(counts_desc)), 2),
        gini=gini(counts_desc),
    )
    d.update(concentration(counts_desc, counts_desc.sum()))
    div[label_name] = d

ec_class = collections.Counter(x for x in EC_LAB if x)
n_pdb_ids = len(OURS)
entries_per_pdb = collections.Counter(pdb_lower)
epp = np.array(sorted(entries_per_pdb.values(), reverse=True), dtype=np.float64)

result("3_diversity",
    n_index_rows=N,
    n_unique_pdb_entries=n_pdb_ids,
    rows_per_pdb_entry_max=int(epp[0]),
    rows_per_pdb_entry_mean=round(float(epp.mean()), 3),
    by_label=div,
    ec_class_distribution=dict(sorted(ec_class.items())),
    top10_uniprot=collections.Counter(x for x in U_LAB if x).most_common(10),
    top10_pfam=collections.Counter(x for x in P_LAB if x).most_common(10),
    NOTE="Gini is over entries-per-target. A high Gini plus a large top-10 share "
         "means the corpus is a handful of heavily-crystallised targets with a "
         "long tail, which is exactly what a redundancy-aware benchmark must "
         "split on (see build_target_disjoint_split.py).",
)

# redundancy curve figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, style in [("uniprot", "-"), ("pfam", "--")]:
    cd = curves[name]
    ax[0].plot(np.arange(1, len(cd) + 1), cd, style, label=name)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("target rank"); ax[0].set_ylabel("entries")
ax[0].set_title("Redundancy curve (entries per target)"); ax[0].legend()

for name, style in [("uniprot", "-"), ("pfam", "--")]:
    cd = curves[name]
    ax[1].plot(np.arange(1, len(cd) + 1) / len(cd), np.cumsum(cd) / cd.sum(), style,
               label=name)
ax[1].plot([0, 1], [0, 1], ":", color="grey", label="uniform")
ax[1].set_xlabel("fraction of targets (ranked by size)")
ax[1].set_ylabel("cumulative fraction of entries")
ax[1].set_title("Concentration (Lorenz-style)"); ax[1].legend()
fig.tight_layout()
fig.savefig(str(OUT_DIR / "fig_redundancy_curve.png"), dpi=160)
print("wrote", OUT_DIR / "fig_redundancy_curve.png")
plt.show()
```

---

## Cell 8 — §4 **R1.2: multi-family qualitative retrieval panel (+ 2 deliberate negatives)**

One query drawn uniformly at random (seed 42) from each of the 10 largest
Pfam families, top-5 with cosine and same-target / same-family labels. Then
two cases chosen *because they fail*: a singleton-family query, which cannot
have a same-family neighbour at all, and a query whose top-1 is the same
ligand in a different protein.

```python
def top_k(qi, k=5):
    D, I = FLAT.search(np.ascontiguousarray(J[qi:qi + 1]), k + 1)
    hits = [(int(j), float(d)) for j, d in zip(I[0], D[0]) if int(j) != qi][:k]
    return hits

def panel_rows(qi, tag):
    rows = []
    q_uni, q_pfam = U_LAB[qi], P_LAB[qi]
    for rank, (j, cos) in enumerate(top_k(qi, 5), 1):
        same_t = ("same-target" if (q_uni and U_LAB[j] == q_uni) else "diff-target")
        same_f = ("same-family" if (q_pfam and P_LAB[j] == q_pfam) else "diff-family")
        same_l = ("same-ligand" if meta[j]["ligand_resname"] == meta[qi]["ligand_resname"]
                  else "diff-ligand")
        rows.append([tag if rank == 1 else "",
                     "%s / %s" % (meta[qi]["pdb_id"].upper(), meta[qi]["ligand_resname"])
                     if rank == 1 else "",
                     str(rank),
                     "%s / %s" % (meta[j]["pdb_id"].upper(), meta[j]["ligand_resname"]),
                     "%.4f" % cos,
                     same_t, same_f, same_l])
    return rows

fam_counts = collections.Counter(x for x in P_LAB if x)
top10_fams = [f for f, _ in fam_counts.most_common(10)]

panel = []
panel_meta = {}
for fam in top10_fams:
    members = np.flatnonzero(P_LAB == fam)
    rs = np.random.default_rng(stable_seed(fam))
    qi = int(rs.choice(members))              # uniformly at random, seed 42
    panel.extend(panel_rows(qi, "%s (n=%d)" % (fam, len(members))))
    panel_meta[fam] = dict(query_row=qi, query_pdb=meta[qi]["pdb_id"],
                           query_ligand=meta[qi]["ligand_resname"],
                           family_size=int(len(members)))

# ---- NEGATIVE CASE 1: singleton family (no same-family answer exists) ------
singletons = [f for f, c in fam_counts.items() if c == 1]
neg1_info = None
if singletons:
    rs = np.random.default_rng(SEED)
    fam1 = sorted(singletons)[int(rs.integers(0, len(singletons)))]
    qi1 = int(np.flatnonzero(P_LAB == fam1)[0])
    panel.extend(panel_rows(qi1, "NEGATIVE 1: %s (singleton family, n=1)" % fam1))
    top = top_k(qi1, 5)
    neg1_info = dict(family=fam1, query_row=qi1, query_pdb=meta[qi1]["pdb_id"],
                     query_ligand=meta[qi1]["ligand_resname"],
                     top1_pdb=meta[top[0][0]]["pdb_id"],
                     top1_cos=round(top[0][1], 4),
                     top1_same_family=bool(P_LAB[top[0][0]] == fam1),
                     WHY="A singleton family has no correct neighbour. The index "
                         "still returns a confident-looking 0.9x cosine, which is "
                         "exactly the failure mode a user must be warned about.")

# ---- NEGATIVE CASE 2: top-1 is the same ligand in a DIFFERENT target -------
scan = np.random.default_rng(SEED).choice(N, size=4000, replace=False)
Dq, Iq = FLAT.search(np.ascontiguousarray(J[scan]), 2)
neg2_info = None
for row, qi in enumerate(scan):
    j = int(Iq[row, 1] if Iq[row, 0] == qi else Iq[row, 0])
    qi = int(qi)
    if (meta[j]["ligand_resname"] == meta[qi]["ligand_resname"]
            and U_LAB[qi] and U_LAB[j] and U_LAB[j] != U_LAB[qi]
            and P_LAB[qi] and P_LAB[j] != P_LAB[qi]):
        panel.extend(panel_rows(qi, "NEGATIVE 2: same-ligand / different-target"))
        neg2_info = dict(query_row=qi, query_pdb=meta[qi]["pdb_id"],
                         query_ligand=meta[qi]["ligand_resname"],
                         query_uniprot=U_LAB[qi], query_pfam=P_LAB[qi],
                         top1_pdb=meta[j]["pdb_id"], top1_ligand=meta[j]["ligand_resname"],
                         top1_uniprot=U_LAB[j], top1_pfam=P_LAB[j],
                         top1_cos=round(float(Dq[row, 1] if Iq[row, 0] == qi
                                              else Dq[row, 0]), 4),
                         WHY="The ligand half of the joined vector is a per-CCD "
                             "fingerprint (26,619 distinct vectors over 72,199 "
                             "rows), so a shared cofactor can pull unrelated "
                             "proteins to rank 1. This case was SELECTED to "
                             "illustrate that mechanism; this notebook does not "
                             "measure how often it fires, so do not describe it "
                             "as the dominant error mode without that number.")
        break

hdr = ["family / case", "query (PDB / lig)", "rank", "hit (PDB / lig)", "cosine",
       "target", "family", "ligand"]
print("\n" + "=" * 72)
print("QUALITATIVE RETRIEVAL PANEL")
print("=" * 72)
panel_md = print_table(hdr, panel)

result("4_qualitative_panel",
    families_shown=top10_fams,
    per_family_query=panel_meta,
    negative_case_1_singleton_family=neg1_info,
    negative_case_2_same_ligand_diff_target=neg2_info,
    table_markdown=panel_md,
    NOTE="Queries were drawn uniformly at random (seed 42) from each family - "
         "not curated. The two negative cases were selected deliberately and are "
         "labelled as such; do not present them as random draws.",
)

print("\n--- paste-ready markdown ---\n")
print(panel_md)
```

---

## Cell 9 — §5a **R1.4: silhouette by family, RAW and MEAN-CENTRED**

Notebook 08 showed the corpus is anisotropic (mean-vector norm 0.84x a typical
row norm). Any separation statistic computed on the raw space inherits that
offset, so both numbers are reported and the centred one is the one to quote.

```python
from sklearn.metrics import silhouette_score

SIL_N = 10000
MIN_FAM = 10          # families need enough members for silhouette to mean anything

lab_ok = np.flatnonzero(P_LAB != "")
fc = collections.Counter(P_LAB[lab_ok])
big = {f for f, c in fc.items() if c >= MIN_FAM}
cand = np.array([i for i in lab_ok if P_LAB[i] in big])
print("candidates: %d entries in %d families with >= %d members"
      % (len(cand), len(big), MIN_FAM))

# stratified: sqrt-proportional per family, floor of MIN_FAM/2, seed 42
sizes = collections.Counter(P_LAB[cand])
fams = sorted(sizes.keys(), key=lambda f: -sizes[f])
w = np.sqrt(np.array([sizes[f] for f in fams], dtype=np.float64))
alloc = np.maximum(5, np.round(SIL_N * w / w.sum())).astype(int)
alloc = np.minimum(alloc, np.array([sizes[f] for f in fams]))

sil_ix = []
for f, k in zip(fams, alloc):
    members = cand[P_LAB[cand] == f]
    rs = np.random.default_rng(stable_seed(f))
    sil_ix.extend(rs.choice(members, size=int(min(k, len(members))),
                            replace=False).tolist())
sil_ix = np.array(sorted(set(sil_ix)))
if len(sil_ix) > SIL_N:
    # Trim at RANDOM, not with [:SIL_N]. Row index tracks PDB-ID order, so
    # keeping the lowest indices would quietly bias the sample toward one end
    # of the corpus instead of trimming it uniformly.
    sil_ix = np.sort(np.random.default_rng(SEED).choice(
        sil_ix, size=SIL_N, replace=False))
sil_lab = P_LAB[sil_ix]
# silhouette needs >= 2 members per cluster. The truncation above can orphan a
# family, so drop those. (Counter hoisted out of the comprehension - rebuilding
# it per element is quadratic.)
_sc = collections.Counter(sil_lab)
keep = np.array([_sc[l] >= 2 for l in sil_lab])
sil_ix, sil_lab = sil_ix[keep], sil_lab[keep]
print("silhouette sample: %d points, %d families" % (len(sil_ix), len(set(sil_lab))))

# MU (the corpus mean vector) was computed once in Cell 1.
X_raw = np.ascontiguousarray(J[sil_ix])
X_raw = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-12)
X_cen = np.ascontiguousarray(J[sil_ix] - MU)
X_cen = X_cen / (np.linalg.norm(X_cen, axis=1, keepdims=True) + 1e-12)

t0 = time.time()
sil_raw = float(silhouette_score(X_raw, sil_lab, metric="cosine"))
sil_cen = float(silhouette_score(X_cen, sil_lab, metric="cosine"))
print("silhouette computed in %.0fs" % (time.time() - t0))

# same statistic on the two modalities separately, for the co-equal-modality claim
Praw = np.asarray(pocket[p_off[sil_ix]], dtype=np.float32)
Lraw = np.asarray(ligand[l_off[sil_ix]], dtype=np.float32)
Psub = Praw / (np.linalg.norm(Praw, axis=1, keepdims=True) + 1e-12)
Lsub = Lraw / (np.linalg.norm(Lraw, axis=1, keepdims=True) + 1e-12)
# Centre on the FULL-corpus pocket mean, not the subsample mean, so this is the
# same correction applied to the joined space above. It must be averaged over
# the pocket rows the JOINED INDEX uses (p_off), not over every row of
# pocket_embeddings.npy: that file also holds the entries the index dropped for
# having a zero-norm ligand, and averaging over them would centre this space on
# a different population than MU centres the joined space on.
# Chunked float64 accumulation over the memmap: no 700 MB temporary, no float32
# summation drift.
_rows = np.sort(p_off)
_acc = np.zeros(pocket.shape[1], dtype=np.float64)
for _st in range(0, len(_rows), 8192):
    _acc += np.asarray(pocket[_rows[_st:_st + 8192]], dtype=np.float64).sum(axis=0)
MU_P = (_acc / len(_rows)).astype(np.float32)
Pcen = Praw - MU_P
Pcen = Pcen / (np.linalg.norm(Pcen, axis=1, keepdims=True) + 1e-12)

sil_p_raw = float(silhouette_score(Psub, sil_lab, metric="cosine"))
sil_p_cen = float(silhouette_score(Pcen, sil_lab, metric="cosine"))
sil_l_raw = float(silhouette_score(Lsub, sil_lab, metric="cosine"))

result("5a_silhouette_by_family",
    n_points=int(len(sil_ix)),
    n_families=int(len(set(sil_lab))),
    min_family_size_in_corpus=MIN_FAM,
    silhouette_joined_RAW=round(sil_raw, 4),
    silhouette_joined_MEAN_CENTRED=round(sil_cen, 4),
    silhouette_pocket_only_RAW=round(sil_p_raw, 4),
    silhouette_pocket_only_MEAN_CENTRED=round(sil_p_cen, 4),
    silhouette_ligand_only_RAW=round(sil_l_raw, 4),
    delta_centred_minus_raw_joined=round(sil_cen - sil_raw, 4),
    INTERPRETATION=("Silhouette near 0 means families overlap heavily in the "
                    "ambient space even where retrieval works: kNN ordering can "
                    "be informative while global cluster geometry is not. Quote "
                    "the MEAN-CENTRED value; the raw value is inflated by the "
                    "corpus mean direction."),
    NOTE="Cosine metric, Pfam labels, stratified sample, seed 42. Families with "
         "fewer than %d entries are excluded (silhouette is undefined for "
         "singletons and unstable for tiny clusters)." % MIN_FAM,
)
```

---

## Cell 10 — §5b **R1.4: UMAP coloured by EC class + the caption the paper must use**

```python
!pip -q install umap-learn
import umap
# Imported HERE as well as in Cells 7 and 9. This cell advertises a fallback for
# "Cell 9 was skipped", and that fallback is unreachable if the cell dies on a
# NameError for silhouette_score first.
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

UMAP_N = 20000
SIL_SAMPLE = 10000        # silhouette on 20k x 3328 twice is needlessly slow

# Sample from the WHOLE index, not just EC-annotated entries. The unannotated
# share is large (Cell 1 prints the exact labelled count) and hiding it would
# make the figure look tidier than the data is.
umap_ix = np.sort(np.random.default_rng(SEED).choice(
    N, size=min(UMAP_N, N), replace=False))

Xu = np.ascontiguousarray(J[umap_ix])
Xu = Xu / (np.linalg.norm(Xu, axis=1, keepdims=True) + 1e-12)
lab_u = np.array([e if e else "none" for e in EC_LAB[umap_ix]])

t0 = time.time()
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                    random_state=SEED, n_components=2, verbose=True)
Y = reducer.fit_transform(Xu)
print("UMAP done in %.0f s" % (time.time() - t0))

EC_NAMES = {"1": "1 oxidoreductase", "2": "2 transferase", "3": "3 hydrolase",
            "4": "4 lyase", "5": "5 isomerase", "6": "6 ligase",
            "7": "7 translocase", "none": "no EC"}
order = sorted(set(lab_u), key=lambda x: (x == "none", x))
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(8, 7))
for k, cls in enumerate(order):
    sel = lab_u == cls
    ax.scatter(Y[sel, 0], Y[sel, 1], s=2, alpha=0.45,
               color=("lightgrey" if cls == "none" else cmap(k % 10)),
               label="%s (n=%d)" % (EC_NAMES.get(cls, cls), int(sel.sum())),
               linewidths=0)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("UMAP of joined pocket+ligand embeddings, coloured by EC class")
ax.legend(markerscale=6, fontsize=8, loc="best", frameon=False)
fig.tight_layout()
fig.savefig(str(OUT_DIR / "fig_umap_ec_class.png"), dpi=160)
print("wrote", OUT_DIR / "fig_umap_ec_class.png")
plt.show()

# per-class local coherence IN THE AMBIENT SPACE (not on the projection)
ii = faiss.IndexFlatIP(Xu.shape[1]); ii.add(Xu)
_, Iu = ii.search(Xu, 11)
purity = {}
for cls in order:
    sel = np.flatnonzero(lab_u == cls)
    if len(sel) < 20:
        continue
    vals = []
    for r in sel:
        nb = [int(j) for j in Iu[r] if int(j) != r][:10]
        vals.append(float(np.mean([lab_u[j] == cls for j in nb])))
    purity[EC_NAMES.get(cls, cls)] = round(float(np.mean(vals)), 4)

sil_ec_raw = float(silhouette_score(Xu, lab_u, metric="cosine",
                                   sample_size=min(SIL_SAMPLE, len(Xu)),
                                   random_state=SEED))
Xu_c = np.ascontiguousarray(J[umap_ix] - MU)
Xu_c = Xu_c / (np.linalg.norm(Xu_c, axis=1, keepdims=True) + 1e-12)
sil_ec_cen = float(silhouette_score(Xu_c, lab_u, metric="cosine",
                                    sample_size=min(SIL_SAMPLE, len(Xu)),
                                    random_state=SEED))

# The Pfam numbers come from Cell 9. Fall back gracefully so this cell still
# produces a usable caption if Cell 9 was skipped.
_s5a = RESULTS.get("5a_silhouette_by_family", {})
pfam_raw = _s5a.get("silhouette_joined_RAW")
pfam_cen = _s5a.get("silhouette_joined_MEAN_CENTRED")
pfam_n = _s5a.get("n_points")
if pfam_raw is None:
    pfam_clause = ("Silhouette by Pfam family was not computed in this run "
                   "(section 5a skipped). ")
else:
    pfam_clause = ("Silhouette by Pfam family on a %d-complex stratified sample "
                   "is %.3f raw and %.3f mean-centred. " % (pfam_n, pfam_raw, pfam_cen))

mean_purity = float(np.mean(list(purity.values()))) if purity else float("nan")

CAPTION = (
    "Figure X. UMAP projection of the joined pocket+ligand embeddings for a "
    "random subsample of %d complexes (cosine metric, n_neighbors=30, "
    "min_dist=0.1, random_state=42), coloured by first-level EC class. "
    "The projection is ILLUSTRATIVE ONLY and must not be read as evidence of "
    "cluster structure: UMAP does not preserve distances or densities, and its "
    "layout is sensitive to n_neighbors and to the random seed. The quantitative "
    "statements are made in the AMBIENT space: silhouette coefficient by EC "
    "class is %.3f raw and %.3f after subtracting the corpus mean vector. "
    "%s"
    "Mean 10-nearest-neighbour EC-class purity in the ambient "
    "space is %.3f. Silhouette values near zero indicate that functional classes "
    "overlap substantially in the embedding space even where nearest-neighbour "
    "retrieval is informative; the embedding supports local retrieval, not global "
    "class separation."
    % (len(umap_ix), sil_ec_raw, sil_ec_cen, pfam_clause, mean_purity)
)

result("5b_umap_and_separation",
    n_points=int(len(umap_ix)),
    umap_params="metric=cosine, n_neighbors=30, min_dist=0.1, random_state=42",
    ec_class_counts={EC_NAMES.get(c, c): int((lab_u == c).sum()) for c in order},
    silhouette_by_EC_RAW=round(sil_ec_raw, 4),
    silhouette_by_EC_MEAN_CENTRED=round(sil_ec_cen, 4),
    knn10_purity_by_EC_class=purity,
    figure_path=str(OUT_DIR / "fig_umap_ec_class.png"),
    CAPTION_TEXT=CAPTION,
    NOTE="The caption states the ambient-space metrics and explicitly labels the "
         "projection illustrative. Use it verbatim; a UMAP with no ambient-space "
         "number attached is what R1.4 objected to.",
)

print("\n--- caption ---\n")
print(CAPTION)
```

---

## Cell 11 — Cross-section summary: the three sentences for the response letter

```python
def _g(sec, key, default=None):
    return RESULTS.get(sec, {}).get(key, default)

lines = []
if "1c_encoder_comparison" in RESULTS:
    lines.append("R1.3 (encoder choice): %s Best language model %s reaches "
                 "same-Pfam hit@1 %s; the 20-d amino-acid-composition control "
                 "(%s) reaches %s over the same %s complexes and the same "
                 "pooled residues (gap %s)."
                 % (_g("1c_encoder_comparison", "HEADLINE", ""),
                    _g("1c_encoder_comparison", "best_language_model"),
                    _g("1c_encoder_comparison", "best_language_model_hit1_pfam"),
                    _g("1c_encoder_comparison", "best_composition_control"),
                    _g("1c_encoder_comparison", "best_composition_control_hit1_pfam"),
                    _g("1c_encoder_comparison", "n_complexes"),
                    _g("1c_encoder_comparison", "gap_lm_minus_control_hit1_pfam")))
if "2_dataset_overlap" in RESULTS:
    lines.append("R2.3 (overlap): %s%% of the %d unique PDB entries in this index "
                 "appear in at least one of %s; %s%% appear in none of them. "
                 "Unreachable sources: %s."
                 % (_g("2_dataset_overlap", "pct_in_at_least_one_fetched_source"),
                    _g("2_dataset_overlap", "n_our_unique_pdb_ids"),
                    _g("2_dataset_overlap", "sources_fetched"),
                    _g("2_dataset_overlap", "pct_in_NONE_of_the_fetched_sources"),
                    list((_g("2_dataset_overlap", "sources_NOT_REACHABLE") or {}).keys())))
if "3_diversity" in RESULTS:
    by = _g("3_diversity", "by_label", {})
    u = by.get("uniprot", {})
    lines.append("Diversity: %d unique UniProt targets, %d unique Pfam families; "
                 "Gini over entries-per-UniProt = %s, top-10 targets account for "
                 "%s of labelled entries and the top 1000 for %s."
                 % (u.get("n_unique", 0), by.get("pfam", {}).get("n_unique", 0),
                    u.get("gini"), u.get("top10_share"), u.get("top1000_share")))
if "5a_silhouette_by_family" in RESULTS:
    lines.append("R1.4 (separation): silhouette by Pfam is %s raw / %s "
                 "mean-centred (cosine, %s complexes). The UMAP is illustrative; "
                 "the ambient-space numbers are the claim."
                 % (_g("5a_silhouette_by_family", "silhouette_joined_RAW"),
                    _g("5a_silhouette_by_family", "silhouette_joined_MEAN_CENTRED"),
                    _g("5a_silhouette_by_family", "n_points")))

result("6_response_letter_sentences", **{("s%d" % i): s for i, s in enumerate(lines, 1)})
for s in lines:
    print("\n* " + s)
```

---

## Cell 12 — Write `audit_results_2.json` + `audit_results_2.md`

```python
out_json = OUT_DIR / "audit_results_2.json"
# encoding pinned: the section titles and captions carry non-ASCII (— and §),
# and write_text() otherwise uses the platform default codec.
out_json.write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")

md = ["# Sci Data revision — remaining reviewer analyses (audit part 2)", "",
      "Generated by `notebooks/10_remaining_analyses.ipynb`. Seed 42 throughout.",
      ""]
for sec, kv in RESULTS.items():
    md.append("## %s" % sec)
    for k, v in kv.items():
        if isinstance(v, str) and "\n" in v:
            md.append("- **%s**:" % k)
            md.append("")
            md.append(v)
            md.append("")
        else:
            md.append("- **%s**: `%s`" % (k, v))
    md.append("")
(OUT_DIR / "audit_results_2.md").write_text("\n".join(md), encoding="utf-8")

print("Wrote:")
for p in [out_json, OUT_DIR / "audit_results_2.md",
          OUT_DIR / "fig_redundancy_curve.png", OUT_DIR / "fig_umap_ec_class.png"]:
    if p.exists():
        print("  %-40s %8.2f MB" % (p.name, p.stat().st_size / 1e6))

print("\nSections completed:", list(RESULTS.keys()))

try:
    from google.colab import files
    files.download(str(out_json))
    files.download(str(OUT_DIR / "audit_results_2.md"))
except Exception:
    pass
```
