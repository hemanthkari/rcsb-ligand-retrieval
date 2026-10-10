# Sci Data revision — full data audit (one run, one notebook)

Every data-dependent number needed for the Scientific Data major revision
(submission `9625e99c-515f-4d60-b5da-cbd12d8d1456`, deadline 02 Sep 2026).

Each section states **which reviewer item it answers** and prints a
`RESULT` block you can paste straight into the response letter. Everything
also accumulates into a `RESULTS` dict that Cell 15 writes to
`audit_results.json` + `audit_results.md`.

**Runtime:** ~45–90 min on a Colab T4/A100 after the ~4 GB download.
No GPU strictly required except the optional encoder comparison.

**Not covered here** (needs the mmCIF archive, separate long-running job):
the 3.M2 residue→token map re-extraction and the V1 chimeric-sequence
control. Those get their own notebook once we know whether the April 2026
mirror still exists.

**Run cells in order.** Cell 1 must run before any analysis cell. After
that, sections are independent — if one fails, skip it and continue.

---

## Cell 0 — Setup: install dependencies and download the dataset

```python
# ~4 GB download. Takes 5-15 min depending on Colab's link.
!pip -q install huggingface_hub faiss-cpu rdkit numpy pandas scipy pyarrow requests h5py

import os, json, math, hashlib, itertools, collections
from pathlib import Path
import numpy as np

from huggingface_hub import snapshot_download

REPO = "HemanthKari/rcsb-ligand-retrieval-v0.1"
LOCAL = Path(snapshot_download(REPO, repo_type="dataset"))
# ---- persistent checkpoint location ------------------------------------
# Every result() below is flushed here immediately, so a Colab disconnect
# never costs completed work. Drive is preferred because /content is wiped
# when the runtime recycles. Mount Drive BEFORE running this cell.
_CKPT_CANDIDATES = [Path("/content/drive/MyDrive/fluxion_audit"),
                    Path("/content/fluxion_audit")]
_DRIVE_ROOT = Path("/content/drive/MyDrive")
CKPT_DIR = None
for _c in _CKPT_CANDIDATES:
    # NEVER fabricate a tree under /content/drive. mkdir(parents=True) on an
    # unmounted Drive creates a fake local /content/drive/MyDrive, which then
    # blocks the real mount with "Mountpoint must not already contain files".
    if str(_c).startswith(str(_DRIVE_ROOT)) and not _DRIVE_ROOT.is_dir():
        continue
    try:
        _c.mkdir(parents=True, exist_ok=True)
        CKPT_DIR = _c; break
    except Exception:
        continue
CKPT_DIR = CKPT_DIR or Path(".")
CKPT_JSON = CKPT_DIR / "audit_results.json"
CKPT_MD   = CKPT_DIR / "audit_results.md"

# Report the TRUTH: on-Drive means genuinely under a mounted MyDrive, not a
# path that merely contains the word "drive".
_ON_DRIVE = _DRIVE_ROOT.is_dir() and str(CKPT_DIR).startswith(str(_DRIVE_ROOT))
print("checkpoint dir :", CKPT_DIR,
      "(ON DRIVE - survives disconnect)" if _ON_DRIVE
      else "(EPHEMERAL - mount Drive and re-run this cell, or you WILL lose results)")

print("Dataset at:", LOCAL)
for p in sorted(LOCAL.rglob("*")):
    if p.is_file():
        print("  %-45s %8.1f MB" % (str(p.relative_to(LOCAL)), p.stat().st_size / 1e6))
```

---

## Cell 1 — Load artifacts, define helpers, init RESULTS

```python
import faiss

# ---- resume from a previous (possibly crashed) run ---------------------
RESULTS = {}
if CKPT_JSON.exists():
    try:
        RESULTS = json.loads(CKPT_JSON.read_text())
        print("RESUMED %d section(s) from %s" % (len(RESULTS), CKPT_JSON))
        print("  already done:", ", ".join(sorted(RESULTS)) or "(none)")
    except Exception as _e:
        print("checkpoint unreadable (%s) - starting fresh" % type(_e).__name__)


def _flush_results():
    """Persist RESULTS after EVERY update, via temp-file replace."""
    try:
        _tmp = CKPT_JSON.with_suffix(".tmp")
        _tmp.write_text(json.dumps(RESULTS, indent=2, default=str))
        _tmp.replace(CKPT_JSON)
        _md = ["# Sci Data revision - data audit results", ""]
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

# ---- core artifacts -------------------------------------------------------
meta = [json.loads(l) for l in open(LOCAL / "metadata.jsonl")]
pocket_manifest = [json.loads(l) for l in open(LOCAL / "pockets" / "pocket_manifest.jsonl")]
ligand_manifest = [json.loads(l) for l in open(LOCAL / "ligands" / "ligand_manifest.jsonl")]

# ---------------------------------------------------------------------------
# EMBEDDING VERSION SWITCH
# Set V02_POCKET to the per-chain re-embedded array from notebook 11 to measure
# v0.2. Leave as None to reproduce the v0.1 numbers.
#
# IMPORTANT: joined_embeddings.npy shipped with v0.1 was built from the v0.1
# pocket half, so it is STALE once the pocket embeddings change. When V02_POCKET
# is set we rebuild the concatenation here, using the same construction as the
# deposited build_index.py (row-wise L2 over [pocket ; ligand]), so that every
# downstream cell measures a self-consistent set of vectors.
# ---------------------------------------------------------------------------
# Leave as "auto" to DETECT the v0.2 per-chain embeddings from notebook 11 and
# fall back to the shipped v0.1 array if they are absent. Set to an explicit path
# to force v0.2, or to None to force v0.1.
V02_POCKET = "auto"
# Falling back to v0.1 is now an EXPLICIT choice. A silent fallback (Drive not
# mounted) previously wasted a full 90-minute run measuring the wrong array.
ALLOW_V01_FALLBACK = False

_CANDIDATES = [
    "/content/drive/MyDrive/fluxion_v02/pocket_embeddings.npy",
    "/content/drive/MyDrive/fluxion_reextract/pocket_embeddings.npy",
    "/content/fluxion_v02/pocket_embeddings.npy",
    "./fluxion_v02/pocket_embeddings.npy",
]

def _mount_drive():
    try:
        from google.colab import drive as _drv
        _drv.mount("/content/drive")
    except Exception:
        pass

if V02_POCKET == "auto":
    if not any(Path(c).exists() for c in _CANDIDATES):
        _mount_drive()
    V02_POCKET = next((c for c in _CANDIDATES if Path(c).exists()), None)
    if V02_POCKET:
        print("auto-detect: %s" % V02_POCKET)
    elif ALLOW_V01_FALLBACK:
        print("auto-detect: no v0.2 array -> measuring SHIPPED v0.1 "
              "(explicitly allowed via ALLOW_V01_FALLBACK)")
    else:
        raise SystemExit(chr(10).join([
            "",
            "=" * 72,
            "STOPPED: no v0.2 pocket embeddings found - Drive is not mounted.",
            "=" * 72,
            "Continuing would silently measure the v0.1 embeddings and waste",
            "the entire run. Fix this before going further:",
            "",
            "  1. Run this in a cell and approve the popup:",
            "",
            "         from google.colab import drive",
            "         drive.mount('/content/drive')",
            "",
            "  2. Re-run Cell 0 (it picks the checkpoint dir), then this cell.",
            "",
            "Paths searched:",
        ] + ["  - " + c for c in _CANDIDATES] + [
            "",
            "To measure v0.1 ON PURPOSE, set ALLOW_V01_FALLBACK = True above.",
            "=" * 72,
        ]))
elif V02_POCKET:
    if not Path(V02_POCKET).exists():
        _mount_drive()
    if not Path(V02_POCKET).exists():
        _msg = [
            "V02_POCKET not found: " + str(V02_POCKET),
            "  * the Drive path is /content/drive/MyDrive/... "
            "(a missing 'drive/' is the usual cause)",
            "  * notebook 11 writes <DRIVE>/fluxion_v02/pocket_embeddings.npy",
            "  * set V02_POCKET = None to measure the shipped v0.1 embeddings",
        ]
        raise SystemExit(chr(10).join(_msg))

ligand  = np.load(LOCAL / "ligands" / "ligand_embeddings.npy", mmap_mode="r")   # unchanged in v0.2
N = len(meta)
p_off = np.array([m["pocket_emb_offset"] for m in meta], dtype=np.int64)
l_off = np.array([m["ligand_emb_offset"] for m in meta], dtype=np.int64)

if V02_POCKET:
    pocket = np.load(V02_POCKET, mmap_mode="r")
    _P = np.asarray(pocket, dtype=np.float32)[p_off]
    _L = np.asarray(ligand, dtype=np.float32)[l_off]
    _J = np.hstack([_P, _L]).astype(np.float32)
    _J /= (np.linalg.norm(_J, axis=1, keepdims=True) + 1e-12)
    joined = _J
    EMB_VERSION = "v0.2 (per-chain)"
    JOINED_SOURCE = "rebuilt in-notebook from v0.2 pocket + unchanged ligand"
    del _P, _L, _J
else:
    pocket = np.load(LOCAL / "pockets" / "pocket_embeddings.npy", mmap_mode="r")
    joined = np.load(LOCAL / "joined_embeddings.npy", mmap_mode="r")
    EMB_VERSION = "v0.1 (concatenated-input)"
    JOINED_SOURCE = "shipped joined_embeddings.npy"

POCKET_DIM, LIGAND_DIM = pocket.shape[1], ligand.shape[1]

print("")
print("#" * 72)
print("#  EMBEDDINGS IN USE : " + EMB_VERSION.ljust(47) + "#")
print("#  joined matrix     : " + JOINED_SOURCE[:47].ljust(47) + "#")
if not V02_POCKET:
    print("#  " + "WARNING: these are the SHIPPED v0.1 embeddings.".ljust(67) + "#")
    print("#  " + "If you meant to measure v0.2, stop and check the path.".ljust(67) + "#")
print("#" * 72)
print("")
print("metadata rows      :", N)
print("joined             :", joined.shape, joined.dtype)
print("pocket             :", pocket.shape)
print("ligand             :", ligand.shape)
print("pocket_dim/lig_dim :", POCKET_DIM, LIGAND_DIM)
RESULTS["0_provenance"] = {"embedding_version": EMB_VERSION,
                           "joined_source": JOINED_SOURCE,
                           "v02_pocket_path": str(V02_POCKET)}

RNG = np.random.default_rng(42)   # every sample in this notebook uses seed 42
```

---

## Cell 1b — **Artifact fingerprints + full-corpus integrity sweep**

Every number here is only meaningful against a specific snapshot. This cell
fingerprints each analysed file, then checks the *entire* corpus — not a sample
— for the internal-consistency failures a referee would probe.

```python
import hashlib as _hl
import h5py
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# ---- 1. fingerprint every analysed artifact ------------------------------
_FILES = ["metadata.jsonl", "joined_embeddings.npy", "retrieval.faiss",
          "pockets/pocket_embeddings.npy", "pockets/pocket_manifest.jsonl",
          "pockets/pocket_coords.h5", "ligands/ligand_embeddings.npy",
          "ligands/ligand_manifest.jsonl", "ligands/ligand_coords.h5",
          "splits/train_indices.npy", "splits/val_indices.npy",
          "raft_pdb_filtered.jsonl", "config.json"]

def _sha(path, cap=256 * 1024 * 1024):
    h = _hl.sha256(); n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk); n += len(chunk)
            if n >= cap:
                break
    return h.hexdigest()[:16] + ("+" if n >= cap else "")

_fp = {}
for _f in _FILES:
    _q = LOCAL / _f
    _fp[_f] = ({"sha256_16": _sha(_q), "bytes": _q.stat().st_size}
               if _q.exists() else "ABSENT")
if V02_POCKET:
    _fp["V02 pocket_embeddings.npy"] = {"sha256_16": _sha(V02_POCKET),
                                        "bytes": Path(V02_POCKET).stat().st_size}
result("1b_artifact_fingerprints", **_fp,
    NOTE="Quote these so every number below is tied to an identifiable snapshot. "
         "A trailing '+' means the hash covers the first 256 MB only.")

# ---- 2. offsets: collisions and coverage (FULL corpus) -------------------
_pu, _lu = np.unique(p_off), np.unique(l_off)
result("1c_offset_integrity",
    n_rows=N,
    distinct_pocket_offsets=int(_pu.size), distinct_ligand_offsets=int(_lu.size),
    pocket_offset_collisions=int(N - _pu.size),
    ligand_offset_collisions=int(N - _lu.size),
    pocket_rows_unused=int(pocket.shape[0] - _pu.size),
    ligand_rows_unused=int(ligand.shape[0] - _lu.size),
    NOTE="A collision means two metadata rows share an embedding row - two "
         "complexes silently carrying the same vector. Must be 0.")

# ---- 3. embeddings: non-finite and degenerate rows (FULL corpus) ---------
_Pa = np.asarray(pocket, dtype=np.float32)
_La = np.asarray(ligand, dtype=np.float32)
_Ja = np.asarray(joined, dtype=np.float32)
_pn_all, _ln_all = np.linalg.norm(_Pa, axis=1), np.linalg.norm(_La, axis=1)
result("1d_embedding_health",
    pocket_nonfinite_rows=int((~np.isfinite(_Pa)).any(axis=1).sum()),
    ligand_nonfinite_rows=int((~np.isfinite(_La)).any(axis=1).sum()),
    joined_nonfinite_rows=int((~np.isfinite(_Ja)).any(axis=1).sum()),
    pocket_zero_norm_rows=int((_pn_all < 1e-8).sum()),
    ligand_zero_norm_rows=int((_ln_all < 1e-8).sum()),
    joined_max_deviation_from_unit_norm=round(
        float(np.abs(np.linalg.norm(_Ja, axis=1) - 1.0).max()), 6),
    NOTE="joined rows must be unit-norm. Ligand zero-norm rows are the 189 "
         "documented parse failures and must not appear in the joined index.")

# ---- 4. HDF5 shapes vs metadata counts (FULL corpus) --------------------
_lig5 = h5py.File(LOCAL / "ligands" / "ligand_coords.h5", "r")
_pkt5 = h5py.File(LOCAL / "pockets" / "pocket_coords.h5", "r")
_lm = _pm = _lmiss = _pmiss = _lbad = _pbad = 0
for _m in meta:
    _k = _m["pocket_key"]
    _lk, _pk = "ligands/" + _k, "pockets/" + _k
    if _lk in _lig5:
        _s = _lig5[_lk].shape
        if len(_s) != 2 or _s[1] != 3:
            _lbad += 1
        elif _m.get("ligand_heavy_atoms") is not None and _s[0] != _m["ligand_heavy_atoms"]:
            _lm += 1
    else:
        _lmiss += 1
    if _pk in _pkt5:
        _s = _pkt5[_pk].shape
        if len(_s) != 2 or _s[1] != 3:
            _pbad += 1
        elif _m.get("pocket_n_residues") is not None and _s[0] != _m["pocket_n_residues"]:
            _pm += 1
    else:
        _pmiss += 1
_lig5.close(); _pkt5.close()
result("1e_hdf5_vs_metadata",
    n_checked=N,
    ligand_datasets_missing=_lmiss, ligand_shape_malformed=_lbad,
    ligand_rows_ne_ligand_heavy_atoms=_lm,
    pocket_datasets_missing=_pmiss, pocket_shape_malformed=_pbad,
    pocket_rows_ne_pocket_n_residues=_pm,
    NOTE="A row-count mismatch means coordinates and metadata describe different "
         "molecules. This is the check that would expose altloc double-counting "
         "in ligand_heavy_atoms.")

# ---- 5. SMILES validity over the WHOLE corpus ---------------------------
_bad_smiles, _empty = [], 0
for _m in meta:
    _s = _m.get("ligand_smiles") or ""
    if not _s:
        _empty += 1
        continue
    if Chem.MolFromSmiles(_s) is None:
        _bad_smiles.append(_m["pocket_key"])
result("1f_smiles_validity_full_corpus",
    n_rows=N, n_empty=_empty, n_unparseable=len(_bad_smiles),
    pct_unparseable=round(100 * len(_bad_smiles) / max(1, N), 4),
    examples=_bad_smiles[:8],
    NOTE="Tests every shipped SMILES, not only the 189 zero-vector entries. Any "
         "unparseable string is redistributed data a user cannot consume.")

# ---- 6. FAISS index vs the matrix it claims to index --------------------
try:
    import faiss as _fa
    _ix = _fa.read_index(str(LOCAL / "retrieval.faiss"))
    try:
        _s = RNG.choice(N, 200, replace=False)
        _rec = np.stack([_ix.reconstruct(int(i)) for i in _s])
        _recon = round(float(np.abs(_rec - np.asarray(joined[_s], dtype=np.float32)).max()), 6)
    except Exception as _e:
        _recon = "reconstruct unsupported: %s" % type(_e).__name__
    result("1g_faiss_consistency",
        index_ntotal=int(_ix.ntotal), metadata_rows=N,
        ntotal_matches_metadata=bool(int(_ix.ntotal) == N),
        index_dim=int(_ix.d), expected_dim=POCKET_DIM + LIGAND_DIM,
        max_reconstruct_error_vs_joined=_recon,
        NOTE=("Under v0.2 the shipped index is a v0.1 artifact, so a reconstruct "
              "mismatch here is EXPECTED and confirms the index must be rebuilt."
              if V02_POCKET else
              "Confirms the released index indexes the released matrix."))
except Exception as _e:
    print("FAISS consistency check skipped:", _e)
```

---

## Cell 1h — **Full-corpus provenance verification (every entry, not a sample)**

One batched GraphQL fetch verifies three Methods claims across all entries and
supplies the deposition dates needed for the temporal analysis in Cell 13i.
Cached to disk, so a re-run costs nothing.

```python
import requests, time as _t, json as _js

# cached on Drive: re-fetching 72,199 entries is the longest step here
_CACHE = CKPT_DIR / "rcsb_entry_meta.json"
_ids = sorted({m["pdb_id"].upper() for m in meta})
print("unique PDB entries to verify: %d" % len(_ids))

_entry_meta = {}
if _CACHE.exists():
    _entry_meta = _js.loads(_CACHE.read_text())
    print("cache hit: %d entries" % len(_entry_meta))

_todo = [i for i in _ids if i not in _entry_meta]
_Q = ("query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id "
      "exptl{method} rcsb_entry_info{resolution_combined} "
      "rcsb_accession_info{initial_release_date deposit_date}}}")

_fail = 0
for _s in range(0, len(_todo), 250):
    _batch = _todo[_s:_s + 250]
    for _attempt in range(3):
        try:
            _r = requests.post("https://data.rcsb.org/graphql",
                               json={"query": _Q, "variables": {"ids": _batch}},
                               timeout=90)
            _d = _r.json().get("data", {}).get("entries") or []
            for _e in _d:
                if not _e:
                    continue
                _ri = _e.get("rcsb_entry_info") or {}
                _ai = _e.get("rcsb_accession_info") or {}
                _res = _ri.get("resolution_combined")
                _entry_meta[_e["rcsb_id"]] = {
                    "methods": [x.get("method") for x in (_e.get("exptl") or [])],
                    "resolution": (float(_res[0]) if isinstance(_res, list) and _res
                                   else (float(_res) if _res is not None else None)),
                    "released": _ai.get("initial_release_date"),
                    "deposited": _ai.get("deposit_date")}
            break
        except Exception:
            _t.sleep(2 * (_attempt + 1))
    else:
        _fail += len(_batch)
    if (_s // 250) % 20 == 0:
        print("  %d/%d fetched" % (len(_entry_meta), len(_ids)), flush=True)
        _CACHE.write_text(_js.dumps(_entry_meta))
_CACHE.write_text(_js.dumps(_entry_meta))

_have = [_entry_meta[i] for i in _ids if i in _entry_meta]
_meth = collections.Counter()
for _e in _have:
    for _mm in (_e["methods"] or ["UNKNOWN"]):
        _meth[_mm] += 1
_xray_only = sum(1 for _e in _have if _e["methods"] == ["X-RAY DIFFRACTION"])
_non_xray = [i for i in _ids if i in _entry_meta
             and "X-RAY DIFFRACTION" not in (_entry_meta[i]["methods"] or [])]
_res_vals = [_e["resolution"] for _e in _have if _e["resolution"] is not None]
_over = [i for i in _ids if i in _entry_meta
         and (_entry_meta[i]["resolution"] or 0) > 2.5]

result("1h_full_corpus_provenance",
    n_entries=len(_ids), n_verified=len(_have), n_fetch_failures=_fail,
    method_counts=dict(_meth),
    n_xray_only=_xray_only,
    n_WITHOUT_xray=len(_non_xray), examples_without_xray=_non_xray[:10],
    pct_with_xray=round(100 * (len(_have) - len(_non_xray)) / max(1, len(_have)), 4),
    resolution_max=round(max(_res_vals), 3) if _res_vals else None,
    n_resolution_over_2p5=len(_over), examples_over_2p5=_over[:10],
    VERDICT=("both Methods claims hold across the full corpus"
             if not _non_xray and not _over else
             "EXCEPTIONS FOUND - the Methods text must be qualified"),
    NOTE="Replaces the 600-entry sample. Verifies BOTH the X-ray claim and the "
         "2.5 A resolution cutoff for every entry, and caches deposition dates "
         "for the temporal analysis.")
```

---

## Cell 1i — **Structural and geometric integrity (full corpus)**

Do the shipped coordinates actually describe the complexes the metadata claims?

```python
import h5py as _h5

_lig5 = _h5.File(LOCAL / "ligands" / "ligand_coords.h5", "r")
_pkt5 = _h5.File(LOCAL / "pockets" / "pocket_coords.h5", "r")

_d_lig_pkt, _nan_rows, _absurd = [], 0, 0
_checked = 0
for _m in meta:
    _k = _m["pocket_key"]
    _lk, _pk = "ligands/" + _k, "pockets/" + _k
    if _lk not in _lig5 or _pk not in _pkt5:
        continue
    _lc = np.asarray(_lig5[_lk][:], dtype=np.float32)
    _pc = np.asarray(_pkt5[_pk][:], dtype=np.float32)
    if _lc.size == 0 or _pc.size == 0:
        continue
    if not (np.isfinite(_lc).all() and np.isfinite(_pc).all()):
        _nan_rows += 1
        continue
    if np.abs(_lc).max() > 1e4 or np.abs(_pc).max() > 1e4:
        _absurd += 1
    # closest pocket-residue centroid to any ligand heavy atom
    _dm = np.sqrt(((_pc[:, None, :] - _lc[None, :, :]) ** 2).sum(-1)).min()
    _d_lig_pkt.append(float(_dm))
    _checked += 1
_lig5.close(); _pkt5.close()

_d = np.array(_d_lig_pkt)
result("1i_geometric_integrity",
    n_checked=_checked,
    nonfinite_coordinate_entries=_nan_rows,
    absurd_coordinate_entries=_absurd,
    min_ligand_to_pocket_centroid_median=round(float(np.median(_d)), 3),
    p95=round(float(np.percentile(_d, 95)), 3),
    max=round(float(_d.max()), 3),
    n_over_8A=int((_d > 8.0).sum()),
    n_over_15A=int((_d > 15.0).sum()),
    pct_over_15A=round(100 * float((_d > 15.0).mean()), 4),
    NOTE="Pocket residues were selected by ANY heavy atom within 8 A, so a "
         "residue CENTROID can legitimately sit slightly beyond 8 A. A centroid "
         "far beyond that (>15 A) indicates ligand and pocket coordinates that "
         "do not belong to the same complex - a mis-pairing this check exists "
         "to catch.")

# ---- metadata vs the two manifests (full corpus) -------------------------
_pm_by = {r["pocket_key"]: r for r in pocket_manifest}
_lm_by = {r["pocket_key"]: r for r in ligand_manifest}
_mis_res = _mis_chain = _mis_num = _absent = 0
for _m in meta:
    _r = _pm_by.get(_m["pocket_key"])
    if _r is None:
        _absent += 1
        continue
    if str(_r.get("ligand_resname")) != str(_m.get("ligand_resname")): _mis_res += 1
    if str(_r.get("ligand_chain")) != str(_m.get("ligand_chain")):     _mis_chain += 1
    if int(_r.get("ligand_resnum", -1)) != int(_m.get("ligand_resnum", -2)): _mis_num += 1
result("1j_metadata_manifest_agreement",
    n_rows=N, pocket_manifest_absent=_absent,
    ligand_resname_mismatch=_mis_res,
    ligand_chain_mismatch=_mis_chain,
    ligand_resnum_mismatch=_mis_num,
    NOTE="metadata.jsonl and pocket_manifest.jsonl must describe the same ligand "
         "for the same pocket_key. Any mismatch is a join error.")

# ---- duplicate detection that is robust to float noise -------------------
_Jf = np.asarray(joined, dtype=np.float32)
_ii = faiss.IndexFlatIP(_Jf.shape[1]); _ii.add(np.ascontiguousarray(_Jf))
_D1, _I1 = _ii.search(np.ascontiguousarray(_Jf), 2)
_nn_cos = np.where(_I1[:, 0] == np.arange(N), _D1[:, 1], _D1[:, 0])
result("1k_duplicates_full_corpus",
    n_rows=N,
    exact_bytewise_groups="see 7_duplicates_and_leakage",
    frac_nn_ge_0p99999=round(float((_nn_cos >= 0.99999).mean()), 4),
    frac_nn_ge_0p9999=round(float((_nn_cos >= 0.9999).mean()), 4),
    frac_nn_ge_0p999=round(float((_nn_cos >= 0.999).mean()), 4),
    frac_nn_ge_0p99=round(float((_nn_cos >= 0.99).mean()), 4),
    NOTE="Byte-identical hashing understates duplication: two float32 rows can be "
         "numerically identical to 1e-7 and hash differently. This is the "
         "full-corpus, tolerance-based count - no sampling.")
```

---

## Cell 2 — §1 Integrity, counts, row alignment  *(editor: count consistency)*

```python
ok_shapes = (joined.shape == (N, POCKET_DIM + LIGAND_DIM))
off_in_range = bool(p_off.max() < pocket.shape[0] and l_off.max() < ligand.shape[0])

# Row alignment: joined[i] should equal L2norm(concat(pocket[p_off[i]], ligand[l_off[i]]))
chk = RNG.choice(N, size=500, replace=False)
errs = []
for i in chk:
    v = np.concatenate([pocket[p_off[i]], ligand[l_off[i]]]).astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    errs.append(np.abs(v - joined[i].astype(np.float64)).max())
errs = np.array(errs)

result("1_integrity",
    embedding_version=EMB_VERSION,
    n_metadata=N,
    joined_shape=str(joined.shape),
    pocket_shape=str(pocket.shape),
    ligand_shape=str(ligand.shape),
    shapes_consistent=ok_shapes,
    offsets_in_range=off_in_range,
    row_alignment_max_abs_err=float(errs.max()),
    row_alignment_verified=bool(errs.max() < 1e-4),
    NOTE=("v0.1: this is a real check of the SHIPPED joined matrix. "
          "v0.2: joined was rebuilt from pocket+ligand in Cell 1, so alignment "
          "holds by construction and this is a self-consistency check only."
          if V02_POCKET else
          "Checks the shipped joined matrix against concat(pocket, ligand)."),
)
```

---

## Cell 3 — §2 **D3: are the shipped splits STALE?**  *(highest-consequence unknown)*

The 2026-05-01 fix regenerated `metadata.jsonl`. Split files are **row
indices into that file**. If the row set changed after the splits were
written, the shipped indices silently address the wrong complexes.

```python
train_ix = np.load(LOCAL / "splits" / "train_indices.npy")
val_ix   = np.load(LOCAL / "splits" / "val_indices.npy")
sinfo    = json.loads((LOCAL / "splits" / "splits_info.json").read_text())

n_tr, n_va = len(train_ix), len(val_ix)
overlap    = len(set(train_ix.tolist()) & set(val_ix.tolist()))
oob        = int((train_ix >= N).sum() + (val_ix >= N).sum())
covered    = n_tr + n_va

# What the shipped code (add_splits_to_hf.py) WOULD produce for this n:
seed     = sinfo.get("seed", 42)
val_frac = sinfo.get("val_frac", 0.05)
n_valid_claimed = sinfo.get("n_valid_total", covered)
would_n_val = max(1, int(n_valid_claimed * val_frac))

# Reproduce the exact permutation on the claimed valid set
rng2 = np.random.RandomState(seed)
perm = rng2.permutation(n_valid_claimed)
val_set = set(perm[:would_n_val].tolist())
repro_val_positions = sorted(val_set)      # positions WITHIN the valid list

print("splits_info.json:", json.dumps(sinfo, indent=2))
print("\nshipped   n_train=%d  n_val=%d  (sum=%d)" % (n_tr, n_va, covered))
print("code path n_val    =%d  (int(%d * %.4f))" % (would_n_val, n_valid_claimed, val_frac))

# Staleness probe: do the split indices point at complexes that still exist
# and look like a coherent 95/5 partition of a contiguous valid subset?
all_ix = np.concatenate([train_ix, val_ix])
missing_from_split = N - len(set(all_ix.tolist()))

verdict = "CONSISTENT"
if oob:
    verdict = "STALE - indices out of range for current metadata.jsonl"
elif overlap:
    verdict = "CORRUPT - train/val overlap"
elif would_n_val != n_va:
    verdict = "MISMATCH - shipped n_val != what the documented seed/val_frac produce"

result("2_split_staleness",
    n_train_shipped=n_tr,
    n_val_shipped=n_va,
    sum_shipped=covered,
    n_val_from_documented_params=would_n_val,
    train_val_overlap=overlap,
    indices_out_of_range=oob,
    metadata_rows_not_in_any_split=missing_from_split,
    VERDICT=verdict,
)

# Second, stronger probe: if metadata.jsonl was regenerated by DROPPING rows,
# the pocket_keys at the shipped val indices would be shifted. We can't prove
# the original ordering, but we CAN check the splits partition exactly the set
# of entries that have usable HDF5 coords (which is what the script claims).
import h5py
lig_h5 = h5py.File(LOCAL / "ligands" / "ligand_coords.h5", "r")
pkt_h5 = h5py.File(LOCAL / "pockets" / "pocket_coords.h5", "r")

def usable(i):
    pk = meta[i]["pocket_key"]
    lk, kk = "ligands/" + pk, "pockets/" + pk
    if lk not in lig_h5 or kk not in pkt_h5:
        return False
    a, b = lig_h5[lk].shape, pkt_h5[kk].shape
    return len(a) == 2 and a[0] >= 3 and a[1] == 3 and len(b) == 2 and b[0] >= 3 and b[1] == 3

valid_now = [i for i in range(N) if usable(i)]
valid_set = set(valid_now)
split_set = set(all_ix.tolist())

result("2_split_staleness_recompute",
    n_usable_now=len(valid_now),
    n_in_shipped_splits=len(split_set),
    in_splits_but_not_usable=len(split_set - valid_set),
    usable_but_not_in_splits=len(valid_set - split_set),
    splits_match_usable_set=bool(split_set == valid_set),
    NOTE="If splits_match_usable_set is False, the shipped splits do NOT "
         "partition the set they claim to and must be reissued.",
)
lig_h5.close(); pkt_h5.close()
```

---

## Cell 4 — §3 **W1: what actually causes the 189 zero-vector ligands?**

The paper says (3×) they are *"transition-metal complexes, polyborane and
carborane clusters, polyoxometalates"*. Morgan returns an empty bitvector
only when `MolFromSmiles` returns `None`. Test it.

```python
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

lig_norms_all = np.linalg.norm(np.asarray(ligand, dtype=np.float32), axis=1)
zero_rows = np.flatnonzero(lig_norms_all < 1e-6)

# map ligand-matrix rows -> SMILES via the ligand manifest offsets
row2smiles, row2res = {}, {}
for r in ligand_manifest:
    row2smiles[r["ligand_offset"]] = r.get("ligand_smiles", "")
    row2res[r["ligand_offset"]] = r.get("ligand_resname", "?")

n_parse_fail = n_parsed_zero_bits = n_no_smiles = 0
examples = []
from rdkit.Chem import rdFingerprintGenerator
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

for r in zero_rows:
    smi = row2smiles.get(int(r), "")
    if not smi:
        n_no_smiles += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_parse_fail += 1
        if len(examples) < 8:
            examples.append((row2res.get(int(r), "?"), smi[:60], "PARSE FAIL"))
    else:
        bits = gen.GetFingerprint(mol).GetNumOnBits()
        if bits == 0:
            n_parsed_zero_bits += 1
            if len(examples) < 8:
                examples.append((row2res.get(int(r), "?"), smi[:60], "parsed, 0 bits"))

print("examples:")
for e in examples:
    print("   %-6s %-62s %s" % e)

result("3_zero_vector_cause",
    n_zero_ligand_rows=int(len(zero_rows)),
    n_smiles_parse_failures=n_parse_fail,
    n_parsed_but_zero_bits=n_parsed_zero_bits,
    n_missing_smiles=n_no_smiles,
    PAPER_CLAIM="no recognizable organic substructure (metal/borane/polyoxometalate)",
    VERDICT=("PAPER CLAIM WRONG - these are SMILES parse failures"
             if n_parse_fail > n_parsed_zero_bits else
             "paper claim supported - molecules parse but yield no bits"),
)
```

---

## Cell 5 — §4 **W2/W3/3.m2: what the "drug-like" filter really is**

```python
raft = [json.loads(l) for l in open(LOCAL / "raft_pdb_filtered.jsonl")]

n_multi = sum(1 for r in raft if len(r.get("ligands", [])) > 1)
lig_counts = collections.Counter(len(r.get("ligands", [])) for r in raft)

# cofactor / buffer share of the SELECTED ligand (metadata = one per entry)
COMMON_ARTIFACTS = {
    "HEM","FAD","ADP","NAP","FMN","GDP","NAD","SAH","MES","CIT","ATP","AMP",
    "NDP","COA","SAM","PLP","GTP","TRS","EPE","ACT","BTB","MPD","IPA","PO4",
    "FES","SF4","HEC","BCL","CLA","UDP","UMP","GSH","NAI","FDA","TPP","B12",
}
sel = collections.Counter(m["ligand_resname"] for m in meta)
n_artifact = sum(c for k, c in sel.items() if k in COMMON_ARTIFACTS)

# the "lower bound excludes solvent and buffer" claim
heavy_by_res = {}
for r in ligand_manifest:
    heavy_by_res.setdefault(r.get("ligand_resname", "?"), r.get("ligand_heavy_atoms"))
probe = {k: heavy_by_res.get(k) for k in ["MES", "CIT", "EPE", "TRS", "MPD", "GOL", "EDO"]}

print("top 15 selected ligands:", sel.most_common(15))
print("heavy-atom counts for buffer-like codes:", probe)

result("4_filter_reality",
    n_entries_in_raft=len(raft),
    n_entries_with_multiple_qualifying_ligands=n_multi,
    pct_entries_multi_ligand=round(100 * n_multi / max(1, len(raft)), 2),
    ligand_count_histogram=dict(sorted(lig_counts.items())[:8]),
    n_selected_is_common_cofactor_or_buffer=n_artifact,
    pct_selected_cofactor_or_buffer=round(100 * n_artifact / N, 2),
    top10_selected=sel.most_common(10),
    buffer_heavy_atom_counts=probe,
    NOTE="MES/CIT pass the 10-heavy-atom lower bound, so the claim that it "
         "'excludes solvent and buffer molecules' is false.",
)
```

---

## Cell 6 — §5 **S2: the 1,022 cap excludes MULTIMERS, not long enzymes**

`protein_length` in raft is the sum over **all** polymer chains.
`metadata.protein_seq` is the **longest single chain**.

```python
raft_len = {r["pdb_id"]: r.get("protein_length") for r in raft}
over_cap = [pid for pid, L in raft_len.items() if L and L > 1022]

# For entries that SURVIVED, compare summed length vs longest-chain length
surv = [(m["pdb_id"], raft_len.get(m["pdb_id"]), len(m.get("protein_seq") or ""))
        for m in meta if m.get("protein_seq")]
multimer_like = sum(1 for _, s, lc in surv if s and lc and s > 1.5 * lc)

result("5_cap_semantics",
    n_raft_entries=len(raft_len),
    n_excluded_by_1022_cap=len(over_cap),
    pct_excluded_by_cap=round(100 * len(over_cap) / max(1, len(raft_len)), 2),
    n_surviving_where_summed_len_gt_1p5x_longest_chain=multimer_like,
    NOTE="Entries excluded by the cap are those whose SUM over chains exceeds "
         "1022 - i.e. multimers - not necessarily long single-chain enzymes.",
)
```

---

## Cell 7 — §6 **THE CORE: embedding geometry, w_p distribution, anisotropy, α-sweep**

Answers R1.5, R2.1, R3.M3 and the editor's *"validity of the embedding"*.

```python
P = np.asarray(pocket, dtype=np.float32)[p_off]     # (N, 1280)
L = np.asarray(ligand, dtype=np.float32)[l_off]     # (N, 2048)

pn = np.linalg.norm(P, axis=1)
ln = np.linalg.norm(L, axis=1)
w_p = pn**2 / (pn**2 + ln**2 + 1e-12)               # pocket share of joint squared norm

def q(a):
    return dict(min=float(a.min()), p25=float(np.percentile(a, 25)),
                median=float(np.median(a)), p75=float(np.percentile(a, 75)),
                max=float(a.max()), mean=float(a.mean()))

on_bits = (L > 0).sum(axis=1)

result("6a_norms",
    pocket_norm=q(pn),
    ligand_norm=q(ln),
    morgan_on_bits=q(on_bits.astype(np.float64)),
    w_p_pocket_share_of_squared_norm=q(w_p),
    w_p_varies_per_entry=bool(w_p.std() > 0.01),
    NOTE="w_p is NOT constant - pairs are compared under different implicit "
         "weightings depending on pocket size and ligand size. The shipped "
         "index therefore does not implement a consistent similarity function.",
)

# ---- anisotropy: is the 0.44 random-pair baseline a floor artifact? -------
S = 20000
ia, ib = RNG.choice(N, S), RNG.choice(N, S)
def cos_rows(A, B):
    num = (A * B).sum(1)
    return num / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)

cos_joint_rand = cos_rows(np.asarray(joined[ia]), np.asarray(joined[ib]))
cos_p_rand = cos_rows(P[ia], P[ib])
cos_l_rand = cos_rows(L[ia], L[ib])

Pc = P - P.mean(0, keepdims=True)          # mean-centred pocket space
cos_p_rand_centred = cos_rows(Pc[ia], Pc[ib])

mu = P.mean(0)
aniso = float(np.linalg.norm(mu) / pn.mean())

result("6b_anisotropy",
    random_pair_cos_joint_median=float(np.median(cos_joint_rand)),
    random_pair_cos_pocket_median=float(np.median(cos_p_rand)),
    random_pair_cos_ligand_median=float(np.median(cos_l_rand)),
    random_pair_cos_pocket_MEAN_CENTRED_median=float(np.median(cos_p_rand_centred)),
    anisotropy_meanvec_norm_over_mean_rownorm=aniso,
    PREDICTION_CHECK="paper reports joint random median 0.440; predicted "
                     "E[cos_p_rand] ~ 0.47 if w_p ~ 0.93",
    NOTE="If the centred median collapses toward 0, the high absolute cosines "
         "are an anisotropy offset and the ORDERING carries the signal.",
)

# ---- alpha sweep ---------------------------------------------------------
# u = [sqrt(a)*p_hat ; sqrt(1-a)*l_hat]  =>  cos(u1,u2) = a*cos_p + (1-a)*cos_l exactly
import faiss
Ph = P / (pn[:, None] + 1e-12)
Lh = L / (ln[:, None] + 1e-12)

QS = 3000
qidx = RNG.choice(N, QS, replace=False)
sweep = {}
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    Ua = np.ascontiguousarray(
        np.hstack([math.sqrt(a) * Ph, math.sqrt(1 - a) * Lh]).astype(np.float32))
    idx = faiss.IndexFlatIP(Ua.shape[1]); idx.add(Ua)
    D, I = idx.search(Ua[qidx], 2)                 # top-1 excluding self
    nn = np.where(I[:, 0] == qidx, I[:, 1], I[:, 0])
    same_lig = np.mean([meta[int(a_)]["ligand_resname"] == meta[int(b_)]["ligand_resname"]
                        for a_, b_ in zip(qidx, nn)])
    sweep["alpha=%.2f" % a] = dict(
        top1_same_ligand_ccd=round(float(same_lig), 4),
        median_top1_cos=round(float(np.median(np.where(I[:, 0] == qidx, D[:, 1], D[:, 0]))), 4),
    )
    del Ua, idx

result("6c_alpha_sweep", **sweep,
    NOTE="alpha=1 is pocket-only, alpha=0 is ligand-only. Same-UniProt hit@1 "
         "per alpha is added in Cell 13 once SIFTS labels are loaded - that is "
         "the number that DECIDES the default index.")
```

---

## Cell 8 — §7 **V3/3.m1: duplicates and split leakage** *(the killer number)*

```python
J = np.ascontiguousarray(np.asarray(joined, dtype=np.float32))

# exact duplicate groups
h = [hashlib.md5(J[i].tobytes()).hexdigest() for i in range(N)]
groups = collections.Counter(h)
dup_groups = {k: v for k, v in groups.items() if v > 1}
n_in_dupes = sum(dup_groups.values())

tr, va = set(train_ix.tolist()), set(val_ix.tolist())
by_hash = collections.defaultdict(list)
for i, hh in enumerate(h):
    by_hash[hh].append(i)
crossing = sum(1 for hh, idxs in by_hash.items()
               if len(idxs) > 1 and any(i in tr for i in idxs) and any(i in va for i in idxs))

# near-duplicate curve: top-1 cosine for a large sample
idx = faiss.IndexFlatIP(J.shape[1]); idx.add(J)
qs = RNG.choice(N, 20000, replace=False)
D, I = idx.search(J[qs], 2)
top1 = np.where(I[:, 0] == qs, D[:, 1], D[:, 0])

near = {("frac_top1_ge_%.2f" % t): round(float((top1 >= t).mean()), 4)
        for t in [0.90, 0.95, 0.99, 0.999]}

# fraction of VAL entries with a TRAIN neighbour above threshold
Jtr = np.ascontiguousarray(J[train_ix]); itr = faiss.IndexFlatIP(J.shape[1]); itr.add(Jtr)
Dv, _ = itr.search(np.ascontiguousarray(J[val_ix]), 1)
val_leak = {("frac_val_with_train_nbr_ge_%.2f" % t): round(float((Dv[:, 0] >= t).mean()), 4)
            for t in [0.90, 0.95, 0.99, 0.999]}

result("7_duplicates_and_leakage",
    n_exact_duplicate_groups=len(dup_groups),
    n_entries_in_duplicate_groups=n_in_dupes,
    pct_entries_in_duplicate_groups=round(100 * n_in_dupes / N, 2),
    n_duplicate_groups_crossing_train_val=crossing,
    **near, **val_leak,
    NOTE="frac_val_with_train_nbr_ge_0.99 is the number that determines whether "
         "the canonical random split is a generalization benchmark at all.",
)
```

---

## Cell 9 — §8 **V4: does the SHIPPED HNSW index reproduce the paper's exact-search numbers?**

Figure 3 and the aggregate evaluation used `IndexFlatIP`. The released
artifact is HNSW. Report recall.

```python
if V02_POCKET:
    # The shipped retrieval.faiss was built over the v0.1 concatenation. Comparing
    # it against a v0.2 exact search measures the embedding change, NOT ANN recall,
    # and would be actively misleading. Skip until build_v02_release.py rebuilds
    # the index; the v0.1 recall figures below remain the correct ones to quote
    # for the v0.1 artifact.
    result("8_ann_recall",
        SKIPPED="shipped retrieval.faiss is a v0.1 artifact; rerun after the v0.2 "
                "indices are built",
        v01_recall_at_1_efSearch16=0.9775, v01_recall_at_1_efSearch256=0.987,
        v01_recall_at_10_efSearch16=0.9737, v01_recall_at_10_efSearch256=0.9905,
        NOTE="Values carried from the v0.1 measurement for reference only.")
    print("\n>> Cell 9 SKIPPED under v0.2 (not an error). "
          "ANN recall must be re-measured after build_v02_release.py "
          "rebuilds the index -- it is still owed for R-V4.")
else:
    hnsw = faiss.read_index(str(LOCAL / "retrieval.faiss"))
    qs2 = RNG.choice(N, 2000, replace=False)
    Q = np.ascontiguousarray(J[qs2])

    _, I_exact = idx.search(Q, 11)          # exact, from Cell 8's flat index
    rec = {}
    for ef in [16, 64, 128, 256]:
        try:
            hnsw.hnsw.efSearch = ef
        except Exception:
            pass
        _, I_ann = hnsw.search(Q, 11)
        r1  = np.mean([I_ann[i, 0] == I_exact[i, 0] for i in range(len(qs2))])
        r10 = np.mean([len(set(I_ann[i, :10]) & set(I_exact[i, :10])) / 10.0
                       for i in range(len(qs2))])
        rec["efSearch=%d" % ef] = dict(recall_at_1=round(float(r1), 4),
                                       recall_at_10=round(float(r10), 4))

    result("8_ann_recall", **rec,
        NOTE="Paper's Fig 3 + aggregate retrieval used exact IndexFlatIP. These are "
             "the numbers a user of the RELEASED HNSW index actually gets.")
```

---

## Cell 10 — §9 **D4: ligand-vector redundancy (per-CCD, not per-instance)**

```python
lh = [hashlib.md5(L[i].tobytes()).hexdigest() for i in range(N)]
distinct_lig = len(set(lh))
distinct_ccd = len({m["ligand_resname"] for m in meta})

# does a CCD code ever map to two different vectors? (it should not)
ccd2vec = collections.defaultdict(set)
for i, m in enumerate(meta):
    ccd2vec[m["ligand_resname"]].add(lh[i])
inconsistent = sum(1 for k, v in ccd2vec.items() if len(v) > 1)

result("9_ligand_redundancy",
    n_rows=N,
    n_distinct_ligand_vectors=distinct_lig,
    n_distinct_ccd_codes=distinct_ccd,
    n_ccd_codes_with_more_than_one_vector=inconsistent,
    redundancy_ratio=round(N / max(1, distinct_lig), 2),
    matrix_size_MB=round(ligand.nbytes / 1e6, 1),
    ccd_table_size_MB=round(distinct_lig * LIGAND_DIM * 4 / 1e6, 1),
    NOTE="If n_distinct == n_ccd_codes, the fingerprint is a pure function of "
         "the 3-letter code - i.e. per-CCD, not per-instance.",
)
```

---

## Cell 11 — §10 **R2.2: when pockets are similar, are the ligands?**

Uses MACCS (non-ECFP) to avoid circularity with the indexed Morgan vector.

```python
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

smi_by_i = [m.get("ligand_smiles", "") for m in meta]
sample = RNG.choice(N, 4000, replace=False)

maccs = {}
def get_maccs(i):
    if i not in maccs:
        mol = Chem.MolFromSmiles(smi_by_i[i]) if smi_by_i[i] else None
        maccs[i] = MACCSkeys.GenMACCSKeys(mol) if mol is not None else None
    return maccs[i]

# top-1 neighbours for the sample (joint index)
Ds, Is = idx.search(np.ascontiguousarray(J[sample]), 2)
nbr = np.where(Is[:, 0] == sample, Is[:, 1], Is[:, 0])

def tan(i, j):
    a, b = get_maccs(int(i)), get_maccs(int(j))
    return DataStructs.TanimotoSimilarity(a, b) if (a and b) else None

t_top1 = [x for x in (tan(i, j) for i, j in zip(sample, nbr)) if x is not None]
rnd = RNG.choice(N, len(sample))
t_rand = [x for x in (tan(i, j) for i, j in zip(sample, rnd)) if x is not None]
same_ccd = np.mean([meta[int(i)]["ligand_resname"] == meta[int(j)]["ligand_resname"]
                    for i, j in zip(sample, nbr)])

# conditional: bin by POCKET-only cosine, report median ligand Tanimoto
cp = cos_rows(P[sample], P[nbr])
bins = [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
cond = {}
for lo, hi in bins:
    sel_b = [(i, j) for i, j, c in zip(sample, nbr, cp) if lo <= c < hi]
    vals = [x for x in (tan(i, j) for i, j in sel_b[:600]) if x is not None]
    cond["pocket_cos_%.1f_%.1f" % (lo, hi)] = dict(
        n=len(sel_b), median_ligand_tanimoto=round(float(np.median(vals)), 4) if vals else None)

result("10_ligand_similarity",
    median_maccs_tanimoto_top1=round(float(np.median(t_top1)), 4),
    median_maccs_tanimoto_random=round(float(np.median(t_rand)), 4),
    frac_top1_shares_query_ccd_code=round(float(same_ccd), 4),
    conditional_by_pocket_cosine=cond,
    NOTE="frac_top1_shares_query_ccd_code is the honest headline: how much of "
         "the 'ligand signal' is exact ligand-identity matching.",
)
```

---

## Cell 12 — §11 **S5: were the entries really all X-ray?**

`rcsb_filter.py` never checks experimental method, and the download query
has a silent tier fallback. Verify against the RCSB Data API.

```python
import requests

pdb_ids = sorted({m["pdb_id"].upper() for m in meta})
probe = list(RNG.choice(np.array(pdb_ids), size=min(600, len(pdb_ids)), replace=False))

methods = collections.Counter()
fails = 0
for pid in probe:
    try:
        r = requests.get("https://data.rcsb.org/rest/v1/core/entry/%s" % pid, timeout=15)
        if r.status_code != 200:
            fails += 1; continue
        for e in r.json().get("exptl", []):
            methods[e.get("method", "?")] += 1
    except Exception:
        fails += 1

tot = sum(methods.values())
result("11_experimental_method",
    n_probed=len(probe),
    n_api_failures=fails,
    method_counts=dict(methods),
    pct_xray=round(100 * methods.get("X-RAY DIFFRACTION", 0) / max(1, tot), 2),
    NOTE="If pct_xray < 100, the Methods claim 'experimental method equal to "
         "X-ray diffraction' is unsupported and must be corrected.",
)
```

---

## Cell 13 — §12 **R1.1/R1.4: SIFTS families, stratified retrieval, discriminativeness**

```python
!wget -q -nc https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz
!wget -q -nc https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_pfam.csv.gz
!wget -q -nc https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_enzyme.csv.gz

import pandas as pd
def load_sifts(fn, col):
    df = pd.read_csv(fn, compression="gzip", skiprows=1)
    df["PDB"] = df["PDB"].str.lower()
    return df.groupby("PDB")[col].first().to_dict()

# SIFTS is re-released periodically and family assignments DO shift between
# releases; a rerun weeks later gives different per-family numbers. Record what
# this run actually used so Table 4 can be pinned to it in Methods.
import datetime, email.utils
_sifts_prov = {}
for _f in ["pdb_chain_uniprot.csv.gz", "pdb_chain_pfam.csv.gz", "pdb_chain_enzyme.csv.gz"]:
    _st = Path(_f)
    _entry = {"sha1_12": None, "downloaded_utc": None, "server_last_modified": None}
    if _st.exists():
        _entry["sha1_12"] = hashlib.sha1(_st.read_bytes()).hexdigest()[:12]
        _entry["downloaded_utc"] = datetime.datetime.utcfromtimestamp(
            _st.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    try:
        import requests as _rq
        _h = _rq.head("https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/" + _f,
                      timeout=20)
        _entry["server_last_modified"] = _h.headers.get("Last-Modified")
    except Exception:
        pass
    _sifts_prov[_f] = _entry
result("12_sifts_provenance", **_sifts_prov,
    NOTE="Quote server_last_modified (or downloaded_utc) as the SIFTS release "
         "in Methods. Per-family numbers are only reproducible against it.")

uni  = load_sifts("pdb_chain_uniprot.csv.gz", "SP_PRIMARY")
pfam = load_sifts("pdb_chain_pfam.csv.gz", "PFAM_ID")
ec   = load_sifts("pdb_chain_enzyme.csv.gz", "EC_NUMBER")

u_lab = np.array([uni.get(m["pdb_id"].lower(), "") for m in meta])
p_lab = np.array([pfam.get(m["pdb_id"].lower(), "") for m in meta])
e_lab = np.array([str(ec.get(m["pdb_id"].lower(), "")).split(".")[0] for m in meta])

def hit_at_k(labels, qsample, k=10, index=idx, mat=J):
    D, I = index.search(np.ascontiguousarray(mat[qsample]), k + 1)
    out = []
    for row, qi in zip(I, qsample):
        nb = [j for j in row if j != qi][:k]
        lq = labels[qi]
        if not lq:
            out.append(None); continue
        out.append([labels[j] == lq for j in nb])
    return out

qs3 = RNG.choice(N, 3000, replace=False)
res_u = hit_at_k(u_lab, qs3)
def rate(res, k):
    v = [any(r[:k]) for r in res if r is not None]
    return round(float(np.mean(v)), 4) if v else None

result("12a_retrieval_overall",
    n_queries=len(qs3),
    resolvable_uniprot=int(sum(1 for r in res_u if r is not None)),
    hit_at_1=rate(res_u, 1), hit_at_5=rate(res_u, 5), hit_at_10=rate(res_u, 10),
)

# STRATIFIED BY FAMILY (rows = family) - this is what R1.1 actually asked for
fam_counts = collections.Counter(p_lab[p_lab != ""])
top_fams = [f for f, _ in fam_counts.most_common(12)]
strat = {}
for f in top_fams:
    members = np.flatnonzero(p_lab == f)
    if len(members) < 20:
        continue
    qq = members if len(members) <= 300 else RNG.choice(members, 300, replace=False)
    r = hit_at_k(u_lab, qq)
    baseline = (len(members) - 1) / (N - 1)
    strat[f] = dict(n_in_family=int(len(members)), n_queried=int(len(qq)),
                    hit_at_1=rate(r, 1), hit_at_10=rate(r, 10),
                    size_adjusted_random_baseline=round(float(baseline), 6))
result("12b_stratified_by_pfam", **strat,
    NOTE="Report the WORST families too - honest failure reporting is what the "
         "editor means by 'thoroughness'.")

# DISCRIMINATIVENESS: kNN purity + AUROC, raw vs mean-centred
def knn_purity(labels, mat, k=10, S=3000):
    ii = faiss.IndexFlatIP(mat.shape[1]); ii.add(np.ascontiguousarray(mat))
    qq = RNG.choice(N, S, replace=False)
    _, I = ii.search(np.ascontiguousarray(mat[qq]), k + 1)
    vals = []
    for row, qi in zip(I, qq):
        nb = [j for j in row if j != qi][:k]
        if not labels[qi]:
            continue
        vals.append(np.mean([labels[j] == labels[qi] for j in nb]))
    return round(float(np.mean(vals)), 4)

Jc = J - J.mean(0, keepdims=True)
Jc = Jc / (np.linalg.norm(Jc, axis=1, keepdims=True) + 1e-12)

result("12c_discriminativeness",
    knn10_purity_uniprot_raw=knn_purity(u_lab, J),
    knn10_purity_uniprot_centred=knn_purity(u_lab, Jc.astype(np.float32)),
    knn10_purity_pfam_raw=knn_purity(p_lab, J),
    knn10_purity_pfam_centred=knn_purity(p_lab, Jc.astype(np.float32)),
    NOTE="If centred >> raw, the raw space is dominated by a common component "
         "and the paper's separation claim rests on an offset.")

# alpha sweep, now with the metric that DECIDES the default
dec = {}
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    Ua = np.ascontiguousarray(
        np.hstack([math.sqrt(a) * Ph, math.sqrt(1 - a) * Lh]).astype(np.float32))
    ii = faiss.IndexFlatIP(Ua.shape[1]); ii.add(Ua)
    D2, I2 = ii.search(Ua[qs3], 2)
    nb2 = np.where(I2[:, 0] == qs3, I2[:, 1], I2[:, 0])
    m = [u_lab[j] == u_lab[i] for i, j in zip(qs3, nb2) if u_lab[i]]
    dec["alpha=%.2f" % a] = round(float(np.mean(m)), 4)
    del Ua, ii
result("12d_alpha_decides_default", same_uniprot_hit_at_1_by_alpha=dec,
    RECOMMENDED_DEFAULT=max(dec, key=dec.get),
    NOTE="Adopt the winning alpha as the v0.2 default index. Ship ONE index, "
         "not a menu.")
```

---

## Cell 13b — **Statistical rigor: CIs, nulls, near-duplicate correction, paired tests**

The headline rates above are point estimates over a pool that contains
near-duplicates. This cell supplies what a referee will otherwise ask for:
bootstrap confidence intervals, a permutation null for k-NN purity, retrieval
recomputed with near-duplicates excluded, and seed stability.

```python
def boot_ci(flags, n_boot=2000, seed=42):
    """Bootstrap 95% CI for a proportion."""
    a = np.asarray(flags, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    bs = a[r.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return (round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4))

# ---- 1. CIs on the overall retrieval rates -------------------------------
_res_u = hit_at_k(u_lab, qs3, k=10)
def flags_at(res, k):
    return [any(r[:k]) for r in res if r is not None]
f1, f5, f10 = flags_at(_res_u, 1), flags_at(_res_u, 5), flags_at(_res_u, 10)
result("13b_retrieval_with_CI",
    n=len(f1),
    hit_at_1=round(float(np.mean(f1)), 4),  hit_at_1_CI95=boot_ci(f1),
    hit_at_5=round(float(np.mean(f5)), 4),  hit_at_5_CI95=boot_ci(f5),
    hit_at_10=round(float(np.mean(f10)), 4), hit_at_10_CI95=boot_ci(f10))

# ---- 2. NEAR-DUPLICATE-EXCLUDED retrieval (the honest number) -------------
# 46% of entries have a >=0.99 neighbour. Counting those as retrieval successes
# conflates "found the same target" with "found a re-deposit of the same
# structure". Recompute excluding (a) same PDB entry and (b) neighbours above a
# cosine threshold, so the reported rate reflects generalisation.
def hit_excluding_neardup(labels, qsample, thresh, k=10, pool=None):
    mat = pool if pool is not None else J
    ii = faiss.IndexFlatIP(mat.shape[1]); ii.add(np.ascontiguousarray(mat))
    D, I = ii.search(np.ascontiguousarray(mat[qsample]), 60)
    out = []
    for row_d, row_i, qi in zip(D, I, qsample):
        lq = labels[qi]
        if not lq:
            out.append(None); continue
        keep = []
        for d, j in zip(row_d, row_i):
            if j == qi:                       continue
            if meta[j]["pdb_id"] == meta[qi]["pdb_id"]: continue   # same entry
            if d >= thresh:                   continue             # near-duplicate
            keep.append(labels[j] == lq)
            if len(keep) >= k:                break
        out.append(keep if keep else None)
    return out

nd = {}
for thr in [1.01, 0.999, 0.99, 0.95]:
    r = hit_excluding_neardup(u_lab, qs3, thr)
    f = [any(x[:1]) for x in r if x]
    nd["excl_ge_%.3f" % thr] = dict(n=len(f),
        hit_at_1=round(float(np.mean(f)), 4), CI95=boot_ci(f))
result("13c_retrieval_neardup_excluded", **nd,
    NOTE="excl_ge_1.010 keeps everything except self/same-PDB (the standard "
         "number). Lower thresholds progressively remove near-identical "
         "neighbours; the drop is the share of apparent performance attributable "
         "to re-deposited structures rather than to generalisation.")

# ---- 3. k-NN purity against a LABEL-PERMUTATION null ----------------------
def knn_purity_with_null(labels, mat, k=10, S=3000, n_perm=5):
    ii = faiss.IndexFlatIP(mat.shape[1]); ii.add(np.ascontiguousarray(mat))
    qq = RNG.choice(N, S, replace=False)
    _, I = ii.search(np.ascontiguousarray(mat[qq]), k + 1)
    obs = []
    for row, qi in zip(I, qq):
        nb = [j for j in row if j != qi][:k]
        if labels[qi]:
            obs.append(np.mean([labels[j] == labels[qi] for j in nb]))
    nulls = []
    for s in range(n_perm):
        perm = np.random.default_rng(100 + s).permutation(len(labels))
        lab = labels[perm]
        v = []
        for row, qi in zip(I, qq):
            nb = [j for j in row if j != qi][:k]
            if lab[qi]:
                v.append(np.mean([lab[j] == lab[qi] for j in nb]))
        nulls.append(float(np.mean(v)))
    return (round(float(np.mean(obs)), 4), boot_ci(obs),
            round(float(np.mean(nulls)), 5), round(float(np.mean(obs))/max(1e-9, np.mean(nulls)), 1))

for name, lab in [("uniprot", u_lab), ("pfam", p_lab)]:
    o, ci, nul, ratio = knn_purity_with_null(lab, J)
    result("13d_knn_purity_vs_null_" + name,
        observed=o, CI95=ci, permutation_null=nul, enrichment_over_null=ratio,
        NOTE="Purity alone is uninterpretable; the null is the chance rate given "
             "this label distribution.")

# ---- 4. seed stability of the alpha decision -----------------------------
seed_tab = {}
for sd in [42, 7, 2024]:
    rr = np.random.default_rng(sd)
    qq = rr.choice(N, 2000, replace=False)
    row = {}
    for a in [0.0, 0.5, 1.0]:
        Ua = np.ascontiguousarray(
            np.hstack([math.sqrt(a) * Ph, math.sqrt(1 - a) * Lh]).astype(np.float32))
        ii = faiss.IndexFlatIP(Ua.shape[1]); ii.add(Ua)
        D2, I2 = ii.search(Ua[qq], 2)
        nb2 = np.where(I2[:, 0] == qq, I2[:, 1], I2[:, 0])
        m = [u_lab[j] == u_lab[i] for i, j in zip(qq, nb2) if u_lab[i]]
        row["alpha=%.2f" % a] = round(float(np.mean(m)), 4)
        del Ua, ii
    seed_tab["seed_%d" % sd] = row
result("13e_alpha_seed_stability", **seed_tab,
    NOTE="If the alpha ordering is identical across seeds, the default-index "
         "choice is not a sampling artifact.")

# ---- 5. save the per-query flags so v0.1 vs v0.2 can be PAIRED later ------
np.savez_compressed("retrieval_flags_%s.npz" % ("v02" if V02_POCKET else "v01"),
                    queries=np.asarray(qs3),
                    hit1=np.asarray([bool(any(r[:1])) if r else False for r in _res_u]),
                    resolvable=np.asarray([r is not None for r in _res_u]))
print("saved per-query flags for a paired v0.1-vs-v0.2 McNemar test")
```

---

## Cell 13f — **Paired significance, held-out α, per-family CIs**

Point estimates are not evidence of a difference. This supplies the tests a
referee would otherwise demand.

```python
import math as _m

def mcnemar(a_flags, b_flags):
    """Paired test on identical queries -> (b01, b10, chi2, p)."""
    a = np.asarray(a_flags, dtype=bool); b = np.asarray(b_flags, dtype=bool)
    b01 = int(np.sum(~a & b))          # b right, a wrong
    b10 = int(np.sum(a & ~b))          # a right, b wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1.0) ** 2 / n        # continuity-corrected
    z = _m.sqrt(chi2)
    p = 2.0 * (1.0 - 0.5 * (1.0 + _m.erf(z / _m.sqrt(2.0))))
    return b01, b10, round(float(chi2), 3), float("%.3g" % p)

def alpha_flags(a, queries):
    Ua = np.ascontiguousarray(
        np.hstack([_m.sqrt(a) * Ph, _m.sqrt(1 - a) * Lh]).astype(np.float32))
    ii = faiss.IndexFlatIP(Ua.shape[1]); ii.add(Ua)
    _, I2 = ii.search(Ua[queries], 2)
    nb = np.where(I2[:, 0] == queries, I2[:, 1], I2[:, 0])
    del Ua, ii
    hits = np.array([bool(u_lab[i]) and u_lab[j] == u_lab[i]
                     for i, j in zip(queries, nb)])
    ok = np.array([bool(u_lab[i]) for i in queries])
    return hits, ok

# ---- 1. pocket-only vs concatenation, PAIRED on identical queries --------
f_pocket, ok_mask = alpha_flags(1.0, qs3)
f_half, _ = alpha_flags(0.5, qs3)
b01, b10, chi2, pv = mcnemar(f_half[ok_mask], f_pocket[ok_mask])
result("13f_paired_pocket_vs_concat",
    n_paired_queries=int(ok_mask.sum()),
    hit_at_1_alpha0p5=round(float(f_half[ok_mask].mean()), 4),
    hit_at_1_pocket_only=round(float(f_pocket[ok_mask].mean()), 4),
    discordant_pocket_wins=b01, discordant_concat_wins=b10,
    mcnemar_chi2=chi2, p_value=pv,
    VERDICT=("pocket-only significantly better (p < 0.05)" if pv < 0.05
             else "NOT significant - do not claim a difference"),
    NOTE="Paired on identical queries; far stronger than comparing two "
         "independent rates.")

# ---- 2. held-out alpha selection ----------------------------------------
_perm = RNG.permutation(len(qs3))
_A = qs3[_perm[: len(qs3) // 2]]
_B = qs3[_perm[len(qs3) // 2:]]
_scores = {}
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    fa, oka = alpha_flags(a, _A)
    _scores[a] = float(fa[oka].mean())
_best = max(_scores, key=_scores.get)
fb, okb = alpha_flags(_best, _B)
result("13g_alpha_heldout",
    selection_half_scores={("alpha=%.2f" % k): round(v, 4) for k, v in _scores.items()},
    alpha_selected_on_half_A=_best,
    hit_at_1_on_UNSEEN_half_B=round(float(fb[okb].mean()), 4),
    CI95_on_half_B=boot_ci(fb[okb]),
    NOTE="alpha is chosen on half A and scored on half B, so the reported figure "
         "is not inflated by selecting and evaluating on the same queries.")

# ---- 3. per-family CIs + multiple-comparison caveat ---------------------
_fam_ci = {}
for _f in top_fams:
    _members = np.flatnonzero(p_lab == _f)
    if len(_members) < 20:
        continue
    _qq = _members if len(_members) <= 300 else RNG.choice(_members, 300, replace=False)
    _r = hit_at_k(u_lab, _qq)
    _fl = [any(x[:1]) for x in _r if x is not None]
    _fam_ci[_f] = dict(n_queried=len(_fl), hit_at_1=round(float(np.mean(_fl)), 4),
                       CI95=boot_ci(_fl))
result("13h_per_family_CI", **_fam_ci,
    n_families_compared=len(_fam_ci),
    NOTE="At ~300 queries a family CI is roughly +/-5 pp, so ADJACENT families "
         "are NOT distinguishable and should not be ranked as if they were. Only "
         "the extremes support a claim, and multiple families were compared - "
         "state that explicitly.")
```

---

## Cell 13i — **Temporal leakage and a time-split evaluation**

The manuscript's own smoke test retrieved six sequentially-deposited entries
(9qp3, 9qp6, 9qoc…) for query 9qp5 — a Protein Data Bank release cluster. If a
large share of retrieval successes are same-release-window neighbours, the
reported rates measure re-deposition rather than generalisation. This is the
axis PLINDER-style benchmarks are built around, and it is the strongest
remaining test of whether this index supports benchmark construction.

Requires the deposition dates cached by Cell 1h.

```python
import datetime as _dt

_rel = {}
for _m in meta:
    _e = _entry_meta.get(_m["pdb_id"].upper())
    _rel[_m["pdb_id"].upper()] = (_e or {}).get("released")

def _ymd(s):
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None

_dates = np.array([_ymd(_rel.get(m["pdb_id"].upper())) for m in meta], dtype=object)
_have_date = np.array([d is not None for d in _dates])
print("entries with a release date: %d / %d" % (int(_have_date.sum()), N))

# ---- 1. how many top-1 hits are same-release-window neighbours? ----------
_ii = faiss.IndexFlatIP(J.shape[1]); _ii.add(np.ascontiguousarray(J))
_qs = RNG.choice(np.flatnonzero(_have_date), min(3000, int(_have_date.sum())),
                 replace=False)
_D, _I = _ii.search(np.ascontiguousarray(J[_qs]), 2)
_nb = np.where(_I[:, 0] == _qs, _I[:, 1], _I[:, 0])

_gaps, _same_week, _same_month = [], 0, 0
for _q, _n in zip(_qs, _nb):
    _a, _b = _dates[_q], _dates[_n]
    if _a is None or _b is None:
        continue
    _g = abs((_a - _b).days)
    _gaps.append(_g)
    _same_week += int(_g <= 7)
    _same_month += int(_g <= 31)
_gaps = np.array(_gaps)
result("13i_release_cluster_leakage",
    n_pairs=len(_gaps),
    median_days_between_query_and_top1=int(np.median(_gaps)),
    frac_top1_within_7_days=round(_same_week / max(1, len(_gaps)), 4),
    frac_top1_within_31_days=round(_same_month / max(1, len(_gaps)), 4),
    frac_top1_within_365_days=round(float((_gaps <= 365).mean()), 4),
    NOTE="A high same-week fraction means the nearest neighbour is typically a "
         "co-deposited structure from the same study, not an independent "
         "observation of the same target.")

# ---- 2. TIME-SPLIT retrieval: query recent, retrieve only from the past ---
_valid = np.flatnonzero(_have_date)
_sorted = sorted(_valid, key=lambda i: _dates[i])
_cut = _sorted[int(0.8 * len(_sorted))]
_cutoff = _dates[_cut]
_past = np.array([i for i in _valid if _dates[i] <= _cutoff])
_future = np.array([i for i in _valid if _dates[i] > _cutoff])
print("time split at %s: %d past / %d future" % (_cutoff, len(_past), len(_future)))

_Jp = np.ascontiguousarray(J[_past])
_ip = faiss.IndexFlatIP(J.shape[1]); _ip.add(_Jp)
_qf = _future if len(_future) <= 3000 else RNG.choice(_future, 3000, replace=False)
_Df, _If = _ip.search(np.ascontiguousarray(J[_qf]), 10)

_ts_hits, _ts_hits_nodup = [], []
for _row_d, _row_i, _q in zip(_Df, _If, _qf):
    if not u_lab[_q]:
        continue
    _lab = [u_lab[_past[j]] == u_lab[_q] for j in _row_i]
    _ts_hits.append(bool(_lab[0]))
    _keep = [l for d, l in zip(_row_d, _lab) if d < 0.99]
    _ts_hits_nodup.append(bool(_keep[0]) if _keep else False)

result("13j_time_split_retrieval",
    cutoff_date=str(_cutoff),
    n_past_pool=len(_past), n_future_queries=len(_ts_hits),
    hit_at_1_time_split=round(float(np.mean(_ts_hits)), 4),
    CI95=boot_ci(_ts_hits),
    hit_at_1_time_split_neardup_excluded=round(float(np.mean(_ts_hits_nodup)), 4),
    CI95_neardup_excluded=boot_ci(_ts_hits_nodup),
    NOTE="Queries are the most recent 20% of entries; the retrieval pool is only "
         "structures released BEFORE the cutoff. This is the closest thing to a "
         "prospective evaluation the dataset permits, and it is the number to "
         "quote for benchmark-construction suitability - not the random-split "
         "figure.")
```

---

## Cell 13k — **Time-split retrieval for the CHOSEN default (pocket-only)**

Cell 13j measured the *concatenated* matrix `J`. Cell 12d selects
**alpha = 1.00 (pocket-only)**, so the prospective number must be
recomputed in the representation actually shipped, or the headline
honest figure describes an index that is being retired.

```python
if "P" not in dir():
    P = np.asarray(pocket, dtype=np.float32)[p_off]
Pn = np.ascontiguousarray(P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12))

_ipp = faiss.IndexFlatIP(Pn.shape[1])
_ipp.add(np.ascontiguousarray(Pn[_past]))
_Dp, _Ip = _ipp.search(np.ascontiguousarray(Pn[_qf]), 10)

_p_hits, _p_hits_nodup = [], []
for _row_d, _row_i, _q in zip(_Dp, _Ip, _qf):
    if not u_lab[_q]:
        continue
    _lab = [u_lab[_past[j]] == u_lab[_q] for j in _row_i]
    _p_hits.append(bool(_lab[0]))
    _keep = [l for d, l in zip(_row_d, _lab) if d < 0.99]
    _p_hits_nodup.append(bool(_keep[0]) if _keep else False)

result("13k_time_split_pocket_only",
    cutoff_date=str(_cutoff),
    n_future_queries=len(_p_hits),
    hit_at_1_time_split=round(float(np.mean(_p_hits)), 4),
    CI95=boot_ci(np.array(_p_hits, dtype=float)),
    hit_at_1_time_split_neardup_excluded=round(float(np.mean(_p_hits_nodup)), 4),
    CI95_neardup_excluded=boot_ci(np.array(_p_hits_nodup, dtype=float)),
    COMPARE_concat_13j="joined J gave 0.414 raw / 0.2769 near-dup-excluded",
    CAVEAT="The 0.99 near-duplicate cut is NOT comparable across "
           "representations: random-pair cosine is 0.72 in pocket space "
           "vs 0.42 in joined space, so the same threshold removes a "
           "different population. Quote the RAW time-split figure when "
           "comparing alpha.",
    NOTE="Prospective number for the index actually shipped (alpha=1.00, "
         "pocket-only). This is the figure to quote for "
         "benchmark-construction suitability.")
```

---
## Cell 14 — §13 **R2.2b: is the 345 MB BindingDB table even joinable?**

```python
import pandas as pd
bdb_path = LOCAL / "bindingdb_staff.tsv"
join_pct = None
try:
    # Pass 1: header only. Finds the PDB-bearing columns without loading data.
    _head = pd.read_csv(bdb_path, sep="	", nrows=0)
    pdb_cols = [c for c in _head.columns if "PDB" in c.upper()]

    # Pass 2: stream ONLY those columns, as str, in chunks.
    # Reading the whole 345 MB table with low_memory=False materialises
    # several GB of object columns and kills the kernel when the embedding
    # matrices are already resident. This keeps it to a few MB.
    ids, n_rows = set(), 0
    _use = pdb_cols if pdb_cols else None
    for _chunk in pd.read_csv(bdb_path, sep="	", usecols=_use, dtype=str,
                              chunksize=200000, on_bad_lines="skip"):
        n_rows += len(_chunk)
        for c in (pdb_cols or []):
            ids |= set(_chunk[c].dropna().astype(str).str.lower().str.strip())
    ids.discard("nan"); ids.discard("")

    ours = {m["pdb_id"].lower() for m in meta}
    inter = ours & ids
    join_pct = round(100 * len(inter) / len(ours), 2)
    result("13_bindingdb_joinability",
        n_bindingdb_rows=n_rows,
        pdb_bearing_columns=pdb_cols[:6],
        n_our_pdb_ids=len(ours),
        n_joinable=len(inter),
        pct_of_index_joinable=join_pct,
        file_size_MB=round(bdb_path.stat().st_size / 1e6, 1),
        NOTE="This table is 9% of the release by size and is used in no "
             "analysis. Report the joinable fraction, or justify shipping it.")
except Exception as e:
    print("BindingDB parse failed:", type(e).__name__, e)
```

---

## Cell 15 — Write the consolidated results **and report completeness**

```python
EXPECTED_SECTIONS = [
    '1b_artifact_fingerprints',
    '1c_offset_integrity',
    '1d_embedding_health',
    '1e_hdf5_vs_metadata',
    '1f_smiles_validity_full_corpus',
    '1g_faiss_consistency',
    '1h_full_corpus_provenance',
    '1i_geometric_integrity',
    '1j_metadata_manifest_agreement',
    '1k_duplicates_full_corpus',
    '1_integrity',
    '2_split_staleness',
    '2_split_staleness_recompute',
    '3_zero_vector_cause',
    '4_filter_reality',
    '5_cap_semantics',
    '6a_norms',
    '6b_anisotropy',
    '6c_alpha_sweep',
    '7_duplicates_and_leakage',
    '8_ann_recall',
    '9_ligand_redundancy',
    '10_ligand_similarity',
    '11_experimental_method',
    '12_sifts_provenance',
    '12a_retrieval_overall',
    '12b_stratified_by_pfam',
    '12c_discriminativeness',
    '12d_alpha_decides_default',
    '13b_retrieval_with_CI',
    '13c_retrieval_neardup_excluded',
    '13d_knn_purity_vs_null_',
    '13e_alpha_seed_stability',
    '13f_paired_pocket_vs_concat',
    '13g_alpha_heldout',
    '13h_per_family_CI',
    '13i_release_cluster_leakage',
    '13j_time_split_retrieval',
    '13k_time_split_pocket_only',
    '13_bindingdb_joinability',
]

out_json = Path('audit_results.json')
out_json.write_text(json.dumps(RESULTS, indent=2, default=str))

lines = ['# Sci Data revision - data audit results', '']
for sec, kv in RESULTS.items():
    lines.append('## %s' % sec)
    for k, v in kv.items():
        lines.append('- **%s**: `%s`' % (k, v))
    lines.append('')
Path('audit_results.md').write_text(chr(10).join(lines), encoding='utf-8')
_flush_results()   # also refresh the Drive copy

missing = [s for s in EXPECTED_SECTIONS if s not in RESULTS]
print('=' * 72)
print('sections present : %d / %d' % (len(RESULTS), len(EXPECTED_SECTIONS)))
print('written to       : %s' % CKPT_MD)
if missing:
    print('')
    print('!! INCOMPLETE - %d section(s) never ran:' % len(missing))
    for s in missing:
        print('     - %s' % s)
    print('')
    print('Re-run the cells that produce them, then run this cell again.')
    print('Nothing already computed is lost - RESULTS is restored from the')
    print('checkpoint, so you only need the cells listed above.')
else:
    print('')
    print('COMPLETE - all %d sections present.' % len(EXPECTED_SECTIONS))
print('=' * 72)

try:
    from google.colab import files
    files.download('audit_results.json')
    files.download('audit_results.md')
except Exception:
    pass
```
