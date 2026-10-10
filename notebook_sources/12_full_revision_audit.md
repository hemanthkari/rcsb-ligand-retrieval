# Sci Data revision — FULL audit (notebooks 08 + 10 merged, one run)

Every data-dependent number for the Scientific Data major revision
(submission `9625e99c-515f-4d60-b5da-cbd12d8d1456`).

**Runtime:** A100/H100 recommended. Only the encoder-comparison section
(ProtT5-XL + ESM-2 ladder) needs the GPU; everything else is CPU/network.

**Every `result()` is flushed to Drive immediately.** A disconnect at any
point keeps all completed sections; re-running resumes and skips nothing
you already have. Mount Drive BEFORE running Cell 0 or the run will stop.

---

## Cell 0 — Setup: dependencies, Drive, dataset, checkpoint

```python
!pip -q install huggingface_hub faiss-cpu rdkit numpy pandas scipy pyarrow requests h5py scikit-learn umap-learn matplotlib fair-esm transformers sentencepiece fsspec aiohttp

import os, io, re, csv, sys, gzip, json, math, time, tarfile, hashlib
import itertools, collections
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from huggingface_hub import snapshot_download

REPO = "HemanthKari/rcsb-ligand-retrieval-v0.1"
LOCAL = Path(snapshot_download(REPO, repo_type="dataset"))
print("dataset at:", LOCAL)

# notebook 09/11 wrote artifacts to Drive. Never mkdir(parents=True) under
# /content/drive when it is unmounted: that fabricates a fake tree and then
# blocks the real mount with 'Mountpoint must not already contain files'.
_DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_DIR = _DRIVE_ROOT / "fluxion_reextract" if _DRIVE_ROOT.is_dir() \
            else Path("./fluxion_reextract")

MAP_DIR = None
for cand in [DRIVE_DIR, Path("/content"), Path(".")]:
    if (cand / "pocket_residue_map.parquet").exists():
        MAP_DIR = cand
        break

OUT_DIR = DRIVE_DIR if DRIVE_DIR.exists() else Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CKPT_CANDIDATES = [_DRIVE_ROOT / "fluxion_audit", Path("/content/fluxion_audit")]
CKPT_DIR = None
for _c in _CKPT_CANDIDATES:
    if str(_c).startswith(str(_DRIVE_ROOT)) and not _DRIVE_ROOT.is_dir():
        continue
    try:
        _c.mkdir(parents=True, exist_ok=True)
        CKPT_DIR = _c; break
    except Exception:
        continue
CKPT_DIR = CKPT_DIR or Path(".")
CKPT_JSON = CKPT_DIR / "audit_results_full.json"
CKPT_MD   = CKPT_DIR / "audit_results_full.md"

_ON_DRIVE = _DRIVE_ROOT.is_dir() and str(CKPT_DIR).startswith(str(_DRIVE_ROOT))
print("residue map dir:", MAP_DIR)
print("output dir     :", OUT_DIR)
print("checkpoint dir :", CKPT_DIR,
      "(ON DRIVE - survives disconnect)" if _ON_DRIVE
      else "(EPHEMERAL - mount Drive and re-run this cell, or you WILL lose results)")
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

## Cell 16 — Load artifacts, SIFTS labels, define helpers, init RESULTS

```python
# checkpoint machinery + result() already defined in 08's Cell 1 above.

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
# Cell 0 resolves MAP_DIR before anything mounts Drive, and notebook 09 may
# have fallen back to ./fluxion_reextract under a different CWD. Re-resolve
# here, now that Drive is definitely mounted, over a wider candidate list.
if MAP_DIR is None:
    for _c in [Path('/content/drive/MyDrive/fluxion_reextract'),
               Path('/content/drive/MyDrive/fluxion_v02'),
               Path('/content/fluxion_reextract'),
               Path('./fluxion_reextract'),
               Path('/content'), Path('.')]:
        if (_c / 'pocket_residue_map.parquet').exists():
            MAP_DIR = _c
            print('re-resolved MAP_DIR ->', MAP_DIR)
            break
    if MAP_DIR is None:
        import glob as _g
        _hits = _g.glob('/content/drive/MyDrive/**/pocket_residue_map.parquet',
                        recursive=True)
        if _hits:
            MAP_DIR = Path(_hits[0]).parent
            print('found by search ->', MAP_DIR)

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

## Cell 27 — Write consolidated results **and report completeness**

```python
EXPECTED_SECTIONS = [
    '0_provenance',
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
    '0_inputs',
    '1a_encoder_subsample',
    '1c_encoder_comparison',
    '2_dataset_overlap',
    '3_diversity',
    '4_qualitative_panel',
    '5a_silhouette_by_family',
    '5b_umap_and_separation',
    '6_response_letter_sentences',
]

_flush_results()
Path("audit_results_full.json").write_text(
    json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")

missing = [s for s in EXPECTED_SECTIONS if s not in RESULTS]
print("=" * 72)
print("sections present : %d / %d" % (len(RESULTS), len(EXPECTED_SECTIONS)))
print("written to       : %s" % CKPT_MD)
if missing:
    print("")
    print("!! INCOMPLETE - %d section(s) never ran:" % len(missing))
    for s in missing:
        print("     - %s" % s)
    print("")
    print("Re-run the cells that produce them, then run this cell again.")
    print("Nothing already computed is lost - RESULTS is restored from the")
    print("checkpoint, so only the cells listed above need to run.")
else:
    print("")
    print("COMPLETE - all %d sections present." % len(EXPECTED_SECTIONS))
print("=" * 72)

try:
    from google.colab import files
    files.download(str(CKPT_MD))
except Exception:
    pass
```
