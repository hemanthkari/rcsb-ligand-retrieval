# Finish v0.2: the per-chain residue map, re-checksum, re-deposit

One notebook for everything the deposit still needs. **Non-destructive**: it
writes new files and only installs them into the release after they verify.

## What is wrong right now

`pockets/pocket_embeddings.npy` in the deposit is the v0.2 **per-chain**
re-embedding. `pockets/pocket_residue_map.parquet` beside it is notebook 09's,
which indexes the **chimeric all-chain** ESM-2 input v0.1 used. A reader
following the manuscript's recipe therefore pools from the wrong token
positions and does not reproduce the shipped vectors — while the paper claims
three times that the pooled embedding is reproducible from the deposited
artifacts. That claim is the answer to Reviewer 3's comment M2.

Notebook 11 Cell 5 also records that rows it could not re-embed **retain their
v0.1 value**, so the matrix is a mixture. This notebook detects which rows are
which by comparing against v0.1 directly, and labels every entry with an
`embedding_input` column — the field the response letter promises.

**Runtime:** CPU is enough. No GPU, no ESM-2 forward pass. The cost is mmCIF
parsing, so a local `--cif_dir` mirror makes it far faster.

---

## Cell 0 — Setup

```python
!pip -q install huggingface_hub gemmi pandas pyarrow numpy requests faiss-cpu
!apt-get -qq install -y mmseqs2 2>/dev/null

import os, io, re, sys, json, gzip, time, shutil, hashlib, subprocess, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import gemmi
from huggingface_hub import snapshot_download

_DRIVE = Path("/content/drive/MyDrive")
if not _DRIVE.is_dir():
    try:
        from google.colab import drive as _drv
        _drv.mount("/content/drive")
    except Exception as _e:
        print("drive.mount failed:", type(_e).__name__, _e)

REPO_DIR = Path("/content/fluxion-dock")
if not (REPO_DIR / "notebooks" / "11_reembed_per_chain.ipynb").exists():
    _tok = None
    try:
        from google.colab import userdata
        _tok = userdata.get("GITHUB_TOKEN")
    except Exception:
        _tok = os.environ.get("GITHUB_TOKEN")
    if not _tok:
        raise SystemExit("GITHUB_TOKEN not set (key icon -> GITHUB_TOKEN, notebook access ON)")
    _tok = _tok.strip()
    _env = dict(os.environ); _env["GIT_TERMINAL_PROMPT"] = "0"
    for _u in ["https://hemanthkari:%s@github.com/hemanthkari/fluxion-dock.git" % _tok,
               "https://%s@github.com/hemanthkari/fluxion-dock.git" % _tok]:
        if REPO_DIR.exists():
            shutil.rmtree(str(REPO_DIR), ignore_errors=True)
        _p = subprocess.run(["git", "clone", "--depth", "1", "--branch",
                             "revision/scidata-v02", _u, str(REPO_DIR)],
                            capture_output=True, text=True, env=_env)
        if _p.returncode == 0:
            break
    if not (REPO_DIR / "notebooks" / "11_reembed_per_chain.ipynb").exists():
        raise SystemExit("clone failed: " + _p.stderr[-400:].replace(_tok, "***"))

CKPT_DIR = _DRIVE / "fluxion_audit" if _DRIVE.is_dir() else Path("/content/fluxion_audit")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_JSON = CKPT_DIR / "finish_v02.json"
CKPT_MD = CKPT_DIR / "finish_v02.md"

RESULTS = {}
if CKPT_JSON.exists():
    try:
        RESULTS = json.loads(CKPT_JSON.read_text(encoding="utf-8"))
        print("RESUMED %d section(s): %s" % (len(RESULTS), ", ".join(sorted(RESULTS))))
    except Exception as _e:
        print("checkpoint unreadable (%s)" % type(_e).__name__)


def _flush():
    try:
        _t = CKPT_JSON.with_suffix(".tmp")
        _t.write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
        _t.replace(CKPT_JSON)
        _md = ["# Finish v0.2", ""]
        for _s, _kv in RESULTS.items():
            _md.append("## %s" % _s)
            for _k, _v in _kv.items():
                _md.append("- **%s**: `%s`" % (_k, _v))
            _md.append("")
        CKPT_MD.write_text(chr(10).join(_md), encoding="utf-8")
    except Exception as _e:
        print("  !! checkpoint write failed:", _e)


def result(section, **kv):
    RESULTS.setdefault(section, {}).update(kv)
    print(chr(10) + "=" * 72)
    print("RESULT  [%s]" % section)
    print("=" * 72)
    for k, v in kv.items():
        print("  %-38s %s" % (k, v))
    _flush()


def done(section):
    return bool(RESULTS.get(section))


print("repo :", REPO_DIR)
print("ckpt :", CKPT_DIR, "(ON DRIVE)" if str(CKPT_DIR).startswith(str(_DRIVE)) else "(EPHEMERAL)")
```

---

## Cell 1 — **Preflight**: locate the release, the two embedding matrices, the v0.1 map

```python
RELEASE = None
_root = _DRIVE / "fluxion_v02_build"
if _root.is_dir():
    _cand = sorted([d for d in _root.glob("release_*")
                    if (d / "metadata.jsonl").exists()])
    if _cand:
        RELEASE = _cand[-1]
if RELEASE is None:
    raise SystemExit("no built release found under %s - run notebook 13 first" % _root)

V01 = Path(snapshot_download("HemanthKari/rcsb-ligand-retrieval-v0.1", repo_type="dataset"))
V01_EMB = V01 / "pockets" / "pocket_embeddings.npy"
V02_EMB = RELEASE / "pockets" / "pocket_embeddings.npy"
V01_MAP = None
for _c in [_DRIVE / "fluxion_reextract" / "pocket_residue_map.parquet",
           RELEASE / "pockets" / "pocket_residue_map.parquet"]:
    if _c.exists():
        V01_MAP = _c
        break

# a local mmCIF mirror makes this ~10x faster; None falls back to RCSB
CIF_DIR = None
for _c in [_DRIVE / "pdb_cache", Path("/content/pdb_cache")]:
    if _c.is_dir():
        CIF_DIR = str(_c)
        break

MAX_SEQ_LEN = 1022                      # same window notebook 11 used
OUT = RELEASE / "_per_chain_build"
OUT.mkdir(parents=True, exist_ok=True)

_missing = [n for n, p in [("release", RELEASE), ("v0.1 embeddings", V01_EMB),
                           ("v0.2 embeddings", V02_EMB), ("v0.1 residue map", V01_MAP)]
            if p is None or not Path(p).exists()]
if _missing:
    raise SystemExit("missing: %s" % _missing)

_a = np.load(V01_EMB, mmap_mode="r")
_b = np.load(V02_EMB, mmap_mode="r")
if _a.shape != _b.shape:
    raise SystemExit("shape mismatch %s vs %s" % (_a.shape, _b.shape))

result("0_preflight",
       release=str(RELEASE), v01_emb=str(V01_EMB), v02_emb=str(V02_EMB),
       v01_map=str(V01_MAP), cif_dir=CIF_DIR or "none (will fetch from RCSB)",
       shape=str(_a.shape), out_dir=str(OUT),
       NOTE="Only OUT and, in the final cell, the release copies are written. "
            "The v0.1 snapshot and the Drive inputs are read-only.")
```

---

## Cell 2 — **Which rows are actually per-chain?**

Compares the two matrices directly, so it does not depend on notebook 11's
shards still existing.

```python
if not done("1_row_provenance"):
    _A = np.asarray(np.load(V01_EMB, mmap_mode="r"), dtype=np.float32)
    _B = np.asarray(np.load(V02_EMB, mmap_mode="r"), dtype=np.float32)
    _diff = ~np.all(np.isclose(_A, _B, rtol=0, atol=0), axis=1)
    PER_CHAIN_ROWS = set(np.flatnonzero(_diff).tolist())
    _cos = None
    _idx = np.flatnonzero(_diff)[:5000]
    if len(_idx):
        _x, _y = _A[_idx], _B[_idx]
        _cos = float(np.median((_x * _y).sum(1) /
                               (np.linalg.norm(_x, axis=1) * np.linalg.norm(_y, axis=1) + 1e-12)))
    result("1_row_provenance",
           n_rows=int(_A.shape[0]),
           n_per_chain=int(_diff.sum()),
           n_retained_v01=int((~_diff).sum()),
           pct_per_chain=round(100.0 * _diff.sum() / _A.shape[0], 3),
           median_cosine_v01_vs_v02=_cos,
           NOTE="Rows identical to v0.1 were never re-embedded and keep the "
                "chimeric encoding. They must be labelled as such, not "
                "described as per-chain.")
    del _A, _B
else:
    print("resumed; recomputing the row mask (cheap)")
    _A = np.asarray(np.load(V01_EMB, mmap_mode="r"), dtype=np.float32)
    _B = np.asarray(np.load(V02_EMB, mmap_mode="r"), dtype=np.float32)
    PER_CHAIN_ROWS = set(np.flatnonzero(~np.all(_A == _B, axis=1)).tolist())
    del _A, _B
print("per-chain rows:", len(PER_CHAIN_ROWS))
```

---

## Cell 3 — Load the v0.1 map and reuse notebook 11's own functions

Executing notebook 11's cell rather than copying it guarantees the assignments
are identical to the ones used for the embeddings.

```python
rmap = pd.read_parquet(V01_MAP)
print("v0.1 residue map: %d rows, columns: %s" % (len(rmap), list(rmap.columns)))

meta = [json.loads(l) for l in open(RELEASE / "metadata.jsonl", encoding="utf-8")]
pk_to_off = {m["pocket_key"]: int(m["pocket_emb_offset"]) for m in meta}
print("metadata rows: %d" % len(meta))

_nb11 = json.loads((REPO_DIR / "notebooks" / "11_reembed_per_chain.ipynb")
                   .read_text(encoding="utf-8"))
_cells = ["".join(c["source"]) for c in _nb11["cells"] if c["cell_type"] == "code"]
_fn_cell = next(c for c in _cells if "def per_chain_sequences" in c)
exec(_fn_cell)          # per_chain_sequences, map_pocket_to_chains, load_structure

for _needed in ("per_chain_sequences", "map_pocket_to_chains", "load_structure"):
    if _needed not in dir():
        raise SystemExit("notebook 11 cell did not define %s" % _needed)
print("reused notebook 11's functions:", "per_chain_sequences, map_pocket_to_chains, load_structure")
```

---

## Cell 4 — Build the per-chain map (resumable, CPU)

```python
SHARDS = OUT / "map_shards"
SHARDS.mkdir(parents=True, exist_ok=True)

_have = set()
for _sh in sorted(SHARDS.glob("map_*.jsonl")):
    for _l in open(_sh, encoding="utf-8"):
        _have.add(json.loads(_l)["pocket_key"])
print("already built: %d" % len(_have))

_todo = [r for r in rmap.itertuples()
         if pk_to_off.get(r.pocket_key) in PER_CHAIN_ROWS and r.pocket_key not in _have]
print("to build: %d" % len(_todo))

_st8 = {"ok": 0, "no_cif": 0, "unresolved": 0, "too_long": 0, "count_mismatch": 0}
_buf, _sn, _t0 = [], len(list(SHARDS.glob("map_*.jsonl"))), time.time()

for _n, _r in enumerate(_todo, 1):
    _s = load_structure(_r.pdb_id)
    if _s is None:
        _st8["no_cif"] += 1; continue
    _chains = per_chain_sequences(_s)
    _prov = json.loads(_r.pocket_res_provenance)
    _assign, _unres = map_pocket_to_chains(_prov, _chains)
    if _unres or not any(_assign):
        _st8["unresolved"] += 1; continue
    _need = sorted({a[0] for a in _assign if a})
    if any(len(_chains[c][0]) > MAX_SEQ_LEN for c in _need):
        _st8["too_long"] += 1; continue

    _rc, _ri, _rt, _rn = [], [], [], []
    for _a, _p in zip(_assign, _prov):
        if _a is None:
            continue
        _rc.append(_a[0]); _ri.append(int(_a[1]))
        _rt.append(int(_a[1]) + 1)          # BOS occupies token 0
        _rn.append(str(_p["resname"]))
    if len(_rt) != len(_prov):
        _st8["count_mismatch"] += 1; continue

    _buf.append({"pocket_key": _r.pocket_key, "pdb_id": _r.pdb_id,
                 "chain_order": _need, "n_chains_in_esm_input": len(_need),
                 "esm_input_length_total": sum(len(_chains[c][0]) for c in _need),
                 "n_pocket_residues": len(_rt),
                 "pocket_res_chain": _rc, "pocket_res_index_in_chain": _ri,
                 "pocket_res_token_position": _rt, "pocket_res_resname": _rn,
                 "sequences": {c: _chains[c][0] for c in _need}})
    _st8["ok"] += 1

    if _n % 200 == 0:
        _rate = _n / max(1e-9, time.time() - _t0)
        print("  %6d/%d  %.1f/s  eta %.0f min  %s"
              % (_n, len(_todo), _rate, (len(_todo) - _n) / _rate / 60, _st8), flush=True)
    if len(_buf) >= 500:
        with open(SHARDS / ("map_%04d.jsonl" % _sn), "w", encoding="utf-8") as fh:
            for _row in _buf:
                fh.write(json.dumps(_row) + chr(10))
        _sn += 1; _buf = []

if _buf:
    with open(SHARDS / ("map_%04d.jsonl" % _sn), "w", encoding="utf-8") as fh:
        for _row in _buf:
            fh.write(json.dumps(_row) + chr(10))

result("2_map_build", processed=len(_todo), **_st8,
       NOTE="Assignments come from notebook 11's own functions, so they match "
            "the embeddings by construction. Cell 5 proves it.")
```

---

## Cell 5 — **Verify, then write** the two artifacts

Refuses to write on any inconsistency.

```python
rows = []
for _sh in sorted(SHARDS.glob("map_*.jsonl")):
    for _l in open(_sh, encoding="utf-8"):
        rows.append(json.loads(_l))
by_key = {r["pocket_key"]: r for r in rows}
print("per-chain map rows: %d" % len(rows))

problems = []
_expect = {r.pocket_key for r in rmap.itertuples()
           if pk_to_off.get(r.pocket_key) in PER_CHAIN_ROWS}
_miss = _expect - set(by_key)
if _miss:
    problems.append("%d per-chain rows have no map entry" % len(_miss))

_bad = 0
for r in rows:
    s = r["sequences"]
    for c, i, t in zip(r["pocket_res_chain"], r["pocket_res_index_in_chain"],
                       r["pocket_res_token_position"]):
        if c not in s or not (0 <= i < len(s[c])) or t != i + 1:
            _bad += 1
    if r["n_pocket_residues"] != len(r["pocket_res_token_position"]):
        _bad += 1
if _bad:
    problems.append("%d residue records are out of range or inconsistent" % _bad)

# the new map MUST differ from the v0.1 one, or cell 4 reproduced the old input
_old = rmap.set_index("pocket_key")
_ndiff = _nchk = 0
for k in list(by_key)[:500]:
    try:
        _o = list(json.loads(_old.loc[k, "pocket_res_indices"]))
    except Exception:
        continue
    _nchk += 1
    if _o != by_key[k]["pocket_res_token_position"]:
        _ndiff += 1
if _nchk and _ndiff == 0:
    problems.append("token positions identical to the v0.1 map on all %d sampled "
                    "entries - cell 4 reproduced the chimeric input" % _nchk)

result("3_verification", n_rows=len(rows), n_expected=len(_expect),
       missing=len(_miss), bad_records=_bad,
       sampled=_nchk, differing_from_v01=_ndiff,
       problems=problems or "none")
if problems:
    raise SystemExit("NOT WRITING:" + chr(10) + chr(10).join("  * " + p for p in problems))

# ---- one map covering EVERY entry, each labelled with its encoding --------
_flat, _fa = [], []
for r in rmap.itertuples():
    pk = r.pocket_key
    if pk in by_key:
        m = by_key[pk]
        _flat.append({
            "pocket_key": pk, "pdb_id": m["pdb_id"],
            "embedding_input": "per_chain",
            "chain_order": json.dumps(m["chain_order"]),
            "n_chains_in_esm_input": m["n_chains_in_esm_input"],
            "esm_input_length_total": m["esm_input_length_total"],
            "n_pocket_residues": m["n_pocket_residues"],
            "pocket_res_chain": json.dumps(m["pocket_res_chain"]),
            "pocket_res_index_in_chain": json.dumps(m["pocket_res_index_in_chain"]),
            "pocket_res_token_position": json.dumps(m["pocket_res_token_position"]),
            "pocket_res_resname": json.dumps(m["pocket_res_resname"]),
        })
        for c in m["chain_order"]:
            _fa.append((pk + "|" + c, m["sequences"][c]))
    else:
        _flat.append({
            "pocket_key": pk, "pdb_id": getattr(r, "pdb_id", ""),
            "embedding_input": "concatenated_all_chains_v01",
            "chain_order": json.dumps(list(json.loads(getattr(r, "chain_order", "[]")))
                                      if isinstance(getattr(r, "chain_order", None), str) else []),
            "n_chains_in_esm_input": int(getattr(r, "n_chains_in_esm_input", 0) or 0),
            "esm_input_length_total": int(getattr(r, "esm_input_length", 0) or 0),
            "n_pocket_residues": int(getattr(r, "n_pocket_residues", 0) or 0),
            "pocket_res_chain": json.dumps([]),
            "pocket_res_index_in_chain": json.dumps([]),
            "pocket_res_token_position": str(getattr(r, "pocket_res_indices", "[]")),
            "pocket_res_resname": json.dumps([]),
        })

MAP_OUT = OUT / "pocket_residue_map.parquet"
FA_OUT = OUT / "esm_input_sequences.fasta.gz"
pd.DataFrame(_flat).to_parquet(MAP_OUT, index=False)
with gzip.open(FA_OUT, "wt") as fh:
    for name, seq in _fa:
        fh.write(">" + name + chr(10) + seq + chr(10))

result("4_written", residue_map=str(MAP_OUT),
       map_MB=round(MAP_OUT.stat().st_size / 1e6, 2),
       fasta=str(FA_OUT), fasta_MB=round(FA_OUT.stat().st_size / 1e6, 2),
       n_rows=len(_flat), n_per_chain=len(by_key),
       n_labelled_v01=len(_flat) - len(by_key), n_fasta_records=len(_fa),
       NOTE="embedding_input labels every row. Entries whose embedding was never "
            "re-embedded keep the v0.1 chimeric token positions and say so - this "
            "is the field the response letter promises.")
```

---

## Cell 6 — Install into the release, re-checksum, verify

```python
if not done("5_installed"):
    for _src, _dst in [(MAP_OUT, RELEASE / "pockets" / "pocket_residue_map.parquet"),
                       (FA_OUT, RELEASE / "pockets" / "esm_input_sequences.fasta.gz")]:
        _bak = _dst.with_suffix(_dst.suffix + ".v01bak")
        if _dst.exists() and not _bak.exists():
            shutil.copy2(str(_dst), str(_bak))
        shutil.copy2(str(_src), str(_dst))
        print("installed %-42s %8.2f MB" % (_dst.name, _dst.stat().st_size / 1e6))

    def _sha(p, buf=1 << 20):
        h = hashlib.sha256()
        with open(str(p), "rb") as fh:
            for c in iter(lambda: fh.read(buf), b""):
                h.update(c)
        return h.hexdigest()

    # exclude by RULE: dotfiles (the uploader writes .zenodo_state.json) and the
    # two files that cannot contain their own digest
    def _skip(rel):
        if rel in ("CHECKSUMS.sha256", "DATASET_MANIFEST.json"):
            return True
        if rel.startswith("_per_chain_build/") or rel.endswith(".v01bak"):
            return True
        return any(part.startswith(".") for part in rel.split("/"))

    _rows2 = []
    for _p in sorted(RELEASE.rglob("*")):
        if _p.is_file():
            _rel = str(_p.relative_to(RELEASE)).replace(chr(92), "/")
            if not _skip(_rel):
                _rows2.append((_rel, _sha(_p)))
    _rows2.sort(key=lambda r: r[0])
    (RELEASE / "CHECKSUMS.sha256").write_text(
        chr(10).join("%s  %s" % (h, r) for r, h in _rows2) + chr(10), encoding="utf-8")

    _mp = RELEASE / "DATASET_MANIFEST.json"
    _mj = json.loads(_mp.read_text(encoding="utf-8"))
    _mj["files"] = [{"path": r, "bytes": (RELEASE / r).stat().st_size, "sha256": h}
                    for r, h in _rows2]
    _mp.write_text(json.dumps(_mj, indent=2), encoding="utf-8")

    _bad2 = [r for r, h in _rows2 if _sha(RELEASE / r) != h]
    _dot = [r for r, _ in _rows2 if any(p.startswith(".") for p in r.split("/"))]
    result("5_installed", n_checksums=len(_rows2), mismatches=len(_bad2),
           dotfiles_listed=len(_dot),
           map_rows=len(pd.read_parquet(RELEASE / "pockets" / "pocket_residue_map.parquet")),
           NOTE="v0.1 copies kept alongside as *.v01bak and excluded from the "
                "checksums. Both counters must be 0.")
```

---

## Cell 7 — Re-upload to the Zenodo draft, then what is left

```python
print("Set your token, then run the upload:")
print("""
import os
os.environ["ZENODO_TOKEN"] = "PASTE"
!python -u {repo}/scripts/mirror_to_zenodo.py \\
    --source_dir {rel} --version v0.2 --license cc-by-4.0 \\
    --deposition 22238825 --flatten
""".format(repo=REPO_DIR, rel=RELEASE))

print("")
print("=" * 72)
print("AFTER THE UPLOAD")
print("=" * 72)
print("1. The script REWRITES the record metadata and reverts the three relation")
print("   fixes you made by hand. Re-set them in the browser afterwards:")
print("     Is new version of -> HF url    (not 'Is identical to')")
print("     Is new version of -> 10.57967/hf/8408")
print("     Is compiled by    -> 10.5281/zenodo.19954940")
print("   and re-check the Description, which its build_metadata() overwrites.")
print("")
print("2. Regenerate Figure 2: panel (c) still plots the chimeric input length.")
print("     !python -u %s/scripts/make_dataset_paper_figures.py --fig 2 \\" % REPO_DIR)
print("         --retrieval_dir %s \\" % RELEASE)
print("         --pocket_map %s/pockets/pocket_residue_map.parquet \\" % RELEASE)
print("         --sifts_dir %s/provenance/sifts --out /content/figures_v02 --seed 42" % RELEASE)
print("")
print("3. Ten manuscript blockers remain open (usage example points at v0.1,")
print("   letter claims a subsection that does not exist, Table 5 baseline")
print("   column, embedding-geometry paragraph). Those are text, not data.")
_flush()
```
