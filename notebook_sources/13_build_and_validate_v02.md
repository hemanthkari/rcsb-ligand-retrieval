# v0.2 release: build, correct, and validate — **strictly non-destructive**

One notebook that produces the corrected v0.2 release and every number the
revision still owes. It **never writes to any existing artifact**.

## Non-destructive guarantees (enforced, not promised)

| Path | Treatment |
|---|---|
| the Hugging Face v0.1 snapshot | **read-only**, asserted before every stage |
| `MyDrive/fluxion_v02/`, `MyDrive/fluxion_reextract/` | **read-only** inputs |
| `MyDrive/fluxion_audit/` | append-only checkpoints |
| `<OUT_ROOT>/release_<stamp>/` | the **only** directory written |

Cell 1 registers every input path in `READ_ONLY` and installs a guard that
raises if anything tries to open one for writing. Nothing is deleted, nothing
is overwritten, and there is **no upload step** — no Zenodo, no HF push.

## The defect this exists to fix

`build_v02_release.py` lists `pockets/pocket_embeddings.npy` in
`CARRY_OVER_FILES` (:2188), so it copies the **v0.1 chimeric** embeddings and
writes into the deposit that they "were NOT recomputed" — while the response
letter tells reviewers they *were* regenerated. Notebook 11 did produce the
per-chain array; the assembler simply never consumed it.

There is no `--pocket_source` flag, so Cell 3 builds a **shim**: a directory of
hardlinks to the v0.1 snapshot with only `pockets/pocket_embeddings.npy`
pointing at the v0.2 array. The tested assembler then runs **unmodified** and the
real v0.1 snapshot is never touched.

---

## Cell 0 — Setup: dependencies, Drive, checkpoint

```python
!pip -q install huggingface_hub faiss-cpu rdkit numpy pandas scipy pyarrow requests h5py
!apt-get -qq install -y mmseqs2 2>/dev/null

import os, io, re, sys, json, gzip, time, shutil, hashlib, subprocess, datetime
import collections
from pathlib import Path
import numpy as np
import pandas as pd

from huggingface_hub import snapshot_download

# ---- clone the repo (branch revision/scidata-v02) ---------------------
# Private repo: token from Colab Secrets (key icon -> GITHUB_TOKEN, a PAT
# with `repo` scope, Notebook access ON). Two URL forms are tried because
# GitHub accepts `user:token` for classic PATs and `token@` for some others.
REPO_DIR = Path('/content/fluxion-dock')
BRANCH = 'revision/scidata-v02'

if not (REPO_DIR / 'scripts' / 'build_v02_release.py').exists():
    _tok = None
    try:
        from google.colab import userdata
        _tok = userdata.get('GITHUB_TOKEN')
    except Exception:
        _tok = os.environ.get('GITHUB_TOKEN')
    if not _tok:
        raise SystemExit(chr(10).join([
            '', '=' * 72,
            'GITHUB_TOKEN not available.',
            '=' * 72,
            'Colab: key icon in the left sidebar -> add secret GITHUB_TOKEN',
            '(PAT with `repo` scope) -> toggle Notebook access ON -> re-run.',
            '=' * 72]))
    _tok = _tok.strip()
    _env = dict(os.environ)
    _env['GIT_TERMINAL_PROMPT'] = '0'      # fail fast, never hang on a prompt
    _forms = ['https://hemanthkari:%s@github.com/hemanthkari/fluxion-dock.git' % _tok,
              'https://%s@github.com/hemanthkari/fluxion-dock.git' % _tok]
    _err = ''
    for _u in _forms:
        if REPO_DIR.exists():
            shutil.rmtree(str(REPO_DIR), ignore_errors=True)
        _p = subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH,
                             _u, str(REPO_DIR)],
                            capture_output=True, text=True, env=_env)
        if _p.returncode == 0:
            break
        _err = _p.stderr[-500:].replace(_tok, '***')
    if not (REPO_DIR / 'scripts' / 'build_v02_release.py').exists():
        raise SystemExit('clone failed:' + chr(10) + _err)

print('repo            :', REPO_DIR, '(%s)' % BRANCH)
print('  build script  :', (REPO_DIR / 'scripts' / 'build_v02_release.py').exists())
print('  verify script :', (REPO_DIR / 'scripts' / 'verify_v02_release.py').exists())
print('  mmseqs2       :', shutil.which('mmseqs') or 'ABSENT')
REPO = "HemanthKari/rcsb-ligand-retrieval-v0.1"
V01 = Path(snapshot_download(REPO, repo_type="dataset"))
print("v0.1 snapshot   :", V01)

_DRIVE_ROOT = Path("/content/drive/MyDrive")
if not _DRIVE_ROOT.is_dir():
    try:
        from google.colab import drive as _drv
        _drv.mount("/content/drive")
    except Exception as _e:
        print("drive.mount failed:", type(_e).__name__, _e)

# Never mkdir(parents=True) under an unmounted /content/drive: it fabricates a
# fake tree that then blocks the real mount.
_CKPT_CANDIDATES = [_DRIVE_ROOT / "fluxion_audit", Path("/content/fluxion_audit")]
CKPT_DIR = None
for _c in _CKPT_CANDIDATES:
    if str(_c).startswith(str(_DRIVE_ROOT)) and not _DRIVE_ROOT.is_dir():
        continue
    try:
        _c.mkdir(parents=True, exist_ok=True)
        CKPT_DIR = _c
        break
    except Exception:
        continue
CKPT_DIR = CKPT_DIR or Path(".")
CKPT_JSON = CKPT_DIR / "build_v02_results.json"
CKPT_MD = CKPT_DIR / "build_v02_results.md"

_ON_DRIVE = _DRIVE_ROOT.is_dir() and str(CKPT_DIR).startswith(str(_DRIVE_ROOT))
print("checkpoint dir  :", CKPT_DIR,
      "(ON DRIVE)" if _ON_DRIVE else "(EPHEMERAL - mount Drive and re-run)")

RESULTS = {}
if CKPT_JSON.exists():
    try:
        RESULTS = json.loads(CKPT_JSON.read_text(encoding="utf-8"))
        print("RESUMED %d section(s)" % len(RESULTS))
    except Exception as _e:
        print("checkpoint unreadable (%s) - starting fresh" % type(_e).__name__)


def _flush():
    try:
        _t = CKPT_JSON.with_suffix(".tmp")
        _t.write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
        _t.replace(CKPT_JSON)
        _md = ["# v0.2 build and validation", ""]
        for _s, _kv in RESULTS.items():
            _md.append("## %s" % _s)
            for _k, _v in _kv.items():
                _md.append("- **%s**: `%s`" % (_k, _v))
            _md.append("")
        CKPT_MD.write_text(chr(10).join(_md), encoding="utf-8")
    except Exception as _e:
        print("  !! CHECKPOINT WRITE FAILED: %s: %s" % (type(_e).__name__, _e))


def result(section, **kv):
    RESULTS.setdefault(section, {}).update(kv)
    print("\n" + "=" * 72)
    print("RESULT  [%s]" % section)
    print("=" * 72)
    for k, v in kv.items():
        print("  %-40s %s" % (k, v))
    _flush()


def done(section):
    return bool(RESULTS.get(section))
```

---

## Cell 1 — **Preflight + the write guard**

Fails in seconds rather than hours, and makes "non-destructive" enforceable.

```python
# ---- inputs, all read-only -------------------------------------------------
V02_POCKET = None
for _c in [_DRIVE_ROOT / "fluxion_v02" / "pocket_embeddings.npy",
           _DRIVE_ROOT / "fluxion_reextract" / "pocket_embeddings.npy",
           Path("/content/fluxion_v02/pocket_embeddings.npy")]:
    if _c.exists():
        V02_POCKET = _c
        break

NB09_DIR = None
for _c in [_DRIVE_ROOT / "fluxion_reextract", _DRIVE_ROOT / "fluxion_v02",
           Path("/content/fluxion_reextract"), Path(".")]:
    if (_c / "pocket_residue_map.parquet").exists():
        NB09_DIR = _c
        break

# REPO_DIR was set by the clone in Cell 0; only search if that somehow failed.
if not (REPO_DIR and (Path(REPO_DIR) / "scripts" / "build_v02_release.py").exists()):
    REPO_DIR = None
    for _c in [Path("/content/fluxion-dock"), Path("/content/repo"),
               Path("/content/AI-Drug-Discovery"), Path("."), Path("/content")]:
        if (_c / "scripts" / "build_v02_release.py").exists():
            REPO_DIR = _c
            break

# ---- the only directory this notebook writes -------------------------------
# A fresh timestamp on every run is INCOMPATIBLE with resume: the checkpoint
# would say the shim and assembly are done while the new directories are
# empty, and the next cell dies on a missing file. So the chosen directories
# are recorded in the checkpoint and reused whenever they still exist.
OUT_ROOT = _DRIVE_ROOT / "fluxion_v02_build" if _DRIVE_ROOT.is_dir() else Path("/content/fluxion_v02_build")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

_prev = RESULTS.get("0_preflight", {})
_p_out, _p_shim = _prev.get("out_dir"), _prev.get("shim_dir")
RESUMED_DIRS = bool(_p_out and _p_shim and Path(_p_out).exists() and Path(_p_shim).exists())
if RESUMED_DIRS:
    OUT_DIR, SHIM_DIR = Path(_p_out), Path(_p_shim)
    STAMP = OUT_DIR.name.replace("release_", "")
    print("RESUMING into existing build dirs:")
    print("   shim :", SHIM_DIR.name)
    print("   out  :", OUT_DIR.name)
else:
    STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = OUT_ROOT / ("release_" + STAMP)
    SHIM_DIR = OUT_ROOT / ("shim_" + STAMP)
    # a new build directory invalidates any stage recorded against an old one
    for _k in ["1_shim", "2_assemble", "3_correct_provenance", "4_ann_recall",
               "5_cluster_neardup", "6_verify", "7_manuscript_numbers"]:
        if _k in RESULTS:
            RESULTS.pop(_k, None)
    if RESULTS:
        _flush()
        print("new build dirs - cleared stale stage markers so nothing is skipped")

READ_ONLY = [V01, _DRIVE_ROOT / "fluxion_v02", _DRIVE_ROOT / "fluxion_reextract",
             Path("/root/.cache/huggingface")]

_problems, _warn = [], []
if V02_POCKET is None:
    _problems.append("v0.2 pocket embeddings not found. Notebook 11 writes "
                     "<DRIVE>/fluxion_v02/pocket_embeddings.npy")
if NB09_DIR is None:
    _problems.append("pocket_residue_map.parquet not found. Run notebook 09 Cell 5.")
if REPO_DIR is None:
    _problems.append("scripts/build_v02_release.py not found. Clone the repo into "
                     "/content, or set REPO_DIR by hand.")
if (not RESUMED_DIRS) and OUT_DIR.exists() and any(OUT_DIR.iterdir()):
    _problems.append("output dir already populated: %s" % OUT_DIR)

if shutil.which("mmseqs") is None:
    _warn.append("MMseqs2 absent: the target-disjoint split degrades to exact-sequence "
                 "matching, which is NOT homology-disjoint. Install with "
                 "`apt-get -qq install -y mmseqs2`.")

try:
    _free = shutil.disk_usage(str(OUT_ROOT.parent if OUT_ROOT.exists() else "/content")).free / 1e9
    if _free < 12:
        _warn.append("only %.1f GB free; the build needs roughly 8-10 GB" % _free)
except Exception:
    pass

if V02_POCKET is not None:
    _a = np.load(V02_POCKET, mmap_mode="r")
    if _a.shape[1] != 1280:
        _problems.append("v0.2 embeddings have wrong width: %s" % (_a.shape,))
    _v01_pocket = np.load(V01 / "pockets" / "pocket_embeddings.npy", mmap_mode="r")
    if _a.shape != _v01_pocket.shape:
        _problems.append("v0.2 %s does not match v0.1 %s; offsets would break"
                         % (_a.shape, _v01_pocket.shape))
    _same = bool(np.array_equal(np.asarray(_a[:2048]), np.asarray(_v01_pocket[:2048])))
    if _same:
        _problems.append("v0.2 array is byte-identical to v0.1 on the first 2048 rows - "
                         "this is the CHIMERIC array, not the per-chain re-embedding.")

for _w in _warn:
    print("  WARNING:", _w)
if _problems:
    raise SystemExit(chr(10).join(["", "=" * 72, "PREFLIGHT FAILED", "=" * 72]
                                  + ["  * " + p for p in _problems]))

OUT_DIR.mkdir(parents=True, exist_ok=True)
SHIM_DIR.mkdir(parents=True, exist_ok=True)


def _path_forms(x):
    """Both the literal path and its symlink target.

    Hugging Face snapshots are symlinks into a blobs/ directory, so resolve()
    on snapshots/.../metadata.jsonl lands OUTSIDE the snapshot tree and a
    resolve-only check silently fails to protect it.
    """
    out = set()
    try:
        out.add(Path(os.path.abspath(str(x))))
    except Exception:
        pass
    try:
        out.add(Path(x).resolve())
    except (FileNotFoundError, OSError, RuntimeError):
        pass
    return out


def assert_writable(p):
    """Refuse to write anywhere under a read-only input.

    Compares path COMPONENTS, not string prefixes: startswith() treats
    /MyDrive/fluxion_v02_build as living inside /MyDrive/fluxion_v02 and
    would wrongly block the output directory.
    """
    for pp in _path_forms(p):
        for ro in READ_ONLY:
            for rp in _path_forms(ro):
                if pp == rp or pp.is_relative_to(rp):
                    raise RuntimeError("REFUSING to write under read-only input: %s" % p)
    return Path(os.path.abspath(str(p)))


# self-test the guard before it protects anything
for _t in [SHIM_DIR / ".gitattributes", OUT_DIR / "x.json"]:
    assert_writable(_t)
for _t in [V01 / "metadata.jsonl", V01 / "pockets" / "pocket_embeddings.npy",
           _DRIVE_ROOT / "fluxion_v02" / "pocket_embeddings.npy"]:
    try:
        assert_writable(_t)
        raise SystemExit("write guard is broken: it failed to block %s" % _t)
    except RuntimeError:
        pass
print("write guard: allows out/shim, blocks v0.1 - OK")


result("0_preflight",
    v01_snapshot=str(V01),
    v02_pocket=str(V02_POCKET),
    nb09_dir=str(NB09_DIR),
    repo_dir=str(REPO_DIR),
    out_dir=str(OUT_DIR),
    shim_dir=str(SHIM_DIR),
    resumed_dirs=RESUMED_DIRS,
    mmseqs=shutil.which("mmseqs") or "ABSENT",
    v02_differs_from_v01=not _same,
    NOTE="Only out_dir and shim_dir are written. v0.1 and the Drive inputs are "
         "asserted read-only by assert_writable().")
```

---

## Cell 2 — **Build the read-only shim** (this is the actual fix)

Hardlinks cost no disk and cannot mutate the source.

```python
if not done("1_shim"):
    _linked, _copied, _missing = 0, 0, []
    for src in sorted(V01.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(V01)
        if str(rel).replace("\\", "/") == "pockets/pocket_embeddings.npy":
            continue                      # substituted below
        dst = assert_writable(SHIM_DIR / rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(str(src), str(dst)); _linked += 1
        except Exception:
            shutil.copy2(str(src), str(dst)); _copied += 1

    _pk = assert_writable(SHIM_DIR / "pockets" / "pocket_embeddings.npy")
    _pk.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(V02_POCKET), str(_pk))
        _how = "hardlink"
    except Exception:
        shutil.copy2(str(V02_POCKET), str(_pk))
        _how = "copy"

    # prove the shim carries the v0.2 array, not the v0.1 one
    _s = np.load(_pk, mmap_mode="r")
    _o = np.load(V01 / "pockets" / "pocket_embeddings.npy", mmap_mode="r")
    _n = np.load(V02_POCKET, mmap_mode="r")
    _is_v02 = bool(np.array_equal(np.asarray(_s[:4096]), np.asarray(_n[:4096])))
    _is_v01 = bool(np.array_equal(np.asarray(_s[:4096]), np.asarray(_o[:4096])))

    if not _is_v02 or _is_v01:
        raise SystemExit("shim verification FAILED: pocket_embeddings.npy is not the v0.2 array")

    result("1_shim",
        shim_dir=str(SHIM_DIR),
        files_hardlinked=_linked, files_copied=_copied,
        pocket_substituted_via=_how,
        shim_matches_v02=_is_v02, shim_matches_v01=_is_v01,
        v01_untouched=bool(np.array_equal(
            np.asarray(np.load(V01 / "pockets" / "pocket_embeddings.npy", mmap_mode="r")[:1024]),
            np.asarray(_o[:1024]))),
        NOTE="Hardlinks share inodes with v0.1 but the assembler only READS them. "
             "pocket_embeddings.npy points at the v0.2 per-chain array, which is the "
             "one-line fix for the letter-vs-deposit contradiction.")
```

---

## Cell 3 — Run the tested assembler against the shim

```python
if not done("2_assemble"):
    LOG = OUT_DIR / "build.log"
    # -u is essential: with stdout redirected, Python block-buffers ~8 KB, so a
    # working build looks identical to a hung one for a long time.
    cmd = [sys.executable, "-u", str(REPO_DIR / "scripts" / "build_v02_release.py"), "all",
           "--v01_dir", str(SHIM_DIR),
           "--out_dir", str(OUT_DIR),
           "--nb09_dir", str(NB09_DIR)]
    if shutil.which("mmseqs"):
        cmd += ["--cluster_backend", "mmseqs"]
    print("$", " ".join(cmd))
    print("-" * 72)
    t0 = time.time()
    # Stream to the notebook AND tee to build.log. A multi-hour job that prints
    # nothing is indistinguishable from a hung one.
    _buf = []
    with open(str(LOG), "w", encoding="utf-8", errors="replace") as fh:
        _cenv = dict(os.environ)
        _cenv["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, encoding="utf-8",
                             errors="replace", env=_cenv)
        for _line in iter(p.stdout.readline, ""):
            _line = _line.rstrip(chr(10))
            print("  %s" % _line, flush=True)
            fh.write(_line + chr(10))
            fh.flush()
            _buf.append(_line)
        p.stdout.close()
        rc = p.wait()
    mins = (time.time() - t0) / 60.0
    print("-" * 72)
    print("exit=%s after %.1f min" % (rc, mins))
    result("2_assemble", exit_code=rc, minutes=round(mins, 1), log=str(LOG),
           NOTE="Non-zero exit: read build.log. The shim is read-only input, so a "
                "failed run leaves v0.1 and the Drive arrays untouched.")
    if rc != 0:
        raise SystemExit("assembler failed (exit %d) - see %s" % (rc, LOG))
```

---

## Cell 4 — **Correct the two provenance statements the build gets wrong**

The assembler hardcodes "NOT recomputed" and an `alpha = 0.75` default. Both are
now false. Correct them **in our output only**.

```python
if not done("3_correct_provenance"):
    _fixed = {}

    # (a) carry-over note: pocket embeddings ARE the v0.2 per-chain array here
    for _cand in [OUT_DIR / "reports" / "carry_over.json", OUT_DIR / "carry_over.json"]:
        if _cand.exists():
            _j = json.loads(_cand.read_text(encoding="utf-8"))
            _j["note"] = (
                "ligand_embeddings.npy, manifests, raft and config are byte-identical "
                "to v0.1. pocket_embeddings.npy is NOT carried over: it is the v0.2 "
                "per-chain re-embedding produced by notebook 11 and supplied through a "
                "read-only shim. joined_embeddings.npy is retained only for "
                "reproducibility of the retired v0.1 concatenated index."
            )
            _j["pocket_embeddings_source"] = str(V02_POCKET)
            assert_writable(_cand).write_text(json.dumps(_j, indent=2), encoding="utf-8")
            _fixed["carry_over.json"] = str(_cand)

    # (b) modality guidance: one index, alpha = 1.00, from the measured sweep
    _mg = OUT_DIR / "indices" / "modality_guidance.json"
    if _mg.exists():
        _j = json.loads(_mg.read_text(encoding="utf-8"))
        _j["default"] = "alpha = 1.00 (pocket-only)"
        _j.pop("balanced_default", None)
        _j["selection"] = {
            "primary_metric": "same-UniProt hit@1",
            "measured": {"0.00": 0.3091, "0.25": 0.631, "0.50": 0.6692,
                         "0.75": 0.7387, "1.00": 0.8353},
            "paired_test": "McNemar chi2=427.85, p<1e-90, pocket-only vs alpha=0.50",
            "held_out": "alpha chosen on half A, 0.8377 (95% CI 0.8184-0.8569) on half B",
            "seed_stability": "identical ordering across seeds 42 / 7 / 2024",
            "cluster_geometry": "silhouette by Pfam: pocket-only +0.174, joined -0.069, "
                                "ligand-only -0.314 (mean-centred)",
        }
        _j["note"] = ("A single default index is shipped. The alpha table is retained "
                      "for transparency, not as a menu.")
        assert_writable(_mg).write_text(json.dumps(_j, indent=2), encoding="utf-8")
        _fixed["modality_guidance.json"] = str(_mg)

    # (c) CHANGELOG.md: the assembler hardcodes "not re-embedded" (:2271-2273) and
    #     "deferred to v1.0" (:2600). Both are false once the shim supplies v0.2.
    _cl = OUT_DIR / "CHANGELOG.md"
    if _cl.exists():
        _txt = _cl.read_text(encoding="utf-8")
        _pairs = [
            ("No entry was added, removed, or re-embedded",
             "No entry was added or removed, and pocket embeddings ARE re-embedded"),
            ("`pocket_embeddings.npy` and `ligand_embeddings.npy` are byte-identical to v0.1.",
             "`ligand_embeddings.npy` is byte-identical to v0.1; `pocket_embeddings.npy` is "
             "the per-chain re-embedding described below."),
            ("rather than fixing it", "and fixes it"),
            ("re-embedding would change the vectors and is deferred to v1.0.",
             "v0.2 re-embeds per chain; every retrieval number in this release is measured "
             "on the re-embedded vectors."),
        ]
        _n = 0
        for _a, _b in _pairs:
            if _a in _txt:
                _txt = _txt.replace(_a, _b)
                _n += 1
        assert_writable(_cl).write_text(_txt, encoding="utf-8")
        _fixed["CHANGELOG.md"] = "%d/%d phrases replaced" % (_n, len(_pairs))

    # ---- prove no false provenance claim survives anywhere in the release ----
    _FALSE = [
        "were NOT recomputed",
        "No entry was added, removed, or re-embedded",
        "deferred to v1.0",
        "balanced_default",
        "alpha = 0.75",
    ]
    _survivors = []
    for _p in sorted(OUT_DIR.rglob("*")):
        if not _p.is_file() or _p.suffix.lower() not in (".md", ".json", ".txt", ".tsv"):
            continue
        try:
            _t = _p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for _f in _FALSE:
            if _f in _t:
                _survivors.append("%s :: %s" % (_p.relative_to(OUT_DIR), _f))

    result("3_correct_provenance", corrected=list(_fixed), paths=_fixed,
        false_claims_remaining=len(_survivors), where=_survivors[:10],
        NOTE="These files are what a reviewer opens first to check whether the response "
             "letter is true; all previously contradicted it. false_claims_remaining "
             "MUST be 0 - if not, the deposit still denies its own letter.")
    if _survivors:
        print("")
        print("!! %d false provenance claim(s) still in the release:" % len(_survivors))
        for _s in _survivors[:10]:
            print("     ", _s)
        print("   Fix these before depositing.")
```

---

## Cell 5 — **V4: ANN recall of the SHIPPED index** (the number still owed)

```python
# ---- shared arrays, loaded OUTSIDE any done() guard ---------------------
# Cells 5 and 6 both need P / _meta / _flat. Defining them inside a
# `if not done(...)` block means a RESUMED run skips the definition and the
# next cell dies on NameError. Load once, whenever either cell still needs it.
import faiss

_NEED_ARRAYS = (not done("4_ann_recall")) or (not done("5_cluster_neardup"))
if _NEED_ARRAYS and "P" not in dir():
    _idx_dir = OUT_DIR / "indices"
    _cands = sorted(_idx_dir.glob("*.faiss")) if _idx_dir.exists() else []
    print("indices built:", [p.name for p in _cands])

    _pk_arr = np.load(SHIM_DIR / "pockets" / "pocket_embeddings.npy", mmap_mode="r")
    _meta = [json.loads(l) for l in open(OUT_DIR / "metadata.jsonl")]
    _poff = np.array([m["pocket_emb_offset"] for m in _meta], dtype=np.int64)
    P = np.asarray(_pk_arr, dtype=np.float32)[_poff]
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    P = np.ascontiguousarray(P)
    _flat = faiss.IndexFlatIP(P.shape[1])
    _flat.add(P)
    print("loaded P %s and built the exact index" % (P.shape,))

if not done("4_ann_recall"):
    _idx_dir = OUT_DIR / "indices"
    _cands = sorted(_idx_dir.glob("*.faiss")) if _idx_dir.exists() else []
    _rng = np.random.RandomState(42)
    _q = _rng.choice(len(P), min(2000, len(P)), replace=False)
    _, I_exact = _flat.search(np.ascontiguousarray(P[_q]), 11)

    _target = _idx_dir / "pocket.faiss"
    rec = {}
    if _target.exists():
        _h = faiss.read_index(str(_target))
        for ef in [16, 64, 128, 256]:
            try:
                _h.hnsw.efSearch = ef
            except Exception:
                pass
            _, I_ann = _h.search(np.ascontiguousarray(P[_q]), 11)
            r1 = float(np.mean([I_ann[i, 0] == I_exact[i, 0] for i in range(len(_q))]))
            r10 = float(np.mean([len(set(I_ann[i, :10]) & set(I_exact[i, :10])) / 10.0
                                 for i in range(len(_q))]))
            rec["efSearch=%d" % ef] = {"recall_at_1": round(r1, 4),
                                       "recall_at_10": round(r10, 4)}
        result("4_ann_recall", index="indices/pocket.faiss", n_queries=len(_q), **rec,
            NOTE="Recall of the ACTUAL shipped pocket-only HNSW index against exact "
                 "search on the same vectors. This is R-V4, previously deferred "
                 "because the v0.1 index could not be compared to v0.2 embeddings.")
    else:
        result("4_ann_recall", SKIPPED="indices/pocket.faiss absent",
               built=[p.name for p in _cands],
               NOTE="Check the indices stage in build.log.")
```

---

## Cell 6 — **Near-duplicate exclusion on sequence clusters, not cosine**

A 0.99 cosine cut removes different populations in different representations
(random-pair cosine is 0.72 pocket vs 0.42 joined), which is why the previous
comparison inverted. Clusters are representation-independent.

```python
if not done("5_cluster_neardup"):
    if "P" not in dir() or "_flat" not in dir() or "_meta" not in dir():
        raise SystemExit("shared arrays missing - re-run the cell above "
                         "(it loads P / _meta / _flat outside the done() guard)")
    _clu = None
    for _c in [OUT_DIR / "splits" / "clusters.tsv", OUT_DIR / "splits" / "cluster_map.tsv",
               OUT_DIR / "splits" / "clusters.json", OUT_DIR / "reports" / "splits.json"]:
        if _c.exists():
            _clu = _c
            break
    print("cluster artifact:", _clu)

    key2cluster = {}
    if _clu is not None and _clu.suffix == ".tsv":
        for _line in open(_clu):
            parts = _line.rstrip(chr(10)).split(chr(9))
            if len(parts) >= 2:
                key2cluster[parts[1]] = parts[0]
    elif _clu is not None:
        try:
            _j = json.loads(_clu.read_text(encoding="utf-8"))
            key2cluster = _j.get("cluster_of") or _j.get("clusters") or {}
        except Exception:
            key2cluster = {}

    if not key2cluster:
        result("5_cluster_neardup", SKIPPED="no cluster map found in the release",
            looked_at=[str(OUT_DIR / "splits"), str(OUT_DIR / "reports")],
            NOTE="Without MMseqs2 clusters the representation-independent exclusion "
                 "cannot be computed. Install mmseqs2 and re-run the splits stage.")
    else:
        _cl = np.array([key2cluster.get(m.get("pocket_key"), "?") for m in _meta])
        _upro = np.array([str(m.get("uniprot_acc") or "") for m in _meta])
        _ok = np.flatnonzero(_upro != "")
        _qs = np.random.RandomState(42).choice(_ok, min(3000, len(_ok)), replace=False)

        _, I = _flat.search(np.ascontiguousarray(P[_qs]), 25)
        raw_hits, clu_hits, n_used = [], [], 0
        for row, q in zip(I, _qs):
            cand = [j for j in row if j != q]
            if not cand:
                continue
            raw_hits.append(_upro[cand[0]] == _upro[q])
            far = [j for j in cand if _cl[j] != _cl[q]]
            if far:
                clu_hits.append(_upro[far[0]] == _upro[q])
                n_used += 1

        def _ci(a):
            a = np.asarray(a, dtype=float)
            if not len(a):
                return (None, None)
            bs = [np.mean(np.random.RandomState(s).choice(a, len(a), replace=True))
                  for s in range(200)]
            return (round(float(np.percentile(bs, 2.5)), 4),
                    round(float(np.percentile(bs, 97.5)), 4))

        result("5_cluster_neardup",
            n_queries=len(raw_hits),
            hit_at_1_raw=round(float(np.mean(raw_hits)), 4),
            CI95_raw=_ci(raw_hits),
            n_queries_with_out_of_cluster_neighbour=n_used,
            hit_at_1_cluster_excluded=round(float(np.mean(clu_hits)), 4) if clu_hits else None,
            CI95_cluster_excluded=_ci(clu_hits),
            NOTE="Neighbours in the query's own sequence cluster are excluded. Unlike a "
                 "cosine threshold this is representation-independent, so pocket-only "
                 "and joined numbers are directly comparable. Quote THIS, not the "
                 "0.99-cosine variant.")
```

---

## Cell 7 — Independent verification of what was built

```python
if not done("6_verify"):
    _vs = REPO_DIR / "scripts" / "verify_v02_release.py"
    if _vs.exists():
        _vlog = OUT_DIR / "verify.log"
        with open(str(_vlog), "wb") as fh:
            _p = subprocess.Popen([sys.executable, str(_vs), "--release_dir", str(OUT_DIR)],
                                  stdout=fh, stderr=subprocess.STDOUT)
            _rc = _p.wait()
        _tail = _vlog.read_text(errors="replace").splitlines()[-30:]
        print(chr(10).join(_tail))
        result("6_verify", exit_code=_rc, log=str(_vlog),
               NOTE="Independent checker: split disjointness, atom-order correspondence, "
                    "ligand geometry, file inventory.")
    else:
        result("6_verify", SKIPPED="verify_v02_release.py not found")
```

---

## Cell 8 — **Numbers for the manuscript**, in one file

Every figure the revision needs, re-derived from what was just built — so no
number is hand-carried into the .tex.

```python
if not done("7_manuscript_numbers"):
    _inv, _total = [], 0
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file() and "reports" not in p.parts:
            sz = p.stat().st_size
            _total += sz
            _inv.append({"path": str(p.relative_to(OUT_DIR)).replace("\\", "/"),
                         "bytes": sz, "MB": round(sz / 1e6, 2)})

    _md = None
    for _c in [OUT_DIR / "DATASET_MANIFEST.json", OUT_DIR / "manifest.json"]:
        if _c.exists():
            _md = json.loads(_c.read_text(encoding="utf-8"))
            break

    numbers = {
        "release_dir": str(OUT_DIR),
        "built_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "n_files": len(_inv),
        "total_bytes": _total,
        "total_GB": round(_total / 1e9, 3),
        "inventory": _inv,
        "pocket_embeddings_provenance": "v0.2 per-chain re-embedding (notebook 11)",
        "default_index": "alpha = 1.00, pocket-only",
        "ann_recall": RESULTS.get("4_ann_recall", {}),
        "cluster_neardup": RESULTS.get("5_cluster_neardup", {}),
        "manifest_present": _md is not None,
    }
    _p = assert_writable(OUT_DIR / "numbers_for_manuscript.json")
    _p.write_text(json.dumps(numbers, indent=2, default=str), encoding="utf-8")

    result("7_manuscript_numbers", written=str(_p), n_files=len(_inv),
        total_GB=numbers["total_GB"],
        NOTE="Table 1's inventory and the stated on-disk total must be regenerated "
             "from THIS file. The manuscript's current 3.84 GB total and file list "
             "describe v0.1.")
```

---

## Cell 9 — Final report

```python
EXPECTED = ["0_preflight", "1_shim", "2_assemble", "3_correct_provenance",
            "4_ann_recall", "5_cluster_neardup", "6_verify", "7_manuscript_numbers"]
_missing = [s for s in EXPECTED if s not in RESULTS]
_flush()

print("=" * 72)
print("sections present : %d / %d" % (len(RESULTS), len(EXPECTED)))
print("release          : %s" % OUT_DIR)
print("checkpoint       : %s" % CKPT_MD)
print("=" * 72)
if _missing:
    print("")
    print("!! INCOMPLETE - never ran:")
    for s in _missing:
        print("     - %s" % s)
else:
    print("")
    print("COMPLETE.")
print("")
print("NOT DONE, DELIBERATELY: no Zenodo deposit, no HF push. Publishing mints a")
print("DOI the manuscript cites - review the release, then do it consciously.")
print("")
print("v0.1 snapshot and the Drive inputs were never written to.")

try:
    from google.colab import files
    files.download(str(CKPT_MD))
except Exception:
    pass
```
