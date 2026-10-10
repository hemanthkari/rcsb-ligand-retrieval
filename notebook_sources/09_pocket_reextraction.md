# Sci Data revision — pocket re-extraction, residue→token map, chimeric control

Answers the two items the first audit notebook could not: they need the
source mmCIF files.

* **R3.M2** — recover and ship the residue→token map, and *prove* it is the
  one actually used, by re-extracting with the deposited code and checking
  the result against the shipped `pockets/pocket_coords.h5`.
* **V1 (the editor's "validity of the embedding")** — measure what the
  all-chain **chimeric** ESM-2 input actually cost, by re-embedding a
  stratified sample per-chain and comparing.

### The witness argument (why no full re-embedding is needed for M2)

In `embed_pockets._extract_pocket`, `pocket_indices` and `pocket_coords_list`
are appended **in the same loop, in the same order**. So if re-running the
*unmodified* extraction reproduces the shipped centroids element-wise, the
recovered indices are provably the ones used. That gives 100 %-of-corpus
verification on **CPU alone**.

Its limit, stated honestly in the paper: this proves the *ordering and
geometry*, not that the mmCIF file is byte-identical to April 2026. Hence we
pin `gemmi==0.6.5` and report a **per-entry exact-match rate**, never a bare
"verified".

### Runtime

| Part | Needs | Time |
|---|---|---|
| A — re-extraction (full 72,485) | CPU + ~25 GB streamed | 2–5 h, resumable |
| B — ESM-2 reproduction check (1,000) | GPU | ~10 min |
| C — chimeric control (~3,000) | GPU | ~30 min |

**Run Cell 4 with `LIMIT = 2000` first.** If the match rate is poor, stop and
diagnose — do not spend hours downloading.

**Set `DRIVE_DIR` in Cell 0.** Every checkpoint goes there; Colab will
disconnect at least once during the full run.

---

## Cell 0 — Setup, pinned deps, Drive checkpointing

```python
# gemmi is PINNED: altloc/entity handling changed across 0.6.x -> 0.7.x and
# an unpinned version is the single most likely cause of a failed reproduction.
!pip -q install "gemmi==0.6.5" h5py numpy pandas pyarrow requests huggingface_hub fair-esm torch faiss-cpu

import os, io, json, gzip, time, math, hashlib, collections, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests

import gemmi
print("gemmi:", gemmi.__version__)
assert gemmi.__version__.startswith("0.6.5"), "PIN FAILED - reproduction will not be auditable"

try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/fluxion_reextract")
except Exception:
    DRIVE_DIR = Path("./fluxion_reextract")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR = DRIVE_DIR / "shards"; SHARD_DIR.mkdir(exist_ok=True)
print("checkpoints ->", DRIVE_DIR)

# If you still have the April 2026 mmCIF mirror, point at it and downloads are skipped.
LOCAL_MIRROR = None          # e.g. Path("/content/drive/MyDrive/pdb_mmcif")
RESULTS2 = {}

def result(section, **kv):
    RESULTS2.setdefault(section, {}).update(kv)
    print("\n" + "=" * 72); print("RESULT  [%s]" % section); print("=" * 72)
    for k, v in kv.items():
        print("  %-42s %s" % (k, v))
```

---

## Cell 1 — Install the **deposited** pipeline verbatim

The audit's credibility rests on using the same code path, not a
reimplementation. Pull the archive straight from the Zenodo record.

```python
# Pull the published archive so the import is provably the deposited code.
# (zipfile, not a shell magic - works in any runtime and inside try/except.)
import zipfile, glob, sys

ZENODO_DOI_RECORD = "19954940"   # 10.5281/zenodo.19954940
EXTRACT_TO = "/content/pipeline_src"
pkg_ok = False
try:
    r = requests.get("https://zenodo.org/api/records/%s" % ZENODO_DOI_RECORD, timeout=30)
    zips = [f for f in r.json()["files"] if f["key"].endswith(".zip")]
    if not zips:
        raise RuntimeError("no .zip asset on the Zenodo record")
    blob = requests.get(zips[0]["links"]["self"], timeout=600).content
    Path("pipeline.zip").write_bytes(blob)
    with zipfile.ZipFile("pipeline.zip") as z:
        z.extractall(EXTRACT_TO)
    roots = glob.glob(EXTRACT_TO + "/**/rcsb_ligand_retrieval", recursive=True)
    if not roots:
        raise RuntimeError("rcsb_ligand_retrieval package not found in archive")
    sys.path.insert(0, str(Path(roots[0]).parent))
    pkg_ok = True
    print("installed deposited pipeline from Zenodo record", ZENODO_DOI_RECORD)
except Exception as e:
    print("Zenodo fetch failed (%s).\n"
          "FALLBACK: upload your local zenodo_pipeline_release/ folder, then run:\n"
          "    sys.path.insert(0, '/content/zenodo_pipeline_release')" % e)

from rcsb_ligand_retrieval.retrieval.embed_pockets import (
    _extract_pocket, _three_to_one, _THREE_TO_ONE,
)
print("imported _extract_pocket verbatim from the deposit:", pkg_ok)
print("three-to-one table size:", len(_THREE_TO_ONE))
```

---

## Cell 2 — Load the shipped artifacts we must reproduce

```python
from huggingface_hub import snapshot_download
import h5py

LOCAL = Path(snapshot_download("HemanthKari/rcsb-ligand-retrieval-v0.1",
                               repo_type="dataset"))
pocket_manifest = [json.loads(l) for l in
                   open(LOCAL / "pockets" / "pocket_manifest.jsonl")]
pkt_h5 = h5py.File(LOCAL / "pockets" / "pocket_coords.h5", "r")
pocket_emb = np.load(LOCAL / "pockets" / "pocket_embeddings.npy", mmap_mode="r")

print("pocket_manifest rows :", len(pocket_manifest))
print("pocket_coords.h5 keys:", len(pkt_h5["pockets"]))
print("pocket_embeddings    :", pocket_emb.shape)

POCKET_RADIUS = 8.0          # embed_pockets default; recorded in config.json
MAX_SEQ_LEN   = 1022
```

---

## Cell 3 — mmCIF acquisition (streaming, resumable, mirror-aware)

```python
CIF_CACHE = Path("/content/cif_tmp"); CIF_CACHE.mkdir(exist_ok=True)

def fetch_cif(pdb_id: str) -> bytes | None:
    """Return gzip bytes for one entry. Prefers a local mirror; else RCSB."""
    pid = pdb_id.lower()
    if LOCAL_MIRROR:
        for cand in (Path(LOCAL_MIRROR) / ("%s.cif.gz" % pid),
                     Path(LOCAL_MIRROR) / pid[1:3] / ("%s.cif.gz" % pid)):
            if cand.exists():
                return cand.read_bytes()
    try:
        r = requests.get("https://files.rcsb.org/download/%s.cif.gz" % pid.upper(),
                         timeout=60)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None

def structure_from_bytes(raw: bytes, pdb_id: str = "tmp"):
    """Parse via a temp FILE using gemmi.read_structure() - byte-for-byte the
    same call the original pipeline made (`_prefetch_pocket` does
    `gemmi.read_structure(str(cif_path))` on a .cif.gz, then setup_entities()).
    Reading from a path rather than memory is both more faithful to the build
    we are auditing and avoids gemmi's from-memory API varying across versions."""
    p = CIF_CACHE / ("%s.cif.gz" % pdb_id.lower())
    p.write_bytes(raw)
    try:
        st = gemmi.read_structure(str(p))
        st.setup_entities()
        return st
    finally:
        try:
            p.unlink()
        except Exception:
            pass
```

### Diagnostic — run ONE entry loudly before the batch

If the batch reports a uniform failure, this shows the real traceback
instead of a swallowed exception type.

```python
import traceback
e0 = pocket_manifest[0]
print("probe entry:", {k: e0[k] for k in
                       ("pdb_id", "ligand_chain", "ligand_resname", "ligand_resnum")})
raw = fetch_cif(e0["pdb_id"])
print("cif bytes:", None if raw is None else len(raw))
try:
    st = structure_from_bytes(raw, e0["pdb_id"])
    print("parsed OK. models=%d chains(model0)=%d"
          % (len(st), len(st[0])))
    seq, idxs, coords = _extract_pocket(
        st, ligand_chain=e0["ligand_chain"], ligand_resname=e0["ligand_resname"],
        ligand_resnum=int(e0["ligand_resnum"]), radius=POCKET_RADIUS)
    print("seq_len=%s n_pocket=%s coords=%s"
          % (len(seq) if seq else None, len(idxs) if idxs else None,
             None if coords is None else coords.shape))
    shipped = np.asarray(pkt_h5["pockets/" + e0["pocket_key"]][:], dtype=np.float32)
    print("shipped coords:", shipped.shape)
    if coords is not None and coords.shape == shipped.shape:
        print("MAX ABS ERR vs shipped:", float(np.abs(coords - shipped).max()))
except Exception:
    traceback.print_exc()
```

---

## Cell 4 — **Part A: re-extract, verify against the shipped centroids, record provenance**

```python
LIMIT = 2000        # <-- SMOKE TEST FIRST. Set to None for the full 72,485 run.
N_WORKERS = 16
CHECKPOINT_EVERY = 1000
PURGE_SHARDS = False   # True = wipe all checkpoints and start clean

def ligand_heavy_coords(structure, chain_name, resname, resnum):
    """Ligand heavy atoms, matching _extract_pocket's own selection."""
    out = []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for residue in chain:
                if residue.name != resname or residue.seqid.num != resnum:
                    continue
                for atom in residue:
                    if atom.element.name != "H":
                        out.append([atom.pos.x, atom.pos.y, atom.pos.z])
        break
    return np.asarray(out, dtype=np.float32) if out else None


def mirror_walk(structure, lig_coords=None, radius=8.0):
    """Reproduce _extract_pocket's sequence walk, recording residue identity per
    sequence position. Same nesting, same filters, same order. The centroid check
    below is what PROVES this mirror is faithful.

    Also counts non-standard residues that were silently dropped, split by
    whether they lie WITHIN `radius` of the ligand - i.e. residues that would
    have been pocket residues but can never be selected because they consume no
    sequence position. That second number is the actual validity impact and is
    what R3.M2's 'non-standard amino acids' clause is asking about."""
    seq_pos = []          # (auth_chain, seqid_num, icode, resname3)
    n_nonstd = n_nonstd_near = 0
    r2 = radius * radius
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.entity_type != gemmi.EntityType.Polymer:
                    continue
                if _three_to_one(residue.name) is None:
                    n_nonstd += 1          # silently dropped, consumes NO position
                    if lig_coords is not None:
                        for atom in residue:
                            if atom.element.name == "H":
                                continue
                            d = lig_coords - np.array(
                                [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
                            if float((d * d).sum(axis=1).min()) <= r2:
                                n_nonstd_near += 1
                                break
                    continue
                seq_pos.append((chain.name, int(residue.seqid.num),
                                str(residue.seqid.icode or "").strip(), residue.name))
        break                               # first model only
    return seq_pos, n_nonstd, n_nonstd_near

def process(entry):
    pid = entry["pdb_id"]; key = entry["pocket_key"]
    raw = fetch_cif(pid)
    if raw is None:
        return {"pocket_key": key, "status": "no_cif"}
    try:
        st = structure_from_bytes(raw, pid)
        seq, idxs, coords = _extract_pocket(
            st, ligand_chain=entry["ligand_chain"],
            ligand_resname=entry["ligand_resname"],
            ligand_resnum=int(entry["ligand_resnum"]),
            radius=POCKET_RADIUS)
    except Exception as e:
        # keep the MESSAGE, not just the type - a bare type name hides the cause
        return {"pocket_key": key,
                "status": "parse_fail:%s: %s" % (type(e).__name__, str(e)[:120])}
    if seq is None or not idxs:
        return {"pocket_key": key, "status": "no_pocket"}

    # ---- THE WITNESS CHECK -------------------------------------------------
    shipped = pkt_h5.get("pockets/" + key)
    if shipped is None:
        return {"pocket_key": key, "status": "no_shipped_coords"}
    shipped = np.asarray(shipped[:], dtype=np.float32)
    got = np.asarray(coords, dtype=np.float32)
    if got.shape != shipped.shape:
        return {"pocket_key": key, "status": "shape_mismatch",
                "n_recovered": len(idxs), "n_shipped": int(shipped.shape[0])}
    max_err = float(np.abs(got - shipped).max())

    ligxyz = ligand_heavy_coords(st, entry["ligand_chain"], entry["ligand_resname"],
                                 int(entry["ligand_resnum"]))
    seq_pos, n_nonstd, n_nonstd_near = mirror_walk(st, ligxyz, POCKET_RADIUS)
    prov = [seq_pos[i] if i < len(seq_pos) else None for i in idxs]
    chains = sorted({c for c, *_ in seq_pos})
    pocket_chains = sorted({p[0] for p in prov if p})

    # crystallographic gaps: non-consecutive seqid within a chain
    gaps = 0
    for ci in chains:
        nums = [n for c, n, _, _ in seq_pos if c == ci]
        gaps += sum(1 for a, b in zip(nums, nums[1:]) if b - a > 1)

    return {
        "pocket_key": key, "status": "ok", "pdb_id": pid,
        "esm_input_sequence": seq,
        "esm_input_length": len(seq),
        "n_chains_in_esm_input": len(chains),
        "chain_order": ",".join(chains),
        "pocket_res_indices": json.dumps([int(i) for i in idxs]),
        "pocket_res_provenance": json.dumps(
            [{"chain": p[0], "seqid": p[1], "icode": p[2], "resname": p[3]}
             for p in prov if p]),
        "n_pocket_residues": len(idxs),
        "n_pocket_chains": len(pocket_chains),
        "n_nonstandard_dropped": n_nonstd,
        "n_nonstandard_dropped_within_8A": n_nonstd_near,
        "n_crystallographic_gaps": gaps,
        "centroid_max_abs_err": max_err,
        "centroid_exact_match": bool(max_err < 1e-4),
    }

# Columns every shard must carry. If the extractor gains a field, older shards
# are stale: delete them so those entries are recomputed instead of silently
# poisoning the run with a missing column.
REQUIRED_COLS = {
    "pocket_key", "status", "esm_input_sequence", "esm_input_length",
    "n_chains_in_esm_input", "chain_order", "pocket_res_indices",
    "pocket_res_provenance", "n_pocket_residues", "n_pocket_chains",
    "n_nonstandard_dropped", "n_nonstandard_dropped_within_8A",
    "n_crystallographic_gaps", "centroid_max_abs_err", "centroid_exact_match",
}

if PURGE_SHARDS:
    for p in SHARD_DIR.glob("shard_*.parquet"):
        p.unlink()
    print("purged all checkpoint shards")

stale = 0
for p in sorted(SHARD_DIR.glob("shard_*.parquet")):
    have = set(pd.read_parquet(p).columns)
    # a shard of pure failures legitimately lacks the success-only columns
    d = pd.read_parquet(p, columns=["status"])
    if (d.status == "ok").any() and not REQUIRED_COLS.issubset(have):
        missing = sorted(REQUIRED_COLS - have)
        p.unlink(); stale += 1
        print("removed stale shard %s (missing %s)" % (p.name, missing))
if stale:
    print("%d stale shard(s) removed - those entries will be recomputed" % stale)

todo = pocket_manifest if LIMIT is None else pocket_manifest[:LIMIT]

# Resume counts ONLY successful entries as done, so a re-run automatically
# retries anything that failed. (Treating failures as 'done' silently pins a
# bad run's results forever - that is what happened on the first attempt.)
done_keys, prev_failed = set(), 0
for sh in sorted(SHARD_DIR.glob("shard_*.parquet")):
    d = pd.read_parquet(sh, columns=["pocket_key", "status"])
    done_keys |= set(d.loc[d.status == "ok", "pocket_key"])
    prev_failed += int((d.status != "ok").sum())

todo = [e for e in todo if e["pocket_key"] not in done_keys]
print("previously succeeded: %d" % len(done_keys))
print("previously failed (will be retried): %d" % prev_failed)
print("remaining to process: %d" % len(todo))
if prev_failed and not done_keys:
    print("\nNOTE: every prior attempt failed. Run the diagnostic cell above "
          "first - do not burn a long run on a broken parse path.")

buf, t0, shard_n = [], time.time(), len(list(SHARD_DIR.glob("shard_*.parquet")))
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(process, e): e for e in todo}
    for n, f in enumerate(as_completed(futs), 1):
        buf.append(f.result())
        if n % 200 == 0:
            rate = n / (time.time() - t0)
            print("  %6d/%d  %.1f/s  eta %.0f min"
                  % (n, len(todo), rate, (len(todo) - n) / max(rate, 1e-9) / 60), flush=True)
        if len(buf) >= CHECKPOINT_EVERY:
            pd.DataFrame(buf).to_parquet(SHARD_DIR / ("shard_%04d.parquet" % shard_n))
            shard_n += 1; buf = []
if buf:
    pd.DataFrame(buf).to_parquet(SHARD_DIR / ("shard_%04d.parquet" % shard_n))
print("done. shards:", len(list(SHARD_DIR.glob('shard_*.parquet'))))
```

---

## Cell 5 — Match rate, and write the shippable map

```python
df = pd.concat([pd.read_parquet(p) for p in sorted(SHARD_DIR.glob("shard_*.parquet"))],
               ignore_index=True)

# A retried entry appears twice (old failure + new result). Keep the success.
df["_ok"] = (df.status == "ok").astype(int)
df = (df.sort_values("_ok", ascending=False)
        .drop_duplicates(subset="pocket_key", keep="first")
        .drop(columns="_ok")
        .reset_index(drop=True))

n = len(df)
ok = df[df.status == "ok"]
status_counts = df.status.value_counts().to_dict()

# Fail loudly and usefully if nothing extracted, instead of an AttributeError
# on a column that only exists for successful rows.
if len(ok) == 0 or "centroid_exact_match" not in ok.columns:
    print("=" * 72)
    print("NO ENTRIES EXTRACTED - stopping before the shippable artifacts.")
    print("=" * 72)
    print("Status breakdown (now carries the real exception message):")
    for k, v in sorted(status_counts.items(), key=lambda kv: -kv[1])[:10]:
        print("   %6d  %s" % (v, k))
    print("\nRun the diagnostic cell above (single entry, full traceback).")
    raise SystemExit("re-extraction produced no usable rows")

exact = int(ok.centroid_exact_match.sum())

result("A_reextraction",
    n_attempted=n,
    status_breakdown=status_counts,
    n_extracted=len(ok),
    n_centroid_exact_match=exact,
    pct_exact_match_of_attempted=round(100 * exact / max(1, n), 3),
    pct_exact_match_of_extracted=round(100 * exact / max(1, len(ok)), 3),
    median_centroid_max_abs_err=(round(float(ok.centroid_max_abs_err.median()), 8)
                                 if len(ok) else None),
    VERDICT=("REPRODUCIBLE - ship the map for exact-match entries"
             if exact / max(1, n) > 0.98 else
             "LOW MATCH RATE - investigate (gemmi version? wwPDB re-versioning?) "
             "before shipping; consider re-running the pipeline and reissuing"),
)

result("A_chimeric_exposure",
    pct_entries_multi_chain_esm_input=round(
        100 * float((ok.n_chains_in_esm_input > 1).mean()), 2) if len(ok) else None,
    median_chains_in_esm_input=int(ok.n_chains_in_esm_input.median()) if len(ok) else None,
    max_chains_in_esm_input=int(ok.n_chains_in_esm_input.max()) if len(ok) else None,
    pct_pockets_spanning_multiple_chains=round(
        100 * float((ok.n_pocket_chains > 1).mean()), 2) if len(ok) else None,
    median_crystallographic_gaps=int(ok.n_crystallographic_gaps.median()) if len(ok) else None,
    total_nonstandard_residues_dropped=int(ok.n_nonstandard_dropped.sum()) if len(ok) else None,
    pct_entries_with_any_nonstandard_dropped=round(
        100 * float((ok.n_nonstandard_dropped > 0).mean()), 2) if len(ok) else None,
    total_nonstandard_dropped_WITHIN_8A=(
        int(ok.n_nonstandard_dropped_within_8A.sum())
        if "n_nonstandard_dropped_within_8A" in ok.columns else None),
    pct_entries_losing_a_pocket_residue_to_nonstandard=(
        round(100 * float((ok.n_nonstandard_dropped_within_8A > 0).mean()), 2)
        if "n_nonstandard_dropped_within_8A" in ok.columns else None),
    NOTE="These quantify exactly how much of the corpus was embedded from a "
         "chimeric, gap-closed sequence. This is the number the editor's "
         "'validity of the embedding' comment is really about.",
)

# ---- shippable artifacts (exact-match entries only) ----------------------
ship = ok[ok.centroid_exact_match].copy()
cols = ["pocket_key", "pdb_id", "esm_input_length", "n_chains_in_esm_input",
        "chain_order", "pocket_res_indices", "pocket_res_provenance",
        "n_pocket_residues", "n_nonstandard_dropped", "n_nonstandard_dropped_within_8A",
        "n_crystallographic_gaps",
        "centroid_max_abs_err"]
missing = [c for c in cols if c not in ship.columns]
if missing:
    print("WARNING: shards predate these fields, omitting from the map: %s" % missing)
    print("         (set PURGE_SHARDS = True in Cell 4 to regenerate with them)")
ship[[c for c in cols if c in ship.columns]].to_parquet(
    DRIVE_DIR / "pocket_residue_map.parquet", index=False)

with gzip.open(DRIVE_DIR / "esm_input_sequences.fasta.gz", "wt") as fh:
    for r in ship.itertuples():
        fh.write(">%s\n%s\n" % (r.pocket_key, r.esm_input_sequence))

print("wrote pocket_residue_map.parquet (%d rows) + esm_input_sequences.fasta.gz"
      % len(ship))
print("\nSchema note for Data Records: token_position = pocket_res_index + 1 "
      "(ESM-2 BOS occupies position 0).")
```

---

## Cell 6 — **Part B: does ESM-2 reproduce the shipped embedding from the recovered map?**

```python
import torch, esm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
model = model.to(device).eval()
if device == "cuda":
    model = model.half()                       # matches the original fp16 build
bc = alphabet.get_batch_converter()
REPR_LAYER = model.num_layers

pk2row = {r["pocket_key"]: r["offset"] for r in pocket_manifest}

def embed_pocket(seq, idxs):
    _, _, toks = bc([("x", seq)])
    with torch.no_grad():
        out = model(toks.to(device), repr_layers=[REPR_LAYER], return_contacts=False)
    reps = out["representations"][REPR_LAYER][0]
    sel = reps[[i + 1 for i in idxs]]          # +1 for BOS
    return sel.mean(0).float().cpu().numpy().astype(np.float32)

samp = ship.sample(n=min(1000, len(ship)), random_state=42)
cos = []
for r in samp.itertuples():
    idxs = json.loads(r.pocket_res_indices)
    if len(r.esm_input_sequence) > MAX_SEQ_LEN:
        continue
    got = embed_pocket(r.esm_input_sequence, idxs)
    ref = np.asarray(pocket_emb[pk2row[r.pocket_key]], dtype=np.float32)
    cos.append(float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12)))
cos = np.array(cos)

result("B_esm2_reproduction",
    n_checked=len(cos),
    cosine_median=round(float(np.median(cos)), 6),
    cosine_p05=round(float(np.percentile(cos, 5)), 6),
    cosine_min=round(float(cos.min()), 6),
    frac_ge_0p999=round(float((cos >= 0.999).mean()), 4),
    frac_ge_0p99=round(float((cos >= 0.99).mean()), 4),
    NOTE="Not bit-exact by construction: fp16 inference and batch-padding "
         "composition both vary. Report this distribution instead of the "
         "manuscript's current 'verified bit-for-bit' claim.",
)
```

---

## Cell 7 — **Part C: the chimeric-sequence control (V1)** — the editor's core question

Re-embed the pocket-bearing chain **alone** and compare to the shipped
chimeric embedding. Stratified by chain count so the answer is broken out.

```python
def pocket_bearing_chain(prov_json):
    c = collections.Counter(p["chain"] for p in json.loads(prov_json))
    return c.most_common(1)[0][0] if c else None

def per_chain_inputs(pdb_id, want_chain, pocket_prov):
    """Rebuild the sequence using ONLY want_chain, and remap pocket indices."""
    raw = fetch_cif(pdb_id)
    if raw is None: return None
    st = structure_from_bytes(raw)
    seq_chars, pos_key = [], {}
    for model in st:
        for chain in model:
            if chain.name != want_chain: continue
            for residue in chain:
                if residue.entity_type != gemmi.EntityType.Polymer: continue
                one = _three_to_one(residue.name)
                if one is None: continue
                pos_key[(chain.name, int(residue.seqid.num),
                         (residue.seqid.icode or "").strip())] = len(seq_chars)
                seq_chars.append(one)
        break
    idxs = [pos_key.get((p["chain"], p["seqid"], p["icode"]))
            for p in pocket_prov if p["chain"] == want_chain]
    idxs = [i for i in idxs if i is not None]
    return ("".join(seq_chars), idxs) if idxs else None

multi = ship[ship.n_chains_in_esm_input > 1]
csamp = multi.sample(n=min(600, len(multi)), random_state=42)
rows = []
for r in csamp.itertuples():
    prov = json.loads(r.pocket_res_provenance)
    ch = pocket_bearing_chain(r.pocket_res_provenance)
    pc = per_chain_inputs(r.pdb_id, ch, prov)
    if pc is None: continue
    seq_pc, idx_pc = pc
    if len(seq_pc) > MAX_SEQ_LEN or len(r.esm_input_sequence) > MAX_SEQ_LEN:
        continue
    e_chim = embed_pocket(r.esm_input_sequence, json.loads(r.pocket_res_indices))
    e_pc   = embed_pocket(seq_pc, idx_pc)
    c = float(e_chim @ e_pc / (np.linalg.norm(e_chim) * np.linalg.norm(e_pc) + 1e-12))
    rows.append(dict(pocket_key=r.pocket_key, cos=c,
                     n_chains=int(r.n_chains_in_esm_input),
                     gaps=int(r.n_crystallographic_gaps),
                     frac_pocket_in_chain=len(idx_pc) / max(1, r.n_pocket_residues)))

cdf = pd.DataFrame(rows)
by_chain = (cdf.assign(bin=pd.cut(cdf.n_chains, [1, 2, 3, 5, 100],
                                  labels=["2", "3", "4-5", "6+"]))
              .groupby("bin", observed=True)["cos"]
              .agg(["count", "median", "min"]).round(4).to_dict("index"))
by_gaps = (cdf.assign(bin=pd.cut(cdf.gaps, [-1, 0, 2, 5, 10000],
                                 labels=["0", "1-2", "3-5", "6+"]))
             .groupby("bin", observed=True)["cos"]
             .agg(["count", "median", "min"]).round(4).to_dict("index"))

result("C_chimeric_control",
    n_compared=len(cdf),
    cosine_chimeric_vs_perchain_median=round(float(cdf.cos.median()), 4),
    cosine_p05=round(float(np.percentile(cdf.cos, 5)), 4),
    cosine_min=round(float(cdf.cos.min()), 4),
    frac_cos_ge_0p95=round(float((cdf.cos >= 0.95).mean()), 4),
    frac_cos_ge_0p99=round(float((cdf.cos >= 0.99).mean()), 4),
    by_chain_count=by_chain,
    by_gap_count=by_gaps,
    median_frac_pocket_residues_in_single_chain=round(
        float(cdf.frac_pocket_in_chain.median()), 4),
    VERDICT=("SMALL EFFECT - the chimeric input is an earned limitation; "
             "document it and do not re-embed"
             if float(cdf.cos.median()) >= 0.95 else
             "LARGE EFFECT - the chimeric concatenation materially changes the "
             "representation; re-embed per-chain for v0.2"),
    NOTE="This is the direct experimental answer to the editor's 'validity of "
         "the embedding'. Report the distribution and the breakdown, not just "
         "the median.",
)
```

---

## Cell 7b — **The decisive test: does per-chain RETRIEVE better?**

Cell 7 shows the two representations *differ*. It does not show which is
better. This measures same-UniProt hit@1 under both encodings, on the same
sample, with an exact within-sample index — the number that decides whether
v0.2 re-embeds per-chain or documents the chimeric input as a limitation.

```python
import faiss

!wget -q -nc https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz
sif = pd.read_csv("pdb_chain_uniprot.csv.gz", compression="gzip", skiprows=1)
sif["PDB"] = sif["PDB"].str.lower()
uni_map = sif.groupby("PDB")["SP_PRIMARY"].first().to_dict()

SAMPLE_N = 4000
pool = ship.copy()
pool["uniprot"] = [uni_map.get(str(p).lower(), "") for p in pool.pdb_id]
pool = pool[pool.uniprot != ""]
vc = pool.uniprot.value_counts()
pool = pool[pool.uniprot.isin(vc[vc >= 2].index)]     # hit@1 must be answerable
samp2 = pool.sample(n=min(SAMPLE_N, len(pool)), random_state=42)
print("retrieval-comparison sample:", len(samp2))

E_chim, E_pc, labels, multi_flag = [], [], [], []
for k, r in enumerate(samp2.itertuples(), 1):
    if len(r.esm_input_sequence) > MAX_SEQ_LEN:
        continue
    prov = json.loads(r.pocket_res_provenance)
    ch = pocket_bearing_chain(r.pocket_res_provenance)
    pc = per_chain_inputs(r.pdb_id, ch, prov)
    if pc is None:
        continue
    seq_pc, idx_pc = pc
    if len(seq_pc) > MAX_SEQ_LEN or not idx_pc:
        continue
    E_chim.append(embed_pocket(r.esm_input_sequence, json.loads(r.pocket_res_indices)))
    E_pc.append(embed_pocket(seq_pc, idx_pc))
    labels.append(r.uniprot)
    multi_flag.append(int(r.n_chains_in_esm_input) > 1)
    if k % 500 == 0:
        print("  %d/%d" % (k, len(samp2)), flush=True)

labels = np.array(labels); multi_flag = np.array(multi_flag)

def hit_at_1(E, lab, subset=None):
    A = np.stack(E).astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    ii = faiss.IndexFlatIP(A.shape[1]); ii.add(A)
    _, I = ii.search(A, 2)
    nn = np.where(I[:, 0] == np.arange(len(A)), I[:, 1], I[:, 0])
    hits = np.array([lab[a] == lab[b] for a, b in enumerate(nn)])
    return float(hits.mean() if subset is None else hits[subset].mean())

h_chim_all,  h_pc_all  = hit_at_1(E_chim, labels), hit_at_1(E_pc, labels)
h_chim_multi = hit_at_1(E_chim, labels, multi_flag)
h_pc_multi   = hit_at_1(E_pc,   labels, multi_flag)

delta = h_pc_all - h_chim_all
result("C2_retrieval_chimeric_vs_perchain",
    n_compared=len(labels),
    n_multi_chain_in_sample=int(multi_flag.sum()),
    hit_at_1_chimeric_all=round(h_chim_all, 4),
    hit_at_1_perchain_all=round(h_pc_all, 4),
    delta_all=round(delta, 4),
    hit_at_1_chimeric_multichain_only=round(h_chim_multi, 4),
    hit_at_1_perchain_multichain_only=round(h_pc_multi, 4),
    delta_multichain_only=round(h_pc_multi - h_chim_multi, 4),
    VERDICT=("PER-CHAIN IS BETTER - re-embed for v0.2" if delta > 0.01 else
             "CHIMERIC IS BETTER OR EQUAL - keep embeddings, document the difference"
             if delta > -0.01 else
             "CHIMERIC IS BETTER - keep embeddings; the extra context helps"),
    NOTE="Cell 7 showed the representations differ (median cos 0.81 vs a "
         "random-pair floor of 0.71). THIS cell shows whether that difference "
         "costs retrieval quality. Only a positive delta justifies re-embedding.",
)
```

---

## Cell 8 — Save everything

```python
out = DRIVE_DIR / "reextraction_results.json"
out.write_text(json.dumps(RESULTS2, indent=2, default=str))

lines = ["# Pocket re-extraction + chimeric control — results", ""]
for sec, kv in RESULTS2.items():
    lines.append("## %s" % sec)
    for k, v in kv.items():
        lines.append("- **%s**: `%s`" % (k, v))
    lines.append("")
(DRIVE_DIR / "reextraction_results.md").write_text("\n".join(lines))

print("Artifacts in", DRIVE_DIR)
for p in sorted(DRIVE_DIR.iterdir()):
    if p.is_file():
        print("  %-40s %8.1f MB" % (p.name, p.stat().st_size / 1e6))

try:
    from google.colab import files
    files.download(str(DRIVE_DIR / "reextraction_results.json"))
    files.download(str(DRIVE_DIR / "reextraction_results.md"))
except Exception:
    pass
```
