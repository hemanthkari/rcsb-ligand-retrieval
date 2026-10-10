# v0.2 — per-chain pocket re-embedding

Corrects the construction defect established by notebook 09: the v0.1 pocket
embeddings were pooled from an ESM-2 input formed by concatenating **every**
polymer chain with no separator tokens.

Two measurements justify the change:

* same pocket, chimeric vs per-chain encoding → median cosine **0.814**, against
  a random-pair floor of **0.706**. The median entry has *all* its pocket
  residues inside one chain, so this is not dropped interface residues.
* same-UniProt hit@1 on 4,000 complexes → **0.588** chimeric vs **0.620**
  per-chain (**+3.25 pp**); multi-chain subset **0.539 → 0.582** (**+4.23 pp**).

**This notebook does better than the control experiment.** The control encoded
only the *pocket-bearing* chain and discarded pocket residues in other chains
(21.1% of pockets span chains). It won anyway. Here, **every chain contributing a
pocket residue is encoded separately, and each residue is pooled from its own
chain** — no fabricated adjacencies *and* no dropped residues.

### Correctness gate

For each pocket residue we recompute its index within its own chain and check
that the residue name there matches `pocket_residue_map.parquet` (already
verified against the released coordinates to 99.99%). Mismatches are **skipped
and counted, never pooled** — a wrong index does not crash, it silently yields a
plausible but wrong embedding, which is the exact failure class that produced
this revision.

### Row alignment

Output rows sit at the same offsets as the released
`pockets/pocket_embeddings.npy`, so metadata offsets, splits and manifests keep
working. Rows that fail retain their v0.1 value and are reported — do not
describe the release as fully per-chain unless that count is zero.

**Prerequisites:** `pocket_residue_map.parquet` from notebook 09, a GPU runtime,
and ideally the mmCIF mirror (set `CIF_DIR`) — without it this re-downloads.

**Run Cell 4 with `LIMIT = 200` first** to confirm the gate passes on real
structures before committing hours.

---

## Cell 0 — Setup

```python
!pip -q install "gemmi==0.6.5" numpy pandas pyarrow requests huggingface_hub fair-esm torch faiss-cpu

import os, json, time, gzip
from pathlib import Path
import numpy as np, pandas as pd, requests
import gemmi

print("gemmi:", gemmi.__version__)
assert gemmi.__version__.startswith("0.6.5"), "PIN FAILED"

try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive")
except Exception:
    DRIVE = Path(".")

RESIDUE_MAP = DRIVE / "fluxion_reextract" / "pocket_residue_map.parquet"
OUT_DIR     = DRIVE / "fluxion_v02";              OUT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR   = OUT_DIR / "embed_shards";           SHARD_DIR.mkdir(exist_ok=True)
CIF_DIR     = None      # e.g. DRIVE / "pdb_mmcif" -- set this if you kept the mirror

MAX_SEQ_LEN = 1022
RESULTS3 = {}

def result(section, **kv):
    RESULTS3.setdefault(section, {}).update(kv)
    print("\n" + "=" * 72); print("RESULT  [%s]" % section); print("=" * 72)
    for k, v in kv.items():
        print("  %-42s %s" % (k, v))

print("residue map:", RESIDUE_MAP, RESIDUE_MAP.exists())
```

---

## Cell 1 — Load the released artifacts and the verified residue map

```python
from huggingface_hub import snapshot_download

LOCAL = Path(snapshot_download("HemanthKari/rcsb-ligand-retrieval-v0.1",
                               repo_type="dataset"))
old_emb  = np.load(LOCAL / "pockets" / "pocket_embeddings.npy", mmap_mode="r")
manifest = [json.loads(l) for l in open(LOCAL / "pockets" / "pocket_manifest.jsonl")]
pk_to_off = {m["pocket_key"]: int(m["offset"]) for m in manifest}

rmap = pd.read_parquet(RESIDUE_MAP)
print("residue map rows :", len(rmap))
print("old embeddings   :", old_emb.shape)
print("manifest rows    :", len(manifest))
```

---

## Cell 2 — Per-chain sequence construction + the correctness gate

Mirrors the deposited `_extract_pocket` residue filter exactly (first model,
polymer only, 22-code table), but keeps chains separate.

```python
THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O",
}

def per_chain_sequences(structure):
    """{chain_name: (sequence, [(seqid, icode, resname3), ...])}"""
    out = {}
    for model in structure:
        for chain in model:
            chars, meta = [], []
            for residue in chain:
                if residue.entity_type != gemmi.EntityType.Polymer:
                    continue
                one = THREE_TO_ONE.get(residue.name.upper())
                if one is None:
                    continue          # non-standard: consumes NO position
                chars.append(one)
                meta.append((int(residue.seqid.num),
                             str(residue.seqid.icode or "").strip(),
                             residue.name))
            if chars:
                out[chain.name] = ("".join(chars), meta)
        break                          # first model only
    return out

def map_pocket_to_chains(provenance, chains):
    """-> ([(chain, idx_within_chain) | None], n_unresolved)"""
    lookup = {}
    for cname, (_seq, meta) in chains.items():
        for i, (num, icode, resname) in enumerate(meta):
            lookup[(cname, num, icode)] = (i, resname)
    assignments, unresolved = [], 0
    for p in provenance:
        key = (p["chain"], int(p["seqid"]), str(p.get("icode") or "").strip())
        hit = lookup.get(key)
        if hit is None:
            assignments.append(None); unresolved += 1; continue
        idx, resname = hit
        # GATE: residue at the recovered index must match the verified map
        if resname.upper() != str(p["resname"]).upper():
            assignments.append(None); unresolved += 1; continue
        assignments.append((p["chain"], idx))
    return assignments, unresolved

def load_structure(pdb_id):
    pid = pdb_id.lower()
    raw = None
    if CIF_DIR:
        for cand in (Path(CIF_DIR) / (pid + ".cif.gz"),
                     Path(CIF_DIR) / pid[1:3] / (pid + ".cif.gz")):
            if cand.exists():
                raw = cand.read_bytes(); break
    if raw is None:
        r = requests.get("https://files.rcsb.org/download/%s.cif.gz" % pid.upper(),
                         timeout=60)
        if r.status_code != 200:
            return None
        raw = r.content
    p = Path("/content/_tmp_%s.cif.gz" % pid)
    p.write_bytes(raw)
    try:
        st = gemmi.read_structure(str(p)); st.setup_entities(); return st
    except Exception:
        return None
    finally:
        try: p.unlink()
        except Exception: pass

# --- inline self-test of the gate, before touching real data ---------------
_built = {"A": ("AG", [(1,"","ALA"), (3,"","GLY")]),      # SEP at 2 dropped
          "B": ("VL", [(10,"","VAL"), (11,"A","LEU")])}
_a, _u = map_pocket_to_chains(
    [{"chain":"A","seqid":3,"icode":"","resname":"GLY"},
     {"chain":"B","seqid":11,"icode":"A","resname":"LEU"}], _built)
assert _u == 0 and _a == [("A",1), ("B",1)], (_a, _u)
_, _bad = map_pocket_to_chains(
    [{"chain":"A","seqid":3,"icode":"","resname":"TRP"}], _built)
assert _bad == 1
_, _miss = map_pocket_to_chains(
    [{"chain":"A","seqid":99,"icode":"","resname":"GLY"}], _built)
assert _miss == 1
print("gate self-test OK: correct mapping accepted; mismatched and missing rejected")
```

---

## Cell 3 — ESM-2 loader (fp16, matching the original build)

```python
import torch, esm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
model = model.to(DEVICE).eval()
if DEVICE.startswith("cuda"):
    model = model.half()
bc = alphabet.get_batch_converter()
LAYER = model.num_layers
print("ESM-2 ready: dim=%d layers=%d device=%s" % (model.embed_dim, LAYER, DEVICE))

def chain_reprs(seqs):
    """seqs: [(name, sequence)] -> {name: (L, D) float32}"""
    _, _, toks = bc(seqs)
    with torch.no_grad():
        out = model(toks.to(DEVICE), repr_layers=[LAYER], return_contacts=False)
    reps = out["representations"][LAYER]
    return {name: reps[b, 1:len(s) + 1].float().cpu().numpy().astype(np.float32)
            for b, (name, s) in enumerate(seqs)}

def embed_entry(chains, assignments, batch_cap=4):
    needed = sorted({a[0] for a in assignments if a})
    if not needed:
        return None
    seqs = [(c, chains[c][0]) for c in needed if len(chains[c][0]) <= MAX_SEQ_LEN]
    if len(seqs) != len(needed):
        return None                      # a required chain exceeds the window
    reprs = {}
    for i in range(0, len(seqs), batch_cap):
        reprs.update(chain_reprs(seqs[i:i + batch_cap]))
    vecs = [reprs[c][idx] for (c, idx) in (a for a in assignments if a)]
    return np.stack(vecs).mean(axis=0).astype(np.float32) if vecs else None
```

---

## Cell 4 — Re-embed (resumable)

```python
LIMIT = 200          # <-- SMOKE TEST FIRST. Set to None for the full run.
CHECKPOINT_EVERY = 500

done = {}
for sh in sorted(SHARD_DIR.glob("emb_*.npz")):
    z = np.load(sh, allow_pickle=True)
    for k, v in zip(z["keys"], z["vecs"]):
        done[str(k)] = v
print("resumed: %d already embedded" % len(done))

todo = [r for r in rmap.itertuples() if r.pocket_key not in done]
if LIMIT:
    todo = todo[:LIMIT]
print("to process: %d" % len(todo))

stats = {"ok": 0, "no_cif": 0, "unresolved": 0, "too_long": 0}
buf_k, buf_v = [], []
shard_n = len(list(SHARD_DIR.glob("emb_*.npz")))
t0 = time.time()

for n, r in enumerate(todo, 1):
    st = load_structure(r.pdb_id)
    if st is None:
        stats["no_cif"] += 1; continue
    chains = per_chain_sequences(st)
    assignments, unres = map_pocket_to_chains(json.loads(r.pocket_res_provenance), chains)
    if unres or not any(assignments):
        stats["unresolved"] += 1; continue
    vec = embed_entry(chains, assignments)
    if vec is None:
        stats["too_long"] += 1; continue
    buf_k.append(r.pocket_key); buf_v.append(vec); stats["ok"] += 1

    if n % 100 == 0:
        rate = n / max(1e-9, time.time() - t0)
        print("  %6d/%d  %.1f/s  eta %.0f min  %s"
              % (n, len(todo), rate, (len(todo)-n)/rate/60, stats), flush=True)
    if len(buf_k) >= CHECKPOINT_EVERY:
        np.savez_compressed(SHARD_DIR / ("emb_%04d.npz" % shard_n),
                            keys=np.array(buf_k, dtype=object), vecs=np.stack(buf_v))
        shard_n += 1; buf_k, buf_v = [], []

if buf_k:
    np.savez_compressed(SHARD_DIR / ("emb_%04d.npz" % shard_n),
                        keys=np.array(buf_k, dtype=object), vecs=np.stack(buf_v))

result("reembed_progress", processed=len(todo), **stats,
       VERDICT=("GATE HEALTHY - proceed to the full run" if stats["unresolved"] <= 0.02*max(1,len(todo))
                else "HIGH UNRESOLVED RATE - stop and diagnose before the full run"))
```

---

## Cell 5 — Assemble, preserving v0.1 row alignment

```python
done = {}
for sh in sorted(SHARD_DIR.glob("emb_*.npz")):
    z = np.load(sh, allow_pickle=True)
    for k, v in zip(z["keys"], z["vecs"]):
        done[str(k)] = v

out = np.array(old_emb, dtype=np.float32)      # start from v0.1
replaced = 0
for pk, vec in done.items():
    off = pk_to_off.get(pk)
    if off is not None:
        out[off] = vec; replaced += 1
retained = [m["pocket_key"] for m in manifest if m["pocket_key"] not in done]

np.save(OUT_DIR / "pocket_embeddings.npy", out)

# how much did the representation actually move?
import numpy.linalg as la
idx = [pk_to_off[k] for k in list(done)[:2000] if k in pk_to_off]
if idx:
    a = np.asarray(old_emb[idx], dtype=np.float32); b = out[idx]
    cos = (a*b).sum(1) / (la.norm(a,axis=1)*la.norm(b,axis=1) + 1e-12)
else:
    cos = np.array([np.nan])

result("reembed_final",
    n_rows=int(out.shape[0]),
    n_replaced_per_chain=replaced,
    n_retained_from_v01=len(retained),
    pct_replaced=round(100*replaced/max(1,out.shape[0]), 3),
    cosine_v01_vs_v02_median=round(float(np.median(cos)), 4),
    cosine_v01_vs_v02_p05=round(float(np.percentile(cos, 5)), 4),
    output=str(OUT_DIR / "pocket_embeddings.npy"),
    NOTE="Rows not replaced retain their v0.1 concatenated-input value. Do NOT "
         "describe the release as fully per-chain unless n_retained_from_v01 is 0.")

(OUT_DIR / "reembed_report.json").write_text(json.dumps(RESULTS3, indent=2, default=str))
print("\nNEXT: re-run notebook 08 pointed at this pocket_embeddings.npy to "
      "regenerate every pocket-dependent number (alpha sweep, geometry, "
      "leakage, retrieval).")
```
---

## Cell 6 — **Per-chain residue map** (CPU only; the artifact v0.2 is missing)

Notebook 09's `pocket_residue_map.parquet` describes the **chimeric** ESM-2
input that v0.1 used. The embeddings assembled in Cell 5 are **per-chain**, so
that map does not reproduce them: a reader following it would pool from the
wrong token positions. This cell writes the matching map.

It re-derives the assignments with the *same* functions Cell 4 used
(`per_chain_sequences`, `map_pocket_to_chains`), so the result is identical by
construction. No GPU and no ESM-2 forward pass is needed — only mmCIF parsing —
so this can run on a CPU runtime even if Cell 3 was never executed.

**Token indexing.** `chain_reprs` returns `reps[b, 1:len(s)+1]`, so array index
*i* within a chain corresponds to ESM-2 token position *i + 1* (BOS occupies 0).
`token_position` below is therefore `index_in_chain + 1`, matching the
convention notebook 09 used for the chimeric input.

```python
import gzip

MAP_SHARDS = OUT_DIR / "map_shards"
MAP_SHARDS.mkdir(parents=True, exist_ok=True)

# Only entries that actually have an embedding may appear in the map.
EMB_KEYS = set()
for _sh in sorted(SHARD_DIR.glob("emb_*.npz")):
    _z = np.load(_sh, allow_pickle=True)
    EMB_KEYS.update(str(k) for k in _z["keys"])
print("embedded entries: %d" % len(EMB_KEYS))
if not EMB_KEYS:
    raise SystemExit("no embedding shards found - run Cell 4 first")

_done = set()
for _sh in sorted(MAP_SHARDS.glob("map_*.jsonl")):
    for _l in open(_sh, encoding="utf-8"):
        _done.add(json.loads(_l)["pocket_key"])
print("map rows already built: %d" % len(_done))

_todo = [r for r in rmap.itertuples()
         if r.pocket_key in EMB_KEYS and r.pocket_key not in _done]
print("to build: %d" % len(_todo))

_stats = {"ok": 0, "no_cif": 0, "unresolved": 0, "too_long": 0, "mismatch": 0}
_buf, _shard = [], len(list(MAP_SHARDS.glob("map_*.jsonl")))
_t0 = time.time()

for _n, _r in enumerate(_todo, 1):
    _st = load_structure(_r.pdb_id)
    if _st is None:
        _stats["no_cif"] += 1; continue
    _chains = per_chain_sequences(_st)
    _prov = json.loads(_r.pocket_res_provenance)
    _assign, _unres = map_pocket_to_chains(_prov, _chains)
    if _unres or not any(_assign):
        _stats["unresolved"] += 1; continue

    # exactly the selection embed_entry() made
    _needed = sorted({a[0] for a in _assign if a})
    _seqs = [(c, _chains[c][0]) for c in _needed if len(_chains[c][0]) <= MAX_SEQ_LEN]
    if len(_seqs) != len(_needed):
        _stats["too_long"] += 1; continue

    # per-residue records, in the SAME order as the pocket residue list
    _res_chain, _res_idx, _res_tok, _res_name = [], [], [], []
    for _a, _p in zip(_assign, _prov):
        if _a is None:
            continue
        _c, _i = _a
        _res_chain.append(_c)
        _res_idx.append(int(_i))
        _res_tok.append(int(_i) + 1)          # BOS occupies position 0
        _res_name.append(str(_p["resname"]))

    if len(_res_tok) != len(_prov):
        _stats["mismatch"] += 1; continue

    _buf.append({
        "pocket_key": _r.pocket_key,
        "pdb_id": _r.pdb_id,
        "chain_order": _needed,
        "n_chains_in_esm_input": len(_needed),
        "chain_lengths": {c: len(_chains[c][0]) for c in _needed},
        "esm_input_length_total": sum(len(_chains[c][0]) for c in _needed),
        "n_pocket_residues": len(_res_tok),
        "pocket_res_chain": _res_chain,
        "pocket_res_index_in_chain": _res_idx,
        "pocket_res_token_position": _res_tok,
        "pocket_res_resname": _res_name,
        "sequences": {c: _chains[c][0] for c in _needed},
    })
    _stats["ok"] += 1

    if _n % 200 == 0:
        _rate = _n / max(1e-9, time.time() - _t0)
        print("  %6d/%d  %.1f/s  eta %.0f min  %s"
              % (_n, len(_todo), _rate, (len(_todo) - _n) / _rate / 60, _stats), flush=True)
    if len(_buf) >= 500:
        with open(MAP_SHARDS / ("map_%04d.jsonl" % _shard), "w", encoding="utf-8") as fh:
            for _row in _buf:
                fh.write(json.dumps(_row) + chr(10))
        _shard += 1; _buf = []

if _buf:
    with open(MAP_SHARDS / ("map_%04d.jsonl" % _shard), "w", encoding="utf-8") as fh:
        for _row in _buf:
            fh.write(json.dumps(_row) + chr(10))

result("per_chain_map_progress", processed=len(_todo), **_stats,
       NOTE="Assignments come from the same functions Cell 4 embedded with, so "
            "the map matches the embeddings by construction. Cell 7 proves it.")
```

---

## Cell 7 — **Verify the map against the embeddings, then write the artifacts**

Refuses to write unless the map covers exactly the embedded entries and the
token positions are in range for the sequences shipped beside them.

```python
rows = []
for _sh in sorted(MAP_SHARDS.glob("map_*.jsonl")):
    for _l in open(_sh, encoding="utf-8"):
        rows.append(json.loads(_l))
print("map rows: %d" % len(rows))

by_key = {r["pocket_key"]: r for r in rows}
_map_keys, _emb_keys = set(by_key), set(EMB_KEYS)

problems = []
if _map_keys != _emb_keys:
    problems.append("key sets differ: %d only-in-map, %d only-in-embeddings"
                    % (len(_map_keys - _emb_keys), len(_emb_keys - _map_keys)))

# every token position must be valid for its own chain's sequence
bad_tok = bad_len = 0
for r in rows:
    seqs = r["sequences"]
    for c, i, t in zip(r["pocket_res_chain"], r["pocket_res_index_in_chain"],
                       r["pocket_res_token_position"]):
        if c not in seqs or not (0 <= i < len(seqs[c])) or t != i + 1:
            bad_tok += 1
    if r["n_pocket_residues"] != len(r["pocket_res_token_position"]):
        bad_len += 1
if bad_tok:
    problems.append("%d residue records have an out-of-range or inconsistent token position" % bad_tok)
if bad_len:
    problems.append("%d rows disagree with their own n_pocket_residues" % bad_len)

# the shipped v0.1 map describes the CHIMERIC input; confirm this one differs
_shared = [k for k in list(_map_keys)[:2000] if k in set(rmap.pocket_key)]
_old = rmap.set_index("pocket_key")
_ndiff = 0
for k in _shared[:500]:
    try:
        old_tok = list(json.loads(_old.loc[k, "pocket_res_indices"]))
    except Exception:
        continue
    if old_tok != by_key[k]["pocket_res_token_position"]:
        _ndiff += 1
print("of 500 sampled entries, %d have token positions differing from the v0.1 map"
      % _ndiff)

result("per_chain_map_verification",
       n_map_rows=len(rows), n_embeddings=len(_emb_keys),
       key_sets_identical=bool(_map_keys == _emb_keys),
       bad_token_records=bad_tok, rows_inconsistent=bad_len,
       n_of_500_differing_from_v01_map=_ndiff,
       problems=problems or "none",
       NOTE="n_of_500_differing should be LARGE: the v0.1 map indexes a chimeric "
            "all-chain input, this one indexes each residue's own chain. If it is "
            "0, Cell 6 reproduced the old map and something is wrong.")

if problems:
    raise SystemExit("NOT WRITING - fix the problems above:" + chr(10)
                     + chr(10).join("  * " + p for p in problems))

# ---- write the two artifacts the deposit needs --------------------------
import pandas as _pd

_flat = [{
    "pocket_key": r["pocket_key"],
    "pdb_id": r["pdb_id"],
    "chain_order": json.dumps(r["chain_order"]),
    "n_chains_in_esm_input": r["n_chains_in_esm_input"],
    "esm_input_length_total": r["esm_input_length_total"],
    "n_pocket_residues": r["n_pocket_residues"],
    "pocket_res_chain": json.dumps(r["pocket_res_chain"]),
    "pocket_res_index_in_chain": json.dumps(r["pocket_res_index_in_chain"]),
    "pocket_res_token_position": json.dumps(r["pocket_res_token_position"]),
    "pocket_res_resname": json.dumps(r["pocket_res_resname"]),
} for r in rows]

_out_map = OUT_DIR / "pocket_residue_map_per_chain.parquet"
_pd.DataFrame(_flat).to_parquet(_out_map, index=False)

_out_fa = OUT_DIR / "esm_input_sequences_per_chain.fasta.gz"
_nrec = 0
with gzip.open(_out_fa, "wt") as fh:
    for r in rows:
        for c in r["chain_order"]:
            fh.write(">%s|%s%s%s%s" % (r["pocket_key"], c, chr(10),
                                       r["sequences"][c], chr(10)))
            _nrec += 1

result("per_chain_map_written",
       residue_map=str(_out_map),
       residue_map_MB=round(_out_map.stat().st_size / 1e6, 2),
       fasta=str(_out_fa),
       fasta_MB=round(_out_fa.stat().st_size / 1e6, 2),
       n_rows=len(_flat), n_fasta_records=_nrec,
       NOTE="Deposit these two files as pockets/pocket_residue_map.parquet and "
            "pockets/esm_input_sequences.fasta.gz, REPLACING the notebook-09 "
            "versions, which describe the chimeric v0.1 input and do not "
            "reproduce the per-chain embeddings shipped in v0.2. FASTA headers "
            "are '<pocket_key>|<chain>'; pool the rows named in "
            "pocket_res_token_position from each residue's own chain.")
```
