"""Independent end-to-end verification of the release about to be deposited.

Deliberately does not call the project's own tooling: build_v02_release wrote these
files and verify_v02_release shares its assumptions. This recomputes from the bytes.
"""

import hashlib
import io
import json
import os
import re

ROOT = "D:/v02_rebuild/release"
fail = []
warn = []


def ok(cond, label, detail=""):
    print("  %-4s %-54s %s" % ("OK" if cond else "FAIL", label, detail))
    if not cond:
        fail.append(label)


def note(label, detail=""):
    print("  %-4s %-54s %s" % ("--", label, detail))


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


print("=" * 80)
print("1. CHECKSUMS.sha256 recomputed from the bytes")
print("=" * 80)
lines = io.open(ROOT + "/CHECKSUMS.sha256", encoding="utf-8").read().strip().split("\n")
listed = {}
for ln in lines:
    ln = ln.strip()
    if not ln:
        continue
    m = re.match(r"^([0-9a-f]{64})\s+\*?(.+)$", ln)
    if m:
        listed[m.group(2).strip()] = m.group(1)
print("  entries in CHECKSUMS.sha256: %d" % len(listed))

bad, missing, checked = [], [], 0
for rel, want in sorted(listed.items()):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.exists(p):
        missing.append(rel)
        continue
    got = sha256(p)
    checked += 1
    if got != want:
        bad.append((rel, want[:12], got[:12]))
ok(not missing, "every listed file exists", "%d missing" % len(missing))
for m in missing[:5]:
    print("        missing: %s" % m)
ok(not bad, "every digest matches", "%d checked, %d mismatched" % (checked, len(bad)))
for rel, w, g in bad[:5]:
    print("        %s  want %s  got %s" % (rel, w, g))

print()
print("=" * 80)
print("2. DATASET_MANIFEST.json agrees with the tree")
print("=" * 80)
man = json.load(io.open(ROOT + "/DATASET_MANIFEST.json", encoding="utf-8"))
entries = man["files"]
mpaths = {f.get("path") or f.get("name") for f in entries}
ondisk = set()
total = 0
for r, _d, fs in os.walk(ROOT):
    for f in fs:
        p = os.path.join(r, f)
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        ondisk.add(rel)
        total += os.path.getsize(p)
self_ref = {"CHECKSUMS.sha256", "DATASET_MANIFEST.json"}
ok(
    man.get("n_files") == len(entries),
    "n_files equals len(files[])",
    "%s vs %d" % (man.get("n_files"), len(entries)),
)
ok(
    ondisk - mpaths == self_ref,
    "only the two self-referential files are unlisted",
    str(sorted(ondisk - mpaths)),
)
ok(
    not (mpaths - ondisk),
    "no manifest entry is missing from disk",
    str(sorted(mpaths - ondisk))[:60],
)
msum = sum(f.get("size", f.get("bytes", 0)) for f in entries)
ok(
    abs(msum - man["total_bytes"]) == 0,
    "total_bytes equals sum(files[].size)",
    "%d vs %d" % (man["total_bytes"], msum),
)
note("tree total incl. self-referential", "%.3f GB over %d files" % (total / 1e9, len(ondisk)))
ok(
    abs(total / 1e9 - 5.45) < 0.01,
    "tree total matches the paper's 5.45 GB",
    "%.3f GB" % (total / 1e9),
)

print()
print("=" * 80)
print("3. The ligand labels")
print("=" * 80)
try:
    import h5py

    with h5py.File(ROOT + "/ligands/ligand_coords.h5", "r") as f:
        groups = list(f.keys())
        n_lig = len(f["ligands"])
        n_lab = len(f["ligand_atom_labels"]) if "ligand_atom_labels" in f else 0
        k = list(f["ligand_atom_labels"])[0]
        dt = f["ligand_atom_labels/" + k].dtype
        rec = f["ligand_atom_labels/" + k][:]
        xyz = f["ligands/" + k][:]
    ok("ligand_atom_labels" in groups, "ligand_atom_labels group present", str(groups))
    ok(n_lig == 72388, "72,388 coordinate datasets", str(n_lig))
    ok(n_lab == 72306, "72,306 label datasets", str(n_lab))
    ok(n_lig - n_lab == 82, "exactly 82 unlabelled", str(n_lig - n_lab))
    ok(set(dt.names) == {"atom_name", "element", "altloc"}, "documented label dtype", str(dt.names))
    ok(
        len(rec) == len(xyz),
        "labels row-aligned to coordinates for %s" % k,
        "%d labels, %d coords" % (len(rec), len(xyz)),
    )
except ImportError:
    note("h5py unavailable", "skipped")

rep = json.load(io.open(ROOT + "/provenance/label_coords_report.json", encoding="utf-8"))["ligands"]
ok(rep["n_labelled"] == 72306, "report agrees: n_labelled", str(rep["n_labelled"]))
ok(
    rep["worst_coord_max_abs_err"] == 0.0,
    "report agrees: worst coord error is exact",
    str(rep["worst_coord_max_abs_err"]),
)
ok(
    rep["unlabelled_reasons"]["no_atoms"] + rep["unlabelled_reasons"]["coord_mismatch"] == 82,
    "82 accounted for by reason",
    str(rep["unlabelled_reasons"]),
)

print()
print("=" * 80)
print("4. No summary file disagrees with another about the same quantity")
print("=" * 80)
mg = json.load(io.open(ROOT + "/indices/modality_guidance.json", encoding="utf-8"))
man_mg = man.get("modality_guidance", {})


def med(t):
    m = re.search(r"median ([0-9.]+)", t or "")
    return m.group(1) if m else None


ok(
    med(mg["why_two_indices"]) == med(man_mg.get("why_two_indices", "")),
    "pocket-share median agrees sidecar vs manifest",
    "%s vs %s" % (med(mg["why_two_indices"]), med(man_mg.get("why_two_indices", ""))),
)
ok(
    mg["same_uniprot_hit_at_1_by_alpha"] == man_mg.get("same_uniprot_hit_at_1_by_alpha"),
    "alpha sweep agrees sidecar vs manifest",
)

ok(
    mg.get("ann_recall_vs_exact", {}).get("efSearch_16")
    == man_mg.get("ann_recall_vs_exact", {}).get("efSearch_16"),
    "ANN recall agrees sidecar vs manifest",
    "%s vs %s"
    % (
        mg.get("ann_recall_vs_exact", {}).get("efSearch_16"),
        man_mg.get("ann_recall_vs_exact", {}).get("efSearch_16"),
    ),
)

fig3 = json.load(io.open("paper/figures/fig3_technical_validation.json", encoding="utf-8"))
a = man_mg.get("anisotropy_warning", "")
want = "%.3f" % fig3["anisotropy_pocket"]
ok(want in a, "manifest anisotropy matches fig3", "expects %s" % want)
ok(
    "%.3f" % fig3["pocket_random_centred_median"] in a,
    "manifest centred median matches fig3 (sign included)",
    "expects %.3f" % fig3["pocket_random_centred_median"],
)

print()
print("=" * 80)
print("5. No preview-era value anywhere in the deposit's text files")
print("=" * 80)
stale = {
    "0.470": "old w_p (per-chain reconstruction)",
    "0.844": "old anisotropy ratio",
    "0.706": "old raw random pocket pair",
    "-0.027": "old centred median (wrong sign)",
    "0.978": "old ANN recall",
    "19954940": "superseded code DOI",
    "convenience mirror": "false HF claim",
}
hits = 0
for r, _d, fs in os.walk(ROOT):
    for f in fs:
        if not f.endswith((".json", ".md", ".txt", ".sha256", ".log")):
            continue
        p = os.path.join(r, f)
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        if rel.startswith("provenance/") and rel.endswith("_report.json"):
            continue  # historical build reports; they record what WAS measured
        try:
            t = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for pat, why in stale.items():
            if pat in t:
                hits += 1
                print("  HIT  %-44s %s (%s)" % (rel, pat, why))
ok(hits == 0, "no preview-era value in shipped text", "%d hit(s)" % hits)

print()
print("=" * 80)
print("RESULT: %s" % ("ALL CHECKS PASS" if not fail else "%d FAILURE(S): %s" % (len(fail), fail)))
print("=" * 80)
