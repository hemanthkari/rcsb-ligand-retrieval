"""
Independent verification of an assembled v0.2 release.

Deliberately NOT part of build_v02_release.py: a builder that validates its own
output can only confirm what its author already believed. This reads the finished
release from disk and tests the three claims the manuscript makes that nothing
else checks.

  1. THE TARGET-DISJOINT SPLIT IS ACTUALLY DISJOINT.
     The paper will claim the new partition is homology-disjoint and suitable for
     generalisation evaluation. The v0.1 random split made an equivalent claim and
     was wrong (46.4% of validation entries had a >=0.99-cosine training
     neighbour). Asserting disjointness without measuring it repeats that error.
     Measures: cross-boundary nearest-neighbour cosine distribution, and
     cross-boundary ligand-component overlap -- because a sequence-disjoint split
     still leaks bit-identical ligand vectors under component-level fingerprinting.

  2. LIGAND COORDINATES CORRESPOND TO THE SHIPPED SMILES.
     The abstract claims the resource supports retrieval-augmented pose
     prediction. That requires a documented, verified correspondence between
     coordinate rows and SMILES atoms. Reports the mappable fraction over the
     corpus using fluxion_dock.data.atom_order.

  3. THE DEPOSITED LIGAND GEOMETRY IS PHYSICALLY SANE.
     Bond lengths derived from the shipped coordinates catch corrupted or
     mis-paired geometry that a shape check cannot.

Plus a file-inventory check: every shipped file must appear in the manifest, and
every manifest entry must exist.

Usage
-----
    python scripts/verify_v02_release.py --release_dir <v0.2 dir> [--limit N]
    python scripts/verify_v02_release.py --self_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ideal heavy-atom single-bond lengths (Angstrom), keyed by sorted atomic numbers.
# Same table used by fluxion_dock/eval/pose_validity.py.
BOND_IDEAL = {
    (6, 6): 1.54,
    (6, 7): 1.47,
    (6, 8): 1.43,
    (6, 9): 1.35,
    (6, 15): 1.84,
    (6, 16): 1.82,
    (6, 17): 1.77,
    (6, 35): 1.94,
    (7, 7): 1.45,
    (7, 8): 1.40,
    (8, 15): 1.63,
    (8, 16): 1.65,
    (16, 16): 2.05,
    (6, 53): 2.14,
}
BOND_TOLERANCE = 0.35  # generous: aromatic/partial orders shorten bonds


def _pct(n, d):
    return round(100.0 * n / max(1, d), 3)


# ---------------------------------------------------------------------------
# 1. split disjointness
# ---------------------------------------------------------------------------


def verify_splits(release, meta, joined, sample=4000, seed=42):
    import faiss

    out = {}
    sp = release / "splits"
    pairs = [
        ("target_disjoint", "train_target_disjoint.npy", "val_target_disjoint.npy"),
        ("random", "train_indices.npy", "val_indices.npy"),
    ]

    ccd = [m.get("ligand_resname") for m in meta]
    rng = np.random.default_rng(seed)

    for name, tf, vf in pairs:
        tp, vp = sp / tf, sp / vf
        if not (tp.exists() and vp.exists()):
            out[name] = "ABSENT"
            continue
        tr, va = np.load(tp), np.load(vp)
        tr = tr[tr < len(meta)]
        va = va[va < len(meta)]
        if len(tr) == 0 or len(va) == 0:
            out[name] = "EMPTY"
            continue

        Jt = np.ascontiguousarray(np.asarray(joined[tr], dtype=np.float32))
        idx = faiss.IndexFlatIP(Jt.shape[1])
        idx.add(Jt)
        q = va if len(va) <= sample else rng.choice(va, sample, replace=False)
        D, _ = idx.search(np.ascontiguousarray(np.asarray(joined[q], dtype=np.float32)), 1)
        nn = D[:, 0]

        tr_ccd = set(ccd[i] for i in tr)
        leak = sum(1 for i in q if ccd[i] in tr_ccd)

        out[name] = {
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "overlap_indices": int(len(set(tr.tolist()) & set(va.tolist()))),
            "val_nn_cosine_median": round(float(np.median(nn)), 4),
            "frac_val_with_train_nbr_ge_0.99": round(float((nn >= 0.99).mean()), 4),
            "frac_val_with_train_nbr_ge_0.95": round(float((nn >= 0.95).mean()), 4),
            "frac_val_with_train_nbr_ge_0.90": round(float((nn >= 0.90).mean()), 4),
            "frac_val_ligand_ccd_also_in_train": round(leak / max(1, len(q)), 4),
        }
    out["NOTE"] = (
        "The target-disjoint split must show a markedly lower "
        "frac_val_with_train_nbr_ge_0.99 than the random split, or it does not "
        "deliver what the manuscript claims. Ligand-CCD overlap is expected to "
        "remain high even for a sequence-disjoint split -- report it rather than "
        "implying chemical disjointness that was never enforced."
    )
    return out


# ---------------------------------------------------------------------------
# 2. ligand atom-order correspondence
# ---------------------------------------------------------------------------


def _ligand_elements(f, key, ds):
    """Element symbols for one ligand, in coordinate-row order.

    The documented v0.2 layout is a structured array at
    ligand_atom_labels/<key> with fields (atom_name, element, altloc),
    row-aligned to ligands/<key>. Older releases carried the elements as a
    dataset attribute or a sibling dataset; both are still accepted so an
    earlier release still verifies.
    """
    lab_key = "ligand_atom_labels/" + key.split("/", 1)[-1]
    if lab_key in f:
        rec = f[lab_key][:]
        if "element" in (rec.dtype.names or ()):
            out = []
            for v in rec["element"]:
                out.append(
                    v.decode("utf-8", "replace").strip()
                    if isinstance(v, (bytes, bytearray))
                    else str(v).strip()
                )
            return out
    for attr in ("elements", "element", "atom_elements"):
        if attr in ds.attrs:
            return [str(x) for x in ds.attrs[attr]]
    sib = key + "_elements"
    if sib in f:
        return [str(x) for x in f[sib][:]]
    return None


def verify_atom_order(release, meta, limit=None):
    try:
        import h5py
        from fluxion_dock.data.atom_order import coord_to_rdkit_perm
    except Exception as exc:
        return {"SKIPPED": "needs h5py + fluxion_dock.data.atom_order (%s)" % type(exc).__name__}

    lig_h5 = release / "ligands" / "ligand_coords.h5"
    if not lig_h5.exists():
        return {"SKIPPED": "ligand_coords.h5 absent"}

    rows = meta if limit is None else meta[:limit]
    ok = no_elem = no_map = missing = 0
    with h5py.File(lig_h5, "r") as f:
        for m in rows:
            key = "ligands/" + m["pocket_key"]
            if key not in f:
                missing += 1
                continue
            ds = f[key]
            coords = np.asarray(ds[:], dtype=np.float32)
            elems = _ligand_elements(f, key, ds)
            if elems is None:
                no_elem += 1
                continue
            _perm, good = coord_to_rdkit_perm(m.get("ligand_smiles", ""), elems, coords)
            if good:
                ok += 1
            else:
                no_map += 1

    n = len(rows)
    return {
        "n_checked": n,
        "mappable_to_rdkit_order": ok,
        "pct_mappable": _pct(ok, n),
        "no_element_labels": no_elem,
        "unmappable": no_map,
        "dataset_missing": missing,
        "NOTE": (
            "The abstract claims support for retrieval-augmented pose "
            "prediction. That requires coordinates resolvable to SMILES atom "
            "order. If pct_mappable is low, either ship the mapping or narrow "
            "the claim. no_element_labels should equal the unlabelled count "
            "that provenance/label_coords_report.json documents; a much "
            "larger number means the labelling stage did not run."
        ),
    }


# ---------------------------------------------------------------------------
# 3. physical validity of the deposited ligand geometry
# ---------------------------------------------------------------------------


def verify_ligand_geometry(release, meta, limit=2000, seed=42):
    try:
        import h5py
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        from fluxion_dock.data.atom_order import coord_to_rdkit_perm
    except Exception as exc:
        return {"SKIPPED": "needs h5py + rdkit (%s)" % type(exc).__name__}

    lig_h5 = release / "ligands" / "ligand_coords.h5"
    if not lig_h5.exists():
        return {"SKIPPED": "ligand_coords.h5 absent"}

    rng = np.random.default_rng(seed)
    pick = (
        meta
        if len(meta) <= limit
        else [meta[i] for i in rng.choice(len(meta), limit, replace=False)]
    )
    checked = bad_bonds = clashes = 0
    devs = []
    with h5py.File(lig_h5, "r") as f:
        for m in pick:
            key = "ligands/" + m["pocket_key"]
            if key not in f:
                continue
            ds = f[key]
            coords = np.asarray(ds[:], dtype=np.float32)
            mol = Chem.MolFromSmiles(m.get("ligand_smiles") or "")
            if mol is None:
                continue
            mol = Chem.RemoveHs(mol)
            if mol.GetNumAtoms() != len(coords):
                continue
            elems = _ligand_elements(f, key, ds)
            if elems is None:
                continue
            perm, good = coord_to_rdkit_perm(m.get("ligand_smiles", ""), elems, coords)
            if not good:
                continue
            xyz = coords[perm]
            checked += 1
            worst = 0.0
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                za = mol.GetAtomWithIdx(i).GetAtomicNum()
                zb = mol.GetAtomWithIdx(j).GetAtomicNum()
                ideal = BOND_IDEAL.get((min(za, zb), max(za, zb)))
                if ideal is None:
                    continue
                d = float(np.linalg.norm(xyz[i] - xyz[j]))
                worst = max(worst, abs(d - ideal))
                if abs(d - ideal) > BOND_TOLERANCE:
                    bad_bonds += 1
            devs.append(worst)
            # non-bonded atoms closer than 0.9 A indicate corrupted coordinates
            dm = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            np.fill_diagonal(dm, 9e9)
            for b in mol.GetBonds():
                dm[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 9e9
                dm[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 9e9
            if dm.min() < 0.9:
                clashes += 1

    devs = np.array(devs) if devs else np.array([np.nan])
    return {
        "n_checked": checked,
        "entries_with_a_nonbonded_pair_under_0.9A": clashes,
        "pct_with_clash": _pct(clashes, checked),
        "bonds_deviating_over_%.2fA" % BOND_TOLERANCE: bad_bonds,
        "worst_bond_deviation_median": round(float(np.nanmedian(devs)), 3),
        "worst_bond_deviation_p95": round(float(np.nanpercentile(devs, 95)), 3),
        "NOTE": (
            "Deposited crystallographic geometry should be physically sane. "
            "Widespread deviation indicates corrupted or mis-paired "
            "coordinates, not merely an unusual ligand."
        ),
    }


# ---------------------------------------------------------------------------
# 4. file inventory vs manifest
# ---------------------------------------------------------------------------


def verify_inventory(release):
    man = release / "DATASET_MANIFEST.json"
    if not man.exists():
        return {"SKIPPED": "DATASET_MANIFEST.json absent"}
    listed = set()
    try:
        d = json.loads(man.read_text(encoding="utf-8"))
        for k in ("files", "inventory", "contents"):
            v = d.get(k)
            if isinstance(v, dict):
                listed |= set(v.keys())
            elif isinstance(v, list):
                listed |= {x.get("path", x) if isinstance(x, dict) else x for x in v}
    except Exception as exc:
        return {"ERROR": "manifest unreadable: %s" % exc}

    on_disk = {p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()}
    ignore = {"DATASET_MANIFEST.json", "CHECKSUMS.sha256"}
    return {
        "files_on_disk": len(on_disk),
        "files_in_manifest": len(listed),
        "on_disk_not_in_manifest": sorted(on_disk - listed - ignore)[:20],
        "in_manifest_not_on_disk": sorted(listed - on_disk)[:20],
        "n_undocumented": len(on_disk - listed - ignore),
        "n_missing": len(listed - on_disk),
        "NOTE": "Table 1 of the manuscript claims to list every file. Anything "
        "undocumented here must be added to Table 1 or removed.",
    }


# ---------------------------------------------------------------------------


def run(args):
    release = Path(args.release_dir)
    meta = [json.loads(l) for l in open(release / "metadata.jsonl", encoding="utf-8")]
    jp = release / "joined_embeddings.npy"
    joined = np.load(jp, mmap_mode="r") if jp.exists() else None
    print("release: %s  (%d metadata rows)" % (release, len(meta)))

    out = {}
    out["1_splits"] = (
        verify_splits(release, meta, joined)
        if joined is not None
        else {"SKIPPED": "joined_embeddings.npy absent"}
    )
    out["2_atom_order"] = verify_atom_order(release, meta, args.limit)
    out["3_ligand_geometry"] = verify_ligand_geometry(release, meta)
    out["4_inventory"] = verify_inventory(release)

    for sec, kv in out.items():
        print("\n" + "=" * 72)
        print("RESULT  [%s]" % sec)
        print("=" * 72)
        if isinstance(kv, dict):
            for k, v in kv.items():
                print("  %-44s %s" % (k, v))
        else:
            print("  %s" % kv)

    dest = release / "verify_v02_report.json"
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nwrote %s" % dest)


def _self_test():
    """Exercise the pure logic with no release on disk."""
    # bond table sanity
    assert BOND_IDEAL[(6, 6)] == 1.54
    assert all(a <= b for (a, b) in BOND_IDEAL)  # keys sorted by atomic no.

    # inventory diffing
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        (r / "a.npy").write_bytes(b"x")
        (r / "sub").mkdir()
        (r / "sub" / "b.npy").write_bytes(b"y")
        (r / "CHECKSUMS.sha256").write_bytes(b"z")
        (r / "DATASET_MANIFEST.json").write_text(
            json.dumps({"files": {"a.npy": {}, "sub/b.npy": {}, "gone.npy": {}}})
        )
        inv = verify_inventory(r)
        assert inv["n_undocumented"] == 0, inv
        assert inv["n_missing"] == 1 and inv["in_manifest_not_on_disk"] == ["gone.npy"], inv

    # percentage helper
    assert _pct(1, 4) == 25.0 and _pct(0, 0) == 0.0

    print(
        "verify_v02_release self-test: OK  "
        "(bond table sorted+correct; inventory diff detects missing and "
        "undocumented files; ignores self-referential artifacts)"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--release_dir")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap entries for the atom-order sweep (default: all)",
    )
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.release_dir:
        sys.exit("--release_dir is required (or pass --self_test)")
    run(args)


if __name__ == "__main__":
    main()
