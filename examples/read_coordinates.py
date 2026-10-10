"""Read one complex's pocket and ligand coordinates without loading the rest.

The companion to retrieve_similar_pockets.py, and the second of the two Usage
Notes listings that now live here rather than in the Data Descriptor.

Both HDF5 files are keyed by pocket_key, so one lookup serves both. Datasets
are read lazily: opening the file does not read the collection.

What the two arrays are is not symmetric, and the difference matters:

  * pockets/<pocket_key> holds ONE HEAVY-ATOM CENTROID PER POCKET RESIDUE --
    not per-atom coordinates, and not C-alpha positions. Residue identifiers
    ship alongside in pocket_residue_labels so each row resolves to a residue.
  * ligands/<pocket_key> holds per-atom heavy-atom coordinates, with atom
    names, element symbols and altloc indicators in ligand_atom_labels,
    row-aligned to the coordinates.

Coordinates are in the deposited Protein Data Bank reference frame.

Usage:
    python examples/read_coordinates.py --local_dir <release root> [--pocket_key KEY]
"""

import argparse
import json
import os

import h5py


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local_dir", required=True, help="staged release root")
    ap.add_argument("--pocket_key", default="", help="defaults to the first record")
    args = ap.parse_args()

    key = args.pocket_key
    if not key:
        with open(os.path.join(args.local_dir, "metadata.jsonl")) as fh:
            key = json.loads(fh.readline())["pocket_key"]
        print("no --pocket_key given; using the first record: %s" % key)

    pocket_h5 = os.path.join(args.local_dir, "pockets", "pocket_coords.h5")
    ligand_h5 = os.path.join(args.local_dir, "ligands", "ligand_coords.h5")

    with h5py.File(pocket_h5, "r") as f:
        pocket_xyz = f["pockets/%s" % key][:]
        # pocket_residue_labels, not pocket_atom_labels: the pocket side stores one
        # row per RESIDUE. Reading the wrong name through f.get() returns None and
        # the script then prints nothing instead of failing, which is how this
        # survived review.
        labels = f.get("pocket_residue_labels/%s" % key)
        pocket_labels = labels[:] if labels is not None else None

    with h5py.File(ligand_h5, "r") as f:
        ligand_xyz = f["ligands/%s" % key][:]
        labels = f.get("ligand_atom_labels/%s" % key)
        ligand_labels = labels[:] if labels is not None else None

    print("pocket: %d residue centroids, shape %s" % (len(pocket_xyz), pocket_xyz.shape))
    if pocket_labels is not None:
        first = pocket_labels[0]
        print(
            "  first residue: chain %s seqid %s resname %s"
            % (first["chain"].decode(), first["seqid"], first["resname"].decode())
        )

    print("ligand: %d heavy atoms, shape %s" % (len(ligand_xyz), ligand_xyz.shape))
    if ligand_labels is not None:
        names = [a["atom_name"].decode().strip() for a in ligand_labels[:8]]
        print("  first atoms: %s" % ", ".join(names))
    else:
        # 82 of 72,388 entries are unlabelled: 75 could not be re-extracted
        # from the source structure and 7 did not reproduce the deposited
        # coordinates. They ship with coordinates and without labels rather
        # than with labels that were not verified.
        print("  no atom labels for this entry (one of the 82 exceptions)")


if __name__ == "__main__":
    raise SystemExit(main())
