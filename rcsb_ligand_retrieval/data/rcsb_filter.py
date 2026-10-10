"""
RCSB PDB filtering.

Given a local mirror of the RCSB PDB (mmCIF files), filter for
protein-ligand complexes suitable for retrieval-index construction.
Output is a JSONL manifest with one row per surviving (pdb_id,
ligand_chain_resname, quality fields) record.

Filtering criteria:
  - Resolution <= 2.5 A
  - At least one non-polymer HET group with 10 <= heavy_atoms <= 70
  - Exclude waters, ions, common crystallization additives
  - Exclude modified amino acids (ligand must be a small molecule, not a
    covalent modification of the protein backbone)
  - Protein chain length >= 50 residues

Usage:
    python -m rcsb_ligand_retrieval.data.rcsb_filter \\
        --pdb_dir /content/pdb_mmcif \\
        --output data/raft_pdb_filtered.jsonl \\
        --min_heavy 10 \\
        --max_heavy 70 \\
        --max_resolution 2.5 \\
        --min_protein_len 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Common crystallization additives and buffer components to exclude.
EXCLUDED_HET = {
    "HOH",
    "WAT",
    "DOD",
    "TIP",
    "NA",
    "K",
    "MG",
    "CA",
    "ZN",
    "FE",
    "CU",
    "MN",
    "CO",
    "NI",
    "CL",
    "BR",
    "IOD",
    "F",
    "SO4",
    "PO4",
    "NO3",
    "ACT",
    "EDO",
    "GOL",
    "PEG",
    "PGE",
    "PG4",
    "MPD",
    "DMS",
    "TRS",
    "BME",
    "DTT",
    "NAG",
    "FUC",
    "BMA",
}


def is_drug_like_ligand(
    resname: str,
    heavy_atoms: int,
    min_heavy: int,
    max_heavy: int,
) -> bool:
    """Simple first-pass filter for a drug-like HET group."""
    if resname in EXCLUDED_HET:
        return False
    if not (min_heavy <= heavy_atoms <= max_heavy):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter RCSB PDB for retrieval-index construction."
    )
    parser.add_argument(
        "--pdb_dir", type=Path, required=True, help="Local rsync mirror of RCSB mmCIF files"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="JSONL manifest of filtered entries"
    )
    parser.add_argument("--min_heavy", type=int, default=10)
    parser.add_argument("--max_heavy", type=int, default=70)
    parser.add_argument("--max_resolution", type=float, default=2.5)
    parser.add_argument("--min_protein_len", type=int, default=50)
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after N entries (for smoke tests)"
    )
    args = parser.parse_args()

    try:
        import gemmi
    except ImportError as exc:
        raise SystemExit("gemmi is required for fast mmCIF parsing. pip install gemmi") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_kept = 0
    with open(args.output, "w") as out_f:
        for cif_path in args.pdb_dir.rglob("*.cif.gz"):
            n_total += 1
            if args.limit is not None and n_kept >= args.limit:
                break
            try:
                structure = gemmi.read_structure(str(cif_path))
            except Exception:
                continue

            if structure.resolution > args.max_resolution or structure.resolution == 0.0:
                continue

            # Protein length check
            protein_len = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.entity_type == gemmi.EntityType.Polymer:
                            protein_len += 1
            if protein_len < args.min_protein_len:
                continue

            # Find drug-like HET groups
            candidates = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.entity_type != gemmi.EntityType.NonPolymer:
                            continue
                        resname = residue.name
                        heavy = sum(1 for atom in residue if atom.element.name != "H")
                        if is_drug_like_ligand(resname, heavy, args.min_heavy, args.max_heavy):
                            candidates.append(
                                {
                                    "chain": chain.name,
                                    "resname": resname,
                                    "resnum": residue.seqid.num,
                                    "heavy_atoms": heavy,
                                }
                            )
            if not candidates:
                continue

            record = {
                "pdb_id": structure.name.lower(),
                "resolution": structure.resolution,
                "protein_length": protein_len,
                "ligands": candidates,
                "path": str(cif_path.relative_to(args.pdb_dir)),
            }
            out_f.write(json.dumps(record) + "\n")
            n_kept += 1

            if n_kept % 1000 == 0:
                print("  filtered: %d kept / %d scanned" % (n_kept, n_total))

    print("Done: %d kept / %d scanned. Output: %s" % (n_kept, n_total, args.output))


if __name__ == "__main__":
    main()
