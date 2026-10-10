"""
ESM-2 pocket embedding runner.

For each (pdb_id, ligand) entry in the filtered manifest, extract the pocket
residues (residues with any heavy atom within POCKET_RADIUS A of the ligand),
run the protein sequence through ESM-2, and mean-pool over the pocket
residue embeddings to produce a (1280,) descriptor.

Outputs:
    <out_dir>/pocket_embeddings.npy     (N, 1280) float32
    <out_dir>/pocket_coords.h5          HDF5 with per-complex heavy-atom coords
    <out_dir>/pocket_manifest.jsonl     one row per successfully embedded entry

The manifest rows carry the offset into pocket_embeddings.npy and the HDF5
dataset name so the retrieval index can cross-reference everything.

Usage:
    python -m rcsb_ligand_retrieval.retrieval.embed_pockets \\
        --manifest data/raft_pdb_filtered.jsonl \\
        --pdb_dir /content/pdb_mmcif \\
        --output data/retrieval_index/pockets/ \\
        --model esm2_t33_650M_UR50D \\
        --pocket_radius 8.0 \\
        --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


POCKET_RADIUS_DEFAULT = 8.0  # Angstroms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pdb_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--pocket_radius", type=float, default=POCKET_RADIUS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help="Save progress to disk every N entries (resume-safe)",
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Disable fp16 inference (use fp32, ~2x slower)"
    )
    args = parser.parse_args()

    try:
        import esm
        import gemmi
        import h5py
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n  pip install fair-esm gemmi h5py torch"
        ) from exc

    args.output.mkdir(parents=True, exist_ok=True)
    emb_path = args.output / "pocket_embeddings.npy"
    coords_path = args.output / "pocket_coords.h5"
    manifest_path = args.output / "pocket_manifest.jsonl"

    print("Loading ESM-2 model: %s" % args.model)
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(args.device).eval()
    # fp16 on GPU: ~2x speedup, negligible accuracy loss for pooled embeddings
    use_fp16 = args.device.startswith("cuda") and not args.no_fp16
    if use_fp16:
        model = model.half()
        print("  fp16 enabled")
    embed_dim = model.embed_dim
    repr_layer = model.num_layers

    # Read manifest
    entries = []
    with open(args.manifest) as f:
        for line in f:
            entries.append(json.loads(line))
    if args.limit:
        entries = entries[: args.limit]
    print("Loaded %d entries to embed" % len(entries))

    # Resume from checkpoint. The embedding run is long enough that an
    # interrupted job must restart from the last checkpoint rather than
    # from scratch. pocket_embeddings.npy and pocket_manifest.jsonl are
    # rewritten every --checkpoint_every entries; pocket_coords.h5 is
    # opened in append mode so previously written datasets survive.
    # On restart, the existing manifest gives the set of completed
    # pdb_ids that the main loop must skip.
    manifest_rows = []
    completed_pdb_ids = set()
    resume_embeddings = None

    if emb_path.exists() and manifest_path.exists():
        try:
            resume_embeddings = np.load(emb_path)
            with open(manifest_path) as f:
                manifest_rows = [json.loads(l) for l in f]
            if len(resume_embeddings) != len(manifest_rows):
                print(
                    "WARNING: checkpoint mismatch (%d embeddings vs %d manifest rows),"
                    " discarding checkpoint and starting fresh"
                    % (
                        len(resume_embeddings),
                        len(manifest_rows),
                    )
                )
                manifest_rows = []
                resume_embeddings = None
            else:
                completed_pdb_ids = {r["pdb_id"] for r in manifest_rows}
                print(
                    "Resuming from checkpoint: %d completed (%d entries remaining)"
                    % (
                        len(manifest_rows),
                        sum(1 for e in entries if e["pdb_id"] not in completed_pdb_ids),
                    )
                )
        except Exception as exc:
            print("WARNING: could not load checkpoint (%s), starting fresh" % exc)
            manifest_rows = []
            resume_embeddings = None

    # Pre-allocate; copy any resumed embeddings into the leading rows
    all_embeddings = np.zeros((len(entries), embed_dim), dtype=np.float32)
    if resume_embeddings is not None:
        n_resume = len(resume_embeddings)
        all_embeddings[:n_resume] = resume_embeddings
        n_ok = n_resume
    else:
        n_ok = 0

    # Open HDF5 in append mode so we don't clobber existing pocket datasets
    h5_mode = "a" if resume_embeddings is not None and coords_path.exists() else "w"
    h5 = h5py.File(coords_path, h5_mode)

    def checkpoint():
        """Flush current progress to disk. Called every --checkpoint_every
        entries and at the end. Safe to call multiple times."""
        h5.flush()
        np.save(emb_path, all_embeddings[:n_ok])
        with open(manifest_path, "w") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")

    # Skip previously completed entries up front so the progress
    # counter reflects work remaining rather than total dataset size.
    if completed_pdb_ids:
        before = len(entries)
        entries = [e for e in entries if e["pdb_id"] not in completed_pdb_ids]
        print(
            "Skipped %d previously completed entries, %d remaining"
            % (
                before - len(entries),
                len(entries),
            )
        )

    # Batched forward pass: accumulate up to --batch_size pocket items,
    # then run ESM-2 once on the padded batch. The batched form
    # substantially improves GPU throughput compared with a
    # per-sequence loop.
    pending = []  # list of dicts w/ seq, pdb_id, lig, pocket_res_indices, pocket_coords

    def flush_pending():
        nonlocal n_ok
        if not pending:
            return
        labels = [(item["pdb_id"], item["seq"]) for item in pending]
        _, _, tokens = batch_converter(labels)
        tokens = tokens.to(args.device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]  # (B, L+2, D)
        for b, item in enumerate(pending):
            idxs = [j + 1 for j in item["pocket_res_indices"]]
            pocket_reps = reps[b, idxs]
            emb = pocket_reps.mean(dim=0).float().cpu().numpy().astype(np.float32)
            offset = n_ok
            all_embeddings[offset] = emb
            lig = item["lig"]
            key = "%s__%s_%d" % (item["pdb_id"], lig["resname"], lig["resnum"])
            if ("pockets/%s" % key) not in h5:
                h5.create_dataset(
                    "pockets/%s" % key,
                    data=item["pocket_coords"],
                    compression="gzip",
                )
            manifest_rows.append(
                {
                    "offset": offset,
                    "pdb_id": item["pdb_id"],
                    "ligand_resname": lig["resname"],
                    "ligand_resnum": lig["resnum"],
                    "ligand_chain": lig["chain"],
                    "pocket_key": key,
                    "pocket_n_residues": len(item["pocket_res_indices"]),
                }
            )
            n_ok += 1
        pending.clear()

    # Prefetch pocket extraction in parallel worker processes; gemmi
    # CIF parsing and pocket-residue selection are CPU-bound and would
    # otherwise leave the GPU idle between batches.
    from concurrent.futures import ProcessPoolExecutor

    prefetch_args = [
        (entry, str(args.pdb_dir), args.pocket_radius, args.max_seq_len) for entry in entries
    ]
    n_workers = max(1, (os.cpu_count() or 4) - 1)
    print("Prefetching pockets with %d worker processes" % n_workers)
    executor = ProcessPoolExecutor(max_workers=n_workers)
    # imap-style ordered streaming so the main thread consumes in the
    # same order entries were submitted. chunksize=4 amortizes IPC.
    stream = executor.map(_prefetch_pocket, prefetch_args, chunksize=4)

    try:
        for i, result in enumerate(stream):
            if result is not None:
                pending.append(result)
                if len(pending) >= args.batch_size:
                    flush_pending()

            if (i + 1) % 100 == 0:
                print(
                    "  %d/%d processed, %d embeddings written"
                    % (
                        i + 1,
                        len(entries),
                        n_ok,
                    ),
                    flush=True,
                )
            if (i + 1) % args.checkpoint_every == 0:
                flush_pending()
                checkpoint()
                print(
                    "  checkpoint saved at %d entries (%d embeddings)" % (i + 1, n_ok),
                    flush=True,
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Flush any remaining items in the last partial batch
    flush_pending()
    checkpoint()
    h5.close()

    print("Done: %d pocket embeddings written to %s" % (n_ok, emb_path))
    print("HDF5 coords: %s" % coords_path)
    print("Manifest: %s" % manifest_path)


def _prefetch_pocket(task):
    """Worker-process entry point: read a CIF and extract one pocket.

    Runs in a ProcessPoolExecutor worker so CIF parsing happens in
    parallel with GPU forward passes on the main process. Returns a
    dict the main loop can append to `pending`, or None to skip.
    """
    entry, pdb_dir, radius, max_seq_len = task
    from pathlib import Path as _Path
    import gemmi as _gemmi

    cif_path = _Path(pdb_dir) / entry["path"]
    if not cif_path.exists():
        return None
    try:
        structure = _gemmi.read_structure(str(cif_path))
        structure.setup_entities()
    except Exception:
        return None

    for lig in entry["ligands"]:
        try:
            seq, pocket_res_indices, pocket_coords = _extract_pocket(
                structure,
                ligand_chain=lig["chain"],
                ligand_resname=lig["resname"],
                ligand_resnum=lig["resnum"],
                radius=radius,
            )
        except Exception:
            continue
        if seq is None or not pocket_res_indices:
            continue
        if len(seq) > max_seq_len:
            continue
        return {
            "pdb_id": entry["pdb_id"],
            "seq": seq,
            "lig": lig,
            "pocket_res_indices": pocket_res_indices,
            "pocket_coords": pocket_coords,
        }
    return None


def _extract_pocket(
    structure,
    ligand_chain: str,
    ligand_resname: str,
    ligand_resnum: int,
    radius: float,
):
    """Return (sequence, pocket_residue_indices, pocket_heavy_coords)."""
    import gemmi

    # Locate the ligand heavy atom coordinates
    lig_atoms = []
    for model in structure:
        for chain in model:
            if chain.name != ligand_chain:
                continue
            for residue in chain:
                if residue.name != ligand_resname:
                    continue
                if residue.seqid.num != ligand_resnum:
                    continue
                for atom in residue:
                    if atom.element.name != "H":
                        lig_atoms.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
        break  # first model only
    if not lig_atoms:
        return None, None, None
    lig_coords = np.stack(lig_atoms)
    r2 = radius * radius

    # Build the protein sequence and find pocket residues (any heavy atom
    # within `radius` of any ligand heavy atom). Residue indices are 0-based
    # positions in the sequence we build, so they align with ESM-2 tokens.
    seq_chars = []
    pocket_indices = []
    pocket_coords_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.entity_type != gemmi.EntityType.Polymer:
                    continue
                three = residue.name
                one = _three_to_one(three)
                if one is None:
                    continue
                idx = len(seq_chars)
                seq_chars.append(one)
                # Check distance to ligand
                residue_coords = []
                is_close = False
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    ax = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    residue_coords.append(ax)
                    if not is_close:
                        diffs = lig_coords - ax
                        if (diffs * diffs).sum(axis=1).min() <= r2:
                            is_close = True
                if is_close and residue_coords:
                    pocket_indices.append(idx)
                    pocket_coords_list.append(np.stack(residue_coords).mean(axis=0))
        break

    if not pocket_indices:
        return None, None, None
    sequence = "".join(seq_chars)
    pocket_coords = np.stack(pocket_coords_list).astype(np.float32)
    return sequence, pocket_indices, pocket_coords


_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


def _three_to_one(three: str):
    return _THREE_TO_ONE.get(three.upper())


if __name__ == "__main__":
    main()
