"""
FAISS retrieval index builder.

Given pocket and ligand embeddings produced by embed_pockets.py and
embed_ligands.py, build an HNSW FAISS index for nearest-neighbor retrieval.

The query vector is the L2-normalized concatenation of pocket + ligand
embeddings, so a single index handles both "similar pocket" and "similar
ligand" as a joint relevance score.

Outputs:
    <out_dir>/retrieval.faiss           FAISS HNSW index
    <out_dir>/metadata.jsonl            one row per index entry, joins
                                        pocket + ligand manifests

Usage:
    python -m rcsb_ligand_retrieval.retrieval.build_index \\
        --pocket_dir data/retrieval_index/pockets/ \\
        --ligand_dir data/retrieval_index/ligands/ \\
        --output data/retrieval_index/ \\
        --index_type HNSW --M 32 --efConstruction 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket_dir", type=Path, required=True)
    parser.add_argument("--ligand_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index_type", default="HNSW", choices=["HNSW", "Flat", "IVF"])
    parser.add_argument("--M", type=int, default=32, help="HNSW graph degree")
    parser.add_argument("--efConstruction", type=int, default=200)
    parser.add_argument("--efSearch", type=int, default=128)
    args = parser.parse_args()

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("pip install faiss-cpu") from exc

    args.output.mkdir(parents=True, exist_ok=True)

    pocket_emb = np.load(args.pocket_dir / "pocket_embeddings.npy")
    ligand_emb = np.load(args.ligand_dir / "ligand_embeddings.npy")

    # Load manifests and key both by pocket_key — the natural primary
    # key for the (pdb_id, ligand_chain, ligand_resname, ligand_resnum)
    # tuple. Indexing both embedding matrices by a shared loop counter
    # is incorrect whenever any ligand fails to embed, because pocket
    # and ligand offsets diverge from that point onward. Pocket and
    # ligand embeddings must be looked up via their respective offset
    # fields, joined on pocket_key.
    pocket_by_key: dict[str, dict] = {}
    with open(args.pocket_dir / "pocket_manifest.jsonl") as f:
        for line in f:
            row = json.loads(line)
            pocket_by_key[row["pocket_key"]] = row

    ligand_by_key: dict[str, dict] = {}
    with open(args.ligand_dir / "ligand_manifest.jsonl") as f:
        for line in f:
            row = json.loads(line)
            ligand_by_key[row["pocket_key"]] = row

    common_keys = sorted(set(pocket_by_key) & set(ligand_by_key))
    joined_embeddings = []
    joined_metadata = []
    n_skipped_zero_ligand = 0
    for key in common_keys:
        prow = pocket_by_key[key]
        lrow = ligand_by_key[key]
        p_off = prow["offset"]
        l_off = lrow["ligand_offset"]

        # Skip entries whose ligand embedding is the zero vector. RDKit's
        # Morgan generator returns an empty bitvector for compounds with no
        # recognizable organic substructure (transition-metal complexes,
        # polyborane clusters, polyoxometalates, etc.), and a zero-vector
        # ligand portion makes the joined embedding degenerate as a
        # joint-modality retrieval anchor. Per-complex pocket and ligand
        # embeddings (and HDF5 coordinates) for these entries remain
        # available; only the joined index excludes them.
        if float(np.linalg.norm(ligand_emb[l_off])) < 1e-6:
            n_skipped_zero_ligand += 1
            continue

        joint = np.concatenate([pocket_emb[p_off], ligand_emb[l_off]]).astype(np.float32)
        norm = np.linalg.norm(joint) + 1e-12
        joined_embeddings.append(joint / norm)
        joined_metadata.append(
            {
                "pdb_id": prow["pdb_id"],
                "ligand_resname": prow["ligand_resname"],
                "ligand_chain": prow["ligand_chain"],
                "ligand_resnum": prow["ligand_resnum"],
                "pocket_key": prow["pocket_key"],
                "ligand_smiles": lrow["ligand_smiles"],
                "ligand_heavy_atoms": lrow["ligand_heavy_atoms"],
                "pocket_n_residues": prow["pocket_n_residues"],
                "pocket_emb_offset": p_off,
                "ligand_emb_offset": l_off,
            }
        )

    if n_skipped_zero_ligand:
        print(
            "Skipped %d entries with zero-norm ligand embedding "
            "(degenerate Morgan fingerprint; entries remain available in "
            "ligand_embeddings.npy and ligand_coords.h5 for downstream use)."
            % n_skipped_zero_ligand
        )

    if not joined_embeddings:
        raise SystemExit("No aligned embeddings found. Did pocket and ligand embeds run?")

    matrix = np.stack(joined_embeddings).astype(np.float32)
    dim = matrix.shape[1]
    print("Index data: %d x %d" % (matrix.shape[0], dim))

    if args.index_type == "HNSW":
        index = faiss.IndexHNSWFlat(dim, args.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.efConstruction
        index.hnsw.efSearch = args.efSearch
    elif args.index_type == "Flat":
        index = faiss.IndexFlatIP(dim)
    elif args.index_type == "IVF":
        nlist = max(64, int(np.sqrt(matrix.shape[0])))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
    else:
        raise ValueError("Unknown index type: %s" % args.index_type)

    index.add(matrix)
    faiss.write_index(index, str(args.output / "retrieval.faiss"))

    with open(args.output / "metadata.jsonl", "w") as f:
        for row in joined_metadata:
            f.write(json.dumps(row) + "\n")

    np.save(args.output / "joined_embeddings.npy", matrix)

    print("Done. FAISS index: %s (%d entries)" % (args.output / "retrieval.faiss", matrix.shape[0]))
    print("Metadata: %s" % (args.output / "metadata.jsonl"))


if __name__ == "__main__":
    main()
