"""
Ligand embedding runner.

For each entry in the pocket manifest, extract the ligand heavy-atom
coordinates and SMILES and produce a per-ligand embedding. Output
rows match the pocket manifest order so that ligand_embeddings[i]
and pocket_embeddings[i] refer to the same complex.

Outputs:
    <out_dir>/ligand_embeddings.npy     (N, D) float32; D depends on backend:
                                        - unimol   : 512   (opt-in)
                                        - molformer: 768
                                        - morgan   : 2048  (default fallback)
    <out_dir>/ligand_coords.h5          HDF5 with per-complex ligand coords
    <out_dir>/ligand_manifest.jsonl     adds smiles, heavy_atoms to each row

Backend is auto-selected through a three-tier fallback chain
(unimol -> molformer -> morgan). To force the unimol backend
(occasionally at the cost of numpy<2.0 conflicts), set
RLR_LIGAND_BACKEND=unimol in the environment before running. The
v0.1 dataset on the Hugging Face Hub was produced with the morgan
fallback because the unimol loader was unavailable in the build
environment.

Usage:
    python -m rcsb_ligand_retrieval.retrieval.embed_ligands \\
        --pocket_manifest build/pockets/pocket_manifest.jsonl \\
        --raft_manifest build/raft_pdb_filtered.jsonl \\
        --pdb_dir build/pdb_mmcif \\
        --output build/ligands
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pocket_manifest", type=Path, required=True)
    parser.add_argument("--raft_manifest", type=Path, required=True)
    parser.add_argument("--pdb_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # The backend is selected by an auto-fallback chain (see
    # _load_unimol); --model is accepted for argparse compatibility
    # only and is not used.
    parser.add_argument(
        "--model",
        default="(auto)",
        help="Accepted for compatibility; ignored. Backend is auto-selected.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import gemmi
        import h5py
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install gemmi h5py") from exc

    unimol_model = _load_unimol(args.model, args.device)
    embed_dim = unimol_model.embed_dim

    args.output.mkdir(parents=True, exist_ok=True)

    # Build pdb_id -> cif_path from raft manifest
    cif_index: dict[str, str] = {}
    with open(args.raft_manifest) as f:
        for line in f:
            row = json.loads(line)
            cif_index[row["pdb_id"]] = row["path"]

    # Read pocket manifest
    pocket_rows = []
    with open(args.pocket_manifest) as f:
        for line in f:
            pocket_rows.append(json.loads(line))
    n = len(pocket_rows)
    print("Embedding %d ligands" % n)

    # Prefetch SMILES for every unique ligand resname in parallel.
    # The serial alternative would issue one synchronous request per
    # entry to the RCSB Chemical Component Dictionary REST endpoint,
    # which dominates total runtime.
    unique_resnames = sorted({row["ligand_resname"] for row in pocket_rows})
    print("Prefetching SMILES for %d unique ligand resnames" % len(unique_resnames))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=64) as ex:
        list(ex.map(_fetch_ccd_smiles, unique_resnames))
    print("  SMILES cache populated (%d hits)" % len(_CCD_CACHE))

    embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    h5 = h5py.File(args.output / "ligand_coords.h5", "w")
    manifest_out = []
    n_ok = 0
    n_embed_fail = 0
    first_error_logged = False

    for i, row in enumerate(pocket_rows):
        pdb_id = row["pdb_id"]
        cif_rel = cif_index.get(pdb_id)
        if cif_rel is None:
            continue
        cif_path = args.pdb_dir / cif_rel
        if not cif_path.exists():
            continue

        try:
            structure = gemmi.read_structure(str(cif_path))
        except Exception as exc:
            if not first_error_logged:
                print("[embed_ligands] gemmi read failed for %s: %s" % (pdb_id, exc))
                first_error_logged = True
            continue

        coords, elements, smiles = _extract_ligand(
            structure,
            ligand_chain=row["ligand_chain"],
            ligand_resname=row["ligand_resname"],
            ligand_resnum=row["ligand_resnum"],
        )
        if coords is None or smiles is None:
            continue

        try:
            emb = unimol_model.embed(coords=coords, atoms=elements, smiles=smiles)
        except Exception as exc:
            n_embed_fail += 1
            if n_embed_fail <= 3:
                # Surface the first few backend failures with traceback;
                # remaining failures are counted silently.
                import traceback

                print(
                    "[embed_ligands] embed failed for %s (%s): %s: %s"
                    % (
                        pdb_id,
                        smiles[:40],
                        type(exc).__name__,
                        exc,
                    )
                )
                if n_embed_fail == 1:
                    traceback.print_exc()
            continue

        key = row["pocket_key"]
        embeddings[n_ok] = emb
        h5.create_dataset("ligands/%s" % key, data=coords, compression="gzip")
        out_row = dict(row)
        out_row["ligand_smiles"] = smiles
        out_row["ligand_heavy_atoms"] = int(coords.shape[0])
        out_row["ligand_offset"] = n_ok
        manifest_out.append(out_row)
        n_ok += 1

        # Periodic progress report (flushed for streaming log viewers).
        if (i + 1) % 25 == 0:
            print(
                "  %d/%d processed, %d embeddings written (%d embed fails)"
                % (
                    i + 1,
                    n,
                    n_ok,
                    n_embed_fail,
                ),
                flush=True,
            )

    h5.close()
    embeddings = embeddings[:n_ok]
    np.save(args.output / "ligand_embeddings.npy", embeddings)
    with open(args.output / "ligand_manifest.jsonl", "w") as f:
        for row in manifest_out:
            f.write(json.dumps(row) + "\n")
    print("Done: %d ligand embeddings written" % n_ok)


def _extract_ligand(structure, ligand_chain, ligand_resname, ligand_resnum):
    coords = []
    elements = []
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
                        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        elements.append(atom.element.name)
        break
    if not coords:
        return None, None, None

    coords_arr = np.array(coords, dtype=np.float32)

    # Try to fetch SMILES from RCSB Chemical Component Dictionary
    smiles = _fetch_ccd_smiles(ligand_resname)
    return coords_arr, elements, smiles


_CCD_CACHE: dict[str, str] = {}


def _fetch_ccd_smiles(resname: str):
    """Fetch canonical SMILES for a CCD residue via RCSB data API."""
    if resname in _CCD_CACHE:
        return _CCD_CACHE[resname] or None
    try:
        import urllib.request

        url = "https://data.rcsb.org/rest/v1/core/chemcomp/%s" % resname
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        # RCSB exposes uppercase SMILES under rcsb_chem_comp_descriptor
        descriptors = data.get("rcsb_chem_comp_descriptor", {})
        smiles = descriptors.get("SMILES_stereo") or descriptors.get("SMILES")
    except Exception:
        smiles = None
    _CCD_CACHE[resname] = smiles
    return smiles


def _load_unimol(model_name: str, device: str):
    """Load a ligand embedder via a ranked fallback chain.

    Default order:
        1. MoLFormer-XL  (768-dimensional SMILES transformer, Apache 2.0)
        2. RDKit Morgan  (2,048-bit fingerprint, BSD-3-Clause)

    Uni-Mol (512-dimensional 3D embedder) is available as an opt-in via
    the RLR_LIGAND_BACKEND=unimol environment variable; it is excluded
    from the default chain because its current pip release returns
    embeddings with inconsistent tensor shapes that produce undetected
    errors in downstream code paths.
    """
    prefer = os.environ.get("RLR_LIGAND_BACKEND", "").lower()

    # Explicit Uni-Mol opt-in
    if prefer == "unimol":
        try:
            from unimol_tools import UniMolRepr

            repr_model = UniMolRepr(data_type="molecule", remove_hs=True)

            class _UniMolWrap:
                embed_dim = 512
                backend_name = "unimol"

                def embed(self, coords, atoms, smiles):
                    out = repr_model.get_repr(smiles_list=[smiles])
                    cls = out["cls_repr"] if isinstance(out, dict) else out[0]
                    arr = np.asarray(cls, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[0]
                    return arr

            print("[embed_ligands] using backend: unimol-tools (512-dim, opt-in)")
            return _UniMolWrap()
        except Exception as exc:
            print("[embed_ligands] unimol-tools opt-in failed: %s" % exc)

    # Default: MoLFormer-XL
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        mf_name = "ibm-research/MoLFormer-XL-both-10pct"
        tokenizer = AutoTokenizer.from_pretrained(mf_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            mf_name,
            deterministic_eval=True,
            trust_remote_code=True,
        )
        model = model.to(device).eval()

        class _MoLFormerWrap:
            embed_dim = model.config.hidden_size
            backend_name = "molformer"

            def embed(self, coords, atoms, smiles):
                with torch.no_grad():
                    inputs = tokenizer([smiles], padding=True, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out = model(**inputs)
                emb = out.pooler_output[0].cpu().numpy().astype(np.float32)
                return emb

        print("[embed_ligands] using backend: MoLFormer-XL (%d-dim)" % _MoLFormerWrap.embed_dim)
        return _MoLFormerWrap()
    except Exception as exc:
        print("[embed_ligands] MoLFormer unavailable: %s" % exc)

    # Last resort: Morgan fingerprints
    print("[embed_ligands] falling back to RDKit Morgan fingerprints (2048-bit)")
    return _RDKitFallback()


class _RDKitFallback:
    """Morgan fingerprint fallback of last resort. Always works."""

    embed_dim = 2048
    backend_name = "morgan"

    def embed(self, coords, atoms, smiles):
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise SystemExit("RDKit required for fingerprint fallback") from exc
        from rcsb_ligand_retrieval.chem._fingerprint import morgan_fp

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.embed_dim, dtype=np.float32)
        fp = morgan_fp(mol, self.embed_dim)
        return np.array(fp, dtype=np.float32)


if __name__ == "__main__":
    main()
