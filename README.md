# rcsb-ligand-retrieval

Construction, validation and deposit code for **Pocket and ligand embeddings for
72,199 protein–ligand complexes from the Protein Data Bank** — a retrieval
resource of ESM-2 pocket embeddings, Morgan ligand fingerprints, FAISS indices
and per-complex coordinates, built end to end from commercially redistributable
sources.

- **Dataset:** <https://doi.org/10.5281/zenodo.22238824> (concept DOI — always
  resolves to the current version)
- **Archived code:** <https://doi.org/10.5281/zenodo.22258470> (MIT, citable)
- **Licence:** MIT

This repository is the browsable copy. The Zenodo archive is the citable one:
cite the DOI, read the code here.

## Layout

```
rcsb_ligand_retrieval/   the construction pipeline (filter, embed, index)
scripts/                 release assembly, verification and deposit
notebooks/               pocket re-extraction and per-chain re-embedding
notebook_sources/        the .md sources those notebooks are generated from
examples/                worked usage, runnable against the released data
REGENERATE.md            the order the stages run in
```

## Examples

Two scripts, both runnable against the deposit:

```bash
python examples/retrieve_similar_pockets.py            # download + nearest pockets
python examples/read_coordinates.py --local_dir <release root>
```

`retrieve_similar_pockets.py` queries `indices/pocket.faiss`, not the retained
concatenated index — the pocket index answers "same target" and the ligand index
answers "same chemotype", and the concatenation is worse than either at its own
task. `read_coordinates.py` shows the asymmetry between the two coordinate
arrays: the pocket side stores one heavy-atom **centroid per residue**, the
ligand side stores **per-atom** coordinates.

## Two things that will bite you

**Stored file names flatten path separators.** Zenodo object keys cannot contain
`/`, so `pockets/pocket_embeddings.npy` is stored as
`pockets__pocket_embeddings.npy`. `PATHS.json` in the deposit maps every stored
key back to its logical path.

**Two token-index conventions exist.** The `token_index` field inside
`pocket_coords.h5` indexes the *preview's* all-chain concatenation. The
positions matching the shipped per-chain embeddings are
`pocket_res_token_position` in `pockets/pocket_residue_map.parquet`. Pooling at
the HDF5 field will not reproduce the released vectors for any multi-chain
entry.

## Reproducing

`REGENERATE.md` gives the stage order. The release assembler and its independent
verifier are deliberately separate programs — a builder that validates its own
output can only confirm what its author already believed:

```bash
python scripts/build_v02_release.py --self_test     # offline logic checks
python scripts/verify_v02_release.py --release_dir <release root>
```

Figures must be built against the SIFTS snapshot the deposit ships, not a fresh
download — SIFTS is a live service and its UniProt mappings move:

```bash
python scripts/make_dataset_paper_figures.py --fig 5 \
    --retrieval_dir <release root> \
    --sifts_dir <release root>/provenance/sifts --no_download
```

## Citing

Cite the dataset by its **version** DOI for reproducible work, and the concept
DOI only where "the latest version" is what you mean. They are not
interchangeable. Please also cite the upstream sources: the RCSB Protein Data
Bank, BindingDB, and ESM-2.
