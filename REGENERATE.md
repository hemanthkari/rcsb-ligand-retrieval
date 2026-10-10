# Regenerating the v0.2 dataset

Companion code for Zenodo dataset 10.5281/zenodo.22238825. Run in order.
Stages 2-3 need a GPU; 4-7 do not. Stage 4 needs MMseqs2.

1. `pipeline/` - the v0.1 base data, unchanged from the previous version of this
   record: targeted_download -> rcsb_filter -> embed_pockets / embed_ligands ->
   build_index.

2. `notebooks/09_pocket_reextraction.ipynb` - recovers pocket residue indices and
   the exact ESM-2 input sequence per complex, verified by reproducing the
   shipped pocket centroids. NOTE its outputs describe the v0.1 CHIMERIC
   all-chain input, not the v0.2 per-chain one.

3. `notebooks/11_reembed_per_chain.ipynb` - re-embeds each pocket from its own
   chain instead of the all-chain concatenation, and in cells 6-7 emits the
   matching per-chain residue map. This is the substantive change in v0.2. Rows
   it cannot re-embed retain their v0.1 value, so the matrix is a mixture; the
   map's `embedding_input` column records which encoding each row uses.

4. `scripts/build_v02_release.py all --v01_dir <snapshot> --out_dir <release>
   --nb09_dir <stage 2 output> --cluster_backend mmseqs`

   IMPORTANT: no flag supplies the stage-3 embeddings. The script reads
   pockets/pocket_embeddings.npy from --v01_dir. Point --v01_dir at a directory
   of links to the v0.1 snapshot with that one file replaced by the stage-3
   output. `notebooks/13_build_and_validate_v02.ipynb` does exactly this and is
   the reference procedure for reproducing the deposited release.

   Without MMseqs2 the target-disjoint split degrades to exact-sequence matching
   and is NOT homology-disjoint.

5. `notebooks/14_finish_v02.ipynb` - installs the per-chain residue map into the
   release, labels every row with `embedding_input`, and regenerates
   CHECKSUMS.sha256 and DATASET_MANIFEST.json. CPU only.

6. `scripts/verify_v02_release.py --release_dir <release>` - independent checks:
   split disjointness, coordinate-row to SMILES-atom correspondence (needs
   fluxion_dock/data/atom_order.py, included here), ligand bond-length sanity,
   file inventory.

7. `scripts/make_dataset_paper_figures.py --fig all --retrieval_dir <release>
   --pocket_map <release>/pockets/pocket_residue_map.parquet
   --sifts_dir <release>/provenance/sifts --out figures --seed 42`
   Figures 1-6 and the JSON sidecars holding every number the captions quote.
   These are the authoritative values reported in the manuscript.

`notebooks/12_full_revision_audit.ipynb` and `10_remaining_analyses.ipynb`
produce the paired-test and held-out statistics quoted in Technical Validation
(McNemar, bootstrap CIs, permutation nulls). They sample a different query set
from the figure script, so their retrieval rates differ slightly; the figure
values are the ones the manuscript reports.

`notebook_sources/` holds the markdown the notebooks compile from, via
`scripts/build_notebook.py`. Edit those, not the .ipynb files.

`scripts/mirror_to_zenodo.py` deposits the DATASET record. Its metadata block is
specific to that record and must not be reused for this code record.
