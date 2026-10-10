"""RCSB Ligand Retrieval Pipeline.

Pipeline that constructs a retrieval index of protein-ligand complexes
from the Research Collaboratory for Structural Bioinformatics (RCSB)
Protein Data Bank, with ESM-2 pocket embeddings, RDKit Morgan ligand
fingerprints, and a FAISS hierarchical-navigable-small-world index
over the joined unit-normalized representation.

Released alongside the dataset of 72,388 protein-ligand complexes
hosted at
https://huggingface.co/datasets/HemanthKari/rcsb-ligand-retrieval-v0.1
under CC BY 4.0.
"""

__version__ = "0.1.0"
