"""Download the release from Zenodo and retrieve the nearest pockets to a query.

This is the worked example that used to be printed in the Data Descriptor's
Usage Notes. It is kept here, runnable, so the paper can point at code that is
executed rather than at a listing that is only read.

Target-level retrieval uses the POCKET index, not the retained concatenated
one: the concatenated index is beaten by the pocket index on same-target
retrieval and by the ligand index on same-chemotype retrieval. See "Choosing an
index" in the Data Descriptor's Usage Notes.

Zenodo flattens path separators to "__" in stored file names, so
pockets/pocket_embeddings.npy is stored as pockets__pocket_embeddings.npy.
PATHS.json in the deposit maps every stored key back to its logical path.

Usage:
    python examples/retrieve_similar_pockets.py [--record 22238825] [--k 5]
"""

import argparse
import json
import os

import faiss
import numpy as np
import requests

API = "https://zenodo.org/api/records/%s"


def file_urls(record):
    """Stored key -> download URL for every file in the deposit."""
    resp = requests.get(API % record, timeout=60)
    resp.raise_for_status()
    return dict((f["key"], f["links"]["self"]) for f in resp.json()["files"])


def fetch(urls, key, out_dir="."):
    """Stream one file to disk. Returns the local path.

    Written to <key>.part and renamed only once the body has arrived, so an
    interrupted download can never be mistaken for a complete file.
    """
    dest = os.path.join(out_dir, key)
    if os.path.exists(dest):
        return dest
    part = dest + ".part"
    resp = requests.get(urls[key], stream=True, timeout=300)
    resp.raise_for_status()
    with open(part, "wb") as fh:
        for chunk in resp.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(part, dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default="22238825", help="Zenodo record id")
    ap.add_argument("--k", type=int, default=5, help="neighbours to return")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    urls = file_urls(args.record)

    index = faiss.read_index(fetch(urls, "indices__pocket.faiss", args.out_dir))
    emb = np.load(fetch(urls, "pockets__pocket_embeddings.npy", args.out_dir), mmap_mode="r")
    with open(fetch(urls, "metadata.jsonl", args.out_dir)) as fh:
        meta = [json.loads(line) for line in fh]

    # Use the first indexed complex as the query. Rows are L2-normalised on
    # write, but normalise again so a hand-built query behaves identically.
    query = emb[meta[0]["pocket_emb_offset"]].astype("float32").reshape(1, -1)
    query /= np.linalg.norm(query)

    distances, indices = index.search(query, args.k)
    print("query: %s %s" % (meta[0]["pdb_id"], meta[0]["ligand_resname"]))
    for d, i in zip(distances[0], indices[0]):
        row = meta[i]
        print("  %s  %-5s  cosine %.4f" % (row["pdb_id"], row["ligand_resname"], float(d)))

    # Absolute cosines in this space are anisotropy-dominated: random pocket
    # pairs have raw median cosine 0.719. Rank the neighbours; do not threshold
    # on the raw value.


if __name__ == "__main__":
    raise SystemExit(main())
