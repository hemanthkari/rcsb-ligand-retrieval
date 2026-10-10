"""
FluxionDock -- fetch protein sequences from RCSB PDB.

The Week 1 pipeline extracted pocket embeddings via ESM-2 but did not
persist the raw protein sequences in any manifest. Boltz-1 needs
the full protein sequence as input, so we fetch them from RCSB's
FASTA endpoint at extraction time.

RCSB FASTA API:
    https://www.rcsb.org/fasta/entry/{PDB_ID}
    Returns multi-chain FASTA. We pick the longest protein chain.

Usage:
    from rcsb_ligand_retrieval.data.fetch_sequences import fetch_sequences_bulk
    seq_map = fetch_sequences_bulk(["1abc", "2def", ...])
    print(seq_map["1abc"])  # "MKTAYIAK..."
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/%s"


def fetch_fasta(pdb_id: str, retries: int = 3) -> Optional[str]:
    """Fetch FASTA text for one PDB entry. Returns None on failure."""
    url = RCSB_FASTA_URL % pdb_id.upper()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
    return None


def parse_longest_protein_chain(fasta_text: str) -> str:
    """Parse multi-chain FASTA and return the longest protein sequence."""
    sequences = []
    current_header = ""
    current_seq = []

    for line in fasta_text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                sequences.append((current_header, "".join(current_seq)))
            current_header = line
            current_seq = []
        elif line:
            current_seq.append(line)
    if current_seq:
        sequences.append((current_header, "".join(current_seq)))

    protein_seqs = []
    for header, seq in sequences:
        # Skip DNA/RNA chains (contain lowercase or non-standard chars)
        is_protein = all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq.upper())
        if is_protein and len(seq) >= 20:
            protein_seqs.append(seq)

    if not protein_seqs:
        return ""
    return max(protein_seqs, key=len)


def fetch_sequences_bulk(
    pdb_ids: list,
    max_workers: int = 8,
    progress_every: int = 100,
) -> dict:
    """Fetch protein sequences for a list of PDB IDs in parallel.

    Returns dict mapping pdb_id -> protein_sequence (longest chain).
    Missing/failed entries are omitted from the result.
    """
    unique_ids = list(set(pdb_ids))
    results = {}
    n_total = len(unique_ids)

    print("Fetching sequences for %d PDB entries from RCSB..." % n_total, flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_fasta, pdb_id): pdb_id for pdb_id in unique_ids}
        n_done = 0
        for future in as_completed(futures):
            pdb_id = futures[future]
            n_done += 1
            try:
                fasta_text = future.result()
                if fasta_text:
                    seq = parse_longest_protein_chain(fasta_text)
                    if seq:
                        results[pdb_id] = seq
            except Exception:
                pass

            if n_done % progress_every == 0 or n_done == n_total:
                print(
                    "  fetched %d / %d sequences (%d OK)" % (n_done, n_total, len(results)),
                    flush=True,
                )

    print("Sequences: %d / %d fetched successfully" % (len(results), n_total), flush=True)
    return results
