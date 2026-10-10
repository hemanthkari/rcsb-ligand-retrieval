"""
Targeted RCSB PDB download.

Instead of rsyncing the full 80 GB mmCIF mirror, query the RCSB Search API
for entries matching the pipeline's filter criteria:

  - Experimental method: X-RAY DIFFRACTION (exclude computed, EM, NMR)
  - Resolution <= max_resolution (default 2.5 A)
  - Has at least one non-polymer entity (small-molecule ligand)
  - Protein chain present
  - Optional: released after/before a date for time-split selection

Then download only the matching entries via HTTPS from files.rcsb.org.

On hosts with ~70 GB free disk this avoids the 80 GB rsync mirror that
otherwise will not fit.

Usage:
    python -m rcsb_ligand_retrieval.data.targeted_download \\
        --output_dir build/pdb_mmcif \\
        --max_resolution 2.5 \\
        --released_before 2024-01-01 \\
        --max_entries 80000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def _terminal(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": attribute,
            "operator": operator,
            "value": value,
        },
    }


def _wrap_query(nodes: list) -> dict:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
        "return_type": "entry",
        "request_options": {
            # return_all_hits and paginate are mutually exclusive.
            "return_all_hits": True,
        },
    }


def _build_query_tiers(max_resolution: float) -> list:
    """Build a list of progressively less strict queries.

    Execution falls through the tiers until one returns a non-empty
    result. The most specific query is tried first; if it returns zero
    hits (probably because one filter name is wrong), we try simpler
    versions that are less sensitive to schema drift. Whatever entries
    pass downstream gemmi-level filtering (experimental method, etc)
    are retained by rcsb_filter.py.

    Attribute names verified against the rcsb-api Python package
    (March 2026 release), which is the authoritative JSON schema source:
      - exptl.method                               (experimental method)
      - rcsb_entry_info.resolution_combined        (resolution in A)
      - rcsb_entry_info.nonpolymer_entity_count    (count of small molecules)
    """
    res = float(max_resolution)

    tier_full = [
        _terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        _terminal("rcsb_entry_info.resolution_combined", "less_or_equal", res),
        _terminal("rcsb_entry_info.nonpolymer_entity_count", "greater", 0),
    ]
    tier_no_method = [
        _terminal("rcsb_entry_info.resolution_combined", "less_or_equal", res),
        _terminal("rcsb_entry_info.nonpolymer_entity_count", "greater", 0),
    ]
    tier_minimal = [
        _terminal("rcsb_entry_info.resolution_combined", "less_or_equal", res),
    ]
    return [
        ("full (method+resolution+ligand)", _wrap_query(tier_full)),
        ("no method (resolution+ligand)", _wrap_query(tier_no_method)),
        ("minimal (resolution only)", _wrap_query(tier_minimal)),
    ]


# Kept for backwards compatibility with any callers
def _default_query(
    max_resolution: float, released_before: str = None, released_after: str = None
) -> dict:
    return _build_query_tiers(max_resolution)[0][1]


def _execute_query(query: dict) -> tuple:
    """POST the query to RCSB and return (status, raw_bytes, parsed_dict_or_none)."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "(no body available)"
        raise SystemExit(
            "RCSB search API %d %s\nresponse body:\n%s"
            % (
                exc.code,
                exc.reason,
                body[:4000],
            )
        )
    except Exception as exc:
        raise SystemExit("RCSB search API request failed: %s" % exc)

    if not raw.strip():
        return status, raw, None
    try:
        return status, raw, json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "RCSB returned non-JSON body (status=%d): %s\nfirst 500 bytes: %r"
            % (
                status,
                exc,
                raw[:500],
            )
        )


def search_rcsb(
    max_resolution: float, released_before: str = None, released_after: str = None
) -> list:
    """Return the list of PDB IDs matching the query.

    Walks progressively less-strict query tiers and returns the result
    from the first tier that matches entries. If a tier returns HTTP 204
    (zero hits), we assume one of its filters has a schema mismatch and
    try the next simpler tier. Downstream filtering (experimental
    method, ligand drug-likeness, release date) is re-applied in Python
    by rcsb_filter.py once we have the per-entry metadata.

    Date filters are intentionally applied in Python later, not at the
    Search API level, since the release_date attribute has had
    inconsistent behavior across API versions.
    """
    tiers = _build_query_tiers(max_resolution)

    for i, (label, query) in enumerate(tiers):
        print("Tier %d: %s" % (i, label))
        if i == 0:
            print("Query JSON:")
            print(json.dumps(query, indent=2))
        status, raw, result = _execute_query(query)
        print("  HTTP %d, content-length=%d" % (status, len(raw)))
        if result is None:
            print("  empty body -> zero hits, trying next tier")
            continue
        ids = [entry["identifier"].lower() for entry in result.get("result_set", [])]
        total = result.get("total_count", len(ids))
        print(
            "RCSB search returned %d hits via tier '%s' (total_count=%d)"
            % (
                len(ids),
                label,
                total,
            )
        )
        if not ids:
            print("  zero ids in result_set; trying next tier")
            continue
        return ids

    print("All %d tiers returned zero hits. API is broken or query schema changed." % len(tiers))
    return []


def download_one(pdb_id: str, output_dir: Path, retries: int = 3) -> bool:
    """Download one mmCIF.gz file from files.rcsb.org. Returns True on success."""
    dest = output_dir / pdb_id[1:3] / ("%s.cif.gz" % pdb_id)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = "https://files.rcsb.org/download/%s.cif.gz" % pdb_id.upper()
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception as exc:
            if attempt == retries - 1:
                print("  FAIL %s: %s" % (pdb_id, exc), file=sys.stderr)
                return False
            time.sleep(1 + attempt * 2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Targeted RCSB download via Search API (disk-safe, parallel)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root directory to store cif.gz files (two-letter subdirs)",
    )
    parser.add_argument("--max_resolution", type=float, default=2.5)
    parser.add_argument("--released_before", type=str, default=None, help="ISO date (YYYY-MM-DD)")
    parser.add_argument("--released_after", type=str, default=None)
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Cap the number of entries to download (useful for testing)",
    )
    parser.add_argument(
        "--ids_only",
        action="store_true",
        help="Only write the list of matching PDB IDs, do not download",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel HTTP download workers (default 16)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ids = search_rcsb(
        max_resolution=args.max_resolution,
        released_before=args.released_before,
        released_after=args.released_after,
    )
    if args.max_entries is not None:
        ids = ids[: args.max_entries]

    id_list_path = args.output_dir / "matched_ids.txt"
    with open(id_list_path, "w") as f:
        for pid in ids:
            f.write(pid + "\n")
    print("Wrote %d PDB IDs to %s" % (len(ids), id_list_path))

    if args.ids_only:
        return

    print("Downloading %d CIFs with %d parallel workers..." % (len(ids), args.workers))
    n_ok, n_fail = _download_parallel(ids, args.output_dir, args.workers)

    total_bytes = 0
    for cif in args.output_dir.rglob("*.cif.gz"):
        total_bytes += cif.stat().st_size
    print(
        "Done: %d ok, %d fail, %.2f GB on disk"
        % (
            n_ok,
            n_fail,
            total_bytes / 1e9,
        )
    )


def _download_parallel(ids: list, output_dir: Path, workers: int) -> tuple:
    """Parallel HTTPS download with progress reporting and ETA.

    Uses concurrent.futures.ThreadPoolExecutor. Downloading is I/O-bound
    (TLS handshake plus a transfer of order 200 KB followed by close), so
    threads suffice in CPython because the GIL is released during the
    underlying urllib socket reads. The default of 16 workers is well
    within the request rate that the RCSB file service tolerates.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_ok = 0
    n_fail = 0
    t0 = time.time()
    report_every = max(100, len(ids) // 50)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, pid, output_dir): pid for pid in ids}
        for i, fut in enumerate(as_completed(futures)):
            try:
                ok = fut.result()
            except Exception as exc:
                ok = False
                print("  worker exception on %s: %s" % (futures[fut], exc), file=sys.stderr)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if (i + 1) % report_every == 0 or (i + 1) == len(ids):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(ids) - i - 1) / max(rate, 1e-6)
                print(
                    "  %d/%d  ok=%d fail=%d  (%.1f files/s, eta %.0fs = %.1fh)"
                    % (
                        i + 1,
                        len(ids),
                        n_ok,
                        n_fail,
                        rate,
                        eta,
                        eta / 3600,
                    )
                )

    return n_ok, n_fail


if __name__ == "__main__":
    main()
