"""
Deposit the RCSB Protein-Ligand Retrieval Index (v0.2) to Zenodo as the
PRIMARY, policy-compliant record for the Scientific Data Data Descriptor.

Nature Scientific Data requires deposition in an approved repository.
Hugging Face is not on their recommended list; Zenodo is. This script makes
Zenodo the record of reference and demotes the Hugging Face copy to a
declared mirror (related_identifier: isIdenticalTo).

What it does
------------
1. Materializes the payload: either an existing local directory
   (--source_dir) or a snapshot pulled from the Hugging Face mirror
   (--hf_repo, downloaded into --source_dir).
2. Hashes every payload file once (SHA-256 + MD5 in a single pass) and
   verifies SHA-256 against CHECKSUMS.sha256 BEFORE any byte is uploaded.
   A mismatch, a missing file, or an unlisted file aborts the run.
3. Builds and validates the Zenodo deposition metadata (upload_type
   dataset, CC-BY-4.0, related_identifiers wiring the record to the HF
   mirror, the paper, PDB, BindingDB and the pipeline-code DOI).
4. Uploads through the Zenodo REST bucket API
   (PUT {bucket_url}/{key}) with bounded exponential backoff, jitter,
   Retry-After support, and *resume*: before every attempt the remote
   bucket is re-listed and any object whose MD5 already matches the local
   file is skipped. Zenodo routinely 502s AFTER it has durably stored a
   multi-hundred-MB object; without that re-check a retry would re-push
   the whole file.
5. Optionally cuts a new version of an existing concept DOI so v0.1 keeps
   its own version DOI, is marked superseded (obsoletes / isObsoletedBy),
   and only genuinely changed files are re-uploaded -- pocket_embeddings.npy
   and ligand_embeddings.npy are byte-identical between v0.1 and v0.2, so
   the new version reuses them instead of re-sending ~1 GB.

The v0.2 payload is ~4.8 GB, comfortably under Zenodo's 50 GB per-record
limit, so no quota request is needed.

The access token is read from the ZENODO_TOKEN environment variable, is
sent only in an Authorization header (never a query string, which servers
log), and is redacted from every message this script prints.

Usage
-----
    # validate metadata + checksums and print the plan, no network:
    python scripts/mirror_to_zenodo.py --source_dir data/zenodo_v0.2 --dry_run

    # snapshot the HF mirror, then create a fresh Zenodo draft:
    export ZENODO_TOKEN=...
    python scripts/mirror_to_zenodo.py \
        --hf_repo HemanthKari/rcsb-ligand-retrieval-v0.1 \
        --source_dir data/zenodo_v0.2 --version v0.2

    # new version of an existing concept DOI, then publish:
    python scripts/mirror_to_zenodo.py --source_dir data/zenodo_v0.2 \
        --new_version 10.5281/zenodo.XXXXXXX --version v0.2 \
        --mark_superseded --publish

    # end-to-end rehearsal against sandbox.zenodo.org (separate token):
    python scripts/mirror_to_zenodo.py --source_dir data/zenodo_v0.2 --sandbox

    # after acceptance: add isSupplementTo to the PUBLISHED record. A published
    # record's files are immutable but its metadata is not, so this is an
    # edit + re-publish, NOT a new version (a new version would mint a second
    # dataset DOI for identical bytes):
    python scripts/mirror_to_zenodo.py --metadata_only \
        --deposition <id> --paper_doi 10.1038/s41597-026-XXXXX-X

    # logic check, no network, no data:
    python scripts/mirror_to_zenodo.py --self_test

Payload note
------------
`scripts/build_v02_release.py` writes CHECKSUMS.sha256 and only afterwards
writes DATASET_MANIFEST.json, so that one file is never covered by the
manifest. It is uploaded, and reported as uncovered, rather than aborting the
run; see MANIFEST_EXEMPT. Every other released file is verified before upload.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ZENODO_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"

HF_DATASET_URL = "https://huggingface.co/datasets/HemanthKari/rcsb-ligand-retrieval-v0.1"
HF_DATASET_DOI = "10.57967/hf/8408"
PIPELINE_CODE_DOI = "10.5281/zenodo.22258470"


def _measured():
    """The builder's MEASURED block, so one edit moves every rendered number."""
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from build_v02_release import MEASURED

    return MEASURED


_M = _measured()


def _pct(x):
    """0.812 -> "81.2". One decimal, which is what the description uses."""
    return "%.1f" % (100.0 * x)


PDB_URL = "https://www.rcsb.org"
BINDINGDB_URL = "https://www.bindingdb.org"

# Per-record and per-file ceilings on zenodo.org (2026). We only warn.
ZENODO_RECORD_LIMIT_BYTES = 50 * 1000**3
ZENODO_FILE_LIMIT_BYTES = 50 * 1000**3

# Relation types accepted by the Zenodo deposit API (DataCite set).
ZENODO_RELATIONS = frozenset(
    [
        "isCitedBy",
        "cites",
        "isSupplementTo",
        "isSupplementedBy",
        "isContinuedBy",
        "continues",
        "isDescribedBy",
        "describes",
        "hasMetadata",
        "isMetadataFor",
        "isNewVersionOf",
        "isPreviousVersionOf",
        "isPartOf",
        "hasPart",
        "isReferencedBy",
        "references",
        "isDocumentedBy",
        "documents",
        "isCompiledBy",
        "compiles",
        "isVariantFormOf",
        "isOriginalFormOf",
        "isIdenticalTo",
        "isAlternateIdentifier",
        "isReviewedBy",
        "reviews",
        "isDerivedFrom",
        "isSourceOf",
        "requires",
        "isRequiredBy",
        "isObsoletedBy",
        "obsoletes",
    ]
)

ZENODO_UPLOAD_TYPES = frozenset(
    [
        "publication",
        "poster",
        "presentation",
        "dataset",
        "image",
        "video",
        "software",
        "lesson",
        "physicalobject",
        "other",
    ]
)

# HTTP statuses worth retrying. Zenodo's proxy layer emits 502/504 on large
# PUTs; 429 is rate limiting; 500/503 are transient backend faults.
RETRY_STATUS = frozenset([408, 429, 500, 502, 503, 504])

# Never shipped to Zenodo regardless of what is on disk.
DEFAULT_EXCLUDES = (
    ".git",
    ".git/*",
    ".git/**",
    ".gitattributes",
    ".gitignore",
    ".cache",
    ".cache/*",
    ".cache/**",
    ".huggingface",
    ".huggingface/*",
    ".huggingface/**",
    "__pycache__",
    "__pycache__/*",
    "**/__pycache__/**",
    "*.pyc",
    "*.lock",
    "*.tmp",
    "*.part",
    ".DS_Store",
    "Thumbs.db",
    ".zenodo_state.json",
)

CHECKSUMS_NAME = "CHECKSUMS.sha256"
DATASET_MANIFEST_NAME = "DATASET_MANIFEST.json"
PATHS_MANIFEST_NAME = "PATHS.json"
STATE_NAME = ".zenodo_state.json"

# Files that cannot appear in CHECKSUMS.sha256 on a clean build, because the
# builder writes them AFTER the manifest.
#   CHECKSUMS.sha256      -- cannot contain its own digest.
#   DATASET_MANIFEST.json -- scripts/build_v02_release.py stage 8 hashes the
#                            tree, writes CHECKSUMS.sha256, and only then writes
#                            DATASET_MANIFEST.json (see _iter_release_files,
#                            whose `skip` set covers CHECKSUMS.sha256 only).
#   PATHS.json            -- generated by this script under --flatten.
#   .zenodo_state.json    -- written by this script; also in DEFAULT_EXCLUDES.
# They are exempt from the UNLISTED check only. A digest MISMATCH on any of
# them is still fatal.
MANIFEST_EXEMPT = frozenset(
    [
        CHECKSUMS_NAME,
        DATASET_MANIFEST_NAME,
        PATHS_MANIFEST_NAME,
        STATE_NAME,
    ]
)

DEFAULT_TITLE = (
    "Pocket and ligand embeddings for 72,199 protein-ligand complexes " "from the Protein Data Bank"
)

DEFAULT_KEYWORDS = [
    "protein-ligand complexes",
    "Protein Data Bank",
    "protein language model",
    "ESM-2",
    "molecular fingerprints",
    "binding site",
    "similarity search",
    "structure-based drug discovery",
    "retrieval",
    "embeddings",
]

# The description is deliberately explicit about (a) which record is
# authoritative, (b) that the pocket and ligand modalities are co-equal,
# and (c) that v0.1's content was regenerated once under a static DOI.
# Point (c) is a citability defect; it belongs in the record, not in a
# private note.
DESCRIPTION_TEMPLATE = """\
<p>Pocket and ligand embeddings for %(n_entries)s protein-ligand complexes
drawn from the Protein Data Bank (X-ray diffraction, resolution &le; 2.5 &Aring;,
one ligand selected per entry by a documented heavy-atom-count filter of 10 to
70 heavy atoms). Each complex carries a 1,280-dimensional ESM-2 pocket
embedding, mean-pooled over binding-site residues within 8 &Aring; of the
ligand; a 2,048-bit RDKit Morgan (ECFP4, radius 2, no stereochemistry) ligand
fingerprint; coordinates in HDF5 -- <em>per-residue heavy-atom centroids</em>
for the pocket (one row per pocket residue, not per atom and not
C&alpha;) and per-atom heavy-atom coordinates for the ligand, both with
aligned identifier labels; and the residue-to-token mapping needed to reproduce
the pocket embedding from the released files.</p>

<p><strong>Selection caveat.</strong> 56.1&nbsp;%% of entries contained more
than one ligand satisfying the filter and the first qualifying candidate was
taken; 25.8&nbsp;%% of the selected ligands are common cofactors or buffers.
Both facts are now carried per row (<code>n_qualifying_ligands</code>,
<code>is_common_cofactor</code>) so the corpus can be re-filtered rather than
taken as a drug-like ligand set.</p>

<p><strong>This Zenodo record is the version of record for the
dataset.</strong> It is the deposit the accompanying Data Descriptor
documents, and the only one its analyses were computed from.</p>

<p><strong>Pocket and ligand are released as co-equal, separately indexed
modalities.</strong> Measured on this corpus, pocket-only retrieval recovers a
same-UniProt complex at rank 1 for %(pocket_uniprot)s&nbsp;%% of queries
against %(concat_uniprot)s&nbsp;%% for the concatenated pocket+ligand vector and
%(ligand_uniprot)s&nbsp;%% for ligand-only, while for chemotype search the
ordering reverses: same-CCD at rank 1 is %(ligand_ccd)s&nbsp;%% ligand-only
against %(pocket_ccd)s&nbsp;%% pocket-only. The concatenated
index is retained and documented, but it is intermediate on every task measured
and better at none; it should be treated as one option rather than as the
primary artifact.</p>

<p><strong>Version %(version)s.</strong> %(version_note)s</p>

<p><strong>Provenance and licensing.</strong> Atomic coordinates derive from
the Protein Data Bank (CC0 1.0). The auxiliary affinity table is restricted to
the staff-curated subset of BindingDB (CC BY 3.0); rows imported from ChEMBL are
filtered at build time. Embeddings are produced with ESM-2 (MIT) and RDKit
(BSD-3-Clause); indexes with FAISS (MIT); structure parsing with gemmi (MPL-2.0).
No non-commercial or share-alike source is included in or required by the
dataset. The build pipeline is archived separately under MIT at
<a href="https://doi.org/%(code_doi)s">https://doi.org/%(code_doi)s</a>.</p>

<p><strong>Integrity.</strong> %(n_files)d files, %(total_human)s. SHA-256
checksums are in <code>%(checksums_name)s</code>, covering every released file
except <code>%(checksums_name)s</code> itself and
<code>%(dataset_manifest_name)s</code> (which the builder writes after the
manifest), and were verified against the payload immediately before
upload.</p>
"""

DEFAULT_VERSION_NOTE = (
    "v0.2 adds the residue-to-token mapping, element symbols and CCD atom "
    "names for ligand coordinates, a corrected train/validation split and a "
    "target-disjoint split, a de-duplicated CCD-to-fingerprint table, and "
    "ligand-selection provenance flags. Embedding values are unchanged: "
    "pocket_embeddings.npy and ligand_embeddings.npy are byte-identical to "
    "v0.1. v0.1 is superseded and retained only for citation continuity."
)

# Disclosure carried in the Zenodo `notes` field. Parts of the v0.1 payload
# behind the Hugging Face DOI were regenerated in place between the 2026-04-15
# initial release and 2026-05-01; a mutable artifact behind a minted DOI is a
# citability problem and is stated, not hidden. Zenodo records are immutable
# per version, which is the fix. The file list matches the dataset card's own
# changelog (scripts/update_hf_readme.py) so the two agree rather than merely
# coexist -- naming the files is what makes the disclosure actionable for
# someone checking a local copy.
DEFAULT_NOTES = (
    "Versioning disclosure: the content distributed under the Hugging Face "
    "identifier %s (dataset v0.1) was revised in place between the initial "
    "release on 2026-04-15 and 2026-05-01, without a change of identifier. "
    "metadata.jsonl, metadata.parquet, joined_embeddings.npy, retrieval.faiss "
    "and the train/validation split arrays were regenerated -- first to add "
    "the protein_seq field and the split arrays, then to correct a "
    "row-position join in the index builder and to drop the 189 zero-norm "
    "ligand rows from the joined index. The per-complex pocket and ligand "
    "embedding matrices and the HDF5 coordinate files were not changed. Users "
    "who obtained v0.1 before 2026-05-01 may therefore hold derived files that "
    "differ from those the identifier resolved to afterwards. Zenodo records "
    "are immutable per version: each version of this dataset receives its own "
    "version DOI, and the concept DOI always resolves to the latest version. "
    "Cite the version DOI for reproducible work." % HF_DATASET_DOI
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def human_size(n: int) -> str:
    """Human-readable byte count using decimal units (as Zenodo reports)."""
    step = 1000.0
    val = float(n)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if val < step or unit == "TB":
            if unit == "B":
                return "%d B" % int(val)
            return "%.1f %s" % (val, unit)
        val /= step
    return "%.1f TB" % val


class _Redactor(object):
    """Replaces the access token with a placeholder in anything printed."""

    def __init__(self) -> None:
        self._secrets = []

    def add(self, secret) -> None:
        if secret and len(str(secret)) >= 8:
            self._secrets.append(str(secret))

    def __call__(self, text) -> str:
        s = str(text)
        for secret in self._secrets:
            s = s.replace(secret, "<ZENODO_TOKEN:redacted>")
        return s


REDACT = _Redactor()


def log(msg) -> None:
    print(REDACT(msg), flush=True)


def backoff_delays(max_retries: int, base: float, cap: float = 120.0):
    """Exponential backoff schedule (seconds), deterministic, capped.

    Returned list has max_retries - 1 entries: the delay taken *after* each
    failed attempt except the last (after which we give up).
    """
    out = []
    for i in range(max(0, max_retries - 1)):
        out.append(min(cap, base * (2.0**i)))
    return out


def hash_file(path: Path, chunk: int = 8 << 20):
    """One pass over the file -> (sha256_hex, md5_hex, size_bytes)."""
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with open(str(path), "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            sha.update(block)
            md5.update(block)
            size += len(block)
    return sha.hexdigest(), md5.hexdigest(), size


# --------------------------------------------------------------------------
# Payload discovery and checksum verification
# --------------------------------------------------------------------------


def is_excluded(rel: str, patterns) -> bool:
    """True if the POSIX-style relative path matches any exclude glob."""
    parts = rel.split("/")
    for pat in patterns:
        if fnmatch.fnmatch(rel, pat):
            return True
        for part in parts:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def discover_files(root: Path, excludes):
    """Return sorted POSIX-relative paths of every payload file under root."""
    out = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames.sort()
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""
        keep_dirs = []
        for d in dirnames:
            rel = (rel_dir + "/" + d).lstrip("/")
            if not is_excluded(rel, excludes):
                keep_dirs.append(d)
        dirnames[:] = keep_dirs
        for name in sorted(filenames):
            rel = (rel_dir + "/" + name).lstrip("/")
            if is_excluded(rel, excludes):
                continue
            out.append(rel)
    out.sort()
    return out


def parse_checksums_file(path: Path):
    """Parse `sha256sum` output into {relative_posix_path: sha256_hex}.

    Accepts both text ("hash  path") and binary ("hash *path") markers,
    tolerates comments, blank lines, a UTF-8 BOM, and backslash separators.
    """
    mapping = {}
    text = path.read_text(encoding="utf-8-sig")
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([0-9a-fA-F]{64})\s+[\*\s]?(.+)$", line)
        if not m:
            raise ValueError("%s line %d is not a sha256sum record: %r" % (path.name, lineno, raw))
        digest = m.group(1).lower()
        rel = m.group(2).strip().replace("\\", "/").lstrip("./")
        if rel in mapping and mapping[rel] != digest:
            raise ValueError("%s lists %s twice with different digests" % (path.name, rel))
        mapping[rel] = digest
    if not mapping:
        raise ValueError("%s contains no checksum records" % path.name)
    return mapping


def write_checksums_file(root: Path, rels, digests, path: Path) -> None:
    """Emit a sha256sum-compatible manifest for the payload."""
    lines = []
    for rel in rels:
        lines.append("%s  %s" % (digests[rel], rel))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums(rels, digests, expected, allow_unlisted: bool):
    """Compare computed digests against the manifest.

    Returns (problems, n_ok). `problems` is a list of human-readable strings;
    an empty list means the payload is exactly what the manifest describes.
    """
    problems = []
    n_ok = 0
    on_disk = set(rels)
    for rel in sorted(expected):
        if rel not in on_disk:
            problems.append("MISSING   %s (listed in manifest, absent on disk)" % rel)
            continue
        if digests[rel] != expected[rel]:
            detail = ""
            if rel == DATASET_MANIFEST_NAME:
                detail = (
                    "\n            cause: build_v02_release.py stage 8 writes "
                    "CHECKSUMS.sha256 BEFORE it writes\n"
                    "                   %s, so on a rebuild in place the manifest "
                    "carries the\n"
                    "                   PREVIOUS run's digest for this file and the "
                    "two can never converge.\n"
                    "                   Fix the builder (add %s to the `skip` set of "
                    "_iter_release_files),\n"
                    "                   or re-issue the manifest from the final tree "
                    "with --write_checksums." % (DATASET_MANIFEST_NAME, DATASET_MANIFEST_NAME)
                )
            problems.append(
                "MISMATCH  %s\n            expected %s\n            computed %s%s"
                % (rel, expected[rel], digests[rel], detail)
            )
            continue
        n_ok += 1
    for rel in rels:
        if rel in expected:
            continue
        if rel in MANIFEST_EXEMPT:
            # Cannot be listed by construction; see MANIFEST_EXEMPT.
            if rel != CHECKSUMS_NAME:
                log(
                    "  note: %s is on disk but not covered by %s (the builder writes "
                    "it after the manifest); it is uploaded unverified" % (rel, CHECKSUMS_NAME)
                )
            continue
        msg = "UNLISTED  %s (on disk, not in manifest)" % rel
        if allow_unlisted:
            log("  warn: " + msg)
        else:
            problems.append(msg)
    return problems, n_ok


# --------------------------------------------------------------------------
# Zenodo file keys
# --------------------------------------------------------------------------


def zenodo_key(rel: str, flatten: bool) -> str:
    """Map a relative payload path to a Zenodo file key.

    Zenodo's bucket API takes a flat key space; whether a "/" is accepted in
    a key has varied across Zenodo/InvenioRDM releases. Default is to
    preserve the path and let the server decide (a 400 is reported with an
    explicit instruction to re-run with --flatten). With --flatten, nested
    paths collapse to "dir__file" and a PATHS.json manifest is added to the
    deposit so the original tree can be rebuilt exactly.
    """
    if not flatten:
        return rel
    return rel.replace("/", "__")


def build_paths_manifest(rels, flatten: bool):
    """{zenodo_key: original_relative_path} -- only meaningful when flattening."""
    return dict((zenodo_key(r, flatten), r) for r in rels)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def build_metadata(args, n_files: int, total_bytes: int) -> dict:
    """Assemble the Zenodo deposition metadata block."""
    creators = [{"name": args.creator}]
    if args.affiliation:
        creators[0]["affiliation"] = args.affiliation
    if args.orcid:
        creators[0]["orcid"] = args.orcid

    # No Hugging Face relation. That copy holds the preview - 18 files
    # against 53, with differing pocket-side content - so isIdenticalTo would
    # be false in the one sense DataCite gives it. Once it is stripped to a
    # pointer it is not a version of this record either.
    related = [
        {
            "relation": "isDerivedFrom",
            "identifier": PDB_URL,
            "resource_type": "dataset",
            "scheme": "url",
        },
        {
            "relation": "isDerivedFrom",
            "identifier": BINDINGDB_URL,
            "resource_type": "dataset",
            "scheme": "url",
        },
        {
            "relation": "isDocumentedBy",
            "identifier": args.code_doi,
            "resource_type": "software",
            "scheme": "doi",
        },
    ]
    if args.paper_doi:
        related.append(
            {
                "relation": "isSupplementTo",
                "identifier": args.paper_doi,
                "resource_type": "publication-datapaper",
                "scheme": "doi",
            }
        )
    if args.supersedes:
        related.append(
            {
                "relation": "obsoletes",
                "identifier": args.supersedes,
                "resource_type": "dataset",
                "scheme": "doi",
            }
        )

    description = DESCRIPTION_TEMPLATE % {
        "n_entries": "{:,}".format(args.n_entries),
        "hf_url": args.hf_url,
        "version": args.version,
        "version_note": args.version_note,
        "code_doi": args.code_doi,
        "n_files": n_files,
        "total_human": human_size(total_bytes),
        "checksums_name": CHECKSUMS_NAME,
        "dataset_manifest_name": DATASET_MANIFEST_NAME,
        # From the builder's MEASURED block, the same source the CHANGELOG and
        # the sidecars render from, so the description cannot drift from the
        # paper the way the hard-coded preview figures did.
        "pocket_uniprot": _pct(_M["same_uniprot_hit_at_1_by_alpha"]["1.00_pocket_only"]),
        "concat_uniprot": _pct(_M["same_uniprot_hit_at_1_by_alpha"]["concatenation_retained"]),
        "ligand_uniprot": _pct(_M["same_uniprot_hit_at_1_by_alpha"]["0.00_ligand_only"]),
        "ligand_ccd": _pct(_M["same_ccd_top_1_by_alpha"]["0.00_ligand_only"]),
        "pocket_ccd": _pct(_M["same_ccd_top_1_by_alpha"]["1.00_pocket_only"]),
    }

    metadata = {
        "upload_type": "dataset",
        "title": args.title,
        "creators": creators,
        "description": description,
        "access_right": "open",
        "license": args.license,
        "keywords": list(DEFAULT_KEYWORDS),
        "version": args.version,
        "language": "eng",
        "publication_date": args.publication_date,
        "related_identifiers": related,
        "notes": args.notes,
    }
    if args.metadata_json:
        patch = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
        if "metadata" in patch and isinstance(patch["metadata"], dict):
            patch = patch["metadata"]
        metadata.update(patch)
    return metadata


def validate_metadata(metadata: dict):
    """Return a list of blocking problems with the metadata block."""
    problems = []

    # The versioning disclosure is not optional. It records that the v0.1
    # content was revised in place under an unchanged minted identifier, which
    # is the reason this dataset moved to per-version DOIs. DEFAULT_NOTES
    # carries it and two self-tests check build_metadata's output - but
    # --metadata_json is merged AFTERWARDS, so a patch could drop it with no
    # test firing. One nearly did. Checked here, on the final block, where
    # every path has to pass.
    _notes = metadata.get("notes") or ""
    for _marker in ("2026-05-01", "revised in place"):
        if _marker not in _notes:
            problems.append(
                "notes has lost the versioning disclosure (missing %r). Do not "
                "override 'notes' in --metadata_json; put release changelog in "
                "the description instead." % _marker
            )
    if metadata.get("upload_type") not in ZENODO_UPLOAD_TYPES:
        problems.append("upload_type %r is not a Zenodo upload type" % metadata.get("upload_type"))
    if metadata.get("upload_type") != "dataset":
        problems.append("upload_type must be 'dataset' for this deposit")
    if not str(metadata.get("title", "")).strip():
        problems.append("title is empty")
    creators = metadata.get("creators") or []
    if not creators:
        problems.append("creators is empty")
    for i, c in enumerate(creators):
        if not str(c.get("name", "")).strip():
            problems.append("creators[%d].name is empty" % i)
        elif "," not in c["name"]:
            problems.append("creators[%d].name %r should be 'Family, Given'" % (i, c["name"]))
        orcid = c.get("orcid")
        if orcid and not re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", str(orcid)):
            problems.append("creators[%d].orcid %r is malformed" % (i, orcid))
    if not str(metadata.get("description", "")).strip():
        problems.append("description is empty")
    if metadata.get("access_right") != "open":
        problems.append("access_right must be 'open' (Scientific Data requires open access)")
    lic = metadata.get("license")
    lic_id = lic.get("id") if isinstance(lic, dict) else lic
    if str(lic_id).lower() != "cc-by-4.0":
        problems.append("license must be cc-by-4.0, got %r" % (lic,))
    pubdate = str(metadata.get("publication_date", ""))
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", pubdate):
        problems.append("publication_date %r is not YYYY-MM-DD" % pubdate)
    if not metadata.get("version"):
        problems.append("version is empty")

    seen = set()
    relations_present = set()
    for i, rid in enumerate(metadata.get("related_identifiers") or []):
        rel = rid.get("relation")
        ident = rid.get("identifier")
        if rel not in ZENODO_RELATIONS:
            problems.append(
                "related_identifiers[%d].relation %r is not accepted by Zenodo" % (i, rel)
            )
        if not str(ident or "").strip():
            problems.append("related_identifiers[%d].identifier is empty" % i)
        scheme = rid.get("scheme")
        if scheme == "doi" and not re.match(r"^10\.\d{4,9}/\S+$", str(ident)):
            problems.append(
                "related_identifiers[%d] claims scheme doi but %r is not a DOI" % (i, ident)
            )
        if scheme == "url" and not str(ident).startswith("http"):
            problems.append(
                "related_identifiers[%d] claims scheme url but %r is not a URL" % (i, ident)
            )
        key = (rel, str(ident))
        if key in seen:
            problems.append("related_identifiers has duplicate %s -> %s" % key)
        seen.add(key)
        relations_present.add(rel)

    # isDerivedFrom is the only relation this deposit must always carry.
    # isIdenticalTo was required here while the Hugging Face copy was
    # described as a mirror of the same content; it holds v0.1, so asserting
    # it would be false. The code relation is isCompiledBy, whose reciprocal
    # on the code record is compiles.
    for required in ("isDerivedFrom",):
        if required not in relations_present:
            problems.append("related_identifiers is missing a %s entry" % required)
    if "isCompiledBy" not in relations_present and "isDocumentedBy" not in relations_present:
        problems.append("related_identifiers names no code record")
    return problems


# --------------------------------------------------------------------------
# Zenodo REST client
# --------------------------------------------------------------------------


class ZenodoError(Exception):
    pass


class _ProgressReader(object):
    """File wrapper that reports upload progress.

    http.client calls read(blocksize) with an 8 kB blocksize; returning a
    larger buffer than requested is legal (the connection sends whatever it
    gets) and avoids millions of 8 kB socket writes on a 1 GB file.
    """

    def __init__(
        self, path: Path, total: int, label: str, chunk: int = 4 << 20, quiet: bool = False
    ) -> None:
        self._fh = open(str(path), "rb")
        self._total = total
        self._sent = 0
        self._label = label
        self._chunk = chunk
        self._quiet = quiet
        self._last = 0.0

    def read(self, n=-1):
        want = self._chunk if (n is None or n < 0 or n < self._chunk) else n
        block = self._fh.read(want)
        self._sent += len(block)
        now = time.time()
        if not self._quiet and block and (now - self._last > 2.0):
            self._last = now
            pct = 100.0 * self._sent / max(1, self._total)
            sys.stdout.write(
                "\r    %s  %5.1f%%  %s / %s"
                % (self._label, pct, human_size(self._sent), human_size(self._total))
            )
            sys.stdout.flush()
        return block

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if not self._quiet:
                sys.stdout.write("\r" + " " * 78 + "\r")
                sys.stdout.flush()


class ZenodoClient(object):
    # Set by callers running under --dry_run. _guard() refuses the mutating
    # verbs, so a code path that forgets its own dry-run check still cannot
    # write. The gate in _run_metadata_only was missing for exactly that
    # reason and reached production.
    read_only = False

    def _guard(self, what: str) -> None:
        if self.read_only:
            raise RuntimeError(
                "refusing %s: client is read-only (--dry_run). This is a bug in "
                "the caller - a dry run should have returned before here." % what
            )

    def __init__(
        self,
        base_url: str,
        token: str,
        max_retries: int = 6,
        backoff: float = 4.0,
        timeout: float = 300.0,
    ) -> None:
        self.base = base_url.rstrip("/")
        self._token = token
        self.max_retries = max(1, max_retries)
        self.backoff = backoff
        self.timeout = timeout

    # -- low level ---------------------------------------------------------

    def _request(
        self, method: str, url: str, data=None, headers=None, timeout=None, parse_json: bool = True
    ):
        req = urllib.request.Request(url, data=data, method=method)
        if self._token:
            req.add_header("Authorization", "Bearer " + self._token)
        req.add_header("Accept", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        resp = urllib.request.urlopen(req, timeout=timeout or self.timeout)
        try:
            body = resp.read()
            status = resp.getcode()
        finally:
            resp.close()
        if not parse_json or not body:
            return status, None
        try:
            return status, json.loads(body.decode("utf-8"))
        except ValueError:
            raise ZenodoError(
                "%s %s returned non-JSON body: %s" % (method, url, REDACT(body[:400]))
            )

    def _retryable(self, describe: str, call):
        """Run `call(attempt)` with bounded exponential backoff + jitter."""
        delays = backoff_delays(self.max_retries, self.backoff)
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return call(attempt)
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", "replace")[:600]
                except Exception:
                    body = ""
                last_error = ZenodoError(
                    "%s -> HTTP %s %s\n%s" % (describe, exc.code, exc.reason, REDACT(body))
                )
                if exc.code not in RETRY_STATUS or attempt >= self.max_retries:
                    raise last_error
                wait = self._retry_after(exc) or self._jitter(delays, attempt)
                log(
                    "  retry %d/%d after HTTP %s (%s) in %.1fs"
                    % (attempt, self.max_retries - 1, exc.code, describe, wait)
                )
            except (http.client.HTTPException, OSError) as exc:
                # URLError, socket timeouts, connection resets, IncompleteRead.
                last_error = ZenodoError(
                    "%s -> %s: %s" % (describe, type(exc).__name__, REDACT(exc))
                )
                if attempt >= self.max_retries:
                    raise last_error
                wait = self._jitter(delays, attempt)
                log(
                    "  retry %d/%d after %s (%s) in %.1fs"
                    % (attempt, self.max_retries - 1, type(exc).__name__, describe, wait)
                )
            time.sleep(wait)
        raise last_error if last_error else ZenodoError(describe + " failed")

    @staticmethod
    def _retry_after(exc):
        raw = None
        try:
            raw = exc.headers.get("Retry-After")
        except Exception:
            raw = None
        if not raw:
            return None
        try:
            return min(300.0, max(1.0, float(raw)))
        except ValueError:
            return None

    @staticmethod
    def _jitter(delays, attempt):
        base = delays[attempt - 1] if attempt - 1 < len(delays) else (delays[-1] if delays else 1.0)
        return base * (0.75 + 0.5 * random.random())

    def get_json(self, url: str):
        return self._retryable("GET " + url, lambda _a: self._request("GET", url)[1])

    def send_json(self, method: str, url: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        return self._retryable(
            "%s %s" % (method, url),
            lambda _a: self._request(method, url, data=body, headers=headers)[1],
        )

    def delete(self, url: str):
        return self._retryable(
            "DELETE " + url, lambda _a: self._request("DELETE", url, parse_json=False)[0]
        )

    # -- depositions -------------------------------------------------------

    def create_deposition(self, metadata: dict):
        url = self.base + "/deposit/depositions"
        return self._post_metadata_tolerant("POST", url, metadata)

    def update_metadata(self, dep_id, metadata: dict):
        url = "%s/deposit/depositions/%s" % (self.base, dep_id)
        return self._post_metadata_tolerant("PUT", url, metadata)

    @staticmethod
    def _flip_license(metadata: dict) -> dict:
        """Swap between the string and {"id": ...} forms of `license`.

        Zenodo's legacy deposit API historically took `license: "cc-by-4.0"`;
        InvenioRDM-backed releases have at times required `{"id": ...}`.
        """
        alt = dict(metadata)
        lic = alt.get("license")
        alt["license"] = lic.get("id") if isinstance(lic, dict) else {"id": str(lic)}
        return alt

    @staticmethod
    def _strip_related_resource_types(metadata: dict) -> dict:
        """Drop the optional `resource_type` from every related identifier.

        `resource_type` on a related identifier is optional and its accepted
        vocabulary differs between the legacy deposit schema and InvenioRDM
        (`publication-datapaper` in particular is not in every version of the
        list). The relation and the identifier -- the parts that carry the
        meaning -- are kept.
        """
        alt = dict(metadata)
        alt["related_identifiers"] = [
            dict((k, v) for k, v in rid.items() if k != "resource_type")
            for rid in (metadata.get("related_identifiers") or [])
        ]
        return alt

    def _post_metadata_tolerant(self, method: str, url: str, metadata: dict):
        """Submit metadata, adapting once per known schema divergence.

        Rather than guess which Zenodo/InvenioRDM schema is live, send the
        documented form and narrow it only when the server names the field it
        rejected. Each fallback is logged, so nothing is silently dropped.
        """
        fallbacks = (
            ("license", self._flip_license, "alternate license form"),
            (
                "resource_type",
                self._strip_related_resource_types,
                "related_identifiers stripped of resource_type",
            ),
            (
                "related_identifier",
                self._strip_related_resource_types,
                "related_identifiers stripped of resource_type",
            ),
        )
        payload = metadata
        tried = set()
        while True:
            try:
                return self.send_json(method, url, {"metadata": payload})
            except ZenodoError as exc:
                text = str(exc)
                lowered = text.lower()
                if "HTTP 400" not in text:
                    raise
                for field, transform, label in fallbacks:
                    if field in lowered and label not in tried:
                        tried.add(label)
                        log("  metadata rejected on %r; retrying with %s" % (field, label))
                        payload = transform(payload)
                        break
                else:
                    raise

    def get_deposition(self, dep_id):
        return self.get_json("%s/deposit/depositions/%s" % (self.base, dep_id))

    def new_version(self, dep_id):
        url = "%s/deposit/depositions/%s/actions/newversion" % (self.base, dep_id)
        return self._retryable("POST " + url, lambda _a: self._request("POST", url)[1])

    def edit(self, dep_id):
        self._guard("actions/edit on %s" % dep_id)
        url = "%s/deposit/depositions/%s/actions/edit" % (self.base, dep_id)
        return self._retryable("POST " + url, lambda _a: self._request("POST", url)[1])

    def publish(self, dep_id):
        self._guard("actions/publish on %s" % dep_id)
        url = "%s/deposit/depositions/%s/actions/publish" % (self.base, dep_id)
        return self._retryable("POST " + url, lambda _a: self._request("POST", url)[1])

    def resolve_latest_deposition(self, reference: str):
        """Concept DOI / version DOI / record id -> the deposition id to branch from."""
        recid = None
        m = re.search(r"zenodo\.(\d+)", str(reference))
        if m:
            recid = m.group(1)
        elif str(reference).isdigit():
            recid = str(reference)
        if recid is None:
            raise ZenodoError(
                "cannot parse %r as a Zenodo DOI or record id "
                "(expected e.g. 10.5281/zenodo.1234567)" % reference
            )
        record = self.get_json("%s/records/%s" % (self.base, recid))
        links = record.get("links") or {}
        latest = links.get("latest") or links.get("latest_html") or ""
        m2 = re.search(r"/(\d+)$", str(latest).rstrip("/"))
        latest_id = m2.group(1) if m2 else str(record.get("id") or recid)
        return latest_id, record

    # -- files -------------------------------------------------------------

    def list_bucket(self, bucket_url: str):
        """{key: {"size": int, "md5": str}} for objects already in the bucket.

        Returns None (not {}) if the bucket cannot be listed, so callers can
        tell "empty bucket" from "no answer" and fall back to the deposition
        files endpoint. The failure is logged: an unlisted bucket silently
        treated as empty would re-push the whole payload.
        """
        try:
            data = self.get_json(bucket_url)
        except ZenodoError as exc:
            log(
                "  warn: could not list the bucket (%s); falling back to the "
                "deposition files endpoint" % REDACT(str(exc).splitlines()[0])
            )
            return None
        out = {}
        for item in (data or {}).get("contents") or []:
            checksum = str(item.get("checksum") or "")
            md5 = checksum.split(":", 1)[1] if ":" in checksum else checksum
            out[item.get("key")] = {"size": int(item.get("size") or 0), "md5": md5.lower()}
        return out

    def list_deposition_files(self, dep_id):
        """Fallback listing via the deposition endpoint (also gives file ids)."""
        data = self.get_json("%s/deposit/depositions/%s/files" % (self.base, dep_id))
        out = {}
        for item in data or []:
            checksum = str(item.get("checksum") or "")
            md5 = checksum.split(":", 1)[1] if ":" in checksum else checksum
            out[item.get("filename")] = {
                "size": int(item.get("filesize") or 0),
                "md5": md5.lower(),
                "id": item.get("id"),
            }
        return out

    def delete_deposition_file(self, dep_id, file_id) -> None:
        self.delete("%s/deposit/depositions/%s/files/%s" % (self.base, dep_id, file_id))

    def put_file(
        self, bucket_url: str, key: str, path: Path, size: int, md5: str, quiet: bool = False
    ):
        """Upload one object, resuming across retries.

        Before every attempt (including the first retry after a 502) the
        bucket is re-listed: Zenodo frequently returns 502 from its proxy
        *after* the object has been stored, and re-pushing several hundred
        megabytes for an upload that already succeeded is the single most
        expensive mistake this script can make.
        """
        target = bucket_url.rstrip("/") + "/" + urllib.parse.quote(key, safe="")

        def attempt(n):
            if n > 1:
                remote = self.list_bucket(bucket_url) or {}
                got = remote.get(key)
                if got and got.get("md5") == md5 and got.get("size") == size:
                    log("    %s already landed server-side; skipping re-upload" % key)
                    return {"key": key, "size": size, "checksum": "md5:" + md5, "resumed": True}
            body = _ProgressReader(path, size, key, quiet=quiet)
            try:
                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(size),
                }
                return self._request("PUT", target, data=body, headers=headers)[1]
            finally:
                body.close()

        try:
            return self._retryable("PUT " + key, attempt)
        except ZenodoError as exc:
            if "HTTP 400" in str(exc) and "/" in key:
                raise ZenodoError(
                    str(exc) + "\n\nThis key contains '/'. Zenodo rejected it. Re-run with "
                    "--flatten to collapse nested paths to 'dir__file' and ship a "
                    "PATHS.json mapping alongside the payload."
                )
            raise


# --------------------------------------------------------------------------
# Upload planning
# --------------------------------------------------------------------------


def plan_uploads(local, remote, force: bool = False):
    """Decide what to upload, skip, and delete.

    local  : {key: {"size": int, "md5": str}}
    remote : {key: {"size": int, "md5": str, "id": optional}}
    Returns (to_upload, to_skip, to_delete), each a sorted list of keys.
    """
    remote = remote or {}
    to_upload, to_skip = [], []
    for key in sorted(local):
        got = remote.get(key)
        if (
            (not force)
            and got
            and got.get("md5") == local[key]["md5"]
            and int(got.get("size", -1)) == int(local[key]["size"])
        ):
            to_skip.append(key)
        else:
            to_upload.append(key)
    to_delete = sorted([k for k in remote if k not in local])
    return to_upload, to_skip, to_delete


# --------------------------------------------------------------------------
# Hugging Face snapshot
# --------------------------------------------------------------------------


def snapshot_from_hf(repo_id: str, revision, dest: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub  (needed for --hf_repo)")
    dest.mkdir(parents=True, exist_ok=True)
    log("Snapshotting %s (revision=%s) into %s" % (repo_id, revision or "main", dest))
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        token=os.environ.get("HF_TOKEN") or None,
    )
    return Path(path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def api_base_for(args) -> str:
    return args.api_base or (ZENODO_SANDBOX_API if args.sandbox else ZENODO_API)


def require_token(base: str) -> str:
    """Read ZENODO_TOKEN or exit with the exact instructions to create one."""
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        sys.exit(
            "ZENODO_TOKEN is not set. Create a personal access token with the "
            "deposit:write and deposit:actions scopes at %s/account/settings/"
            "applications/tokens/new/ and export it:\n"
            "    export ZENODO_TOKEN=...        # bash\n"
            "    $env:ZENODO_TOKEN = '...'      # PowerShell" % base.replace("/api", "")
        )
    REDACT.add(token)
    return token


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Deposit the v0.2 dataset to Zenodo as the primary record.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_argument_group("payload")
    src.add_argument(
        "--source_dir",
        type=Path,
        help="Local directory holding the payload (created by --hf_repo).",
    )
    src.add_argument(
        "--hf_repo", type=str, help="Snapshot this HF dataset repo into --source_dir first."
    )
    src.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="HF revision/branch/tag to snapshot (default: main).",
    )
    src.add_argument(
        "--checksums",
        type=Path,
        default=None,
        help="Path to CHECKSUMS.sha256 (default: <source_dir>/%s)" % CHECKSUMS_NAME,
    )
    src.add_argument(
        "--write_checksums",
        action="store_true",
        help="Generate CHECKSUMS.sha256 from the payload instead of verifying "
        "against it. Only valid for a payload you just built locally.",
    )
    src.add_argument(
        "--allow_unlisted",
        action="store_true",
        help="Permit files on disk that the manifest does not list.",
    )
    src.add_argument(
        "--exclude", action="append", default=[], help="Extra exclude glob (repeatable)."
    )
    src.add_argument(
        "--flatten",
        action="store_true",
        help="Collapse nested paths to 'dir__file' keys and ship PATHS.json.",
    )

    meta = ap.add_argument_group("metadata")
    meta.add_argument("--title", default=DEFAULT_TITLE)
    meta.add_argument("--creator", default="Kari, Hemanth")
    meta.add_argument(
        "--affiliation",
        default="Independent researcher, Hammond, Indiana, USA",
    )
    meta.add_argument(
        "--orcid",
        default=os.environ.get("ORCID", "0000-0001-5755-2242"),
    )
    meta.add_argument("--version", default="v0.2")
    meta.add_argument("--version_note", default=DEFAULT_VERSION_NOTE)
    meta.add_argument("--notes", default=DEFAULT_NOTES)
    meta.add_argument("--n_entries", type=int, default=72199)
    meta.add_argument("--license", default="cc-by-4.0")
    meta.add_argument("--publication_date", default=time.strftime("%Y-%m-%d"))
    meta.add_argument("--hf_url", default=HF_DATASET_URL)
    meta.add_argument("--hf_doi", default=HF_DATASET_DOI)
    meta.add_argument("--code_doi", default=PIPELINE_CODE_DOI)
    meta.add_argument(
        "--paper_doi",
        default="",
        help="DOI of the Data Descriptor once assigned; adds isSupplementTo.",
    )
    meta.add_argument(
        "--supersedes", default="", help="Version DOI of v0.1; adds an 'obsoletes' relation."
    )
    meta.add_argument(
        "--metadata_json",
        type=Path,
        default=None,
        help="JSON file whose keys override the generated metadata.",
    )

    dep = ap.add_argument_group("deposition")
    dep.add_argument(
        "--new_version",
        default="",
        help="Concept DOI, version DOI, or record id to cut a new version of.",
    )
    dep.add_argument("--deposition", default="", help="Attach to an existing draft deposition id.")
    dep.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved draft state and start a new deposition.",
    )
    dep.add_argument(
        "--publish",
        action="store_true",
        help="Publish the deposition (mints the DOI). Default: leave as draft.",
    )
    dep.add_argument(
        "--metadata_only",
        action="store_true",
        help="Edit metadata on an existing deposition (--deposition <id>) "
        "without touching files, then re-publish if it was published. "
        "This is the post-acceptance path for adding --paper_doi; a "
        "published record's METADATA is editable, only its files are "
        "immutable, so this does not need --new_version.",
    )
    dep.add_argument(
        "--mark_superseded",
        action="store_true",
        help="After publishing, edit the previous version to add isObsoletedBy.",
    )
    dep.add_argument(
        "--keep_remote_extras",
        action="store_true",
        help="Do not delete draft files that are absent from the payload.",
    )
    dep.add_argument(
        "--force_reupload",
        action="store_true",
        help="Re-upload every file even when the remote checksum matches.",
    )

    net = ap.add_argument_group("transport")
    net.add_argument(
        "--sandbox", action="store_true", help="Target sandbox.zenodo.org (needs a sandbox token)."
    )
    net.add_argument("--api_base", default="", help="Override the API base URL entirely.")
    net.add_argument("--max_retries", type=int, default=6)
    net.add_argument(
        "--backoff", type=float, default=4.0, help="Base seconds for exponential backoff."
    )
    net.add_argument(
        "--timeout", type=float, default=300.0, help="Per-request socket timeout in seconds."
    )

    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Verify checksums, validate metadata, print the plan; no network.",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress per-file progress lines.")
    ap.add_argument(
        "--self_test", action="store_true", help="Run the offline logic checks and exit."
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.self_test:
        _self_test()
        return

    if args.metadata_only:
        _run_metadata_only(args)
        return

    if args.source_dir is None:
        sys.exit("--source_dir is required (or pass --self_test)")

    source_dir = args.source_dir.resolve()
    if args.hf_repo:
        snapshot_from_hf(args.hf_repo, args.hf_revision, source_dir)
    if not source_dir.is_dir():
        sys.exit("Source dir not found: %s (pass --hf_repo to snapshot it)" % source_dir)

    excludes = list(DEFAULT_EXCLUDES) + list(args.exclude)

    # ---- 1. discover + hash ------------------------------------------------
    log("Payload: %s" % source_dir)
    rels = discover_files(source_dir, excludes)
    if not rels:
        sys.exit("No files found under %s after exclusions" % source_dir)

    sha_of, md5_of, size_of = {}, {}, {}
    total_bytes = 0
    t0 = time.time()
    for i, rel in enumerate(rels, 1):
        sha, md5, size = hash_file(source_dir / rel)
        sha_of[rel], md5_of[rel], size_of[rel] = sha, md5, size
        total_bytes += size
        if not args.quiet:
            sys.stdout.write("\r  hashed %d/%d  %s" % (i, len(rels), human_size(total_bytes)))
            sys.stdout.flush()
    if not args.quiet:
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()
    log("Hashed %d files, %s, in %.1fs" % (len(rels), human_size(total_bytes), time.time() - t0))

    # ---- 2. checksum gate --------------------------------------------------
    checksums_path = args.checksums or (source_dir / CHECKSUMS_NAME)
    if args.write_checksums:
        manifest_abs = checksums_path.resolve()
        payload_rels = [r for r in rels if (source_dir / r).resolve() != manifest_abs]
        write_checksums_file(source_dir, payload_rels, sha_of, checksums_path)
        log("Wrote %s (%d records)" % (checksums_path, len(payload_rels)))
        log(
            "NOTE: this manifest was generated from the payload just now, so verifying "
            "against it proves only internal consistency, not that the payload matches "
            "the build. Ship the manifest produced by the build for a real check."
        )
        # The manifest is itself part of the payload. Re-hash anything that
        # appeared or changed as a result of writing it, so the MD5 we later
        # compare against Zenodo's is the post-write value, not a stale one.
        rels = discover_files(source_dir, excludes)
        for rel in rels:
            path = source_dir / rel
            if rel in sha_of and path.resolve() != manifest_abs:
                continue
            sha, md5, size = hash_file(path)
            total_bytes += size - size_of.get(rel, 0)
            sha_of[rel], md5_of[rel], size_of[rel] = sha, md5, size
    if not checksums_path.exists():
        sys.exit(
            "Checksum manifest not found: %s\n"
            "Refusing to upload an unverified payload. Ship the build's "
            "CHECKSUMS.sha256, or pass --write_checksums if you built this "
            "payload locally right now." % checksums_path
        )
    expected = parse_checksums_file(checksums_path)
    problems, n_ok = verify_checksums(rels, sha_of, expected, args.allow_unlisted)
    if problems:
        log("SHA-256 verification FAILED (%d problem(s)):" % len(problems))
        for p in problems:
            log("  " + p)
        sys.exit("Aborting: payload does not match %s" % checksums_path.name)
    log("SHA-256 verified: %d/%d files match %s" % (n_ok, len(expected), checksums_path.name))

    # ---- 3. keys ------------------------------------------------------------
    keys = {}
    for rel in rels:
        key = zenodo_key(rel, args.flatten)
        if key in keys:
            sys.exit("Key collision: %r and %r both map to %r" % (keys[key], rel, key))
        keys[key] = rel
    local = dict((k, {"size": size_of[r], "md5": md5_of[r], "rel": r}) for k, r in keys.items())

    paths_manifest_path = None
    if args.flatten:
        paths_manifest_path = source_dir / PATHS_MANIFEST_NAME
        body = json.dumps(build_paths_manifest(rels, True), indent=2, sort_keys=True).encode(
            "utf-8"
        )
        if args.dry_run:
            log(
                "Would write %s (flattened-key -> original-path map, %d bytes); "
                "--dry_run does not touch the payload directory" % (paths_manifest_path, len(body))
            )
        else:
            paths_manifest_path.write_bytes(body)
            log("Wrote %s (flattened-key -> original-path map)" % paths_manifest_path)
        # Whether or not it was written, it is part of the upload, so it counts
        # towards the file count and total size reported in the record.
        local[PATHS_MANIFEST_NAME] = {
            "size": len(body),
            "md5": hashlib.md5(body).hexdigest(),
            "rel": PATHS_MANIFEST_NAME,
        }
        # If a stale PATHS.json from an earlier run was already discovered and
        # hashed, replace its contribution rather than adding a second one.
        total_bytes += len(body) - size_of.get(PATHS_MANIFEST_NAME, 0)

    nested = [k for k in local if "/" in k]
    base = api_base_for(args)

    # ---- 4. metadata -------------------------------------------------------
    # Built here, after PATHS.json is accounted for, so the file count and
    # total size baked into the (permanent) record description match what is
    # actually uploaded.
    metadata = build_metadata(args, len(local), total_bytes)
    md_problems = validate_metadata(metadata)
    if md_problems:
        log("Metadata validation FAILED:")
        for p in md_problems:
            log("  " + p)
        sys.exit("Aborting: fix the metadata before depositing")
    log(
        "Metadata validated: %d related identifiers, license=%s, version=%s"
        % (len(metadata["related_identifiers"]), metadata["license"], metadata["version"])
    )
    if not args.paper_doi:
        log(
            "WARN: no --paper_doi, so no isSupplementTo relation was written. Once the "
            "Data Descriptor DOI is assigned, re-run with "
            "--metadata_only --deposition <id> --paper_doi <doi>."
        )

    log("")
    log("=== Deposit plan ===")
    log("  target        : %s" % base)
    log("  files         : %d" % len(local))
    log("  total size    : %s" % human_size(total_bytes))
    log(
        "  record limit  : %s (%.1f%% used)"
        % (human_size(ZENODO_RECORD_LIMIT_BYTES), 100.0 * total_bytes / ZENODO_RECORD_LIMIT_BYTES)
    )
    if args.new_version:
        log("  mode          : new version of %s" % args.new_version)
    elif args.deposition:
        log("  mode          : resume deposition %s" % args.deposition)
    else:
        log("  mode          : new deposition")
    log("  publish       : %s" % ("yes" if args.publish else "no (draft)"))
    for key in sorted(local, key=lambda k: -local[k]["size"])[:15]:
        log(
            "    %-52s %10s  md5=%s" % (key, human_size(local[key]["size"]), local[key]["md5"][:12])
        )
    if len(local) > 15:
        log("    ... and %d smaller files" % (len(local) - 15))
    if total_bytes > ZENODO_RECORD_LIMIT_BYTES:
        sys.exit("Payload exceeds Zenodo's 50 GB per-record limit; request a quota increase.")
    oversize = [k for k in local if local[k]["size"] > ZENODO_FILE_LIMIT_BYTES]
    if oversize:
        sys.exit("Files over the per-file limit: %s" % ", ".join(oversize))
    if nested and not args.flatten:
        log(
            "  NOTE: %d keys contain '/' (e.g. %s). Preserving the tree. If Zenodo "
            "returns HTTP 400 on those keys, re-run with --flatten." % (len(nested), nested[0])
        )

    if args.dry_run:
        log("")
        log("DRY RUN: checksums verified, metadata validated, nothing uploaded.")
        return

    # ---- 5. token ----------------------------------------------------------
    token = require_token(base)
    client = ZenodoClient(base, token, args.max_retries, args.backoff, args.timeout)

    # ---- 6. resolve or create the deposition -------------------------------
    state_path = source_dir / STATE_NAME
    state = {}
    if state_path.exists() and not args.fresh:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except ValueError:
            state = {}
        if state.get("base") != base:
            state = {}

    previous_record = None
    dep_id = args.deposition or state.get("deposition_id") or ""
    if args.new_version and dep_id and not args.deposition:
        log(
            "WARN: --new_version was requested but %s already records draft %s for %s, "
            "so no new version is cut and that draft is reused. Pass --fresh to ignore "
            "the saved state." % (STATE_NAME, dep_id, base)
        )

    if args.new_version and not dep_id:
        latest_id, previous_record = client.resolve_latest_deposition(args.new_version)
        log("Concept %s -> latest published record %s" % (args.new_version, latest_id))
        draft = client.new_version(latest_id)
        links = draft.get("links") or {}
        latest_draft = links.get("latest_draft") or ""
        m = re.search(r"/(\d+)$", str(latest_draft).rstrip("/"))
        dep_id = m.group(1) if m else str(draft.get("id"))
        log("New version draft: deposition %s" % dep_id)

    if dep_id:
        deposition = client.get_deposition(dep_id)
        log(
            "Using deposition %s (state=%s, submitted=%s)"
            % (dep_id, deposition.get("state"), deposition.get("submitted"))
        )
        if deposition.get("submitted") and deposition.get("state") != "unsubmitted":
            sys.exit(
                "Deposition %s is already published; a published record's FILES are "
                "immutable.\n"
                "  - to change the files, cut a new version: --new_version %s\n"
                "  - to change only the metadata (e.g. add --paper_doi after "
                "acceptance), use:\n"
                "        --metadata_only --deposition %s --paper_doi <doi>"
                % (dep_id, dep_id, dep_id)
            )
    else:
        deposition = client.create_deposition(metadata)
        dep_id = str(deposition.get("id"))
        log("Created deposition %s" % dep_id)

    bucket_url = ((deposition.get("links") or {}).get("bucket") or "").rstrip("/")
    if not bucket_url:
        deposition = client.get_deposition(dep_id)
        bucket_url = ((deposition.get("links") or {}).get("bucket") or "").rstrip("/")
    if not bucket_url:
        sys.exit("Deposition %s exposes no bucket link; cannot use the bucket API." % dep_id)

    deposition = client.update_metadata(dep_id, metadata)
    log("Metadata written to deposition %s" % dep_id)

    state.update({"base": base, "deposition_id": dep_id, "bucket": bucket_url})
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ---- 7. upload ---------------------------------------------------------
    remote = client.list_bucket(bucket_url)
    dep_files = client.list_deposition_files(dep_id)
    if remote is None:
        remote = dep_files

    # Adopt the convention the remote already uses. A new version that
    # addresses its objects differently from the version it descends from
    # makes every key stale, so nothing can be skipped - and here the keys are
    # load-bearing: the Data Descriptor documents the "__" flattening,
    # PATHS.json exists to invert it, and the published usage example fetches
    # "indices__pocket.faiss" by name. Switching silently breaks all three,
    # and Zenodo rejects "/" in a key anyway (HTTP 404 on PUT).
    remote_flat = sum(1 for k in remote if "__" in k)
    local_nested = sum(1 for k in local if "/" in k)
    if remote_flat and local_nested and not args.flatten:
        log(
            "  NOTE: the remote stores %d flattened key(s) but this payload would "
            "produce %d nested one(s). Adopting the remote's convention; pass "
            "--flatten explicitly to silence this." % (remote_flat, local_nested)
        )
        local = {zenodo_key(v["rel"], True): v for v in local.values()}

    to_upload, to_skip, to_delete = plan_uploads(local, remote, args.force_reupload)

    log("")
    log(
        "Remote holds %d object(s): %d already correct, %d to upload, %d stale"
        % (len(remote), len(to_skip), len(to_upload), len(to_delete))
    )
    for key in to_skip:
        log("  skip   %-52s (checksum match)" % key)

    # Deletions happen AFTER the uploads below. Deleting first means a
    # failure part-way leaves the draft missing files; uploading first leaves
    # it holding extras, which is harmless and self-corrects next run. The
    # order mattered: an aborted run once left a draft with 14 of 53 objects.
    if to_delete and args.keep_remote_extras:
        log(
            "  keeping %d remote file(s) not in the payload (--keep_remote_extras)" % len(to_delete)
        )

    uploaded_bytes = 0
    started = time.time()
    for n, key in enumerate(to_upload, 1):
        info = local[key]
        path = source_dir / info["rel"]
        log("  [%d/%d] PUT %s (%s)" % (n, len(to_upload), key, human_size(info["size"])))
        result = client.put_file(bucket_url, key, path, info["size"], info["md5"], quiet=args.quiet)
        got = str((result or {}).get("checksum") or "")
        got_md5 = got.split(":", 1)[1] if ":" in got else got
        if got_md5 and got_md5.lower() != info["md5"]:
            sys.exit(
                "Checksum mismatch after upload of %s: local md5 %s, Zenodo %s"
                % (key, info["md5"], got_md5)
            )
        uploaded_bytes += info["size"]
        state.setdefault("uploaded", {})[key] = info["md5"]
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if to_delete and not args.keep_remote_extras:
        for key in to_delete:
            info = dep_files.get(key) or {}
            if info.get("id"):
                client.delete_deposition_file(dep_id, info["id"])
            else:
                client.delete(bucket_url + "/" + urllib.parse.quote(key, safe=""))
            log("  delete %s (stale; uploads already succeeded)" % key)

    if to_upload:
        elapsed = max(0.001, time.time() - started)
        log(
            "Uploaded %s in %.1f min (%.1f MB/s)"
            % (human_size(uploaded_bytes), elapsed / 60.0, uploaded_bytes / 1e6 / elapsed)
        )

    # ---- 8. post-upload verification ---------------------------------------
    final = client.list_bucket(bucket_url)
    if final is None:
        final = client.list_deposition_files(dep_id)
    missing = [k for k in local if k not in final]
    bad = [k for k in local if k in final and final[k]["md5"] != local[k]["md5"]]
    if missing or bad:
        for k in missing:
            log("  MISSING on Zenodo: %s" % k)
        for k in bad:
            log("  CHECKSUM MISMATCH on Zenodo: %s" % k)
        sys.exit("Aborting before publish: remote payload does not match local payload.")
    log("Verified %d/%d objects on Zenodo by MD5" % (len(local), len(local)))

    # ---- 9. publish --------------------------------------------------------
    deposition = client.get_deposition(dep_id)
    if not args.publish:
        log("")
        log("Draft ready but NOT published (default). Review it, then either:")
        log(
            "  - publish in the browser: %s"
            % ((deposition.get("links") or {}).get("html") or (base + "/deposit/" + dep_id))
        )
        log("  - or re-run this command with --publish")
        _print_dois(deposition)
        return

    published = client.publish(dep_id)
    log("Published.")
    _print_dois(published)

    # ---- 10. mark the previous version superseded --------------------------
    if args.mark_superseded:
        prev_id = (previous_record or {}).get("id")
        # --supersedes is a DOI, not a deposition id: resolve it. Without this
        # the flag was accepted and then silently ignored.
        for reference in (args.supersedes, args.new_version):
            if prev_id or not reference:
                continue
            prev_id, previous_record = client.resolve_latest_deposition(reference)
        if not prev_id:
            log(
                "WARN: --mark_superseded needs the previous record; pass --new_version "
                "or --supersedes with the v0.1 DOI. Skipped."
            )
        elif str(prev_id) == str(dep_id):
            log(
                "WARN: --supersedes/--new_version resolves to the record just "
                "published (%s); refusing to mark it obsoleted by itself. Skipped." % dep_id
            )
        else:
            _mark_superseded(client, prev_id, published.get("doi") or "", args.version)

    if paths_manifest_path is not None:
        log(
            "Note: %s was generated for the flattened layout and left in the payload dir."
            % paths_manifest_path.name
        )


def _run_metadata_only(args) -> None:
    """Rewrite the metadata of an existing deposition; never touches files.

    This is the path the revision documentation prescribes for the one edit
    that cannot be made before acceptance: adding the isSupplementTo relation
    to the article DOI. On Zenodo a published record's FILES are immutable but
    its METADATA is not, so this needs actions/edit + actions/publish, not a
    new version -- cutting a new version here would mint a second dataset DOI
    for identical bytes, which is exactly what the Data Availability text
    tells readers not to do.
    """
    if not args.deposition:
        sys.exit("--metadata_only requires --deposition <id>")
    base = api_base_for(args)
    token = require_token(base)
    client = ZenodoClient(base, token, args.max_retries, args.backoff, args.timeout)
    if args.dry_run:
        client.read_only = True

    dep_id = str(args.deposition)
    deposition = client.get_deposition(dep_id)
    was_published = bool(deposition.get("submitted")) and deposition.get("state") != "unsubmitted"
    log(
        "Deposition %s (state=%s, submitted=%s)"
        % (dep_id, deposition.get("state"), deposition.get("submitted"))
    )

    files = client.list_deposition_files(dep_id)
    n_files = len(files)
    total_bytes = sum(int(v.get("size") or 0) for v in files.values())
    if not n_files:
        log(
            "WARN: the deposition reports no files; the description's file count "
            "and total size will be written as 0."
        )

    metadata = build_metadata(args, n_files, total_bytes)
    problems = validate_metadata(metadata)
    if problems:
        log("Metadata validation FAILED:")
        for p in problems:
            log("  " + p)
        sys.exit("Aborting: fix the metadata before editing the record")
    if not args.paper_doi:
        log(
            "WARN: --metadata_only without --paper_doi does not add the "
            "isSupplementTo relation, which is usually the reason to run it."
        )

    if args.dry_run:
        # Placed after build+validate so a dry run still shows exactly what
        # would be sent, and before client.edit(), which is the first call
        # that mutates a PUBLISHED record. Without this, --dry_run reaches
        # production: main() branches here long before the upload path's own
        # dry-run gate.
        log("")
        log("DRY RUN: nothing was sent. The metadata that WOULD be written:")
        log(json.dumps(metadata, indent=2, sort_keys=True))
        log("")
        log("Re-run without --dry_run to apply. Note that for an already")
        log("published record this re-publishes immediately; there is no draft.")
        return

    if was_published:
        log("Opening the published record for metadata editing (actions/edit)")
        client.edit(dep_id)
    client.update_metadata(dep_id, metadata)
    log(
        "Metadata rewritten on deposition %s (%d files, %s -- unchanged)"
        % (dep_id, n_files, human_size(total_bytes))
    )

    if was_published or args.publish:
        record = client.publish(dep_id)
        log("Re-published; the DOI is unchanged.")
        _print_dois(record)
    else:
        log("Draft metadata updated; not published (pass --publish).")
        _print_dois(client.get_deposition(dep_id))


def _print_dois(record: dict) -> None:
    doi = record.get("doi") or (record.get("metadata") or {}).get("doi") or "(not yet minted)"
    conceptdoi = record.get("conceptdoi") or (record.get("metadata") or {}).get("conceptdoi") or ""
    links = record.get("links") or {}
    log("")
    log("  version DOI : %s" % doi)
    if conceptdoi:
        log("  concept DOI : %s   (always resolves to the latest version)" % conceptdoi)
    if links.get("html"):
        log("  record page : %s" % links.get("html"))
    log("")
    log("  Cite the VERSION DOI in Data Availability; cite the CONCEPT DOI only where")
    log("  'latest version' is the intent. They are not interchangeable.")


def _mark_superseded(client: ZenodoClient, prev_id, new_doi: str, new_version: str) -> None:
    """Edit the previous published version to record that it is superseded."""
    if not new_doi:
        log("WARN: no new DOI available; cannot mark the previous version superseded.")
        return
    log("Marking record %s superseded by %s" % (prev_id, new_doi))
    draft = client.edit(prev_id)
    metadata = dict((draft.get("metadata") or {}))
    related = list(metadata.get("related_identifiers") or [])
    already = any(
        r.get("relation") == "isObsoletedBy" and str(r.get("identifier")) == new_doi
        for r in related
    )
    if not already:
        related.append(
            {
                "relation": "isObsoletedBy",
                "identifier": new_doi,
                "resource_type": "dataset",
                "scheme": "doi",
            }
        )
        metadata["related_identifiers"] = related
    banner = (
        "<p><strong>SUPERSEDED.</strong> This version has been replaced by %s "
        "(%s). It remains available for citation continuity only; new work should "
        "use the current version.</p>" % (new_version, new_doi)
    )
    description = str(metadata.get("description") or "")
    if "SUPERSEDED" not in description:
        metadata["description"] = banner + description
    client.update_metadata(prev_id, metadata)
    client.publish(prev_id)
    log("Previous record %s updated and re-published as superseded." % prev_id)


# --------------------------------------------------------------------------
# Offline self-test
# --------------------------------------------------------------------------


def _self_test() -> None:
    """Exercise every non-network code path. No token, no sockets, no data."""
    checks = []

    def ok(name, cond):
        assert cond, "self-test FAILED: " + name
        checks.append(name)

    # -- human_size / backoff ------------------------------------------------
    ok(
        "human_size",
        human_size(0) == "0 B"
        and human_size(1500) == "1.5 kB"
        and human_size(4_800_000_000).endswith("GB"),
    )
    delays = backoff_delays(6, 4.0)
    ok("backoff length", len(delays) == 5)
    ok("backoff monotonic", all(delays[i] <= delays[i + 1] for i in range(len(delays) - 1)))
    ok("backoff capped", max(backoff_delays(20, 4.0)) <= 120.0)
    ok("backoff empty", backoff_delays(1, 4.0) == [])

    # -- redaction -----------------------------------------------------------
    red = _Redactor()
    red.add("supersecrettoken123")
    red.add("x")  # too short; must be ignored
    ok("redaction", "supersecrettoken123" not in red("Bearer supersecrettoken123 failed"))
    ok("redaction short ignored", red("x marks") == "x marks")

    # -- filesystem-backed checks -------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="zenodo_selftest_"))
    (tmp / "pockets").mkdir()
    (tmp / "__pycache__").mkdir()
    (tmp / "metadata.jsonl").write_bytes(b'{"pdb_id": "1abc"}\n')
    (tmp / "pockets" / "pocket_embeddings.npy").write_bytes(b"\x00\x01\x02\x03" * 64)
    (tmp / "__pycache__" / "junk.pyc").write_bytes(b"junk")
    (tmp / ".gitattributes").write_bytes(b"*.npy filter=lfs")

    rels = discover_files(tmp, DEFAULT_EXCLUDES)
    ok("discover excludes caches", rels == ["metadata.jsonl", "pockets/pocket_embeddings.npy"])
    ok("is_excluded nested", is_excluded("pockets/__pycache__/a.pyc", DEFAULT_EXCLUDES))
    ok("is_excluded keeps payload", not is_excluded("pockets/pocket_coords.h5", DEFAULT_EXCLUDES))

    sha_of, md5_of, size_of = {}, {}, {}
    for rel in rels:
        s, m, z = hash_file(tmp / rel)
        sha_of[rel], md5_of[rel], size_of[rel] = s, m, z
    ok(
        "hash_file sha256",
        sha_of["metadata.jsonl"] == hashlib.sha256(b'{"pdb_id": "1abc"}\n').hexdigest(),
    )
    ok(
        "hash_file md5",
        md5_of["metadata.jsonl"] == hashlib.md5(b'{"pdb_id": "1abc"}\n').hexdigest(),
    )
    ok("hash_file size", size_of["pockets/pocket_embeddings.npy"] == 256)

    manifest = tmp / CHECKSUMS_NAME
    write_checksums_file(tmp, rels, sha_of, manifest)
    parsed = parse_checksums_file(manifest)
    ok("checksum roundtrip", parsed == dict((r, sha_of[r]) for r in rels))

    problems, n_ok = verify_checksums(rels + [CHECKSUMS_NAME], sha_of, parsed, False)
    ok("verify clean", problems == [] and n_ok == 2)

    # tolerate binary markers, CRLF, comments, backslashes, BOM
    weird = tmp / "weird.sha256"
    weird.write_text(
        "﻿# generated by the build\r\n"
        + sha_of["metadata.jsonl"]
        + " *metadata.jsonl\r\n"
        + sha_of["pockets/pocket_embeddings.npy"]
        + "  pockets\\pocket_embeddings.npy\r\n",
        encoding="utf-8",
    )
    ok("checksum parser tolerant", parse_checksums_file(weird) == parsed)
    try:
        (tmp / "bad.sha256").write_text("not a checksum line\n", encoding="utf-8")
        parse_checksums_file(tmp / "bad.sha256")
        ok("checksum parser rejects junk", False)
    except ValueError:
        ok("checksum parser rejects junk", True)

    # corruption is caught
    corrupt = dict(sha_of)
    corrupt["metadata.jsonl"] = "0" * 64
    problems, _ = verify_checksums(rels, corrupt, parsed, False)
    ok("verify catches mismatch", len(problems) == 1 and problems[0].startswith("MISMATCH"))
    # a missing file is caught
    problems, _ = verify_checksums(["metadata.jsonl"], sha_of, parsed, False)
    ok("verify catches missing", any(p.startswith("MISSING") for p in problems))
    # an extra file is caught unless allowed
    extra_sha = dict(sha_of)
    extra_sha["surprise.npy"] = "1" * 64
    problems, _ = verify_checksums(rels + ["surprise.npy"], extra_sha, parsed, False)
    ok("verify catches unlisted", any(p.startswith("UNLISTED") for p in problems))
    problems, _ = verify_checksums(rels + ["surprise.npy"], extra_sha, parsed, True)
    ok("verify allows unlisted when asked", problems == [])

    # DATASET_MANIFEST.json is written by build_v02_release.py AFTER
    # CHECKSUMS.sha256, so a clean build can never list it. It must not abort
    # the run, and the exemption must not extend to a digest mismatch.
    with_dm = dict(sha_of)
    with_dm[DATASET_MANIFEST_NAME] = "2" * 64
    problems, _ = verify_checksums(rels + [DATASET_MANIFEST_NAME], with_dm, parsed, False)
    ok("verify exempts DATASET_MANIFEST.json from UNLISTED", problems == [])
    stale = dict(parsed)
    stale[DATASET_MANIFEST_NAME] = "3" * 64
    problems, _ = verify_checksums(rels + [DATASET_MANIFEST_NAME], with_dm, stale, False)
    ok(
        "verify still catches a stale DATASET_MANIFEST.json digest",
        len(problems) == 1 and problems[0].startswith("MISMATCH"),
    )
    ok("stale manifest diagnostic names the builder", "build_v02_release.py" in problems[0])
    ok("checksums manifest itself exempt", CHECKSUMS_NAME in MANIFEST_EXEMPT)
    ok("state file exempt", STATE_NAME in MANIFEST_EXEMPT)

    # -- key mapping ---------------------------------------------------------
    ok("key preserve", zenodo_key("pockets/pocket_coords.h5", False) == "pockets/pocket_coords.h5")
    ok("key flatten", zenodo_key("pockets/pocket_coords.h5", True) == "pockets__pocket_coords.h5")
    pm = build_paths_manifest(["a/b.npy", "c.npy"], True)
    ok("paths manifest", pm == {"a__b.npy": "a/b.npy", "c.npy": "c.npy"})
    ok(
        "url quoting",
        urllib.parse.quote("pockets/pocket map.parquet", safe="")
        == "pockets%2Fpocket%20map.parquet",
    )

    # -- upload planning -----------------------------------------------------
    local = {
        "a.npy": {"size": 10, "md5": "aa"},
        "b.npy": {"size": 20, "md5": "bb"},
        "c.npy": {"size": 30, "md5": "cc"},
    }
    remote = {
        "a.npy": {"size": 10, "md5": "aa"},  # identical -> skip
        "b.npy": {"size": 20, "md5": "zz"},  # changed   -> upload
        "old.npy": {"size": 5, "md5": "dd"},  # stale     -> delete
    }
    up, skip, dele = plan_uploads(local, remote)
    ok("plan skip identical", skip == ["a.npy"])
    ok("plan upload changed+new", up == ["b.npy", "c.npy"])
    ok("plan delete stale", dele == ["old.npy"])
    up2, skip2, _ = plan_uploads(local, remote, force=True)
    ok("plan force", up2 == ["a.npy", "b.npy", "c.npy"] and skip2 == [])
    up3, skip3, del3 = plan_uploads(local, None)
    ok("plan empty remote", up3 == ["a.npy", "b.npy", "c.npy"] and del3 == [])
    # size differing with equal md5 must still re-upload (truncated object)
    remote_trunc = {"a.npy": {"size": 4, "md5": "aa"}}
    up4, skip4, _ = plan_uploads({"a.npy": local["a.npy"]}, remote_trunc)
    ok("plan catches truncated remote", up4 == ["a.npy"] and skip4 == [])

    # -- metadata ------------------------------------------------------------
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source_dir",
            str(tmp),
            "--version",
            "v0.2",
            "--publication_date",
            "2026-08-06",
            "--paper_doi",
            "10.1038/s41597-026-00000-0",
            "--supersedes",
            "10.5281/zenodo.11111111",
        ]
    )
    metadata = build_metadata(args, len(rels), 4_800_000_000)
    ok("metadata valid", validate_metadata(metadata) == [])
    ok("metadata upload_type", metadata["upload_type"] == "dataset")
    ok("metadata license", metadata["license"] == "cc-by-4.0")
    rel_pairs = set((r["relation"], r["identifier"]) for r in metadata["related_identifiers"])
    # Inverted deliberately. The Hugging Face copy is the preview, so these
    # relations were false; the test now fails if they come back.
    ok("no isIdenticalTo -> HF url", ("isIdenticalTo", HF_DATASET_URL) not in rel_pairs)
    ok("no isIdenticalTo -> HF doi", ("isIdenticalTo", HF_DATASET_DOI) not in rel_pairs)
    ok(
        "relation isSupplementTo paper",
        ("isSupplementTo", "10.1038/s41597-026-00000-0") in rel_pairs,
    )
    ok("relation isDerivedFrom PDB", ("isDerivedFrom", PDB_URL) in rel_pairs)
    ok("relation isDerivedFrom BindingDB", ("isDerivedFrom", BINDINGDB_URL) in rel_pairs)
    ok("relation isDocumentedBy code DOI", ("isDocumentedBy", PIPELINE_CODE_DOI) in rel_pairs)
    ok("relation obsoletes v0.1", ("obsoletes", "10.5281/zenodo.11111111") in rel_pairs)
    ok(
        "all relations legal",
        all(r["relation"] in ZENODO_RELATIONS for r in metadata["related_identifiers"]),
    )
    ok("regeneration disclosed in notes", "2026-05-01" in metadata["notes"])
    ok(
        "regeneration disclosure names the affected files",
        all(
            name in metadata["notes"]
            for name in ("metadata.jsonl", "joined_embeddings.npy", "retrieval.faiss")
        ),
    )
    ok(
        "regeneration disclosure names what did NOT change",
        "HDF5 coordinate files were not changed" in metadata["notes"],
    )
    ok(
        "integrity paragraph does not overclaim checksum coverage",
        "checksums for every file" not in metadata["description"]
        and DATASET_MANIFEST_NAME in metadata["description"],
    )
    ok("no convenience-mirror claim", "convenience mirror" not in metadata["description"])
    ok(
        "description quotes the measured pocket-only rate",
        _pct(_M["same_uniprot_hit_at_1_by_alpha"]["1.00_pocket_only"]) in metadata["description"],
    )
    ok("co-equal modalities in description", "co-equal" in metadata["description"])
    ok("description has no stray format tokens", "%(" not in metadata["description"])

    # metadata patch file
    patch = tmp / "patch.json"
    patch.write_text(json.dumps({"metadata": {"title": "Patched title"}}), encoding="utf-8")
    args2 = parser.parse_args(
        [
            "--source_dir",
            str(tmp),
            "--metadata_json",
            str(patch),
            "--publication_date",
            "2026-08-06",
        ]
    )
    ok("metadata_json patch", build_metadata(args2, 1, 1)["title"] == "Patched title")

    # validator actually rejects things
    bad = dict(metadata)
    bad["license"] = "cc-by-nc-4.0"
    ok("validator rejects NC license", any("license" in p for p in validate_metadata(bad)))
    bad = dict(metadata)
    bad["access_right"] = "restricted"
    ok("validator rejects closed access", any("access_right" in p for p in validate_metadata(bad)))
    bad = dict(metadata)
    bad["publication_date"] = "06/08/2026"
    ok("validator rejects bad date", any("publication_date" in p for p in validate_metadata(bad)))
    bad = dict(metadata)
    bad["upload_type"] = "software"
    ok(
        "validator rejects wrong upload_type",
        any("upload_type" in p for p in validate_metadata(bad)),
    )
    bad = dict(metadata)
    bad["creators"] = [{"name": "Hemanth Kari"}]
    ok(
        "validator rejects unformatted creator",
        any("Family, Given" in p for p in validate_metadata(bad)),
    )
    bad = dict(metadata)
    bad["creators"] = [{"name": "Kari, Hemanth", "orcid": "1234"}]
    ok("validator rejects bad orcid", any("orcid" in p for p in validate_metadata(bad)))
    bad = dict(metadata)
    bad["related_identifiers"] = [
        {"relation": "isMirrorOf", "identifier": "https://x", "scheme": "url"}
    ]
    problems_b = validate_metadata(bad)
    ok("validator rejects bogus relation", any("isMirrorOf" in p for p in problems_b))
    ok(
        "validator requires a code relation",
        any("names no code record" in p for p in problems_b),
    )
    # isCompiledBy satisfies it, and an absent isIdenticalTo is no longer a
    # problem: the Hugging Face copy holds v0.1, so claiming identity is false.
    good_rel = dict(metadata)
    good_rel["related_identifiers"] = [
        {"relation": "isDerivedFrom", "identifier": "https://www.rcsb.org", "scheme": "url"},
        {"relation": "isCompiledBy", "identifier": "10.5281/zenodo.22258470", "scheme": "doi"},
    ]
    ok(
        "validator accepts isCompiledBy without isIdenticalTo",
        not validate_metadata(good_rel),
    )
    bad = dict(metadata)
    bad["related_identifiers"] = list(metadata["related_identifiers"]) + [
        {"relation": "isDocumentedBy", "identifier": "not-a-doi", "scheme": "doi"}
    ]
    ok("validator rejects fake DOI", any("not a DOI" in p for p in validate_metadata(bad)))

    # -- description must not restate defects the revision corrects ----------
    desc = metadata["description"]
    ok(
        "description does not call the pocket coords heavy-atom coordinates",
        "heavy-atom coordinates for pocket" not in desc,
    )
    ok(
        "description states the pocket centroid definition",
        "per-residue heavy-atom centroids" in desc,
    )
    ok(
        "description does not claim a drug-like ligand set",
        "drug-like ligand per entry" not in desc,
    )
    ok("description discloses the cofactor share", "25.8" in desc)
    # These were hard-coded preview figures (80.7 / 64.8 / 29.6 / 63.1 / 34.5).
    # They now render from MEASURED, so the test asserts the MEASURED values
    # appear and that no preview figure survives.
    _want = [
        _pct(_M["same_uniprot_hit_at_1_by_alpha"]["1.00_pocket_only"]),
        _pct(_M["same_uniprot_hit_at_1_by_alpha"]["concatenation_retained"]),
        _pct(_M["same_uniprot_hit_at_1_by_alpha"]["0.00_ligand_only"]),
        _pct(_M["same_ccd_top_1_by_alpha"]["0.00_ligand_only"]),
        _pct(_M["same_ccd_top_1_by_alpha"]["1.00_pocket_only"]),
    ]
    ok(
        "description carries the measured alpha-sweep numbers",
        all(v in desc for v in _want),
    )
    ok(
        "description carries no preview-era alpha-sweep number",
        not any(v in desc for v in ("80.7", "64.8", "29.6", "63.1", "34.5")),
    )
    # Whitespace-insensitive: the template wraps this pair across a line, and
    # HTML collapses that anyway, so a single-line match would fail on correct
    # output.
    _flat = " ".join(desc.split())
    ok(
        "description names the ligand-only/pocket-only same-CCD pair explicitly",
        "%s&nbsp;%% ligand-only against %s&nbsp;%% pocket-only" % (_want[3], _want[4]) in _flat,
    )

    # -- metadata fallbacks are pure transforms ------------------------------
    flipped = ZenodoClient._flip_license({"license": "cc-by-4.0"})
    ok("license flip to object", flipped["license"] == {"id": "cc-by-4.0"})
    ok("license flip back", ZenodoClient._flip_license(flipped)["license"] == "cc-by-4.0")
    stripped = ZenodoClient._strip_related_resource_types(metadata)
    ok(
        "resource_type stripped",
        all("resource_type" not in r for r in stripped["related_identifiers"]),
    )
    ok(
        "resource_type strip keeps relations",
        set((r["relation"], r["identifier"]) for r in stripped["related_identifiers"]) == rel_pairs,
    )
    ok(
        "resource_type strip leaves the original intact",
        any("resource_type" in r for r in metadata["related_identifiers"]),
    )

    # -- metadata_only requires a deposition id ------------------------------
    args3 = parser.parse_args(["--metadata_only", "--publication_date", "2026-08-06"])
    ok("metadata_only needs no source_dir", args3.source_dir is None)
    try:
        _run_metadata_only(args3)
        ok("metadata_only rejects a missing --deposition", False)
    except SystemExit as exc:
        ok("metadata_only rejects a missing --deposition", "--deposition" in str(exc))

    # -- record id resolution (pure parsing, no network) ---------------------
    for ref, expect in (
        ("10.5281/zenodo.19954940", "19954940"),
        ("https://doi.org/10.5281/zenodo.123", "123"),
        ("456789", "456789"),
    ):
        m = re.search(r"zenodo\.(\d+)", ref)
        got = m.group(1) if m else (ref if ref.isdigit() else None)
        ok("recid parse %s" % ref, got == expect)

    # -- client construction without a token must not explode ----------------
    client = ZenodoClient(ZENODO_SANDBOX_API, "", max_retries=3, backoff=1.0)
    ok("client base normalized", client.base == ZENODO_SANDBOX_API)
    ok(
        "jitter within bounds",
        all(0.7 <= ZenodoClient._jitter([2.0, 4.0], 1) / 2.0 <= 1.3 for _ in range(50)),
    )
    ok(
        "retry statuses",
        502 in RETRY_STATUS and 404 not in RETRY_STATUS and 403 not in RETRY_STATUS,
    )

    # -- progress reader returns exactly the file, in big blocks -------------
    big = tmp / "big.bin"
    big.write_bytes(b"z" * 100000)
    reader = _ProgressReader(big, 100000, "big.bin", chunk=1 << 16, quiet=True)
    blocks, total = 0, 0
    while True:
        chunk = reader.read(8192)
        if not chunk:
            break
        blocks += 1
        total += len(chunk)
    reader.close()
    ok("progress reader byte-exact", total == 100000)
    ok("progress reader coalesces", blocks <= 3)

    print("mirror_to_zenodo self-test: OK (%d checks)" % len(checks))
    print("  payload discovery, sha256/md5 hashing, manifest parse+verify,")
    print("  key mapping, upload/skip/delete planning, metadata build+validation,")
    print("  backoff schedule, token redaction, streaming reader.")


if __name__ == "__main__":
    main()
