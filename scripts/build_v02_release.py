"""
Assemble the v0.2 release of the RCSB protein-ligand retrieval dataset.

Input is the immutable v0.1 snapshot (HuggingFace HemanthKari/rcsb-ligand-retrieval-v0.1)
plus the artifacts produced by notebook 09 (pocket_residue_map.parquet,
esm_input_sequences.fasta.gz). Nothing under --v01_dir is ever written to.

What v0.2 changes, and why (every item traces to a measured audit finding):

  1. LABELLED COORDINATES. v0.1 ships bare (n, 3) float32 arrays. Element symbols
     were extracted by embed_ligands.py and then discarded; pocket residue identity
     was never recorded at all. v0.2 writes parallel, row-aligned label datasets
     into the same HDF5 files (the xyz dataset paths are unchanged, so v0.1 reader
     code keeps working) and documents the atom order explicitly (ATOM_ORDER_DOC /
     POCKET_ORDER_DOC below). Ligand labels are re-extracted from mmCIF with the
     *same* walk embed_ligands.py used, and are written only for entries whose
     re-extracted coordinates match the shipped array (witness check), so a label
     can never be silently misaligned with a coordinate.

  2. CCD -> FINGERPRINT TABLE. ligand_embeddings.npy is 593 MB of 2.71x-redundant
     rows: the Morgan fingerprint is a pure function of the CCD code (26,619
     distinct vectors over the 72,199 RELEASED complexes; 0 CCD codes map to >1
     vector). v0.2 ships ccd_fingerprints.npz (bit-packed, ~7 MB) keyed by CCD code,
     plus ligand_row_to_ccd.npy and a ligand_ccd metadata column that make the
     redundancy self-evident. The per-row matrix is retained for compatibility.
     NOTE: the table itself is built from ligand_manifest.jsonl, which also covers
     the ~189 zero-fingerprint ligands the joined index dropped, so the report
     carries BOTH the manifest-scoped and the metadata-scoped (= published) counts.

  3. PER-MODALITY INDICES. The alpha sweep (same-UniProt hit@1: ligand-only 0.298,
     pocket-only 0.812, retained concatenation 0.640; same-CCD top-1 runs the other
     way, 0.624 vs 0.366) shows the concatenation is not a good default for either
     job. v0.2 makes pocket.faiss and ligand.faiss co-equal primary artifacts and
     ships the measured guidance table; the concatenated index is retained. The
     guidance also carries the measured PER-PFAM-FAMILY stratification (PF00069
     protein kinase, the largest family, is the WORST at hit@1 0.643 vs 0.983 for
     the best of the top 12) - the aggregate number alone is misleading.

  4. FIXED SPLITS. The shipped v0.1 split (68,459/3,608) neither matches its own
     documented seed+val_frac (68,464/3,603) nor partitions the usable set: 34
     entries in the splits are unusable, 132 usable entries are in no split. v0.2
     recomputes it so it exactly partitions the usable set, and adds a
     target-disjoint split clustered on the REAL ESM-2 input sequences (not the
     post-hoc protein_seq annotation, which is missing for 457 entries), with a
     ligand-CCD overlap report across the boundary.

  5. ENRICHED METADATA, row order preserved (the splits index into it).

  6. UNPARSEABLE SMILES FLAGGED. The 189 zero-vector ligands are 189/189 RDKit
     SMILES parse failures, not "metal/borane/polyoxometalate" chemistry. v0.2
     ships them as an explicit table instead of dropping them silently.

  7. CHECKSUMS.sha256, DATASET_MANIFEST.json, CHANGELOG.md.

Every stage is independently runnable and re-runnable:

    python scripts/build_v02_release.py label_coords --v01_dir DIR --nb09_dir DIR \
        --out_dir DIR --pdb_dir MIRROR
    python scripts/build_v02_release.py ccd_table       --v01_dir DIR --out_dir DIR
    python scripts/build_v02_release.py flag_unparseable --v01_dir DIR --out_dir DIR
    python scripts/build_v02_release.py metadata        --v01_dir DIR --nb09_dir DIR \
        --out_dir DIR --sifts_dir DIR
    python scripts/build_v02_release.py indices         --v01_dir DIR --out_dir DIR
    python scripts/build_v02_release.py splits          --v01_dir DIR --nb09_dir DIR \
        --out_dir DIR
    python scripts/build_v02_release.py carry_over      --v01_dir DIR --out_dir DIR
    python scripts/build_v02_release.py manifest        --v01_dir DIR --out_dir DIR
    python scripts/build_v02_release.py all             --v01_dir DIR --nb09_dir DIR \
        --out_dir DIR --pdb_dir MIRROR --sifts_dir DIR

    python scripts/build_v02_release.py --self_test     # synthetic data, no dataset

Style note: this repository forbids f-strings; all formatting is %-style.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RELEASE_VERSION = "0.2.0"
BUILDER_VERSION = "1.0.0"
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

# ---------------------------------------------------------------------------
# Documented contracts
# ---------------------------------------------------------------------------

ATOM_ORDER_DOC = (
    "ligands/<pocket_key> is (n_atoms, 3) float32 Cartesian coordinates in Angstrom, in "
    "the deposited mmCIF frame (no centering, no superposition). "
    "ligand_atom_labels/<pocket_key> is (n_atoms,) with fields "
    "(atom_name S8, element S4, altloc S2); row i labels row i of the coordinate array. "
    "ORDER: mmCIF _atom_site record order, restricted to the FIRST model, the chain whose "
    "auth_asym_id equals metadata.ligand_chain, and residues whose comp_id equals the ligand "
    "CCD code AND whose auth_seq_id equals metadata.ligand_resnum; atoms with element == H are "
    "skipped. Alternate conformers are NOT filtered - every altloc present in the file is kept, "
    "in file order, and the altloc field records which. This reproduces the order used by "
    "rcsb_ligand_retrieval/retrieval/embed_ligands.py::_extract_ligand, i.e. the order the "
    "shipped v0.1 coordinates are already in. It is NOT RDKit canonical order and NOT the CCD "
    "reference atom order; use fluxion_dock/data/atom_order.py to map to an RDKit molecule."
)

POCKET_ORDER_DOC = (
    "pockets/<pocket_key> is (n_residues, 3) float32: the per-residue MEAN of that residue's "
    "heavy-atom coordinates (not CA, not backbone centroid), in the deposited mmCIF frame. "
    "pocket_residue_labels/<pocket_key> is (n_residues,) with fields "
    "(token_index i4, chain S8, seqid i4, icode S2, resname S4); row i labels row i of the "
    "coordinate array. token_index is the 0-based position of the residue in the ESM-2 INPUT "
    "sequence, which is the concatenation of every polymer chain of model 0 whose residue name "
    "maps to a standard one-letter code (48.85% of entries are multi-chain, so this input is "
    "chimeric; 8,454 non-standard residues consume no sequence position and were dropped). "
    "The ESM-2 token position is token_index + 1 because BOS occupies position 0. "
    "ORDER: increasing token_index, i.e. the order residues were encountered in the model-0 "
    "chain walk, which is the order rcsb_ligand_retrieval/retrieval/embed_pockets.py::"
    "_extract_pocket appended them. Labels come from notebook 09's pocket_residue_map.parquet "
    "and are written only for entries whose re-extraction reproduced the shipped coordinates "
    "exactly."
)

# Measured on the v0.1 corpus (audit, notebook 08). Shipped with the indices so users
# pick a combination rule from numbers rather than from the layout of a concatenation.
# Authoritative measurements for this release. Every number below is produced
# by scripts/make_dataset_paper_figures.py against the deposited arrays and is
# mirrored in paper/figures/fig4_stratified_retrieval.json and
# fig5_embedding_geometry.json. The previous values were measured on the preview
# corpus BEFORE the per-chain re-embedding and were baked in as literal text in
# three separate places, which is why the published deposit disagreed with the
# manuscript. Change them here and nowhere else.
MEASURED = {
    "same_uniprot_hit_at_1_by_alpha": {
        "0.00_ligand_only": 0.298,
        "0.25": 0.612,
        "0.50": 0.657,
        "0.75": 0.726,
        "1.00_pocket_only": 0.812,
        "concatenation_retained": 0.640,
    },
    "same_ccd_top_1_by_alpha": {
        "0.00_ligand_only": 0.624,
        "1.00_pocket_only": 0.366,
        "concatenation_retained": 0.604,
    },
    # Two different objects, and they must not be mixed up again.
    # w_p: a concatenation of the RELEASED PER-CHAIN matrices - the index a user
    #      would build today. Used by the Supplementary Information.
    # w_p_shipped_joint_index: the RETAINED retrieval.faiss / joined_embeddings.npy,
    #      carried over from the preview, whose pocket block is the all-chain
    #      encoding. Measured over all 72,199 rows. Any sentence about the SHIPPED
    #      index must use this one; the deposit already states it correctly.
    "w_p": {"median": 0.470, "iqr": [0.366, 0.603], "range": [0.152, 0.968]},
    # Measured on fig3_technical_validation.json, which is generated from the
    # released matrices. The deposited text had 0.844 / 0.706 / -0.027, and the
    # centred median had the wrong SIGN.
    "anisotropy": {
        "pocket_mean_norm_ratio": 0.832,
        "pocket_random_raw": 0.719,
        "pocket_random_centred": 0.005,
    },
    # recall@1 of the retained concatenated index against exact search.
    "ann_recall_concat": {"efSearch_16": 0.957, "efSearch_256": 0.971},
    "w_p_shipped_joint_index": {
        "median": 0.513,
        "iqr": [0.408, 0.631],
        "range": [0.166, 0.966],
    },
    "duplicates": {
        "groups": 1653,
        "entries": 5143,
        "pct": 7.12,
        "groups_crossing_split": 212,
        "val_frac_train_nbr_ge_099": 0.457,
        "val_frac_exact_duplicate": 0.030,
    },
    "per_family_hit_at_1": {"PF00069": 0.643, "PF02898": 0.983},
}


def _split_warning():
    """The randomsplit caveat, rendered from MEASURED so it cannot go stale."""
    d = MEASURED["duplicates"]
    return (
        "This split is RANDOM. Direct measurement over the released index finds "
        "%d exact-duplicate groups covering %d entries (%.2f%%), %d of them "
        "crossing the boundary; %.1f%% of val entries have a train neighbour at "
        "cosine >= 0.99 and %.1f%% are an exact duplicate. It is a reproducibility "
        "split, NOT a generalization benchmark - use the target-disjoint split "
        "for that."
        % (
            d["groups"],
            d["entries"],
            d["pct"],
            d["groups_crossing_split"],
            100 * d["val_frac_train_nbr_ge_099"],
            100 * d["val_frac_exact_duplicate"],
        )
    )


MODALITY_GUIDANCE = {
    "score_rule": (
        "score(q, r) = alpha * cos(pocket_q, pocket_r) + (1 - alpha) * cos(ligand_q, ligand_r); "
        "both indices store L2-normalised vectors under METRIC_INNER_PRODUCT, so the FAISS "
        "score IS the cosine and the two are directly combinable."
    ),
    # Rendered, not hardcoded: DATASET_MANIFEST.json embeds this dict, so a literal
    # here is a second copy that refresh_release_sidecars.py cannot reach. It drifted
    # to 0.470 - the per-chain reconstruction - while the sidecar said 0.513.
    "why_two_indices": (
        "The retained concatenated index is equivalent to a FIXED alpha determined by the "
        "relative norms of the two blocks. Those norms are not constant: the pocket share "
        "of the joint squared norm has median %.3f, IQR %.3f-%.3f and range %.3f-%.3f, so "
        "the effective alpha varies per query. Separate indices make the weighting explicit."
        % (
            MEASURED["w_p_shipped_joint_index"]["median"],
            MEASURED["w_p_shipped_joint_index"]["iqr"][0],
            MEASURED["w_p_shipped_joint_index"]["iqr"][1],
            MEASURED["w_p_shipped_joint_index"]["range"][0],
            MEASURED["w_p_shipped_joint_index"]["range"][1],
        )
    ),
    "same_uniprot_hit_at_1_by_alpha": MEASURED["same_uniprot_hit_at_1_by_alpha"],
    "same_ccd_top_1_by_alpha": MEASURED["same_ccd_top_1_by_alpha"],
    "recommended": {
        "find_same_target": "alpha = 1.0 (pocket.faiss alone)",
        "find_same_or_similar_ligand": "alpha = 0.0 (ligand.faiss alone)",
    },
    "anisotropy_warning": (
        "The corpus mean vector has norm %.3fx a typical row norm; a random pocket pair "
        "scores %.3f raw but %.3f after mean-centering. RANKINGS are robust to this (kNN "
        "purity UniProt 0.443 -> 0.446, Pfam 0.557 -> 0.566 when centred) but ABSOLUTE "
        "cosines are not interpretable as similarity. Do not threshold on raw cosine."
        % (
            MEASURED["anisotropy"]["pocket_mean_norm_ratio"],
            MEASURED["anisotropy"]["pocket_random_raw"],
            MEASURED["anisotropy"]["pocket_random_centred"],
        )
    ),
    "ann_recall_vs_exact": {
        "efSearch_16": MEASURED["ann_recall_concat"]["efSearch_16"],
        "efSearch_256": MEASURED["ann_recall_concat"]["efSearch_256"],
        "note": "HNSW recall@1 against exact search on the retained concatenated index.",
    },
}

# The EXACT set the audit (notebook 08, cell 5, "4_filter_reality") used to measure
# "25.8% of selected ligands are common cofactors or buffer components". It is
# reproduced here VERBATIM so the number quoted in the paper stays reproducible from
# the shipped release. Do not extend this set - extend COMMON_COFACTOR_CCD instead.
AUDIT_COMMON_ARTIFACTS_CCD = {
    "HEM",
    "FAD",
    "ADP",
    "NAP",
    "FMN",
    "GDP",
    "NAD",
    "SAH",
    "MES",
    "CIT",
    "ATP",
    "AMP",
    "NDP",
    "COA",
    "SAM",
    "PLP",
    "GTP",
    "TRS",
    "EPE",
    "ACT",
    "BTB",
    "MPD",
    "IPA",
    "PO4",
    "FES",
    "SF4",
    "HEC",
    "BCL",
    "CLA",
    "UDP",
    "UMP",
    "GSH",
    "NAI",
    "FDA",
    "TPP",
    "B12",
}

# Broader, versioned list backing the new is_common_cofactor annotation. v0.1 had no
# such field. The flag is ADVISORY - it marks ligands that are usually incidental to the
# crystallisation rather than the biological subject of the structure. It is a SUPERSET
# of AUDIT_COMMON_ARTIFACTS_CCD and therefore flags MORE entries than the audit's 25.8%;
# both percentages are reported side by side so the two can never be confused.
# It is deliberately separate from EXCLUDED_HET (the 35-code build-time blacklist in
# rcsb_ligand_retrieval/data/rcsb_filter.py), which is reproduced in the manifest. A few
# codes below (PGE, PG4, TRS, PEG-family, ACT, PO4, MPD) are also in EXCLUDED_HET and are
# therefore inert - they can never reach metadata.jsonl as a selected ligand; the
# metadata report counts the overlap explicitly rather than hiding it.
COMMON_COFACTOR_CCD = {
    # adenine/guanine/pyrimidine nucleotides and non-hydrolysable analogues
    "ATP",
    "ADP",
    "AMP",
    "ANP",
    "ACP",
    "AGS",
    "APC",
    "ADX",
    "A2P",
    "AP5",
    "GTP",
    "GDP",
    "GMP",
    "GNP",
    "GSP",
    "GCP",
    "G3P",
    "CTP",
    "CDP",
    "CMP",
    "UTP",
    "UDP",
    "UMP",
    "TTP",
    "DTP",
    "UPG",
    # nicotinamide cofactors
    "NAD",
    "NAI",
    "NAP",
    "NDP",
    "NAX",
    "NAH",
    "NMN",
    # flavins
    "FAD",
    "FDA",
    "FMN",
    "FNS",
    "RBF",
    # coenzyme A and thioesters
    "COA",
    "ACO",
    "SCA",
    "MLC",
    "CMX",
    "COO",
    # methyl donors / one-carbon
    "SAM",
    "SAH",
    "MTA",
    "FOL",
    "THF",
    "DHF",
    "MTE",
    # haem and porphyrins
    "HEM",
    "HEC",
    "HEA",
    "HAS",
    "DHE",
    "SRM",
    "COH",
    "HDD",
    "VER",
    # vitamin-derived cofactors
    "PLP",
    "PMP",
    "PLR",
    "TPP",
    "BTN",
    "B12",
    "COB",
    "CNC",
    "MQ7",
    "UQ1",
    "UQ2",
    # peptide-like / redox cofactors and metabolites
    "GSH",
    "GDS",
    "PYR",
    "CIT",
    "AKG",
    "SIN",
    "MLI",
    "MLA",
    "TAR",
    "FLC",
    "OXL",
    # sugars and polyols frequently present as cryo/additive or glycan fragments
    "GLC",
    "BGC",
    "MAN",
    "GAL",
    "GLA",
    "SUC",
    "TRE",
    "MAL",
    "LAT",
    "SOR",
    "XYS",
    "INS",
    "SRT",
    "MBG",
    "NDG",
    "A2G",
    # buffers / cryoprotectants / detergents that clear the >=10 heavy-atom cut
    "MES",
    "EPE",
    "BTB",
    "TRS",
    "BIS",
    "CAC",
    "IMD",
    "TAM",
    "POP",
    "HED",
    "PGE",
    "PG4",
    "1PE",
    "2PE",
    "P6G",
    "PE4",
    "PEU",
    "XPE",
    "P33",
    "7PE",
    "BOG",
    "LDA",
    "LMT",
    "BNG",
    "C8E",
    "OCT",
    "DDQ",
    "TRT",
    "SPD",
    "SPM",
    "MPO",
    "MRD",
    "PGO",
    "PIG",
    "TOE",
    "DXG",
    # fatty acids / lipids that are usually adventitious
    "PLM",
    "MYR",
    "OLA",
    "STE",
    "LAU",
    "DAO",
    "CHD",
    "CLR",
    "CHO",
    "PC1",
}

# Superset property enforced by construction, not by hand-maintenance.
COMMON_COFACTOR_CCD |= AUDIT_COMMON_ARTIFACTS_CCD

# Measured in the audit (notebook 08, cells 12a/12b) on the SHIPPED v0.1 concatenated
# index. Shipped with the release because "how well does retrieval work, and for which
# protein families" is the question a user of this dataset actually has, and the answer
# is very uneven. Only the endpoints of the per-family sweep were recorded; the rest of
# the top-12 families are reproducible from notebook 08 and are deliberately NOT
# invented here.
RETRIEVAL_BENCHMARK = {
    "task": (
        "same-UniProt hit@k: query the index with a corpus row, exclude the query itself, "
        "count a hit if any of the top k neighbours shares the query's SIFTS SP_PRIMARY "
        "accession."
    ),
    "index": "retrieval.faiss (v0.1 concatenated [pocket || ligand], L2-normalised)",
    "n_queries": 3000,
    # hit@1 is the figure-script measurement on the released arrays;
    # hit@5 and hit@10 are from the audit run and have no re-measured
    # counterpart, which this note records rather than implying one source.
    "overall": {
        "hit_at_1": MEASURED["same_uniprot_hit_at_1_by_alpha"]["concatenation_retained"],
        "hit_at_5": 0.744,
        "hit_at_10": 0.769,
        "hit_at_5_10_source": "audit run; not re-measured after re-embedding",
    },
    "published_in_v0_1_paper": {"hit_at_1": 0.651, "hit_at_5": 0.746, "hit_at_10": 0.761},
    "stratified_by_pfam_hit_at_1": {
        "PF00069_protein_kinase": {
            "hit_at_1": MEASURED["per_family_hit_at_1"]["PF00069"],
            "rank": "WORST of the top-12 families, and the LARGEST family in the corpus",
        },
        "PF02898": {
            "hit_at_1": MEASURED["per_family_hit_at_1"]["PF02898"],
            "rank": "best of the top-12 families",
        },
    },
    "interpretation": (
        "The aggregate number is NOT representative. Retrieval is weakest exactly where "
        "the corpus is densest: protein kinases (PF00069), the single largest family, are "
        "the worst-performing of the top-12 families at hit@1 0.643, against 0.983 for the "
        "best. Anyone benchmarking on this dataset should report per-family numbers; an "
        "aggregate hit@1 hides a ~0.37 spread across families. Per-family stratification is "
        "possible directly from the shipped metadata via the new pfam_acc column."
    ),
}

LIGAND_LABEL_DTYPE = np.dtype([("atom_name", "S8"), ("element", "S4"), ("altloc", "S2")])
POCKET_LABEL_DTYPE = np.dtype(
    [("token_index", "i4"), ("chain", "S8"), ("seqid", "i4"), ("icode", "S2"), ("resname", "S4")]
)

COORD_MATCH_TOL = 1e-3  # Angstrom; float32 round-trip through gzip HDF5 is exact, this is slack


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def log(msg):
    print(msg, flush=True)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path):
    rows = []
    with open(str(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def open_text_write(path):
    """Open for writing with LF endings on EVERY platform.

    Python's text mode translates "\\n" to "\\r\\n" on Windows. That silently makes
    CHECKSUMS.sha256 unusable with `sha256sum -c` (every path comes back with a
    trailing CR), and it makes the whole release non-reproducible byte-for-byte
    against a Linux build. Every text artifact in the release goes through here.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(str(path), "w", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open_text_write(path) as fh:
        for row in rows:
            # default=_json_default: pandas/parquet round-trips hand back numpy scalars,
            # and a bare json.dumps would raise mid-file, leaving a TRUNCATED
            # metadata.jsonl that still looks like a valid release artifact.
            fh.write(json.dumps(row, default=_json_default) + "\n")


def clean_str(value):
    """Normalise a SIFTS / pandas cell to a real string or None.

    float('nan') is TRUTHY, so the obvious `str(x) if x else None` silently writes the
    literal string "nan" into uniprot_acc / pfam_acc for every entry SIFTS has no
    annotation for. This is the guard for that.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    s = str(value).strip()
    if s in ("", "nan", "NaN", "None", "?", "-"):
        return None
    return s


def opt_int(value):
    """int() a possibly-absent / possibly-NaN cell without raising."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def opt_float(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def write_json(path, payload):
    with open_text_write(path) as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def report_path(out_dir, stage):
    return Path(out_dir) / "provenance" / (stage + "_report.json")


def write_report(out_dir, stage, payload):
    payload = dict(payload)
    payload["_stage"] = stage
    payload["_written_utc"] = utcnow()
    payload["_builder_version"] = BUILDER_VERSION
    path = report_path(out_dir, stage)
    write_json(path, payload)
    log("[%s] report -> %s" % (stage, path))
    return path


def load_report(out_dir, stage):
    path = report_path(out_dir, stage)
    if not path.exists():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log(
            "[load_report] WARNING: %s is unreadable (%s); the manifest will NOT carry "
            "this stage's report" % (path, exc)
        )
        return None


def _load_module_from_path(name, path):
    """Import a sibling script by path so its logic is REUSED, not duplicated."""
    import importlib.util

    path = Path(path)
    if not path.exists():
        raise SystemExit("required sibling module not found: %s" % path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load module %s from %s" % (name, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_excluded_het():
    """Import EXCLUDED_HET live from the deposited pipeline. Returns (sorted list, error).

    Imported rather than copied so the blacklist reproduced in DATASET_MANIFEST.json can
    never drift from the one the build actually applied.
    """
    path = (
        REPO_ROOT / "zenodo_pipeline_release" / "rcsb_ligand_retrieval" / "data" / "rcsb_filter.py"
    )
    try:
        rf = _load_module_from_path("_v02_rcsb_filter", path)
        return sorted(rf.EXCLUDED_HET), None
    except Exception as exc:
        return None, "%s: %s (looked in %s)" % (type(exc).__name__, exc, path)


def require(module_name, hint):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit("%s is required for this stage. %s" % (module_name, hint)) from exc


def git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _bfield(value, width):
    """ASCII-encode and truncate for a fixed-width HDF5 string field."""
    if value is None:
        return b""
    s = str(value).strip()
    return s.encode("ascii", "replace")[:width]


def copy_file(src, dst, mode="copy"):
    """mode: copy | link (hardlink, falls back to copy) | skip."""
    src = Path(src)
    dst = Path(dst)
    if mode == "skip":
        return "skipped"
    if not src.exists():
        return "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "link":
        try:
            os.link(str(src), str(dst))
            return "hardlinked"
        except Exception:
            pass
    shutil.copy2(str(src), str(dst))
    return "copied"


def l2_normalise_rows(mat, eps=1e-12):
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    zero = norms < eps
    norms[zero] = 1.0
    return (mat / norms).astype(np.float32), int(zero.sum())


# ---------------------------------------------------------------------------
# Stage 1: label the coordinate files
# ---------------------------------------------------------------------------


def extract_ligand_atom_labels(structure, ligand_chain, ligand_resname, ligand_resnum):
    """Re-walk one ligand exactly as embed_ligands.py::_extract_ligand did.

    Returns (coords (n,3) float32, labels (n,) LIGAND_LABEL_DTYPE). The loop
    structure - first model only, auth chain name match, comp id AND auth seq id
    match, element != H, no altloc filtering, no break on first match - is a
    deliberate mirror: any deviation would misalign the labels against the shipped
    coordinates, which is why the caller also runs a coordinate witness check.
    """
    coords = []
    labels = []
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
                    element = atom.element.name
                    if element == "H":
                        continue
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    altloc = getattr(atom, "altloc", "") or ""
                    altloc = str(altloc).replace("\x00", "").strip()
                    labels.append((_bfield(atom.name, 8), _bfield(element, 4), _bfield(altloc, 2)))
        break  # first model only
    if not coords:
        return None, None
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels_arr = np.zeros(len(labels), dtype=LIGAND_LABEL_DTYPE)
    for i, tup in enumerate(labels):
        labels_arr[i] = tup
    return coords_arr, labels_arr


def pocket_labels_from_map(indices_json, provenance_json, n_rows):
    """Build the row-aligned pocket residue label array. Returns (array, status)."""
    try:
        idxs = json.loads(indices_json) if isinstance(indices_json, str) else list(indices_json)
        prov = (
            json.loads(provenance_json)
            if isinstance(provenance_json, str)
            else list(provenance_json)
        )
    except Exception as exc:
        return None, "map_json_error:%s" % type(exc).__name__
    if len(idxs) != n_rows:
        return None, "index_count_mismatch:%d_vs_%d" % (len(idxs), n_rows)
    if len(prov) != n_rows:
        # notebook 09 drops provenance entries it could not resolve, so a short
        # provenance list means the mapping is ambiguous - refuse to guess.
        return None, "provenance_count_mismatch:%d_vs_%d" % (len(prov), n_rows)
    arr = np.zeros(n_rows, dtype=POCKET_LABEL_DTYPE)
    for i in range(n_rows):
        p = prov[i]
        arr[i] = (
            int(idxs[i]),
            _bfield(p.get("chain", ""), 8),
            int(p.get("seqid", 0)),
            _bfield(p.get("icode", ""), 2),
            _bfield(p.get("resname", ""), 4),
        )
    return arr, "ok"


def _load_pocket_residue_map(nb09_dir):
    """pocket_key -> dict of notebook-09 columns. Requires pandas."""
    path = Path(nb09_dir) / "pocket_residue_map.parquet"
    if not path.exists():
        raise SystemExit("pocket_residue_map.parquet not found in %s" % nb09_dir)
    pd = require("pandas", "pip install pandas pyarrow")
    df = pd.read_parquet(str(path))
    out = {}
    for row in df.to_dict("records"):
        out[row["pocket_key"]] = row
    return out


def _cif_bytes(pdb_id, pdb_dir, cif_index, cif_cache, fetch_missing):
    """Return the path to a readable .cif.gz for pdb_id, or None."""
    pid = pdb_id.lower()
    candidates = []
    if pdb_dir is not None:
        rel = cif_index.get(pid)
        if rel:
            candidates.append(Path(pdb_dir) / rel)
        candidates.append(Path(pdb_dir) / ("%s.cif.gz" % pid))
        candidates.append(Path(pdb_dir) / pid[1:3] / ("%s.cif.gz" % pid))
    if cif_cache is not None:
        candidates.append(Path(cif_cache) / ("%s.cif.gz" % pid))
    for cand in candidates:
        if cand.exists():
            return cand
    if not fetch_missing or cif_cache is None:
        return None
    try:
        import urllib.request

        url = "https://files.rcsb.org/download/%s.cif.gz" % pid.upper()
        dst = Path(cif_cache) / ("%s.cif.gz" % pid)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        if not data:
            return None
        dst.write_bytes(data)
        return dst
    except Exception:
        return None


def stage_label_coords(args):
    h5py = require("h5py", "pip install h5py")
    v01 = Path(args.v01_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = {
        "atom_order_doc": ATOM_ORDER_DOC,
        "pocket_order_doc": POCKET_ORDER_DOC,
        "coord_match_tolerance_angstrom": COORD_MATCH_TOL,
    }

    do_pockets = args.mode in ("both", "pockets")
    do_ligands = args.mode in ("both", "ligands")

    # ---------------- pockets ----------------
    if do_pockets:
        src_path = v01 / "pockets" / "pocket_coords.h5"
        dst_path = out / "pockets" / "pocket_coords.h5"
        if not src_path.exists():
            raise SystemExit("missing %s" % src_path)
        pmap = _load_pocket_residue_map(args.nb09_dir)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {"copied": 0, "labelled": 0, "no_map_row": 0}
        reasons = {}
        centroid_errs = []
        t0 = time.time()
        with h5py.File(str(src_path), "r") as src, h5py.File(str(dst_path), "w") as dst:
            dst.attrs["pocket_order_doc"] = POCKET_ORDER_DOC
            dst.attrs["release_version"] = RELEASE_VERSION
            dst.attrs["source_file"] = str(src_path)
            keys = sorted(src["pockets"].keys()) if "pockets" in src else []
            if args.limit:
                keys = keys[: args.limit]
            for n, key in enumerate(keys, 1):
                xyz = np.asarray(src["pockets/" + key][:], dtype=np.float32)
                ds = dst.create_dataset("pockets/" + key, data=xyz, compression="gzip")
                ds.attrs["order"] = "increasing token_index (see /pocket_order_doc)"
                counts["copied"] += 1
                row = pmap.get(key)
                if row is None:
                    counts["no_map_row"] += 1
                    reasons["no_map_row"] = reasons.get("no_map_row", 0) + 1
                    continue
                # WITNESS CHECK. POCKET_ORDER_DOC promises labels only where notebook
                # 09's re-extraction reproduced the shipped coordinates. Notebook 09
                # already filters the parquet to centroid_exact_match rows, but the
                # promise must be ENFORCED here too: an older or hand-edited parquet
                # would otherwise silently attach residue identities to coordinates
                # they do not belong to, which is exactly the failure the ligand side
                # guards against. A missing label is recoverable; a wrong one is not.
                cerr = opt_float(row.get("centroid_max_abs_err"))
                if "centroid_max_abs_err" in row and cerr is None:
                    reasons["centroid_err_unreadable"] = (
                        reasons.get("centroid_err_unreadable", 0) + 1
                    )
                    continue
                if cerr is not None and cerr > COORD_MATCH_TOL:
                    reasons["centroid_not_reproduced"] = (
                        reasons.get("centroid_not_reproduced", 0) + 1
                    )
                    continue
                if cerr is None:
                    reasons["_note_no_centroid_column"] = (
                        reasons.get("_note_no_centroid_column", 0) + 1
                    )
                labels, status = pocket_labels_from_map(
                    row.get("pocket_res_indices"),
                    row.get("pocket_res_provenance"),
                    int(xyz.shape[0]),
                )
                if labels is None:
                    reasons[status] = reasons.get(status, 0) + 1
                    continue
                lds = dst.create_dataset(
                    "pocket_residue_labels/" + key, data=labels, compression="gzip"
                )
                lds.attrs["aligned_to"] = "pockets/" + key
                if cerr is not None:
                    lds.attrs["centroid_max_abs_err"] = cerr
                    centroid_errs.append(cerr)
                for extra in (
                    "esm_input_length",
                    "n_chains_in_esm_input",
                    "chain_order",
                    "n_nonstandard_dropped",
                    "n_nonstandard_dropped_within_8A",
                    "n_crystallographic_gaps",
                ):
                    if extra in row and row[extra] is not None:
                        try:
                            lds.attrs[extra] = row[extra]
                        except Exception:
                            lds.attrs[extra] = str(row[extra])
                counts["labelled"] += 1
                if n % 5000 == 0:
                    log(
                        "  pockets %d/%d (%d labelled, %.0fs)"
                        % (n, len(keys), counts["labelled"], time.time() - t0)
                    )
        rep["pockets"] = {
            "source": str(src_path),
            "output": str(dst_path),
            "n_datasets": counts["copied"],
            "n_labelled": counts["labelled"],
            "pct_labelled": round(100.0 * counts["labelled"] / max(1, counts["copied"]), 3),
            "unlabelled_reasons": reasons,
            "witness_check": (
                "labels written only where notebook 09 reported "
                "centroid_max_abs_err <= %g A against the shipped coordinates" % COORD_MATCH_TOL
            ),
            "n_map_rows": len(pmap),
            "median_centroid_max_abs_err": (
                round(float(np.median(centroid_errs)), 8) if centroid_errs else None
            ),
            "worst_centroid_max_abs_err": (
                round(float(np.max(centroid_errs)), 8) if centroid_errs else None
            ),
        }
        log(
            "[label_coords] pockets: %d copied, %d labelled"
            % (counts["copied"], counts["labelled"])
        )

    # ---------------- ligands ----------------
    if do_ligands:
        src_path = v01 / "ligands" / "ligand_coords.h5"
        dst_path = out / "ligands" / "ligand_coords.h5"
        if not src_path.exists():
            raise SystemExit("missing %s" % src_path)

        entries_path = v01 / "ligands" / "ligand_manifest.jsonl"
        if not entries_path.exists():
            entries_path = v01 / "metadata.jsonl"
        entries = {}
        for row in read_jsonl(entries_path):
            entries[row["pocket_key"]] = row

        cif_index = {}
        raft = Path(args.raft_manifest) if args.raft_manifest else (v01 / "raft_pdb_filtered.jsonl")
        if raft.exists():
            for row in read_jsonl(raft):
                cif_index[row["pdb_id"].lower()] = row.get("path")

        if args.fetch_missing and not args.cif_cache:
            # Otherwise _cif_bytes silently declines to fetch anything and every entry
            # comes back "no_cif", i.e. the user asks for downloads and gets zero labels
            # with no error.
            raise SystemExit("--fetch_missing requires --cif_cache (where downloads land)")

        gemmi = None
        if args.pdb_dir is not None or args.fetch_missing:
            try:
                import gemmi as _gemmi

                gemmi = _gemmi
            except ImportError:
                log("[label_coords] WARNING: gemmi unavailable; ligand labels cannot be built")
        else:
            log(
                "[label_coords] no --pdb_dir and no --fetch_missing: copying ligand coordinates "
                "without labels"
            )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {"copied": 0, "labelled": 0}
        reasons = {}
        max_errs = []
        t0 = time.time()
        with h5py.File(str(src_path), "r") as src, h5py.File(str(dst_path), "w") as dst:
            dst.attrs["atom_order_doc"] = ATOM_ORDER_DOC
            dst.attrs["release_version"] = RELEASE_VERSION
            dst.attrs["source_file"] = str(src_path)
            keys = sorted(src["ligands"].keys()) if "ligands" in src else []
            if args.limit:
                keys = keys[: args.limit]
            for n, key in enumerate(keys, 1):
                xyz = np.asarray(src["ligands/" + key][:], dtype=np.float32)
                ds = dst.create_dataset("ligands/" + key, data=xyz, compression="gzip")
                ds.attrs["order"] = "mmCIF _atom_site order, heavy atoms only (see /atom_order_doc)"
                counts["copied"] += 1
                if gemmi is None:
                    continue
                entry = entries.get(key)
                if entry is None:
                    reasons["no_manifest_row"] = reasons.get("no_manifest_row", 0) + 1
                    continue
                cif_path = _cif_bytes(
                    entry["pdb_id"], args.pdb_dir, cif_index, args.cif_cache, args.fetch_missing
                )
                if cif_path is None:
                    reasons["no_cif"] = reasons.get("no_cif", 0) + 1
                    continue
                try:
                    structure = gemmi.read_structure(str(cif_path))
                except Exception as exc:
                    reasons["parse_fail:" + type(exc).__name__] = (
                        reasons.get("parse_fail:" + type(exc).__name__, 0) + 1
                    )
                    continue
                coords, labels = extract_ligand_atom_labels(
                    structure,
                    ligand_chain=entry["ligand_chain"],
                    ligand_resname=entry["ligand_resname"],
                    ligand_resnum=int(entry["ligand_resnum"]),
                )
                if coords is None:
                    reasons["no_atoms"] = reasons.get("no_atoms", 0) + 1
                    continue
                if coords.shape != xyz.shape:
                    reasons["shape_mismatch"] = reasons.get("shape_mismatch", 0) + 1
                    continue
                err = float(np.abs(coords - xyz).max())
                if err > COORD_MATCH_TOL:
                    reasons["coord_mismatch"] = reasons.get("coord_mismatch", 0) + 1
                    continue
                lds = dst.create_dataset(
                    "ligand_atom_labels/" + key, data=labels, compression="gzip"
                )
                lds.attrs["aligned_to"] = "ligands/" + key
                lds.attrs["coord_max_abs_err"] = err
                lds.attrs["ligand_ccd"] = str(entry["ligand_resname"])
                lds.attrs["pdb_id"] = str(entry["pdb_id"])
                max_errs.append(err)
                counts["labelled"] += 1
                if n % 2000 == 0:
                    log(
                        "  ligands %d/%d (%d labelled, %.0fs)"
                        % (n, len(keys), counts["labelled"], time.time() - t0)
                    )
        rep["ligands"] = {
            "source": str(src_path),
            "output": str(dst_path),
            "n_datasets": counts["copied"],
            "n_labelled": counts["labelled"],
            "pct_labelled": round(100.0 * counts["labelled"] / max(1, counts["copied"]), 3),
            "unlabelled_reasons": reasons,
            "median_coord_max_abs_err": (
                round(float(np.median(max_errs)), 8) if max_errs else None
            ),
            "worst_coord_max_abs_err": (round(float(np.max(max_errs)), 8) if max_errs else None),
        }
        log(
            "[label_coords] ligands: %d copied, %d labelled"
            % (counts["copied"], counts["labelled"])
        )

    write_report(out, "label_coords", rep)
    return rep


# ---------------------------------------------------------------------------
# Stage 2: CCD -> fingerprint table
# ---------------------------------------------------------------------------


def build_ccd_table(rows, emb, offset_field, ccd_field="ligand_resname", smiles_field=None):
    """Collapse a per-row fingerprint matrix to a per-CCD table.

    Returns a dict with:
      codes           list[str]                 CCD codes, first-appearance order
      vectors         (n_ccd, D) float32        one representative vector per code
      row_to_ccd      (n_rows,) int32           index into codes for every input row
      n_rows_per_ccd  (n_ccd,) int32            redundancy count
      smiles          list[str|None]
      conflicts       list[dict]                codes whose rows disagreed (audit says 0)
      is_binary       bool
    """
    codes = []
    index_of = {}
    vectors = []
    smiles = []
    conflicts = []
    row_to_ccd = np.zeros(len(rows), dtype=np.int32)
    counts = []
    for i, row in enumerate(rows):
        ccd = str(row[ccd_field])
        off = int(row[offset_field])
        vec = np.asarray(emb[off], dtype=np.float32)
        if ccd not in index_of:
            index_of[ccd] = len(codes)
            codes.append(ccd)
            vectors.append(vec.copy())
            counts.append(0)
            smiles.append(row.get(smiles_field) if smiles_field else None)
        else:
            ref = vectors[index_of[ccd]]
            if not np.array_equal(ref, vec):
                conflicts.append(
                    {
                        "ccd": ccd,
                        "row": i,
                        "pocket_key": row.get("pocket_key"),
                        "max_abs_diff": float(np.abs(ref - vec).max()),
                    }
                )
            if smiles_field and smiles[index_of[ccd]] in (None, "") and row.get(smiles_field):
                smiles[index_of[ccd]] = row.get(smiles_field)
        j = index_of[ccd]
        row_to_ccd[i] = j
        counts[j] += 1
    mat = (
        np.stack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, int(emb.shape[1])), dtype=np.float32)
    )
    is_binary = bool(mat.size == 0 or np.isin(mat, (0.0, 1.0)).all())
    return {
        "codes": codes,
        "vectors": mat,
        "row_to_ccd": row_to_ccd,
        "n_rows_per_ccd": np.asarray(counts, dtype=np.int32),
        "smiles": smiles,
        "conflicts": conflicts,
        "is_binary": is_binary,
    }


def stage_ccd_table(args):
    v01 = Path(args.v01_dir)
    out = Path(args.out_dir)
    (out / "ligands").mkdir(parents=True, exist_ok=True)

    emb_path = v01 / "ligands" / "ligand_embeddings.npy"
    if not emb_path.exists():
        raise SystemExit("missing %s" % emb_path)
    emb = np.load(str(emb_path), mmap_mode="r")

    # The manifest covers all 72,388 embedded ligands (including the ones the joined
    # index dropped); metadata covers only the 72,199 that survived. Build the table
    # from the manifest when available so no CCD code is lost.
    man_path = v01 / "ligands" / "ligand_manifest.jsonl"
    if man_path.exists():
        source_rows = read_jsonl(man_path)
        offset_field = "ligand_offset"
        source_name = "ligands/ligand_manifest.jsonl"
    else:
        source_rows = read_jsonl(v01 / "metadata.jsonl")
        offset_field = "ligand_emb_offset"
        source_name = "metadata.jsonl"
    table = build_ccd_table(
        source_rows, emb, offset_field, ccd_field="ligand_resname", smiles_field="ligand_smiles"
    )

    # row_to_ccd must be aligned to METADATA rows (that is what the splits index into),
    # which is a subset of the manifest rows, so recompute it against metadata.
    meta_rows = read_jsonl(v01 / "metadata.jsonl")
    code_index = {c: i for i, c in enumerate(table["codes"])}
    meta_row_to_ccd = np.zeros(len(meta_rows), dtype=np.int32)
    missing_meta_ccd = 0
    meta_ccd_codes = set()
    for i, row in enumerate(meta_rows):
        code = str(row["ligand_resname"])
        j = code_index.get(code)
        if j is None:
            missing_meta_ccd += 1
            j = -1
        else:
            meta_ccd_codes.add(code)
        meta_row_to_ccd[i] = j
    if missing_meta_ccd:
        # -1 would silently index the LAST table row, i.e. hand back a wrong
        # fingerprint. The manifest is a superset of metadata by construction, so this
        # is a real inconsistency in the inputs, not a tolerable edge case.
        raise SystemExit(
            "%d metadata rows have a ligand_resname absent from %s; the CCD table "
            "cannot be row-mapped. Refusing to write a table with -1 sentinels."
            % (missing_meta_ccd, source_name)
        )

    vectors = table["vectors"]
    n_bits = int(vectors.shape[1])
    if table["is_binary"]:
        packed = np.packbits(vectors.astype(np.uint8), axis=1)
    else:
        packed = None
    n_on_bits = vectors.sum(axis=1).astype(np.int32)
    nonzero = n_on_bits > 0

    npz_path = out / "ligands" / "ccd_fingerprints.npz"
    payload = {
        "ccd_codes": np.asarray(table["codes"], dtype="U8"),
        "n_on_bits": n_on_bits,
        "n_rows_per_ccd": table["n_rows_per_ccd"],
        "fingerprint_n_bits": np.asarray([n_bits], dtype=np.int32),
    }
    if packed is not None:
        payload["fingerprint_packed"] = packed
    else:
        payload["fingerprint_float32"] = vectors
    np.savez_compressed(str(npz_path), **payload)

    row_map_path = out / "ligands" / "ligand_row_to_ccd.npy"
    np.save(str(row_map_path), meta_row_to_ccd)

    # RDKit re-check: distinguishes "SMILES did not parse" from "parsed but no bits set".
    rdkit_ok = {}
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        for code, smi in zip(table["codes"], table["smiles"]):
            if not smi:
                rdkit_ok[code] = None
                continue
            rdkit_ok[code] = Chem.MolFromSmiles(smi) is not None
    except ImportError:
        log("[ccd_table] rdkit unavailable; skipping SMILES re-parse check")

    table_rows = []
    for j, code in enumerate(table["codes"]):
        table_rows.append(
            {
                "ccd": code,
                "table_row": j,
                "smiles": table["smiles"][j],
                "n_on_bits": int(n_on_bits[j]),
                "n_dataset_rows": int(table["n_rows_per_ccd"][j]),
                "fingerprint_nonzero": bool(nonzero[j]),
                "rdkit_parse_ok": rdkit_ok.get(code),
                "is_common_cofactor": code in COMMON_COFACTOR_CCD,
            }
        )
    write_jsonl(out / "ligands" / "ccd_fingerprints.jsonl", table_rows)
    try:
        import pandas as pd

        pd.DataFrame(table_rows).to_parquet(
            str(out / "ligands" / "ccd_fingerprints.parquet"), index=False
        )
    except Exception as exc:
        log("[ccd_table] parquet not written (%s)" % exc)

    legacy_status = copy_file(emb_path, out / "ligands" / "ligand_embeddings.npy", mode=args.legacy)

    n_rows = len(source_rows)
    n_ccd = len(table["codes"])
    # The audit / paper numbers (26,619 distinct vectors over 72,199 rows, 2.71x) are
    # measured over METADATA rows - the 72,199 released complexes - NOT over the
    # ligand_manifest rows, which also contain the ~189 zero-fingerprint ligands the
    # joined index dropped. Reporting only the manifest-scoped figure would quietly
    # contradict the published number, so BOTH are reported and the changelog quotes
    # the metadata-scoped one.
    n_meta_rows = len(meta_rows)
    n_meta_ccd = len(meta_ccd_codes)
    packed_bytes = int(packed.nbytes) if packed is not None else int(vectors.nbytes)
    rep = {
        "source_rows_from": source_name,
        "n_rows": n_rows,
        "n_distinct_ccd": n_ccd,
        "redundancy_factor": round(float(n_rows) / max(1, n_ccd), 4),
        "scope_note": (
            "n_rows / n_distinct_ccd / redundancy_factor cover every EMBEDDED ligand "
            "(%s). The figures quoted in the paper cover the RELEASED complexes "
            "(metadata.jsonl) and are the *_in_metadata fields below." % source_name
        ),
        "n_rows_in_metadata": n_meta_rows,
        "n_distinct_ccd_in_metadata": n_meta_ccd,
        "redundancy_factor_in_metadata": round(float(n_meta_rows) / max(1, n_meta_ccd), 4),
        "n_ccd_with_multiple_distinct_vectors": len({c["ccd"] for c in table["conflicts"]}),
        "conflicts_sample": table["conflicts"][:20],
        "fingerprint_n_bits": n_bits,
        "fingerprint_is_binary": table["is_binary"],
        # Two different medians. median_on_bits_per_ccd is over DISTINCT codes;
        # median_on_bits_per_metadata_row is over released complexes and is the one the
        # audit reports (median 46 on-bits, i.e. ||ligand|| = sqrt(46) = 6.78). Quoting
        # the per-code median as if it were the per-row one would misstate the paper.
        "median_on_bits_per_ccd": int(np.median(n_on_bits[nonzero])) if nonzero.any() else 0,
        "median_on_bits_per_metadata_row": (
            int(np.median(n_on_bits[meta_row_to_ccd][n_on_bits[meta_row_to_ccd] > 0]))
            if (n_on_bits[meta_row_to_ccd] > 0).any()
            else 0
        ),
        "median_ligand_norm_per_metadata_row": (
            round(
                float(
                    np.median(
                        np.sqrt(
                            n_on_bits[meta_row_to_ccd][n_on_bits[meta_row_to_ccd] > 0].astype(
                                np.float64
                            )
                        )
                    )
                ),
                4,
            )
            if (n_on_bits[meta_row_to_ccd] > 0).any()
            else None
        ),
        "n_ccd_zero_fingerprint": int((~nonzero).sum()),
        "n_ccd_rdkit_parse_failed": sum(1 for v in rdkit_ok.values() if v is False),
        "bytes_per_row_matrix": int(emb.shape[0]) * int(emb.shape[1]) * 4,
        "bytes_ccd_table_packed": packed_bytes,
        "compression_vs_row_matrix": round(
            (int(emb.shape[0]) * int(emb.shape[1]) * 4) / max(1, packed_bytes), 2
        ),
        "metadata_rows": len(meta_rows),
        "metadata_rows_without_ccd_entry": missing_meta_ccd,
        "legacy_matrix_status": legacy_status,
        "outputs": {
            "npz": str(npz_path),
            "jsonl": str(out / "ligands" / "ccd_fingerprints.jsonl"),
            "row_map": str(row_map_path),
        },
        "unpack_recipe": (
            "d = np.load('ligands/ccd_fingerprints.npz'); "
            "fp = np.unpackbits(d['fingerprint_packed'], axis=1)[:, :int(d['fingerprint_n_bits'][0])]"
            "; row_i_vector = fp[np.load('ligands/ligand_row_to_ccd.npy')[i]]"
        ),
    }
    write_report(out, "ccd_table", rep)
    log(
        "[ccd_table] embedded: %d rows -> %d CCD codes (%.2fx); released (metadata): "
        "%d rows -> %d CCD codes (%.2fx); %d conflicts"
        % (
            n_rows,
            n_ccd,
            rep["redundancy_factor"],
            n_meta_rows,
            n_meta_ccd,
            rep["redundancy_factor_in_metadata"],
            len(table["conflicts"]),
        )
    )
    return rep


# ---------------------------------------------------------------------------
# Stage 3: per-modality FAISS indices
# ---------------------------------------------------------------------------


def _build_hnsw(faiss, dim, M, ef_construction, ef_search):
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    return index


def _add_offsets_to_index(index, src, offsets, chunk):
    n_zero = 0
    for start in range(0, len(offsets), chunk):
        sel = offsets[start : start + chunk]
        block = np.asarray(src[sel], dtype=np.float32)
        normed, z = l2_normalise_rows(block)
        n_zero += z
        index.add(np.ascontiguousarray(normed))
    return n_zero


def stage_indices(args):
    faiss = require("faiss", "pip install faiss-cpu")
    v01 = Path(args.v01_dir)
    out = Path(args.out_dir)
    (out / "indices").mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.metadata) if args.metadata else None
    if meta_path is None:
        meta_path = out / "metadata.jsonl"
        if not meta_path.exists():
            meta_path = v01 / "metadata.jsonl"
    meta_rows = read_jsonl(meta_path)
    n = len(meta_rows)
    log("[indices] %d metadata rows from %s" % (n, meta_path))

    p_off = np.asarray([int(r["pocket_emb_offset"]) for r in meta_rows], dtype=np.int64)
    l_off = np.asarray([int(r["ligand_emb_offset"]) for r in meta_rows], dtype=np.int64)
    np.save(str(out / "indices" / "pocket_row_offsets.npy"), p_off)
    np.save(str(out / "indices" / "ligand_row_offsets.npy"), l_off)
    with open_text_write(out / "indices" / "index_row_pocket_keys.txt") as fh:
        for r in meta_rows:
            fh.write(str(r["pocket_key"]) + "\n")

    pocket_emb = np.load(str(v01 / "pockets" / "pocket_embeddings.npy"), mmap_mode="r")
    ligand_emb = np.load(str(v01 / "ligands" / "ligand_embeddings.npy"), mmap_mode="r")

    p_dim = int(pocket_emb.shape[1])
    l_dim = int(ligand_emb.shape[1])

    log("[indices] building pocket.faiss (%d x %d)" % (n, p_dim))
    p_index = _build_hnsw(faiss, p_dim, args.M, args.efConstruction, args.efSearch)
    p_zero = _add_offsets_to_index(p_index, pocket_emb, p_off, args.chunk)
    faiss.write_index(p_index, str(out / "indices" / "pocket.faiss"))

    log("[indices] building ligand.faiss (%d x %d)" % (n, l_dim))
    l_index = _build_hnsw(faiss, l_dim, args.M, args.efConstruction, args.efSearch)
    l_zero = _add_offsets_to_index(l_index, ligand_emb, l_off, args.chunk)
    faiss.write_index(l_index, str(out / "indices" / "ligand.faiss"))

    # Concatenated index: retained, not primary.
    joint_status = None
    joint_ntotal = None
    joint_path = out / "retrieval.faiss"
    if args.rebuild_joint:
        joined_path = v01 / "joined_embeddings.npy"
        if not joined_path.exists():
            raise SystemExit("--rebuild_joint needs %s" % joined_path)
        joined = np.load(str(joined_path), mmap_mode="r")
        if int(joined.shape[0]) != n:
            raise SystemExit(
                "joined_embeddings has %d rows but metadata has %d" % (joined.shape[0], n)
            )
        j_index = _build_hnsw(
            faiss, int(joined.shape[1]), args.M, args.efConstruction, args.efSearch
        )
        _add_offsets_to_index(j_index, joined, np.arange(n, dtype=np.int64), args.chunk)
        faiss.write_index(j_index, str(joint_path))
        joint_status = "rebuilt"
        joint_ntotal = int(j_index.ntotal)
    else:
        joint_status = copy_file(v01 / "retrieval.faiss", joint_path, mode="copy")
        if joint_path.exists():
            try:
                jj = faiss.read_index(str(joint_path))
                joint_ntotal = int(jj.ntotal)
            except Exception as exc:
                log("[indices] could not read carried-over joint index: %s" % exc)

    if p_zero or l_zero:
        # A zero vector scores 0.0 against everything under inner product: it is a dead
        # row that will never be retrieved and will never retrieve anything sensible.
        # metadata.jsonl should contain no such row (build_index.py dropped them), so
        # this is a loud failure signal, not a footnote.
        log(
            "[indices] WARNING: %d pocket and %d ligand rows have zero norm and were "
            "added as dead rows. metadata.jsonl is expected to contain NONE - check "
            "ligands/unparseable_smiles.jsonl and the metadata source." % (p_zero, l_zero)
        )

    guidance = dict(MODALITY_GUIDANCE)
    guidance["measured_retrieval_performance"] = RETRIEVAL_BENCHMARK
    guidance["row_alignment"] = (
        "Row i of pocket.faiss, row i of ligand.faiss, row i of retrieval.faiss and line i of "
        "metadata.jsonl are the SAME complex. indices/index_row_pocket_keys.txt gives the "
        "pocket_key of every index row so the alignment can be checked without assuming it."
    )
    guidance["index_params"] = {
        "type": "IndexHNSWFlat",
        "metric": "METRIC_INNER_PRODUCT on L2-normalised vectors (= cosine)",
        "M": args.M,
        "efConstruction": args.efConstruction,
        "efSearch_default": args.efSearch,
        "pocket_dim": p_dim,
        "ligand_dim": l_dim,
    }
    write_json(out / "indices" / "modality_guidance.json", guidance)

    rep = {
        "n_rows": n,
        "metadata_source": str(meta_path),
        "pocket_index": {
            "path": str(out / "indices" / "pocket.faiss"),
            "dim": p_dim,
            "ntotal": int(p_index.ntotal),
            "n_zero_norm_rows": p_zero,
        },
        "ligand_index": {
            "path": str(out / "indices" / "ligand.faiss"),
            "dim": l_dim,
            "ntotal": int(l_index.ntotal),
            "n_zero_norm_rows": l_zero,
        },
        "joint_index": {
            "path": str(joint_path),
            "status": joint_status,
            "ntotal": joint_ntotal,
            "row_aligned_with_metadata": (joint_ntotal == n) if joint_ntotal is not None else None,
        },
        "params": guidance["index_params"],
        "n_dead_zero_norm_rows": int(p_zero) + int(l_zero),
        "measured_retrieval_performance": RETRIEVAL_BENCHMARK,
    }
    write_report(out, "indices", rep)
    log(
        "[indices] pocket ntotal=%d, ligand ntotal=%d, joint ntotal=%s"
        % (p_index.ntotal, l_index.ntotal, joint_ntotal)
    )
    return rep


# ---------------------------------------------------------------------------
# Stage 4: splits
# ---------------------------------------------------------------------------


def read_fasta_gz(path):
    """pocket_key -> sequence, from notebook 09's esm_input_sequences.fasta.gz."""
    out = {}
    key = None
    chunks = []
    with gzip.open(str(path), "rt") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if key is not None:
                    out[key] = "".join(chunks)
                key = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if key is not None:
        out[key] = "".join(chunks)
    return out


def exact_sequence_clusters(seqs):
    """Fallback clustering: identical sequence strings share a cluster.

    Strictly weaker than MMseqs2 at 30% identity - it removes only exact-sequence
    leakage, not homolog leakage. Recorded as such in target_disjoint_info.json so a
    split built this way can never be mistaken for a homology-disjoint benchmark.
    """
    cluster_of_index = {}
    seen = {}
    next_cid = 0
    for i, s in enumerate(seqs):
        if not s:
            cluster_of_index[i] = next_cid
            next_cid += 1
            continue
        if s not in seen:
            seen[s] = next_cid
            next_cid += 1
        cluster_of_index[i] = seen[s]
    return cluster_of_index


def ccd_overlap_report(splits, ccds):
    """Ligand-CCD leakage across split boundaries.

    A sequence-disjoint split still hands the model bit-identical ligand vectors: the
    Morgan fingerprint is a pure function of the CCD code, so any held-out entry whose
    CCD also occurs in train has an exact ligand-embedding twin in train.
    """
    train = splits.get("train", [])
    train_ccds = {ccds[i] for i in train if ccds[i]}
    out = {
        "n_train": len(train),
        "n_train_distinct_ccd": len(train_ccds),
    }
    for name in ("val", "test"):
        held = splits.get(name, [])
        if not held:
            continue
        held_ccds = [ccds[i] for i in held]
        seen_in_train = [c for c in held_ccds if c and c in train_ccds]
        distinct_held = {c for c in held_ccds if c}
        out[name] = {
            "n": len(held),
            "n_distinct_ccd": len(distinct_held),
            "n_entries_whose_ccd_occurs_in_train": len(seen_in_train),
            "pct_entries_whose_ccd_occurs_in_train": round(
                100.0 * len(seen_in_train) / max(1, len(held_ccds)), 3
            ),
            "n_distinct_ccd_shared_with_train": len(distinct_held & train_ccds),
            "n_distinct_ccd_unseen_in_train": len(distinct_held - train_ccds),
        }
    out["interpretation"] = (
        "pct_entries_whose_ccd_occurs_in_train is the fraction of held-out entries whose "
        "ligand fingerprint is bit-identical to at least one training entry's. A "
        "target-disjoint split does NOT drive this to zero; report it alongside any "
        "generalisation claim."
    )
    return out


def stage_splits(args):
    v01 = Path(args.v01_dir)
    out = Path(args.out_dir)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    addsplits = _load_module_from_path("_v02_add_splits", SCRIPTS_DIR / "add_splits_to_hf.py")
    tdisjoint = _load_module_from_path(
        "_v02_target_disjoint", SCRIPTS_DIR / "build_target_disjoint_split.py"
    )

    # Which directory do we scan for usable HDF5 entries?
    if args.usable_from == "out":
        scan_dir = out
    elif args.usable_from == "v01":
        scan_dir = v01
    else:
        has_out = (
            (out / "metadata.jsonl").exists()
            and (out / "ligands" / "ligand_coords.h5").exists()
            and (out / "pockets" / "pocket_coords.h5").exists()
        )
        scan_dir = out if has_out else v01
    log("[splits] scanning usable entries in %s" % scan_dir)

    meta_rows = read_jsonl(scan_dir / "metadata.jsonl")
    n_meta = len(meta_rows)
    usable = addsplits.scan_valid(scan_dir)  # same rule that produced the v0.1 splits
    usable_set = set(usable)

    # ---- random split, recomputed so it exactly partitions the usable set ----
    train_ix, val_ix = addsplits.split_indices(usable, args.seed, args.val_frac)
    # Explicit raises, not bare asserts: `python -O` strips asserts, and "the split
    # exactly partitions the usable set" is the entire point of this stage.
    if set(train_ix) & set(val_ix):
        raise SystemExit(
            "random split overlaps: %d shared indices" % len(set(train_ix) & set(val_ix))
        )
    if set(train_ix) | set(val_ix) != usable_set:
        raise SystemExit(
            "random split does not partition the usable set (%d usable, %d covered)"
            % (len(usable_set), len(set(train_ix) | set(val_ix)))
        )
    np.save(str(out / "splits" / "train_indices.npy"), np.asarray(train_ix, dtype=np.int64))
    np.save(str(out / "splits" / "val_indices.npy"), np.asarray(val_ix, dtype=np.int64))

    # ---- diff against the shipped v0.1 split, so the fix is auditable ----
    v01_diff = None
    v01_splits = Path(args.v01_splits_dir) if args.v01_splits_dir else (v01 / "splits")
    v01_train_p = v01_splits / "train_indices.npy"
    v01_val_p = v01_splits / "val_indices.npy"
    if v01_train_p.exists() and v01_val_p.exists():
        old_train = np.load(str(v01_train_p)).tolist()
        old_val = np.load(str(v01_val_p)).tolist()
        old_all = set(old_train) | set(old_val)
        v01_diff = {
            "v01_n_train": len(old_train),
            "v01_n_val": len(old_val),
            "v01_entries_in_a_split_but_not_usable": len(old_all - usable_set),
            "v01_usable_entries_in_no_split": len(usable_set - old_all),
            "v01_indices_out_of_metadata_range": len([i for i in old_all if i >= n_meta or i < 0]),
            "v01_train_val_overlap": len(set(old_train) & set(old_val)),
            "v02_n_train": len(train_ix),
            "v02_n_val": len(val_ix),
            "v02_entries_in_a_split_but_not_usable": 0,
            "v02_usable_entries_in_no_split": 0,
        }

    splits_info = {
        "method": "deterministic random permutation via np.random.RandomState(seed) over the "
        "usable set (identical rule to scripts/add_splits_to_hf.py, recomputed against the "
        "v0.2 metadata and coordinate files)",
        "seed": args.seed,
        "val_frac": args.val_frac,
        "n_metadata_rows": n_meta,
        "n_usable": len(usable),
        "n_train": len(train_ix),
        "n_val": len(val_ix),
        "usable_rule": "pocket_key present in both HDF5 files with shape (n>=3, 3)",
        "usable_scanned_in": str(scan_dir),
        "partition_asserted": True,
        "v01_comparison": v01_diff,
        "WARNING": _split_warning(),
    }
    write_json(out / "splits" / "splits_info.json", splits_info)
    log(
        "[splits] random: %d train / %d val (usable %d)" % (len(train_ix), len(val_ix), len(usable))
    )

    # ---- target-disjoint split on the REAL ESM-2 input sequences ----
    fasta = (
        Path(args.esm_fasta)
        if args.esm_fasta
        else (Path(args.nb09_dir) / "esm_input_sequences.fasta.gz" if args.nb09_dir else None)
    )
    td_info = None
    td_splits = None
    if fasta is None or not Path(fasta).exists():
        log(
            "[splits] esm_input_sequences.fasta.gz not found; skipping the target-disjoint split "
            "(pass --nb09_dir or --esm_fasta)"
        )
    else:
        seq_by_key = read_fasta_gz(fasta)
        seqs = [seq_by_key.get(str(r["pocket_key"]), "") for r in meta_rows]
        n_with_seq = sum(1 for s in seqs if s)
        log(
            "[splits] %d/%d rows have a real ESM input sequence (backend=%s)"
            % (n_with_seq, n_meta, args.cluster_backend)
        )
        if args.cluster_backend == "mmseqs":
            cluster_of_index = tdisjoint.run_mmseqs_cluster(
                seqs, args.min_seq_id, args.coverage, args.mmseqs
            )
            backend_desc = "MMseqs2 easy-cluster --min-seq-id %s -c %s" % (
                args.min_seq_id,
                args.coverage,
            )
        else:
            cluster_of_index = exact_sequence_clusters(seqs)
            backend_desc = (
                "EXACT SEQUENCE IDENTITY fallback (no MMseqs2). Removes exact-sequence leakage "
                "ONLY - homologs still cross the boundary. Not a homology-disjoint benchmark."
            )
        usable_clusters = {i: c for i, c in cluster_of_index.items() if i in usable_set}
        td_train, td_val, td_test = tdisjoint.assign_clusters_to_splits(
            usable_clusters, args.val_frac, args.test_frac, args.seed
        )
        if set(td_train) | set(td_val) | set(td_test) != usable_set:
            raise SystemExit(
                "target-disjoint split does not partition the usable set "
                "(%d usable, %d covered)"
                % (len(usable_set), len(set(td_train) | set(td_val) | set(td_test)))
            )
        np.save(
            str(out / "splits" / "train_target_disjoint.npy"), np.asarray(td_train, dtype=np.int64)
        )
        np.save(str(out / "splits" / "val_target_disjoint.npy"), np.asarray(td_val, dtype=np.int64))
        np.save(
            str(out / "splits" / "test_target_disjoint.npy"), np.asarray(td_test, dtype=np.int64)
        )
        td_splits = {"train": td_train, "val": td_val, "test": td_test}
        td_info = {
            "method": "whole-cluster assignment (scripts/build_target_disjoint_split.py)",
            "clustering_backend": args.cluster_backend,
            "clustering_description": backend_desc,
            "sequence_source": str(fasta),
            "sequence_source_note": (
                "Clustered on the ACTUAL ESM-2 input sequence recovered by notebook 09 - the "
                "chimeric concatenation the pocket embedding was computed from - NOT the "
                "post-hoc protein_seq annotation, which is the longest chain only and is missing "
                "for 457 entries."
            ),
            "min_seq_id": args.min_seq_id if args.cluster_backend == "mmseqs" else None,
            "coverage": args.coverage if args.cluster_backend == "mmseqs" else None,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "n_rows_with_sequence": n_with_seq,
            "n_rows_without_sequence_singleton_clusters": n_meta - n_with_seq,
            "n_clusters_over_usable": len(set(usable_clusters.values())),
            "n_usable": len(usable),
            "n_train": len(td_train),
            "n_val": len(td_val),
            "n_test": len(td_test),
            "disjoint_asserted": True,
        }

    # ---- ligand-CCD overlap across both boundaries ----
    ccds = [str(r.get("ligand_ccd") or r.get("ligand_resname") or "") for r in meta_rows]
    leakage = {
        "random_split": ccd_overlap_report({"train": train_ix, "val": val_ix}, ccds),
    }
    if td_splits is not None:
        leakage["target_disjoint_split"] = ccd_overlap_report(td_splits, ccds)
    write_json(out / "splits" / "split_leakage_report.json", leakage)
    if td_info is not None:
        td_info["ligand_ccd_leakage"] = leakage.get("target_disjoint_split")
        write_json(out / "splits" / "target_disjoint_info.json", td_info)
        log(
            "[splits] target-disjoint: %d train / %d val / %d test"
            % (td_info["n_train"], td_info["n_val"], td_info["n_test"])
        )

    rep = {"random": splits_info, "target_disjoint": td_info, "ligand_ccd_leakage": leakage}
    write_report(out, "splits", rep)
    return rep


# ---------------------------------------------------------------------------
# Stage 5: metadata enrichment
# ---------------------------------------------------------------------------


def load_sifts_map(path, column):
    """pdb_id(lower) -> (first value, sorted unique values) from a SIFTS flatfile."""
    pd = require("pandas", "pip install pandas")
    df = pd.read_csv(str(path), compression="gzip", skiprows=1)
    df["PDB"] = df["PDB"].astype(str).str.lower()
    # Identical rule to notebook 08 cell 13's load_sifts() so the stratification the
    # audit measured is reproducible from the shipped columns.
    first = df.groupby("PDB")[column].first().to_dict()

    def _uniq(series):
        vals = set()
        for x in series:
            s = clean_str(x)
            if s:
                vals.add(s)
        return sorted(vals)

    allv = df.groupby("PDB")[column].apply(_uniq).to_dict()
    return first, allv


def download_sifts(dest_dir):
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/"
    files = ["pdb_chain_uniprot.csv.gz", "pdb_chain_pfam.csv.gz", "pdb_chain_enzyme.csv.gz"]
    for name in files:
        dst = dest_dir / name
        if dst.exists():
            continue
        log("[metadata] downloading %s" % name)
        urllib.request.urlretrieve(base + name, str(dst))
    return dest_dir


def fetch_exptl_methods(pdb_ids, cache_path, workers=16):
    """pdb_id(lower) -> experimental method string, via the RCSB Data API, cached."""
    cache = {}
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            with open(str(cache_path), "r", encoding="utf-8") as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}
    todo = [p for p in pdb_ids if p not in cache]
    if not todo:
        return cache
    log("[metadata] fetching experimental method for %d PDB IDs" % len(todo))
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def one(pid):
        url = "https://data.rcsb.org/rest/v1/core/entry/%s" % pid.upper()
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            methods = [e.get("method") for e in data.get("exptl", []) if e.get("method")]
            if not methods:
                info = data.get("rcsb_entry_info", {}) or {}
                methods = info.get("experimental_method") and [info["experimental_method"]] or []
            return pid, ";".join(methods) if methods else None
        except Exception:
            return pid, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pid, method in ex.map(one, todo):
            cache[pid] = method
            done += 1
            if done % 2000 == 0:
                log("  exptl %d/%d" % (done, len(todo)))
                if cache_path:
                    write_json(cache_path, cache)
    if cache_path:
        write_json(cache_path, cache)
    return cache


def enrich_metadata_rows(
    meta_rows,
    pocket_map,
    raft_by_pdb,
    exptl,
    uniprot,
    pfam,
    ec,
    row_to_ccd,
):
    """Return (enriched rows, stats). Row ORDER and row COUNT are preserved exactly."""
    out_rows = []
    stats = {
        "n_rows": len(meta_rows),
        "n_with_exptl_method": 0,
        "n_with_n_qualifying_ligands": 0,
        "n_common_cofactor": 0,
        "n_common_cofactor_audit_set": 0,
        "n_with_esm_input_length": 0,
        "n_with_uniprot": 0,
        "n_with_pfam": 0,
        "n_with_ec": 0,
    }
    for i, src in enumerate(meta_rows):
        row = dict(src)  # preserves v0.1 key order; new keys append
        pid = str(row["pdb_id"]).lower()
        ccd = str(row["ligand_resname"])

        row["ligand_ccd"] = ccd
        if row_to_ccd is not None and i < len(row_to_ccd):
            row["ligand_ccd_table_row"] = int(row_to_ccd[i])

        method = exptl.get(pid) if exptl else None
        row["exptl_method"] = method
        if method:
            stats["n_with_exptl_method"] += 1

        raft = raft_by_pdb.get(pid) if raft_by_pdb else None
        nq = len(raft.get("ligands", [])) if raft else None
        row["n_qualifying_ligands"] = nq
        if nq is not None:
            stats["n_with_n_qualifying_ligands"] += 1

        is_cof = ccd in COMMON_COFACTOR_CCD
        row["is_common_cofactor"] = bool(is_cof)
        if is_cof:
            stats["n_common_cofactor"] += 1
        # Audit-comparable flag: the exact 36-code set notebook 08 used to measure the
        # 25.8% quoted in the paper. Shipped alongside the broader advisory flag so the
        # published number stays reproducible from the release itself.
        is_cof_audit = ccd in AUDIT_COMMON_ARTIFACTS_CCD
        row["is_common_cofactor_audit_set"] = bool(is_cof_audit)
        if is_cof_audit:
            stats["n_common_cofactor_audit_set"] += 1

        pm = pocket_map.get(str(row["pocket_key"])) if pocket_map else None
        if pm is not None:
            row["esm_input_length"] = opt_int(pm.get("esm_input_length"))
            row["n_chains_in_esm_input"] = opt_int(pm.get("n_chains_in_esm_input"))
            row["chain_order"] = clean_str(pm.get("chain_order"))
            row["n_nonstandard_dropped_within_8A"] = opt_int(
                pm.get("n_nonstandard_dropped_within_8A")
            )
            if row["esm_input_length"] is not None:
                stats["n_with_esm_input_length"] += 1
        else:
            row["esm_input_length"] = None
            row["n_chains_in_esm_input"] = None
            row["chain_order"] = None
            row["n_nonstandard_dropped_within_8A"] = None

        # clean_str, not `str(x) if x else None`: SIFTS groupby().first() yields NaN for
        # unannotated entries and float('nan') is truthy, which would write the literal
        # string "nan" into the accession columns of every unannotated row.
        row["uniprot_acc"] = clean_str(uniprot[0].get(pid)) if uniprot else None
        row["uniprot_acc_all"] = (uniprot[1].get(pid) if uniprot else None) or None
        if row["uniprot_acc"]:
            stats["n_with_uniprot"] += 1

        row["pfam_acc"] = clean_str(pfam[0].get(pid)) if pfam else None
        row["pfam_acc_all"] = (pfam[1].get(pid) if pfam else None) or None
        if row["pfam_acc"]:
            stats["n_with_pfam"] += 1

        row["ec_number"] = clean_str(ec[0].get(pid)) if ec else None
        if row["ec_number"]:
            stats["n_with_ec"] += 1

        out_rows.append(row)
    return out_rows, stats


def stage_metadata(args):
    v01 = Path(args.v01_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_rows = read_jsonl(v01 / "metadata.jsonl")
    n = len(meta_rows)
    log("[metadata] %d v0.1 rows" % n)

    pocket_map = None
    if args.nb09_dir:
        try:
            pocket_map = _load_pocket_residue_map(args.nb09_dir)
        except SystemExit as exc:
            log("[metadata] pocket map unavailable: %s" % exc)

    raft_by_pdb = {}
    raft = Path(args.raft_manifest) if args.raft_manifest else (v01 / "raft_pdb_filtered.jsonl")
    if raft.exists():
        for r in read_jsonl(raft):
            raft_by_pdb[str(r["pdb_id"]).lower()] = r
        log("[metadata] raft manifest: %d entries" % len(raft_by_pdb))
    else:
        log("[metadata] raft manifest not found; n_qualifying_ligands will be null")

    exptl = {}
    if args.exptl_source == "api":
        pdb_ids = sorted({str(r["pdb_id"]).lower() for r in meta_rows})
        cache = args.exptl_cache or str(out / "provenance" / "exptl_cache.json")
        exptl = fetch_exptl_methods(pdb_ids, cache, workers=args.workers)
    elif args.exptl_source == "cache" and args.exptl_cache and Path(args.exptl_cache).exists():
        with open(str(args.exptl_cache), "r", encoding="utf-8") as fh:
            exptl = json.load(fh)

    uniprot = pfam = ec = None
    sifts_dir = Path(args.sifts_dir) if args.sifts_dir else None
    if args.download_sifts:
        sifts_dir = download_sifts(sifts_dir or (out / "provenance" / "sifts"))
    if sifts_dir is not None:
        try:
            uniprot = load_sifts_map(sifts_dir / "pdb_chain_uniprot.csv.gz", "SP_PRIMARY")
            pfam = load_sifts_map(sifts_dir / "pdb_chain_pfam.csv.gz", "PFAM_ID")
            ec = load_sifts_map(sifts_dir / "pdb_chain_enzyme.csv.gz", "EC_NUMBER")
            log("[metadata] SIFTS loaded from %s" % sifts_dir)
        except Exception as exc:
            log("[metadata] SIFTS unavailable (%s); uniprot/pfam/ec will be null" % exc)
            uniprot = pfam = ec = None

    row_to_ccd = None
    rmap = out / "ligands" / "ligand_row_to_ccd.npy"
    if rmap.exists():
        row_to_ccd = np.load(str(rmap))
        if len(row_to_ccd) != n:
            log("[metadata] WARNING: ligand_row_to_ccd length %d != %d" % (len(row_to_ccd), n))
            row_to_ccd = None

    enriched, stats = enrich_metadata_rows(
        meta_rows, pocket_map, raft_by_pdb, exptl, uniprot, pfam, ec, row_to_ccd
    )

    excluded_het, _het_err = load_excluded_het()
    n_cofactor_blacklisted = len(COMMON_COFACTOR_CCD & set(excluded_het)) if excluded_het else None

    # Row-order contract: identical count, identical pocket_key sequence. A plain
    # `assert` would vanish under `python -O`, and this is the invariant every split
    # index array depends on, so it is enforced unconditionally.
    if len(enriched) != n:
        raise SystemExit("metadata row count changed: %d -> %d" % (n, len(enriched)))
    for i in range(n):
        if enriched[i]["pocket_key"] != meta_rows[i]["pocket_key"]:
            raise SystemExit(
                "metadata row ORDER changed at row %d (%s -> %s); the split index arrays "
                "would silently point at the wrong complexes"
                % (i, meta_rows[i]["pocket_key"], enriched[i]["pocket_key"])
            )

    write_jsonl(out / "metadata.jsonl", enriched)
    parquet_status = "not_written"
    try:
        import pandas as pd

        pd.DataFrame(enriched).to_parquet(str(out / "metadata.parquet"), index=False)
        parquet_status = "written"
    except Exception as exc:
        parquet_status = "failed:%s" % exc
        log("[metadata] parquet not written (%s)" % exc)

    rep = {
        "n_rows": n,
        "row_order_preserved": True,
        "added_fields": [
            "exptl_method",
            "n_qualifying_ligands",
            "is_common_cofactor",
            "is_common_cofactor_audit_set",
            "esm_input_length",
            "n_chains_in_esm_input",
            "chain_order",
            "n_nonstandard_dropped_within_8A",
            "uniprot_acc",
            "uniprot_acc_all",
            "pfam_acc",
            "pfam_acc_all",
            "ec_number",
            "ligand_ccd",
            "ligand_ccd_table_row",
        ],
        "coverage": stats,
        "pct_common_cofactor": round(100.0 * stats["n_common_cofactor"] / max(1, n), 3),
        "pct_common_cofactor_audit_set": round(
            100.0 * stats["n_common_cofactor_audit_set"] / max(1, n), 3
        ),
        "audit_reference_pct_selected_cofactor_or_buffer": 25.8,
        "audit_set_drift_pp": round(
            100.0 * stats["n_common_cofactor_audit_set"] / max(1, n) - 25.8, 3
        ),
        "common_cofactor_list_size": len(COMMON_COFACTOR_CCD),
        "common_cofactor_audit_list_size": len(AUDIT_COMMON_ARTIFACTS_CCD),
        "n_cofactor_codes_also_in_excluded_het": n_cofactor_blacklisted,
        "common_cofactor_note": (
            "TWO flags are shipped. is_common_cofactor_audit_set uses the exact 36-code set "
            "notebook 08 used to measure the 25.8%% quoted in the paper - "
            "pct_common_cofactor_audit_set MUST reproduce it (see audit_set_drift_pp). "
            "is_common_cofactor uses the broader %d-code COMMON_COFACTOR_CCD list defined in "
            "scripts/build_v02_release.py (new in v0.2, advisory, a strict superset), so it "
            "flags MORE entries; do NOT quote pct_common_cofactor as the audit's 25.8%%. Both "
            "are independent of EXCLUDED_HET, the 35-code build-time blacklist in "
            "rcsb_filter.py; %s of the cofactor codes are also blacklisted and are therefore "
            "inert." % (len(COMMON_COFACTOR_CCD), n_cofactor_blacklisted)
        ),
        "exptl_source": args.exptl_source,
        "sifts_dir": str(sifts_dir) if sifts_dir else None,
        "parquet": parquet_status,
    }
    write_report(out, "metadata", rep)
    log(
        "[metadata] wrote %d rows; exptl %d, uniprot %d, pfam %d, ec %d, esm %d, "
        "cofactor %d (broad) / %d (audit set, %.2f%% vs audit 25.8%%)"
        % (
            n,
            stats["n_with_exptl_method"],
            stats["n_with_uniprot"],
            stats["n_with_pfam"],
            stats["n_with_ec"],
            stats["n_with_esm_input_length"],
            stats["n_common_cofactor"],
            stats["n_common_cofactor_audit_set"],
            rep["pct_common_cofactor_audit_set"],
        )
    )
    if abs(rep["audit_set_drift_pp"]) > 1.0:
        log(
            "[metadata] WARNING: audit-set cofactor share is %.2f%%, the audit measured "
            "25.8%% (drift %+.2f pp). Reconcile before quoting either number in the paper."
            % (rep["pct_common_cofactor_audit_set"], rep["audit_set_drift_pp"])
        )
    return rep


# ---------------------------------------------------------------------------
# Stage 6: flag unparseable SMILES
# ---------------------------------------------------------------------------


def stage_flag_unparseable(args):
    v01 = Path(args.v01_dir)
    out = Path(args.out_dir)
    (out / "ligands").mkdir(parents=True, exist_ok=True)

    emb = np.load(str(v01 / "ligands" / "ligand_embeddings.npy"), mmap_mode="r")
    # chunked so the 593 MB matrix is never materialised in RAM
    norms = np.zeros(int(emb.shape[0]), dtype=np.float64)
    for start in range(0, int(emb.shape[0]), 8192):
        block = np.asarray(emb[start : start + 8192], dtype=np.float32)
        norms[start : start + block.shape[0]] = np.linalg.norm(block, axis=1)
    zero_rows = np.nonzero(norms < 1e-6)[0]

    man_path = v01 / "ligands" / "ligand_manifest.jsonl"
    by_offset = {}
    if man_path.exists():
        for r in read_jsonl(man_path):
            by_offset[int(r["ligand_offset"])] = r
    meta_keys = {str(r["pocket_key"]) for r in read_jsonl(v01 / "metadata.jsonl")}

    chem = None
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        chem = Chem
    except ImportError:
        log("[flag_unparseable] rdkit unavailable; reason will be 'unverified'")

    records = []
    reason_counts = {}
    for off in zero_rows.tolist():
        r = by_offset.get(off, {})
        smi = r.get("ligand_smiles")
        if chem is None:
            reason = "unverified_no_rdkit"
        elif not smi:
            reason = "no_smiles_from_ccd_api"
        elif chem.MolFromSmiles(smi) is None:
            reason = "rdkit_smiles_parse_failure"
        else:
            reason = "parsed_but_empty_fingerprint"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        records.append(
            {
                "ligand_emb_offset": int(off),
                "pocket_key": r.get("pocket_key"),
                "pdb_id": r.get("pdb_id"),
                "ligand_ccd": r.get("ligand_resname"),
                "ligand_smiles": smi,
                "ligand_heavy_atoms": r.get("ligand_heavy_atoms"),
                "reason": reason,
                "in_v01_metadata": (r.get("pocket_key") in meta_keys)
                if r.get("pocket_key")
                else None,
                "in_v02_metadata": False,
                "v01_disposition": "silently excluded from metadata.jsonl and the joined index",
                "v02_disposition": "listed here explicitly; still excluded from the row-aligned "
                "indices because a zero vector has no direction",
            }
        )

    write_jsonl(out / "ligands" / "unparseable_smiles.jsonl", records)
    distinct_ccd = sorted({r["ligand_ccd"] for r in records if r.get("ligand_ccd")})
    rep = {
        "n_ligand_embedding_rows": int(emb.shape[0]),
        "n_zero_vector_rows": int(len(zero_rows)),
        "reason_counts": reason_counts,
        "n_distinct_ccd_affected": len(distinct_ccd),
        "distinct_ccd_affected": distinct_ccd[:200],
        "n_still_present_in_v01_metadata": sum(1 for r in records if r.get("in_v01_metadata")),
        "output": str(out / "ligands" / "unparseable_smiles.jsonl"),
        "correction": (
            "The v0.1 documentation attributed these to transition-metal complexes, polyborane "
            "clusters and polyoxometalates. Re-checking every SMILES with RDKit shows they are "
            "SMILES PARSE FAILURES; the chemistry explanation is wrong and is corrected in v0.2."
        ),
    }
    write_report(out, "flag_unparseable", rep)
    log(
        "[flag_unparseable] %d zero-vector rows; reasons %s"
        % (len(zero_rows), json.dumps(reason_counts))
    )
    return rep


# ---------------------------------------------------------------------------
# Stage 7: carry over the v0.1 files that v0.2 does not change
# ---------------------------------------------------------------------------

# ligands/ligand_embeddings.npy is handled by the ccd_table stage (it is the input to
# that stage) and is deliberately absent here. pocket_embeddings.npy is NOT optional:
# metadata.pocket_emb_offset indexes into it, so a release without it is broken.
CARRY_OVER_FILES = [
    "pockets/pocket_embeddings.npy",
    "pockets/pocket_manifest.jsonl",
    "ligands/ligand_manifest.jsonl",
    "raft_pdb_filtered.jsonl",
    "config.json",
    "bindingdb_staff.tsv",
    "joined_embeddings.npy",
]


def stage_carry_over(args):
    v01 = Path(args.v01_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = list(CARRY_OVER_FILES) + list(getattr(args, "extra", None) or [])
    statuses = {}
    for rel in wanted:
        statuses[rel] = copy_file(v01 / rel, out / rel, mode=args.legacy)
    rep = {
        "mode": args.legacy,
        "files": statuses,
        "n_copied": sum(1 for v in statuses.values() if v in ("copied", "hardlinked")),
        "n_missing_in_v01": sum(1 for v in statuses.values() if v == "missing_source"),
        "note": (
            "Files listed here are copied from --v01_dir unchanged. Whether that includes "
            "pocket_embeddings.npy depends on what --v01_dir supplies: a build that passes a "
            "shim with re-embedded pockets ships those, not the v0.1 array. Check pocket_embeddings_source. "
            "joined_embeddings.npy is retained only to reproduce the v0.1 concatenated index; it is "
            "NOT derivable from the pocket and ligand matrices in this release unless the pocket "
            "embeddings were also carried over unchanged."
        ),
    }
    write_report(out, "carry_over", rep)
    log("[carry_over] %d files carried over (%s)" % (rep["n_copied"], args.legacy))
    return rep


# ---------------------------------------------------------------------------
# Stage 8: checksums, manifest, changelog
# ---------------------------------------------------------------------------


def _iter_release_files(out_dir):
    out_dir = Path(out_dir)
    # Both files are excluded because neither can contain its own digest.
    # DATASET_MANIFEST.json in particular is written AFTER hashing, so including it
    # would record a stale digest on every re-run and make `sha256sum -c` fail on a
    # release that is in fact intact.
    skip = {"CHECKSUMS.sha256", "DATASET_MANIFEST.json"}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out_dir).as_posix()
        if rel in skip:
            continue
        # Dotfiles are tooling state, not payload. mirror_to_zenodo.py writes its
        # resume state as .zenodo_state.json INTO this directory and exempts it
        # from its own upload, but this walk did not know that - so a failed
        # deposit left the state file hashed into the manifest, listing a 54th
        # file the deposit would never contain.
        if any(part.startswith(".") for part in path.relative_to(out_dir).parts):
            continue
        yield rel, path


def build_changelog(reports):
    ccd = reports.get("ccd_table") or {}
    idx = reports.get("indices") or {}
    spl = reports.get("splits") or {}
    met = reports.get("metadata") or {}
    unp = reports.get("flag_unparseable") or {}
    lab = reports.get("label_coords") or {}
    rnd = spl.get("random") or {}
    td = spl.get("target_disjoint") or {}
    leak = spl.get("ligand_ccd_leakage") or {}

    def g(d, *keys, default="n/a"):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur

    lines = []
    lines.append("# Changelog: rcsb-ligand-retrieval v0.1 -> v0.2")
    lines.append("")
    lines.append("Generated %s by scripts/build_v02_release.py %s." % (utcnow(), BUILDER_VERSION))
    lines.append("")
    lines.append(
        "v0.2 is a re-release of the SAME 72,199 protein-ligand complexes. No entry was "
        "added or removed. `pocket_embeddings.npy` is NOT byte-identical to v0.1: it is "
        "the per-chain re-embedding described below, and the pocket-side numbers in this "
        "release are measured on it. `ligand_embeddings.npy` is unchanged. Everything "
        "else is packaging, labelling, annotation and correction."
    )
    lines.append("")
    lines.append("## 1. Coordinate files are now labelled")
    lines.append("")
    lines.append(
        "v0.1 shipped bare `(n, 3)` float32 arrays. Element symbols were extracted during the "
        "build and discarded; pocket residue identity was never recorded. A coordinate array "
        "with no atom identity cannot be checked, re-ordered, or matched to a molecule."
    )
    lines.append("")
    lines.append(
        "- `ligands/ligand_coords.h5` gains `ligand_atom_labels/<pocket_key>`: "
        "`(atom_name S8, element S4, altloc S2)`, row-aligned to `ligands/<pocket_key>`. "
        "Labelled entries: %s / %s."
        % (g(lab, "ligands", "n_labelled"), g(lab, "ligands", "n_datasets"))
    )
    lines.append(
        "- `pockets/pocket_coords.h5` gains `pocket_residue_labels/<pocket_key>`: "
        "`(token_index i4, chain S8, seqid i4, icode S2, resname S4)`, row-aligned to "
        "`pockets/<pocket_key>`. Labelled entries: %s / %s."
        % (g(lab, "pockets", "n_labelled"), g(lab, "pockets", "n_datasets"))
    )
    lines.append(
        "- The xyz dataset paths are UNCHANGED, so v0.1 reader code keeps working verbatim."
    )
    lines.append(
        "- Ligand labels are written only where the re-extracted coordinates reproduce the "
        "shipped array (tolerance %s A); pocket labels only where notebook 09's re-extraction "
        "matched the shipped centroids exactly. A label is therefore never silently misaligned."
        % COORD_MATCH_TOL
    )
    lines.append("- The atom order is documented in the HDF5 root attributes:")
    lines.append("")
    lines.append("```")
    lines.append(ATOM_ORDER_DOC)
    lines.append("")
    lines.append(POCKET_ORDER_DOC)
    lines.append("```")
    lines.append("")
    lines.append("## 2. Ligand fingerprints are keyed by CCD code")
    lines.append("")
    lines.append(
        "The Morgan fingerprint is a pure function of the CCD code: the %s released "
        "complexes carry only %s distinct ligand vectors (%sx redundancy), and %s CCD "
        "codes map to more than one vector. Counting every embedded ligand rather than "
        "every released complex, it is %s rows over %s codes (%sx)."
        % (
            g(ccd, "n_rows_in_metadata"),
            g(ccd, "n_distinct_ccd_in_metadata"),
            g(ccd, "redundancy_factor_in_metadata"),
            g(ccd, "n_ccd_with_multiple_distinct_vectors"),
            g(ccd, "n_rows"),
            g(ccd, "n_distinct_ccd"),
            g(ccd, "redundancy_factor"),
        )
    )
    lines.append("")
    lines.append("New files:")
    lines.append("")
    lines.append(
        "- `ligands/ccd_fingerprints.npz` - bit-packed table keyed by CCD code "
        "(%s bytes vs %s bytes for the per-row matrix, %sx smaller)."
        % (
            g(ccd, "bytes_ccd_table_packed"),
            g(ccd, "bytes_per_row_matrix"),
            g(ccd, "compression_vs_row_matrix"),
        )
    )
    lines.append(
        "- `ligands/ccd_fingerprints.jsonl` / `.parquet` - CCD, SMILES, on-bits, "
        "row count, RDKit parse status, cofactor flag."
    )
    lines.append(
        "- `ligands/ligand_row_to_ccd.npy` - int32, one entry per metadata row, index into the "
        "table. The redundancy is now visible in the layout instead of hidden in the matrix."
    )
    lines.append("- `metadata.jsonl` gains a `ligand_ccd` column.")
    lines.append("- `ligands/ligand_embeddings.npy` is RETAINED unchanged for compatibility.")
    lines.append("")
    lines.append("Reconstruction: `%s`" % g(ccd, "unpack_recipe", default="see the manifest"))
    lines.append("")
    lines.append(
        "The fingerprints carry NO stereochemistry: identical CCD codes have identical vectors, "
        "so enantiomers and diastereomers with the same code are indistinguishable in ligand "
        "space."
    )
    lines.append("")
    lines.append("## 3. Pocket and ligand are now co-equal primary indices")
    lines.append("")
    lines.append(
        "v0.1 shipped one index over the L2-normalised concatenation `[pocket || ligand]`. That "
        "is a fixed, implicit weighting, and the weight is not even constant: the pocket share "
        "of the joint squared norm has median %.3f, IQR %.3f-%.3f, range %.3f-%.3f."
        % (
            MEASURED["w_p_shipped_joint_index"]["median"],
            MEASURED["w_p_shipped_joint_index"]["iqr"][0],
            MEASURED["w_p_shipped_joint_index"]["iqr"][1],
            MEASURED["w_p_shipped_joint_index"]["range"][0],
            MEASURED["w_p_shipped_joint_index"]["range"][1],
        )
    )
    lines.append("")
    lines.append("Measured same-UniProt hit@1 as a function of alpha:")
    lines.append("")
    lines.append(
        "| alpha | 0.00 (ligand only) | 0.25 | 0.50 | 0.75 | 1.00 (pocket only) | v0.1 concat |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    _a = MEASURED["same_uniprot_hit_at_1_by_alpha"]
    lines.append(
        "| hit@1 | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
        % (
            _a["0.00_ligand_only"],
            _a["0.25"],
            _a["0.50"],
            _a["0.75"],
            _a["1.00_pocket_only"],
            _a["concatenation_retained"],
        )
    )
    lines.append("")
    lines.append(
        "Same-CCD top-1 runs the other way (ligand-only %.3f, pocket-only %.3f). No "
        "single concatenation is right for both jobs, so v0.2 ships both modalities "
        "separately:"
        % (
            MEASURED["same_ccd_top_1_by_alpha"]["0.00_ligand_only"],
            MEASURED["same_ccd_top_1_by_alpha"]["1.00_pocket_only"],
        )
    )
    lines.append("")
    lines.append(
        "- `indices/pocket.faiss` - %s-d, L2-normalised, HNSW M=%s efConstruction=%s."
        % (g(idx, "pocket_index", "dim"), g(idx, "params", "M"), g(idx, "params", "efConstruction"))
    )
    lines.append(
        "- `indices/ligand.faiss` - %s-d, L2-normalised, HNSW M=%s efConstruction=%s."
        % (g(idx, "ligand_index", "dim"), g(idx, "params", "M"), g(idx, "params", "efConstruction"))
    )
    lines.append(
        "- `retrieval.faiss` - the v0.1 concatenated index, retained at its original path "
        "and demoted to one documented option."
    )
    # Rendered from MEASURED. These were literals, and a third copy of numbers that
    # MODALITY_GUIDANCE and the sidecar also carry: the CHANGELOG said the centred
    # median was -0.027 when it is +0.005, and quoted recall 0.978/0.987 where the
    # figure measures 0.957/0.971.
    lines.append(
        "- `indices/modality_guidance.json` - the measured alpha table, the anisotropy warning "
        "(random pocket pair scores %.3f raw, %.3f mean-centred; rankings survive, absolute "
        "cosines do not), and the HNSW recall figures (%.3f at efSearch 16, %.3f at 256)."
        % (
            MEASURED["anisotropy"]["pocket_random_raw"],
            MEASURED["anisotropy"]["pocket_random_centred"],
            MEASURED["ann_recall_concat"]["efSearch_16"],
            MEASURED["ann_recall_concat"]["efSearch_256"],
        )
    )
    lines.append(
        "- `indices/index_row_pocket_keys.txt` - the pocket_key of every index row, so the "
        "row alignment can be verified rather than assumed."
    )
    lines.append("")
    lines.append("### 3a. Measured retrieval performance, stratified by protein family")
    lines.append("")
    lines.append(
        "The v0.1 release reported a single aggregate same-UniProt hit@k. That number is "
        "not representative, and v0.2 says so in the artifacts themselves "
        "(`indices/modality_guidance.json` -> `measured_retrieval_performance`)."
    )
    lines.append("")
    lines.append("| | hit@1 | hit@5 | hit@10 |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| overall (n=%s queries, re-measured) | %s | %s | %s |"
        % (
            RETRIEVAL_BENCHMARK["n_queries"],
            RETRIEVAL_BENCHMARK["overall"]["hit_at_1"],
            RETRIEVAL_BENCHMARK["overall"]["hit_at_5"],
            RETRIEVAL_BENCHMARK["overall"]["hit_at_10"],
        )
    )
    lines.append(
        "| as published for v0.1 | %s | %s | %s |"
        % (
            RETRIEVAL_BENCHMARK["published_in_v0_1_paper"]["hit_at_1"],
            RETRIEVAL_BENCHMARK["published_in_v0_1_paper"]["hit_at_5"],
            RETRIEVAL_BENCHMARK["published_in_v0_1_paper"]["hit_at_10"],
        )
    )
    lines.append(
        "| PF00069 protein kinase (largest family) | %.3f | - | - |"
        % RETRIEVAL_BENCHMARK["stratified_by_pfam_hit_at_1"]["PF00069_protein_kinase"]["hit_at_1"]
    )
    lines.append(
        "| PF02898 (best of the top-12 families) | %.3f | - | - |"
        % RETRIEVAL_BENCHMARK["stratified_by_pfam_hit_at_1"]["PF02898"]["hit_at_1"]
    )
    lines.append("")
    lines.append(
        "Retrieval is WEAKEST where the corpus is densest: protein kinases, the "
        "single largest Pfam family in the dataset, are the worst-performing of the "
        "top-12 families at hit@1 %.3f against %.3f for the best. An aggregate "
        "hit@1 hides that spread. The new `pfam_acc` / `pfam_acc_all` metadata "
        "columns exist so this stratification is reproducible directly from the "
        "release."
        % (MEASURED["per_family_hit_at_1"]["PF00069"], MEASURED["per_family_hit_at_1"]["PF02898"])
    )
    lines.append("")
    lines.append("## 4. The splits are fixed")
    lines.append("")
    lines.append(
        "The shipped v0.1 split did not partition the usable set and did not match its own "
        "documented parameters."
    )
    lines.append("")
    lines.append("| | v0.1 shipped | v0.2 |")
    lines.append("|---|---|---|")
    lines.append(
        "| train / val | %s / %s | %s / %s |"
        % (
            g(rnd, "v01_comparison", "v01_n_train"),
            g(rnd, "v01_comparison", "v01_n_val"),
            g(rnd, "n_train"),
            g(rnd, "n_val"),
        )
    )
    lines.append(
        "| entries in a split that are not usable | %s | 0 |"
        % g(rnd, "v01_comparison", "v01_entries_in_a_split_but_not_usable")
    )
    lines.append(
        "| usable entries in no split | %s | 0 |"
        % g(rnd, "v01_comparison", "v01_usable_entries_in_no_split")
    )
    lines.append("")
    lines.append(
        "New: a TARGET-DISJOINT split (`splits/{train,val,test}_target_disjoint.npy`), whole "
        "clusters assigned to one side, clustered with %s on the ACTUAL ESM-2 input sequences "
        "recovered by notebook 09 - not the post-hoc `protein_seq` annotation, which is the "
        "longest chain only and is missing for 457 entries. Counts: %s train / %s val / %s test "
        "over %s clusters."
        % (
            g(td, "clustering_backend"),
            g(td, "n_train"),
            g(td, "n_val"),
            g(td, "n_test"),
            g(td, "n_clusters_over_usable"),
        )
    )
    lines.append("")
    lines.append(
        "`splits/split_leakage_report.json` reports LIGAND-CCD overlap across every boundary. A "
        "sequence-disjoint split does not stop bit-identical ligand vectors from crossing: %s%% "
        "of target-disjoint val entries have a ligand CCD that also occurs in train (random "
        "split: %s%%). Report this number with any generalisation claim."
        % (
            g(leak, "target_disjoint_split", "val", "pct_entries_whose_ccd_occurs_in_train"),
            g(leak, "random_split", "val", "pct_entries_whose_ccd_occurs_in_train"),
        )
    )
    lines.append("")
    lines.append(
        "The random split remains a REPRODUCIBILITY split, not a generalization "
        "benchmark: %d exact-duplicate groups cover %d entries (%.2f%%), %d of them "
        "cross the boundary, %.1f%% of val entries have a train neighbour at cosine "
        ">= 0.99 and %.1f%% are an exact duplicate."
        % (
            MEASURED["duplicates"]["groups"],
            MEASURED["duplicates"]["entries"],
            MEASURED["duplicates"]["pct"],
            MEASURED["duplicates"]["groups_crossing_split"],
            100 * MEASURED["duplicates"]["val_frac_train_nbr_ge_099"],
            100 * MEASURED["duplicates"]["val_frac_exact_duplicate"],
        )
    )
    lines.append("")
    lines.append("## 5. Metadata is enriched (row order preserved)")
    lines.append("")
    lines.append(
        "Row count and row order are unchanged - the splits index into this file - and the "
        "builder asserts the `pocket_key` sequence is identical to v0.1. New columns:"
    )
    lines.append("")
    lines.append("| column | coverage |")
    lines.append("|---|---|")
    lines.append("| exptl_method | %s |" % g(met, "coverage", "n_with_exptl_method"))
    lines.append(
        "| n_qualifying_ligands | %s |" % g(met, "coverage", "n_with_n_qualifying_ligands")
    )
    lines.append(
        "| is_common_cofactor (broad, advisory) | %s flagged (%s%%) |"
        % (g(met, "coverage", "n_common_cofactor"), g(met, "pct_common_cofactor"))
    )
    lines.append(
        "| is_common_cofactor_audit_set (36-code audit set) | %s flagged (%s%%) |"
        % (
            g(met, "coverage", "n_common_cofactor_audit_set"),
            g(met, "pct_common_cofactor_audit_set"),
        )
    )
    lines.append(
        "| esm_input_length, n_chains_in_esm_input, chain_order | %s |"
        % g(met, "coverage", "n_with_esm_input_length")
    )
    lines.append("| uniprot_acc (+ _all) | %s |" % g(met, "coverage", "n_with_uniprot"))
    lines.append("| pfam_acc (+ _all) | %s |" % g(met, "coverage", "n_with_pfam"))
    lines.append("| ec_number | %s |" % g(met, "coverage", "n_with_ec"))
    lines.append("| ligand_ccd, ligand_ccd_table_row | all rows |")
    lines.append("")
    lines.append(
        "`n_qualifying_ligands` makes the selection ambiguity visible: 56.1% of entries had more "
        "than one ligand passing the filter and only one was kept."
    )
    lines.append("")
    lines.append(
        "TWO cofactor flags are shipped and they are NOT interchangeable. "
        "`is_common_cofactor_audit_set` uses the exact %s-code set the audit used, so it "
        "reproduces the 25.8%% figure quoted in the paper (measured here: %s%%). "
        "`is_common_cofactor` uses a broader %s-code advisory list that is a strict superset "
        "and therefore flags more entries (%s%%) - do not quote it as the audit number. Both "
        "are independent of `EXCLUDED_HET`, the 35-code build-time blacklist that v0.1 never "
        "documented and that is now reproduced in DATASET_MANIFEST.json."
        % (
            g(met, "common_cofactor_audit_list_size"),
            g(met, "pct_common_cofactor_audit_set"),
            g(met, "common_cofactor_list_size"),
            g(met, "pct_common_cofactor"),
        )
    )
    lines.append("")
    lines.append("## 6. Unparseable SMILES are flagged, not dropped")
    lines.append("")
    lines.append(
        "%s ligand rows carry a zero fingerprint. v0.1 excluded them from the joined index "
        "silently and attributed them to transition-metal complexes, polyborane clusters and "
        "polyoxometalates. That explanation is WRONG. Re-checking every SMILES with RDKit shows "
        "they are SMILES PARSE FAILURES, not exotic chemistry. Verified breakdown: %s."
        % (g(unp, "n_zero_vector_rows"), json.dumps(unp.get("reason_counts", {})))
    )
    lines.append("")
    lines.append(
        "`ligands/unparseable_smiles.jsonl` now lists every one with its pocket_key, CCD code, "
        "SMILES and the verified reason. They remain outside the row-aligned indices - a zero "
        "vector has no direction - but they are no longer invisible."
    )
    lines.append("")
    lines.append("## 7. Integrity")
    lines.append("")
    lines.append(
        "- `CHECKSUMS.sha256` - `sha256sum -c` compatible, covers every file in the "
        "release except itself and `DATASET_MANIFEST.json` (neither can contain its "
        "own digest; the manifest is written after hashing)."
    )
    lines.append(
        "- `DATASET_MANIFEST.json` - per-file size and digest, the row-alignment contract, the "
        "build parameters, the EXCLUDED_HET blacklist, and every stage report."
    )
    lines.append("- `provenance/*_report.json` - the machine-readable output of each build stage.")
    lines.append("")
    lines.append("## Known limitations carried over unchanged")
    lines.append("")
    lines.append(
        "- The preview computed pocket embeddings from a CHIMERIC ESM-2 input (all polymer "
        "chains of model 0 concatenated), which placed non-adjacent residues next to one "
        "another in 48.85% of entries. THIS IS FIXED IN THIS RELEASE: each pocket is now "
        "encoded from its own chain, and the `embedding_input` column of "
        "pockets/pocket_residue_map.parquet records which encoding every row used. The "
        "21.09% of pockets that span chains have every contributing chain encoded."
    )
    lines.append(
        "- 8,454 non-standard residues within 8 A of a ligand consume no ESM-2 sequence position "
        "and can never be selected as pocket residues (4.31% of entries affected); "
        "`n_nonstandard_dropped_within_8A` now records the per-entry count."
    )
    lines.append(
        "- Ligand fingerprints remain 2,048-bit Morgan (radius 2) with no stereochemistry."
    )
    lines.append(
        "- Only 865 / 72,199 entries (1.2%) are joinable to BindingDB; the affinity subset is not "
        "a coverage claim."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def stage_manifest(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports = {}
    for stage in (
        "label_coords",
        "ccd_table",
        "flag_unparseable",
        "metadata",
        "indices",
        "splits",
        "carry_over",
    ):
        r = load_report(out, stage)
        if r is not None:
            reports[stage] = r

    changelog = build_changelog(reports)
    with open_text_write(out / "CHANGELOG.md") as fh:
        fh.write(changelog)

    files = []
    total_bytes = 0
    for rel, path in _iter_release_files(out):
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": rel, "bytes": size, "sha256": sha256_file(path)})
        if len(files) % 200 == 0:
            log("  hashed %d files" % len(files))

    lines = []
    for f in sorted(files, key=lambda d: d["path"]):
        lines.append("%s  %s" % (f["sha256"], f["path"]))
    with open_text_write(out / "CHECKSUMS.sha256") as fh:
        fh.write("\n".join(lines) + "\n")

    excluded_het, het_err = load_excluded_het()
    if het_err:
        log("[manifest] WARNING: could not import EXCLUDED_HET (%s)" % het_err)

    n_rows = None
    if (out / "metadata.jsonl").exists():
        n_rows = sum(1 for _ in open(str(out / "metadata.jsonl"), "r", encoding="utf-8"))

    manifest = {
        "name": "rcsb-ligand-retrieval",
        "version": RELEASE_VERSION,
        "supersedes": "0.1",
        "created_utc": utcnow(),
        "builder": "scripts/build_v02_release.py %s" % BUILDER_VERSION,
        "git_sha": git_sha(),
        "license": {
            "data": "CC-BY-4.0",
            "pipeline_code": "MIT",
            "note": "Every component is commercially redistributable; see the dataset card.",
        },
        "n_metadata_rows": n_rows,
        "n_files": len(files),
        "total_bytes": total_bytes,
        "row_alignment_contract": {
            "primary_key": "pocket_key = <pdb_id>__<ligand_ccd>_<ligand_resnum>",
            "statement": (
                "Line i of metadata.jsonl, row i of indices/pocket.faiss, row i of "
                "indices/ligand.faiss, row i of retrieval.faiss, row i of "
                "ligands/ligand_row_to_ccd.npy and line i of "
                "indices/index_row_pocket_keys.txt all describe the SAME complex. Split index "
                "arrays contain values of i. The legacy per-row embedding matrices are NOT "
                "row-aligned to metadata: index them through metadata.pocket_emb_offset and "
                "metadata.ligand_emb_offset (equivalently indices/pocket_row_offsets.npy and "
                "indices/ligand_row_offsets.npy)."
            ),
        },
        "atom_order": {"ligand": ATOM_ORDER_DOC, "pocket": POCKET_ORDER_DOC},
        "build_parameters": {
            "pocket_radius_angstrom": 8.0,
            "pocket_model": "ESM-2 esm2_t33_650M_UR50D, mean pooled over pocket residues",
            "max_esm_input_length": 1022,
            "ligand_fingerprint": "RDKit Morgan, radius 2, 2048 bits, no stereochemistry",
            "ligand_heavy_atom_range": [10, 70],
            "max_resolution_angstrom": 2.5,
            "min_protein_length": 50,
            "excluded_het_blacklist": excluded_het,
            "excluded_het_source": (
                "imported live from zenodo_pipeline_release/rcsb_ligand_retrieval/data/"
                "rcsb_filter.py::EXCLUDED_HET"
            ),
            "excluded_het_import_error": het_err,
            "excluded_het_n_codes": len(excluded_het) if excluded_het else None,
            "excluded_het_note": (
                "Undocumented in v0.1. Note that MES (12 heavy atoms), CIT (13) and EPE (15) are "
                "NOT on this list and therefore pass the >=10 heavy-atom lower bound as ligands."
            ),
        },
        "modality_guidance": MODALITY_GUIDANCE,
        "measured_retrieval_performance": RETRIEVAL_BENCHMARK,
        "checksum_scope": (
            "CHECKSUMS.sha256 covers every file in the release EXCEPT itself and "
            "DATASET_MANIFEST.json, neither of which can contain its own digest. "
            "`sha256sum -c CHECKSUMS.sha256` from the release root must report OK for "
            "all %d files." % len(files)
        ),
        "stage_reports": reports,
        "files": sorted(files, key=lambda d: d["path"]),
    }
    write_json(out / "DATASET_MANIFEST.json", manifest)
    log(
        "[manifest] %d files, %.2f GB, CHECKSUMS.sha256 + DATASET_MANIFEST.json + CHANGELOG.md"
        % (len(files), total_bytes / 1e9)
    )
    return {"n_files": len(files), "total_bytes": total_bytes}


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------


def stage_all(args):
    log("=== stage 1/8: label_coords ===")
    stage_label_coords(args)
    log("=== stage 2/8: ccd_table ===")
    stage_ccd_table(args)
    log("=== stage 3/8: flag_unparseable ===")
    stage_flag_unparseable(args)
    log("=== stage 4/8: metadata ===")
    stage_metadata(args)
    log("=== stage 5/8: indices ===")
    stage_indices(args)
    log("=== stage 6/8: splits ===")
    stage_splits(args)
    log("=== stage 7/8: carry_over ===")
    stage_carry_over(args)
    log("=== stage 8/8: manifest ===")
    stage_manifest(args)
    log("v0.2 release assembled in %s" % args.out_dir)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


class _StubPos(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _StubElement(object):
    def __init__(self, name):
        self.name = name


class _StubAtom(object):
    def __init__(self, name, element, xyz, altloc=""):
        self.name = name
        self.element = _StubElement(element)
        self.pos = _StubPos(*xyz)
        self.altloc = altloc


class _StubSeqId(object):
    def __init__(self, num):
        self.num = num


class _StubResidue(object):
    def __init__(self, name, num, atoms):
        self.name = name
        self.seqid = _StubSeqId(num)
        self._atoms = atoms

    def __iter__(self):
        return iter(self._atoms)


class _StubChain(object):
    def __init__(self, name, residues):
        self.name = name
        self._residues = residues

    def __iter__(self):
        return iter(self._residues)


class _StubStructure(object):
    """Minimal stand-in for a gemmi.Structure, sufficient for the ligand walk."""

    def __init__(self, models):
        self._models = models

    def __iter__(self):
        return iter(self._models)


def _synth_dataset(root, n_rows=24, n_ccd=6, p_dim=8, l_dim=16, n_extra_ligand=3):
    """Write a miniature but structurally faithful v0.1 snapshot + notebook-09 outputs."""
    root = Path(root)
    (root / "v01" / "ligands").mkdir(parents=True, exist_ok=True)
    (root / "v01" / "pockets").mkdir(parents=True, exist_ok=True)
    (root / "v01" / "splits").mkdir(parents=True, exist_ok=True)
    (root / "nb09").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)

    codes = ["LG%02d" % i for i in range(n_ccd)]
    ccd_vec = {}
    for j, c in enumerate(codes):
        v = np.zeros(l_dim, dtype=np.float32)
        v[(j * 3) % l_dim] = 1.0
        v[(j * 5 + 1) % l_dim] = 1.0
        ccd_vec[c] = v

    n_lig_rows = n_rows + n_extra_ligand
    lig_emb = np.zeros((n_lig_rows, l_dim), dtype=np.float32)
    pkt_emb = rng.randn(n_rows + 2, p_dim).astype(np.float32)

    lig_manifest = []
    metadata = []
    keys = []
    for i in range(n_rows):
        ccd = codes[i % n_ccd]
        pdb_id = "%04x" % (0x1000 + (i // 2))
        key = "%s__%s_%d" % (pdb_id, ccd, 100 + i)
        keys.append(key)
        lig_emb[i] = ccd_vec[ccd]
        lig_manifest.append(
            {
                "pocket_key": key,
                "pdb_id": pdb_id,
                "ligand_chain": "A",
                "ligand_resname": ccd,
                "ligand_resnum": 100 + i,
                "ligand_smiles": "C" * (i % 4 + 1),
                "ligand_heavy_atoms": 4,
                "ligand_offset": i,
            }
        )
        metadata.append(
            {
                "pdb_id": pdb_id,
                "ligand_resname": ccd,
                "ligand_chain": "A",
                "ligand_resnum": 100 + i,
                "pocket_key": key,
                "ligand_smiles": "C" * (i % 4 + 1),
                "ligand_heavy_atoms": 4,
                "pocket_n_residues": 5,
                "pocket_emb_offset": i,
                "ligand_emb_offset": i,
                "protein_seq": "AAAA" if i % 7 else "",
            }
        )
    # extra ligand rows that never made it into metadata; the last ones are zero vectors
    for k in range(n_extra_ligand):
        i = n_rows + k
        ccd = "ZZ%d" % k
        pdb_id = "9zz%d" % k
        key = "%s__%s_%d" % (pdb_id, ccd, 900 + k)
        keys.append(key)
        lig_emb[i] = 0.0  # unparseable SMILES -> zero fingerprint
        lig_manifest.append(
            {
                "pocket_key": key,
                "pdb_id": pdb_id,
                "ligand_chain": "A",
                "ligand_resname": ccd,
                "ligand_resnum": 900 + k,
                "ligand_smiles": "not-a-smiles[[",
                "ligand_heavy_atoms": 4,
                "ligand_offset": i,
            }
        )

    np.save(str(root / "v01" / "ligands" / "ligand_embeddings.npy"), lig_emb)
    np.save(str(root / "v01" / "pockets" / "pocket_embeddings.npy"), pkt_emb)
    write_jsonl(root / "v01" / "ligands" / "ligand_manifest.jsonl", lig_manifest)
    write_jsonl(root / "v01" / "metadata.jsonl", metadata)

    raft = []
    for pdb_id in sorted({m["pdb_id"] for m in metadata}):
        raft.append(
            {
                "pdb_id": pdb_id,
                "resolution": 2.0,
                "protein_length": 120,
                "ligands": [{"chain": "A", "resname": "LG00", "resnum": 1, "heavy_atoms": 12}] * 2,
                "path": "%s/%s.cif.gz" % (pdb_id[1:3], pdb_id),
            }
        )
    write_jsonl(root / "v01" / "raft_pdb_filtered.jsonl", raft)

    # a v0.1-style broken split: one out-of-usable index, one usable index omitted
    old_train = list(range(0, n_rows - 4))
    old_val = [n_rows - 4, n_rows - 3, n_rows + 100]
    np.save(
        str(root / "v01" / "splits" / "train_indices.npy"), np.asarray(old_train, dtype=np.int64)
    )
    np.save(str(root / "v01" / "splits" / "val_indices.npy"), np.asarray(old_val, dtype=np.int64))

    try:
        import h5py
    except ImportError:
        return {"codes": codes, "keys": keys, "h5": False, "metadata": metadata}

    with h5py.File(str(root / "v01" / "ligands" / "ligand_coords.h5"), "w") as fh:
        for i, key in enumerate(keys):
            fh.create_dataset("ligands/" + key, data=(rng.randn(4, 3) * 5).astype(np.float32))
    with h5py.File(str(root / "v01" / "pockets" / "pocket_coords.h5"), "w") as fh:
        for i, key in enumerate(keys[:n_rows]):
            fh.create_dataset("pockets/" + key, data=(rng.randn(5, 3) * 5).astype(np.float32))

    # notebook 09 outputs
    try:
        import pandas as pd

        rows = []
        for i, key in enumerate(keys[:n_rows]):
            prov = [
                {"chain": "A", "seqid": 10 + r, "icode": "", "resname": "ALA"} for r in range(5)
            ]
            rows.append(
                {
                    "pocket_key": key,
                    "pdb_id": metadata[i]["pdb_id"],
                    "esm_input_length": 40 + i,
                    "n_chains_in_esm_input": 1 + (i % 3),
                    "chain_order": "A,B" if i % 3 else "A",
                    "pocket_res_indices": json.dumps([3, 4, 5, 6, 7]),
                    "pocket_res_provenance": json.dumps(prov),
                    "n_pocket_residues": 5,
                    "n_nonstandard_dropped": i % 2,
                    "n_nonstandard_dropped_within_8A": i % 2,
                    "n_crystallographic_gaps": 0,
                    # row 1 did NOT reproduce the shipped coordinates: the witness gate
                    # must refuse to label it rather than attach residue identities to
                    # coordinates they may not belong to.
                    "centroid_max_abs_err": 0.5 if i == 1 else 0.0,
                }
            )
        pd.DataFrame(rows).to_parquet(
            str(root / "nb09" / "pocket_residue_map.parquet"), index=False
        )
    except ImportError:
        pass

    with gzip.open(str(root / "nb09" / "esm_input_sequences.fasta.gz"), "wt") as fh:
        for i, key in enumerate(keys[:n_rows]):
            # three sequence families so exact-sequence clustering has something to do
            fh.write(">%s\n%s\n" % (key, "ACDEFGHIKL"[: 4 + (i % 3)] * 4))

    return {"codes": codes, "keys": keys, "h5": True, "metadata": metadata}


def _self_test():
    import tempfile

    log("=" * 72)
    log("build_v02_release self-test (synthetic data, no dataset required)")
    log("=" * 72)
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        log("  %-52s %s %s" % (name, "OK" if cond else "FAIL", detail))
        if not cond:
            raise AssertionError("self-test failed: %s %s" % (name, detail))

    def skip(name, why):
        checks.append((name, None, why))
        log("  %-52s SKIP (%s)" % (name, why))

    tmp = Path(tempfile.mkdtemp(prefix="v02_selftest_"))
    try:
        synth = _synth_dataset(tmp)
        v01 = tmp / "v01"
        out = tmp / "out"
        nb09 = tmp / "nb09"

        # --- pure logic: ligand atom label extraction against a stub structure ---
        atoms = [
            _StubAtom("C1", "C", (1.0, 2.0, 3.0)),
            _StubAtom("H1", "H", (9.0, 9.0, 9.0)),  # must be skipped
            _StubAtom("N2", "N", (4.0, 5.0, 6.0), altloc="A"),
            _StubAtom("ZN", "Zn", (7.0, 8.0, 9.0)),
        ]
        st = _StubStructure(
            [
                [
                    _StubChain("B", [_StubResidue("LIG", 1, [_StubAtom("X", "C", (0, 0, 0))])]),
                    _StubChain(
                        "A",
                        [
                            _StubResidue("OTH", 1, [_StubAtom("Y", "C", (0, 0, 0))]),
                            _StubResidue("LIG", 1, atoms),
                            _StubResidue("LIG", 2, [_StubAtom("Z", "C", (0, 0, 0))]),
                        ],
                    ),
                ],
                [_StubChain("A", [_StubResidue("LIG", 1, atoms)])],  # second model, ignored
            ]
        )
        coords, labels = extract_ligand_atom_labels(st, "A", "LIG", 1)
        check("ligand walk skips hydrogens", coords.shape == (3, 3), str(coords.shape))
        check(
            "ligand walk keeps mmCIF order",
            [b.decode() for b in labels["atom_name"]] == ["C1", "N2", "ZN"],
        )
        check(
            "ligand walk keeps elements",
            [b.decode() for b in labels["element"]] == ["C", "N", "Zn"],
        )
        check("ligand walk records altloc", labels["altloc"][1].decode() == "A")
        check("ligand walk honours chain filter", float(coords[0][0]) == 1.0)
        check("labels are row-aligned to coords", len(labels) == coords.shape[0])
        missing = extract_ligand_atom_labels(st, "Q", "LIG", 1)
        check("ligand walk returns None when absent", missing == (None, None))

        # --- pure logic: pocket labels ---
        prov = json.dumps(
            [{"chain": "A", "seqid": 10 + i, "icode": "", "resname": "ALA"} for i in range(5)]
        )
        arr, status = pocket_labels_from_map(json.dumps([3, 4, 5, 6, 7]), prov, 5)
        check("pocket labels build", status == "ok" and len(arr) == 5, status)
        check("pocket token_index preserved", list(arr["token_index"]) == [3, 4, 5, 6, 7])
        bad, status = pocket_labels_from_map(json.dumps([3, 4]), prov, 5)
        check("pocket labels refuse count mismatch", bad is None, status)
        bad, status = pocket_labels_from_map(
            json.dumps([3, 4, 5, 6, 7]), json.dumps([{"chain": "A", "seqid": 1}]), 5
        )
        check("pocket labels refuse short provenance", bad is None, status)

        # --- pure logic: NaN-safe cell coercion (float('nan') is TRUTHY) ---
        nan = float("nan")
        check("clean_str rejects NaN", clean_str(nan) is None)
        check("clean_str rejects the string 'nan'", clean_str("nan") is None)
        check("clean_str keeps a real accession", clean_str("  P00533 ") == "P00533")
        check("opt_int rejects NaN", opt_int(nan) is None)
        check("opt_int rejects a missing key", opt_int(None) is None)
        check("opt_int coerces a numpy int", opt_int(np.int64(7)) == 7)
        check("opt_float rejects NaN", opt_float(nan) is None)
        check("opt_float keeps 0.0", opt_float(0.0) == 0.0)

        # --- pure logic: CCD table collapse ---
        emb = np.load(str(v01 / "ligands" / "ligand_embeddings.npy"))
        rows = read_jsonl(v01 / "ligands" / "ligand_manifest.jsonl")
        table = build_ccd_table(rows, emb, "ligand_offset", smiles_field="ligand_smiles")
        n_expected = len({r["ligand_resname"] for r in rows})
        check(
            "ccd table collapses rows", len(table["codes"]) == n_expected, str(len(table["codes"]))
        )
        check("ccd table finds no conflicts", not table["conflicts"])
        check("ccd table row map covers all rows", len(table["row_to_ccd"]) == len(rows))
        check(
            "ccd table reconstructs every row",
            all(
                np.array_equal(
                    table["vectors"][table["row_to_ccd"][i]], emb[int(r["ligand_offset"])]
                )
                for i, r in enumerate(rows)
            ),
        )
        check("ccd table detects binary fingerprints", table["is_binary"])
        packed = np.packbits(table["vectors"].astype(np.uint8), axis=1)
        unpacked = np.unpackbits(packed, axis=1)[:, : table["vectors"].shape[1]]
        check(
            "bit-packing round-trips", np.array_equal(unpacked, table["vectors"].astype(np.uint8))
        )
        # a genuine conflict must be reported, not swallowed
        bad_emb = emb.copy()
        bad_emb[6] = 1.0 - bad_emb[6]
        bad_table = build_ccd_table(rows, bad_emb, "ligand_offset")
        check("ccd table reports vector conflicts", len(bad_table["conflicts"]) >= 1)

        # --- pure logic: exact-sequence clustering + CCD leakage ---
        cl = exact_sequence_clusters(["AAA", "BBB", "AAA", "", ""])
        check("exact clustering groups identical seqs", cl[0] == cl[2])
        check("exact clustering separates distinct seqs", cl[0] != cl[1])
        check("empty sequences are singletons", cl[3] != cl[4] and cl[3] != cl[0])
        ccds = ["A", "A", "B", "C", "C"]
        leak = ccd_overlap_report({"train": [0, 2], "val": [1, 3], "test": [4]}, ccds)
        check(
            "ccd leakage counts train-seen val entries",
            leak["val"]["n_entries_whose_ccd_occurs_in_train"] == 1,
            json.dumps(leak["val"]),
        )
        check("ccd leakage counts unseen codes", leak["val"]["n_distinct_ccd_unseen_in_train"] == 1)

        # --- stage: ccd_table ---
        args = argparse.Namespace(
            v01_dir=str(v01), out_dir=str(out), nb09_dir=str(nb09), legacy="copy"
        )
        rep = stage_ccd_table(args)
        check("stage ccd_table redundancy > 1", rep["redundancy_factor"] > 1.0)
        check("stage ccd_table no conflicts", rep["n_ccd_with_multiple_distinct_vectors"] == 0)
        # The published redundancy figure is metadata-scoped, not manifest-scoped; the
        # two differ whenever a ligand was embedded but dropped from the joined index.
        check(
            "ccd_table reports a metadata-scoped redundancy separately",
            rep["n_rows_in_metadata"] == 24 and rep["n_distinct_ccd_in_metadata"] == 6,
            "%s rows / %s codes" % (rep["n_rows_in_metadata"], rep["n_distinct_ccd_in_metadata"]),
        )
        check(
            "metadata-scoped redundancy differs from manifest-scoped",
            rep["redundancy_factor_in_metadata"] != rep["redundancy_factor"],
            "%s vs %s" % (rep["redundancy_factor_in_metadata"], rep["redundancy_factor"]),
        )
        check(
            "on-bits median reported per-row as well as per-CCD",
            "median_on_bits_per_metadata_row" in rep and "median_on_bits_per_ccd" in rep,
        )
        # ||ligand|| = sqrt(on-bits) for a binary fingerprint. The two medians are taken
        # independently, so they coincide only up to the reported rounding (and, for an
        # even row count with unequal middle values, up to the median of a square root
        # not being the square root of the median).
        check(
            "median ligand norm tracks sqrt(median on-bits)",
            abs(
                rep["median_ligand_norm_per_metadata_row"]
                - rep["median_on_bits_per_metadata_row"] ** 0.5
            )
            < 1e-3,
            str(rep["median_ligand_norm_per_metadata_row"]),
        )
        d = np.load(str(out / "ligands" / "ccd_fingerprints.npz"))
        fp = np.unpackbits(d["fingerprint_packed"], axis=1)[:, : int(d["fingerprint_n_bits"][0])]
        row_map = np.load(str(out / "ligands" / "ligand_row_to_ccd.npy"))
        meta = read_jsonl(v01 / "metadata.jsonl")
        check("row_to_ccd matches metadata length", len(row_map) == len(meta))
        check(
            "npz recipe reproduces the shipped matrix",
            all(
                np.array_equal(fp[row_map[i]].astype(np.float32), emb[int(m["ligand_emb_offset"])])
                for i, m in enumerate(meta)
            ),
        )
        check("legacy matrix retained", (out / "ligands" / "ligand_embeddings.npy").exists())

        # --- stage: flag_unparseable ---
        rep = stage_flag_unparseable(args)
        check(
            "zero-vector rows found", rep["n_zero_vector_rows"] == 3, str(rep["n_zero_vector_rows"])
        )
        recs = read_jsonl(out / "ligands" / "unparseable_smiles.jsonl")
        check("every zero row is listed", len(recs) == 3)
        check("zero rows resolve to a pocket_key", all(r["pocket_key"] for r in recs))

        # --- stage: label_coords ---
        if synth["h5"] and (nb09 / "pocket_residue_map.parquet").exists():
            largs = argparse.Namespace(
                v01_dir=str(v01),
                out_dir=str(out),
                nb09_dir=str(nb09),
                mode="both",
                pdb_dir=None,
                raft_manifest=None,
                cif_cache=None,
                fetch_missing=False,
                limit=None,
            )
            rep = stage_label_coords(largs)
            check("pockets copied", rep["pockets"]["n_datasets"] == 24, str(rep["pockets"]))
            check(
                "pockets labelled (witness gate rejects 1 of 24)",
                rep["pockets"]["n_labelled"] == 23,
                str(rep["pockets"]["n_labelled"]),
            )
            check(
                "witness gate names the reason",
                rep["pockets"]["unlabelled_reasons"].get("centroid_not_reproduced") == 1,
                json.dumps(rep["pockets"]["unlabelled_reasons"]),
            )
            check("ligands copied", rep["ligands"]["n_datasets"] == 27)
            import h5py

            with (
                h5py.File(str(out / "pockets" / "pocket_coords.h5"), "r") as fh,
                h5py.File(str(v01 / "pockets" / "pocket_coords.h5"), "r") as src,
            ):
                key = synth["keys"][0]
                check(
                    "v0.1 xyz path still readable",
                    np.array_equal(fh["pockets/" + key][:], src["pockets/" + key][:]),
                )
                lab = fh["pocket_residue_labels/" + key][:]
                check("pocket labels row-aligned", len(lab) == fh["pockets/" + key].shape[0])
                check("atom order documented in file", "pocket_order_doc" in fh.attrs)
                check(
                    "unreproduced entry keeps its xyz but gets NO labels",
                    ("pockets/" + synth["keys"][1]) in fh
                    and ("pocket_residue_labels/" + synth["keys"][1]) not in fh,
                )
            with h5py.File(str(out / "ligands" / "ligand_coords.h5"), "r") as fh:
                check("ligand order documented in file", "atom_order_doc" in fh.attrs)
        else:
            skip("stage label_coords", "h5py/pandas unavailable")

        # --- stage: metadata ---
        margs = argparse.Namespace(
            v01_dir=str(v01),
            out_dir=str(out),
            nb09_dir=str(nb09),
            raft_manifest=None,
            exptl_source="none",
            exptl_cache=None,
            sifts_dir=None,
            download_sifts=False,
            workers=4,
        )
        rep = stage_metadata(margs)
        enriched = read_jsonl(out / "metadata.jsonl")
        check("metadata row count preserved", len(enriched) == len(meta))
        check(
            "metadata row ORDER preserved",
            all(enriched[i]["pocket_key"] == meta[i]["pocket_key"] for i in range(len(meta))),
        )
        check("ligand_ccd added", all("ligand_ccd" in r for r in enriched))
        check(
            "ligand_ccd equals ligand_resname",
            all(r["ligand_ccd"] == r["ligand_resname"] for r in enriched),
        )
        check("n_qualifying_ligands filled from raft", enriched[0]["n_qualifying_ligands"] == 2)
        check("is_common_cofactor present", all("is_common_cofactor" in r for r in enriched))
        check(
            "audit-set cofactor flag present",
            all("is_common_cofactor_audit_set" in r for r in enriched),
        )
        check(
            "audit cofactor set is a subset of the broad list",
            AUDIT_COMMON_ARTIFACTS_CCD <= COMMON_COFACTOR_CCD,
        )
        check(
            "metadata report carries the audit reference pct",
            rep["audit_reference_pct_selected_cofactor_or_buffer"] == 25.8
            and "pct_common_cofactor_audit_set" in rep,
        )
        check(
            "uniprot/pfam never carry the string 'nan'",
            not any(r.get("uniprot_acc") == "nan" or r.get("pfam_acc") == "nan" for r in enriched),
        )
        if (nb09 / "pocket_residue_map.parquet").exists():
            check("esm_input_length joined", enriched[0]["esm_input_length"] == 40)
        check("uniprot_acc key present even when unavailable", "uniprot_acc" in enriched[0])
        check(
            "v0.1 fields untouched",
            all(enriched[i]["ligand_smiles"] == meta[i]["ligand_smiles"] for i in range(len(meta))),
        )

        # --- stage: indices ---
        try:
            import faiss  # noqa: F401

            iargs = argparse.Namespace(
                v01_dir=str(v01),
                out_dir=str(out),
                metadata=None,
                M=8,
                efConstruction=40,
                efSearch=32,
                chunk=8,
                rebuild_joint=False,
            )
            rep = stage_indices(iargs)
            check("pocket index built", rep["pocket_index"]["ntotal"] == len(meta))
            check("ligand index built", rep["ligand_index"]["ntotal"] == len(meta))
            check("pocket index dim", rep["pocket_index"]["dim"] == 8)
            check("ligand index dim", rep["ligand_index"]["dim"] == 16)
            keys_file = (
                (out / "indices" / "index_row_pocket_keys.txt")
                .read_text(encoding="utf-8")
                .split("\n")
            )
            check("index row keys written", keys_file[0] == meta[0]["pocket_key"])
            # a self-query must return itself first: proves normalisation + alignment
            import faiss as _f

            pidx = _f.read_index(str(out / "indices" / "pocket.faiss"))
            pemb = np.load(str(v01 / "pockets" / "pocket_embeddings.npy"))
            q = pemb[int(meta[3]["pocket_emb_offset"])][None, :].astype(np.float32)
            q = q / np.linalg.norm(q)
            _, idxs = pidx.search(np.ascontiguousarray(q), 1)
            check("pocket self-query returns its own row", int(idxs[0][0]) == 3, str(idxs[0][0]))
        except ImportError:
            skip("stage indices", "faiss unavailable")

        # --- stage: splits ---
        if synth["h5"]:
            sargs = argparse.Namespace(
                v01_dir=str(v01),
                out_dir=str(out),
                nb09_dir=str(nb09),
                esm_fasta=None,
                usable_from="out",
                seed=42,
                val_frac=0.2,
                test_frac=0.2,
                min_seq_id=0.3,
                coverage=0.8,
                mmseqs="mmseqs",
                cluster_backend="exact_sequence",
                v01_splits_dir=None,
            )
            rep = stage_splits(sargs)
            tr = np.load(str(out / "splits" / "train_indices.npy")).tolist()
            va = np.load(str(out / "splits" / "val_indices.npy")).tolist()
            check("random split is disjoint", not (set(tr) & set(va)))
            check(
                "random split partitions the usable set",
                len(set(tr) | set(va)) == rep["random"]["n_usable"],
            )
            check(
                "v0.1 split defect detected",
                rep["random"]["v01_comparison"]["v01_entries_in_a_split_but_not_usable"] >= 1,
                json.dumps(rep["random"]["v01_comparison"]),
            )
            check(
                "v0.2 split has no orphans",
                rep["random"]["v01_comparison"]["v02_usable_entries_in_no_split"] == 0,
            )
            td = rep["target_disjoint"]
            check("target-disjoint split written", td is not None)
            ttr = np.load(str(out / "splits" / "train_target_disjoint.npy")).tolist()
            tva = np.load(str(out / "splits" / "val_target_disjoint.npy")).tolist()
            tte = np.load(str(out / "splits" / "test_target_disjoint.npy")).tolist()
            check(
                "target-disjoint partitions the usable set",
                len(set(ttr) | set(tva) | set(tte)) == rep["random"]["n_usable"],
            )
            seq_by_key = read_fasta_gz(nb09 / "esm_input_sequences.fasta.gz")
            enriched_keys = [r["pocket_key"] for r in enriched]

            def seq_of(ix):
                return seq_by_key.get(enriched_keys[ix], "")

            train_seqs = {seq_of(i) for i in ttr if seq_of(i)}
            held_seqs = {seq_of(i) for i in tva + tte if seq_of(i)}
            check("no sequence crosses the boundary", not (train_seqs & held_seqs))
            check(
                "ccd leakage reported for both splits",
                "random_split" in rep["ligand_ccd_leakage"]
                and "target_disjoint_split" in rep["ligand_ccd_leakage"],
            )
        else:
            skip("stage splits", "h5py unavailable")

        # --- stage: carry_over ---
        rep = stage_carry_over(
            argparse.Namespace(v01_dir=str(v01), out_dir=str(out), legacy="copy", extra=[])
        )
        check(
            "pocket_embeddings carried over",
            (out / "pockets" / "pocket_embeddings.npy").exists(),
        )
        check(
            "carried-over file is byte-identical",
            sha256_file(out / "pockets" / "pocket_embeddings.npy")
            == sha256_file(v01 / "pockets" / "pocket_embeddings.npy"),
        )
        check(
            "absent v0.1 files are reported, not fatal",
            rep["files"]["config.json"] == "missing_source",
        )

        # --- stage: manifest ---
        rep = stage_manifest(argparse.Namespace(out_dir=str(out)))
        check("checksums written", (out / "CHECKSUMS.sha256").exists())
        check("manifest written", (out / "DATASET_MANIFEST.json").exists())
        check("changelog written", (out / "CHANGELOG.md").exists())
        # CRLF in CHECKSUMS.sha256 makes `sha256sum -c` fail on EVERY line (each path
        # picks up a trailing CR), and CRLF anywhere makes the release differ
        # byte-for-byte from a Linux build. Text artifacts must be LF on all platforms.
        crlf = [
            rel
            for rel in (
                "CHECKSUMS.sha256",
                "CHANGELOG.md",
                "metadata.jsonl",
                "indices/index_row_pocket_keys.txt",
                "ligands/ccd_fingerprints.jsonl",
                "splits/splits_info.json",
                "DATASET_MANIFEST.json",
            )
            if (out / rel).exists() and b"\r" in (out / rel).read_bytes()
        ]
        check("text artifacts use LF, not CRLF", not crlf, ",".join(crlf))
        lines = [
            ln for ln in (out / "CHECKSUMS.sha256").read_text(encoding="utf-8").split("\n") if ln
        ]
        check(
            "checksum lines look like sha256sum output",
            all(len(ln.split("  ")[0]) == 64 for ln in lines),
        )
        first = lines[0].split("  ", 1)
        check(
            "checksum verifies",
            sha256_file(out / first[1]) == first[0],
            first[1],
        )
        with open(str(out / "DATASET_MANIFEST.json"), "r", encoding="utf-8") as fh:
            man = json.load(fh)
        check("manifest lists every file", man["n_files"] == len(lines))
        check("manifest carries the alignment contract", "row_alignment_contract" in man)
        check("manifest carries stage reports", "ccd_table" in man["stage_reports"])
        changelog = (out / "CHANGELOG.md").read_text(encoding="utf-8")
        check("changelog mentions the split fix", "target-disjoint" in changelog.lower())
        check("changelog corrects the SMILES claim", "SMILES PARSE FAILURES" in changelog)
        check(
            "changelog reports the per-family stratification the reviewer asked for",
            "PF00069" in changelog
            and ("%.3f" % MEASURED["per_family_hit_at_1"]["PF00069"]) in changelog
            and ("%.3f" % MEASURED["per_family_hit_at_1"]["PF02898"]) in changelog,
        )
        check(
            "changelog has no unescaped %% artifacts",
            "%%" not in changelog,
            changelog[max(0, changelog.find("%%") - 60) : changelog.find("%%") + 20],
        )
        check(
            "changelog separates the two cofactor percentages",
            "is_common_cofactor_audit_set" in changelog and "25.8" in changelog,
        )
        check(
            "manifest carries the measured retrieval benchmark",
            man["measured_retrieval_performance"]["stratified_by_pfam_hit_at_1"][
                "PF00069_protein_kinase"
            ]["hit_at_1"]
            == MEASURED["per_family_hit_at_1"]["PF00069"],
        )
        check(
            "EXCLUDED_HET imported live from the deposited pipeline",
            man["build_parameters"]["excluded_het_n_codes"] in (35, 36)
            and man["build_parameters"]["excluded_het_import_error"] is None,
            str(man["build_parameters"]["excluded_het_n_codes"]),
        )
        # Regression: DATASET_MANIFEST.json is written AFTER hashing, so if it were
        # included in its own checksum file every re-run would ship a release that
        # fails `sha256sum -c` on a file that is in fact intact.
        stage_manifest(argparse.Namespace(out_dir=str(out)))
        relines = [
            ln for ln in (out / "CHECKSUMS.sha256").read_text(encoding="utf-8").split("\n") if ln
        ]
        stale = []
        for ln in relines:
            dig, rel = ln.split("  ", 1)
            if sha256_file(out / rel) != dig:
                stale.append(rel)
        check("checksums still verify after a manifest re-run", not stale, ",".join(stale))

        n_ok = sum(1 for _, c, _ in checks if c is True)
        n_skip = sum(1 for _, c, _ in checks if c is None)
        log("-" * 72)
        log("self-test PASSED: %d checks OK, %d skipped" % (n_ok, n_skip))
        return 0
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(sp, need_nb09=False):
    sp.add_argument("--v01_dir", type=str, required=True, help="Read-only v0.1 snapshot")
    sp.add_argument("--out_dir", type=str, required=True, help="v0.2 release root (written)")
    sp.add_argument(
        "--nb09_dir",
        type=str,
        required=need_nb09,
        default=None,
        help="Directory holding pocket_residue_map.parquet and esm_input_sequences.fasta.gz",
    )


def build_parser():
    ap = argparse.ArgumentParser(
        description="Assemble the v0.2 release from the v0.1 snapshot + notebook 09 outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--self_test", action="store_true", help="Run the synthetic self-test and exit."
    )
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("label_coords", aliases=["label-coords"], help="Label the HDF5 coordinates")
    _add_common(p, need_nb09=True)
    p.add_argument("--mode", choices=["both", "pockets", "ligands"], default="both")
    p.add_argument("--pdb_dir", type=str, default=None, help="Local mmCIF mirror (.cif.gz)")
    p.add_argument("--raft_manifest", type=str, default=None)
    p.add_argument("--cif_cache", type=str, default=None, help="Where fetched mmCIF files land")
    p.add_argument("--fetch_missing", action="store_true", help="Download absent mmCIF from RCSB")
    p.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N datasets")
    p.set_defaults(func=stage_label_coords)

    p = sub.add_parser("ccd_table", aliases=["ccd-table"], help="Build the CCD->fingerprint table")
    _add_common(p)
    p.add_argument("--legacy", choices=["copy", "link", "skip"], default="copy")
    p.set_defaults(func=stage_ccd_table)

    p = sub.add_parser("indices", help="Build per-modality FAISS indices")
    _add_common(p)
    p.add_argument("--metadata", type=str, default=None, help="Override the metadata.jsonl used")
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--efConstruction", type=int, default=200)
    p.add_argument("--efSearch", type=int, default=128)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--rebuild_joint", action="store_true", help="Rebuild instead of carrying over")
    p.set_defaults(func=stage_indices)

    p = sub.add_parser("splits", help="Fix the random split and add a target-disjoint split")
    _add_common(p)
    p.add_argument("--esm_fasta", type=str, default=None)
    p.add_argument("--usable_from", choices=["auto", "out", "v01"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--min_seq_id", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--mmseqs", type=str, default="mmseqs")
    p.add_argument(
        "--cluster_backend",
        choices=["mmseqs", "exact_sequence"],
        default="mmseqs",
        help="exact_sequence is a weaker, dependency-free fallback and is labelled as such",
    )
    p.add_argument("--v01_splits_dir", type=str, default=None)
    p.set_defaults(func=stage_splits)

    p = sub.add_parser("metadata", help="Enrich metadata.jsonl (row order preserved)")
    _add_common(p)
    p.add_argument("--raft_manifest", type=str, default=None)
    p.add_argument("--exptl_source", choices=["api", "cache", "none"], default="api")
    p.add_argument("--exptl_cache", type=str, default=None)
    p.add_argument("--sifts_dir", type=str, default=None)
    p.add_argument("--download_sifts", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=stage_metadata)

    p = sub.add_parser(
        "flag_unparseable", aliases=["flag-unparseable"], help="List the zero-fingerprint ligands"
    )
    _add_common(p)
    p.set_defaults(func=stage_flag_unparseable)

    p = sub.add_parser(
        "carry_over", aliases=["carry-over"], help="Copy the v0.1 files v0.2 does not change"
    )
    _add_common(p)
    p.add_argument("--legacy", choices=["copy", "link", "skip"], default="copy")
    p.add_argument("--extra", nargs="*", default=[], help="Additional relative paths to carry")
    p.set_defaults(func=stage_carry_over)

    p = sub.add_parser("manifest", help="CHECKSUMS.sha256 + DATASET_MANIFEST.json + CHANGELOG.md")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--v01_dir", type=str, default=None)
    p.set_defaults(func=stage_manifest)

    p = sub.add_parser("all", help="Run every stage in order")
    _add_common(p, need_nb09=True)
    p.add_argument("--mode", choices=["both", "pockets", "ligands"], default="both")
    p.add_argument("--pdb_dir", type=str, default=None)
    p.add_argument("--raft_manifest", type=str, default=None)
    p.add_argument("--cif_cache", type=str, default=None)
    p.add_argument("--fetch_missing", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--legacy", choices=["copy", "link", "skip"], default="copy")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--efConstruction", type=int, default=200)
    p.add_argument("--efSearch", type=int, default=128)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--rebuild_joint", action="store_true")
    p.add_argument("--esm_fasta", type=str, default=None)
    p.add_argument("--usable_from", choices=["auto", "out", "v01"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--min_seq_id", type=float, default=0.3)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--mmseqs", type=str, default="mmseqs")
    p.add_argument("--cluster_backend", choices=["mmseqs", "exact_sequence"], default="mmseqs")
    p.add_argument("--v01_splits_dir", type=str, default=None)
    p.add_argument("--exptl_source", choices=["api", "cache", "none"], default="api")
    p.add_argument("--exptl_cache", type=str, default=None)
    p.add_argument("--sifts_dir", type=str, default=None)
    p.add_argument("--download_sifts", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=stage_all)

    sub.add_parser("self_test", aliases=["self-test"], help="Synthetic self-test").set_defaults(
        func=lambda args: _self_test()
    )
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self_test" in argv or "--self-test" in argv:
        return _self_test()
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "func", None) is None:
        ap.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
