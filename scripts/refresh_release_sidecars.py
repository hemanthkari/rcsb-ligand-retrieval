"""Refresh the derived JSON sidecars of a staged release from MEASURED.

splits/splits_info.json and indices/modality_guidance.json are written by the
splits and indices stages. Re-running those stages would recompute the splits
and rebuild a gigabyte of FAISS indices purely to correct prose, so this
rewrites only the fields that are functions of MEASURED and leaves every
computed value (split counts, usable set, index parameters) untouched.

The stale text these replace was measured on the preview corpus before the
per-chain re-embedding. Run this before `manifest`, so DATASET_MANIFEST.json
and CHECKSUMS.sha256 pick the corrected files up.

Usage:
    python scripts/refresh_release_sidecars.py --release D:/v02_rebuild/release
"""

import argparse
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / "scripts"))
from build_v02_release import MEASURED, _split_warning  # noqa: E402


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def save(p, obj):
    io.open(p, "w", encoding="utf-8", newline="").write(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    args = ap.parse_args()
    root = Path(args.release)
    changed = []

    # ---- splits/splits_info.json : only the WARNING prose is stale ----
    p = root / "splits" / "splits_info.json"
    if p.exists():
        d = load(p)
        before = d.get("WARNING", "")
        d["WARNING"] = _split_warning()
        if d["WARNING"] != before:
            save(p, d)
            changed.append(str(p.relative_to(root)))
            print("  splits_info.json WARNING refreshed")
            print("    was: %s..." % before[:96])
            print("    now: %s..." % d["WARNING"][:96])

    # ---- indices/modality_guidance.json : the measured tables ----
    p = root / "indices" / "modality_guidance.json"
    if p.exists():
        d = load(p)
        d["same_uniprot_hit_at_1_by_alpha"] = MEASURED["same_uniprot_hit_at_1_by_alpha"]
        d["same_ccd_top_1_by_alpha"] = MEASURED["same_ccd_top_1_by_alpha"]
        # This sentence is about the RETAINED index, so it takes the shipped
        # figures. MEASURED["w_p"] is the per-chain reconstruction and writing
        # it here would overwrite a value the deposit already has right.
        w = MEASURED["w_p_shipped_joint_index"]
        d["why_two_indices"] = (
            "The retained concatenated index is equivalent to a FIXED alpha determined "
            "by the relative norms of the two blocks. Those norms are not constant: the "
            "pocket share of the joint squared norm has median %.3f, IQR %.3f-%.3f and "
            "range %.3f-%.3f, so the effective alpha varies per query. Separate indices "
            "make the weighting explicit."
            % (w["median"], w["iqr"][0], w["iqr"][1], w["range"][0], w["range"][1])
        )
        mrp = d.get("measured_retrieval_performance")
        if isinstance(mrp, dict):
            strat = mrp.get("stratified_by_pfam_hit_at_1")
            if isinstance(strat, dict):
                for key, code in (("PF00069_protein_kinase", "PF00069"), ("PF02898", "PF02898")):
                    if key in strat and isinstance(strat[key], dict):
                        strat[key]["hit_at_1"] = MEASURED["per_family_hit_at_1"][code]
            ov = mrp.get("overall")
            if isinstance(ov, dict):
                ov["hit_at_1"] = MEASURED["same_uniprot_hit_at_1_by_alpha"][
                    "concatenation_retained"
                ]
            mrp["interpretation"] = (
                "The aggregate number is NOT representative. Retrieval is weakest where "
                "the corpus is densest: protein kinases (PF00069), the single largest "
                "family, are the worst-performing of the top-12 families at hit@1 %.3f, "
                "against %.3f for the best. Report per-family numbers when benchmarking "
                "on this dataset."
                % (
                    MEASURED["per_family_hit_at_1"]["PF00069"],
                    MEASURED["per_family_hit_at_1"]["PF02898"],
                )
            )
        # the selection block held a third, differently-sampled copy
        sel = d.get("selection")
        if isinstance(sel, dict):
            sel["measured"] = {
                k.replace("_ligand_only", "").replace("_pocket_only", ""): v
                for k, v in MEASURED["same_uniprot_hit_at_1_by_alpha"].items()
                if k != "concatenation_retained"
            }
            sel["source"] = (
                "scripts/make_dataset_paper_figures.py against the released arrays; "
                "identical to the values reported in the Data Descriptor"
            )
        # The sidecar carried its own ann_recall copy. Fixing MODALITY_GUIDANCE
        # without this created a fresh disagreement between the sidecar (0.978)
        # and the manifest (0.957) - the same one-copy-fixed failure this whole
        # refresh exists to prevent.
        if isinstance(d.get("ann_recall_vs_exact"), dict):
            d["ann_recall_vs_exact"]["efSearch_16"] = MEASURED["ann_recall_concat"]["efSearch_16"]
            d["ann_recall_vs_exact"]["efSearch_256"] = MEASURED["ann_recall_concat"]["efSearch_256"]
            d["ann_recall_vs_exact"]["note"] = (
                "HNSW recall@1 against exact search on the retained concatenated index."
            )
        d["anisotropy_warning"] = (
            "The corpus mean vector has a large component in common with every row, so "
            "random pocket pairs score well above zero on raw cosine. RANKINGS are "
            "robust to this, but ABSOLUTE cosines are not interpretable as similarity. "
            "Do not threshold on raw cosine; mean-centre first."
        )
        save(p, d)
        changed.append(str(p.relative_to(root)))
        print("  modality_guidance.json refreshed")

    # ---- provenance reports: DATASET_MANIFEST.json embeds these ----
    p = root / "provenance" / "splits_report.json"
    if p.exists():
        d = load(p)
        touched = False
        for key in ("random", "target_disjoint"):
            blk = d.get(key)
            if isinstance(blk, dict) and "WARNING" in blk:
                blk["WARNING"] = _split_warning()
                touched = True
        if touched:
            save(p, d)
            changed.append(str(p.relative_to(root)))
            print("  splits_report.json WARNING refreshed")

    p = root / "provenance" / "indices_report.json"
    if p.exists():
        raw = io.open(p, encoding="utf-8").read()
        fixed = raw
        for stale, good in (
            ("0.610", "%.3f" % MEASURED["per_family_hit_at_1"]["PF00069"]),
            ("0.61,", "%.3f," % MEASURED["per_family_hit_at_1"]["PF00069"]),
            (
                "0.648",
                "%.3f" % MEASURED["same_uniprot_hit_at_1_by_alpha"]["concatenation_retained"],
            ),
            ("0.98,", "%.3f," % MEASURED["per_family_hit_at_1"]["PF02898"]),
            ("0.980", "%.3f" % MEASURED["per_family_hit_at_1"]["PF02898"]),
        ):
            fixed = fixed.replace(stale, good)
        if fixed != raw:
            io.open(p, "w", encoding="utf-8", newline="").write(fixed)
            changed.append(str(p.relative_to(root)))
            print("  indices_report.json per-family values refreshed")

    print()
    print("refreshed %d file(s): %s" % (len(changed), ", ".join(changed) or "none"))
    print("run `build_v02_release.py manifest` next so the digests match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
