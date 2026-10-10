"""Do the paper, the figure sidecars and the rebuilt deposit now agree?

Three independent artefacts state the same quantities. Until today they did
not: the deposit carried preview-era numbers, the paper carried corrected ones,
and the SI documented the gap in a table. This checks the gap is closed rather
than assuming the regeneration closed it.
"""

import io
import json
import os
import re

REL = "D:/v02_rebuild/release/"
ms = io.open("paper/nature_dataset_paper.tex", encoding="utf-8").read()
si = io.open("paper/supplementary_information.tex", encoding="utf-8").read()
both = ms + si


def j(p):
    return json.load(io.open(p, encoding="utf-8"))


f5 = j("paper/figures/fig5_embedding_geometry.json")
f4 = j("paper/figures/fig4_stratified_retrieval.json")
mg = j(REL + "indices/modality_guidance.json")
man = j(REL + "DATASET_MANIFEST.json")
lab = j(REL + "provenance/label_coords_report.json")
ver = j(REL + "verify_v02_report.json")

alpha = dict(zip([round(a, 3) for a in f5["alphas"]], f5["same_uniprot_hit1"]))
ccd = dict(zip([round(a, 3) for a in f5["alphas"]], f5["same_ccd_top1"]))

rows = []


def chk(name, figure, deposit, paper_str):
    # Also probe the PERCENTAGE form. Both documents state these quantities as
    # percentages as well as decimals, and a check that only looked for "0.298"
    # passed while both still printed "29.5%".
    pct = "%.1f" % (100.0 * figure)
    stale_pct = [
        p
        for p in re.findall(r"\b(\d\d\.\d)\\?%", both)
        if abs(float(p) - 100.0 * figure) < 1.0 and p != pct
    ]
    if stale_pct:
        print(
            "   %-38s PERCENTAGE FORM DISAGREES: text has %s, figure says %s"
            % (name, sorted(set(stale_pct)), pct)
        )
    in_paper = paper_str in both
    agree = abs(figure - deposit) < 5e-4
    rows.append((name, figure, deposit, paper_str, in_paper, agree))


chk(
    "same-UniProt hit@1, pocket-only",
    alpha[1.0],
    mg["same_uniprot_hit_at_1_by_alpha"]["1.00_pocket_only"],
    "81.2",
)
chk(
    "same-UniProt hit@1, ligand-only",
    alpha[0.0],
    mg["same_uniprot_hit_at_1_by_alpha"]["0.00_ligand_only"],
    "0.298",
)
chk(
    "same-UniProt hit@1, concatenation",
    f5["shipped_concat_same_uniprot_hit1"],
    mg["same_uniprot_hit_at_1_by_alpha"]["concatenation_retained"],
    "64.0",
)
chk(
    "same-component top-1, ligand-only",
    ccd[0.0],
    mg["same_ccd_top_1_by_alpha"]["0.00_ligand_only"],
    "62.4",
)
chk(
    "same-component top-1, pocket-only",
    ccd[1.0],
    mg["same_ccd_top_1_by_alpha"]["1.00_pocket_only"],
    "36.6",
)
chk(
    "same-component top-1, concatenation",
    f5["shipped_concat_same_ccd_top1"],
    mg["same_ccd_top_1_by_alpha"]["concatenation_retained"],
    "60.4",
)

print("=" * 94)
print("%-38s %9s %9s  %-8s %-7s %s" % ("quantity", "figure", "deposit", "paper", "in doc", "agree"))
print("=" * 94)
bad = 0
for name, fg, dp, ps, inp, ag in rows:
    flag = "" if (inp and ag) else "   <-- CHECK"
    if not (inp and ag):
        bad += 1
    print("%-38s %9.4f %9.4f  %-8s %-7s %-5s%s" % (name, fg, dp, ps, inp, ag, flag))

print()
print("--- deposit integrity ---")
n_disk = sum(len(f) for _, _, f in os.walk(REL))
tot = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(REL) for f in fs)
print("  files on disk              : %d" % n_disk)
print("  manifest n_files / entries : %s / %d" % (man.get("n_files"), len(man["files"])))
print("  manifest total_bytes       : %.3f GB" % (man["total_bytes"] / 1e9))
print(
    "  on-disk total              : %.3f GB   paper says 5.45 GB -> %s"
    % (tot / 1e9, "MATCH" if abs(tot / 1e9 - 5.45) < 0.01 else "MISMATCH")
)
print(
    "  labels                     : %d / %d (%.3f%%)"
    % (lab["ligands"]["n_labelled"], lab["ligands"]["n_datasets"], lab["ligands"]["pct_labelled"])
)
print("  worst coord error          : %s A" % lab["ligands"]["worst_coord_max_abs_err"])
print("  paper claims 72,306        : %s" % ("72,306" in ms))
print("  paper claims 466.8 MB      : %s" % ("466.8 MB" in ms))

print()
print("--- verifier, 4 checks ---")
for k in sorted(ver):
    if isinstance(ver[k], dict):
        keys = [x for x in ver[k] if x != "NOTE"]
        print("  %-18s ran, %d measurements" % (k, len(keys)))

print()
print("--- stale-value sweep over both documents ---")
# 0.982 is deliberately absent: it is a legitimate pocket-index recall@10
# value ("recall@10 of 0.944, 0.974, 0.980 and 0.982"), not the superseded
# Figure 3 top-1 median. A probe that cannot tell them apart only trains the
# reader to ignore it.
stale = {
    "0.295": "old ligand-only",
    "51.7": "old pp gap",
    # 17.3 is CORRECT against the deposited SIFTS snapshot. 17.2 came from a
    # regeneration that silently downloaded a newer snapshot from EBI, so that is
    # the value that must not reappear.
    "17.2": "pp gap from a live-SIFTS regeneration, not the deposited snapshot",
    "29.5": "old ligand-only, percentage form",
    "5.23 GB": "old deposit size",
    "22-entry": "old residue table",
    "0.440": "old random median",
    "median 0.982": "old fig3 top-1 median",
}
hits = 0
for pat, why in stale.items():
    # A bare substring match fires inside longer numbers: "29.5" matched the
    # "529.5 MB" file size in Table 3. Anchor numeric probes so a digit may not
    # immediately precede or follow them.
    rx = re.escape(pat)
    if pat[0].isdigit():
        rx = r"(?<![\d.])" + rx + r"(?![\d])"
    c = len(re.findall(rx, both))
    if c:
        hits += 1
        print("  %-12s %-22s %d occurrence(s)" % (pat, why, c))
if not hits:
    print("  none")

print()
print(
    "RESULT: %s"
    % ("ALL CONSISTENT" if bad == 0 and hits == 0 else "%d disagreement(s)" % (bad + hits))
)
