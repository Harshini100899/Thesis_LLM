"""Analyze test_results.csv and produce thesis-ready metrics."""

import ast
import pandas as pd
from pathlib import Path
from collections import Counter
from metrics import (
    calculate_metrics, print_metrics,
    per_label_report, print_per_label_report, per_label_report_df,
)

RESULTS_CSV = Path("test_results.csv")


def parse_list(raw):
    """Parse string representation of list."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return []


def main():
    if not RESULTS_CSV.exists():
        print(f"Error: {RESULTS_CSV} not found")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")

    # Parse lists
    df["pred"] = df["predicted_categories"].apply(parse_list)
    df["gt"] = df["ground_truth"].apply(parse_list)

    # Filter successful with ground truth
    mask = (df["success"] == True) & (df["gt"].apply(len) > 0) & (df["pred"].apply(len) > 0)
    df_eval = df[mask].copy()
    print(f"Evaluatable: {len(df_eval)} projects")
    print(f"Failed/empty: {len(df) - len(df_eval)} projects")

    predictions = df_eval["pred"].tolist()
    ground_truths = df_eval["gt"].tolist()

    # ── Overall Metrics ──
    metrics = calculate_metrics(predictions, ground_truths)
    print_metrics(metrics, title=f"THESIS RESULTS ({len(df_eval)} projects)")

    # ── Per-Label Report ──
    report = per_label_report(predictions, ground_truths)
    print_per_label_report(report)

    # ── Compact Summary ──
    print("\n  === THESIS TABLE: Per-Label Performance ===")
    print(f"  {'Category':<28} {'P':>6} {'R':>6} {'F1':>6} {'HL':>6} {'Supp':>6}")
    print(f"  {'-'*58}")
    for label, s in sorted(report.items(), key=lambda x: -x[1]["support"]):
        print(f"  {label:<28} {s['precision']:.3f}  {s['recall']:.3f}  "
              f"{s['f1']:.3f}  {s['hamming_loss']:.3f}  {s['support']:>5}")

    # ── Save CSV for thesis ──
    df_perlabel = per_label_report_df(predictions, ground_truths)
    df_perlabel.to_csv("thesis_per_label_metrics.csv", index=False)
    print(f"\n  Saved: thesis_per_label_metrics.csv")

    # ── Prediction Distribution ──
    print(f"\n  === PREDICTION STATISTICS ===")
    pred_counts = Counter()
    for p in predictions:
        for label in p:
            pred_counts[label] += 1
    gt_counts = Counter()
    for g in ground_truths:
        for label in g:
            gt_counts[label] += 1

    print(f"  {'Category':<28} {'Predicted':>10} {'GroundTruth':>12} {'Ratio':>8}")
    print(f"  {'-'*62}")
    all_labels = sorted(set(list(pred_counts.keys()) + list(gt_counts.keys())))
    for label in sorted(all_labels, key=lambda x: -gt_counts.get(x, 0)):
        p_count = pred_counts.get(label, 0)
        g_count = gt_counts.get(label, 0)
        ratio = p_count / g_count if g_count > 0 else float('inf')
        marker = " ⚠️" if ratio < 0.5 or ratio > 1.5 else ""
        print(f"  {label:<28} {p_count:>10} {g_count:>12} {ratio:>7.2f}{marker}")

    # ── Avg labels per project ──
    avg_pred = sum(len(p) for p in predictions) / len(predictions)
    avg_gt = sum(len(g) for g in ground_truths) / len(ground_truths)
    print(f"\n  Avg predicted labels/project: {avg_pred:.1f}")
    print(f"  Avg ground truth labels/project: {avg_gt:.1f}")

    # ── Error Analysis: worst mismatches ──
    print(f"\n  === ERROR ANALYSIS: Biggest Mismatches ===")
    df_eval["jaccard"] = df_eval.apply(
        lambda r: len(set(r["pred"]) & set(r["gt"])) / len(set(r["pred"]) | set(r["gt"]))
        if len(set(r["pred"]) | set(r["gt"])) > 0 else 1.0, axis=1
    )
    worst = df_eval.nsmallest(10, "jaccard")
    for _, row in worst.iterrows():
        print(f"\n  [{row['joss_id']}] {row['title'][:60]}")
        print(f"    Pred: {row['pred']}")
        print(f"    GT:   {row['gt']}")
        print(f"    Jaccard: {row['jaccard']:.3f}")

    # ── Ground Truth Quality Check ──
    print(f"\n  === GROUND TRUTH QUALITY CHECK ===")
    sa_projects = df_eval[df_eval["gt"].apply(lambda x: "Software Analytics" in x)]
    print(f"  Projects with 'Software Analytics' in GT: {len(sa_projects)}")
    print(f"  Sample titles with SA label:")
    for _, row in sa_projects.head(10).iterrows():
        pred_has = "Software Analytics" in row["pred"]
        print(f"    {'✅' if pred_has else '❌'} {row['title'][:70]}")


if __name__ == "__main__":
    main()
