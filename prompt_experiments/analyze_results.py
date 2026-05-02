"""
Load all experiment CSVs and produce:
  1. Console comparison table (all metrics)
  2. Per-label F1 comparison across variants
  3. Bar chart: Micro F1 per variant
  4. Heatmap: per-label F1 across variants

Usage:
    python analyze_results.py
    python analyze_results.py --metric macro_f1
"""

import argparse
import ast
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import calculate_metrics, per_label_report

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = Path(__file__).parent / "plots"

from prompt_variants import VARIANTS

# ── Parse helpers ────────────────────────────────────────────────────────────────
def parse_list(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass
    return []

def load_variant(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df["pred"] = df["predicted_categories"].apply(parse_list)
    df["gt"]   = df["ground_truth"].apply(parse_list)
    mask = df["success"] & df["gt"].apply(len).gt(0) & df["pred"].apply(len).gt(0)
    df_eval = df[mask]
    if df_eval.empty:
        return None
    preds = df_eval["pred"].tolist()
    gts   = df_eval["gt"].tolist()
    m = calculate_metrics(preds, gts)
    m["n_evaluated"] = len(df_eval)
    m["n_total"]     = len(df)
    m["report"]      = per_label_report(preds, gts)
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="micro_f1",
                        choices=["micro_f1", "macro_f1", "jaccard_similarity", "f1_score", "recall"])
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all ──
    data = {}
    for v in VARIANTS:
        csv = RESULTS_DIR / f"{v['name']}.csv"
        if not csv.exists():
            continue
        m = load_variant(csv)
        if m:
            data[v["name"]] = {"desc": v["description"], **m}

    if not data:
        print("No results found. Run run_experiments.py first.")
        return

    names = list(data.keys())

    # ── Table 1: Overall metrics ──
    metrics_to_show = [
        ("N Eval",    "n_evaluated"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1 Score",  "f1_score"),
        ("Jaccard",   "jaccard_similarity"),
        ("Hamming",   "hamming_loss"),
        ("Micro F1",  "micro_f1"),
        ("Macro F1",  "macro_f1"),
    ]
    print(f"\n{'='*110}")
    print(f"  PROMPT VARIANT COMPARISON — Overall Metrics (30% test set, same split)")
    print(f"{'='*110}")
    col_w = 14
    print(f"  {'Metric':<18}", end="")
    for n in names:
        short = n[2:18]  # strip v0_ prefix + trim
        print(f"  {short:>{col_w}}", end="")
    print()
    print(f"  {'-'*105}")
    for display, key in metrics_to_show:
        print(f"  {display:<18}", end="")
        vals = {}
        for n in names:
            v = data[n].get(key, 0)
            vals[n] = v
            if isinstance(v, float):
                print(f"  {v:>{col_w}.4f}", end="")
            else:
                print(f"  {v:>{col_w}}", end="")
        if vals and key not in ("n_evaluated", "n_total"):
            if key == "hamming_loss":
                best = min(vals, key=vals.get)
            else:
                best = max(vals, key=vals.get)
            print(f"  ← {best[:15]}", end="")
        print()
    print(f"{'='*110}")

    # ── Table 2: Per-label F1 heatmap data ──
    all_labels = sorted({
        lbl
        for m in data.values()
        for lbl in m.get("report", {})
    })
    print(f"\n{'='*110}")
    print(f"  PER-LABEL F1 SCORES ACROSS VARIANTS")
    print(f"{'='*110}")
    print(f"  {'Category':<28}", end="")
    for n in names:
        print(f"  {n[2:14]:>{col_w}}", end="")
    print()
    print(f"  {'-'*105}")
    label_f1_df = {}
    for lbl in all_labels:
        print(f"  {lbl:<28}", end="")
        row = {}
        for n in names:
            f1 = data[n]["report"].get(lbl, {}).get("f1", 0)
            row[n] = f1
            print(f"  {f1:>{col_w}.3f}", end="")
        label_f1_df[lbl] = row
        print()

    # ── Save summary CSV ──
    summary_rows = []
    for n in names:
        row = {"variant": n, "description": data[n]["desc"]}
        for display, key in metrics_to_show:
            row[key] = data[n].get(key, 0)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary saved: {summary_csv}")

    # ── Plot 1: Bar chart — primary metric per variant ──
    fig, ax = plt.subplots(figsize=(12, 5))
    vals = [data[n].get(args.metric, 0) for n in names]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[3:] for n in names], rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0, min(1.15, max(vals) + 0.15))
    ax.set_ylabel(args.metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Prompt Variant Comparison — {args.metric.replace('_',' ').title()}\n(LLaMA 3.3 70b · 30% hold-out)",
                 fontsize=12, fontweight="bold")
    ax.axhline(max(vals), color="red", linestyle="--", linewidth=0.8, alpha=0.5, label=f"Best: {max(vals):.4f}")
    ax.legend(fontsize=9)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"variant_comparison_{args.metric}.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"  Plot saved: {plot_path}")

    # ── Compute best variant (needed for plots below) ──
    best = max(names, key=lambda n: data[n].get("micro_f1", 0))

    # ── Plot 2: Heatmap of per-label F1 ──
    f1_matrix = np.array([[data[n]["report"].get(lbl, {}).get("f1", 0)
                            for n in names] for lbl in all_labels])
    fig, ax = plt.subplots(figsize=(max(10, len(names)*1.4), max(5, len(all_labels)*0.55)))
    im = ax.imshow(f1_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[3:] for n in names], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=9)
    for i in range(len(all_labels)):
        for j in range(len(names)):
            val = f1_matrix[i, j]
            color = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=color)
    plt.colorbar(im, ax=ax, fraction=0.03, label="F1 Score")
    ax.set_title("Per-Label F1 Score Heatmap — Prompt Variants\n(LLaMA 3.3 70b · 30% hold-out)",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    hm_path = PLOTS_DIR / "perlabel_f1_heatmap.png"
    plt.savefig(hm_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Heatmap saved: {hm_path}")

    # ── Plot 3: Micro F1 per-label heatmap (variants × labels) ──
    best_idx = names.index(best)
    fig, ax = plt.subplots(figsize=(max(12, len(all_labels)*0.85), max(5, len(names)*0.75)))
    im = ax.imshow(f1_matrix.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [("★ " + n[3:] if i == best_idx else n[3:]) for i, n in enumerate(names)],
        fontsize=11,
        fontweight="bold"
    )
    ax.get_yticklabels()[best_idx].set_color("#B8860B")
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=40, ha="right", fontsize=13)
    for i in range(len(names)):
        for j in range(len(all_labels)):
            val = f1_matrix[j, i]
            color = "white" if val < 0.25 or val > 0.80 else "black"
            weight = "bold" if i == best_idx else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=13, color=color, fontweight=weight)

    # Gold highlight rectangle around best variant row
    rect = plt.Rectangle(
        (-0.5, best_idx - 0.5), len(all_labels), 1,
        linewidth=3, edgecolor="gold", facecolor="none", zorder=5
    )
    ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("F1 Score", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f"Per-Label F1 Score per Variant (Variants × Labels)\n"
        f"★ Best: {best[3:]}  |  Micro F1 = {data[best].get('micro_f1',0):.4f}  "
        f"|  LLaMA 3.3 70b · 30% hold-out",
        fontsize=13, fontweight="bold", pad=14, color="#222"
    )
    plt.tight_layout()
    hm2_path = PLOTS_DIR / "variant_perlabel_f1_heatmap.png"
    plt.savefig(hm2_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Variant×Label heatmap saved: {hm2_path}")

    # ── Best variant recommendation ──
    print(f"\n  BEST VARIANT: {best}")
    print(f"    Micro F1  = {data[best].get('micro_f1',0):.4f}")
    print(f"    Macro F1  = {data[best].get('macro_f1',0):.4f}")
    print(f"    Jaccard   = {data[best].get('jaccard_similarity',0):.4f}")
    print(f"    Recall    = {data[best].get('recall',0):.4f}")
    print(f"    Desc: {data[best]['desc']}")


if __name__ == "__main__":
    main()
