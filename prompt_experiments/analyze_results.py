"""
Analyze prompt variant experiment results.

Usage:
    python prompt_experiments/analyze_results.py
    python prompt_experiments/analyze_results.py --model llama3.3:70b
    python prompt_experiments/analyze_results.py --model llama3.3:70b --sculpting-only
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import CATEGORIES
from metrics import calculate_metrics

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = Path(__file__).parent / "plots"

CATEGORIES_ORDERED = sorted(CATEGORIES)

# ── helpers ────────────────────────────────────────────────────────────────────────

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


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pred"] = df["predicted_categories"].apply(parse_list)
    df["gt"]   = df["ground_truth"].apply(parse_list)
    mask = df["success"] & df["gt"].apply(len).gt(0) & df["pred"].apply(len).gt(0)
    return df[mask].copy()


def compute_metrics(df: pd.DataFrame) -> dict:
    m = calculate_metrics(df["pred"].tolist(), df["gt"].tolist())
    m["n"]              = len(df)
    m["avg_latency_s"]  = df["latency_s"].mean() if "latency_s" in df.columns else float("nan")
    return m


def per_label_f1(df: pd.DataFrame) -> dict:
    """Return per-label F1 dict for a result dataframe."""
    scores = {}
    for cat in CATEGORIES_ORDERED:
        y_true = [int(cat in row) for row in df["gt"]]
        y_pred = [int(cat in row) for row in df["pred"]]
        tp = sum(a and b for a, b in zip(y_true, y_pred))
        fp = sum((not a) and b for a, b in zip(y_true, y_pred))
        fn = sum(a and (not b) for a, b in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        scores[cat] = round(f1, 4)
    return scores


# ── main ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default=None, help="Filter to one model slug, e.g. llama3.3-70b")
    parser.add_argument("--sculpting-only", action="store_true")
    parser.add_argument("--base-only",      action="store_true")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── discover result files ──────────────────────────────────────────────────────
    pattern = f"*__{args.model}.csv" if args.model else "*.csv"
    csv_files = [f for f in RESULTS_DIR.glob(pattern)
                 if not f.name.endswith(".tmp.csv")
                 and f.name != "summary_metrics.csv"
                 and f.name != "test_split_ids.json"]

    if not csv_files:
        print(f"No result CSVs found in {RESULTS_DIR}  (pattern: {pattern})")
        sys.exit(1)

    if args.sculpting_only:
        csv_files = [f for f in csv_files if "sc_" in f.stem or f.stem.endswith("_sc")]
    elif args.base_only:
        csv_files = [f for f in csv_files if "sc_" not in f.stem and not f.stem.endswith("_sc")]

    csv_files = sorted(csv_files)
    print(f"Found {len(csv_files)} result files\n")

    # ── collect metrics ────────────────────────────────────────────────────────────
    summary_rows  = []
    label_f1_rows = {}   # variant_model → {cat: f1}

    for path in csv_files:
        stem = path.stem   # e.g. v02_seven_cot_desc__llama3.3-70b
        parts = stem.rsplit("__", 1)
        variant = parts[0]
        model   = parts[1] if len(parts) == 2 else "unknown"

        try:
            df = load_csv(path)
        except Exception as e:
            print(f"  ⚠  {path.name}: {e}")
            continue

        if df.empty:
            print(f"  ⚠  {path.name}: no valid rows")
            continue

        m = compute_metrics(df)
        label_f1_rows[stem] = per_label_f1(df)

        summary_rows.append({
            "variant":       variant,
            "model":         model,
            "n":             m["n"],
            "micro_f1":      round(m.get("micro_f1", 0), 4),
            "macro_f1":      round(m.get("macro_f1", 0), 4),
            "precision":     round(m.get("precision", 0), 4),
            "recall":        round(m.get("recall", 0), 4),
            "jaccard":       round(m.get("jaccard_similarity", 0), 4),
            "avg_latency_s": round(m.get("avg_latency_s", 0), 1),
        })
        print(f"  ✅ {stem:<55}  micro_f1={m.get('micro_f1',0):.4f}  "
              f"macro_f1={m.get('macro_f1',0):.4f}  n={m['n']}")

    if not summary_rows:
        print("No data to summarise.")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "micro_f1"], ascending=[True, False])
    out_csv = RESULTS_DIR / "summary_metrics.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n📄 Summary saved → {out_csv}")

    # ── print table ────────────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  {'Variant':<42} {'Model':<18} {'N':>5} {'MicroF1':>8} {'MacroF1':>8} "
          f"{'Prec':>7} {'Recall':>7} {'Jaccard':>8} {'AvgLat':>8}")
    print(f"  {'-'*100}")
    for _, row in summary_df.iterrows():
        print(f"  {row['variant']:<42} {row['model']:<18} {row['n']:>5} "
              f"{row['micro_f1']:>8.4f} {row['macro_f1']:>8.4f} "
              f"{row['precision']:>7.4f} {row['recall']:>7.4f} "
              f"{row['jaccard']:>8.4f} {row['avg_latency_s']:>7.1f}s")

    # ── Plot 1: Micro F1 bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(summary_df) * 0.55), 5))
    colors = ["#e07b39" if "sc_" in v or v.endswith("_sc") else "#4878cf"
              for v in summary_df["variant"]]
    bars = ax.bar(range(len(summary_df)), summary_df["micro_f1"], color=colors)
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(
        [f"{r['variant']}\n({r['model']})" for _, r in summary_df.iterrows()],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("Micro F1")
    ax.set_title("Prompt Variant Ablation — Micro F1")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, summary_df["micro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4878cf", label="no sculpting"),
                        Patch(color="#e07b39", label="with sculpting")], fontsize=8)
    plt.tight_layout()
    p1 = PLOTS_DIR / "variant_comparison_micro_f1.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"📊 Plot saved → {p1}")

    # ── Plot 2: Per-label F1 heatmap ───────────────────────────────────────────────
    if label_f1_rows:
        heatmap_df = pd.DataFrame(label_f1_rows, index=CATEGORIES_ORDERED).T
        heatmap_df.index = [i.rsplit("__", 1)[0] for i in heatmap_df.index]  # strip model slug

        fig, ax = plt.subplots(figsize=(max(8, len(CATEGORIES_ORDERED) * 0.9),
                                        max(4, len(heatmap_df) * 0.4)))
        sns.heatmap(heatmap_df.astype(float), ax=ax, annot=True, fmt=".2f",
                    cmap="YlGnBu", vmin=0, vmax=1,
                    linewidths=0.3, linecolor="white",
                    annot_kws={"size": 7})
        ax.set_title("Per-Label F1 by Variant")
        ax.set_xlabel("Category")
        ax.set_ylabel("Variant")
        plt.tight_layout()
        p2 = PLOTS_DIR / "perlabel_f1_heatmap.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"📊 Plot saved → {p2}")

    # ── Plot 3: sculpting comparison ───────────────────────────────────────────────
    sc_pairs = [
        ("v02_seven_cot_desc",    "v13sc_seven_cot_desc"),
        ("v06_seven_cot_no_desc", "v12sc_seven_cot_no_desc"),
        ("v05_seven_no_cot_desc", "v11sc_seven_no_cot_desc"),
        ("v03_ten_cot_desc",      "v14sc_ten_cot_desc"),
    ]
    pair_data = []
    for base, sc in sc_pairs:
        b_rows = summary_df[summary_df["variant"] == base]
        s_rows = summary_df[summary_df["variant"] == sc]
        if not b_rows.empty and not s_rows.empty:
            pair_data.append({
                "pair": base.replace("v0", "").replace("v1", ""),
                "no_sculpting": b_rows.iloc[0]["micro_f1"],
                "sculpting":    s_rows.iloc[0]["micro_f1"],
            })
    if pair_data:
        pair_df = pd.DataFrame(pair_data)
        x = np.arange(len(pair_df))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.2, pair_df["no_sculpting"], 0.35, label="no sculpting", color="#4878cf")
        ax.bar(x + 0.2, pair_df["sculpting"],    0.35, label="sculpting",    color="#e07b39")
        ax.set_xticks(x)
        ax.set_xticklabels(pair_df["pair"], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Micro F1")
        ax.set_ylim(0, 1.0)
        ax.set_title("Sculpting vs No-Sculpting (Matched Pairs)")
        ax.legend()
        plt.tight_layout()
        p3 = PLOTS_DIR / "sculpting_comparison.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"📊 Plot saved → {p3}")

    print(f"\n✅ Done. Copy {out_csv} to your thesis.")


if __name__ == "__main__":
    main()
