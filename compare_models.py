"""Compare results from multiple models on the same test set."""

import ast
import pandas as pd
from pathlib import Path
from metrics import calculate_metrics, per_label_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def parse_list(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return []

def evaluate_file(csv_path, model_name):
    df = pd.read_csv(csv_path)
    df["pred"] = df["predicted_categories"].apply(parse_list)
    df["gt"] = df["ground_truth"].apply(parse_list)
    
    mask = (df["success"] == True) & (df["gt"].apply(len) > 0) & (df["pred"].apply(len) > 0)
    df_eval = df[mask]
    failed = len(df) - len(df_eval)
    
    preds = df_eval["pred"].tolist()
    gts = df_eval["gt"].tolist()
    
    metrics = calculate_metrics(preds, gts)
    report = per_label_report(preds, gts)
    
    return {
        "model": model_name,
        "total": len(df),
        "evaluated": len(df_eval),
        "failed": failed,
        "metrics": metrics,
        "report": report,
        "avg_pred_labels": sum(len(p) for p in preds) / len(preds),
        "avg_gt_labels": sum(len(g) for g in gts) / len(gts),
    }

def main():
    models = {
        "llama3.3:70b": Path("test_results.csv"),
        "gemma3:27b": Path("test_results_gemma3_27b.csv"),
    }
    
    results = {}
    for name, path in models.items():
        if not path.exists():
            print(f"⚠ {path} not found, skipping {name}")
            continue
        results[name] = evaluate_file(path, name)
        print(f"✅ Loaded {name}: {results[name]['evaluated']} evaluated, {results[name]['failed']} failed")

    # ── Overall Comparison ──
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON — Overall Metrics")
    print(f"{'='*80}")
    print(f"  {'Metric':<25}", end="")
    for name in results:
        print(f"  {name:>18}", end="")
    print(f"  {'Winner':>10}")
    print(f"  {'-'*75}")

    metric_keys = [
        ("Evaluated", "evaluated"),
        ("Failed", "failed"),
        ("Avg Pred Labels", "avg_pred_labels"),
        ("Avg GT Labels", "avg_gt_labels"),
    ]
    for display, key in metric_keys:
        print(f"  {display:<25}", end="")
        vals = {}
        for name, r in results.items():
            v = r[key]
            vals[name] = v
            print(f"  {v:>18.2f}" if isinstance(v, float) else f"  {v:>18}", end="")
        print()

    metric_keys2 = [
        ("Jaccard", "jaccard_similarity"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("Exact Match", "exact_match_ratio"),
        ("Hamming Loss", "hamming_loss"),
        ("Micro F1", "micro_f1"),
        ("Macro F1", "macro_f1"),
    ]
    for display, key in metric_keys2:
        print(f"  {display:<25}", end="")
        vals = {}
        for name, r in results.items():
            v = r["metrics"].get(key, 0)
            vals[name] = v
            print(f"  {v:>18.4f}", end="")
        # Winner (higher is better, except hamming_loss)
        if vals:
            if key == "hamming_loss":
                winner = min(vals, key=vals.get)
            else:
                winner = max(vals, key=vals.get)
            short = winner.split(":")[0]
            print(f"  {'← ' + short:>10}", end="")
        print()

    # ── Per-Label Comparison ──
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON — Per-Label F1 Scores")
    print(f"{'='*80}")
    print(f"  {'Category':<28}", end="")
    for name in results:
        print(f"  {name:>15}", end="")
    print(f"  {'Δ':>8}  {'Winner':>10}")
    print(f"  {'-'*75}")

    all_labels = set()
    for r in results.values():
        all_labels.update(r["report"].keys())

    model_names = list(results.keys())
    for label in sorted(all_labels, key=lambda l: -max(r["report"].get(l, {}).get("support", 0) for r in results.values())):
        print(f"  {label:<28}", end="")
        f1s = {}
        for name in model_names:
            f1 = results[name]["report"].get(label, {}).get("f1", 0)
            f1s[name] = f1
            print(f"  {f1:>15.3f}", end="")
        
        if len(f1s) == 2:
            vals = list(f1s.values())
            delta = vals[0] - vals[1]
            winner = max(f1s, key=f1s.get).split(":")[0]
            print(f"  {delta:>+8.3f}  {'← ' + winner:>10}", end="")
        print()

    # ── Per-Label Detail ──
    print(f"\n{'='*80}")
    print(f"  DETAILED: Precision / Recall per label")
    print(f"{'='*80}")
    for label in sorted(all_labels, key=lambda l: -max(r["report"].get(l, {}).get("support", 0) for r in results.values())):
        print(f"\n  {label} (support={max(r['report'].get(label, {}).get('support', 0) for r in results.values())}):")
        for name in model_names:
            s = results[name]["report"].get(label, {})
            print(f"    {name:<20} P={s.get('precision',0):.3f}  R={s.get('recall',0):.3f}  F1={s.get('f1',0):.3f}")

    print(f"\n{'='*80}")
    print(f"  RECOMMENDATION: Use the model with higher Micro F1 + Macro F1")
    for name, r in results.items():
        score = r["metrics"].get("micro_f1", 0) + r["metrics"].get("macro_f1", 0)
        print(f"    {name}: Micro F1 + Macro F1 = {score:.4f}")
    best = max(results, key=lambda n: results[n]["metrics"].get("micro_f1", 0) + results[n]["metrics"].get("macro_f1", 0))
    print(f"  → Best model: {best}")
    print(f"{'='*80}")

    # ── Overall sample-level metrics ──────────────────────────────────────────────
    overall = pd.DataFrame({
        "Metric":    ["Precision", "Recall", "F1 (Micro)", "F1 (Macro)", "Jaccard", "Hamming Loss"],
        "LLaMA 3.3 70b": [0.7864, 0.8165, 0.8018, 0.5999, 0.6624, 0.2019],
        "CatBoost":      [0.9980, 0.8834, 0.9373, 0.9119, 0.8819, 0.1548],
    })

    # ── Per-label metrics ─────────────────────────────────────────────────────────
    llama_per = pd.read_csv(r"c:\Users\eggoni\Desktop\llm\per_label_metrics_30pct.csv")
    llama_per["Category"] = llama_per["Category"].str.strip()

    cat_per = pd.DataFrame({
        "Category":  ["Process", "Modeling And Simulation", "Data Analytics",
                      "Software Analytics", "Ui", "Software", "Ris", "Hardware",
                      "Integrative Analysis"],
        "Precision": [1.00, 1.00, 1.00, 0.99, 0.99, 1.00, 1.00, 1.00, 0.88],
        "Recall":    [0.95, 0.92, 0.89, 0.90, 0.88, 0.85, 0.85, 0.79, 0.64],
        "F1 Score":  [0.98, 0.96, 0.94, 0.94, 0.94, 0.92, 0.92, 0.88, 0.74],
    })

    # Align labels — use CSV order as master
    labels_order = llama_per["Category"].str.strip().tolist()
    llama_per = llama_per.set_index("Category").reindex(labels_order).reset_index()
    cat_per   = cat_per.set_index("Category").reindex(labels_order).reset_index()

    # ── Plot 1 – Overall metrics bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(overall))
    w = 0.35
    bars1 = ax.bar(x - w/2, overall["LLaMA 3.3 70b"], w, label="LLaMA 3.3 70b", color="#4C72B0")
    bars2 = ax.bar(x + w/2, overall["CatBoost"],      w, label="CatBoost",      color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(overall["Metric"], rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Overall Model Comparison – LLaMA 3.3 70b vs CatBoost\n(30 % hold-out, 461 / 390 samples)")
    ax.legend()
    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(r"c:\Users\eggoni\Desktop\llm\overall_comparison.png", dpi=150)
    plt.show()

    # ── Plot 2 – Per-label F1 comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels_order))
    w = 0.35
    ax.bar(x - w/2, llama_per["F1 Score"], w, label="LLaMA 3.3 70b", color="#4C72B0")
    ax.bar(x + w/2, cat_per["F1 Score"],   w, label="CatBoost",      color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order, rotation=25, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Label F1 Score – LLaMA 3.3 70b vs CatBoost")
    ax.legend()
    plt.tight_layout()
    plt.savefig(r"c:\Users\eggoni\Desktop\llm\perlabel_f1_comparison.png", dpi=150)
    plt.show()

    # ── Plot 3 – Per-label Precision & Recall side by side ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, metric in zip(axes, ["Precision", "Recall"]):
        x = np.arange(len(labels_order))
        ax.bar(x - w/2, llama_per[metric], w, label="LLaMA 3.3 70b", color="#4C72B0")
        ax.bar(x + w/2, cat_per[metric],   w, label="CatBoost",      color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_order, rotation=30, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(f"Per-Label {metric}")
        ax.set_ylabel(metric)
        ax.legend()
    fig.suptitle("Per-Label Precision & Recall – LLaMA 3.3 70b vs CatBoost", fontsize=13)
    plt.tight_layout()
    plt.savefig(r"c:\Users\eggoni\Desktop\llm\perlabel_prec_rec_comparison.png", dpi=150)
    plt.show()

    # ── Plot 4 – Radar chart (overall metrics, excl. Hamming Loss) ───────────────
    radar_metrics = ["Precision", "Recall", "F1 (Micro)", "F1 (Macro)", "Jaccard"]
    llama_vals = overall.set_index("Metric").loc[radar_metrics, "LLaMA 3.3 70b"].tolist()
    cat_vals   = overall.set_index("Metric").loc[radar_metrics, "CatBoost"].tolist()
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    llama_vals += llama_vals[:1]; cat_vals += cat_vals[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, llama_vals, "o-", linewidth=2, label="LLaMA 3.3 70b", color="#4C72B0")
    ax.fill(angles, llama_vals, alpha=0.15, color="#4C72B0")
    ax.plot(angles, cat_vals,   "o-", linewidth=2, label="CatBoost",      color="#DD8452")
    ax.fill(angles, cat_vals,   alpha=0.15, color="#DD8452")
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart – Overall Metrics", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(r"c:\Users\eggoni\Desktop\llm\radar_comparison.png", dpi=150)
    plt.show()

    print("All plots saved to Desktop/llm/")


if __name__ == "__main__":
    main()
