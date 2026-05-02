"""
Classify ALL labeled projects and evaluate against ground truth.
No splitting — classifies 100% and computes metrics on 100%.

Usage:
    python evaluate_all.py --model llama3.3:70b --limit 5 --verbose
    python evaluate_all.py --model llama3.3:70b --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import List
from collections import Counter

from config import (
    CATEGORIES,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    LABELED_CSV,
)
from data import load_projects, save_results, Project
from classifiers import OllamaClassifier
from utils import check_ollama_connection
from metrics import (
    calculate_metrics, print_metrics,
    per_label_report, print_per_label_report, per_label_report_df,
)

# Output files
ALL_RESULTS_CSV = Path("all_labeled_results.csv")
ALL_PERLABEL_CSV = Path("per_label_metrics_all.csv")


def classify_batch(projects, classifier, verbose=False):
    """Classify projects using title, language, dependencies."""
    results = []
    total = len(projects)

    for i, project in enumerate(projects, 1):
        if verbose or i % 10 == 0:
            print(f"  [{i}/{total}] {project.title[:60]}...")

        result = classifier.classify(
            title=project.title,
            description="",
            language=project.language,
            dependencies=project.dependencies,
        )

        results.append({
            "joss_id": project.joss_id,
            "title": project.title,
            "language": project.language,
            "dependencies": str(project.dependencies[:20]),
            "predicted_categories": result.categories,
            "ground_truth": project.ground_truth if hasattr(project, "ground_truth") else [],
            "reasoning": result.reasoning,
            "success": result.success,
            "error": result.error or "",
        })

        if verbose and not result.success:
            print(f"    ⚠ Failed: {result.error}")
        elif verbose:
            print(f"    → {result.categories}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Classify ALL labeled projects and evaluate")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None, help="Limit projects (for testing)")
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ── Connect ──
    print(f"Connecting to {args.host}...")
    ok, msg = check_ollama_connection(args.host)
    if not ok:
        print(f"Error: {msg}")
        sys.exit(1)
    print(f"  {msg}")

    classifier = OllamaClassifier(
        model=args.model, host=args.host,
        temperature=args.temperature, use_few_shot=not args.no_few_shot,
    )
    if not classifier.is_available():
        print(f"Error: Model '{args.model}' not available")
        sys.exit(1)
    print(f"  Model: {args.model}")

    # ── Step 1: Load ALL labeled data ──
    print(f"\n{'='*60}")
    print(f"Step 1: Loading ALL labeled data from {LABELED_CSV}")
    print(f"{'='*60}")

    if not LABELED_CSV.exists():
        print(f"Error: {LABELED_CSV} not found")
        sys.exit(1)

    projects = load_projects(LABELED_CSV, limit=None, require_dependencies=True)
    projects = [p for p in projects if hasattr(p, "ground_truth") and p.ground_truth]
    print(f"  Total labeled projects: {len(projects)}")

    # Label distribution
    label_counts = Counter()
    for p in projects:
        for l in p.ground_truth:
            label_counts[l] += 1
    print(f"  Label distribution:")
    for label, count in label_counts.most_common():
        print(f"    {label:<28} {count:>5} ({count/len(projects)*100:.1f}%)")

    if args.limit:
        projects = projects[:args.limit]
        print(f"\n  (Limited to {args.limit} for testing)")

    # ── Step 2: Classify ALL ──
    print(f"\n{'='*60}")
    print(f"Step 2: Classifying {len(projects)} projects")
    print(f"{'='*60}")

    results = classify_batch(projects, classifier, verbose=args.verbose)

    # Save all results
    save_results(results, ALL_RESULTS_CSV)
    print(f"\n  Results saved to {ALL_RESULTS_CSV}")

    # ── Step 3: Evaluate ──
    successful = [r for r in results if r["success"] and r["ground_truth"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        print(f"\n{'='*60}")
        print(f"Step 3: Evaluation — ALL {len(successful)} projects")
        print(f"{'='*60}")

        predictions = [r["predicted_categories"] for r in successful]
        ground_truths = [r["ground_truth"] for r in successful]

        metrics = calculate_metrics(predictions, ground_truths)
        print_metrics(metrics, title=f"ALL Labeled ({len(successful)} projects) — LLM vs Ground Truth")

        report = per_label_report(predictions, ground_truths)
        print_per_label_report(report)

        # Compact summary
        print("\n  === ALL 9 LABELS SUMMARY ===")
        for label, s in sorted(report.items(), key=lambda x: -x[1]["support"]):
            print(f"    {label:<28} P={s['precision']:.3f}  R={s['recall']:.3f}  "
                  f"F1={s['f1']:.3f}  Hamming={s['hamming_loss']:.3f}  Support={s['support']}")

        # Save per-label CSV
        df_perlabel = per_label_report_df(predictions, ground_truths)
        df_perlabel.to_csv(ALL_PERLABEL_CSV, index=False)
        print(f"\n  Per-label metrics saved to {ALL_PERLABEL_CSV}")
    else:
        print("\n  No successful classifications.")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Total:      {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Output:     {ALL_RESULTS_CSV}")
    print(f"  Metrics:    {ALL_PERLABEL_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
