"""
Classify JOSS projects into 9 research software categories using LLM.

Pipeline:
  1. Load labeled data, stratified split into 30% eval / 70% rest
  2. Classify 30% eval subset → compute metrics (unbiased test set)
  3. Classify remaining 70% → compute metrics on 100% (full picture)

Usage:
    python classify_projects.py --model llama3.3:70b --eval-only --verbose
    python classify_projects.py --model llama3.3:70b --limit 5 --verbose
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List
from collections import Counter
import time

import numpy as np
import pandas as pd

from config import (
    CATEGORIES,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    EVAL_SPLIT_RATIO,
    RANDOM_SEED,
    LABELED_CSV,
    TEST_RESULTS_CSV,
)
from data import load_projects, save_results, Project
from classifiers import OllamaClassifier
from utils import check_ollama_connection
from metrics import (
    calculate_metrics, print_metrics,
    per_label_report, print_per_label_report, per_label_report_df,
)


def stratified_multilabel_split(projects, eval_ratio, categories, seed=42):
    """Stratified split for multi-label data using iterative stratification."""
    np.random.seed(seed)
    n = len(projects)
    n_eval = int(n * eval_ratio)

    cat_list = sorted(categories)
    cat_to_idx = {c: i for i, c in enumerate(cat_list)}
    n_labels = len(cat_list)
    label_matrix = np.zeros((n, n_labels), dtype=int)
    for i, p in enumerate(projects):
        for lbl in p.ground_truth:
            if lbl in cat_to_idx:
                label_matrix[i, cat_to_idx[lbl]] = 1

    desired_eval = label_matrix.sum(axis=0) * eval_ratio
    assignments = np.zeros(n, dtype=int)
    label_order = np.argsort(label_matrix.sum(axis=0))

    for label_idx in label_order:
        has_label = label_matrix[:, label_idx] == 1
        unassigned = assignments == 0
        candidates = np.where(has_label & unassigned)[0]
        if len(candidates) == 0:
            continue
        eval_mask = assignments == 1
        current_eval = label_matrix[eval_mask, label_idx].sum() if eval_mask.any() else 0
        needed_eval = max(0, round(desired_eval[label_idx]) - current_eval)
        space_in_eval = max(0, n_eval - (assignments == 1).sum())
        needed_eval = min(needed_eval, space_in_eval, len(candidates))

        if needed_eval > 0 and needed_eval < len(candidates):
            np.random.shuffle(candidates)
            assignments[candidates[:needed_eval]] = 1
            assignments[candidates[needed_eval:]] = 2
        elif needed_eval >= len(candidates):
            assignments[candidates] = 1
        else:
            assignments[candidates] = 2

    unassigned = np.where(assignments == 0)[0]
    current_eval = (assignments == 1).sum()
    remaining_for_eval = max(0, n_eval - current_eval)
    if remaining_for_eval > 0 and len(unassigned) > 0:
        np.random.shuffle(unassigned)
        assignments[unassigned[:remaining_for_eval]] = 1
        assignments[unassigned[remaining_for_eval:]] = 2
    else:
        assignments[unassigned] = 2

    eval_projects = [projects[i] for i in range(n) if assignments[i] == 1]
    inference_projects = [projects[i] for i in range(n) if assignments[i] == 2]
    return eval_projects, inference_projects


def classify_batch(projects, classifier, verbose=False):
    """Classify a list of projects."""
    results = []
    total = len(projects)
    for i, project in enumerate(projects, 1):
        if verbose or i % 10 == 0:
            print(f"  [{i}/{total}] {project.title[:60]}...", flush=True)
        
        t0 = time.time()
        result = classifier.classify(
            title=project.title,
            description="",
            language=project.language,
            dependencies=project.dependencies,
        )
        elapsed = time.time() - t0

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
            print(f"    ⚠ Failed ({elapsed:.1f}s): {result.error}", flush=True)
        elif verbose:
            print(f"    → {result.categories} ({elapsed:.1f}s)", flush=True)
    return results


def evaluate_and_print(results, title):
    """Compute and print metrics for a set of results."""
    successful = [r for r in results if r["success"] and r["ground_truth"]]
    if not successful:
        print("  No successful classifications to evaluate.")
        return

    predictions = [r["predicted_categories"] for r in successful]
    ground_truths = [r["ground_truth"] for r in successful]

    metrics = calculate_metrics(predictions, ground_truths)
    print_metrics(metrics, title=title)

    report = per_label_report(predictions, ground_truths)
    print_per_label_report(report)

    # Compact summary
    print("\n  === ALL 9 LABELS SUMMARY ===")
    for label, s in sorted(report.items(), key=lambda x: -x[1]["support"]):
        print(f"    {label:<28} P={s['precision']:.3f}  R={s['recall']:.3f}  "
              f"F1={s['f1']:.3f}  Support={s['support']}")

    return predictions, ground_truths


def main():
    parser = argparse.ArgumentParser(description="Classify JOSS research software")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Only 30% eval, skip rest")
    parser.add_argument("--rest-only", action="store_true", help="Only 70% rest, skip 30% eval")
    parser.add_argument("--check-connection", action="store_true")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output files (e.g. gemma27b)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Determine output prefix from model name if not specified
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        prefix = args.model.replace(":", "_").replace("/", "_")

    if args.check_connection:
        ok, msg = check_ollama_connection(args.host)
        print(f"{'OK' if ok else 'FAILED'} — {msg}")
        sys.exit(0 if ok else 1)

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

    # ── Step 1: Load and split ──
    print(f"\n{'='*60}")
    print(f"Step 1: Loading labeled data from {LABELED_CSV}")
    print(f"{'='*60}")

    if not LABELED_CSV.exists():
        print(f"Error: {LABELED_CSV} not found")
        sys.exit(1)

    labeled_projects = load_projects(LABELED_CSV, limit=None, require_dependencies=True)
    labeled_projects = [p for p in labeled_projects if hasattr(p, "ground_truth") and p.ground_truth]
    print(f"  Labeled projects: {len(labeled_projects)}")

    # Ground truth stats
    label_counts = Counter()
    for p in labeled_projects:
        for l in p.ground_truth:
            label_counts[l] += 1
    avg_labels = sum(len(p.ground_truth) for p in labeled_projects) / len(labeled_projects)
    print(f"  Avg labels/project: {avg_labels:.1f}")
    print(f"  Label distribution:")
    for label, count in label_counts.most_common():
        print(f"    {label:<28} {count:>5} ({count/len(labeled_projects)*100:.1f}%)")

    # Stratified split
    eval_projects, rest_projects = stratified_multilabel_split(
        labeled_projects, EVAL_SPLIT_RATIO, CATEGORIES, seed=RANDOM_SEED
    )
    print(f"\n  Eval (30%): {len(eval_projects)}")
    print(f"  Rest (70%): {len(rest_projects)}")

    # Verify stratification
    eval_label_counts = Counter()
    for p in eval_projects:
        for l in p.ground_truth:
            eval_label_counts[l] += 1
    print(f"\n  Stratification check:")
    for label, count in label_counts.most_common():
        overall_pct = count / len(labeled_projects) * 100
        eval_pct = eval_label_counts.get(label, 0) / len(eval_projects) * 100
        diff = abs(eval_pct - overall_pct)
        print(f"    {label:<28} overall={overall_pct:5.1f}%  eval={eval_pct:5.1f}%  diff={diff:.1f}%")

    # ── Step 2: Classify 30% eval ──
    if not args.rest_only:
        print(f"\n{'='*60}")
        print(f"Step 2: Classifying {len(eval_projects)} eval projects (30%)")
        print(f"{'='*60}")

        if args.limit:
            eval_projects_subset = eval_projects[:args.limit]
            print(f"  (Limited to {args.limit})")
        else:
            eval_projects_subset = eval_projects

        eval_results = classify_batch(eval_projects_subset, classifier, verbose=args.verbose)
        eval_csv = Path(f"test_results_{prefix}.csv")
        save_results(eval_results, eval_csv)
        print(f"  Saved to {eval_csv}")

        # ── Step 3: Metrics on 30% ──
        print(f"\n{'='*60}")
        print(f"Step 3: Evaluation Metrics — 30% Test Set ({prefix})")
        print(f"{'='*60}")
        evaluate_and_print(eval_results, f"30% Test Set — {prefix} ({sum(1 for r in eval_results if r['success'])} projects)")

        successful = [r for r in eval_results if r["success"] and r["ground_truth"]]
        if successful:
            preds = [r["predicted_categories"] for r in successful]
            gts = [r["ground_truth"] for r in successful]
            df = per_label_report_df(preds, gts)
            perlabel_csv = f"per_label_metrics_30pct_{prefix}.csv"
            df.to_csv(perlabel_csv, index=False)
            print(f"  Saved: {perlabel_csv}")
    else:
        eval_results = []
        print(f"\n  --rest-only: Skipping 30% eval set.")

    # ── Step 4: Classify 70% rest ──
    if args.eval_only:
        print(f"\n  --eval-only: Done.")
        return

    print(f"\n{'='*60}")
    print(f"Step 4: Classifying {len(rest_projects)} remaining projects (70%)")
    print(f"{'='*60}")

    if args.limit:
        rest_subset = rest_projects[:args.limit]
        print(f"  (Limited to {args.limit})")
    else:
        rest_subset = rest_projects

    rest_results = classify_batch(rest_subset, classifier, verbose=args.verbose)

    # ── Metrics on 70% rest only ──
    if args.rest_only:
        print(f"\n{'='*60}")
        print(f"Step 5: Evaluation Metrics — 70% Rest Set ({prefix})")
        print(f"{'='*60}")
        evaluate_and_print(rest_results, f"70% Rest Set — {prefix} ({sum(1 for r in rest_results if r['success'])} projects)")
        rest_successful = [r for r in rest_results if r["success"] and r["ground_truth"]]
        if rest_successful:
            preds = [r["predicted_categories"] for r in rest_successful]
            gts = [r["ground_truth"] for r in rest_successful]
            df = per_label_report_df(preds, gts)
            df.to_csv(f"per_label_metrics_70pct_{prefix}.csv", index=False)
            print(f"  Saved: per_label_metrics_70pct_{prefix}.csv")
        save_results(rest_results, Path(f"rest_results_{prefix}.csv"))
        print(f"\n  Done! 70% rest: {len(rest_results)} ({sum(1 for r in rest_results if r['success'])} success)")
        return

    # ── Step 5: Metrics on 100% ──
    all_results = eval_results + rest_results
    print(f"\n{'='*60}")
    print(f"Step 5: Evaluation Metrics — 100% Labeled")
    print(f"{'='*60}")
    evaluate_and_print(all_results, f"100% Labeled ({sum(1 for r in all_results if r['success'])} projects)")

    # Save all
    save_results(all_results, Path(f"all_labeled_results_{prefix}.csv"))
    all_successful = [r for r in all_results if r["success"] and r["ground_truth"]]
    if all_successful:
        preds = [r["predicted_categories"] for r in all_successful]
        gts = [r["ground_truth"] for r in all_successful]
        df = per_label_report_df(preds, gts)
        df.to_csv(f"per_label_metrics_all_{prefix}.csv", index=False)
        print(f"  Saved: per_label_metrics_all_{prefix}.csv")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  30% eval: {len(eval_results)} ({sum(1 for r in eval_results if r['success'])} success)")
    print(f"  70% rest: {len(rest_results)} ({sum(1 for r in rest_results if r['success'])} success)")
    print(f"  100% all: {len(all_results)} ({sum(1 for r in all_results if r['success'])} success)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
