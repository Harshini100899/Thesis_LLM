"""Evaluation metrics for multi-label classification."""

from typing import List, Dict, Set
from collections import Counter

# ─── All 9 categories ─────────────────────────────────────────────────────────
ALL_CATEGORIES = {
    "Modeling And Simulation",
    "Data Analytics",
    "Software Analytics",
    "Integrative Analysis",
    "Hardware",
    "Software",
    "Ui",
    "Process",
    "Ris",
}


def jaccard_similarity(pred: Set[str], true: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    return len(pred & true) / len(pred | true)


def precision_at_k(pred: List[str], true: Set[str], k: int = None) -> float:
    if k:
        pred = pred[:k]
    if not pred:
        return 0.0
    return sum(1 for p in pred if p in true) / len(pred)


def recall_at_k(pred: List[str], true: Set[str], k: int = None) -> float:
    if k:
        pred = pred[:k]
    if not true:
        return 1.0
    return sum(1 for p in pred if p in true) / len(true)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def exact_match(pred: Set[str], true: Set[str]) -> int:
    return 1 if pred == true else 0


def hamming_loss(pred: Set[str], true: Set[str], all_labels: Set[str]) -> float:
    if not all_labels:
        return 0.0
    wrong = sum(1 for label in all_labels if (label in pred) != (label in true))
    return wrong / len(all_labels)


def calculate_metrics(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    all_labels: Set[str] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-label classification.
    
    Args:
        predictions: List of predicted label lists
        ground_truths: List of ground truth label lists
        all_labels: Set of all possible labels (for Hamming loss)
    
    Returns:
        Dictionary of metric names to values
    """
    if not predictions or not ground_truths:
        return {}

    n = len(predictions)
    if all_labels is None:
        all_labels = set(ALL_CATEGORIES)

    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    exact_matches = []
    hamming_losses = []

    for pred, true in zip(predictions, ground_truths):
        pred_set = set(pred)
        true_set = set(true)
        jaccard_scores.append(jaccard_similarity(pred_set, true_set))
        prec = precision_at_k(pred, true_set)
        rec = recall_at_k(pred, true_set)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1_score(prec, rec))
        exact_matches.append(exact_match(pred_set, true_set))
        hamming_losses.append(hamming_loss(pred_set, true_set, all_labels))

    metrics = {
        "num_samples": n,
        "jaccard_similarity": sum(jaccard_scores) / n,
        "precision": sum(precision_scores) / n,
        "recall": sum(recall_scores) / n,
        "f1_score": sum(f1_scores) / n,
        "exact_match_ratio": sum(exact_matches) / n,
        "hamming_loss": sum(hamming_losses) / n,
    }

    # Micro/Macro
    label_tp = Counter()
    label_fp = Counter()
    label_fn = Counter()
    for pred, true in zip(predictions, ground_truths):
        pred_set = set(pred)
        true_set = set(true)
        for label in pred_set & true_set:
            label_tp[label] += 1
        for label in pred_set - true_set:
            label_fp[label] += 1
        for label in true_set - pred_set:
            label_fn[label] += 1

    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    metrics["micro_precision"] = micro_p
    metrics["micro_recall"] = micro_r
    metrics["micro_f1"] = f1_score(micro_p, micro_r)

    label_f1s = []
    for label in all_labels:
        tp, fp, fn = label_tp[label], label_fp[label], label_fn[label]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        label_f1s.append(f1_score(p, r))
    metrics["macro_f1"] = sum(label_f1s) / len(label_f1s) if label_f1s else 0

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Classification Metrics"):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"  Samples evaluated:    {metrics.get('num_samples', 0)}")
    print(f"\n  Sample-level metrics:")
    print(f"    Jaccard Similarity: {metrics.get('jaccard_similarity', 0):.4f}")
    print(f"    Precision:          {metrics.get('precision', 0):.4f}")
    print(f"    Recall:             {metrics.get('recall', 0):.4f}")
    print(f"    F1 Score:           {metrics.get('f1_score', 0):.4f}")
    print(f"    Exact Match Ratio:  {metrics.get('exact_match_ratio', 0):.4f}")
    print(f"    Hamming Loss:       {metrics.get('hamming_loss', 0):.4f}")
    print(f"\n  Label-level metrics:")
    print(f"    Micro Precision:    {metrics.get('micro_precision', 0):.4f}")
    print(f"    Micro Recall:       {metrics.get('micro_recall', 0):.4f}")
    print(f"    Micro F1:           {metrics.get('micro_f1', 0):.4f}")
    print(f"    Macro F1:           {metrics.get('macro_f1', 0):.4f}")
    print(f"{'='*60}\n")


def per_label_report(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    all_labels: Set[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Generate per-label precision, recall, F1, hamming loss, exact match, support."""
    if all_labels is None:
        all_labels = set(ALL_CATEGORIES)

    n = len(predictions)
    report = {}

    for label in sorted(all_labels):
        tp = fp = fn = tn = 0
        label_exact = 0

        for pred, true in zip(predictions, ground_truths):
            pred_has = label in pred
            true_has = label in true

            if pred_has and true_has:
                tp += 1
            elif pred_has and not true_has:
                fp += 1
            elif not pred_has and true_has:
                fn += 1
            else:
                tn += 1

            # Per-label exact match: both agree on this label
            if pred_has == true_has:
                label_exact += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(precision, recall)
        support = tp + fn  # number of true positives for this label
        label_hamming = (fp + fn) / n if n > 0 else 0  # per-label hamming contribution
        label_exact_ratio = label_exact / n if n > 0 else 0

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "hamming_loss": label_hamming,
            "exact_match_ratio": label_exact_ratio,
        }

    return report


def print_per_label_report(report: Dict[str, Dict[str, float]]):
    """Pretty print per-label report with all requested metrics."""
    total_support = sum(s["support"] for s in report.values())
    print(f"\n{'='*100}")
    print(f"  Per-Label Classification Metrics")
    print(f"{'='*100}")
    print(f"  {'Category':<28} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} "
          f"{'Hamming':>10} {'ExactMatch':>12} {'Support':>10}")
    print(f"  {'-'*98}")

    weighted_p = weighted_r = weighted_f1 = weighted_h = weighted_em = 0

    for label, s in sorted(report.items(), key=lambda x: -x[1]["support"]):
        sup = s["support"]
        weighted_p += s["precision"] * sup
        weighted_r += s["recall"] * sup
        weighted_f1 += s["f1"] * sup
        weighted_h += s["hamming_loss"] * sup
        weighted_em += s["exact_match_ratio"] * sup

        print(f"  {label:<28} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1']:>10.4f} "
              f"{s['hamming_loss']:>10.4f} {s['exact_match_ratio']:>12.4f} {s['support']:>10}")

    print(f"  {'-'*98}")
    if total_support > 0:
        print(f"  {'Weighted Average':<28} {weighted_p/total_support:>10.4f} "
              f"{weighted_r/total_support:>10.4f} {weighted_f1/total_support:>10.4f} "
              f"{weighted_h/total_support:>10.4f} {weighted_em/total_support:>12.4f} "
              f"{total_support:>10}")
    print(f"{'='*100}\n")


def per_label_report_df(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    all_labels: Set[str] = None,
) -> "pd.DataFrame":
    """Generate per-label report as a pandas DataFrame."""
    import pandas as pd

    report = per_label_report(predictions, ground_truths, all_labels)

    rows = []
    for label, s in sorted(report.items(), key=lambda x: -x[1]["support"]):
        rows.append({
            "Category": label,
            "Precision": round(s["precision"], 4),
            "Recall": round(s["recall"], 4),
            "F1 Score": round(s["f1"], 4),
            "Hamming Loss": round(s["hamming_loss"], 4),
            "Exact Match Ratio": round(s["exact_match_ratio"], 4),
            "Support": s["support"],
        })

    return pd.DataFrame(rows)
