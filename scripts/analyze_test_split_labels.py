import json
import csv
import ast
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Load test split IDs
with open("prompt_experiments/results/test_split_ids.json") as f:
    test_ids = set(json.load(f))

# Load CSV and filter test samples
csv_path = "datasets/joss_all_with_dependency_labels1.csv"

label_counter = Counter()
found_ids = set()
total_samples = 0

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        joss_id = str(row['joss_id']).strip()
        if joss_id in test_ids:
            found_ids.add(joss_id)
            total_samples += 1
            raw_labels = row.get('dependency_labels', '[]').strip()
            try:
                labels = ast.literal_eval(raw_labels)
            except Exception:
                labels = []
            for label in labels:
                label_counter[label.strip()] += 1

print(f"Test IDs requested: {len(test_ids)}")
print(f"Test IDs found in CSV: {len(found_ids)}")
print(f"Total test samples matched: {total_samples}")
print(f"\nClass distribution in test set ({total_samples} samples):")
print(f"{'Label':<40} {'Count':>6}  {'% of samples':>12}")
print("-" * 62)
for label, count in sorted(label_counter.items(), key=lambda x: -x[1]):
    pct = 100.0 * count / total_samples if total_samples else 0
    print(f"{label:<40} {count:>6}  {pct:>11.1f}%")

# Also show which test IDs were NOT found in the CSV
missing = test_ids - found_ids
if missing:
    print(f"\nTest IDs not found in CSV ({len(missing)}):")
    print(sorted(missing))

# After the print block, add visualization:
if label_counter:
    labels_sorted = [l for l, _ in sorted(label_counter.items(), key=lambda x: -x[1])]
    counts_sorted = [label_counter[l] for l in labels_sorted]
    pcts_sorted = [100.0 * c / total_samples for c in counts_sorted]

    # Use a colormap so each bar has a distinct color based on its rank
    greens = plt.cm.Greens(np.linspace(0.4, 0.9, len(labels_sorted)))
    colors = greens[::-1]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"Label Distribution in Test Set (n={total_samples})", fontsize=18, fontweight='bold')

    bars = ax.barh(labels_sorted[::-1], counts_sorted[::-1], color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel("Count", fontsize=14)
    ax.set_title("Sample Count per Label", fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    ax.spines[['top', 'right']].set_visible(False)
    for bar, count, pct in zip(bars, counts_sorted[::-1], pcts_sorted[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{count}  ({pct:.1f}%)", va='center', fontsize=12)
    ax.set_xlim(0, max(counts_sorted) * 1.25)

    plt.tight_layout()
    out_path = "plots/test_split_label_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {out_path}")
