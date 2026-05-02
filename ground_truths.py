"""Load ground truth dependency lists from CSV.

Note: This module loads RAW DEPENDENCIES (not category labels).
Category labels are loaded by data.py from joss_all_with_dependency_labels1.csv.
"""
import csv
import ast


def load_ground_truths(csv_path=r"c:\Users\eggoni\Desktop\llm\joss_all_with_dependency_labels1.csv"):
    """Load ground truth dependencies from the CSV file.
    
    Returns:
        dict: Mapping of joss_id -> list of dependency strings
    """
    ground_truths = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joss_id = row.get("joss_id", "").strip()
            deps_raw = row.get("dependecies_found", "").strip()
            if not joss_id or not deps_raw:
                continue
            try:
                deps = ast.literal_eval(deps_raw)
            except (ValueError, SyntaxError):
                deps = []
            ground_truths[joss_id] = deps
    return ground_truths


if __name__ == "__main__":
    gt = load_ground_truths()
    print(f"Loaded ground truths for {len(gt)} projects")
    for joss_id, deps in list(gt.items())[:3]:
        print(f"\nJOSS ID {joss_id}: {len(deps)} dependencies")
        print(f"  First 5: {deps[:5]}")
