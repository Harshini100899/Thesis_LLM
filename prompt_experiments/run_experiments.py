"""
Run prompt variant experiments on the same 30% stratified test split.

Usage:
    # Run all variants
    python run_experiments.py --model llama3.3:70b

    # Run a single variant
    python run_experiments.py --model llama3.3:70b --variant v5_five_shot_cot

    # Dry run (limit 5 projects)
    python run_experiments.py --model llama3.3:70b --limit 5
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from collections import Counter
import os

import numpy as np
import pandas as pd

# ── Resolve parent dir (llm/src) so we can import from there ───────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    CATEGORIES, DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE, EVAL_SPLIT_RATIO, RANDOM_SEED, LABELED_CSV,
)
from data import load_projects
from utils import check_ollama_connection
from metrics import calculate_metrics, per_label_report, per_label_report_df

from prompt_variants import VARIANTS, VARIANT_MAP

RESULTS_DIR = Path(__file__).parent / "results"
SPLIT_FILE = RESULTS_DIR / "test_split_ids.json"

# ── Stratified split (same logic as classify_projects.py) ─────────────────────
def stratified_multilabel_split(projects, eval_ratio, categories, seed=42):
    np.random.seed(seed)
    n = len(projects)
    n_eval = int(n * eval_ratio)
    cat_list = sorted(categories)
    cat_to_idx = {c: i for i, c in enumerate(cat_list)}
    n_labels = len(cat_list)
    label_matrix = np.zeros((n, n_labels), dtype=int)
    for i, p in enumerate(projects):
        for lbl in getattr(p, "ground_truth", []):
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
    return eval_projects

# ── Parse LLM response ─────────────────────────────────────────────────────────
def parse_categories_from_response(text: str) -> list:
    if not text:
        return []
    # Find JSON block
    for start in [text.find("{"), text.rfind("{")]:
        if start == -1:
            continue
        end = text.find("}", start)
        if end == -1:
            continue
        try:
            obj = json.loads(text[start:end + 1])
            cats = obj.get("categories", [])
            if isinstance(cats, list):
                return [str(c).strip() for c in cats if c]
        except Exception:
            continue
    return []

# ── Ollama call ─────────────────────────────────────────────────────────────────
import requests

def _auth_headers() -> dict:
    """Return Authorization header if OLLAMA_API_KEY is set."""
    key = os.environ.get("OLLAMA_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

def call_ollama(host: str, model: str, system: str, user: str, temperature: float,
                timeout: int = 600, retries: int = 3):
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,   # ← stream tokens so connection stays alive
        "options": {"temperature": temperature},
    }
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=timeout, stream=True,
                                 headers=_auth_headers())
            resp.raise_for_status()
            content = ""
            tokens = 0
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                delta = chunk.get("message", {}).get("content", "")
                content += delta
                tokens += 1
                if tokens % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"      ⏳ {tokens} tokens, {elapsed:.0f}s...", end="\r", flush=True)
                if chunk.get("done"):
                    break
            latency = time.time() - t0
            return content, None, latency
        except requests.exceptions.Timeout:
            latency = time.time() - t0
            wait = 15 * attempt
            if attempt < retries:
                print(f"      ⏳ Timeout (attempt {attempt}/{retries}), waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                return "", "Request timed out after all retries", latency
        except Exception as e:
            return "", str(e), time.time() - t0
    return "", "Unknown error", 0.0

# ── Run a single variant on test set ───────────────────────────────────────────
def run_variant(variant: dict, test_projects: list, model: str, host: str,
                temperature: float, verbose: bool, limit: int = None,
                timeout: int = 600) -> Path:
    name = variant["name"]
    system_prompt = variant.get("system_prompt", "")
    user_template = variant.get("user_prompt") or variant.get("user_template") or ""
    if not user_template:
        raise ValueError(f"Variant '{name}' has no 'user_prompt' or 'user_template' key. Keys: {list(variant.keys())}")
    
    # ── Include model name in filename e.g. v7_nodesc_five_shot_cot__llama3.1-8b.csv
    model_slug = model.replace(":", "-").replace("/", "-")
    out_csv = RESULTS_DIR / f"{name}__{model_slug}.csv"
    tmp_csv = RESULTS_DIR / f"{name}__{model_slug}.tmp.csv"  # partial progress

    # ── Resume from partial progress ──────────────────────────────────────────
    done_ids = set()
    existing_rows = []
    if out_csv.exists():
        print(f"  ⏭  {name} — already done, skipping (delete to rerun)")
        return out_csv
    if tmp_csv.exists():
        try:
            tmp_df = pd.read_csv(tmp_csv)
            existing_rows = tmp_df.to_dict("records")
            done_ids = set(tmp_df["joss_id"].astype(str).tolist())
            print(f"  ↩  {name} — resuming from {len(done_ids)} already done rows")
        except Exception:
            pass

    projects = test_projects[:limit] if limit else test_projects
    # Skip already-done projects
    projects_todo = [p for p in projects if str(p.joss_id) not in done_ids]
    total = len(projects)
    print(f"\n  Running {name} ({variant['description']}) — {len(projects_todo)}/{total} remaining...")

    rows = list(existing_rows)
    for i, p in enumerate(projects_todo, len(done_ids) + 1):
        deps = ", ".join(str(d) for d in (p.dependencies or [])[:80])
        user_msg = user_template.format(
            title=p.title or "Unknown",
            language=p.language or "Unknown",
            dependencies=deps or "none",
        )
        content, error, latency = call_ollama(host, model, system_prompt, user_msg, temperature, timeout=timeout)
        cats = parse_categories_from_response(content) if not error else []
        success = bool(cats) and not error

        rows.append({
            "joss_id": p.joss_id,
            "title": p.title,
            "language": p.language,
            "predicted_categories": str(cats),
            "ground_truth": str(getattr(p, "ground_truth", [])),
            "success": success,
            "error": error or "",
            "latency_s": round(latency, 2),
            "raw_response": content[:500] if content else "",
        })

        # ── Save progress after every row ─────────────────────────────────────
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(tmp_csv, index=False)

        if verbose or i % 20 == 0:
            status = f"→ {cats}" if success else f"⚠ {error}"
            print(f"    [{i}/{total}] {p.title[:45]:<45} {latency:5.1f}s {status}", flush=True)

    # ── Finalize: rename tmp → final ──────────────────────────────────────────
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    tmp_csv.unlink(missing_ok=True)
    print(f"  ✅ Saved {out_csv} ({sum(r['success'] for r in rows)}/{total} success)")
    return out_csv

# ── Quick metrics from CSV ──────────────────────────────────────────────────────
def compute_metrics_from_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    def parse(raw):
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            val = ast.literal_eval(raw)
            if isinstance(val, (list, tuple)):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
        return []
    df["pred"] = df["predicted_categories"].apply(parse)
    df["gt"]   = df["ground_truth"].apply(parse)
    mask = df["success"] & df["gt"].apply(len).gt(0) & df["pred"].apply(len).gt(0)
    df_eval = df[mask]
    if df_eval.empty:
        return {}
    preds = df_eval["pred"].tolist()
    gts   = df_eval["gt"].tolist()
    m = calculate_metrics(preds, gts)
    m["n_evaluated"] = len(df_eval)
    m["n_total"]     = len(df)
    m["n_failed"]    = len(df) - len(df_eval)
    if "latency_s" in df.columns:
        m["avg_latency_s"] = df["latency_s"].mean()
        m["median_latency_s"] = df["latency_s"].median()
    return m

# ── Server load check ──────────────────────────────────────────────────────
def check_server_load(host: str, model: str) -> tuple:
    """Send a tiny request and measure response time to gauge server load."""
    import time
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": "Reply with one word: ready", "stream": False}
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120,
                             headers=_auth_headers())
        elapsed = time.time() - t0
        if resp.status_code == 200:
            if elapsed < 15:
                return elapsed, "🟢 Free (fast response)"
            elif elapsed < 45:
                return elapsed, "🟡 Moderate load"
            else:
                return elapsed, "🔴 Busy (slow response) — consider waiting"
        else:
            return elapsed, f"⚠ HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return 120, "🔴 Timed out — server very busy"
    except Exception as e:
        return -1, f"❌ Error: {e}"

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--host",        default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit",       type=int,   default=None)
    parser.add_argument("--timeout",     type=int,   default=600, help="Seconds per request (default 600)")
    parser.add_argument("--variant",     default=None, help="Run only this variant name")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--check-load", action="store_true", help="Check server load and exit")
    args = parser.parse_args()

    print(f"Connecting to {args.host}...")
    ok, msg = check_ollama_connection(args.host)
    if not ok:
        print(f"Error: {msg}")
        sys.exit(1)
    print(f"  {msg}")

    # ── Server load check ──────────────────────────────────────────────────────
    if args.check_load:
        print(f"\nChecking server load for {args.model}...")
        elapsed, status = check_server_load(args.host, args.model)
        print(f"  Response time: {elapsed:.1f}s")
        print(f"  Status: {status}")
        print(f"\n  Rule of thumb:")
        print(f"    < 15s  → 🟢 Good time to run experiments")
        print(f"    15-45s → 🟡 Will work but slower")
        print(f"    > 45s  → 🔴 Wait for off-peak (night/weekend)")
        sys.exit(0)

    # ── Load & split ──
    print(f"\nLoading labeled data from {LABELED_CSV}...")
    if not LABELED_CSV.exists():
        print(f"Error: {LABELED_CSV} not found"); sys.exit(1)

    projects = load_projects(LABELED_CSV, limit=None, require_dependencies=True)
    projects = [p for p in projects if getattr(p, "ground_truth", None)]
    print(f"  Total labeled: {len(projects)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load or create the canonical split ──
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as f:
            saved_ids = set(json.load(f))
        test_projects = [p for p in projects if str(p.joss_id) in saved_ids]
        print(f"  ✅ Loaded existing split from {SPLIT_FILE} ({len(test_projects)} projects)")
    else:
        test_projects = stratified_multilabel_split(
            projects, EVAL_SPLIT_RATIO, CATEGORIES, seed=RANDOM_SEED
        )
        split_ids = [str(p.joss_id) for p in test_projects]
        with open(SPLIT_FILE, "w") as f:
            json.dump(split_ids, f, indent=2)
        print(f"  💾 Saved split to {SPLIT_FILE} ({len(test_projects)} projects)")

    # ── Select variants to run ──
    variants = VARIANTS if not args.variant else [VARIANT_MAP[args.variant]]

    print(f"  Test split (30%): {len(test_projects)} projects (seed={RANDOM_SEED})")
    print(f"  This SAME split is used for ALL {len(variants)} variants — fair comparison ✅")

    for variant in variants:
        run_variant(
            variant, test_projects, args.model, args.host,
            args.temperature, args.verbose, args.limit, args.timeout,
        )

    # ── Quick summary — only show variants that have a result file ─────────────
    print(f"\n{'='*95}")
    print(f"  QUICK METRICS SUMMARY")
    print(f"{'='*95}")
    print(f"  {'Variant':<42} {'N':>5} {'Micro F1':>9} {'Macro F1':>9} {'Jaccard':>9} {'Recall':>8} {'AvgLat':>8}")
    print(f"  {'-'*95}")
    model_slug = args.model.replace(":", "-").replace("/", "-")
    for variant in VARIANTS:
        csv = RESULTS_DIR / f"{variant['name']}__{model_slug}.csv"
        if not csv.exists():
            print(f"  {variant['name']:<42} (not run yet)")
            continue
        m = compute_metrics_from_csv(csv)
        if not m:
            print(f"  {variant['name']:<42} (no data)")
            continue
        print(f"  {variant['name']:<42} {m['n_evaluated']:>5} "
              f"{m.get('micro_f1',0):>9.4f} {m.get('macro_f1',0):>9.4f} "
              f"{m.get('jaccard_similarity',0):>9.4f} {m.get('recall',0):>8.4f} "
              f"{m.get('avg_latency_s',0):>7.1f}s")
    print(f"\nAll results in: {RESULTS_DIR}")
    print(f"File pattern : <variant_name>__{model_slug}.csv  (model-namespaced, safe to run multiple models)")


if __name__ == "__main__":
    main()
