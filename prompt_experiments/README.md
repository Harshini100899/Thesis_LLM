# Prompt Variant Experiments

Ablation study for thesis Section: *Prompt Engineering Evaluation*.

## Design — 16-Variant Grid (was 28, reduced for practicality)

| Axis | Fixed | Varying | # Variants |
|------|-------|---------|-----------|
| A — Shot count | CoT+desc, no sculpting | 0 / 5 / 7 / 10 | 4 |
| B — CoT | 7-shot, no sculpting | no-CoT×no-desc / no-CoT×desc / CoT×no-desc | 3 |
| C — 5-shot no-CoT | no sculpting | no-desc / desc | 2 |
| D — Sculpting | CoT+desc | 0 / 5 / 7 / 10 shot | 4 + 3 intermediate | 7 |
| **Total** | | | **16** |

Est. cost: **16 × ~461 ≈ 7,376 LLM calls** (vs 12,908 for 28 variants).

Key comparison pairs (sculpting ablation):

| No Sculpting | With Sculpting |
|---|---|
| v02_seven_cot_desc ← production | v13sc_seven_cot_desc ← Ccot flagship |
| v06_seven_cot_no_desc | v12sc_seven_cot_no_desc |
| v05_seven_no_cot_desc | v11sc_seven_no_cot_desc |
| v03_ten_cot_desc | v14sc_ten_cot_desc |

## Run

```bash


# Run all 16 variants (results saved as <variant>__llama3.3-70b.csv — never overwrites)
python prompt_experiments/run_experiments.py --model llama3.3:70b

# Run one variant only
python prompt_experiments/run_experiments.py --model llama3.3:70b --variant v02_seven_cot_desc

# Quick test (5 projects per variant)
python prompt_experiments/run_experiments.py --model llama3.3:70b --limit 5

# Compare a second model without touching first model's results
python prompt_experiments/run_experiments.py --model llama3.1:8b

# Analyze results
python prompt_experiments/analyze_results.py
```



```bash
python prompt_experiments/run_experiments.py --model llama3.3:70b

```

## Output — Model-Namespaced Files (no accidental overwrites)

```
prompt_experiments/results/
├── test_split_ids.json                    ← canonical split, shared by ALL runs
├── v02_seven_cot_desc__llama3.3-70b.csv
├── v13sc_seven_cot_desc__llama3.3-70b.csv
├── v02_seven_cot_desc__llama3.1-8b.csv   ← different model, different file
├── ...
└── summary_metrics.csv
```