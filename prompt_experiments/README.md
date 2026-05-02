# Prompt Variant Experiments

Ablation study for thesis Section: *Prompt Engineering Evaluation*.

## Variants

| Variant | Examples | CoT | Category Descriptions |
|---------|----------|-----|-----------------------|
| v0_zero_no_desc | 0 | ✗ | ✗ |
| v1_zero_with_desc | 0 | ✗ | ✓ |
| v2_one_shot_no_cot | 1 | ✗ | ✓ |
| v3_three_shot_no_cot | 3 | ✗ | ✓ |
| v4_five_shot_no_cot | 5 | ✗ | ✓ |
| v5_five_shot_cot | 5 | ✓ | ✓ |
| v6_seven_shot_cot | 7 | ✓ | ✓ ← current best |
| v7_nodesc_five_shot_cot | 5 | ✓ | ✗ |

All variants use the **same 30% stratified test split** (seed=42, ~461 projects).

## Run

```bash
cd /home/coder/llm

# Run all variants sequentially (~8 × 461 calls)
python prompt_experiments/run_experiments.py --model llama3.3:70b

# Run one variant only
python prompt_experiments/run_experiments.py --model llama3.3:70b --variant v5_five_shot_cot

# Quick test (5 projects per variant)
python prompt_experiments/run_experiments.py --model llama3.3:70b --limit 5

# Analyze results (after all variants run)
python prompt_experiments/analyze_results.py
```

## Output

```
prompt_experiments/
├── results/
│   ├── v0_zero_no_desc.csv
│   ├── v1_zero_with_desc.csv
│   ├── ...
│   ├── v7_nodesc_five_shot_cot.csv
│   └── summary_metrics.csv       ← copy this to thesis
└── plots/
    ├── variant_comparison_micro_f1.png
    └── perlabel_f1_heatmap.png
```

## Thesis Write-Up Guidance

- **Table**: `summary_metrics.csv` → Table X: Prompt Variant Comparison
- **Finding 1**: 0-shot vs few-shot gap (v0 vs v4) → value of demonstrations
- **Finding 2**: CoT effect (v4 vs v5) → value of reasoning traces
- **Finding 3**: Descriptions effect (v7 vs v5) → value of category definitions
- **Finding 4**: Shot count effect (v2→v3→v4→v5) → diminishing returns
- **Selected prompt**: v6 (best Micro F1 on validation → used for final test)
