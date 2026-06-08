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
| v02_seven_cot_desc ← production | v13sc_seven_cot_desc ← sculpted flagship |
| v06_seven_cot_no_desc | v12sc_seven_cot_no_desc |
| v05_seven_no_cot_desc | v11sc_seven_no_cot_desc |
| v03_ten_cot_desc | v14sc_ten_cot_desc |

## Run

```bash
cd /home/coder/llm

# Check server load first
python prompt_experiments/run_experiments.py --model llama3.3:70b --check-load

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

## Copy & Run (full workflow)

```bash
# 1. Copy updated files to server (run on local PC)
scp -r c:\Users\eggoni\Desktop\llm\prompt_experiments\*.py coder.LLMtune:/home/coder/llm/prompt_experiments/
scp c:\Users\eggoni\Desktop\llm\prompt_experiments\get_token.sh coder.LLMtune:/home/coder/llm/prompt_experiments/

# 2. SSH into server
coder ssh LLMtune
cd /home/coder/llm
source .venv/bin/activate

# 3. Get bearer token (device flow — opens a browser URL, valid ~1 hour)
bash prompt_experiments/get_token.sh
# Open the printed URL, enter the code, then run:
export OLLAMA_API_KEY=<paste_token_here>

# 4. Run experiments (token picked up from env var)
python prompt_experiments/run_experiments.py --model llama3.3:70b

# Or pass token explicitly
python prompt_experiments/run_experiments.py --model llama3.3:70b --api-key $OLLAMA_API_KEY

# 5. Copy results back to local PC
scp -r coder.LLMtune:/home/coder/llm/prompt_experiments/results/ "C:\Users\eggoni\Desktop\llm\prompt_experiments\results\"
scp -r coder.LLMtune:/home/coder/llm/prompt_experiments/plots/ "C:\Users\eggoni\Desktop\llm\prompt_experiments\plots\"
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