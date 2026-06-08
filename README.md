# LLM Research Software Classifier

Classifies JOSS research software repositories into 9 categories using an LLM (Ollama) based on **title, programming language, and dependencies**.

## Categories

| Category | Description |
|----------|-------------|
| **Modeling And Simulation** | Numerical solvers, agent-based models, ODE/PDE simulators |
| **Data Analytics** | Statistical analysis, ML pipelines, data processing |
| **Software Analytics** | Code analysis, repository mining, software quality tools |
| **Integrative Analysis** | Multi-source integration, data assimilation, visualization |
| **Hardware** | Embedded software, device drivers, hardware control |
| **Software** | OS components, compilers, runtimes, system utilities |
| **Ui** | GUIs, web frontends, dashboards, interactive interfaces |
| **Process** | Workflow automation, CI/CD, build/deploy systems |
| **Ris** | Experiment control, data management, HPC, lab tools |

## How to Run

### Option 1: Running on Google Colab (Recommended for GPU Access)

1. Upload the project folder (`src/`, `scripts/`, `datasets/`, `requirements.txt`) to your Google Drive.
2. Open a Jupyter notebook in Google Colab and change the runtime to a **GPU runtime** (e.g., T4, L4, or A100).
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Navigate to the project directory and install dependencies:
   ```bash
   %cd /content/drive/MyDrive/llm
   !pip install -r requirements.txt
   ```
5. Run the classification script:
   ```bash
   # Quick test (5 projects only)
   !python scripts/classify_projects.py --model llama3.3:70b --limit 5 --verbose
   
   # Full run
   !python scripts/classify_projects.py --model llama3.3:70b --verbose
   ```

### Option 2: Running Locally or on a Custom Server

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Test Ollama API server connection:
   ```bash
   python scripts/diagnose_timeout.py
   ```
3. Run classification:
   ```bash
   python scripts/classify_projects.py --model llama3.3:70b --limit 5 --verbose
   ```

## Expected Output

```
============================================================
Step 1: Loading labeled data from joss_all_with_dependency_labels1.csv
============================================================
  Labeled projects with ground truth: ~1100
  Evaluation subset (30%): ~330
  Inference subset (70%): ~770

============================================================
Step 2: Classifying 330 evaluation projects
============================================================
  [1/330] pySYD: Automated measurements...
    → ['Modeling And Simulation', 'Data Analytics', 'Ris']
  ...

============================================================
Step 3: Evaluation Metrics (330 projects)
============================================================
  Sample-level metrics:
    Jaccard Similarity: 0.XXXX
    Precision:          0.XXXX
    Recall:             0.XXXX
    F1 Score:           0.XXXX
    Exact Match Ratio:  0.XXXX
    Hamming Loss:       0.XXXX

====================================================================================================
  Per-Label Classification Metrics
====================================================================================================
  Category                     Precision     Recall   F1 Score    Hamming  ExactMatch    Support
  --------------------------------------------------------------------------------------------------
  Data Analytics                  0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Modeling And Simulation         0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Software                        0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Ris                             0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Process                         0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Ui                              0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Hardware                        0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Software Analytics              0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  Integrative Analysis            0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
  --------------------------------------------------------------------------------------------------
  Weighted Average                0.XXXX     0.XXXX     0.XXXX     0.XXXX       0.XXXX       XXX
====================================================================================================
```

## Project Structure

```
├── src/                    # Core Python package / module files
│   ├── config.py           # Configuration parameters and category definitions
│   ├── prompts.py          # LLM system and user prompt builders
│   ├── categories.py       # Category constants metadata
│   ├── metrics.py          # Performance metrics calculation (F1, Jaccard, etc.)
│   ├── utils.py            # Diagnostic connection utilities
│   ├── ground_truths.py    # Raw dependencies file loader
│   ├── data/               # Project data loader package
│   └── classifiers/        # Ollama API classifier package
├── scripts/                # Execution, utility, and setup scripts
│   ├── classify_projects.py
│   ├── evaluate_all.py
│   ├── compare_models.py
│   └── ...
├── datasets/               # Labeled and inference datasets (.csv)
├── results/                # Output classification results (.csv)
├── plots/                  # Graph outputs and comparisons (.png)
├── logs/                   # System runtime logs (.log)
├── prompt_experiments/     # Ablation study sub-experiments
├── README.md
└── requirements.txt
```
```
qwen3.5:122b, embeddinggemma:300m, deepseek-r1:70b, llama3.3:70b, llama3.1:8b, nomic-embed-text:latest, gemma3:1b, gemma3:27b
  Model: llama3.3:70b
    Label distribution:
    Software                      1360 (87.2%)
    Data Analytics                1262 (80.9%)
    Ris                           1228 (78.8%)
    Modeling And Simulation       1094 (70.2%)
    Process                        826 (53.0%)
    Ui                             670 (43.0%)
    Software Analytics             439 (28.2%)
    Hardware                        90 (5.8%)
    Integrative Analysis            43 (2.8%)
    
  Performing stratified multi-label split (eval=30%)...
  Evaluation subset (30%): 461
  Inference subset (70%): 1098

  Stratification check (eval % vs overall %):
    Software                     overall= 87.2%  eval= 88.5%  diff=1.3% ✅
    Data Analytics               overall= 80.9%  eval= 82.2%  diff=1.3% ✅
    Ris                          overall= 78.8%  eval= 79.8%  diff=1.1% ✅
    Modeling And Simulation      overall= 70.2%  eval= 71.1%  diff=1.0% ✅
    Process                      overall= 53.0%  eval= 53.8%  diff=0.8% ✅
    Ui                           overall= 43.0%  eval= 43.6%  diff=0.6% ✅
    Software Analytics           overall= 28.2%  eval= 28.6%  diff=0.5% ✅
    Hardware                     overall=  5.8%  eval=  5.9%  diff=0.1% ✅
    Integrative Analysis         overall=  2.8%  eval=  2.8%  diff=0.1% ✅

    # Running on Google Colab:
    # 1. Open Google Colab and configure GPU runtime.
    # 2. Upload this directory (src/, scripts/, datasets/, requirements.txt) to Google Drive.
    # 3. Mount Google Drive and run:
    #    %cd /content/drive/MyDrive/llm
    #    !pip install -r requirements.txt
    #    !python scripts/classify_projects.py --model llama3.3:70b --limit 5 --verbose
