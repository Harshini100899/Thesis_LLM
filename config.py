"""Configuration and constants for the classifier."""

from pathlib import Path

# ─── File Paths ───────────────────────────────────────────────────────────────
# Full dataset for inference (title, language, dependencies)
INFERENCE_CSV = Path("joss_all_language_depends.csv")
# Labeled dataset for evaluation (has dependency_labels column)
LABELED_CSV = Path("joss_all_with_dependency_labels1.csv")
# Output files
TEST_RESULTS_CSV = Path("test_results.csv")
PREDICTION_RESULTS_CSV = Path("prediction_results.csv")

# ─── Ollama Settings ─────────────────────────────────────────────────────────
DEFAULT_OLLAMA_HOST = "https://ai.compute.isst.fraunhofer.de"
DEFAULT_OLLAMA_MODEL = "llama3.3:70b"

# ─── Model Settings ──────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 600  # 10 min for 70B model

# ─── Evaluation Settings ─────────────────────────────────────────────────────
EVAL_SPLIT_RATIO = 0.30  # 30% of labeled data for evaluation
RANDOM_SEED = 42

# ─── 9 Categories (matching thesis definitions) ──────────────────────────────
CATEGORIES = [
    "Modeling And Simulation",
    "Data Analytics",
    "Software Analytics",
    "Integrative Analysis",
    "Hardware",
    "Software",
    "Ui",
    "Process",
    "Ris",
]
