"""Prompt definitions for research software classification.

Follows prompt engineering best practices:
  1. Clear role + task definition (system prompt)
  2. No category descriptions — CoT examples provide implicit decision rules (v7)
  3. Well-calibrated 5-shot examples WITH chain-of-thought reasoning
  4. Concise, positive framing — no contradictory rules
  5. Explicit output format constraint

Classification is based on: title, programming language, and dependencies.
Categories follow: Hasselbring et al. (2025) Multi-Dimensional Categorization.
Prompt variant: v7_nodesc_five_shot_cot
  → Best ablation result: Micro F1=0.785, Macro F1=0.594, Recall=0.801 (n=461)
"""

from config import CATEGORIES

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Role + Task + Category Names ONLY (no descriptions)
# v7 finding: removing descriptions improves Macro F1 (+0.044) and Recall (+0.038)
# CoT examples embed decision rules implicitly → descriptions add noise
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert research software classifier with deep knowledge of scientific computing ecosystems.

TASK: Given a software project's title, programming language, and dependency list, assign 1 or more of the 9 categories below. Use the dependency signals and title keywords as primary evidence.

VALID CATEGORIES:
1. Modeling And Simulation
2. Data Analytics
3. Software Analytics
4. Integrative Analysis
5. Hardware
6. Software
7. Ui
8. Process
9. Ris

RULES:
- Assign ALL categories that apply. Most JOSS projects receive 3–5 labels.
- Use dependency signals as primary evidence; title provides domain context.
- Return ONLY valid JSON: {"categories": ["Cat1", "Cat2", ...]}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES — 5-shot WITH chain-of-thought (v7 configuration)
# Prompt Engineering: Reasoning before Output — model justifies before deciding
# Coverage: Hardware, Ris, Modeling, Analytics, Software Analytics, Ui, Process
# ═══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = """
=== EXAMPLES ===

Example 1:
  Title: emg3d: A multigrid solver for 3D electromagnetic diffusion
  Language: Python
  Dependencies: h5py, matplotlib, numba, numpy, pytest, scipy, xarray
  Reasoning: solver+scipy→Modeling, numpy+xarray→Analytics, electromagnetic→Ris, library→Software, pytest+h5py (multiple tools)→Process
  Output: {"categories": ["Modeling And Simulation", "Data Analytics", "Ris", "Software", "Process"]}

Example 2:
  Title: measr: Bayesian psychometric measurement using Stan
  Language: R
  Dependencies: StanHeaders, rstan, dplyr, tidyr, ggplot2, Rcpp, testthat, knitr, shiny
  Reasoning: rstan+StanHeaders→Modeling, dplyr+ggplot2→Analytics, psychometric→Ris, Rcpp→Software, shiny→Ui, testthat+knitr→Process
  Output: {"categories": ["Data Analytics", "Modeling And Simulation", "Ris", "Software", "Ui", "Process"]}

Example 3:
  Title: PyDriller: Python Framework for Mining Software Repositories
  Language: Python
  Dependencies: gitpython, lizard, pytest, sphinx, click
  Reasoning: "Mining Software Repositories"+gitpython+lizard→Software Analytics (purpose=analyse code, not scientific data), library→Software, pytest+sphinx→Process
  Output: {"categories": ["Software Analytics", "Software", "Process"]}

Example 4:
  Title: covidregionaldata: Subnational data for COVID-19 epidemiology
  Language: R
  Dependencies: dplyr, ggplot2, httr, jsonlite, knitr, lubridate, memoise, purrr, rlang, rmarkdown, sf, tidyr, testthat
  Reasoning: dplyr+tidyr→Analytics, COVID epidemiology→Ris, package→Software, knitr+rmarkdown→Ui, testthat+knitr→Process
  Output: {"categories": ["Data Analytics", "Ris", "Software", "Ui", "Process"]}

Example 5:
  Title: BrightEyes-MCS: Control software for multichannel scanning microscopy
  Language: Python
  Dependencies: nifpga, PySide6, PyQtGraph, h5py, numpy, scipy, pytest
  Reasoning: nifpga→Hardware (FPGA device interface), microscopy→Ris, PySide6→Ui, reusable package→Software, only 1 dev tool (pytest)→no Process
  Output: {"categories": ["Hardware", "Ris", "Software", "Ui"]}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION TEMPLATE — User prompt structure
# Prompt Engineering: Clear input demarcation + output constraint
# ═══════════════════════════════════════════════════════════════════════════════

CLASSIFICATION_TEMPLATE = """{few_shot_examples}
=== CLASSIFY THIS SOFTWARE ===

Title: {title}
Language: {language}
Dependencies: [{dependencies}]

Return ONLY valid JSON: {{"categories": ["Cat1", "Cat2", ...]}}
"""


def build_system_prompt() -> str:
    """Return the system prompt (v7: no category descriptions)."""
    return SYSTEM_PROMPT


def build_user_prompt(
    title: str,
    language: str,
    dependencies: list = None,
    use_few_shot: bool = True,
) -> str:
    """Build classification prompt from title, language, and dependencies ONLY.

    Our research approach classifies using only:
    - Title
    - Programming language
    - Dependency list
    No README, documentation, or source code is used.

    Prompt variant: v7_nodesc_five_shot_cot
    - use_few_shot=True  → 5-shot CoT examples included (default, recommended)
    - use_few_shot=False → zero-shot fallback
    """
    if dependencies is None:
        dependencies = []

    deps_str = ", ".join(str(d) for d in dependencies[:80])
    if len(dependencies) > 80:
        deps_str += f" ... and {len(dependencies) - 80} more"

    return CLASSIFICATION_TEMPLATE.format(
        few_shot_examples=FEW_SHOT_EXAMPLES if use_few_shot else "",
        title=title or "Unknown",
        language=language or "Unknown",
        dependencies=deps_str or "none found",
    )