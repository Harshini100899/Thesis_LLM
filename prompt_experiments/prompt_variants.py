"""
Prompt variants for ablation study.
All { } in JSON output lines are escaped as {{ }} so .format() works correctly.
"""

VALID_CATEGORIES = (
    "Modeling And Simulation, Data Analytics, Software Analytics, "
    "Integrative Analysis, Hardware, Software, Ui, Process, Ris"
)

# ── Shared category descriptions ────────────────────────────────────────────────
CATEGORY_DESCRIPTIONS = """CATEGORIES AND DEPENDENCY SIGNALS:

1. Modeling And Simulation
   Builds or executes formal models: numerical solvers, ODE/PDE, agent-based, physics/chemical.
   Signals: scipy.optimize, sympy, fenics, pyomo, cvxpy, petsc, deSolve, StanHeaders, rstan, deap, pymoo

2. Data Analytics
   Processes, analyses, or visualises data: statistics, ML, preprocessing.
   Signals: pandas, scikit-learn, statsmodels, xgboost, xarray, dask, nltk, spacy, ggplot2, dplyr, tidyr, caret

3. Software Analytics
   Analyses SOFTWARE artifacts: code metrics, repo mining, static analysis.
   Signals: gitpython, pydriller, radon, lizard, tree-sitter, perceval, bandit, codecarbon
   NOTE: pytest/black/pylint used DURING development → Process, NOT Software Analytics.

4. Integrative Analysis
   Combines MULTIPLE HETEROGENEOUS data sources or model outputs for joint interpretation.
   Signals: intake, pangeo, vtk, pyvista, muon, mofapy2
   NOTE: plotly/networkx alone does NOT qualify.

5. Hardware
   Interfaces with physical devices: sensors, serial/GPIB/SPI, embedded control.
   Signals: pyserial, nidaqmx, pyvisa, smbus2, RPi.GPIO, nifpga, firmata
   NOTE: GPU compute (torch/jax) is NOT hardware.

6. Software
   Reusable library/toolkit/package published for others to build on.
   Signals: setuptools, pybind11, Rcpp, cython — or the project IS a published library.

7. Ui
   Provides a user-facing interface: GUI, web app, dashboard, Shiny, interactive notebook.
   Signals: shiny, dash, streamlit, gradio, PyQt5/6, tkinter, ipywidgets, napari, flask
   NOTE: matplotlib/ggplot2 for static plots alone does NOT qualify.

8. Process
   Supports dev workflow: testing, CI/CD, docs, linting, build.
   Signals: pytest, tox, coverage, sphinx, mkdocs, black, pre-commit, testthat, cmake, gtest
   RULE: Requires 2+ dev/build/test/docs tools.

9. Ris (Research Infrastructure Software)
   Serves a specific scientific domain: domain-specific data handling, experiment support.
   Signals: astropy, biopython, scanpy, pymatgen, rdkit, obspy, nibabel, mne, qiskit, h5py, ase, DESeq2, Seurat
"""

# ── System prompt builders ───────────────────────────────────────────────────────
def make_system(with_desc: bool) -> str:
    role = (
        "You are an expert research software classifier. "
        "Given title, language, and dependencies, assign 1+ categories from the 9 below. "
        "Assign ALL that apply — most projects receive 3–5 labels.\n\n"
    )
    if with_desc:
        return role + CATEGORY_DESCRIPTIONS + f"\nValid categories: {VALID_CATEGORIES}\n"
    else:
        return role + f"Valid categories: {VALID_CATEGORIES}\n"

# ── Examples (all JSON braces escaped as {{ }}) ──────────────────────────────────

# 0-shot: no examples
EXAMPLES_0 = ""

# 1-shot (1 example, no CoT)
EXAMPLES_1_NO_COT = """\
=== EXAMPLES ===

Example 1:
  Title: emg3d: A multigrid solver for 3D electromagnetic diffusion | Language: Python
  Dependencies: h5py, matplotlib, numba, numpy, pytest, scipy, xarray
  Output: {{"categories": ["Modeling And Simulation", "Data Analytics", "Ris", "Software", "Process"]}}
"""

# 3-shot (3 examples, no CoT)
EXAMPLES_3_NO_COT = """\
=== EXAMPLES ===

Example 1:
  Title: emg3d: A multigrid solver for 3D electromagnetic diffusion | Language: Python
  Dependencies: h5py, matplotlib, numba, numpy, pytest, scipy, xarray
  Output: {{"categories": ["Modeling And Simulation", "Data Analytics", "Ris", "Software", "Process"]}}

Example 2:
  Title: measr: Bayesian psychometric measurement using Stan | Language: R
  Dependencies: StanHeaders, rstan, dplyr, tidyr, ggplot2, Rcpp, testthat, knitr, shiny
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Ris", "Software", "Ui", "Process"]}}

Example 3:
  Title: PyDriller: Python Framework for Mining Software Repositories | Language: Python
  Dependencies: gitpython, lizard, pytest, sphinx, click
  Output: {{"categories": ["Software Analytics", "Software", "Process"]}}
"""

# 5-shot (5 examples, no CoT)
EXAMPLES_5_NO_COT = EXAMPLES_3_NO_COT + """\
Example 4:
  Title: covidregionaldata: Subnational data for COVID-19 epidemiology | Language: R
  Dependencies: dplyr, ggplot2, httr, knitr, lubridate, rmarkdown, sf, tidyr, testthat
  Output: {{"categories": ["Data Analytics", "Ris", "Software", "Ui", "Process"]}}

Example 5:
  Title: BrightEyes-MCS: Control software for multichannel scanning microscopy | Language: Python
  Dependencies: nifpga, PySide6, PyQtGraph, h5py, numpy, scipy, pytest
  Output: {{"categories": ["Hardware", "Ris", "Software", "Ui"]}}
"""

# 5-shot WITH chain-of-thought
EXAMPLES_5_COT = """\
=== EXAMPLES ===

Example 1:
  Title: emg3d: A multigrid solver for 3D electromagnetic diffusion | Language: Python
  Dependencies: h5py, matplotlib, numba, numpy, pytest, scipy, xarray
  Reasoning: solver+scipy→Modeling, numpy+xarray→Analytics, electromagnetic→Ris, library→Software, pytest+h5py→Process
  Output: {{"categories": ["Modeling And Simulation", "Data Analytics", "Ris", "Software", "Process"]}}

Example 2:
  Title: measr: Bayesian psychometric measurement using Stan | Language: R
  Dependencies: StanHeaders, rstan, dplyr, tidyr, ggplot2, Rcpp, testthat, knitr, shiny
  Reasoning: rstan+StanHeaders→Modeling, dplyr+ggplot2→Analytics, psychometric→Ris, Rcpp→Software, shiny→Ui, testthat+knitr→Process
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Ris", "Software", "Ui", "Process"]}}

Example 3:
  Title: PyDriller: Python Framework for Mining Software Repositories | Language: Python
  Dependencies: gitpython, lizard, pytest, sphinx, click
  Reasoning: gitpython+lizard→Software Analytics (purpose=analyse code), library→Software, pytest+sphinx→Process
  Output: {{"categories": ["Software Analytics", "Software", "Process"]}}

Example 4:
  Title: covidregionaldata: Subnational data for COVID-19 epidemiology | Language: R
  Dependencies: dplyr, ggplot2, httr, knitr, lubridate, rmarkdown, sf, tidyr, testthat
  Reasoning: dplyr+tidyr→Analytics, COVID domain→Ris, package→Software, knitr+rmarkdown→Ui, testthat+knitr→Process
  Output: {{"categories": ["Data Analytics", "Ris", "Software", "Ui", "Process"]}}

Example 5:
  Title: BrightEyes-MCS: Control software for multichannel scanning microscopy | Language: Python
  Dependencies: nifpga, PySide6, PyQtGraph, h5py, numpy, scipy, pytest
  Reasoning: nifpga→Hardware (FPGA device), microscopy→Ris, PySide6→Ui, reusable→Software, only 1 dev tool→no Process
  Output: {{"categories": ["Hardware", "Ris", "Software", "Ui"]}}
"""

# 7-shot WITH CoT — must match production prompts.py FEW_SHOT_EXAMPLES exactly
EXAMPLES_7_COT = """\
=== EXAMPLES ===

Example 1 (R, 6 labels — covers Ui via shiny, Process via testthat+knitr):
  Title: measr: Bayesian psychometric measurement using Stan
  Language: R
  Dependencies: StanHeaders, rstan, dplyr, tidyr, ggplot2, Rcpp, testthat, knitr, shiny
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Ris", "Software", "Ui", "Process"]}}
  Reasoning: rstan+StanHeaders→Modeling, dplyr+ggplot2→Analytics, psychometric→Ris, Rcpp→Software, shiny→Ui, testthat+knitr→Process

Example 2 (Python, 4 labels — covers Hardware via nifpga):
  Title: BrightEyes-MCS: Control software for multichannel scanning microscopy
  Language: Python
  Dependencies: nifpga, PySide6, PyQtGraph, h5py, numpy, scipy, pytest
  Output: {{"categories": ["Hardware", "Ris", "Software", "Ui"]}}
  Reasoning: nifpga→Hardware, microscopy→Ris, PySide6→Ui, reusable package→Software. Only 1 dev tool (pytest), so no Process.

Example 3 (Python, 3 labels — covers Software Analytics):
  Title: PyDriller: Python Framework for Mining Software Repositories
  Language: Python
  Dependencies: gitpython, lizard, pytest, sphinx, click
  Output: {{"categories": ["Software Analytics", "Software", "Process"]}}
  Reasoning: Mining Software Repositories+gitpython+lizard→Software Analytics, library→Software, pytest+sphinx→Process

Example 4 (Python, 5 labels — covers Integrative Analysis):
  Title: Jupyter Scatter: Interactive Exploration of Large-Scale Datasets
  Language: Python
  Dependencies: anywidget, numpy, pandas, scikit-learn, pytest, sphinx
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Software", "Ui", "Process"]}}
  Reasoning: pandas+sklearn→Analytics, interactive widget for multi-source exploration→Integrative Analysis+Ui, library→Software, pytest+sphinx→Process

Example 5 (Python, 5 labels — standard scientific Python package):
  Title: emg3d: A multigrid solver for 3D electromagnetic diffusion
  Language: Python
  Dependencies: h5py, matplotlib, numba, numpy, pytest, scipy, xarray
  Output: {{"categories": ["Modeling And Simulation", "Data Analytics", "Ris", "Software", "Process"]}}
  Reasoning: solver+scipy→Modeling, numpy+xarray→Analytics, electromagnetic→Ris, library→Software, pytest+h5py→Process

Example 6 (R, 5 labels — typical R data package):
  Title: covidregionaldata: Subnational data for COVID-19 epidemiology
  Language: R
  Dependencies: dplyr, ggplot2, httr, jsonlite, knitr, lubridate, memoise, purrr, rlang, rmarkdown, sf, tidyr, testthat
  Output: {{"categories": ["Data Analytics", "Ris", "Software", "Ui", "Process"]}}
  Reasoning: dplyr+tidyr→Analytics, COVID epidemiology→Ris, package→Software, knitr+rmarkdown→Ui, testthat+knitr→Process

Example 7 (C++, 4 labels — HPC simulation without Hardware):
  Title: Cabana: A Performance Portable Library for Particle-Based Simulations
  Language: C++
  Dependencies: Kokkos, HDF5, GTest, Doxygen, cmake
  Output: {{"categories": ["Modeling And Simulation", "Ris", "Software", "Process"]}}
  Reasoning: Particle simulations+Kokkos→Modeling, physics domain→Ris, library→Software, GTest+cmake+Doxygen→Process. No hardware interfacing despite HPC.
"""

# Derive 5-shot CoT from first 5 examples of EXAMPLES_7_COT
EXAMPLES_5_COT = "\n".join(
    EXAMPLES_7_COT.split("\n\n")[:6]  # header + 5 examples
) + "\n"

EXAMPLES_5_COT_NO_DESC = EXAMPLES_5_COT  # same examples, system differs

# 5-shot, same examples as v7, NO CoT reasoning lines — clean CoT ablation partner for v7
def _strip_reasoning(examples_str: str) -> str:
    """Remove 'Reasoning: ...' lines from CoT examples."""
    lines = [l for l in examples_str.splitlines() if not l.strip().startswith("Reasoning:")]
    return "\n".join(lines) + "\n"

EXAMPLES_5_COT_NO_REASONING = _strip_reasoning(EXAMPLES_5_COT)
EXAMPLES_7_COT_NO_REASONING = _strip_reasoning(EXAMPLES_7_COT)

# 10-shot WITH CoT, no descriptions — extends EXAMPLES_7_COT with 3 more
EXAMPLES_10_COT = EXAMPLES_7_COT + """\
Example 8 (Python, 4 labels — covers Data Analytics + Ui via dashboard):
  Title: Lumen: A framework for building data-driven dashboards
  Language: Python
  Dependencies: panel, param, pandas, bokeh, pytest, sphinx
  Output: {{"categories": ["Data Analytics", "Software", "Ui", "Process"]}}
  Reasoning: pandas→Analytics, panel+bokeh→Ui (interactive dashboard), reusable framework→Software, pytest+sphinx→Process. No domain signal→no Ris.

Example 9 (Python, 3 labels — covers Process-heavy dev tool):
  Title: pytest-benchmark: A pytest fixture for benchmarking code
  Language: Python
  Dependencies: pytest, setuptools, sphinx, coverage, tox
  Output: {{"categories": ["Software", "Process"]}}
  Reasoning: Benchmarking tool for developers→Process, pytest+tox+coverage+sphinx→confirms Process, reusable plugin→Software. No data, no domain→nothing else.

Example 10 (Julia, 5 labels — HPC with Integrative Analysis):
  Title: AMDGPU.jl: AMD GPU programming in Julia
  Language: Julia
  Dependencies: LLVM, Documenter, Test, LinearAlgebra, Statistics
  Output: {{"categories": ["Hardware", "Software", "Process"]}}
  Reasoning: GPU programming→Hardware (device interfacing), reusable library→Software, Test+Documenter→Process. LinearAlgebra is stdlib not domain signal→no Ris.
"""

# ── Output format ────────────────────────────────────────────────────────────────
OUTPUT_FORMAT = 'Return ONLY valid JSON: {{"categories": ["Cat1", "Cat2", ...]}}'

# ── User template — {title}, {language}, {dependencies} are filled at runtime ───
# {examples} and {output_format} are filled when building VARIANTS
USER_TEMPLATE = """{examples}
=== CLASSIFY THIS SOFTWARE ===

Title: {{title}}
Language: {{language}}
Dependencies: [{{dependencies}}]

{output_format}
"""

# ── Variant definitions ──────────────────────────────────────────────────────────
VARIANTS = [
    {
        "name": "v0_zero_no_desc",
        "description": "0-shot, no category descriptions",
        "system_prompt": make_system(with_desc=False),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_0, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v1_zero_with_desc",
        "description": "0-shot, with category descriptions",
        "system_prompt": make_system(with_desc=True),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_0, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v2_one_shot_no_cot",
        "description": "1-shot, no chain-of-thought, with descriptions",
        "system_prompt": make_system(with_desc=True),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_1_NO_COT, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v4_five_shot_no_cot",
        "description": "5-shot, no chain-of-thought, with descriptions",
        "system_prompt": make_system(with_desc=True),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_NO_COT, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v5_five_shot_cot",
        "description": "5-shot, with chain-of-thought, with descriptions",
        "system_prompt": make_system(with_desc=True),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_COT, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v6_seven_shot_cot",
        "description": "7-shot, with chain-of-thought, with descriptions (production prompt)",
        "system_prompt": make_system(with_desc=True),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_COT, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v7_nodesc_five_shot_cot",
        "description": "5-shot, with chain-of-thought, NO descriptions",
        "system_prompt": make_system(with_desc=False),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_COT_NO_DESC, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v8_ten_shot_cot_no_desc",
        "description": "10-shot, with chain-of-thought, NO descriptions — extends v7",
        "system_prompt": make_system(with_desc=False),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_10_COT, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v9_seven_shot_cot_no_desc",
        "description": "7-shot, with chain-of-thought, NO descriptions — clean comparison to v6",
        "system_prompt": make_system(with_desc=False),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_COT, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v10_five_shot_no_cot_no_desc",
        "description": "5-shot, NO chain-of-thought, NO descriptions — clean CoT ablation vs v7 (same examples)",
        "system_prompt": make_system(with_desc=False),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_COT_NO_REASONING, output_format=OUTPUT_FORMAT),
    },
    {
        "name": "v11_seven_shot_no_cot_no_desc",
        "description": "7-shot, NO chain-of-thought, NO descriptions — clean CoT ablation vs v9 (same examples)",
        "system_prompt": make_system(with_desc=False),
        "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_COT_NO_REASONING, output_format=OUTPUT_FORMAT),
    },
]

VARIANT_MAP = {v["name"]: v for v in VARIANTS}

if __name__ == "__main__":
    # Quick sanity check — try formatting each template
    test = {"title": "Test", "language": "Python", "dependencies": "numpy, scipy"}
    print("Sanity check — all variants format without error:\n")
    for v in VARIANTS:
        try:
            msg = v["user_template"].format(**test)
            print(f"  ✅ {v['name']:<35} ({v['description']})")
        except KeyError as e:
            print(f"  ❌ {v['name']:<35} KeyError: {e}")
