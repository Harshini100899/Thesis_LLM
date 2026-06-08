"""
Prompt variants for ablation study.
Axes: shots (0/5/7/10) × CoT (no/yes) × descriptions (no/yes) × sculpting (no/yes)

All examples are verified against joss_all_with_dependency_labels1.csv ground truth.
All { } in JSON output lines are escaped as {{ }} so .format() works correctly.
"""

VALID_CATEGORIES = (
    "Modeling And Simulation, Data Analytics, Software Analytics, "
    "Integrative Analysis, Hardware, Software, Ui, Process, Ris"
)

# ── Shared category descriptions ─────────────────────────────────────────────────
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

# ── Sculpting rules: hard NOT/ONLY-IF boundaries ─────────────────────────────────
# Targeted at the lowest-scoring categories in stratification:
#   Process (53%), Ui (43%), Software Analytics (28%), Hardware (5.8%), Integrative Analysis (2.8%)
SCULPTING_RULES = """
HARD BOUNDARY RULES — apply these before assigning any label:

PROCESS — assign ONLY when 2+ distinct dev/build/test/docs tools are present:
  ASSIGN   pytest + sphinx, tox + coverage, cmake + gtest + doxygen
  NEVER    pytest alone, a single linter, or a single doc tool
  NEVER    pytest/black/pylint that analyse others' code → that is Software Analytics

HARDWARE — assign ONLY when the code interfaces with a physical device:
  ASSIGN   pyserial, nidaqmx, pyvisa, RPi.GPIO, smbus2, nifpga, firmata
  NEVER    torch, jax, cupy, numba (GPU compute is NOT hardware)
  NEVER    HPC / parallel computing frameworks alone (Kokkos, MPI, OpenMP)

UI — assign ONLY when the project provides an interactive user-facing interface:
  ASSIGN   shiny, dash, streamlit, gradio, PyQt5/6, tkinter, ipywidgets, napari, flask
  NEVER    matplotlib or ggplot2 used only for static plots
  NEVER    knitr or rmarkdown used only for static reports/notebooks

INTEGRATIVE ANALYSIS — assign ONLY when MULTIPLE HETEROGENEOUS sources are joined:
  ASSIGN   intake + vtk, pangeo + muon, multi-modal fusion pipelines
  NEVER    plotly or networkx used alone for visualisation
  NEVER    pandas + numpy operating on a single homogeneous dataset

SOFTWARE ANALYTICS — assign ONLY when the project analyses SOFTWARE ARTIFACTS:
  ASSIGN   gitpython, pydriller, radon, lizard, tree-sitter, perceval
  NEVER    analysing scientific/experimental data (→ Data Analytics or Ris instead)
  NEVER    dev tools used inside the project's own pipeline (→ Process)

RIS — assign ONLY when a specific scientific/research domain is the subject:
  ASSIGN   astropy (astronomy), biopython (biology), rdkit (chemistry), qiskit (quantum)
  NEVER    generic data tools with no domain focus
  NEVER    infer Ris from "data" or "analytics" in the project title alone
"""

# ── System prompt builders ────────────────────────────────────────────────────────
def make_system(with_desc: bool, with_sculpting: bool = False) -> str:
    n_cats = 9
    role = (
        "You are an expert research software classifier. "
        f"Given title, language, and dependencies, assign 1+ categories from the {n_cats} below. "
        "Assign ALL that apply — most projects receive 3–5 labels.\n\n"
    )
    body = ""
    if with_desc:
        body += CATEGORY_DESCRIPTIONS + "\n"
    if with_sculpting:
        body += SCULPTING_RULES + "\n"
    return role + body + f"Valid categories: {VALID_CATEGORIES}\n"

# ── Examples (all JSON braces escaped as {{ }}) ───────────────────────────────────
# Selection criteria (prompt engineering best practice):
#   1. Label coverage  — every category appears in at least one example
#   2. Language diversity — Python / R / C++ / C
#   3. Contrastive pairs — explicit ABSENT label reasoning teaches boundaries
#   4. Cardinality range — 2-to-8 labels represented
#   5. Ground truth verified — every output matches CSV dependency_labels column

EXAMPLES_0 = ""

# ── 5-shot no-CoT (compact inline format) ────────────────────────────────────────
# Verified JOSS IDs: 5742, 5662, 3290, 7302, 8504
EXAMPLES_5_NO_COT = """\
=== EXAMPLES ===

Example 1:
  Title: measr: Bayesian psychometric measurement using Stan | Language: R
  Dependencies: StanHeaders, rstan, dplyr, tidyr, Rcpp, testthat, knitr, rmarkdown
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Modeling And Simulation", "Process", "Ris", "Software", "Ui"]}}

Example 2:
  Title: PYDAQ: Data Acquisition and Experimental Analysis with Python | Language: Python
  Dependencies: PySide6, nidaqmx, pyserial, matplotlib, numpy
  Output: {{"categories": ["Data Analytics", "Hardware", "Modeling And Simulation", "Ris", "Software", "Ui"]}}

Example 3:
  Title: covidregionaldata: Subnational data for COVID-19 epidemiology | Language: R
  Dependencies: dplyr, ggplot2, httr, knitr, lubridate, purrr, rmarkdown, sf, tidyr, testthat
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Process", "Ris", "Software"]}}

Example 4:
  Title: EXP: a Python/C++ package for basis function expansion methods in galactic dynamics | Language: C++
  Dependencies: CUDAToolkit, Doxygen, Eigen3, HDF5, MPI, OpenMP, pybind11, VTK
  Output: {{"categories": ["Data Analytics", "Hardware", "Modeling And Simulation", "Ris", "Software", "Software Analytics"]}}

Example 5:
  Title: PyBispectra: A toolbox for advanced electrophysiological signal processing | Language: Python
  Dependencies: mne, scipy, numba, numpy, ipywidgets, pytest, sphinx, pre-commit, coverage
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Process", "Ris", "Software", "Software Analytics", "Ui"]}}
"""

# ── 5-shot WITH CoT ───────────────────────────────────────────────────────────────
# Verified JOSS IDs: 5742, 5662, 3290, 7302, 8504
# Covers: all 9 categories | Languages: R x2, Python x2, C++ x1
# Contrastive: Ex3 shows rmarkdown ≠ Ui; Ex4 shows CUDAToolkit = Hardware in C++;
#              Ex2 shows no Process when <2 dev tools present
EXAMPLES_5_COT = """\
=== EXAMPLES ===

Example 1 (R, 7 labels — Bayesian stats; rmarkdown=Ui, testthat+knitr=Process, rstan=Modeling):
  Title: measr: Bayesian psychometric measurement using Stan
  Language: R
  Dependencies: StanHeaders, rstan, dplyr, tidyr, Rcpp, testthat, knitr, rmarkdown
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Modeling And Simulation", "Process", "Ris", "Software", "Ui"]}}
  Reasoning: rstan and StanHeaders signal Modeling And Simulation. dplyr and tidyr signal Data Analytics. The package integrates psychometric and measurement data from multiple sources so Integrative Analysis applies. The psychometric measurement domain signals Ris. Rcpp signals Software. rmarkdown provides interactive report rendering so Ui applies. testthat and knitr together (2+ tools) signal Process.

Example 2 (Python, 6 labels — physical instrument control; NO Process because <2 dev tools):
  Title: PYDAQ: Data Acquisition and Experimental Analysis with Python
  Language: Python
  Dependencies: PySide6, nidaqmx, pyserial, matplotlib, numpy, packaging, shiboken6
  Output: {{"categories": ["Data Analytics", "Hardware", "Modeling And Simulation", "Ris", "Software", "Ui"]}}
  Reasoning: nidaqmx and pyserial signal Hardware because they interface with physical data acquisition instruments. PySide6 signals Ui. Physical measurement and system identification signals Modeling And Simulation and Ris. matplotlib and numpy signal Data Analytics. The project is a reusable library so Software applies. No dev/build/test tools are listed so Process does not qualify.

Example 3 (R, 5 labels — epidemiology data package; rmarkdown = static reporting NOT interactive Ui):
  Title: covidregionaldata: Subnational data for COVID-19 epidemiology
  Language: R
  Dependencies: dplyr, ggplot2, httr, knitr, lubridate, purrr, rmarkdown, sf, tidyr, testthat
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Process", "Ris", "Software"]}}
  Reasoning: dplyr and tidyr signal Data Analytics. The package integrates subnational COVID-19 data from multiple heterogeneous sources so Integrative Analysis applies. COVID-19 epidemiology is a specific scientific domain so Ris applies. The project is a reusable package so Software applies. testthat and knitr together (2+ tools) signal Process. rmarkdown is used for static reports only, not an interactive user-facing interface, so Ui does not apply.

Example 4 (C++, 6 labels — GPU device programming; CUDAToolkit = Hardware; Doxygen alone ≠ Process):
  Title: EXP: a Python/C++ package for basis function expansion methods in galactic dynamics
  Language: C++
  Dependencies: CUDAToolkit, Doxygen, Eigen3, HDF5, MPI, OpenMP, pybind11, VTK
  Output: {{"categories": ["Data Analytics", "Hardware", "Modeling And Simulation", "Ris", "Software", "Software Analytics"]}}
  Reasoning: CUDAToolkit signals Hardware because the C++ code directly programs GPU devices at low level, unlike high-level ML frameworks. N-body galactic simulations signal Modeling And Simulation. Galactic dynamics is a specific scientific domain so Ris applies. pybind11 signals Software. VTK and the analysis pipeline signal Software Analytics. HDF5 and numpy signal Data Analytics. Doxygen alone is only one documentation tool so Process does not qualify.

Example 5 (Python, 7 labels — neuroscience signal toolbox; 4 dev tools = Process):
  Title: PyBispectra: A toolbox for advanced electrophysiological signal processing using the bispectrum
  Language: Python
  Dependencies: mne, scipy, numba, numpy, ipywidgets, pytest, sphinx, pre-commit, coverage
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Process", "Ris", "Software", "Software Analytics", "Ui"]}}
  Reasoning: mne signals Ris because it is a domain-specific neuroscience library. scipy and numba signal Modeling And Simulation for spectral analysis. ipywidgets signals Ui for interactive notebooks. pytest, sphinx, pre-commit, and coverage together (4 tools) signal Process. numpy and scipy signal Data Analytics. The toolbox analyses electrophysiological signal data so Software Analytics applies. The project is a reusable library so Software applies.

Example 6 (Python, 7 labels — agent-based simulation; solara+ipyvuetify = Ui; scipy+networkx = M&S):
  Title: Mesa 3: Agent-based modeling with Python in 2025
  Language: Python
  Dependencies: networkx, scipy, numpy, matplotlib, pytest, sphinx, coverage, solara, ipyvue, ipyvuetify
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Process", "Ris", "Software", "Software Analytics", "Ui"]}}
  Reasoning: Mesa is the canonical agent-based modeling framework so Modeling And Simulation applies. scipy and networkx signal Data Analytics. solara, ipyvue, and ipyvuetify provide an interactive user-facing visualization interface so Ui applies. The project is a reusable library so Software applies. Complex systems research signals Ris. pytest, sphinx, and coverage together (3 tools) signal Process. The framework includes analysis tools for simulation outputs so Software Analytics applies.

Example 7 (R, 4 labels — political event data API; rmarkdown present but NO Ui, NO Process):
  Title: UTDEventData: An R package to access political event data
  Language: R
  Dependencies: countrycode, curl, jsonlite, knitr, methods, rjson, rmarkdown, stats
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Ris", "Software"]}}
  Reasoning: jsonlite and countrycode signal Data Analytics. The package retrieves and aggregates political event data from a remote API across multiple sources so Integrative Analysis applies. Political science data infrastructure is a specific research domain so Ris applies. The project is a reusable R package so Software applies. knitr and rmarkdown are used only for static documentation, not an interactive interface, so Ui does not apply. No test tools are present so Process does not qualify.
"""

# ── 7-shot WITH CoT ───────────────────────────────────────────────────────────────
# Extends 5-shot by adding:
#   Ex6 (Python): mesa — agent-based Modeling, Ui via solara, covers M&S clearly
#   Ex7 (R): daiquiri — strong Integrative Analysis, Software Analytics in R
# Verified JOSS IDs: 5742, 5662, 3290, 7302, 8504, 7668, 5034
# Languages: R x3, Python x3, C++ x1 | Cardinality: 5, 6, 7, 6, 7, 7, 8
EXAMPLES_7_COT = """\
=== EXAMPLES ===

Example 1 (R, 7 labels — Bayesian stats; rmarkdown=Ui, testthat+knitr=Process, rstan=Modeling):
  Title: measr: Bayesian psychometric measurement using Stan
  Language: R
  Dependencies: StanHeaders, rstan, dplyr, tidyr, Rcpp, testthat, knitr, rmarkdown
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Modeling And Simulation", "Process", "Ris", "Software", "Ui"]}}
  Reasoning: rstan and StanHeaders signal Modeling And Simulation. dplyr and tidyr signal Data Analytics. The package integrates psychometric and measurement data from multiple sources so Integrative Analysis applies. The psychometric measurement domain signals Ris. Rcpp signals Software. rmarkdown provides interactive report rendering so Ui applies. testthat and knitr together (2+ tools) signal Process.

Example 2 (Python, 6 labels — physical instrument control; NO Process because <2 dev tools):
  Title: PYDAQ: Data Acquisition and Experimental Analysis with Python
  Language: Python
  Dependencies: PySide6, nidaqmx, pyserial, matplotlib, numpy, packaging, shiboken6
  Output: {{"categories": ["Data Analytics", "Hardware", "Modeling And Simulation", "Ris", "Software", "Ui"]}}
  Reasoning: nidaqmx and pyserial signal Hardware because they interface with physical data acquisition instruments. PySide6 signals Ui. Physical measurement and system identification signals Modeling And Simulation and Ris. matplotlib and numpy signal Data Analytics. The project is a reusable library so Software applies. No dev/build/test tools are listed so Process does not qualify.

Example 3 (R, 5 labels — epidemiology data package; rmarkdown = static reporting NOT interactive Ui):
  Title: covidregionaldata: Subnational data for COVID-19 epidemiology
  Language: R
  Dependencies: dplyr, ggplot2, httr, knitr, lubridate, purrr, rmarkdown, sf, tidyr, testthat
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Process", "Ris", "Software"]}}
  Reasoning: dplyr and tidyr signal Data Analytics. The package integrates subnational COVID-19 data from multiple heterogeneous sources so Integrative Analysis applies. COVID-19 epidemiology is a specific scientific domain so Ris applies. The project is a reusable package so Software applies. testthat and knitr together (2+ tools) signal Process. rmarkdown is used for static reports only, not an interactive user-facing interface, so Ui does not apply.

Example 4 (C++, 6 labels — GPU device programming; CUDAToolkit = Hardware; Doxygen alone ≠ Process):
  Title: EXP: a Python/C++ package for basis function expansion methods in galactic dynamics
  Language: C++
  Dependencies: CUDAToolkit, Doxygen, Eigen3, HDF5, MPI, OpenMP, pybind11, VTK
  Output: {{"categories": ["Data Analytics", "Hardware", "Modeling And Simulation", "Ris", "Software", "Software Analytics"]}}
  Reasoning: CUDAToolkit signals Hardware because the C++ code directly programs GPU devices at low level, unlike high-level ML frameworks. N-body galactic simulations signal Modeling And Simulation. Galactic dynamics is a specific scientific domain so Ris applies. pybind11 signals Software. VTK and the analysis pipeline signal Software Analytics. HDF5 and numpy signal Data Analytics. Doxygen alone is only one documentation tool so Process does not qualify.

Example 5 (Python, 7 labels — neuroscience signal toolbox; 4 dev tools = Process):
  Title: PyBispectra: A toolbox for advanced electrophysiological signal processing using the bispectrum
  Language: Python
  Dependencies: mne, scipy, numba, numpy, ipywidgets, pytest, sphinx, pre-commit, coverage
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Process", "Ris", "Software", "Software Analytics", "Ui"]}}
  Reasoning: mne signals Ris because it is a domain-specific neuroscience library. scipy and numba signal Modeling And Simulation for spectral analysis. ipywidgets signals Ui for interactive notebooks. pytest, sphinx, pre-commit, and coverage together (4 tools) signal Process. numpy and scipy signal Data Analytics. The toolbox analyses electrophysiological signal data so Software Analytics applies. The project is a reusable library so Software applies.

Example 6 (Python, 7 labels — agent-based simulation; solara+ipyvuetify = Ui; scipy+networkx = M&S):
  Title: Mesa 3: Agent-based modeling with Python in 2025
  Language: Python
  Dependencies: networkx, scipy, numpy, matplotlib, pytest, sphinx, coverage, solara, ipyvue, ipyvuetify
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Process", "Ris", "Software", "Software Analytics", "Ui"]}}
  Reasoning: Mesa is the canonical agent-based modeling framework so Modeling And Simulation applies. scipy and networkx signal Data Analytics. solara, ipyvue, and ipyvuetify provide an interactive user-facing visualization interface so Ui applies. The project is a reusable library so Software applies. Complex systems research signals Ris. pytest, sphinx, and coverage together (3 tools) signal Process. The framework includes analysis tools for simulation outputs so Software Analytics applies.

Example 7 (R, 4 labels — political event data API; rmarkdown present but NO Ui, NO Process):
  Title: UTDEventData: An R package to access political event data
  Language: R
  Dependencies: countrycode, curl, jsonlite, knitr, methods, rjson, rmarkdown, stats
  Output: {{"categories": ["Data Analytics", "Integrative Analysis", "Ris", "Software"]}}
  Reasoning: jsonlite and countrycode signal Data Analytics. The package retrieves and aggregates political event data from a remote API across multiple sources so Integrative Analysis applies. Political science data infrastructure is a specific research domain so Ris applies. The project is a reusable R package so Software applies. knitr and rmarkdown are used only for static documentation, not an interactive interface, so Ui does not apply. No test tools are present so Process does not qualify.
"""

# ── 10-shot WITH CoT ─────────────────────────────────────────────────────────────
# Extends 7-shot with 3 more contrastive examples:
#   Ex8 (C): MPTRAC — C language, GPU via cudart=Hardware, atmospheric science=Ris
#   Ex9 (Python): actisleep-tracker — pure dashboard, 5 dev tools=Process, Ui via dash
#   Ex10 (Python): Jupyter Scatter — interactive widget; SA from analysis pipeline; NO Integrative Analysis
# Verified JOSS IDs: 5742, 5662, 3290, 7302, 8504, 7668, 5034, 8177, 8181, 7059
EXAMPLES_10_COT = EXAMPLES_7_COT + """\
Example 8 (C, 6 labels — atmospheric dispersion model; cudart+curand = Hardware in C):
  Title: MPTRAC: A high-performance Lagrangian transport model for atmospheric air parcel dispersion
  Language: C
  Dependencies: cudart, curand, hdf5, hdf5_hl, netcdf, gsl, gslcblas, sz, zstd
  Reasoning: cudart and curand signal Hardware because they are CUDA device runtime libraries for direct GPU programming in C. Lagrangian particle transport modelling signals Modeling And Simulation. Atmospheric science is a specific scientific domain so Ris applies. hdf5 and netcdf signal Data Analytics. The project is a reusable model library so Software applies. No dev/test/docs tools present so Process does not apply.
  Output: {{"categories": ["Data Analytics", "Hardware", "Modeling And Simulation", "Ris", "Software"]}}

Example 9 (Python, 7 labels — actigraphy dashboard; 5 dev tools = strong Process; dash = Ui):
  Title: ActiSleep Tracker: a Python-based dashboard for adjusting automatic sleep predictions of actigraphy data
  Language: Python
  Dependencies: dash, pandas, plotly, numpy, xarray, pre-commit, pytest, coverage, mypy, ruff
  Reasoning: pandas and xarray signal Data Analytics. Actigraphy and sleep research is a specific scientific domain so Ris applies. The project uses machine learning for sleep predictions so Modeling And Simulation applies. dash and plotly provide an interactive user-facing dashboard so Ui applies. The project is a reusable package so Software applies. pre-commit, pytest, coverage, mypy, and ruff together (5 tools) signal Process. The dashboard analyses data pipeline quality so Software Analytics applies.
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Process", "Ris", "Software", "Software Analytics", "Ui"]}}

Example 10 (Python, 6 labels — interactive scatter plot widget; anywidget = Ui; NO Ris, NO Integrative Analysis):
  Title: Jupyter Scatter: Interactive Exploration of Large-Scale Datasets
  Language: Python
  Dependencies: anywidget, pandas, scikit-learn, numpy, pytest, pre-commit, ruff, scipy, hdbscan
  Reasoning: pandas and scikit-learn signal Data Analytics. hdbscan and scipy signal Modeling And Simulation for clustering and spatial analysis. anywidget provides an interactive Jupyter scatter plot interface so Ui applies. pytest and pre-commit together (2+ tools) signal Process. The project is a reusable widget library so Software applies. The exploration pipeline analyses dataset characteristics so Software Analytics applies. No domain-specific scientific library (e.g. astropy, biopython, rdkit) is present so Ris does NOT apply. This project operates on a SINGLE dataset type, not multiple heterogeneous sources, so Integrative Analysis does NOT apply.
  Output: {{"categories": ["Data Analytics", "Modeling And Simulation", "Process", "Software", "Software Analytics", "Ui"]}}
"""

# ── Strip Reasoning lines for no-CoT ablation variants ───────────────────────────
def _strip_reasoning(examples_str: str) -> str:
    """Remove 'Reasoning: ...' lines to produce no-CoT variants."""
    lines = [l for l in examples_str.splitlines() if not l.strip().startswith("Reasoning:")]
    return "\n".join(lines) + "\n"

EXAMPLES_5_NO_COT_R  = _strip_reasoning(EXAMPLES_5_COT)   # 5-shot, no CoT
EXAMPLES_7_NO_COT_R  = _strip_reasoning(EXAMPLES_7_COT)   # 7-shot, no CoT
EXAMPLES_10_NO_COT_R = _strip_reasoning(EXAMPLES_10_COT)  # 10-shot, no CoT

# ── Output format ─────────────────────────────────────────────────────────────────
OUTPUT_FORMAT = 'Return ONLY valid JSON: {{"categories": ["Cat1", "Cat2", ...]}}'

# ── User template — {title}, {language}, {dependencies} are filled at runtime ────
# {examples} and {output_format} are filled when building VARIANTS
USER_TEMPLATE = """{examples}
=== CLASSIFY THIS SOFTWARE ===

Title: {{title}}
Language: {{language}}
Dependencies: [{{dependencies}}]

{output_format}
"""

# ── Variant definitions ───────────────────────────────────────────────────────────
#
# REDUCED GRID: 16 variants (was 28)
# Rationale: isolate one axis at a time; 7-shot is the production zone.
#
#   Axis A — Shot count   : 0-shot, 5-shot, 7-shot, 10-shot   (fixed: CoT+desc, no sculpting)
#   Axis B — CoT          : no / yes                           (fixed: 7-shot, desc, no sculpting)
#   Axis C — Descriptions : no / yes                           (fixed: 7-shot, CoT, no sculpting)
#   Axis D — Sculpting    : no / yes                           (fixed: 7-shot, CoT+desc)
#             + two extra sculpting checkpoints at 5-shot & 10-shot
#
VARIANTS = [
    # ── Axis A: Shot-count ablation (CoT+desc, no sculpting) ────────────────────
    {"name": "v00_zero_cot_desc",         "description": "0-shot  | CoT | desc | no sculpting  — baseline",
     "system_prompt": make_system(True, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_0,         output_format=OUTPUT_FORMAT)},

    {"name": "v01_five_cot_desc",         "description": "5-shot  | CoT | desc | no sculpting",
     "system_prompt": make_system(True, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_COT,     output_format=OUTPUT_FORMAT)},

    {"name": "v02_seven_cot_desc",        "description": "7-shot  | CoT | desc | no sculpting  ← production",
     "system_prompt": make_system(True, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_COT,     output_format=OUTPUT_FORMAT)},

    {"name": "v03_ten_cot_desc",          "description": "10-shot | CoT | desc | no sculpting",
     "system_prompt": make_system(True, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_10_COT,    output_format=OUTPUT_FORMAT)},

    # ── Axis B: CoT ablation (7-shot, desc, no sculpting) ───────────────────────
    {"name": "v04_seven_no_cot_no_desc",  "description": "7-shot  | no CoT | no desc | no sculpting",
     "system_prompt": make_system(False, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_NO_COT_R, output_format=OUTPUT_FORMAT)},

    {"name": "v05_seven_no_cot_desc",     "description": "7-shot  | no CoT | desc    | no sculpting",
     "system_prompt": make_system(True, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_NO_COT_R, output_format=OUTPUT_FORMAT)},

    {"name": "v06_seven_cot_no_desc",     "description": "7-shot  | CoT    | no desc | no sculpting",
     "system_prompt": make_system(False, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_COT,      output_format=OUTPUT_FORMAT)},
    # v02 already covers 7-shot CoT+desc (no sculpting)

    # ── Axis C: 5-shot no-CoT variants (compare with v01) ───────────────────────
    {"name": "v07_five_no_cot_no_desc",   "description": "5-shot  | no CoT | no desc | no sculpting",
     "system_prompt": make_system(False, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_NO_COT_R, output_format=OUTPUT_FORMAT)},

    {"name": "v08_five_no_cot_desc",      "description": "5-shot  | no CoT | desc    | no sculpting",
     "system_prompt": make_system(True, False),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_NO_COT_R, output_format=OUTPUT_FORMAT)},

    # ── Axis D: Sculpting ablation (fixed: CoT+desc, vs no sculpting) ───────────
    # 5-shot sculpting checkpoint
    {"name": "v09sc_five_cot_desc",       "description": "5-shot  | CoT | desc | sculpting",
     "system_prompt": make_system(True, True),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_5_COT,      output_format=OUTPUT_FORMAT)},

    # 7-shot sculpting — main comparison pair with v02
    {"name": "v10sc_seven_no_cot_no_desc","description": "7-shot  | no CoT | no desc | sculpting",
     "system_prompt": make_system(False, True),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_NO_COT_R, output_format=OUTPUT_FORMAT)},

    {"name": "v11sc_seven_no_cot_desc",   "description": "7-shot  | no CoT | desc    | sculpting",
     "system_prompt": make_system(True, True),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_NO_COT_R, output_format=OUTPUT_FORMAT)},

    {"name": "v12sc_seven_cot_no_desc",   "description": "7-shot  | CoT    | no desc | sculpting",
     "system_prompt": make_system(False, True),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_COT,      output_format=OUTPUT_FORMAT)},

    {"name": "v13sc_seven_cot_desc",      "description": "7-shot  | CoT    | desc    | sculpting  ← sculpted production",
     "system_prompt": make_system(True, True),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_7_COT,      output_format=OUTPUT_FORMAT)},

    # 10-shot sculpting checkpoint
    {"name": "v14sc_ten_cot_desc",        "description": "10-shot | CoT | desc | sculpting",
     "system_prompt": make_system(True, True),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_10_COT,     output_format=OUTPUT_FORMAT)},

    # 0-shot sculpting baseline
    {"name": "v15sc_zero_cot_desc",       "description": "0-shot  | CoT | desc | sculpting  — sculpting baseline",
     "system_prompt": make_system(True, True),
     "user_template": USER_TEMPLATE.format(examples=EXAMPLES_0,          output_format=OUTPUT_FORMAT)},
]

VARIANT_MAP = {v["name"]: v for v in VARIANTS}

# ── Head-to-head sculpting pairs (for focused analysis) ──────────────────────────
SCULPTING_PAIRS = [
    ("v02_seven_cot_desc",     "v13sc_seven_cot_desc"),    # ← flagship comparison
    ("v06_seven_cot_no_desc",  "v12sc_seven_cot_no_desc"),
    ("v05_seven_no_cot_desc",  "v11sc_seven_no_cot_desc"),
    ("v03_ten_cot_desc",       "v14sc_ten_cot_desc"),
]

SCULPTING_VARIANTS = [v for v in VARIANTS if v["name"].endswith("_sc") or "sc_" in v["name"]]
BASE_VARIANTS      = [v for v in VARIANTS if v not in SCULPTING_VARIANTS]

# ─────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test = {"title": "Test", "language": "Python", "dependencies": "numpy, scipy"}
    print(f"{'Ablation grid — focused 16-variant design':─<70}")
    print(f"\n  Total variants : {len(VARIANTS)}")
    print(f"  Base (no sculpting) : {len(BASE_VARIANTS)}")
    print(f"  With sculpting      : {len(SCULPTING_VARIANTS)}")
    print(f"  Est. LLM calls      : {len(VARIANTS)} × ~461 = ~{len(VARIANTS)*461:,}\n")
    print(f"  {'Name':<42} {'OK':<4} Description")
    print("─" * 105)
    for v in VARIANTS:
        try:
            v["user_template"].format(**test)
            status = "✅"
        except KeyError as e:
            status = f"❌ {e}"
        sc = " [SC]" if "sc" in v["name"] else ""
        print(f"  {v['name']:<42} {status:<4} {v['description']}{sc}")

    print(f"\nScuplting pairs (no-sc → sc):")
    for a, b in SCULPTING_PAIRS:
        print(f"  {a:<38} → {b}")