"""Category definitions for research software classification.

Based on: Hasselbring et al. (2025) Multi-Dimensional Categorization
of Research Software.
"""

CATEGORIES = {
    "Modeling And Simulation": {
        "id": "1.1",
        "description": "Numerical solvers, agent-based models, ODE/PDE simulators, process models",
        "parent": "Research Software",
    },
    "Data Analytics": {
        "id": "1.2",
        "description": "Statistical analysis, ML pipelines, data processing, feature engineering",
        "parent": "Research Software",
    },
    "Software Analytics": {
        "id": "1.3",
        "description": "Static/dynamic code analysis, repository mining, software quality tools",
        "parent": "Research Software",
    },
    "Integrative Analysis": {
        "id": "1.4",
        "description": "Multi-source data integration, data assimilation, scientific visualization",
        "parent": "Research Software",
    },
    "Hardware": {
        "id": "2.1",
        "description": "Embedded software, firmware, device drivers, hardware control",
        "parent": "Technology Research Software",
    },
    "Software": {
        "id": "2.2",
        "description": "OS components, compilers, runtimes, language tools, system utilities",
        "parent": "Technology Research Software",
    },
    "Ui": {
        "id": "2.3",
        "description": "GUIs, web frontends, dashboards, interactive interfaces",
        "parent": "Technology Research Software",
    },
    "Process": {
        "id": "2.4",
        "description": "Workflow automation, build/deploy, CI/CD, process management",
        "parent": "Technology Research Software",
    },
    "Ris": {
        "id": "3.1",
        "description": "Experiment control, data management, HPC, lab tools, collaboration",
        "parent": "Research Infrastructure Software",
    },
}

CATEGORY_NAMES = list(CATEGORIES.keys())


def get_category_text() -> str:
    """Generate formatted category text for prompts."""
    lines = []
    current_parent = None
    for name, info in CATEGORIES.items():
        if info["parent"] != current_parent:
            current_parent = info["parent"]
            lines.append(f"\n{current_parent}:")
        lines.append(f"  {info['id']} {name}: {info['description']}")
    return "\n".join(lines)
