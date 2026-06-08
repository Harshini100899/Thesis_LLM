"""Data loading and parsing utilities."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


@dataclass
class Project:
    """Represents a project to classify."""
    joss_id: str
    title: str
    description: str
    language: str
    dependencies: List[str]
    ground_truth: List[str] = field(default_factory=list)  # From dependency_labels
    
    @classmethod
    def from_row(cls, row: pd.Series) -> "Project":
        """Create a Project from a DataFrame row."""
        return cls(
            joss_id=str(row.get("joss_id", "")),
            title=str(row.get("title", "") or row.get("Repository Name", "")),
            description=str(row.get("Description", "") or ""),
            language=str(row.get("Language", "") or ""),
            dependencies=parse_list_field(row.get("dependecies_found", "")),
            ground_truth=parse_list_field(row.get("dependency_labels", "")),
        )


def parse_list_field(raw: Any) -> List[str]:
    """Parse a list field from CSV string format."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    
    return []


# Keep old function name for compatibility
parse_dependencies = parse_list_field


def load_projects(
    csv_path: Path,
    limit: Optional[int] = None,
    require_dependencies: bool = True,
) -> List[Project]:
    """Load projects from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        limit: Maximum number of projects to load
        require_dependencies: If True, only load projects with dependencies
    
    Returns:
        List of Project objects
    """
    df = pd.read_csv(csv_path)
    
    projects = []
    for _, row in df.iterrows():
        project = Project.from_row(row)
        
        if require_dependencies and not project.dependencies:
            continue
        
        projects.append(project)
        
        if limit and len(projects) >= limit:
            break
    
    return projects


def save_results(
    results: List[dict],
    output_path: Path,
) -> pd.DataFrame:
    """Save classification results to CSV.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the CSV
    
    Returns:
        DataFrame of results
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df
