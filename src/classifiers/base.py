"""Base classifier interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClassificationResult:
    """Result of a classification."""
    categories: List[str]
    reasoning: str
    raw_output: str
    success: bool
    error: Optional[str] = None


class BaseClassifier(ABC):
    """Abstract base class for classifiers."""
    
    @abstractmethod
    def classify(
        self,
        title: str,
        description: str,
        language: str,
        dependencies: List[str],
    ) -> ClassificationResult:
        """Classify a single project."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the classifier is available."""
        pass
