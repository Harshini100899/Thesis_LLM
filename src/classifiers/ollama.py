"""Ollama-based classifier implementation."""

import json
from typing import List

import requests

from .base import BaseClassifier, ClassificationResult
from prompts import build_system_prompt, build_user_prompt
from config import CATEGORIES, DEFAULT_OLLAMA_HOST, DEFAULT_TIMEOUT


class OllamaClassifier(BaseClassifier):
    """Classifier using Ollama API."""

    def __init__(
        self,
        model: str = "llama3.3:70b",
        host: str = DEFAULT_OLLAMA_HOST,
        temperature: float = 0.1,
        timeout: int = DEFAULT_TIMEOUT,
        use_few_shot: bool = True,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.use_few_shot = use_few_shot
        self.system_prompt = build_system_prompt()

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=15, verify=True)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(self.model in name or name in self.model for name in model_names)
        except requests.RequestException:
            return False

    def classify(
        self,
        title: str,
        description: str,
        language: str,
        dependencies: List[str],
    ) -> ClassificationResult:
        """Classify a single project using title, language, and dependencies."""

        user_prompt = build_user_prompt(
            title=title,
            language=language,
            dependencies=dependencies,
            use_few_shot=self.use_few_shot,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": 512,
            },
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
                verify=True,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return ClassificationResult(
                categories=[], reasoning="", raw_output="", success=False,
                error=f"Could not connect to Ollama at {self.host}",
            )
        except requests.exceptions.HTTPError as e:
            return ClassificationResult(
                categories=[], reasoning="", raw_output="", success=False,
                error=f"HTTP error: {e}",
            )
        except requests.exceptions.Timeout:
            return ClassificationResult(
                categories=[], reasoning="", raw_output="", success=False,
                error="Request timed out",
            )

        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        return self._parse_response(content)

    def _parse_response(self, content: str) -> ClassificationResult:
        """Parse the LLM response into a ClassificationResult."""
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)

                categories = parsed.get("categories", [])

                # Exact match first
                valid_categories = [
                    str(c).strip() for c in categories
                    if str(c).strip() in CATEGORIES
                ]

                # Fallback: case-insensitive match
                if not valid_categories and categories:
                    cat_map = {c.lower(): c for c in CATEGORIES}
                    valid_categories = [
                        cat_map[str(c).strip().lower()]
                        for c in categories
                        if str(c).strip().lower() in cat_map
                    ]

                return ClassificationResult(
                    categories=valid_categories,
                    reasoning=str(parsed.get("reasoning", "")),
                    raw_output=content,
                    success=bool(valid_categories),
                    error="" if valid_categories else "No valid categories in response",
                )
            else:
                return ClassificationResult(
                    categories=[], reasoning="", raw_output=content,
                    success=False, error="No JSON found in response",
                )
        except json.JSONDecodeError as e:
            return ClassificationResult(
                categories=[], reasoning="", raw_output=content,
                success=False, error=f"JSON parse error: {e}",
            )