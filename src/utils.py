"""Utility functions."""

import sys
from typing import Iterable, TypeVar

T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 40,
) -> Iterable[T]:
    """Simple progress bar for iteration."""
    
    def print_progress(iteration: int):
        percent = 100 * iteration / total
        filled = int(length * iteration // total)
        bar = "█" * filled + "-" * (length - filled)
        sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}")
        sys.stdout.flush()
    
    print_progress(0)
    for i, item in enumerate(iterable, 1):
        yield item
        print_progress(i)
    print()  # New line after completion


def check_ollama_connection(host: str) -> tuple[bool, str]:
    """Check if Ollama is running.

    Returns:
        Tuple of (is_running, message)
    """
    import requests

    try:
        resp = requests.get(f"{host}/api/tags", timeout=15, verify=True)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return True, f"Connected. Models: {', '.join(model_names)}"
        return False, f"Unexpected status code: {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to {host}. Check VPN/network."
    except Exception as e:
        return False, f"Error: {e}"
