"""Diagnose LLM API connectivity and response times."""

import time
import requests
from config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, DEFAULT_TIMEOUT

API_BASE = DEFAULT_OLLAMA_HOST
MODEL = DEFAULT_OLLAMA_MODEL


def test_connection():
    print(f"Testing connection to {API_BASE}...")
    try:
        r = requests.get(f"{API_BASE}/api/tags", timeout=15, verify=True)
        print(f"  ✅ Connected. Status: {r.status_code}")
        for m in r.json().get("models", []):
            marker = " ← TARGET" if MODEL in m["name"] else ""
            size_gb = m.get("size", 0) / 1e9
            print(f"    - {m['name']} ({size_gb:.1f}GB){marker}")
        return True
    except requests.ConnectionError:
        print(f"  ❌ Cannot connect to {API_BASE}")
        print("  Check: VPN connected? Server URL correct?")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_classify():
    print(f"\nTest classification with {MODEL}...")
    print(f"  (70B model may take 30-90s on first request while loading into GPU)")

    # Minimal payload to reduce response time
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Reply with only: {\"categories\": [\"Software\"]}"},
        ],
        "options": {"temperature": 0.1, "num_predict": 50},
        "stream": False,
    }

    start = time.time()
    print(f"  Waiting", end="", flush=True)

    # Use a thread to show progress dots
    import threading
    stop_dots = threading.Event()

    def print_dots():
        while not stop_dots.is_set():
            print(".", end="", flush=True)
            stop_dots.wait(5)

    dot_thread = threading.Thread(target=print_dots, daemon=True)
    dot_thread.start()

    try:
        r = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=DEFAULT_TIMEOUT, verify=True)
        stop_dots.set()
        elapsed = time.time() - start
        content = r.json().get("message", {}).get("content", "")
        print(f"\n  ✅ Response in {elapsed:.1f}s")
        print(f"  Output: {content[:200]}")
        if elapsed > 60:
            print(f"  ⚠️ Slow response. First request loads model into GPU — subsequent requests will be faster.")
    except requests.Timeout:
        stop_dots.set()
        print(f"\n  ❌ Timed out after {time.time() - start:.1f}s")
        print(f"  Current timeout: {DEFAULT_TIMEOUT}s. The 70B model may need more time on first load.")
    except Exception as e:
        stop_dots.set()
        print(f"\n  ❌ Error: {e}")


def test_prompt_size():
    try:
        from prompts import build_system_prompt, build_user_prompt
        sys_prompt = build_system_prompt()
        user_prompt = build_user_prompt("Test", "Python", dependencies=["numpy", "pandas"])
        total = len(sys_prompt) + len(user_prompt)
        tokens_est = total // 4
        print(f"\nPrompt size: {total:,} chars (~{tokens_est:,} tokens)")
        if tokens_est > 4000:
            print("  ⚠️ Large prompt — may cause slow responses")
        else:
            print("  ✅ Prompt size OK")
    except ImportError as e:
        print(f"\n  Could not import prompts: {e}")


if __name__ == "__main__":
    if test_connection():
        test_classify()
    test_prompt_size()
