"""Check available models on the Fraunhofer AI server."""

import requests
import json

API_BASE = "https://ai.compute.isst.fraunhofer.de"
TARGET_MODEL = "llama3.3:70b"

# Step 1: List available models
print(f"Fetching models from {API_BASE}...\n")
try:
    r = requests.get(f"{API_BASE}/api/tags", timeout=15, verify=True)
    r.raise_for_status()
    models = r.json().get("models", [])
    
    print(f"Found {len(models)} model(s):\n")
    print(f"{'Name':<40} {'Size':<12} {'Params':<15} {'Quant'}")
    print("-" * 85)
    found_target = False
    for m in models:
        name = m.get("name", "?")
        size_gb = m.get("size", 0) / 1e9
        details = m.get("details", {})
        params = details.get("parameter_size", "?")
        quant = details.get("quantization_level", "?")
        marker = " ← TARGET" if TARGET_MODEL in name or name in TARGET_MODEL else ""
        if marker:
            found_target = True
        print(f"{name:<40} {size_gb:.1f} GB     {params:<15} {quant}{marker}")
    
    if not found_target:
        print(f"\n⚠️  Target model '{TARGET_MODEL}' not found in available models!")

except requests.ConnectionError:
    print(f"❌ Cannot connect to {API_BASE}")
    print("Check if you need VPN or if the URL is correct.")
except requests.HTTPError as e:
    print(f"❌ HTTP Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

# Step 2: Quick test with target model
print(f"\n--- Quick test with '{TARGET_MODEL}' ---")
try:
    payload = {
        "model": TARGET_MODEL,
        "prompt": "Reply with just: hello",
        "stream": False,
        "options": {"num_predict": 10}
    }
    r = requests.post(f"{API_BASE}/api/generate", json=payload, timeout=120, verify=True)
    r.raise_for_status()
    output = r.json().get("response", "")
    print(f"✅ Response: {output.strip()}")
except Exception as e:
    print(f"❌ Test failed: {e}")
