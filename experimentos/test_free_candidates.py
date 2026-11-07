# Creado por Francisco Grandón Vergara
import requests

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

CANDIDATES = [
    "liquid/lfm-2.5-2.6b:free",
    "liquid/lfm-3-1.7b:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3.5-lightning:free",
    "recursive/flint-v1.0:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
]

for m in CANDIDATES:
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
            "model": m,
            "messages": [{"role": "user", "content": "What is the capital of France? Reply with only the answer."}],
            "max_tokens": 50
        }, timeout=15).json()
        
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        txt = msg.get("content") or msg.get("reasoning") or ""
        if txt:
            print(f"  [OK] {m} -> {txt.strip()[:60]!r}")
        else:
            err = res.get("error", {}).get("message", str(res))[:80]
            print(f"  [FAIL] {m} -> {err}")
    except Exception as e:
        print(f"  [ERROR] {m} -> {e}")
