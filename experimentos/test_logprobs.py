# Creado por Francisco Grandón Vergara
import requests

URL = "http://127.0.0.1:11434/api/generate"
for m in ["gemma3:4b", "gemma4:e4b"]:
    r = requests.post(URL, json={
        "model": m,
        "prompt": "What is the capital of Chile? Reply with only the answer.",
        "stream": False,
        "logprobs": True,
        "options": {"temperature": 0.0, "num_predict": 8, "seed": 42}
    }, timeout=60).json()
    lps = r.get("logprobs")
    print(f"[{m}] Text: {r.get('response', '').strip()!r}, has_logprobs: {bool(lps)}, count: {len(lps) if lps else 0}")
    if lps:
        print(f"[{m}] Sample token: {lps[0]}")
