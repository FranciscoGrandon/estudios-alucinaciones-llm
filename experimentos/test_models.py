# Creado por Francisco Grandón Vergara
import requests, time

URL = "http://127.0.0.1:11434/api/generate"
MODELS = ["gemma2:2b", "gemma3:4b", "gemma4:e4b"]

for m in MODELS:
    t0 = time.time()
    try:
        r = requests.post(URL, json={
            "model": m,
            "prompt": "What is the capital of France? Reply with only the answer.",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10, "seed": 42}
        }, timeout=60).json()
        dt = time.time() - t0
        print(f"[{m}] OK ({dt:.2f}s): {r.get('response', '').strip()!r} (eval_count={r.get('eval_count')})")
    except Exception as e:
        print(f"[{m}] ERROR: {e}")
