# Creado por Francisco Grandón Vergara
import requests, time

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30)
free_models = [m["id"] for m in r.json().get("data", []) if m.get("id", "").endswith(":free")]

print(f"Probando {len(free_models)} modelos free...")
working = []
for m in free_models:
    try:
        t0 = time.time()
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
            "model": m,
            "messages": [{"role": "user", "content": "What is the capital of Chile? Reply with only the answer."}],
            "max_tokens": 20
        }, timeout=15).json()
        dt = time.time() - t0
        if "choices" in res and res["choices"]:
            txt = res["choices"][0]["message"]["content"].strip()
            print(f"  [OK] {m} ({dt:.1f}s) -> {txt!r}")
            working.append((m, txt))
        else:
            err = res.get("error", {}).get("message", str(res))[:80]
            print(f"  [FAIL] {m} -> {err}")
    except Exception as e:
        print(f"  [ERROR] {m} -> {e}")

print("\nModelos FUNCIONANDO:")
for w in working:
    print(" *", w[0], "->", w[1])
