# Creado por Francisco Grandón Vergara
import requests, time

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
MODELS = [
    "gemma-4-31b-it",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest"
]

for m in MODELS:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={API_KEY}"
    t0 = time.time()
    try:
        r = requests.post(url, json={
            "contents": [{"parts": [{"text": "What is the capital of France? Reply with only the answer."}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 20}
        }, timeout=15)
        dt = time.time() - t0
        print(f"[{m}] Status: {r.status_code} ({dt:.2f}s)")
        if r.status_code == 200:
            txt = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            print(f"  -> {txt!r}")
        else:
            print(f"  -> {r.text[:100]}")
    except Exception as e:
        print(f"[{m}] Exception: {e}")
