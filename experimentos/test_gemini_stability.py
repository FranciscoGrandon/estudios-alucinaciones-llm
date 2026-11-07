# Creado por Francisco Grandón Vergara
import requests

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
MODELS_TO_TEST = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]

for m in MODELS_TO_TEST:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={API_KEY}"
    try:
        r = requests.post(url, json={
            "contents": [{"parts": [{"text": "What is the capital of France? Reply with only the answer."}]}]
        }, timeout=10)
        print(f"[{m}] Status: {r.status_code}")
        if r.status_code == 200:
            txt = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            print(f"  -> {txt.strip()[:40]!r}")
    except Exception as e:
        print(f"[{m}] Exception: {e}")
