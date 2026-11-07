# Creado por Francisco Grandón Vergara
import requests, time

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com",
    "X-Title": "Doctoral Research Hallucinations"
}

TEST_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free"
]

for m in TEST_MODELS:
    t0 = time.time()
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
            "model": m,
            "messages": [{"role": "user", "content": "What is the capital of Chile? Reply with only the answer, no explanation."}],
            "temperature": 0.0,
            "max_tokens": 10
        }, timeout=30).json()
        dt = time.time() - t0
        msg = r.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(f"[{m}] OK ({dt:.2f}s): {msg!r}")
    except Exception as e:
        print(f"[{m}] ERROR: {e}")
