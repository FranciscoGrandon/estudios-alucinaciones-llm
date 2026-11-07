# Creado por Francisco Grandón Vergara
import requests

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MODELS = [
    "liquid/lfm-2.5-2.6b:free",
    "liquid/lfm-3-1.7b:free",
    "cohere/north-mini-code:free"
]

TEST_QUESTIONS = [
    "What is the capital of France? Reply with only the answer.",
    "Who wrote Romeo and Juliet? Reply with only the answer.",
    "What is the chemical symbol for gold? Reply with only the answer."
]

for m in MODELS:
    print(f"=== {m} ===")
    for q in TEST_QUESTIONS:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
            "model": m,
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 25,
            "temperature": 0.0
        }, timeout=20).json()
        choice = res.get("choices", [{}])[0]
        msg = choice.get("message", {})
        txt = (msg.get("content") or msg.get("reasoning") or "").strip()
        print(f"  {q[:30]} -> {txt!r}")
