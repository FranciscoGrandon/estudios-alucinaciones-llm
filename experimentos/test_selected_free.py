# Creado por Francisco Grandón Vergara
import requests, json

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

for m in ["nvidia/nemotron-3-ultra-550b-a55b:free", "xingchen-agi/xing4.0-29b-a4b:free"]:
    print(f"=== {m} ===")
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
        "model": m,
        "messages": [{"role": "user", "content": "What is the capital of France? Reply with only the answer."}],
        "max_tokens": 30
    }, timeout=20).json()
    print(json.dumps(res, indent=2))
