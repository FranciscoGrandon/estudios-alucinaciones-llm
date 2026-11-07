# Creado por Francisco Grandón Vergara
import requests, json

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
    "model": "liquid/lfm-2.5-2.6b:free",
    "messages": [{"role": "user", "content": "What is the capital of France? Reply with only the answer."}],
    "max_tokens": 50
}, timeout=20).json()

print(json.dumps(res, indent=2))
