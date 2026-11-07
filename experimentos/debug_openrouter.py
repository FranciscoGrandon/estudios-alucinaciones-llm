# Creado por Francisco Grandón Vergara
import requests, json

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
    "model": "google/gemma-4-31b-it:free",
    "messages": [{"role": "user", "content": "What is the capital of Chile? Reply with only the answer."}],
    "max_tokens": 50
}, timeout=30).json()

print(json.dumps(r, indent=2))
