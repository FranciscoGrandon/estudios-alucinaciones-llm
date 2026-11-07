# Creado por Francisco Grandón Vergara
import requests, json

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent?key={API_KEY}"

payload = {
    "contents": [{"parts": [{"text": "What is the capital of France? Answer with only the answer, no explanation."}]}],
    "generationConfig": {
        "temperature": 0.0,
        "maxOutputTokens": 100
    }
}

r = requests.post(URL, json=payload, timeout=20).json()
txt = r.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
print("Respuesta completa de gemma-4-31b-it:")
print(repr(txt))
