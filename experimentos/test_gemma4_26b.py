# Creado por Francisco Grandón Vergara
import requests

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent?key={API_KEY}"

payload = {
    "contents": [{"parts": [{"text": "What is the capital of France? Reply with only the word."}]}],
    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 100}
}

r = requests.post(URL, json=payload, timeout=20)
print("Status 26B:", r.status_code)
if r.status_code == 200:
    print("Respuesta 26B:", r.json()["candidates"][0]["content"]["parts"][0]["text"])
else:
    print("Error 26B:", r.text[:200])
