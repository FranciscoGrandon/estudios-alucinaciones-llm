# Creado por Francisco Grandón Vergara
import requests, json

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent?key={API_KEY}"

# Probar payload minimal sin generationConfig
payload_simple = {
    "contents": [{"parts": [{"text": "What is the capital of France?"}]}]
}
r1 = requests.post(URL, json=payload_simple, timeout=20)
print("Payload simple:", r1.status_code)
if r1.status_code == 200:
    print("Respuesta:", r1.json()["candidates"][0]["content"]["parts"][0]["text"][:60])
else:
    print("Error:", r1.text[:150])

# Probar payload v1 en vez de v1beta
URL_V1 = f"https://generativelanguage.googleapis.com/v1/models/gemma-4-26b-a4b-it:generateContent?key={API_KEY}"
r2 = requests.post(URL_V1, json=payload_simple, timeout=20)
print("Endpoint v1:", r2.status_code)
if r2.status_code == 200:
    print("Respuesta v1:", r2.json()["candidates"][0]["content"]["parts"][0]["text"][:60])
