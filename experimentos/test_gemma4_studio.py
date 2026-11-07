# Creado por Francisco Grandón Vergara
import requests, json, time

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent?key={API_KEY}"

payload = {
    "contents": [{"parts": [{"text": "What is the capital of France? Reply with only the answer, no explanation."}]}],
    "generationConfig": {
        "temperature": 0.0,
        "maxOutputTokens": 20
    }
}

t0 = time.time()
r = requests.post(URL, json=payload, timeout=20)
print("Status:", r.status_code, f"({time.time()-t0:.2f}s)")
if r.status_code == 200:
    res = r.json()
    print("Response JSON:")
    print(json.dumps(res, indent=2))
else:
    print("Error:", r.text[:300])
