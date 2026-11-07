# Creado por Francisco Grandón Vergara
import requests, time

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent?key={API_KEY}"

prompts = [
    "What is the capital of France?",
    "What is the capital of France? Answer with only the answer, no explanation.",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?"
]

for p in prompts:
    t0 = time.time()
    r = requests.post(URL, json={"contents": [{"parts": [{"text": p}]}]}, timeout=15)
    dt = time.time() - t0
    print(f"[{dt:.1f}s] Status: {r.status_code} | Prompt: {p[:30]}...")
    if r.status_code == 200:
        print("  ->", r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()[:40])
    time.sleep(2)
