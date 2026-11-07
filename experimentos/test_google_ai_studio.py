# Creado por Francisco Grandón Vergara
import requests, json

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

try:
    r = requests.get(URL, timeout=15)
    print("Status Code:", r.status_code)
    if r.status_code == 200:
        models = r.json().get("models", [])
        print(f"Total modelos accesibles: {len(models)}")
        gemma_models = [m["name"] for m in models if "gemma" in m["name"].lower()]
        print("Modelos Gemma encontrados:")
        for gm in gemma_models:
            print(" -", gm)
        all_names = [m["name"] for m in models]
        print("\nMuestra de otros modelos:")
        for name in all_names[:10]:
            print(" *", name)
    else:
        print("Error response:", r.text[:300])
except Exception as e:
    print("Exception:", e)
