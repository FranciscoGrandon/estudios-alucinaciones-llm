# Creado por Francisco Grandón Vergara
import requests

API_KEY = "OPENROUTER_API_KEY_HERE"
headers = {"Authorization": f"Bearer {API_KEY}"}

try:
    r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30)
    data = r.json().get("data", [])
    free_models = [m["id"] for m in data if m.get("id", "").endswith(":free")]
    print(f"Total modelos: {len(data)}, Modelos FREE encontrados: {len(free_models)}")
    for m in free_models:
        print(" -", m)
except Exception as e:
    print("ERROR:", e)
