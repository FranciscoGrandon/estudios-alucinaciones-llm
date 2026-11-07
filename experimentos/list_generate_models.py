# Creado por Francisco Grandón Vergara
import requests

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
r = requests.get(URL).json()
for m in r.get("models", []):
    methods = m.get("supportedGenerationMethods", [])
    if "generateContent" in methods:
        print(m["name"], "|", m.get("displayName", ""), "| input_limit:", m.get("inputTokenLimit"))
