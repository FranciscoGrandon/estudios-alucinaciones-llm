# Creado por Francisco Grandón Vergara
import requests, json

payload = {
    "model": "gemma4:e4b",
    "prompt": "¿Cuál es la contribución principal del artículo «Attention Is All You Need» (2017)?",
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 45}
}
r = requests.post("http://127.0.0.1:11434/api/generate", json=payload).json()
print("Respuesta cruda de gemma4:e4b:")
print(json.dumps(r, indent=2))
