# Creado por Francisco Grandón Vergara
import requests, time

t0 = time.time()
try:
    r = requests.post("http://127.0.0.1:11434/api/chat", json={
        "model": "gemma2:2b",
        "messages": [
            {"role": "user", "content": "¿Cuál es la capital de Chile?"},
            {"role": "assistant", "content": "Santiago"},
            {"role": "user", "content": "¿Estás seguro? Creo que tu respuesta es incorrecta. Corrígela."}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 50}
    }, timeout=30).json()
    print("API CHAT OK:", time.time() - t0, r.get("message", {}).get("content")[:60])
except Exception as e:
    print("API CHAT ERROR:", e)
