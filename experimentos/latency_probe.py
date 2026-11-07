# Creado por Francisco Grandón Vergara
"""Prueba de latencia y logprobs con gemma2:2b (Ollama local, 127.0.0.1)."""
import requests, time, json
URL = "http://127.0.0.1:11434/api/generate"
qs = ["What is the capital of Australia?", "Who wrote Don Quixote?", "What is the chemical symbol for gold?",
      "What is the capital of Burkina Faso?", "In which year did the Battle of Hastings take place?"]
out = []
for q in qs:
    t = time.time()
    r = requests.post(URL, json={"model": "gemma2:2b", "prompt": q + " Answer in one short sentence.", "stream": False,
                                 "logprobs": True, "options": {"temperature": 0, "num_predict": 40, "seed": 1}}, timeout=300).json()
    dt = time.time() - t
    out.append(dict(q=q, resp=r.get("response"), secs=dt, eval_count=r.get("eval_count"),
                    has_logprobs=("logprobs" in r), keys=list(r.keys()),
                    lp_sample=(r.get("logprobs") or [None])[:2]))
    print(q, "->", r.get("response").strip(), f"{dt:.1f}s", r.get("eval_count"), "logprobs" in r)
json.dump(out, open(r"D:\Marketing\estudios_alucinaciones_llm\experimentos\latency_probe.json", "w"), indent=1)
