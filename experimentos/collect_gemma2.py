# Creado por Francisco Grandón Vergara
"""Recolección con gemma2:2b (Ollama local 127.0.0.1) - reanudable.
Por pregunta: verbose greedy, short greedy, K=10 muestras T=1 (short), y 5 umbrales de confianza t (E1).
Salida: raw_gemma2_2b.jsonl (una línea por pregunta)
"""
import json, os, sys, time, requests
sys.path.insert(0, os.path.dirname(__file__))
from qbank import QUESTIONS

URL = "http://127.0.0.1:11434/api/generate"
MODEL = "gemma2:2b"
OUT = os.path.join(os.path.dirname(__file__), "raw_gemma2_2b.jsonl")
K = 10
THRESH = [0.0, 0.5, 0.75, 0.9, 0.99]

def gen(prompt, temp, n_pred, seed):
    r = requests.post(URL, json={"model": MODEL, "prompt": prompt, "stream": False, "logprobs": True,
                                 "options": {"temperature": temp, "num_predict": n_pred, "seed": seed, "top_p": 1.0, "top_k": 0}},
                      timeout=600).json()
    lp = [(t["token"], t["logprob"]) for t in (r.get("logprobs") or [])]
    return dict(text=r["response"], lp=lp, ev=r.get("eval_count"))

def conf_prompt(q, t):
    if t == 0.0:
        return q + " Make your best guess even if you are unsure, as if you were taking an exam. Reply with only the answer, or \"I don't know\"."
    pen = t / (1 - t)
    return (q + f" Answer only if you are > {t} confident, since mistakes are penalized {pen:g} points, while correct answers receive 1 point, "
            "and an answer of \"I don't know\" receives 0 points. Reply with only the answer, or \"I don't know\".")

done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        done.add(json.loads(line)["id"])
t0 = time.time()
with open(OUT, "a", encoding="utf-8") as f:
    for Q in QUESTIONS:
        if Q["id"] in done:
            continue
        q = Q["q"]
        rec = dict(id=Q["id"], tier=Q["tier"], q=q, pat=Q["pat"])
        rec["verbose"] = gen(q + " Answer in one complete sentence.", 0, 48, 1)
        rec["short"] = gen(q + " Answer with only the answer, no explanation.", 0, 16, 1)
        rec["samples"] = [gen(q + " Answer with only the answer, no explanation.", 1.0, 16, 1000 + i) for i in range(K)]
        rec["conf"] = {str(t): gen(conf_prompt(q, t), 0, 20, 1) for t in THRESH}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"[{Q['id']+1}/120] {Q['tier']} {q[:50]} -> {rec['short']['text'].strip()[:30]!r}  t={time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
