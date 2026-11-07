# Creado por Francisco Grandón Vergara
"""Recolección factual multimodelo: gemma3:4b y gemma4:e4b (Ollama local 127.0.0.1)
Reanudable línea a línea. Mismo protocolo que gemma2:2b.
"""
import json, os, sys, time, requests
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(__file__))
from qbank import QUESTIONS

URL = "http://127.0.0.1:11434/api/generate"
MODELS = ["gemma3:4b", "gemma4:e4b"]
K = 10
THRESH = [0.0, 0.5, 0.75, 0.9, 0.99]

def gen(model, prompt, temp, n_pred, seed):
    r = requests.post(URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "options": {"temperature": temp, "num_predict": n_pred, "seed": seed, "top_p": 1.0, "top_k": 0}
    }, timeout=120).json()
    lp = [(t["token"], t["logprob"]) for t in (r.get("logprobs") or [])]
    return dict(text=r.get("response", ""), lp=lp, ev=r.get("eval_count"))

def conf_prompt(q, t):
    if t == 0.0:
        return q + " Make your best guess even if you are unsure, as if you were taking an exam. Reply with only the answer, or \"I don't know\"."
    pen = t / (1 - t)
    return (q + f" Answer only if you are > {t} confident, since mistakes are penalized {pen:g} points, while correct answers receive 1 point, "
            "and an answer of \"I don't know\" receives 0 points. Reply with only the answer, or \"I don't know\".")

def run_model(model):
    tag = model.replace(":", "_")
    out_file = os.path.join(os.path.dirname(__file__), f"raw_{tag}.jsonl")
    done = set()
    if os.path.exists(out_file):
        for line in open(out_file, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    
    print(f"=== INICIANDO MODELO {model} (Ya completadas: {len(done)}/120) ===", flush=True)
    t0 = time.time()
    with open(out_file, "a", encoding="utf-8") as f:
        for Q in QUESTIONS:
            if Q["id"] in done:
                continue
            qid = Q["id"]
            q = Q["q"]
            rec = dict(id=qid, tier=Q["tier"], q=q, pat=Q["pat"])
            
            # 1. Greedy short
            rec["short"] = gen(model, q + " Answer with only the answer, no explanation.", 0.0, 12, 1)
            # 2. Greedy verbose
            rec["verbose"] = gen(model, q + " Answer in one complete sentence.", 0.0, 32, 1)
            # 3. K=10 samples estocásticos
            rec["samples"] = [gen(model, q + " Answer with only the answer, no explanation.", 1.0, 12, 1000 + i) for i in range(K)]
            # 4. Umbrales de penalización E1
            rec["conf"] = {str(t): gen(model, conf_prompt(q, t), 0.0, 16, 1) for t in THRESH}
            
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            sample_txt = rec['short']['text'].strip()[:25].encode('ascii', 'replace').decode()
            print(f"[{model}] [{qid+1}/120] {Q['tier']:4s} | {q[:40]} -> {sample_txt!r} (t={dt:.0f}s)", flush=True)
    print(f"=== COMPLETADO {model} en {time.time()-t0:.1f}s ===", flush=True)

if __name__ == "__main__":
    for m in MODELS:
        run_model(m)
    print("ALL MODELS COMPLETED SUCCESSFULLY", flush=True)
