# Creado por Francisco Grandón Vergara
"""Recolección OpenRouter Free:
1. nvidia/nemotron-3-ultra-550b-a55b:free (550B MoE Frontier)
2. liquid/lfm-2.5-2.6b:free (2.6B Liquid Foundation Model)
Evalúa el banco de 120 preguntas (qbank.py) con pausas de 1s para respetar rate limits.
Reanudable línea a línea.
"""
import json, os, sys, time, requests
sys.path.insert(0, os.path.dirname(__file__))
from qbank import QUESTIONS

API_KEY = "OPENROUTER_API_KEY_HERE"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com",
    "X-Title": "Doctoral Research Hallucinations"
}
URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_MODELS = [
    ("nvidia/nemotron-3-ultra-550b-a55b:free", "raw_openrouter_nemotron_550b.jsonl"),
    ("liquid/lfm-2.5-2.6b:free", "raw_openrouter_lfm_2_6b.jsonl")
]
K_SAMPLES = 5
THRESH = [0.0, 0.5, 0.99]

def call_api(model, prompt, temp=0.0, max_tokens=20, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.post(URL, headers=HEADERS, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": max_tokens
            }, timeout=30)
            res = r.json()
            if "choices" in res and res["choices"]:
                msg = res["choices"][0].get("message", {})
                txt = msg.get("content") or msg.get("reasoning") or ""
                return {"text": txt.strip()}
            elif "error" in res:
                err_msg = res["error"].get("message", "")
                if "rate" in err_msg.lower() or res["error"].get("code") == 429:
                    time.sleep(3 + attempt * 2)
                    continue
        except Exception as e:
            time.sleep(2)
    return {"text": ""}

def conf_prompt(q, t):
    if t == 0.0:
        return q + " Make your best guess even if unsure. Reply with only the answer, or \"I don't know\"."
    pen = t / (1.0 - t) if t < 1.0 else 99.0
    return (q + f" Answer only if you are > {t} confident, since mistakes are penalized {pen:g} points, while correct answers receive 1 point, "
            "and an answer of \"I don't know\" receives 0 points. Reply with only the answer, or \"I don't know\".")

def run_openrouter(model_id, out_name):
    out_file = os.path.join(os.path.dirname(__file__), out_name)
    done = set()
    if os.path.exists(out_file):
        for line in open(out_file, encoding="utf-8"):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
            
    print(f"=== INICIANDO OPENROUTER: {model_id} (Completadas: {len(done)}/120) ===", flush=True)
    t0 = time.time()
    with open(out_file, "a", encoding="utf-8") as f:
        for Q in QUESTIONS:
            qid = Q["id"]
            if qid in done: continue
            q = Q["q"]
            rec = dict(id=qid, tier=Q["tier"], q=q, pat=Q["pat"])
            
            # 1. Short greedy (max_tokens=65 para acomodar thinking previo)
            rec["short"] = call_api(model_id, q + " Answer with only the answer, no explanation.", 0.0, 65)
            time.sleep(0.6)
            # 2. Verbose greedy
            rec["verbose"] = call_api(model_id, q + " Answer in one complete sentence.", 0.0, 65)
            time.sleep(0.6)
            # 3. K samples estocásticos
            rec["samples"] = []
            for i in range(K_SAMPLES):
                rec["samples"].append(call_api(model_id, q + " Answer with only the answer, no explanation.", 1.0, 65))
                time.sleep(0.4)
            # 4. Umbrales de penalización
            rec["conf"] = {}
            for t in THRESH:
                rec["conf"][str(t)] = call_api(model_id, conf_prompt(q, t), 0.0, 65)
                time.sleep(0.6)
                
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            print(f"[{model_id.split('/')[1]}] [{qid+1}/120] {Q['tier']:4s} | {q[:35]} -> {rec['short']['text'][:25]!r} (t={dt:.0f}s)", flush=True)
    print(f"=== COMPLETADO {model_id} en {time.time()-t0:.1f}s ===", flush=True)

if __name__ == "__main__":
    for mid, fname in OPENROUTER_MODELS:
        run_openrouter(mid, fname)
    print("OPENROUTER EVALUATION COMPLETE", flush=True)
