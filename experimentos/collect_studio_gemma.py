# Creado por Francisco Grandón Vergara
"""Recolección con Google AI Studio: gemma-4-26b-a4b-it
Temporización estricta: Máximo 1 pregunta por minuto (>=62 segundos por ciclo de pregunta).
6.0 segundos de enfriamiento entre llamadas atómicas.
Reanudable línea a línea desde raw_studio_gemma4_26b.jsonl.
"""
import json, os, sys, time, requests, io
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, io.UnsupportedOperation):
    pass
sys.path.insert(0, os.path.dirname(__file__))
from qbank import QUESTIONS

API_KEY = "GOOGLE_AI_STUDIO_API_KEY_HERE"
MODEL = "gemma-4-26b-a4b-it"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
OUT_FILE = os.path.join(os.path.dirname(__file__), "raw_studio_gemma4_26b.jsonl")
K_SAMPLES = 5
THRESH = [0.0, 0.5, 0.99]
INTER_CALL_DELAY = 6.2  # segundos entre llamadas atómicas
MIN_QUESTION_CYCLE = 65.0  # segundos mínimos por pregunta completa (máx 1 pregunta por minuto)

def call_studio(prompt, max_retries=5):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(URL, json=payload, timeout=30)
            if r.status_code == 200:
                res = r.json()
                cands = res.get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    if parts:
                        txt = parts[0].get("text", "").strip()
                        return {"text": txt}
                return {"text": ""}
            elif r.status_code == 429:
                wait_sec = 35 * (attempt + 1)
                print(f"[Studio 429 RateLimit] Enfriando {wait_sec}s...", flush=True)
                time.sleep(wait_sec)
                continue
            else:
                print(f"[Studio Error {r.status_code}] {r.text[:80]}", flush=True)
                time.sleep(INTER_CALL_DELAY)
        except Exception as e:
            print(f"[Studio Exception] {e}", flush=True)
            time.sleep(10)
    return {"text": ""}

def conf_prompt(q, t):
    if t == 0.0:
        return q + " Make your best guess even if unsure. Reply with only the answer, or \"I don't know\"."
    pen = t / (1.0 - t) if t < 1.0 else 99.0
    return (q + f" Answer only if you are > {t} confident, since mistakes are penalized {pen:g} points, while correct answers receive 1 point, "
            "and an answer of \"I don't know\" receives 0 points. Reply with only the answer, or \"I don't know\".")

def main():
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
            
    print(f"=== INICIANDO GOOGLE AI STUDIO: {MODEL} (Completadas: {len(done)}/120) ===", flush=True)
    print(f"Política de cadencia: máximo 1 pregunta cada {MIN_QUESTION_CYCLE:.0f}s ({INTER_CALL_DELAY}s inter-llamada)", flush=True)
    t0 = time.time()
    
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for Q in QUESTIONS:
            qid = Q["id"]
            if qid in done:
                continue
            t_q_start = time.time()
            q = Q["q"]
            rec = dict(id=qid, tier=Q["tier"], q=q, pat=Q["pat"])
            
            # 1. Short greedy
            rec["short"] = call_studio(q + " Answer with only the answer, no explanation.")
            time.sleep(INTER_CALL_DELAY)
            
            # 2. Verbose greedy
            rec["verbose"] = call_studio(q + " Answer in one complete sentence.")
            time.sleep(INTER_CALL_DELAY)
            
            # 3. K samples estocásticos
            rec["samples"] = []
            for _ in range(K_SAMPLES):
                rec["samples"].append(call_studio(q + " Answer with only the answer, no explanation."))
                time.sleep(INTER_CALL_DELAY)
                
            # 4. Umbrales de penalización
            rec["conf"] = {}
            for t in THRESH:
                rec["conf"][str(t)] = call_studio(conf_prompt(q, t))
                time.sleep(INTER_CALL_DELAY)
                
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            
            elapsed_q = time.time() - t_q_start
            if elapsed_q < MIN_QUESTION_CYCLE:
                time.sleep(MIN_QUESTION_CYCLE - elapsed_q)
                
            dt = time.time() - t0
            sample_txt = rec['short']['text'].strip()[:25].encode('ascii', 'replace').decode()
            print(f"[{MODEL}] [{qid+1}/120] {Q['tier']:4s} | {q[:35]} -> {sample_txt!r} (t_total={dt/60:.1f}min)", flush=True)
            
    print(f"=== COMPLETADO {MODEL} en {(time.time()-t0)/60:.1f} minutos ===", flush=True)
    print(">>> Ejecutando análisis comparativo multimodelo...", flush=True)
    try:
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "analyze_multimodel.py")], check=True)
        print(">>> Análisis multimodelo completado exitosamente.", flush=True)
    except subprocess.SubprocessError as e:
        print(f"Error al ejecutar analyze_multimodel.py: {e}", flush=True)

if __name__ == "__main__":
    main()
