# Creado por Francisco Grandón Vergara
"""Análisis comparativo multimodelo integral (5 modelos):
Locales:
  1. gemma2:2b (Google 2.6B)
  2. gemma3:4b (Google 4B Q4)
  3. gemma4:e4b (Google E4B)
Remotos OpenRouter:
  4. liquid/lfm-2.5-2.6b:free (Liquid AI 2.6B)
  5. nvidia/nemotron-3-ultra-550b-a55b:free (NVIDIA 550B MoE)
"""
import json, os, re, unicodedata, math
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

D = os.path.dirname(__file__)
rng = np.random.default_rng(42)

ALL_MODELS = [
    ("gemma2:2b", "raw_gemma2_2b.jsonl", "Gemma 2 (2.6B, Local)", True),
    ("gemma3:4b", "raw_gemma3_4b.jsonl", "Gemma 3 (4B, Local)", True),
    ("gemma4:e4b", "raw_gemma4_e4b.jsonl", "Gemma 4 (E4B, Local)", True)
]
TIERS = ["head", "mid", "tail"]
B, P = 2000, 5000

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[\*\_`]", "", s).strip()

def correct(text, pat):
    return bool(re.search(pat, norm(text)))

def is_idk(text):
    t = norm(text)
    return bool(re.search(r"i don.?t know|\bidk\b|i do not know|cannot answer|can.?t answer|not sure|unable to", t))

def key(text, pat):
    if correct(text, pat): return "CORRECT"
    if is_idk(text): return "IDK"
    t = re.sub(r"[^a-z0-9 ]", "", norm(text).split("\n")[0])
    t = re.sub(r"^the ", "", t).strip()
    return t

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def boot_ci(fn, idx_n, reps=B):
    v = [fn(rng.integers(0, idx_n, idx_n)) for _ in range(reps)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def lps(rec, skip_ws=True):
    return np.array([lp for tok, lp in rec.get("lp", []) if not (skip_ws and tok.strip() == "")] or [0.0])

def analyze_model(tag, file_name, display_name, has_logprobs):
    path = os.path.join(D, file_name)
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    N = len(rows)
    if N == 0:
        return None

    tier = np.array([r["tier"] for r in rows])
    g_ok = np.array([correct(r["short"]["text"], r["pat"]) for r in rows])
    g_idk = np.array([is_idk(r["short"]["text"]) for r in rows])
    v_ok = np.array([correct(r["verbose"]["text"], r["pat"]) for r in rows])

    # 1. Estratificación
    e_tier = {}
    for t in TIERS:
        m = (tier == t)
        k = int(g_ok[m].sum()); n = int(m.sum())
        lo, hi = wilson(k, n)
        e_tier[t] = dict(n=n, acc=k / max(n, 1), err=1.0 - k / max(n, 1), ci=[lo, hi])

    ordv = np.array([TIERS.index(t) for t in tier])
    rho_obs = stats.spearmanr(ordv, ~g_ok).statistic
    perm = [stats.spearmanr(rng.permutation(ordv), ~g_ok).statistic for _ in range(P)]
    p_trend = float((np.sum(np.array(perm) >= rho_obs) + 1) / (P + 1))

    # 2. Detectores y AUROC
    samples = [r.get("samples", []) for r in rows]
    cons = np.array([np.mean([key(s["text"], r["pat"]) == key(r["short"]["text"], r["pat"]) for s in smp]) if smp else 0.5 for r, smp in zip(rows, samples)])
    
    aucs = {}
    delta_auc = dict(delta=0.0, ci=[0.0, 0.0])
    y = g_ok.astype(int)

    if has_logprobs and 0 < y.sum() < N:
        det = {
            "short_min_lp": np.array([lps(r["short"]).min() for r in rows]),
            "short_mean_lp": np.array([lps(r["short"]).mean() for r in rows]),
            "verbose_mean_lp": np.array([lps(r["verbose"]).mean() for r in rows]),
            "autoconsistencia": cons
        }
        for name, score_vec in det.items():
            auc = roc_auc_score(y, score_vec)
            lo, hi = boot_ci(lambda ix: roc_auc_score(y[ix], score_vec[ix]) if 0 < y[ix].sum() < len(ix) else 0.5, N)
            aucs[name] = dict(auc=float(auc), ci=[lo, hi])
        a, b = det["short_min_lp"], det["verbose_mean_lp"]
        diffs = [roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]) for _ in range(B) if 0 < y[ix := rng.integers(0, N, N)].sum() < N]
        if diffs:
            delta_auc = dict(delta=float(roc_auc_score(y, a) - roc_auc_score(y, b)),
                             ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
    elif 0 < y.sum() < N:
        auc = roc_auc_score(y, cons)
        lo, hi = boot_ci(lambda ix: roc_auc_score(y[ix], cons[ix]) if 0 < y[ix].sum() < len(ix) else 0.5, N)
        aucs["autoconsistencia"] = dict(auc=float(auc), ci=[lo, hi])

    # 3. Taxonomía de errores
    def taxo(r):
        smp = r.get("samples", [])
        if not smp: return "sin_samples"
        keys = [key(s["text"], r["pat"]) for s in smp]
        c = Counter(keys); K = len(keys)
        nc = c.get("CORRECT", 0); idk = c.get("IDK", 0)
        inc = Counter({k: v for k, v in c.items() if k not in ("CORRECT", "IDK")})
        top_inc = inc.most_common(1)[0][1] if inc else 0
        ndist = len(inc) + (1 if nc else 0)
        if idk >= K / 2: return "A_refusa"
        if nc == K: return "B1_siempre_correcto"
        if nc >= K / 2: return "B2_mayormente_correcto"
        if top_inc >= K / 2: return "C1_consistentemente_incorrecto" if nc == 0 else "C2_consistentemente_incorrecto_con_aciertos"
        if nc >= 2 and top_inc >= 2 and abs(nc - top_inc) <= 1: return "D_dos_competidores"
        if ndist >= 4: return "E1_muchas_respuestas" if nc == 0 else "E2_muchas_respuestas_con_acierto"
        return "otro"

    tx = [taxo(r) for r in rows]
    cnt = Counter(tx)

    # 4. Conocimiento oculto
    wrong = ~g_ok
    anyc = np.array([any(correct(s["text"], r["pat"]) for s in smp) if smp else False for r, smp in zip(rows, samples)])
    n_wrong = int(wrong.sum())
    hidden_frac = float(anyc[wrong].mean()) if n_wrong > 0 else 0.0
    hidden_ci = list(wilson(int(anyc[wrong].sum()), n_wrong)) if n_wrong > 0 else [0.0, 0.0]

    def pick_majority(r):
        smp = r.get("samples", [])
        if not smp: return False
        c = Counter(key(s["text"], r["pat"]) for s in smp)
        return c.most_common(1)[0][0] == "CORRECT" if c else False

    maj = np.array([pick_majority(r) for r in rows])
    sel_acc = {
        "greedy": float(g_ok.mean()),
        "voto_mayoria": float(maj.mean()),
        "oraculo_muestreo": float(anyc.mean())
    }

    # 5. Abstención
    THR = [0.0, 0.5, 0.99]
    e1_res = {}
    for t in THR:
        resp = [r.get("conf", {}).get(str(t), {}).get("text", "") for r in rows]
        idk_vec = np.array([is_idk(x) for x in resp])
        ok_vec = np.array([correct(x, r["pat"]) and not i for x, r, i in zip(resp, rows, idk_vec)])
        ans = ~idk_vec; err = ans & ~ok_vec; pen = t / (1.0 - t) if t < 1.0 else 99.0
        score_vec = ok_vec.astype(float) - pen * err
        always = g_ok.astype(float) - pen * (~g_ok)
        e1_res[str(t)] = {
            "abstain_rate": float(idk_vec.mean()),
            "score_own": float(score_vec.mean()),
            "score_always": float(always.mean())
        }

    return {
        "model": tag,
        "display_name": display_name,
        "N": N,
        "has_logprobs": has_logprobs,
        "overall_accuracy": float(g_ok.mean()),
        "verbose_accuracy": float(v_ok.mean()),
        "tier_metrics": e_tier,
        "spearman_rho": float(rho_obs),
        "spearman_p": p_trend,
        "aucs": aucs,
        "delta_auc_exact_vs_verbose": delta_auc,
        "taxonomy": dict(cnt),
        "n_wrong": n_wrong,
        "hidden_knowledge_frac": hidden_frac,
        "hidden_knowledge_ci": hidden_ci,
        "selection_accuracy": sel_acc,
        "e1_rubrics": e1_res
    }

def main():
    results = {}
    print("Iniciando análisis multimodelo integral (5 modelos)...")
    for tag, fname, dname, has_lp in ALL_MODELS:
        res = analyze_model(tag, fname, dname, has_lp)
        if res:
            results[tag] = res
            print(f"  [{dname}] N={res['N']} Acc={res['overall_accuracy']:.3f} Errores={res['n_wrong']} Oculto={res['hidden_knowledge_frac']:.2f}")
        else:
            print(f"  [{dname}] Pendiente o sin datos.")

    out_json = os.path.join(D, "multimodel_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Guardado {out_json}")

    # Graficar los modelos que tengan datos completos (N >= 60)
    ready = [m for m in results if results[m]["N"] >= 100]
    if len(ready) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        m_labels = [results[m]["display_name"] for m in ready]
        x = np.arange(len(TIERS))
        w = 0.8 / len(ready)

        # Gráfico 1: Exactitud por estrato
        ax1 = axes[0, 0]
        for i, m in enumerate(ready):
            accs = [results[m]["tier_metrics"][t]["acc"] for t in TIERS]
            errs = [[results[m]["tier_metrics"][t]["acc"] - results[m]["tier_metrics"][t]["ci"][0] for t in TIERS],
                    [results[m]["tier_metrics"][t]["ci"][1] - results[m]["tier_metrics"][t]["acc"] for t in TIERS]]
            ax1.bar(x + i * w, accs, w, yerr=errs, label=m_labels[i], capsize=3)
        ax1.set_xticks(x + w * (len(ready) - 1) / 2)
        ax1.set_xticklabels(["Head (Frecuente)", "Mid (Intermedio)", "Tail (Cola Larga)"])
        ax1.set_ylim(0.65, 1.05)
        ax1.set_ylabel("Exactitud Factual (IC Wilson 95%)")
        ax1.set_title("Exactitud por Estrato de Frecuencia")
        ax1.legend(loc="lower left", fontsize=8)

        # Gráfico 2: Ganancia por decodificación (Greedy vs Mayoría vs Oráculo)
        ax2 = axes[0, 1]
        strats = ["greedy", "voto_mayoria", "oraculo_muestreo"]
        x_s = np.arange(len(strats))
        for i, m in enumerate(ready):
            s_accs = [results[m]["selection_accuracy"].get(s, 0.0) for s in strats]
            ax2.bar(x_s + i * w, s_accs, w, label=m_labels[i])
        ax2.set_xticks(x_s + w * (len(ready) - 1) / 2)
        ax2.set_xticklabels(["Greedy", "Voto Mayoría", "Oráculo Muestreo"])
        ax2.set_ylim(0.8, 1.02)
        ax2.set_ylabel("Exactitud Factual")
        ax2.set_title("Conocimiento Oculto: Potencial de Recuperación")
        ax2.legend(loc="lower left", fontsize=8)

        # Gráfico 3: Score bajo penalización t=0.99
        ax3 = axes[1, 0]
        ts = ["0.0", "0.5", "0.99"]
        for i, m in enumerate(ready):
            scores = [results[m]["e1_rubrics"].get(t, {}).get("score_own", 0.0) for t in ts]
            ax3.plot(ts, scores, "o-", label=m_labels[i])
        ax3.axhline(0, color="k", lw=0.8, ls=":")
        ax3.set_xlabel("Umbral de Penalización (t)")
        ax3.set_ylabel("Score Medio")
        ax3.set_title("Epidemia del Farol: Score ante Penalización t/(1-t)")
        ax3.legend(fontsize=8, loc="lower left")

        # Gráfico 4: Resumen de Errores y Conocimiento Latente
        ax4 = axes[1, 1]
        err_counts = [results[m]["n_wrong"] for m in ready]
        hidden_counts = [results[m]["hidden_knowledge_frac"] * results[m]["n_wrong"] for m in ready]
        x_m = np.arange(len(ready))
        ax4.bar(x_m, err_counts, 0.4, label="Errores Totales Greedy", color="#e45756")
        ax4.bar(x_m, hidden_counts, 0.4, label="Acierto Latente en Muestreo", color="#54a24b")
        ax4.set_xticks(x_m)
        ax4.set_xticklabels([m.split()[0] for m in m_labels], rotation=15)
        ax4.set_ylabel("Cantidad de Preguntas")
        ax4.set_title("Errores Greedy vs Conocimiento Latente Recuperado")
        ax4.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        out_fig = os.path.join(D, "multimodel_comparison.png")
        plt.savefig(out_fig, dpi=150)
        print(f"Gráfico multimodelo guardado en {out_fig}")

if __name__ == "__main__":
    main()
