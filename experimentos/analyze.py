# Creado por Francisco Grandón Vergara
"""Análisis E1 (umbral de penalización) y E3 (tier/singleton-proxy, detectores, taxonomía, conocimiento oculto)
sobre raw_gemma2_2b.jsonl. Bootstrap (2000) y tests de permutación (5000), semilla fija.
Salida: analysis_results.json, e1_e3_fig.png, analysis_log.txt (stdout)
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
rng = np.random.default_rng(7)
rows = [json.loads(l) for l in open(os.path.join(D, "raw_gemma2_2b.jsonl"), encoding="utf-8")]
N = len(rows)
TIERS = ["head", "mid", "tail"]
B, P = 2000, 5000

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[\*\_`]", "", s).strip()
def correct(text, pat): return bool(re.search(pat, norm(text)))
def is_idk(text):
    t = norm(text)
    return bool(re.search(r"i don.?t know|\bidk\b|i do not know|cannot answer|can.?t answer|not sure|unable to", t))
def key(text, pat):
    if correct(text, pat): return "CORRECT"
    if is_idk(text): return "IDK"
    t = re.sub(r"[^a-z0-9 ]", "", norm(text).split("\n")[0]); t = re.sub(r"^the ", "", t).strip()
    return t
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
def boot_ci(fn, idx_n, reps=B):
    v = [fn(rng.integers(0, idx_n, idx_n)) for _ in range(reps)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

out = {"N": N}
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

tier = np.array([r["tier"] for r in rows])
g_ok = np.array([correct(r["short"]["text"], r["pat"]) for r in rows])
g_idk = np.array([is_idk(r["short"]["text"]) for r in rows])
v_ok = np.array([correct(r["verbose"]["text"], r["pat"]) for r in rows])
say(f"N={N} greedy-short acc={g_ok.mean():.3f}  verbose acc={v_ok.mean():.3f} idk(short)={g_idk.mean():.3f}")

# ---------- E3a: error por estrato (proxy de frecuencia) ----------
e3a = {}
for t in TIERS:
    m = tier == t; k = int(g_ok[m].sum()); n = int(m.sum()); lo, hi = wilson(k, n)
    e3a[t] = dict(n=n, acc=k / n, err=1 - k / n, ci=[lo, hi])
    say(f"  tier {t}: n={n} acc={k/n:.3f} CI[{lo:.3f},{hi:.3f}]")
ordv = np.array([TIERS.index(t) for t in tier])
rho_obs = stats.spearmanr(ordv, ~g_ok).statistic
perm = [stats.spearmanr(rng.permutation(ordv), ~g_ok).statistic for _ in range(P)]
p_trend = float((np.sum(np.array(perm) >= rho_obs) + 1) / (P + 1))
say(f"  Spearman(tier ordinal, error)={rho_obs:.3f}  permutation p(one-sided)={p_trend:.4f}")
e3a["spearman_tier_error"] = float(rho_obs); e3a["perm_p_one_sided"] = p_trend
out["E3a_tier_error"] = e3a

# ---------- E3b: detectores (AUROC de correcto) ----------
def lps(rec, skip_ws=True):
    return np.array([lp for tok, lp in rec["lp"] if not (skip_ws and tok.strip() == "")] or [0.0])
cons = np.array([np.mean([key(s["text"], r["pat"]) == key(r["short"]["text"], r["pat"]) for s in r["samples"]]) for r in rows])
def cluster_entropy(r):
    c = Counter(key(s["text"], r["pat"]) for s in r["samples"]); p = np.array(list(c.values())) / sum(c.values())
    return -(p * np.log(p)).sum()
negent = np.array([-cluster_entropy(r) for r in rows])
det = {
    "verbose_mean_lp (todos los tokens)": np.array([lps(r["verbose"]).mean() for r in rows]),
    "verbose_min_lp (todos los tokens)": np.array([lps(r["verbose"]).min() for r in rows]),
    "short_mean_lp (tokens respuesta)": np.array([lps(r["short"]).mean() for r in rows]),
    "short_min_lp (tokens respuesta)": np.array([lps(r["short"]).min() for r in rows]),
    "short_first_token_lp": np.array([lps(r["short"])[0] for r in rows]),
    "autoconsistencia (K=10)": cons,
    "-entropia de clusters (K=10)": negent,
}
e3b = {}
y = g_ok.astype(int)
say("E3b AUROC (label = greedy short correcto):")
for name, s in det.items():
    auc = roc_auc_score(y, s)
    lo, hi = boot_ci(lambda ix: roc_auc_score(y[ix], s[ix]) if 0 < y[ix].sum() < len(ix) else 0.5, N)
    pm = [roc_auc_score(rng.permutation(y), s) for _ in range(P)]
    p = float((np.sum(np.array(pm) >= auc) + 1) / (P + 1))
    e3b[name] = dict(auc=float(auc), ci=[lo, hi], perm_p=p)
    say(f"  {name}: AUC={auc:.3f} CI[{lo:.3f},{hi:.3f}] p_perm={p:.4f}")
# diferencia pareada exacto vs verbose
a, b = det["short_min_lp (tokens respuesta)"], det["verbose_mean_lp (todos los tokens)"]
diffs = []
for _ in range(B):
    ix = rng.integers(0, N, N)
    if 0 < y[ix].sum() < N: diffs.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
e3b["delta_short_min_minus_verbose_mean"] = dict(delta=float(roc_auc_score(y, a) - roc_auc_score(y, b)),
                                                 ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
say("  delta AUC (short_min - verbose_mean):", e3b["delta_short_min_minus_verbose_mean"])
out["E3b_detectors"] = e3b

# ---------- E3c: taxonomía de errores (adaptada, K=10) ----------
def taxo(r):
    keys = [key(s["text"], r["pat"]) for s in r["samples"]]; c = Counter(keys); K = len(keys)
    nc = c.get("CORRECT", 0); idk = c.get("IDK", 0)
    inc = Counter({k: v for k, v in c.items() if k not in ("CORRECT", "IDK")})
    top_inc = inc.most_common(1)[0][1] if inc else 0
    ndist = len(inc) + (1 if nc else 0)
    if idk >= K / 2: return "A_refusa"
    if nc == K: return "B1_siempre_correcto"
    if nc >= K / 2: return "B2_mayormente_correcto"
    if top_inc >= K / 2: return "C1_consistentemente_incorrecto" if nc == 0 else "C2_consistentemente_incorrecto_con_aciertos"
    if nc >= 3 and top_inc >= 3 and abs(nc - top_inc) <= 1: return "D_dos_competidores"
    if ndist >= 5: return "E1_muchas_respuestas" if nc == 0 else "E2_muchas_respuestas_con_acierto"
    return "otro"
tx = [taxo(r) for r in rows]
cnt = Counter(tx); say("E3c taxonomía:", dict(cnt))
bytier = {t: dict(Counter(x for x, tt in zip(tx, tier) if tt == t)) for t in TIERS}
out["E3c_taxonomy"] = dict(overall=dict(cnt), by_tier=bytier)
say("  por estrato:", bytier)

# ---------- E3d: conocimiento oculto / selección ----------
def pick_majority(r):
    c = Counter(key(s["text"], r["pat"]) for s in r["samples"]); k = c.most_common(1)[0][0]
    return k == "CORRECT"
def pick_logprob(r):
    best, bs = None, -1e9
    for s in r["samples"]:
        m = lps(s).mean()
        if m > bs: bs, best = m, s
    return correct(best["text"], r["pat"])
maj = np.array([pick_majority(r) for r in rows]); lpb = np.array([pick_logprob(r) for r in rows])
rnd = np.array([np.mean([correct(s["text"], r["pat"]) for s in r["samples"]]) for r in rows])
anyc = np.array([any(correct(s["text"], r["pat"]) for s in r["samples"]) for r in rows])
wrong = ~g_ok
hidden = float(anyc[wrong].mean()) if wrong.sum() else float("nan")
lo, hi = wilson(int(anyc[wrong].sum()), int(wrong.sum()))
say(f"E3d greedy-wrong n={wrong.sum()}; con respuesta correcta en >=1 de 10 muestras: {hidden:.3f} CI[{lo:.3f},{hi:.3f}]")
def signflip(a, b):
    d = a.astype(float) - b.astype(float); obs = d.mean()
    ps = [(d * rng.choice([-1, 1], len(d))).mean() for _ in range(P)]
    return float(obs), float((np.sum(np.abs(ps) >= abs(obs)) + 1) / (P + 1))
sel = {"greedy": float(g_ok.mean()), "azar(K=10)": float(rnd.mean()), "voto_mayoria": float(maj.mean()),
       "max_mean_logprob": float(lpb.mean()), "oraculo_any": float(anyc.mean())}
say("  accuracy por estrategia:", sel)
cmp_ = {}
for nm, arr in [("voto_mayoria", maj), ("max_mean_logprob", lpb), ("azar", rnd)]:
    d, p = signflip(arr, g_ok); ix_ci = boot_ci(lambda ix: float((arr[ix].astype(float) - g_ok[ix]).mean()), N)
    cmp_[nm] = dict(delta_vs_greedy=d, ci=list(ix_ci), p_signflip=p); say(f"  {nm} - greedy: {d:+.3f} CI{ix_ci} p={p:.4f}")
out["E3d_hidden_knowledge"] = dict(n_wrong=int(wrong.sum()), frac_correct_in_samples=hidden, ci=[lo, hi], accuracy=sel, comparisons=cmp_)

# ---------- E1: umbrales de penalización ----------
THR = [0.0, 0.5, 0.75, 0.9, 0.99]
e1 = {}
say("E1 (score = (aciertos - pen*errores)/N, pen=t/(1-t)):")
own_abst = {}
for t in THR:
    resp = [r["conf"][str(t)]["text"] for r in rows]
    idk = np.array([is_idk(x) for x in resp]); ok = np.array([correct(x, r["pat"]) and not i for x, r, i in zip(resp, rows, idk)])
    ans = ~idk; err = ans & ~ok; pen = t / (1 - t)
    score_vec = ok.astype(float) - pen * err
    # políticas de referencia con los mismos datos
    always = g_ok.astype(float) - pen * (~g_ok)                         # responde siempre (greedy short)
    selfc = np.where(cons > t, g_ok.astype(float) - pen * (~g_ok), 0.0)  # responde si autoconsistencia > t
    oracle = np.where(g_ok, 1.0, 0.0)                                    # abstiene iff incorrecto
    e1[str(t)] = dict(pen=pen, abstain_rate=float(idk.mean()), answered=int(ans.sum()), correct=int(ok.sum()), wrong=int(err.sum()),
                      acc_answered=float(ok.sum() / max(ans.sum(), 1)), err_rate_answered=float(err.sum() / max(ans.sum(), 1)),
                      target_err=1 - t, score_own=float(score_vec.mean()), score_always=float(always.mean()),
                      score_selfconsistency=float(selfc.mean()), score_oracle=float(oracle.mean()),
                      own_minus_always_ci=list(boot_ci(lambda ix: float((score_vec[ix] - always[ix]).mean()), N)),
                      selfc_minus_own_ci=list(boot_ci(lambda ix: float((selfc[ix] - score_vec[ix]).mean()), N)))
    own_abst[t] = idk
    e = e1[str(t)]
    say(f"  t={t}: pen={pen:.1f} abst={e['abstain_rate']:.3f} acc|ans={e['acc_answered']:.3f} err|ans={e['err_rate_answered']:.3f}(obj<= {1-t:.2f}) "
        f"score own={e['score_own']:.3f} always={e['score_always']:.3f} selfcons={e['score_selfconsistency']:.3f} oracle={e['score_oracle']:.3f}")
# McNemar exacto abstención t=0.99 vs t=0
a0, a9 = own_abst[0.0], own_abst[0.99]
b01 = int(np.sum(~a0 & a9)); b10 = int(np.sum(a0 & ~a9))
pm = stats.binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0
e1["mcnemar_abstain_t0.99_vs_t0"] = dict(only_t099=b01, only_t0=b10, p=float(pm))
say(f"  McNemar abstención t=.99 vs t=0: {b01} vs {b10}  p={pm:.4g}")
# calibración conductual: abstiene más en preguntas que son incorrectas?
tt = 0.9; idk9 = own_abst[0.9]
auc_abst = roc_auc_score(~g_ok, idk9.astype(int)) if 0 < (~g_ok).sum() < N and 0 < idk9.sum() < N else float("nan")
e1["abstention_t0.9_AUC_for_predicting_wrong"] = float(auc_abst)
say(f"  AUC(abstención@t=.9 predice error de greedy) = {auc_abst:.3f}")
out["E1_threshold"] = e1

# ---------- figura ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
x = np.arange(3); acc = [e3a[t]["acc"] for t in TIERS]
ax[0, 0].bar(x, acc, yerr=[[e3a[t]["acc"] - e3a[t]["ci"][0] for t in TIERS], [e3a[t]["ci"][1] - e3a[t]["acc"] for t in TIERS]], color=["#4c78a8", "#f58518", "#e45756"])
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(TIERS); ax[0, 0].set_ylim(0, 1.05); ax[0, 0].set_ylabel("accuracy greedy (IC Wilson 95%)"); ax[0, 0].set_title("E3a: accuracy por estrato (proxy frecuencia)")
names = list(det); aucs = [e3b[n]["auc"] for n in names]
ax[0, 1].barh(range(len(names)), aucs, xerr=[[e3b[n]["auc"] - e3b[n]["ci"][0] for n in names], [e3b[n]["ci"][1] - e3b[n]["auc"] for n in names]], color="#54a24b")
ax[0, 1].axvline(0.5, color="k", ls=":"); ax[0, 1].set_yticks(range(len(names))); ax[0, 1].set_yticklabels(names, fontsize=7); ax[0, 1].set_xlabel("AUROC (IC bootstrap 95%)"); ax[0, 1].set_title("E3b: detectores de error")
ts = THR
for k, c, lab in [("score_own", "#e45756", "modelo (prompt con t)"), ("score_always", "#999", "responder siempre"),
                  ("score_selfconsistency", "#4c78a8", "autoconsistencia>t"), ("score_oracle", "#54a24b", "oraculo")]:
    ax[1, 0].plot(range(len(ts)), [e1[str(t)][k] for t in ts], "o-", color=c, label=lab)
ax[1, 0].set_xticks(range(len(ts))); ax[1, 0].set_xticklabels([f"t={t}" for t in ts]); ax[1, 0].set_ylabel("score medio"); ax[1, 0].axhline(0, color="k", lw=.5); ax[1, 0].legend(fontsize=7); ax[1, 0].set_title("E1: score con penalizacion t/(1-t)")
ax[1, 1].bar(list(sel.keys()), list(sel.values()), color="#b279a2"); ax[1, 1].set_ylim(0, 1.05); ax[1, 1].tick_params(axis="x", labelrotation=25, labelsize=8); ax[1, 1].set_title(f"E3d: estrategias (oculto: {hidden:.2f} de errores tiene la correcta en K=10)")
plt.tight_layout(); plt.savefig(os.path.join(D, "e1_e3_fig.png"), dpi=140)
json.dump(out, open(os.path.join(D, "analysis_results.json"), "w"), indent=1, default=float)
open(os.path.join(D, "analysis_log.txt"), "w", encoding="utf-8").write("\n".join(log))
print("OK")
