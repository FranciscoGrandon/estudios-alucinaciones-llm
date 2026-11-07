# Creado por Francisco Grandón Vergara
"""E2a: Cota de singleton-rate (Teorema 2, Kalai et al. 2025) en el modelo de 'Hechos Arbitrarios'.
E2b: Corolario 2 (trigram) - error >= 1/2 con prompts c1,c2 indistinguibles.
E2c: Estimador Good-Turing: sr vs masa faltante real.
Todo con numpy, semilla fija. Salidas: e2_results.json, e2_fig.png
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260611)
OUT = r"D:\Marketing\estudios_alucinaciones_llm\experimentos"
res = {}

def run(nprompts, zipf_s, N, L=365, reps=200):
    ranks = np.arange(1, nprompts + 1)
    mu = ranks ** (-zipf_s); mu /= mu.sum()
    rows = []
    for _ in range(reps):
        sample = rng.choice(nprompts, size=N, p=mu)
        counts = np.bincount(sample, minlength=nprompts)
        sr = (counts == 1).sum() / N
        mm = mu[counts == 0].sum()
        err_expected = mm * (1 - 1 / L)
        test = rng.choice(nprompts, size=20000, p=mu)
        unseen = counts[test] == 0
        guess_ok = rng.random(unseen.sum()) < (1 / L)
        err_emp = (unseen.sum() - guess_ok.sum()) / len(test)
        bound = sr - 2 / (L - 1) - (35 + 6 * math.log(N)) / math.sqrt(N)
        rows.append((sr, mm, err_emp, err_expected, bound))
    return np.array(rows)

configs = []
for N in [500, 2000, 10000, 50000, 200000]:
    for s in [0.8, 1.0, 1.3]:
        r = run(nprompts=100000, zipf_s=s, N=N)
        sr, mm, err_emp, err_exp, bound = r.mean(0)
        viol = float((r[:, 2] < r[:, 4]).mean())
        configs.append(dict(N=N, zipf_s=s, sr=sr, missing_mass=mm, err_emp=err_emp,
                            err_expected=err_exp, bound_thm2=bound, violation_rate=viol,
                            gt_gap_mean=float((r[:, 0] - r[:, 1]).mean()),
                            gt_abs_gap_max=float(np.abs(r[:, 0] - r[:, 1]).max())))
res["E2a_arbitrary_facts"] = configs
print("N, s, sr, MM, err_emp, bound, violation_rate")
for c in configs:
    print(c["N"], c["zipf_s"], round(c["sr"], 4), round(c["missing_mass"], 4), round(c["err_emp"], 4), round(c["bound_thm2"], 4), c["violation_rate"])

def trigram_experiment(n_train=5000, n_test=20000, ncontexts=2):
    c_train = rng.integers(0, ncontexts, n_train)
    r_train = c_train.copy()
    p_hat = np.bincount(r_train, minlength=ncontexts) / n_train
    c_test = rng.integers(0, ncontexts, n_test)
    r_sample = rng.choice(ncontexts, size=n_test, p=p_hat)
    err_sample = float((r_sample != c_test).mean())
    r_arg = np.full(n_test, p_hat.argmax())
    err_arg = float((r_arg != c_test).mean())
    return err_sample, err_arg

tri = []
for k in [2, 3, 5, 10]:
    runs = [trigram_experiment(ncontexts=k) for _ in range(30)]
    tri.append(dict(C=k, err_sampling=float(np.mean([a for a, b in runs])),
                    err_argmax=float(np.mean([b for a, b in runs])),
                    theory=1 - 1 / k))
res["E2b_trigram_corollary2"] = tri
print("trigram:", tri)

def full_context_error(ncontexts=2, n_train=5000, n_test=20000):
    c_train = rng.integers(0, ncontexts, n_train)
    table = {int(c): int(c) for c in c_train}
    c_test = rng.integers(0, ncontexts, n_test)
    pred = np.array([table.get(int(c), -1) for c in c_test])
    return float((pred != c_test).mean())
res["E2b_control_full_window_error"] = full_context_error()
print("control ventana completa:", res["E2b_control_full_window_error"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for s, m in zip([0.8, 1.0, 1.3], ["o", "s", "^"]):
    sub = [c for c in configs if c["zipf_s"] == s]
    ax[0].plot([c["N"] for c in sub], [c["sr"] for c in sub], m + "-", label=f"sr (Zipf s={s})")
    ax[0].plot([c["N"] for c in sub], [c["missing_mass"] for c in sub], m + "--", alpha=.5, label=f"masa faltante s={s}")
ax[0].set_xscale("log"); ax[0].set_xlabel("N muestras"); ax[0].set_ylabel("tasa")
ax[0].set_title("Good-Turing: singleton rate vs masa faltante"); ax[0].legend(fontsize=6)
ks = [t["C"] for t in tri]
ax[1].plot(ks, [t["err_sampling"] for t in tri], "o-", label="trigram (muestreo)")
ax[1].plot(ks, [t["err_argmax"] for t in tri], "s-", label="trigram (argmax)")
ax[1].plot(ks, [1 - 1 / k for k in ks], "k--", label="1-1/C (teoria)")
ax[1].axhline(0.5, color="r", ls=":", label="Cor.2: >=1/2")
ax[1].set_xlabel("C contextos indistinguibles"); ax[1].set_ylabel("error de generacion")
ax[1].set_title("Corolario 2 (trigram)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT + r"\e2_fig.png", dpi=140)
json.dump(res, open(OUT + r"\e2_results.json", "w"), indent=1)
print("OK")
