# Creado por Francisco Grandón Vergara
"""Genera gráficos científicos en alta resolución para el Test 12 de Alucinaciones.
Compara Gemma 2 (2.6B), Gemma 3 (4B) y Gemma 4 (E4B).
Salida: test12_comparison.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_EXP = os.path.dirname(__file__)
JSON_PATH = os.path.join(D_EXP, "test12_results.json")
OUT_FIG = os.path.join(D_EXP, "test12_comparison.png")

def main():
    if not os.path.exists(JSON_PATH):
        print("No existe test12_results.json aún.")
        return
        
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
        
    models = list(data.keys())
    m_labels = [data[m]["name"] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(models))
    w = 0.25
    
    # 1. Tasa de Alucinación por Plantilla (Impacto de la Presión)
    ax1 = axes[0, 0]
    plantillas = ["con_salida", "neutra", "forzada"]
    p_labels = ["Con Salida (Permite IDK)", "Neutra (Estándar)", "Forzada (Prohibido IDK)"]
    colors = ["#4c78a8", "#f58518", "#e45756"]
    
    for i, p in enumerate(plantillas):
        vals = [data[m]["metrics"][p]["tasa_alucinacion"] * 100 for m in models]
        ax1.bar(x + (i - 1) * w, vals, w, label=p_labels[i], color=colors[i])
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(m_labels, fontsize=10)
    ax1.set_ylabel("Tasa de Alucinación (%)")
    ax1.set_title("1. Tasa de Alucinación por Plantilla (Efecto Presión)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # 2. Alucinación en Preguntas No Respondibles (Sensibilidad a Falsedad/Inexistencia)
    ax2 = axes[0, 1]
    for i, p in enumerate(["con_salida", "forzada"]):
        vals = [data[m]["metrics"][p]["alucinacion_no_respondibles"] * 100 for m in models]
        ax2.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=f"No Respondibles ({p})", color=["#54a24b", "#e45756"][i])
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(m_labels, fontsize=10)
    ax2.set_ylabel("Alucinación en No Respondibles (%)")
    ax2.set_title("2. Invención ante Entidades Inexistentes y Premisas Falsas")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # 3. Inconsistencia Estocástica (T=0.8)
    ax3 = axes[1, 0]
    incons = [data[m]["metrics"]["inconsistencia_temp_0.8"] * 100 for m in models]
    bars = ax3.bar(x, incons, 0.5, color="#72b7b2")
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(m_labels, fontsize=10)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Preguntas Inconsistentes (%)")
    ax3.set_title("3. Inconsistencia Estocástica (Variabilidad entre 5 Muestras T=0.8)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    
    # 4. Adulación en Turno 2 (Sycophancy)
    ax4 = axes[1, 1]
    adul = [data[m]["metrics"]["adulacion_turno_2"] * 100 for m in models]
    bars4 = ax4.bar(x, adul, 0.5, color="#b279a2")
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(m_labels, fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("Tasa de Adulación (%)")
    ax4.set_title("4. Adulación en Turno 2 (Retractación ante Presión del Usuario)")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"Gráfico guardado en {OUT_FIG}")

if __name__ == "__main__":
    main()
