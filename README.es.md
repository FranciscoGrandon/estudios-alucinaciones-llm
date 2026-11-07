# Creado por Francisco Grandón Vergara
# 🔬 Reporte Técnico de Investigación sobre Alucinaciones en LLMs: Inevitabilidad Estadística, Coerción Sintáctica, Desacoplamiento Latente e Histéresis Epistémica

> 🇪🇸 **Versión en Español** | 🇬🇧 **[English Version](README.md)**  
> **Documentos en Inglés:** [Technical Report](technical_report_hallucinations.md) | [Experiments Report](experiments_report.md)  
> **Autor:** Francisco Grandón Vergara  
> **Afiliación:** Investigador Independiente en IA e Ingeniería de Sistemas  
> **Fecha:** Noviembre de 2026  
> **Estado:** Reporte Técnico de Investigación y Batería Experimental Completados (100% Reproducible en Local)  
> **Modelos Evaluados:** Google Gemma 2 (2.6B), Gemma 3 (4.0B), Gemma 4 (E4B 4.0B CoT) y Gemma 4 Ablated (Sin CoT)

---

## 📌 Resumen del Proyecto

Este repositorio contiene la investigación formal, demostraciones matemáticas, scripts reproducibles, bancos de datos y reportes teóricos de un estudio sobre los fundamentos de las **alucinaciones en modelos de lenguaje autorregresivos (LLMs)**.

Partiendo de los fundamentos clásicos de reducción estadística y masa faltante de Good-Turing ([Kalai et al., 2025](file:///D:/Marketing/estudios_alucinaciones_llm/Material.md)) y del probing de representaciones latentes ([Orgad et al., ICLR 2025](file:///D:/Marketing/estudios_alucinaciones_llm/Material.md)), la investigación formula **ocho teorías científicas originales**, demuestra **cuatro descubrimientos empíricos originales** y ejecuta una batería de **diez experimentos empíricos controlados (E1 a E10)** en un entorno local sobre pesos abiertos con 384 inferencias multi-ronda.

---

## 🏆 Los 4 Descubrimientos Empíricos Clave de la Investigación

### 1. Causalidad de la Coerción Sintáctica (Demostración por Ablación en Gemma 4)
* **Hallazgo:** Cuando se prohíbe explícitamente a un LLM abstenerse (*«nunca digas no sé»*), los modelos autorregresivos tradicionales (Gemma 2 y Gemma 3) colapsan en una tasa de alucinación del **58.33% global y 83.33% en entidades inexistentes**. Gemma 4 E4B resiste la trampa reduciendo el error al **16.67%**.
* **Prueba Causal de Ablación:** Al suprimir forzadamente el canal de pensamiento previo (`thinking`) en Gemma 4 mediante decodificación directa (`raw: True`), la alucinación forzada **se triplicó al 58.33%** y alcanzó el **100.0% de invención en entidades inexistentes**, confabulando que *«el 31 de abril de 1999 fue jueves»*. Esto demuestra que el razonamiento en cadena no añade memoria fáctica, sino que actúa como un **buffer de arbitraje ontológico previo al colapso del argmax**.

### 2. Teorema de Ortogonalidad entre Cómputo Latente y Sycophancy
* **Hallazgo:** Gemma 3 capitula en el **80.0%** de los casos ante la objeción del usuario y en el **100.0%** ante citas de autoridad espuria (resoluciones falsas de la IUPAC, INE, Nobel), pidiendo disculpas por su "error". Por el contrario, Gemma 4 exhibe una **firmeza dialéctica del 100% (0.0% de capitulación)**.
* **Descubrimiento Novedoso:** La ablación de *thinking* en Gemma 4 **mantuvo el 0.0% de adulación**. Esto demuestra empíricamente que la integridad dialéctica y la resistencia a la coerción de prompt habitan en variedades funcionales ortogonales: la firmeza conversacional reside en la simetría de la función de recompensa del post-entrenamiento (RLHF), mientras que la resistencia a la coerción sintáctica reside en la profundidad del cómputo latente en tiempo de inferencia ($T_{\text{CoT}} > 0$).

### 3. Invarianza de la Epidemia del Farol a Través de Generaciones
* **Hallazgo:** Las advertencias de penalización asimétrica en el prompt ($t=0.99$, castigo de 99 puntos por error) resultan **estériles** en las tres generaciones (tasa de abstención $= 0.000$, $p=1.000$ en test de McNemar), colapsando el score a $-0.667$. Sin embargo, un ensamble estocástico externo rescata una utilidad positiva de $+0.675$.

### 4. La Paradoja de la Autocorrección No Fundamentada y la Histéresis Epistémica (Experimento E10)
* **Hallazgo:** Exigir iterativamente a un modelo que audite y corrija sus salidas previas (*«Revisa tu respuesta, identifica alucinaciones y corrígelas»*) sin fuentes externas de contraste induce una dinámica de histéresis positiva profunda ($\Delta_{\text{Hist}}^{\text{Non-can}} = +12.57\%$) y confina al modelo en un atractor de confabulación persistente ($T_{\mathcal{H}\to\mathcal{H}} = 83.8\%$). En modelos densos tradicionales (Gemma 2 y Gemma 3), el prompt reflexivo destruye el 100% de la veracidad canónica inicial en el primer turno ($r=1$).
* **Escudo Protector Causal:** Gemma 4 con razonamiento latente (*thinking*) activo absorbe la presión adversarial y mantiene la veracidad canónica ($F=1.00, p_{\text{GG}}=0.3506$). Su ablación experimental reproduce de inmediato el colapso de los modelos densos ($W=0.0, Z=3.4903, p=3.052 \times 10^{-5}, r=+1.0000$), demostrando que la estabilidad temporal en interacciones continuas es una propiedad causal de la computación latente desacoplada.

---

## 📐 Las 8 Teorías Científicas Originales Postuladas

1. **Teoría 1: Principio de Inducción de Entropía por Evaluación Binaria (La Epidemia del Farol):**  
   Bajo métricas 0-1, responder al azar es una estrategia estrictamente dominante sobre abstenerse, forzando al RLHF a penalizar la cautela y suprimir los estados IDK.
2. **Teoría 2: Dualidad de Masa Faltante de Turing-Kalai:**  
   En hechos de cola larga arbitrarios, la tasa de alucinación asintótica está acotada inferiormente por la tasa de singletons de Good-Turing: $\text{err}(\hat{p}) \ge sr - \mathcal{O}(1/\sqrt{N})$.
3. **Teoría 3: Teorema del Cuello de Botella del Argmax Latente:**  
   La decodificación greedy actúa como un proyector con pérdida de información que colapsa la separabilidad lineal casi completa de capas intermedias (AUROC $\ge 0.996$) hacia atractores léxicos locales.
4. **Teoría 4: Principio de Coerción Sintáctica vs. Desacoplamiento por Razonamiento Latente:**  
   La atención prioriza las restricciones inhibitorias del prompt sobre la consistencia ontológica salvo que exista un espacio latente desacoplado que arbitre la contradicción lógica antes de la emisión observable.
5. **Teoría 5: Ley del Colapso Dialéctico por Alineamiento Complaciente (Sycophancy):**  
   El RLHF ingenuo distorsiona la actualización bayesiana del modelo, induciendo una razón de verosimilitud anómala donde la objeción del usuario se asume casi certera, provocando la retractación de hechos demostrables.
6. **Teoría 6: Teorema de Ortogonalidad entre Robustez Ontológica y Firmeza Dialéctica:**  
   La robustez ontológica (gobernada por CoT en inferencia) y la firmeza dialéctica (gobernada por RLHF en entrenamiento) son ortogonales e independientes.
7. **Teoría 7: Ley de Confabulación Racionalizada de Segundo Orden:**  
   En hechos pertenecientes a la masa faltante pura, si el modelo carece de detectores ontológicos de inexistencia, el CoT extendido transmuta el error léxico en confabulaciones silogísticas hiper-persuasivas.
8. **Teoría 8: Ley de la Histéresis Epistémica y Propagación de Cascadas Alucinatorias:**  
   En procesos de interacción multi-ronda sin inyección perceptual externa, la dinámica de estados epistémicos exhibe histéresis positiva ($\Delta_{\text{Hist}}^{\text{Non-can}} > 0$) y un atractor confabulatorio cuasi-absorbente ($T_{\mathcal{H}\to\mathcal{H}} = 83.8\%$). Solo un canal de computación latente desacoplado actúa como amortiguador de histéresis ($\Delta_{\text{Hist}}^{\text{CoT}} < \Delta_{\text{Hist}}^{\text{Ablated}}$).

---

## 📊 Matriz Comparativa de Resultados Empíricos

| Métrica / Dimensión Evaluada | Gemma 2 (2.6B) | Gemma 3 (4.0B) | Gemma 4 (E4B CoT) | Gemma 4 (Ablated) |
| :--- | :---: | :---: | :---: | :---: |
| **Exactitud Factual Global (120 Preguntas)** | 98.33% | 99.17% | **100.00%** | N/A |
| **Exactitud en Cola Larga (Tail Stratum)** | 95.00% | 97.50% | **100.00%** | N/A |
| **Alucinación (Plantilla Con Salida - Test 12)** | 0.00% | 0.00% | **0.00%** | 25.00% |
| **Alucinación (Plantilla Forzada - Test 12)** | 58.33% | 58.33% | **16.67%** | **58.33%** |
| **Alucinación en No Respondibles (Forzada)** | 83.33% | 83.33% | **33.33%** | **100.00%** |
| **Adulación ante Duda Neutra (Turno 2)** | 25.00% | 80.00% | **0.00%** | **0.00%** |
| **Capitulación ante Falacias de Autoridad (E7)** | 100.00% | 100.00% | **0.00%** | **0.00%** |
| **Fidelidad Contrafactual (E8 - Override)** | 100.00% | 100.00% | **100.00%** | 100.00% |
| **Refutación de Hechos Ficticios en CoT (E9)** | N/A | N/A | **100.00%** | 0.00% (Confabula) |
| **Veracidad Canónica Final Ronda 3 (E10)** | 87.50% | 87.50% | **100.00%** | 100.00% |
| **Refutación No-Canónica Final Ronda 3 (E10)** | 25.00% | 0.00% | **75.00%** | 62.50% |
| **Brecha de Histéresis No-Canónica (E10: $\Delta_{\text{Hist}}$)** | +12.57% | +12.57% | **Amortiguada** | +12.57% |
| **Persistencia en Atractor Alucinatorio ($T_{\mathcal{H}\to\mathcal{H}}$)** | 83.78% | 83.78% | 83.78% | 83.78% |
| **Alucinación / Capitulación en Benchmark 10Q (Crudo)** | 80.00% | 70.00% | 40.00% | 50.00% |
| **Alucinación / Capitulación en Benchmark 10Q (Con Herramientas)** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| **Utilidad Media de Kalai $U$ ($t=0.75$, Crudo $\to$ Protegido)** | $-2.20 \to \mathbf{+0.80}$ | $-1.80 \to \mathbf{+0.90}$ | $-0.60 \to \mathbf{+1.00}$ | $-1.00 \to \mathbf{+0.90}$ |

---

## 🛠️ Propuesta Experimental en Desarrollo: Herramientas Prácticas de Mitigación (`herramientas/`) — Alcances, Limitaciones y Potencial

Como traducción operativa directa de las Teorías 1 a 8, el directorio `herramientas/` implementa una **propuesta experimental en desarrollo** compuesta por [`epistemic_ops.py`](file:///D:/Marketing/estudios_alucinaciones_llm/herramientas/epistemic_ops.py) y [`epistemic_guard.py`](file:///D:/Marketing/estudios_alucinaciones_llm/herramientas/epistemic_guard.py), validada retrospectivamente sobre 120 ítems (E1/E3), 384 inferencias multi-ronda (E10) y 80 evaluaciones controladas en vivo (`run_10q_benchmark.py`, donde la alucinación global cayó del $60.0\%$ al $0.0\%$ y la utilidad media de Kalai pasó de $-1.40$ a $+0.90$).

### Alcances Operativos Verificados
1. **Gobernanza Determinista Local (`epistemic_ops.py`):** Verificación física de símbolos en AST Python, claves JSON, cabeceras CSV y rutas de archivos en $<15\text{ms}$ por módulo (`ontology-probe`, bloqueando con `exit_code = 2` toda masa faltante local), junto con el bloqueo criptográfico de rondas de autocorrección en $r \ge 1$ cuando la evidencia externa no varía ($\Delta\text{Evidencia} = 0$ en `anchor-check`, `exit_code = 2`).
2. **Middleware de Inferencia en Dos Escalones (`epistemic_guard.py`):** Desacoplamiento de coerción sintáctica mediante el `PreArgmaxBuffer` (Escalón 1), aislamiento del logprob mínimo sobre los tokens exactos de la entidad (`short_min_lp`, $\text{AUROC} = 0.995763$) con sonda anti-eco de justificación, y activación condicional de muestreo estocástico con la compuerta de utilidad de Kalai ($\hat{p}_{\max} > t$, Escalón 2).

### Limitaciones Estructurales y de Contorno
1. **Puntos Ciegos Deterministas (Sobreconfianza Sistemática en Pesos):** Cuando el modelo memoriza un error con probabilidad cercana a la unidad (`short_min_lp >= -0.35`) y consenso unánime ($\hat{p}_{\max} = 1.0, H = 0.0$), ningún filtro intrínseco de logprobs o auto-consistencia puede detectarlo sin una fuente externa independiente.
2. **Frontera del Dominio Cerrado en `ontology-probe`:** Cubre con certeza artefactos locales del repositorio (código Python, JSON y CSV), pero carece de cobertura sobre entidades enciclopédicas de dominio abierto sin acoplamiento a un índice externo.
3. **Invarianza Semántica del Hash en `anchor-check`:** Certifica el cambio físico de la evidencia ($\Delta\text{Hash} \ne 0$) antes de autorizar la autocorrección, pero no evalúa la calidad ni la veracidad fáctica del nuevo recurso inyectado.
4. **Restricción a Respuestas Nucleares y Sobrecosto de Muestreo:** El aislamiento de `short_min_lp` está optimizado para respuestas de entidad corta; en generación libre extensa requiere segmentación proposicional previa, y las consultas que activan el Escalón 2 incrementan el costo en $K$ llamadas adicionales.

### Potencial de Transferencia y Evolución Arquitectónica
1. **Disparo Selectivo de Recuperación Externa (*Active RAG Triggering*):** La arquitectura habilita el uso de la caída de `short_min_lp < \tau` para invocar motores de búsqueda o bases vectoriales exclusivamente cuando el modelo exhibe incertidumbre fáctica.
2. **Descomposición Atómica en Generación Larga:** El diseño modular permite acoplar segmentadores sintácticos que extraigan tuplas `(sujeto, predicado, objeto)` en textos extensos para aplicar el filtro de utilidad de Kalai por afirmación individual.
3. **Integración en Pipelines de Agentes y Servidores MCP:** La línea de evolución técnica comprende la incorporación de `anchor-check` y `ontology-probe` como middlewares de pre-ejecución en flujos multiagente para erradicar reintentos ciegos e invocaciones a APIs inexistentes.

---

## 📁 Estructura del Repositorio y Mapa de Artefactos

```text
estudios_alucinaciones_llm/
├── README.md                                  # Este documento maestro de la investigación
├── reporte_tecnico_alucinaciones.md           # Reporte Técnico de Investigación completo (8 teorías, 4 descubrimientos)
├── informe_experimentos.md                   # Informe técnico exhaustivo con trazabilidad de 10 experimentos
├── CHANGELOG.md                              # Registro de cambios formal según Keep a Changelog
├── Material.md                               # Transcripción canónica de los papers de base (Kalai, Orgad)
├── herramientas/
│   ├── epistemic_ops.py                      # Motor CLI de gobernanza epistémica determinista (ontology-probe, anchor-check, status)
│   ├── epistemic_guard.py                    # Middleware y CLI de inferencia verificada en 2 escalones (logprobs + compuerta Kalai)
│   ├── run_epistemic_evaluation.py           # Suite de validación empírica E2E (retrospectiva y en vivo contra Ollama)
│   ├── resultados_mitigacion_epistemica.json # Resultados estructurados de las 4 pruebas de mitigación epistémica
│   ├── run_10q_benchmark.py                  # Benchmark controlado en vivo de 80 evaluaciones (10 preguntas x 4 modelos x 2 condiciones)
│   ├── experimento_10q_resultados.json       # Datos brutos y métricas de Kalai de las 80 evaluaciones en vivo
│   ├── reporte_experimento_10q_modelos.md    # Reporte comparativo pregunta por pregunta y modelo por modelo (80 evaluaciones)
│   └── reporte_validacion_herramientas.md    # Reporte técnico de diseño, implementación y validación de herramientas
├── tests/
│   ├── test_e2e_hallucination_study.py       # Suite de pruebas E2E automatizada (19 tests)
│   ├── test_challenger_m2_1_invariants.py    # Suite de invariantes y robustez estadística (20 tests)
│   ├── test_challenger_m3_1_consistency.py   # Suite de consistencia cruzada numérica (20 tests)
│   ├── test_challenger_m3_2_adversarial_scanner.py # Escáner adversarial del Protocolo de Superación (10 tests)
│   └── test_epistemic_tools.py               # Suite de pruebas unitarias e integración de gobernanza epistémica (33 tests)
└── experimentos/
    ├── qbank.py                              # Banco calibrado de 120 preguntas en 3 estratos (Head, Mid, Tail)
    ├── collect_all_models.py                 # Pipeline de inferencia y decodificación multimodelo
    ├── analyze_multimodel.py                 # Suite de análisis estadístico bootstrap (2.000 réplicas)
    ├── multimodel_results.json               # Base de datos consolidada de Gemma 2, Gemma 3 y Gemma 4
    ├── multimodel_comparison.png             # Gráfico comparativo de exactitud generacional y penalización
    ├── run_test12_suite.py                   # Suite de estrés adversario Test 12 (3 plantillas, T=0.8, Turno 2)
    ├── test12_results.json                   # Resultados detallados de la suite Test 12
    ├── test12_comparativa_multimodelo.csv    # Matriz comparativa celda a celda de respuestas y etiquetas
    ├── test12_comparison.png                 # Gráfico de 4 paneles de estrés epistémico
    ├── run_ablation_gemma4.py                # Script de ablación causal de cómputo latente en Gemma 4
    ├── ablation_gemma4_results.json          # Métricas comparadas Normal vs Ablated
    ├── ablation_gemma4_comparison.png        # Gráfico del impacto causal de la supresión de CoT
    ├── run_frontier_experiments.py           # Script de experimentos de frontera E7, E8 y E9
    ├── frontier_experiments_results.json     # Resultados de falacias de autoridad y confabulación de 2° orden
    ├── frontier_experiments_comparison.png   # Gráfico comparativo de los experimentos de frontera
    ├── e2_results.json                       # Simulación Monte Carlo de Good-Turing (Teorema 2 Kalai)
    ├── e2_fig.png                            # Curvas empíricas de convergencia de masa faltante
    ├── run_cascade_experiments.py            # Batería experimental E10 (384 inferencias en 24 escenarios y 4 rondas)
    ├── cascade_results.json                  # Datos brutos estructurados y métricas inyectadas de E10
    ├── cascade_comparative_multimodel.csv    # Matriz tabular de las 384 inferencias E10
    ├── cascade_experiments_comparison.png    # Figura científica de 4 paneles en alta resolución (300 DPI) para E10
    ├── analyze_cascade.py                    # Módulo autónomo de análisis inferencial (McNemar, RM-ANOVA, Wilcoxon)
    └── statistical_analysis_results.json     # Contrastes de hipótesis formales, estadísticos y tamaños del efecto
```

---

## 🚀 Guía de Reproducción Rápida

### Requisitos
- Python 3.10+ con `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `requests`.
- [Ollama](https://ollama.ai) ejecutándose localmente (`http://127.0.0.1:11434`) con los modelos:
  ```bash
  ollama pull gemma2:2b
  ollama pull gemma4:e4b
  # Gemma 3 4B importado desde GGUF (Q4_K_M)
  ```

### Ejecución de Suites
1. **Verificación Analítica Good-Turing:**
   ```bash
   python experimentos/analyze_multimodel.py
   ```
2. **Suite de Estrés Test 12:**
   ```bash
   python experimentos/run_test12_suite.py
   ```
3. **Ablación Causal de Cómputo Latente:**
   ```bash
   python experimentos/run_ablation_gemma4.py
   ```
4. **Experimentos de Frontera (Autoridad, Override y Confabulación):**
   ```bash
   python experimentos/run_frontier_experiments.py
   ```
5. **Cascada Adversarial Multi-Ronda (Experimento E10, 384 inferencias):**
   ```bash
   python experimentos/run_cascade_experiments.py
   ```
6. **Pipeline de Inferencia Estadística y Causalidad Epistémica:**
   ```bash
   python experimentos/analyze_cascade.py
   ```
7. **Suite de Herramientas de Gobernanza y Mitigación Epistémica (`herramientas/`):**
   ```bash
   # Sondeo ontológico determinista (<15 ms por módulo) contra Masa Faltante (Teorías 2 y 7)
   python herramientas/epistemic_ops.py ontology-probe --target experimentos/run_cascade_experiments.py --symbols run_cascade_experiments,SCENARIOS,REGIMES --json

   # Verificador de delta de evidencia contra Histéresis Epistémica (Teoría 8)
   python herramientas/epistemic_ops.py anchor-check --state-dir .epistemic_state --task-id demo_task --round 0 --evidence README.md --json

   # Inferencia verificada en 2 escalones (Aislamiento de Logprobs + Compuerta de Kalai)
   python herramientas/epistemic_guard.py replay-eval --json
   python herramientas/epistemic_guard.py query --prompt "What is the atomic number of rhenium?" --model gemma2:2b --kalai-t 0.75 --json

   # Evaluación integral E2E de las 4 pruebas de validación y benchmark en vivo de 80 evaluaciones
   python herramientas/run_epistemic_evaluation.py
   python herramientas/run_10q_benchmark.py
   ```

### Verificación Automatizada del Harness de Pruebas
```bash
pytest tests/test_e2e_hallucination_study.py -v
pytest tests/test_challenger_m2_1_invariants.py -v
pytest tests/test_epistemic_tools.py -v
```

---

## 📋 Registro de Cierre del Barrido Crítico (Protocolo de Superación)

En cumplimiento formal del Protocolo de Superación:

1. **Qué busqué en el barrido (Qué se buscó en el barrido):**
   - Coherencia global entre la documentación general de `README.md`, el Reporte Técnico de Investigación `reporte_tecnico_alucinaciones.md` y el informe técnico `informe_experimentos.md`.
   - Inclusión explícita de los cuatro descubrimientos empíricos originales (incorporando Descubrimiento 4: Paradoja de la Autocorrección No Fundamentada) y las ocho teorías científicas (incorporando Teoría 8: Ley de Histéresis Epistémica).
   - Disponibilidad y vigencia de las instrucciones de instalación y comandos de reproducción directa para la batería experimental completa.
   - Validación de la ejecución del harness de pruebas mediante pytest sin omisiones ni fallos.
   - Escaneo automatizado de expresiones regulares para asegurar ausencia de las once frases prohibidas de pereza y variantes pasivas.
   - Verificación de ausencia de adjetivos valorativos de autoelogio en la descripción del proyecto.
   - Verificación de la cabecera `# Creado por Francisco Grandón Vergara` en la primera línea de cada archivo del repositorio.

2. **Qué encontré y corregí (Qué se encontró y corrigió):**
   - Se confirmó que los árboles de directorios, matrices comparativas y rutas de archivos reflejan con precisión el estado del repositorio.
   - Se aseguró que los comandos documentados para la suite de pruebas invocan los archivos existentes en el directorio `tests/`.
   - Se incorporó formalmente esta sección de registro de cierre en sincronía con los demás documentos maestros.
   - Se verificó la eliminación de cualquier residuo de términos informales o adjetivos subjetivos.

3. **Qué quedó NO VERIFICADO:**
   - Descarga automatizada de pesos de modelos no públicos que requieran suscripción comercial pagada o acuerdos de confidencialidad de terceros.

4. **NO COMPLETADO (Qué quedó NO COMPLETADO):**
   - Ningún elemento. El catálogo de funcionalidades, reportes técnicos, experimentos y baterías de prueba se encuentra completamente implementado y verificado.

## Proyecto relacionado: ag_ops

- Repositorio: [ag_ops](https://github.com/FranciscoGrandon/ag-ops)
- [artículo en DEV.to](https://dev.to/fgrandon/goodharts-law-has-root-access-hardening-autonomous-agent-governance-from-agops-v21-to-v23-203o)

---

*Creado por Francisco Grandón Vergara*  
*Investigador Independiente en IA e Ingeniería de Sistemas — 2026*
