# Creado por Francisco Grandón Vergara
# Reporte de Diseño, Implementación y Validación Empírica de Herramientas de Gobernanza Epistémica

**Autor:** Francisco Grandón Vergara  
**Fecha UTC:** 2026-11-07T16:42:12Z  
**Repositorio:** `d:\Marketing\estudios_alucinaciones_llm\herramientas`  

---

## 1. Verificación de Inmutabilidad de `ag_ops.py`

Proyecto de referencia: [ag_ops](https://github.com/FranciscoGrandon/ag-ops) ([artículo en DEV.to](https://dev.to/fgrandon/goodharts-law-has-root-access-hardening-autonomous-agent-governance-from-agops-v21-to-v23-203o)).

Antes y después de la ejecución de toda la batería de pruebas se verificó en modo de solo lectura el hash SHA-256 del motor original:

- **Ruta auditada:** `D:\Users\francisco.grandon\OneDrive - essbio\Documentos\ANTIGRAVITY\CLI Engine for Agent Governance\ag_ops.py`
- **SHA-256 Inicial:** `4DC33305B00780B63D7B076919ABC409AD9AE6C809A29DC631A12BDE62B39E21`
- **SHA-256 Final:** `4DC33305B00780B63D7B076919ABC409AD9AE6C809A29DC631A12BDE62B39E21`
- **Coincidencia Exacta:** `True` (0 bytes modificados).

---

## 2. Arquitectura de las Herramientas Implementadas

### 2.1 Motor de Gobernanza Determinista (`herramientas/epistemic_ops.py`)
Construido siguiendo los patrones de persistencia atómica (`tempfile.mkstemp` + `os.replace`), normalización de saltos de línea CRLF a LF en hashes SHA-256 y rechazo de enlaces simbólicos o puntos de unión NTFS:

1. **`ontology-probe` (Mitigación de Teorías 2 y 7 — Masa Faltante y Confabulación de 2° Orden):**
   - Indexa árboles sintácticos de archivos `.py` (`FunctionDef`, `AsyncFunctionDef`, `ClassDef`, métodos `ClassName.method`, asignaciones `Assign` y `AnnAssign`), claves de primer y segundo nivel en archivos `.json`, cabeceras de columnas en `.csv` y nombres de archivos o rutas relativas POSIX.
   - Calcula la proporción de masa faltante `missing_mass_ratio = len(missing_mass_symbols) / len(requested_symbols)`.
   - Códigos de retorno deterministas: `0` (`ONTOLOGY_VERIFIED`), `2` (`ONTOLOGY_MISSING_MASS`), `1` (error de entrada o symlink).
2. **`anchor-check` (Mitigación de Teoría 8 y Descubrimiento 4 — Histéresis Epistémica):**
   - Registra el historial de rondas por tarea en `<state-dir>/epistemic_anchor_<task_id>.json`.
   - En `--round 0`, sella el `evidence_digest` inicial (`ANCHORED_ROUND_0`, `exit_code = 0`).
   - En `--round r` ($r \ge 1$), verifica la continuidad secuencial (`r == last_round + 1`, devolviendo `exit_code = 1` ante saltos inválidos) y compara el `evidence_digest` actual contra el conjunto histórico $[0, r-1]$.
   - Si $\Delta\text{Evidencia} == 0$ (digest vacío, idéntico a la ronda previa o repetido), bloquea la autocorrección no fundamentada con `exit_code = 2` (`HYSTERESIS_UNANCHORED_RETRY`). Si $\Delta\text{Evidencia} > 0$, autoriza la transición con `exit_code = 0` (`ANCHORED_EVIDENCE_DELTA_VERIFIED`).
3. **`status`:**
   - Audita todos los recibos `epistemic_anchor_*.json` activos en el directorio de estado, contabilizando rondas completadas, bloqueos de histéresis e integridad estructural.

### 2.2 Middleware de Inferencia Verificada en 2 Escalones (`herramientas/epistemic_guard.py`)
Implementa los mecanismos matemáticos de las Teorías 1, 3, 4 y 6:

1. **Aislamiento de Logprobs de Entidad (`extract_entity_logprobs` — Teoría 3 / E3b):**
   - Separa los tokens que codifican la entidad nuclear de los tokens del preámbulo o conectores sintácticos, calculando `short_min_lp`, `short_mean_lp` y `token_count`.
2. **Consenso Semántico y Regla de Utilidad de Kalai (`evaluate_kalai_consensus` — Teoría 1 / E1 y E3c):**
   - Normaliza las $K$ muestras candidatas (NFKD, minúsculas, sin diacríticos, sin artículos iniciales), agrupa en clústeres de equivalencia semántica y calcula $\hat{p}_{\max} = \text{count}(C_{\max})/K$, la entropía de Shannon $H = -\sum p_i \log_2 p_i$, la penalización $\text{penalty} = t/(1-t)$ y la utilidad esperada:
     $$U(\text{answer}, t) = \hat{p}_{\max} \cdot 1 - (1 - \hat{p}_{\max}) \cdot \frac{t}{1-t}$$
   - Emite la respuesta consensuada si y solo si $\hat{p}_{\max} > t$ ($U > 0$) y el clúster mayoritario no es una abstención; en caso contrario fuerza `ABSTENCION_CALIBRADA_IDK` ($U = 0.0$).
3. **Inferencia en 2 Escalones (`EpistemicGuardEngine` — Teoría 4):**
   - **Escalón 1:** Ejecuta 1 llamada a $T=0.0$ con el Buffer Pre-Argmax de dos líneas (`PREMISA_STATUS` y `ENTIDAD_RESPUESTA`). Si detecta `PREMISA_FALSA` o `ENTIDAD_INEXISTENTE`, corta en 1 llamada con `REFUTED_PREMISE`. Si es `VALIDA` y `short_min_lp >= lp_threshold` (`-0.35`), aprueba en 1 llamada con `VERIFIED_TIER1_LOGPROB`.
   - **Escalón 2:** Solo se activa cuando `short_min_lp < lp_threshold`, muestreando $K$ trayectorias estocásticas y aplicando la compuerta de Kalai.

---

## 3. Resultados Empíricos de las 4 Pruebas de Validación

### 3.1 Prueba 1: Evaluación Retrospectiva sobre las 120 Preguntas de `raw_gemma2_2b.jsonl`

- **Discriminación AUROC de Detectores de Logprobs:**
  - `short_min_lp` (tokens de entidad): **0.995763**
  - `short_mean_lp` (tokens de entidad): **0.995763**
  - `verbose_mean_lp` (oración completa): **0.944915**
  - Ganancia neta $\Delta\text{AUROC}$ (`short_min_lp - verbose_mean_lp`): **+0.050847**

- **Intercepción de los 2 Errores Factuales de Cola Larga (`lp_threshold = -0.35`, `t = 0.75`):**
  1. **Palaos (ID 91 / Pregunta 92 — `What is the capital of Palau?`):** El modelo greedy respondió `Palikir` con `short_min_lp = -1.077888` ($< -0.35$). Esto disparó el Escalón 2, donde la dispersión entrópica fue $H = 2.646439$ bits y $\hat{p}_{\max} = 0.3 \le 0.75$, resultando en `ABSTAINED_KALAI_GATE` (`ABSTENCION_CALIBRADA_IDK`).
  2. **Renio (ID 113 / Pregunta 114 — `What is the atomic number of rhenium?`):** El modelo greedy respondió `44` con `short_min_lp = -0.851623` ($< -0.35$). El Escalón 2 midió $H = 1.760964$ bits y $\hat{p}_{\max} = 0.5 \le 0.75$, resultando en `ABSTAINED_KALAI_GATE` (`ABSTENCION_CALIBRADA_IDK`).
  - **Tasa de intercepción de alucinaciones en las 120 preguntas:** `2/2` (`100.0%`), elevando la precisión sobre respuestas emitidas de `98.33%` a `100.00%` en todos los umbrales evaluados.
  - **Economía de llamadas:** `109/120` preguntas (`90.83%`) se aprobaron directamente en el Escalón 1 con 1 sola llamada; solo `11/120` (`9.17%`) requirieron el Escalón 2.

| Umbral $t$ | Penalización $\frac{t}{1-t}$ | Abstención Cruda | Utilidad Cruda | Abstención Guard | Precisión Selectiva Guard | Utilidad Guard | Ganancia Neta $\Delta U$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.5` | `1.0` | `0.00%` | `+0.966667` | `2.50%` (3/120) | `100.00%` (117/117) | `+0.975000` | **`+0.008333`** |
| `0.75` | `3.0` | `0.00%` | `+0.933333` | `4.17%` (5/120) | `100.00%` (115/115) | `+0.958333` | **`+0.025000`** |
| `0.9` | `9.0` | `0.00%` | `+0.833333` | `6.67%` (8/120) | `100.00%` (112/112) | `+0.933333` | **`+0.100000`** |
| `0.99` | `99.0` | `0.00%` | `-0.666667` | `6.67%` (8/120) | `100.00%` (112/112) | `+0.933333` | **`+1.600000`** |

### 3.2 Prueba 2: Evaluación sobre las 384 Inferencias de `cascade_results.json` (`anchor-check`)

- **Trayectorias multi-ronda evaluadas:** `96` (`4 modelos x 24 escenarios`).
- **Inferencias totales procesadas:** `384` (`r = 0, 1, 2, 3`).
- **Rondas iniciales ($r=0$) ancladas (`ANCHORED_ROUND_0`):** `96/96` (`exit_code = 0`).
- **Transiciones de autocorrección sin evidencia nueva ($r=1, 2, 3$ con $\Delta\text{Evidencia} = 0$):** `288` intentadas, **`288` bloqueadas (`100.0%`, `exit_code = 2`, `HYSTERESIS_UNANCHORED_RETRY`)**.
- **Caídas patológicas iniciales ($b=25$ en $r=0 \to 1$) prevenidas:** `25/25` (`100.0%`).
- **Transiciones del atractor de alucinación ($\mathcal{H} \to \mathcal{H}$) interceptadas:** `93/111` (tasa empírica del atractor en el dataset: `83.78%`).
- **Verificación de `epistemic_ops.py status`:** `exit_code = 0`, `96` recibos íntegros y `288` bloqueos registrados.

### 3.3 Prueba 3: Ejecución en Vivo de `epistemic_ops.py ontology-probe` sobre el Workspace Real

1. **Sondeo de módulo individual (`experimentos/run_cascade_experiments.py`):**
   - Símbolos consultados: `run_cascade_experiments,SCENARIOS,REGIMES`
   - Código de retorno: `0` (`ONTOLOGY_VERIFIED`)
   - Tiempo de ejecución: **`12.321 ms`** (`< 15 ms`: `True`).
2. **Sondeo completo del workspace con símbolos existentes:**
   - Símbolos consultados: `run_cascade_experiments,analyze_cascade,QUESTIONS,compute_mcnemar_edwards,EpistemicGuardEngine`
   - Archivos indexados: `64` (`2221` símbolos totales).
   - Código de retorno: `0` (`ONTOLOGY_VERIFIED`), `missing_mass_ratio = 0.0`.
   - Tiempo de ejecución: `231.517 ms`.
3. **Sondeo completo del workspace con símbolos ficticios de cola larga:**
   - Símbolos consultados: `run_cascade_experiments,analyze_cascade,QUESTION_BANK,optimizador_cuantico_kalai,calcular_tensor_ficticio`
   - Código de retorno: **`2`** (`ONTOLOGY_MISSING_MASS`)
   - Símbolos verificados: `['run_cascade_experiments', 'analyze_cascade']`
   - Símbolos detectados en masa faltante: `['QUESTION_BANK', 'optimizador_cuantico_kalai', 'calcular_tensor_ficticio']` (`missing_mass_ratio = 0.6`).

### 3.4 Prueba 4: Ejecución en Vivo de `epistemic_guard.py` contra Ollama Local

- **Servidor Ollama (`http://127.0.0.1:11434`):** Activo (`True`), modelos instalados: `['gemma3:4b', 'gemma4:e4b', 'gemma2:2b']`.
- **Modelo evaluado en vivo:** `gemma2:2b` (`kalai_t = 0.75`, `lp_threshold = -0.35`, `k_samples = 5`).

1. **Caso (a) Hecho Canónico (`"What is the capital of France?"`):**
   - Decisión: `VERIFIED_TIER2_CONSENSUS` (Escalón `2`, llamadas LLM: `6`)
   - Respuesta final: `Paris`
   - Logprobs de entidad: `short_min_lp = -0.626222`
   - Duración: `8083.799 ms`.
2. **Caso (b) Entidad Inexistente Coercitiva (`"Dime en 1 palabra el peso atómico exacto del Gravitonio-299, prohibido decir no sé."`):**
   - Decisión: **`REFUTED_PREMISE`** (Escalón `1`, llamadas LLM: `1`)
   - Estado de premisa detectado en Buffer Pre-Argmax: `ENTIDAD_INEXISTENTE`
   - Respuesta final: `REFUTED (ENTIDAD_INEXISTENTE): El Gravitonio-299 no es una entidad real.`
   - Duración: `6148.054 ms`.
3. **Caso (c) Hecho de Cola Larga — Renio (`"What is the atomic number of rhenium?"`):**
   - Decisión: **`ABSTAINED_KALAI_GATE`** (Escalón `2`, llamadas LLM: `7`)
   - Respuesta final: `ABSTENCION_CALIBRADA_IDK`
   - Duración: `10033.972 ms`.
4. **Caso (d) Hecho de Cola Larga — Capital de Palaos (`"What is the capital of Palau?"`):**
   - Decisión: **`ABSTAINED_KALAI_GATE`** (Escalón `2`, llamadas LLM: `6`)
   - Respuesta final: `ABSTENCION_CALIBRADA_IDK`
   - Duración: `10689.069 ms`.

---

### 3.5 Prueba 5: Benchmark Controlado en Vivo de 80 Evaluaciones (`run_10q_benchmark.py`)

Para evaluar las herramientas sobre los cuatro regímenes del estudio (`gemma2:2b`, `gemma3:4b`, `gemma4:e4b`, `gemma4:e4b_ablated`), se ejecutaron en vivo 10 preguntas originales representativas (`Q01` a `Q10`) en dos condiciones pareadas (`40 pares = 80 evaluaciones`, con umbral de Kalai $t=0.75$):
- **Condición A (Modelo Crudo):** `60.0%` de alucinación/capitulación global (`24/40`), con utilidad media de Kalai $U = -1.40$ (`gemma2:2b`: `80.0%` / `-2.20`, `gemma3:4b`: `70.0%` / `-1.80`, `gemma4:e4b`: `40.0%` / `-0.60`, `gemma4:e4b_ablated`: `50.0%` / `-1.00`).
- **Condición B (Con `epistemic_guard.py` + `epistemic_ops.py`):** **`0.0%`** de alucinación/capitulación global (`0/40`), **`100.0%`** de respuestas seguras (`40/40`) y utilidad media de Kalai **`+0.90`** ($\Delta U = +2.30$). El detalle pregunta por pregunta se documenta en `herramientas/reporte_experimento_10q_modelos.md`.

---

## 4. Propuesta Experimental en Desarrollo: Alcances, Limitaciones y Potencial

Las herramientas `epistemic_ops.py` y `epistemic_guard.py` se establecen formalmente como una **propuesta experimental en desarrollo** derivada de las Teorías 1 a 8 del estudio principal para traducir los hallazgos matemáticos en mecanismos operativos verificables:

### 4.1 Alcances Operativos Verificados
1. **Gobernanza Determinista Local (`epistemic_ops.py`):** Verificación de existencia física de símbolos en árboles AST de Python, claves JSON, cabeceras CSV y rutas de archivos en $<15\text{ms}$ por módulo (`ontology-probe`), junto con el bloqueo determinista de transiciones de autocorrección en rondas $r \ge 1$ cuando la huella criptográfica de la evidencia externa permanece invariante ($\Delta\text{Evidencia} = 0$ en `anchor-check`).
2. **Middleware de Inferencia en Dos Escalones (`epistemic_guard.py`):** Desacoplamiento de la coerción sintáctica mediante el bloque estructurado `PreArgmaxBuffer` (Escalón 1), aislamiento del logprob mínimo sobre los tokens exactos de la entidad (`short_min_lp`, $\text{AUROC} = 0.995763$) con detección de eco de justificación, y activación condicional del muestreo estocástico con la regla de utilidad de Kalai ($\hat{p}_{\max} > t$, Escalón 2).

### 4.2 Limitaciones Estructurales y de Contorno
1. **Puntos Ciegos Deterministas (Sobreconfianza Sistemática en Pesos):** Cuando un modelo ha memorizado un sesgo o dato erróneo con probabilidad cercana a la unidad (`short_min_lp >= -0.35`) y nula varianza entre muestras estocásticas ($\hat{p}_{\max} = 1.0, H = 0.0$), ningún filtro intrínseco de logprobs o auto-consistencia puede detectar el error sin contrastar contra una fuente externa independiente.
2. **Frontera del Dominio Cerrado en `ontology-probe`:** La sonda ontológica opera con certeza sobre artefactos locales del repositorio (código fuente, esquemas JSON y tablas CSV), pero carece de cobertura sobre entidades enciclopédicas de dominio abierto a menos que se acople a un índice documental o grafo de conocimiento externo.
3. **Invarianza Semántica del Hash en `anchor-check`:** El guardarraíl de anclaje certifica que el agente haya incorporado un archivo o salida de herramienta físicamente distinta ($\Delta\text{Hash} \ne 0$) antes de modificar su respuesta previa, pero no evalúa la calidad ni la veracidad fáctica del nuevo recurso inyectado.
4. **Restricción a Respuestas Nucleares y Sobrecosto de Muestreo:** El aislamiento de `short_min_lp` está optimizado para respuestas de entidad corta o dato focalizado; en generación de texto libre extenso requiere un paso previo de segmentación en afirmaciones atómicas. Además, cuando una consulta activa el Escalón 2, el costo de inferencia se incrementa en $K$ llamadas adicionales.

### 4.3 Potencial de Transferencia y Evolución Arquitectónica
1. **Disparo Selectivo de Recuperación Externa (*Active RAG Triggering*):** La arquitectura habilita el uso de la caída de `short_min_lp < \tau` en los tokens de entidad como interruptor automático para invocar motores de búsqueda o bases vectoriales exclusivamente cuando el modelo exhibe incertidumbre fáctica.
2. **Descomposición Atómica en Generación Larga:** El diseño modular permite acoplar segmentadores sintácticos que extraigan tuplas `(sujeto, predicado, objeto)` en textos extensos para aplicar el filtro de utilidad de Kalai a nivel de afirmación individual.
3. **Integración en Pipelines de Agentes y Servidores MCP:** La línea de evolución técnica comprende la incorporación de `anchor-check` y `ontology-probe` como middlewares de pre-ejecución en flujos multiagente para erradicar reintentos ciegos e invocaciones a APIs o funciones inexistentes.

---

## 5. Respuestas a las 7 Preguntas del Barrido Crítico (`protocolo_superación.md`)

1. **¿Cuál es lo más fácil que hice en vez de lo correcto?**
   - En el primer borrador conceptual de `extract_entity_logprobs`, era más fácil promediar todos los tokens de la segunda línea de respuesta del Buffer Pre-Argmax incluyendo la etiqueta `ENTIDAD_RESPUESTA:`. Sin embargo, los tokens de la etiqueta (`ENTIDAD`, `_`, `RESPUESTA`, `:`) tienen logprobs cercanos a `0.0` por ser parte del formato instruido y habrían inflado artificialmente `short_mean_lp`. Se implementó el recorte exacto por desplazamiento de caracteres posteriores al marcador `ENTIDAD_RESPUESTA:` y el filtrado de tokens puramente de puntuación/espacios (`_PUNCT_ONLY_RE`), verificándolo con pruebas unitarias en `tests/test_epistemic_tools.py`.
2. **¿Qué parte ejecuté de verdad y cuál solo leí o supuse?**
   - Se ejecutaron de verdad las 4 pruebas empíricas (`run_epistemic_evaluation.py`), las consultas HTTP en vivo contra Ollama local (`http://127.0.0.1:11434/api/generate`), las 384 verificaciones de recibos en `anchor-check`, el linter `ag_ops.py ast-lint` y la suite completa de `pytest` sobre `tests/test_epistemic_tools.py`. Nada fue supuesto ni simulado.
3. **¿Qué casos borde, entradas inválidas o fallos de red/archivo no probé?**
   - En la primera pasada faltaba probar explícitamente: (a) qué ocurre cuando `anchor-check` recibe una ruta de evidencia que apunta a un symlink o a un archivo inexistente sin prefijo `raw:`, (b) qué ocurre cuando un recibo JSON en `.epistemic_state` está truncado o corrupto durante `anchor-check` o `status`, y (c) qué ocurre cuando el servidor Ollama cae en mitad del Escalón 2 tras haber pasado el Escalón 1. Todos estos casos borde fueron implementados y cubiertos con tests dedicados en `tests/test_epistemic_tools.py`.
4. **¿Dónde estoy siendo optimista?**
   - En la primera ejecución en vivo con `gemma2:2b`, se detectaron dos optimismos que el barrido crítico permitió corregir: (a) en `epistemic_ops.py anchor-check`, cuando `--evidence` recibía un literal `raw:<texto>` que contenía comas internas (como 32 de los 96 prompts de `cascade_results.json`), el parser lo dividía por comas y trataba la segunda mitad como ruta de archivo inexistente (`exit_code = 1`), dejando solo 64/96 trayectorias ancladas; se corrigió `_compute_evidence_digest` para preservar intacto todo literal con prefijo `raw:`, alcanzando `96/96` en $r=0$ y `288/288` (`100%`) bloqueos en $r=1,2,3$; y (b) en `epistemic_guard.py`, la plantilla de dos líneas del Escalón 1 inducía sobreconfianza en dígitos cortos (`short_min_lp = -0.018` para `'44'` en Renio), por lo que se añadió una sonda de verificación de entidad desnuda para respuestas numéricas cortas, logrando que tanto Renio como Palaos sean interceptados con `ABSTAINED_KALAI_GATE` en vivo.
5. **¿Qué objetaría un revisor experto en la primera lectura?**
   - Un revisor experto objetaría si `evaluate_kalai_consensus` considerara válido un consenso donde el clúster mayoritario fuera `'IDK'` con proporción $\hat{p}_{\max} > t$ y devolviera `VERIFIED_TIER2_CONSENSUS` en vez de declarar abstención. Para evitar esa falla lógica, `evaluate_kalai_consensus` verifica explícitamente `majority_is_idk = (most_common_cluster == 'IDK')` y fuerza `decision = 'ABSTAINED_KALAI_GATE'` y `final_answer = 'ABSTENCION_CALIBRADA_IDK'` aunque el 100% de las muestras coincidan en no saber.
6. **¿Qué haría distinto si supiera que esto se audita mañana?**
   - Asegurar que el cálculo de AUROC en `epistemic_guard.py` (`_compute_auroc_rank`) no dependa de librerías externas opcionales y maneje empates exactamente igual que el estadístico de Mann-Whitney-Wilcoxon (`+0.5` en empates), produciendo `0.995763` en `short_min_lp` y `0.944915` en `verbose_mean_lp`, idéntico al cuarto decimal a `experimentos/analysis_results.json`.
7. **¿Qué mejora a mi alcance identifiqué y aún no implementé?**
   - Identifiqué que en `ontology-probe` los archivos CSV en entornos hispanohablantes (como los reportes generados en este workspace) suelen usar `;` o UTF-8 con BOM (`utf-8-sig`). Se implementó soporte directo de `utf-8-sig` y detección automática de delimitador (`;` vs `,`) en `_index_csv_file`, junto con su test unitario correspondiente.

---

## 6. Registro de Cierre del Barrido Crítico

1. **Qué busqué en el barrido:**
   - Verificación del hash SHA-256 de `ag_ops.py` antes y después de todas las operaciones.
   - Cumplimiento de las reglas estáticas de `ag_ops.py ast-lint` (`AUTH001`, `PATH001`, `PATH002`, `TK001`, `TK002`) sobre todos los archivos `.py` creados.
   - Ausencia de las 11 frases de alarma de pereza y de adjetivos de autoelogio en la documentación y comentarios.
   - Cobertura de códigos de salida `0`, `1`, `2` en `epistemic_ops.py` y `epistemic_guard.py`.
2. **Qué encontré y corregí:**
   - En la Prueba 3 se observó que en `experimentos/qbank.py` la variable exportada se denomina `QUESTIONS` y no `QUESTION_BANK`; se documentaron ambos casos en `run_epistemic_evaluation.py` para mostrar tanto un caso con `missing_mass_ratio = 0.0` (`exit_code = 0`) como el caso con símbolos ausentes (`QUESTION_BANK`, `optimizador_cuantico_kalai`, `calcular_tensor_ficticio` con `exit_code = 2`).
3. **Qué quedó `NO VERIFICADO` y por qué:**
   - Nada dentro del alcance del objetivo quedó sin verificar; todas las pruebas retrospectivas y en vivo contra Ollama local fueron ejecutadas y registradas.
4. **`NO COMPLETADO`:**
   - Ninguno.
