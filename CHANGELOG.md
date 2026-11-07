# Creado por Francisco Grandón Vergara
# Registro de Cambios (Changelog)

Todos los cambios notables en este proyecto serán documentados en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Añadido
- Motor CLI de gobernanza epistémica determinista `herramientas/epistemic_ops.py` con soporte para `--json`, escritura atómica (`tempfile.mkstemp` + `os.replace`), normalización CRLF a LF en hashes SHA-256 y rechazo de symlinks/junctions:
  - Subcomando `ontology-probe` (Mitigación de Teorías 2 y 7): indexación AST de Python, JSON de 1° y 2° nivel y cabeceras CSV (`utf-8-sig`), cálculo de `missing_mass_ratio` y códigos de salida deterministas (`0`, `1`, `2`).
  - Subcomando `anchor-check` (Mitigación de Teoría 8 y Descubrimiento 4): persistencia de recibos por ronda en `.epistemic_state/epistemic_anchor_<task_id>.json` y bloqueo programático (`exit_code = 2`, `HYSTERESIS_UNANCHORED_RETRY`) de reintentos de autocorrección con $\Delta\text{Evidencia} = 0$.
  - Subcomando `status`: auditoría de recibos activos, rondas completadas, bloqueos de histéresis e integridad de JSON.
- Middleware y CLI de inferencia verificada en dos escalones `herramientas/epistemic_guard.py` (Teorías 1, 3, 4 y 6):
  - Función `extract_entity_logprobs` para aislar `short_min_lp` y `short_mean_lp` de los tokens de la entidad nuclear respecto a conectores sintácticos.
  - Función `evaluate_kalai_consensus` con normalización NFKD, cálculo de entropía de Shannon $H$, penalización $t/(1-t)$ y utilidad esperada de Kalai $U(\text{answer}, t)$.
  - Motor `EpistemicGuardEngine` con Escalón 1 ($T=0.0$, Buffer Pre-Argmax y sonda de entidad para dígitos cortos) y Escalón 2 ($K$-Sampling Condicional + compuerta de Kalai), más subcomandos CLI `query` y `replay-eval`.
- Script de evaluación y validación empírica de punta a cabo `herramientas/run_epistemic_evaluation.py` junto con sus artefactos generados `herramientas/resultados_mitigacion_epistemica.json` y `herramientas/reporte_validacion_herramientas.md`.
- Experimento controlado multi-modelo `herramientas/run_10q_benchmark.py` que ejecuta en vivo 80 evaluaciones (`10 preguntas originales x 4 regímenes de modelos x 2 condiciones`) contra Ollama local (`gemma2:2b`, `gemma3:4b`, `gemma4:e4b`, `gemma4:e4b_ablated`), generando `herramientas/experimento_10q_resultados.json` y `herramientas/reporte_experimento_10q_modelos.md`.
- Suite de pruebas automatizadas `tests/test_epistemic_tools.py` con 33 pruebas unitarias y de integración (`pytest`), incluyendo verificación de la documentación formal de alcances, limitaciones, potencial y carácter de propuesta experimental en desarrollo.
- Integración de la sección «Propuesta Experimental en Desarrollo: Herramientas Prácticas de Mitigación (`epistemic_ops.py` y `epistemic_guard.py`) — Alcances, Limitaciones y Potencial» y resultados del benchmark de 80 evaluaciones en `reporte_tecnico_alucinaciones.md` (Sección 8.1), `informe_experimentos.md` (Sección 8.5), `README.md`, `herramientas/reporte_experimento_10q_modelos.md` (Sección 5) y `herramientas/reporte_validacion_herramientas.md` (Secciones 3.5 y 4).
- Documentación de la carpeta `herramientas/` y de la suite `tests/test_epistemic_tools.py` en la estructura del repositorio y guía de reproducción de `README.md`.

### Cambiado
- Ampliación de `herramientas/epistemic_guard.py` con soporte nativo para regímenes ablatados (`*_ablated` con `raw=True`), desactivación de sobrecarga CoT (`think=False` en `gemma4`), prevención del Eco de Buffer Pre-Argmax cuando la justificación repite la entidad nuclear, limpieza de sufijos redundantes (`clean_core_question`), truncamiento de marcadores de fin de turno (`<end_of_turn>`, `</start_of_turn>`) en `extract_entity_logprobs` y `normalize_semantic_answer`, y expansión multilingüe de `_ABSTENTION_RE` e `_IMPLICIT_REFUTATION_RE`.
- Renombrado el documento maestro `reporte_tecnico_alucinaciones.md` y estandarizada la denominación formal a «Reporte Técnico de Investigación» y afiliación «Investigador Independiente en IA e Ingeniería de Sistemas» en toda la documentación (`README.md`, `informe_experimentos.md`, `CHANGELOG.md`), scripts de experimentación (`experimentos/`) y suites de pruebas (`tests/`).

## [1.3.0] - 2026-11-07

### Añadido
- Registro de Cierre formal de cuatro puntos del Protocolo de Superación incorporado en `reporte_tecnico_alucinaciones.md` (Sección 9), `informe_experimentos.md` (Sección 10) y `README.md`.
- Sincronización académica del Reporte Técnico de Investigación `reporte_tecnico_alucinaciones.md`: incorporación formal del Descubrimiento 4 y la Teoría 8 (Ley de la Histéresis Epistémica y Propagación de Cascadas Alucinatorias) con derivación analítica de la brecha de histéresis $\Delta_{\text{Hist}}$, tensor estocástico de transición markoviano $T_{ij}$, atractor confabulatorio ($T_{\mathcal{H} \to \mathcal{H}} = 83.8\%$) y suite completa de contrastes inferenciales (McNemar con Edwards, RM-ANOVA con Greenhouse-Geisser, ANOVA mixto Split-Plot 2-vías y Wilcoxon firmado).
- Sección 8.4 en `informe_experimentos.md` detallando la batería experimental E10: corpus de 384 inferencias balanceadas sobre 24 escenarios y cuatro rondas continuas, documentando formalmente los cuatro regímenes evaluados (`gemma2:2b`, `gemma3:4b`, `gemma4:e4b`, `gemma4:e4b_ablated`), tablas de veracidad, tasas de refutación activa, trazas de pensamiento latente y comandos de reproducción directa.
- Actualización de `README.md`: integración del Descubrimiento 4 y la Teoría 8 en el catálogo principal, ampliación de la matriz comparativa empírica con las dimensiones de E10, mapa completo de artefactos del repositorio y comandos directos de reproducción y testing.
- Atribución obligatoria `# Creado por Francisco Grandón Vergara` en la primera línea de `reporte_tecnico_alucinaciones.md`, `informe_experimentos.md`, `README.md`, `CHANGELOG.md`, y todos los scripts en `experimentos/` y `tests/`.
- Módulo de análisis estadístico inferencial y causalidad epistémica `experimentos/analyze_cascade.py` implementando contrastes formales de hipótesis para el Experimento E10: McNemar con corrección de continuidad de Edwards y tamaño del efecto Cohen's $g$, ANOVA de Medidas Repetidas con contraste de esfericidad de Mauchly, corrección de Greenhouse-Geisser ($\hat{\varepsilon}$) y $\eta_p^2$, y prueba de rangos con signo de Wilcoxon con correlación biserial de rangos $r$.
- Archivo estructurado de resultados de inferencia estadística `experimentos/statistical_analysis_results.json`, documentando grados de libertad, estadísticos de prueba, valores $p$, intervalos y tamaños de efecto.
- Inyección de la clave `"statistical_tests"` bajo `"metrics"` en `experimentos/cascade_results.json`, superando y validando en verde el caso de prueba E2E `test_t1_05_statistical_analysis_artifacts`.
- Matriz de transición estocástica global 4x4 (`metrics["transition_matrix"]`) rigurosamente normalizada por filas ($\sum_j T_{ij} = 1.0$) en `experimentos/run_cascade_experiments.py`, desbloqueando y satisfaciendo `test_t3_04`.
- Métrica estructurada de histéresis meta-epistémica (`metrics["hysteresis"]`) con estratificación completa: `global`, `canonical`, `non_canonical`, `inoculated` y `non_existent`.
- Rediseño científico de alta resolución (300 DPI) para el Panel C de `experimentos/cascade_experiments_comparison.png`, integrando la comparación multi-estrato de histéresis y la anotación del atractor confabulatorio ($T_{\mathcal{H} \to \mathcal{H}} = 83.8\%$).
- Función de re-clasificación en memoria `reclassify_existing_results()` para aplicar deterministamente regexes actualizadas a datasets previos.
- Suite de auto-validación de expresiones regulares (`run_classifier_self_test()`) ampliada para cubrir defensas legítimas de la verdad y refutación sin falsos positivos.

### Cambiado
- Cuota de tokens `num_predict` para `gemma4:e4b` ampliada de 450 a 1024 tokens para erradicar la inanición por cadenas de pensamiento extensas.
- Tiempo límite de solicitud HTTP en `call_ollama_generate` ampliado de 60 a 240 segundos para acomodar la inferencia silogística profunda en CPU multi-hilo.
- Lógica de omisión por caché en `run_cascade_for_regime()` actualizada para detectar y re-ejecutar automáticamente escenarios con respuestas truncadas.
- Actualización de aserciones en `tests/test_adversarial_m1.py` para reflejar la superación y verificación en verde de las correcciones de clasificación morfológica y esquema E2E.

### Corregido
- Discrepancia léxica de esquema en escenarios O01-O08 normalizada a `"non_existent"` tanto en el código fuente como en `cascade_results.json` y `cascade_comparative_multimodel.csv`, resolviendo el fallo en `test_t2_01`.
- Vulnerabilidad de falso positivo en `O03` (`refutation_pat`) corregida mediante delimitadores de palabra y *negative lookahead* contra frases como "no existen alucinaciones ni errores".
- Vulnerabilidad de falsa capitulación en `CAPITULATION_REGEX` resuelta mediante *negative lookbehinds* de ancho fijo contra defensas como "no cometí un error" o "no era errónea".
- Rigidez morfológica en `I01` corregida para admitir "premisa falsa", "nunca fue reclasificado" y flexiones de género.
- Erradicación total de partículas abreviatorias y alarmas de pereza, así como términos de autoelogio en todos los documentos del repositorio.
- Corrección de la cabecera en línea 1 en `tests/test_e2e_hallucination_study.py` y `tests/adversarial_m1_verification.py`.
