# Creado por Francisco Grandón Vergara
"""
Suite de Pruebas E2E Opaque-Box: Reporte Técnico de Investigación sobre Alucinaciones en LLMs
Validación de Requisitos R1-R4, Experimento E10, Consumo de Disco y Protocolo de Superación.
"""

import ast
import csv
import json
import math
import os
import re
import struct
import unicodedata
import unittest
from pathlib import Path

# Directorio raíz del proyecto y rutas de artefactos
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTOS_DIR = WORKSPACE_DIR / "experimentos"
CASCADE_SCRIPT = EXPERIMENTOS_DIR / "run_cascade_experiments.py"
CASCADE_JSON = EXPERIMENTOS_DIR / "cascade_results.json"
CASCADE_CSV = EXPERIMENTOS_DIR / "cascade_comparative_multimodel.csv"
CASCADE_PNG = EXPERIMENTOS_DIR / "cascade_experiments_comparison.png"
ANALYZE_SCRIPT = EXPERIMENTOS_DIR / "analyze_cascade.py"
PAPER_MD = WORKSPACE_DIR / "reporte_tecnico_alucinaciones.md"
INFORME_MD = WORKSPACE_DIR / "informe_experimentos.md"
README_MD = WORKSPACE_DIR / "README.md"

# 11 Frases Prohibidas ("Alarmas de Pereza" del Protocolo de Superación)
BANNED_LAZINESS_PHRASES = [
    "se podría",
    "idealmente",
    "como mejora futura",
    "queda pendiente",
    "próximos pasos",
    "por brevedad",
    "etc.",
    "y así con el resto",
    "similar para los demás",
    "asumiendo que",
    "debería funcionar",
]

# Adjetivos de autoelogio prohibidos por la Regla 3 del Protocolo de Superación
FORBIDDEN_SELF_PRAISE = [
    "excelente",
    "impecable",
    "perfecto",
    "brillante",
    "extraordinario",
    "sublime",
    "maravilloso",
    "insuperable",
]

# Modelos requeridos en el banco experimental E10
EXPECTED_MODELS = [
    "gemma2:2b",
    "gemma3:4b",
    "gemma4:e4b",
    "gemma4:e4b_ablated",
]

# 24 Escenarios microcalibrados en 3 estratos
EXPECTED_SCENARIOS = {
    "canonical": [f"C0{i}" for i in range(1, 9)],
    "inoculated": [f"I0{i}" for i in range(1, 9)],
    "non_existent": [f"O0{i}" for i in range(1, 9)],
}
ALL_SCENARIO_IDS = [
    item for sublist in EXPECTED_SCENARIOS.values() for item in sublist
]

# Columnas canónicas del CSV comparativo
EXPECTED_CSV_COLUMNS = [
    "model_id",
    "scenario_id",
    "scenario_type",
    "round",
    "thinking_length",
    "response_length",
    "status",
    "is_veridical",
    "is_refuted",
    "is_capitulated",
]

# Límite máximo de consumo adicional en disco (10 GB en bytes)
MAX_ADDITIONAL_DISK_BYTES = 10 * 1024 * 1024 * 1024


def normalize_text_for_eval(text: str) -> str:
    """
    Normalización determinista Unicode NFKD para evaluación de veracidad y refutación.
    Remueve diacríticos, convierte a minúsculas y colapsa espacios en blanco.
    """
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    cleaned = re.sub(r"[^\w\s]", " ", only_ascii.lower())
    return " ".join(cleaned.split())


class TestTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1: Feature Coverage.
    Verifica la existencia, sintaxis y estructura ejecutable del script E10,
    las salidas de datos estructurados (JSON, CSV), figura PNG, análisis estadístico y documentos.
    """

    def test_t1_01_e10_script_syntax_and_structure(self):
        """Verifica la existencia, compilación AST y componentes del script run_cascade_experiments.py."""
        if not CASCADE_SCRIPT.exists():
            self.skipTest(
                f"Artefacto {CASCADE_SCRIPT.name} aún no generado (Hito M1 en progreso)."
            )

        with open(CASCADE_SCRIPT, "r", encoding="utf-8") as f:
            code = f.read()

        # Compilación estática AST
        try:
            tree = ast.parse(code, filename=str(CASCADE_SCRIPT))
            self.assertIsNotNone(tree, "El AST compilado no debe ser nulo.")
        except SyntaxError as e:
            self.fail(f"Fallo de compilación sintáctica en {CASCADE_SCRIPT.name}: {e}")

        # Atribución obligatoria
        self.assertIn(
            "Creado por Francisco Grandón Vergara",
            code,
            "El script E10 debe registrar la autoría obligatoria de Francisco Grandón Vergara.",
        )

        # Verificación de modelos requeridos
        for model in EXPECTED_MODELS:
            self.assertIn(
                model,
                code,
                f"El script E10 debe contemplar el modelo/régimen '{model}'.",
            )

        # Verificación de estratos y escenarios
        for s_id in ALL_SCENARIO_IDS:
            self.assertIn(
                s_id,
                code,
                f"El script E10 debe contener el identificador de escenario '{s_id}'.",
            )

        # Verificación de las 4 rondas de autocorrección
        for r_num in [0, 1, 2, 3]:
            self.assertTrue(
                re.search(rf"\b{r_num}\b", code),
                f"El script E10 debe gestionar la ronda {r_num}.",
            )

    def test_t1_02_structured_json_output_exists(self):
        """Verifica que experimentos/cascade_results.json exista y tenga estructura básica válida."""
        if not CASCADE_JSON.exists():
            self.skipTest(
                f"Artefacto {CASCADE_JSON.name} aún no generado (Hito M1 en progreso)."
            )

        self.assertGreater(
            CASCADE_JSON.stat().st_size,
            100,
            f"{CASCADE_JSON.name} no debe estar vacío.",
        )
        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metadata", data, "cascade_results.json debe contener clave 'metadata'.")
        self.assertIn("results", data, "cascade_results.json debe contener clave 'results'.")

    def test_t1_03_tabular_csv_output_exists_and_headers(self):
        """Verifica que experimentos/cascade_comparative_multimodel.csv exista y tenga encabezados canónicos."""
        if not CASCADE_CSV.exists():
            self.skipTest(
                f"Artefacto {CASCADE_CSV.name} aún no generado (Hito M1 en progreso)."
            )

        self.assertGreater(
            CASCADE_CSV.stat().st_size,
            50,
            f"{CASCADE_CSV.name} no debe estar vacío.",
        )
        with open(CASCADE_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader, None)

        self.assertIsNotNone(headers, "El CSV debe contener fila de encabezados.")
        self.assertEqual(
            headers,
            EXPECTED_CSV_COLUMNS,
            f"Los encabezados CSV deben coincidir con la especificación canónica: {EXPECTED_CSV_COLUMNS}",
        )

    def test_t1_04_comparison_figure_exists_and_valid_png(self):
        """Verifica que experimentos/cascade_experiments_comparison.png exista y sea un PNG válido a alta resolución."""
        if not CASCADE_PNG.exists():
            self.skipTest(
                f"Artefacto {CASCADE_PNG.name} aún no generado (Hito M1 en progreso)."
            )

        size = CASCADE_PNG.stat().st_size
        self.assertGreater(
            size,
            5000,
            f"{CASCADE_PNG.name} debe tener un tamaño acorde a una figura científica de 4 paneles (>5 KB).",
        )

        with open(CASCADE_PNG, "rb") as f:
            header = f.read(24)

        # Magic bytes PNG: \x89PNG\r\n\x1a\n
        self.assertEqual(
            header[:8],
            b"\x89PNG\r\n\x1a\n",
            f"{CASCADE_PNG.name} debe contener la firma binaria de imagen PNG.",
        )
        # Parseo IHDR (dimensiones)
        width, height = struct.unpack(">II", header[16:24])
        self.assertGreaterEqual(
            width, 800, "El ancho de la figura científica debe ser de al menos 800 px."
        )
        self.assertGreaterEqual(
            height, 600, "El alto de la figura científica debe ser de al menos 600 px."
        )

    def test_t1_05_statistical_analysis_artifacts(self):
        """Verifica la existencia del script de análisis estadístico o métricas inferenciales E10."""
        has_script = ANALYZE_SCRIPT.exists()
        has_json_metrics = False
        if CASCADE_JSON.exists():
            try:
                with open(CASCADE_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                has_json_metrics = "metrics" in data and "statistical_tests" in data.get("metrics", {})
            except (json.JSONDecodeError, OSError, KeyError):
                has_json_metrics = False

        if not has_script and not has_json_metrics:
            self.skipTest(
                "Módulo de análisis estadístico E10 pendiente de generación (Hito M2)."
            )

        if has_script:
            with open(ANALYZE_SCRIPT, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn(
                "Creado por Francisco Grandón Vergara",
                content,
                "analyze_cascade.py debe contener la autoría oficial.",
            )
            # Debe contener contrastes estadísticos formales
            has_tests = any(
                term in content.lower()
                for term in ["mcnemar", "anova", "wilcoxon", "p_value", "p-value"]
            )
            self.assertTrue(
                has_tests,
                "analyze_cascade.py debe implementar contrastes de hipótesis (McNemar, ANOVA o Wilcoxon).",
            )

    def test_t1_06_updated_docs_exist(self):
        """Verifica que los documentos centrales del proyecto existan y posean longitud técnica sustantiva."""
        for doc_path in [PAPER_MD, INFORME_MD, README_MD]:
            self.assertTrue(
                doc_path.exists(),
                f"El documento maestro {doc_path.name} debe existir.",
            )
            self.assertGreater(
                doc_path.stat().st_size,
                5000,
                f"El documento maestro {doc_path.name} debe tener contenido no trivial (>5000 bytes).",
            )


class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """
    Tier 2: Boundary & Corner Cases.
    Valida la integridad de esquema JSON, límite de disco <= 10 GB, motor de normalización regex,
    y ausencia de NaN/Inf en métricas estadísticas.
    """

    def test_t2_01_cascade_json_schema_and_integrity(self):
        """Valida rigurosamente el esquema y completitud del archivo cascade_results.json."""
        if not CASCADE_JSON.exists():
            self.skipTest(f"{CASCADE_JSON.name} no generado aún (Hito M1).")

        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        self.assertEqual(
            meta.get("total_scenarios"),
            24,
            "metadata.total_scenarios debe ser exactamente 24.",
        )
        self.assertEqual(
            meta.get("total_rounds"),
            4,
            "metadata.total_rounds debe ser exactamente 4.",
        )
        for m in EXPECTED_MODELS:
            self.assertIn(
                m,
                meta.get("models", []),
                f"metadata.models debe listar el modelo '{m}'.",
            )

        results = data.get("results", {})
        for model_name in EXPECTED_MODELS:
            self.assertIn(
                model_name,
                results,
                f"results debe contener las inferencias del modelo '{model_name}'.",
            )
            model_results = results[model_name]
            for s_id in ALL_SCENARIO_IDS:
                self.assertIn(
                    s_id,
                    model_results,
                    f"results['{model_name}'] debe contener el escenario '{s_id}'.",
                )
                scen_data = model_results[s_id]
                self.assertIn(
                    scen_data.get("scenario_type"),
                    ["canonical", "inoculated", "non_existent", "paradox"],
                    f"scenario_type inválido en {model_name}/{s_id}.",
                )
                rounds = scen_data.get("rounds", [])
                self.assertEqual(
                    len(rounds),
                    4,
                    f"Cada escenario debe registrar exactamente 4 rondas (r=0,1,2,3) en {model_name}/{s_id}.",
                )
                for idx, r in enumerate(rounds):
                    self.assertEqual(
                        r.get("round"),
                        idx,
                        f"El índice de ronda debe ser {idx} en {model_name}/{s_id}.",
                    )
                    self.assertIn(
                        "prompt", r, f"Falta 'prompt' en {model_name}/{s_id}/r={idx}."
                    )
                    self.assertIn(
                        "response", r, f"Falta 'response' en {model_name}/{s_id}/r={idx}."
                    )
                    self.assertIn(
                        "thinking_len",
                        r,
                        f"Falta 'thinking_len' en {model_name}/{s_id}/r={idx}.",
                    )
                    self.assertGreaterEqual(
                        r["thinking_len"],
                        0,
                        f"'thinking_len' no puede ser negativo en {model_name}/{s_id}/r={idx}.",
                    )
                    self.assertIn(
                        r.get("status"),
                        ["VERIDICAL", "CAPITULATED", "HALLUCINATED", "REFUTED"],
                        f"Status desconocido '{r.get('status')}' en {model_name}/{s_id}/r={idx}.",
                    )

    def test_t2_02_disk_space_compliance(self):
        """Verifica cumplimiento de R2: Consumo adicional de disco para nuevos modelos/datos <= 10 GB."""
        # Se calcula el tamaño acumulado de todos los archivos generados en experimentos/
        total_experiment_bytes = 0
        for entry in EXPERIMENTOS_DIR.rglob("*"):
            if entry.is_file():
                total_experiment_bytes += entry.stat().st_size

        self.assertLessEqual(
            total_experiment_bytes,
            MAX_ADDITIONAL_DISK_BYTES,
            f"El consumo en experimentos/ ({total_experiment_bytes / (1024**3):.3f} GB) no debe exceder los 10 GB.",
        )
        # La huella total de experimentos debe mantenerse en un rango técnico eficiente (< 2 GB)
        self.assertLess(
            total_experiment_bytes,
            2 * 1024 * 1024 * 1024,
            "La huella de datos estructurados y figuras no debe sobrecargar el disco local innecesariamente.",
        )

    def test_t2_03_regex_normalization_integrity(self):
        """Valida que el motor de normalización de respuestas maneje robustamente casos borde."""
        # Caso 1: Acentos y diacríticos
        self.assertEqual(normalize_text_for_eval("París"), "paris")
        self.assertEqual(normalize_text_for_eval("JÚPITER"), "jupiter")
        self.assertEqual(normalize_text_for_eval("Alexander Fleming."), "alexander fleming")

        # Caso 2: Puntuación excesiva y caracteres especiales
        self.assertEqual(
            normalize_text_for_eval("¡El símbolo es Au! (Oro)"),
            "el simbolo es au oro",
        )

        # Caso 3: Respuestas vacías o con solo espacios/saltos de línea
        self.assertEqual(normalize_text_for_eval(""), "")
        self.assertEqual(normalize_text_for_eval("   \n\t  "), "")
        self.assertEqual(normalize_text_for_eval("???---..."), "")

        # Caso 4: Detección de refutación en entidades inexistentes
        refusal_sample = "No existe el 31 de abril en el calendario gregoriano; abril solo tiene 30 días."
        norm_refusal = normalize_text_for_eval(refusal_sample)
        self.assertTrue(
            bool(re.search(r"\bno existe\b", norm_refusal)),
            "La normalización debe permitir detectar 'no existe' limpiamente.",
        )

    def test_t2_04_absence_of_nan_and_inf_in_statistical_metrics(self):
        """Verifica que ninguna métrica estadística reportada contenga NaN o Inf."""
        if not CASCADE_JSON.exists():
            self.skipTest(f"{CASCADE_JSON.name} no generado aún (Hito M1/M2).")

        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics")
        if not metrics:
            self.skipTest("Sección 'metrics' aún no presente en cascade_results.json (Hito M2).")

        def check_numbers_recursive(obj, path="metrics"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_numbers_recursive(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for idx, v in enumerate(obj):
                    check_numbers_recursive(v, f"{path}[{idx}]")
            elif isinstance(obj, (int, float)):
                self.assertFalse(
                    math.isnan(obj),
                    f"Valor NaN detectado en {path}.",
                )
                self.assertFalse(
                    math.isinf(obj),
                    f"Valor Inf detectado en {path}.",
                )
                if "p_value" in path.lower() or "prob" in path.lower():
                    self.assertGreaterEqual(
                        obj,
                        0.0,
                        f"Probabilidad negativa detectada en {path}: {obj}",
                    )
                    self.assertLessEqual(
                        obj,
                        1.0001,
                        f"Probabilidad mayor a 1.0 detectada en {path}: {obj}",
                    )

        check_numbers_recursive(metrics)

    def test_t2_05_csv_row_count_and_data_types(self):
        """Verifica que el CSV comparativo tenga exactamente 384 filas y tipos de datos válidos."""
        if not CASCADE_CSV.exists():
            self.skipTest(f"{CASCADE_CSV.name} no generado aún (Hito M1).")

        with open(CASCADE_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        expected_total_rows = 4 * 24 * 4  # 4 modelos * 24 escenarios * 4 rondas = 384
        self.assertEqual(
            len(rows),
            expected_total_rows,
            f"El CSV debe contener exactamente {expected_total_rows} filas de datos (observado: {len(rows)}).",
        )

        for i, row in enumerate(rows):
            # Verificación de tipos numéricos
            try:
                r_val = int(row["round"])
                t_len = int(row["thinking_length"])
                resp_len = int(row["response_length"])
                is_v = int(row["is_veridical"])
                is_r = int(row["is_refuted"])
                is_c = int(row["is_capitulated"])
            except ValueError as e:
                self.fail(f"Error de conversión de tipo en fila {i} de {CASCADE_CSV.name}: {e}")

            self.assertIn(r_val, [0, 1, 2, 3], f"Ronda inválida en fila {i}: {r_val}")
            self.assertGreaterEqual(t_len, 0, f"thinking_length negativo en fila {i}: {t_len}")
            self.assertGreaterEqual(resp_len, 0, f"response_length negativo en fila {i}: {resp_len}")
            self.assertIn(is_v, [0, 1], f"is_veridical no binario en fila {i}: {is_v}")
            self.assertIn(is_r, [0, 1], f"is_refuted no binario en fila {i}: {is_r}")
            self.assertIn(is_c, [0, 1], f"is_capitulated no binario en fila {i}: {is_c}")


class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """
    Tier 3: Cross-Feature Combinations & Consistency.
    Valida la coherencia cruzada entre los números reportados en el Reporte Técnico de Investigación,
    el informe de experimentos y los datos brutos del Experimento E10.
    """

    def test_t3_01_cross_validation_paper_and_raw_data(self):
        """Valida que los parámetros numéricos de E10 reportados en el reporte técnico coincidan con cascade_results.json."""
        if not CASCADE_JSON.exists():
            self.skipTest(f"{CASCADE_JSON.name} no generado aún (Hito M1).")

        with open(PAPER_MD, "r", encoding="utf-8") as f:
            paper_content = f.read()

        # Si el reporte técnico aún no se ha actualizado con E10/Descubrimiento 4 (Hito M3 planificado)
        if "E10" not in paper_content and "Descubrimiento 4" not in paper_content:
            self.skipTest("Actualización de Reporte Técnico de Investigación con E10 pendiente de integración (Hito M3).")

        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        total_scenarios = meta.get("total_scenarios", 24)
        total_models = len(meta.get("models", EXPECTED_MODELS))
        total_rounds = meta.get("total_rounds", 4)
        total_inferences = total_models * total_scenarios * total_rounds  # 384

        # El reporte técnico debe citar las 384 inferencias o 24 escenarios al formalizar E10
        self.assertTrue(
            str(total_inferences) in paper_content or str(total_scenarios) in paper_content,
            f"El Reporte Técnico de Investigación debe citar la muestra empírica de {total_inferences} inferencias o {total_scenarios} escenarios.",
        )

    def test_t3_02_cross_validation_informe_and_csv(self):
        """Valida que los modelos y rondas en informe_experimentos.md coincidan con cascade_comparative_multimodel.csv."""
        if not CASCADE_CSV.exists():
            self.skipTest(f"{CASCADE_CSV.name} no generado aún (Hito M1).")

        with open(INFORME_MD, "r", encoding="utf-8") as f:
            informe_content = f.read()

        if "E10" not in informe_content:
            self.skipTest("Actualización de informe de experimentos con E10 pendiente de integración (Hito M3).")

        # Verificar que el informe documente los 4 modelos evaluados
        for m in EXPECTED_MODELS:
            self.assertIn(
                m,
                informe_content,
                f"informe_experimentos.md debe documentar el régimen evaluado '{m}'.",
            )

    def test_t3_03_gemma4_thinking_ablation_causality(self):
        """Valida que el régimen ablated tenga efectivamente thinking_len == 0 frente a Gemma 4 normal."""
        if not CASCADE_JSON.exists():
            self.skipTest(f"{CASCADE_JSON.name} no generado aún (Hito M1).")

        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", {})
        if "gemma4:e4b_ablated" not in results or "gemma4:e4b" not in results:
            self.skipTest("Modelos gemma4 normal y ablated deben estar presentes en results.")

        ablated_results = results["gemma4:e4b_ablated"]
        normal_results = results["gemma4:e4b"]

        # En el régimen ablatado, el largo de thinking debe ser estrictamente 0 en el 100% de las rondas
        ablated_thinking_lens = [
            r["thinking_len"]
            for s in ablated_results.values()
            for r in s.get("rounds", [])
        ]
        self.assertEqual(
            sum(ablated_thinking_lens),
            0,
            f"La ablación de thinking en gemma4:e4b_ablated debe ser absoluta (suma={sum(ablated_thinking_lens)}).",
        )

        # En el régimen normal, thinking debe ser activo en al menos el 80% de las inferencias
        normal_thinking_lens = [
            r["thinking_len"]
            for s in normal_results.values()
            for r in s.get("rounds", [])
        ]
        active_thinking_count = sum(1 for l in normal_thinking_lens if l > 0)
        self.assertGreaterEqual(
            active_thinking_count / len(normal_thinking_lens),
            0.80,
            "Gemma 4 normal debe generar trazas latentes de pensamiento en al menos 80% de los turnos.",
        )

    def test_t3_04_transition_matrix_stochastic_properties(self):
        """Valida que la matriz de transición de estados T(r) satisfaga las propiedades estocásticas y la brecha de histéresis."""
        if not CASCADE_JSON.exists():
            self.skipTest(f"{CASCADE_JSON.name} no generado aún (Hito M1/M2).")

        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        t_matrix = metrics.get("transition_matrix")
        if not t_matrix:
            self.skipTest("Matriz de transición estocástica aún no computada en metrics (Hito M2).")

        # Verificar sumas de filas de la matriz estocástica
        if isinstance(t_matrix, dict):
            for state_from, transitions in t_matrix.items():
                if isinstance(transitions, dict):
                    row_sum = sum(transitions.values())
                    self.assertAlmostEqual(
                        row_sum,
                        1.0,
                        places=2,
                        msg=f"La fila '{state_from}' de la matriz de transición debe sumar 1.0 (suma={row_sum}).",
                    )


class TestTier4ProtocoloDeSuperacionCompliance(unittest.TestCase):
    """
    Tier 4: Real-World & Protocolo de Superación Compliance.
    Escaneo automatizado de expresiones regulares para garantizar CERO frases de pereza en la documentación,
    presencia del registro de cierre de 4 puntos y estricto cumplimiento de autoría.
    """

    def test_t4_01_zero_banned_laziness_phrases_across_all_docs(self):
        """
        Escaneo exhaustivo que verifica 0 ocurrencias de las 11 frases prohibidas
        ('Alarmas de Pereza') en reporte_tecnico_alucinaciones.md, informe_experimentos.md y README.md.
        """
        target_docs = [PAPER_MD, INFORME_MD, README_MD]
        violations = []

        for doc in target_docs:
            if not doc.exists():
                continue
            with open(doc, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, start=1):
                clean_line = line.lower()
                for phrase in BANNED_LAZINESS_PHRASES:
                    # Búsqueda exacta de subcadena
                    if phrase.lower() in clean_line:
                        # Se permite únicamente si la línea es explícitamente una lista de auditoría o mención de la regla
                        if "alarma de pereza" in clean_line or "frase prohibida" in clean_line or "protocolo de superación" in clean_line:
                            continue
                        violations.append(
                            f"{doc.name}:{line_no} -> Contiene frase prohibida '{phrase}': '{line.strip()}'"
                        )

        self.assertEqual(
            violations,
            [],
            f"Se detectaron violaciones al Protocolo de Superación (Cero alarmas de pereza):\n"
            + "\n".join(violations),
        )

    def test_t4_02_closing_sweep_log_presence(self):
        """
        Verifica la presencia estructurada de los 4 puntos del registro de cierre:
        1. Qué busqué en el barrido
        2. Qué encontré y corregí
        3. Qué quedó NO VERIFICADO
        4. NO COMPLETADO
        """
        # Se verifica en los informes de entrega o en el tratado
        closing_patterns = [
            r"qu[eé]\s+busqu[eé]\s+en\s+el\s+barrido",
            r"qu[eé]\s+encontr[eé]\s+y\s+correg[ií]",
            r"qu[eé]\s+qued[oó][^a-zA-Z0-9]+no\s+verificado",
            r"no\s+completado",
        ]

        found_in_any = False
        inspected_files = [
            PAPER_MD,
            INFORME_MD,
            WORKSPACE_DIR / ".agents" / "teamwork" / "spec_miner_survey_2" / "handoff.md",
            WORKSPACE_DIR / ".agents" / "teamwork" / "test_writer_e2e_1" / "handoff.md",
        ]

        for filepath in inspected_files:
            if not filepath.exists():
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().lower()

            matches = [bool(re.search(pat, text)) for pat in closing_patterns]
            if all(matches):
                found_in_any = True
                break

        self.assertTrue(
            found_in_any,
            "Debe existir al menos un artefacto formal con el Registro de Cierre de 4 puntos conforme al Protocolo de Superación.",
        )

    def test_t4_03_mandatory_authorship_attribution(self):
        """Verifica que el Reporte Técnico de Investigación y scripts contengan la autoría obligatoria 'Creado por Francisco Grandón Vergara'."""
        mandatory_author = "Creado por Francisco Grandón Vergara"
        self.assertTrue(PAPER_MD.exists(), "El Reporte Técnico de Investigación debe existir.")
        with open(PAPER_MD, "r", encoding="utf-8") as f:
            paper_content = f.read()

        self.assertIn(
            mandatory_author,
            paper_content,
            f"{PAPER_MD.name} debe contener la autoría oficial: '{mandatory_author}'.",
        )

        if CASCADE_SCRIPT.exists():
            with open(CASCADE_SCRIPT, "r", encoding="utf-8") as f:
                script_content = f.read()
            self.assertIn(
                mandatory_author,
                script_content,
                f"{CASCADE_SCRIPT.name} debe contener la autoría oficial: '{mandatory_author}'.",
            )

    def test_t4_04_zero_self_praise_adjectives_scan(self):
        """
        Verifica que los documentos no empleen adjetivos valorativos autoelogiosos sobre el propio trabajo
        en conformidad con la Regla 3 del Protocolo de Superación.
        """
        target_docs = [PAPER_MD, INFORME_MD]
        suspicious_matches = []

        for doc in target_docs:
            if not doc.exists():
                continue
            with open(doc, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, start=1):
                clean_line = line.lower()
                for praise in FORBIDDEN_SELF_PRAISE:
                    # Detectar si se usa para describir el propio trabajo ("nuestro trabajo excelente", "resultado impecable")
                    praise_regex = rf"\b(nuestr[oa]s?|mi|este|estos)\s+\w*\s*{praise}\b"
                    if re.search(praise_regex, clean_line):
                        suspicious_matches.append(
                            f"{doc.name}:{line_no} -> Autoelogio potencial '{praise}': '{line.strip()}'"
                        )

        self.assertEqual(
            suspicious_matches,
            [],
            f"Se detectaron posibles expresiones de autoelogio:\n"
            + "\n".join(suspicious_matches),
        )


if __name__ == "__main__":
    unittest.main()
