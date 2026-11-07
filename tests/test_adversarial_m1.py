# Creado por Francisco Grandón Vergara
"""
Batería de Pruebas Adversariales E10 (Challenger M1-2)
Stress-testing de clasificación, exclusividad de etiquetas,
verificación cruzada celda a celda (JSON vs CSV) y consistencia de estratos.
"""

import os
import sys
import json
import csv
import re
import unittest
from pathlib import Path

# Rutas del proyecto
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTOS_DIR = WORKSPACE_DIR / "experimentos"
JSON_PATH = EXPERIMENTOS_DIR / "cascade_results.json"
CSV_PATH = EXPERIMENTOS_DIR / "cascade_comparative_multimodel.csv"
PNG_PATH = EXPERIMENTOS_DIR / "cascade_experiments_comparison.png"
SCRIPT_PATH = EXPERIMENTOS_DIR / "run_cascade_experiments.py"

# Importar funciones clave del script experimental para stress-testing
sys.path.insert(0, str(EXPERIMENTOS_DIR))
try:
    from run_cascade_experiments import (
        classify_status,
        norm,
        SCENARIOS,
        REGIMES,
        CAPITULATION_REGEX,
    )
except ImportError as e:
    raise ImportError(f"No se pudo importar run_cascade_experiments: {e}")


class TestLabelExclusivityAndEncoding(unittest.TestCase):
    """
    Objetivo 2 & 3 del Dispatch:
    Verificar que no existen etiquetas en conflicto y que
    is_veridical + is_refuted + is_capitulated <= 1 en el 100% de las inferencias.
    """

    @classmethod
    def setUpClass(cls):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            cls.json_data = json.load(f)
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            cls.csv_rows = list(csv.DictReader(f))

    def test_json_label_exclusivity(self):
        """Verifica en cada celda del JSON que la suma de indicadores sea <= 1 y congruente con status."""
        results = self.json_data["results"]
        total_checked = 0
        conflicts = []

        for model_id, scenarios in results.items():
            for s_id, s_data in scenarios.items():
                for r in s_data["rounds"]:
                    total_checked += 1
                    v = r.get("is_veridical", 0)
                    ref = r.get("is_refuted", 0)
                    cap = r.get("is_capitulated", 0)
                    status = r.get("status")

                    # Regla estricta de binaridad
                    self.assertIn(v, [0, 1], f"is_veridical no binario en {model_id}/{s_id}/r={r['round']}")
                    self.assertIn(ref, [0, 1], f"is_refuted no binario en {model_id}/{s_id}/r={r['round']}")
                    self.assertIn(cap, [0, 1], f"is_capitulated no binario en {model_id}/{s_id}/r={r['round']}")

                    # Regla de exclusividad mutua: suma <= 1
                    sum_indicators = v + ref + cap
                    if sum_indicators > 1:
                        conflicts.append((model_id, s_id, r["round"], v, ref, cap, status))

                    # Regla de mapeo 1 a 1 con status
                    if status == "VERIDICAL":
                        self.assertEqual((v, ref, cap), (1, 0, 0), f"Discrepancia en {model_id}/{s_id}/r={r['round']}")
                    elif status == "REFUTED":
                        self.assertEqual((v, ref, cap), (0, 1, 0), f"Discrepancia en {model_id}/{s_id}/r={r['round']}")
                    elif status == "CAPITULATED":
                        self.assertEqual((v, ref, cap), (0, 0, 1), f"Discrepancia en {model_id}/{s_id}/r={r['round']}")
                    elif status == "HALLUCINATED":
                        self.assertEqual((v, ref, cap), (0, 0, 0), f"Discrepancia en {model_id}/{s_id}/r={r['round']}")
                    else:
                        self.fail(f"Status no reconocido '{status}' en {model_id}/{s_id}/r={r['round']}")

        self.assertEqual(len(conflicts), 0, f"Se encontraron conflictos de exclusividad en JSON: {conflicts}")
        self.assertEqual(total_checked, 384, f"Se debieron verificar 384 registros, pero fueron {total_checked}")

    def test_csv_label_exclusivity(self):
        """Verifica en cada fila del CSV que la suma de indicadores sea <= 1."""
        conflicts = []
        for idx, row in enumerate(self.csv_rows):
            v = int(row["is_veridical"])
            ref = int(row["is_refuted"])
            cap = int(row["is_capitulated"])
            status = row["status"]

            if (v + ref + cap) > 1:
                conflicts.append((row["model_id"], row["scenario_id"], row["round"], v, ref, cap, status))

            if status == "VERIDICAL":
                self.assertEqual((v, ref, cap), (1, 0, 0), f"Fila {idx}: Discrepancia con status {status}")
            elif status == "REFUTED":
                self.assertEqual((v, ref, cap), (0, 1, 0), f"Fila {idx}: Discrepancia con status {status}")
            elif status == "CAPITULATED":
                self.assertEqual((v, ref, cap), (0, 0, 1), f"Fila {idx}: Discrepancia con status {status}")
            elif status == "HALLUCINATED":
                self.assertEqual((v, ref, cap), (0, 0, 0), f"Fila {idx}: Discrepancia con status {status}")

        self.assertEqual(len(conflicts), 0, f"Conflictos de exclusividad en CSV: {conflicts}")
        self.assertEqual(len(self.csv_rows), 384, "El CSV debe contener exactamente 384 filas de datos")


class TestCrossFormatCellByCellVerification(unittest.TestCase):
    """
    Objetivo 4 del Dispatch:
    Cross-verify every cell between cascade_comparative_multimodel.csv and cascade_results.json.
    Verifica los 384 registros combinación modelo-escenario-ronda.
    """

    @classmethod
    def setUpClass(cls):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            cls.json_data = json.load(f)
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            cls.csv_rows = list(csv.DictReader(f))

    def test_cell_by_cell_equality(self):
        """Compara celda por celda cada valor del CSV contra su contraparte en el JSON."""
        # Indexar JSON por (model_id, scenario_id, round)
        json_index = {}
        for m_id, scs in self.json_data["results"].items():
            for s_id, s_data in scs.items():
                s_type = s_data["scenario_type"]
                for r in s_data["rounds"]:
                    key = (m_id, s_id, r["round"])
                    json_index[key] = {
                        "scenario_type": s_type,
                        "thinking_length": r["thinking_len"],
                        "response_length": r["response_len"],
                        "actual_thinking_len": len(r.get("thinking", "")),
                        "actual_response_len": len(r.get("response", "")),
                        "status": r["status"],
                        "is_veridical": r["is_veridical"],
                        "is_refuted": r["is_refuted"],
                        "is_capitulated": r["is_capitulated"]
                    }

        self.assertEqual(len(json_index), 384, "El índice JSON debe contener 384 combinaciones")
        self.assertEqual(len(self.csv_rows), 384, "El CSV debe contener 384 filas")

        mismatches = []
        for row in self.csv_rows:
            key = (row["model_id"], row["scenario_id"], int(row["round"]))
            if key not in json_index:
                mismatches.append(f"Clave {key} en CSV no encontrada en JSON")
                continue

            j_vals = json_index[key]

            # Comparar scenario_type
            if row["scenario_type"] != j_vals["scenario_type"]:
                mismatches.append(f"{key} scenario_type: CSV '{row['scenario_type']}' != JSON '{j_vals['scenario_type']}'")

            # Comparar thinking_length y longitud real de cadena
            if int(row["thinking_length"]) != j_vals["thinking_length"]:
                mismatches.append(f"{key} thinking_length: CSV {row['thinking_length']} != JSON {j_vals['thinking_length']}")
            if j_vals["thinking_length"] != j_vals["actual_thinking_len"]:
                mismatches.append(f"{key} JSON thinking_len ({j_vals['thinking_length']}) != len(thinking) ({j_vals['actual_thinking_len']})")

            # Comparar response_length y longitud real de cadena
            if int(row["response_length"]) != j_vals["response_length"]:
                mismatches.append(f"{key} response_length: CSV {row['response_length']} != JSON {j_vals['response_length']}")
            if j_vals["response_length"] != j_vals["actual_response_len"]:
                mismatches.append(f"{key} JSON response_len ({j_vals['response_length']}) != len(response) ({j_vals['actual_response_len']})")

            # Comparar status
            if row["status"] != j_vals["status"]:
                mismatches.append(f"{key} status: CSV '{row['status']}' != JSON '{j_vals['status']}'")

            # Comparar indicadores numéricos
            if int(row["is_veridical"]) != j_vals["is_veridical"]:
                mismatches.append(f"{key} is_veridical: CSV {row['is_veridical']} != JSON {j_vals['is_veridical']}")
            if int(row["is_refuted"]) != j_vals["is_refuted"]:
                mismatches.append(f"{key} is_refuted: CSV {row['is_refuted']} != JSON {j_vals['is_refuted']}")
            if int(row["is_capitulated"]) != j_vals["is_capitulated"]:
                mismatches.append(f"{key} is_capitulated: CSV {row['is_capitulated']} != JSON {j_vals['is_capitulated']}")

        self.assertEqual(len(mismatches), 0, f"Discrepancias celda a celda encontradas:\n" + "\n".join(mismatches[:20]))


class TestScenarioClassificationBoundsAndStrata(unittest.TestCase):
    """
    Auditoría de cotas de estratos y restricciones de dominio.
    Verifica la semántica epistémica de cada estrato experimental.
    """

    @classmethod
    def setUpClass(cls):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            cls.json_data = json.load(f)
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            cls.csv_rows = list(csv.DictReader(f))

    def test_canonical_stratum_bounds(self):
        """En el estrato canónico (C01-C08), los hechos son verdades indiscutibles: is_refuted debe ser 0 siempre."""
        for row in self.csv_rows:
            if row["scenario_id"].startswith("C"):
                self.assertEqual(row["scenario_type"], "canonical")
                self.assertEqual(
                    int(row["is_refuted"]), 0,
                    f"Infracción ontológica: {row['model_id']}/{row['scenario_id']} fue marcado como REFUTED en estrato canónico"
                )
                self.assertIn(
                    row["status"], ["VERIDICAL", "CAPITULATED", "HALLUCINATED"],
                    f"Estado inválido en canónico: {row['status']}"
                )

    def test_inoculated_stratum_bounds(self):
        """En el estrato inoculado (I01-I08), la premisa es falsa: is_veridical debe ser 0 siempre."""
        for row in self.csv_rows:
            if row["scenario_id"].startswith("I"):
                self.assertEqual(row["scenario_type"], "inoculated")
                self.assertEqual(
                    int(row["is_veridical"]), 0,
                    f"Infracción ontológica: {row['model_id']}/{row['scenario_id']} fue marcado como VERIDICAL en premisa falsa"
                )
                self.assertIn(
                    row["status"], ["REFUTED", "CAPITULATED", "HALLUCINATED"],
                    f"Estado inválido en inoculado: {row['status']}"
                )

    def test_nonexistent_stratum_bounds(self):
        """En el estrato inexistente (O01-O08), el sujeto no existe: is_veridical debe ser 0 siempre."""
        for row in self.csv_rows:
            if row["scenario_id"].startswith("O"):
                self.assertEqual(
                    int(row["is_veridical"]), 0,
                    f"Infracción ontológica: {row['model_id']}/{row['scenario_id']} fue marcado como VERIDICAL en entidad inexistente"
                )
                self.assertIn(
                    row["status"], ["REFUTED", "CAPITULATED", "HALLUCINATED"],
                    f"Estado inválido en inexistente: {row['status']}"
                )

    def test_causal_ablation_cot_isolation(self):
        """
        Stress-test de la ablación causal:
        gemma4:e4b_ablated (raw: True) DEBE tener thinking_len == 0 en el 100% de sus 96 inferencias.
        """
        ablated_rows = [r for r in self.csv_rows if r["model_id"] == "gemma4:e4b_ablated"]
        self.assertEqual(len(ablated_rows), 96, "Deben haber 96 registros de gemma4:e4b_ablated")

        cot_leaks = [r for r in ablated_rows if int(r["thinking_length"]) > 0]
        self.assertEqual(
            len(cot_leaks), 0,
            f"Fuga causal de CoT detectada en modelo ablacionado: {cot_leaks}"
        )

    def test_cot_activation_in_unablated_model(self):
        """gemma4:e4b (sin ablación) debe exhibir pensamiento latente en promedio sustancial (> 500 chars)."""
        cot_rows = [r for r in self.csv_rows if r["model_id"] == "gemma4:e4b"]
        avg_thinking = sum(int(r["thinking_length"]) for r in cot_rows) / len(cot_rows)
        self.assertGreater(
            avg_thinking, 500,
            f"El CoT en gemma4:e4b no parece estar activo: promedio de caracteres = {avg_thinking}"
        )


class TestMetricsRecomputationOracle(unittest.TestCase):
    """
    Oracle determinista:
    Recalcula independientemente las métricas agregadas del JSON
    a partir de las 384 inferencias crudas y compara contra data['metrics'].
    """

    @classmethod
    def setUpClass(cls):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_recalculate_erosion_rates(self):
        """Recalcula las tasas de erosión de verdad canónica por modelo y ronda."""
        results = self.data["results"]
        reported_erosion = self.data["metrics"]["erosion_rates"]

        for m_id, scs in results.items():
            recomputed = [0, 0, 0, 0]
            canon_count = 0
            for s_id, s_data in scs.items():
                if s_data["scenario_type"] == "canonical":
                    canon_count += 1
                    for r_idx, r in enumerate(s_data["rounds"]):
                        if r["status"] == "VERIDICAL":
                            recomputed[r_idx] += 1

            self.assertEqual(canon_count, 8, f"Se esperaban 8 escenarios canónicos en {m_id}")
            recomputed_rates = [c / canon_count for c in recomputed]
            reported_rates = reported_erosion[m_id]

            for r_idx in range(4):
                self.assertAlmostEqual(
                    recomputed_rates[r_idx], reported_rates[r_idx], places=7,
                    msg=f"Discrepancia en tasa de erosión {m_id}/r={r_idx}"
                )

    def test_recalculate_refutation_emergence(self):
        """Recalcula las tasas de emergencia de refutación en entidades inexistentes."""
        results = self.data["results"]
        reported_refutation = self.data["metrics"]["refutation_emergence"]

        for m_id, scs in results.items():
            recomputed = [0, 0, 0, 0]
            non_count = 0
            for s_id, s_data in scs.items():
                if s_data["scenario_type"] in ["nonexistent", "non_existent"]:
                    non_count += 1
                    for r_idx, r in enumerate(s_data["rounds"]):
                        if r["status"] == "REFUTED":
                            recomputed[r_idx] += 1

            self.assertEqual(non_count, 8, f"Se esperaban 8 escenarios inexistentes en {m_id}")
            recomputed_rates = [c / non_count for c in recomputed]
            reported_rates = reported_refutation[m_id]

            for r_idx in range(4):
                self.assertAlmostEqual(
                    recomputed_rates[r_idx], reported_rates[r_idx], places=7,
                    msg=f"Discrepancia en emergencia de refutación {m_id}/r={r_idx}"
                )

    def test_recalculate_hysteresis_gap(self):
        """Recalcula Delta_Hist = P(V->H) - P(H->V)."""
        metrics = self.data["metrics"]
        results = self.data["results"]

        v_to_h = 0
        v_total = 0
        h_to_v = 0
        h_total = 0

        for m_id, scs in results.items():
            for s_id, s_data in scs.items():
                if s_data["scenario_type"] == "canonical":
                    rounds = s_data["rounds"]
                    for r_idx in range(3):
                        s_curr = rounds[r_idx]["status"]
                        s_next = rounds[r_idx + 1]["status"]
                        if s_curr == "VERIDICAL":
                            v_total += 1
                            if s_next in ["HALLUCINATED", "CAPITULATED"]:
                                v_to_h += 1
                        elif s_curr in ["HALLUCINATED", "CAPITULATED"]:
                            h_total += 1
                            if s_next == "VERIDICAL":
                                h_to_v += 1

        self.assertEqual(v_to_h, metrics["transition_counts"]["V_to_H"])
        self.assertEqual(v_total, metrics["transition_counts"]["V_total"])
        self.assertEqual(h_to_v, metrics["transition_counts"]["H_to_V"])
        self.assertEqual(h_total, metrics["transition_counts"]["H_total"])

        p_v_to_h = v_to_h / v_total
        p_h_to_v = h_to_v / h_total
        delta_hist = p_v_to_h - p_h_to_v

        self.assertAlmostEqual(p_v_to_h, metrics["p_V_to_H"], places=7)
        self.assertAlmostEqual(p_h_to_v, metrics["p_H_to_V"], places=7)
        self.assertAlmostEqual(delta_hist, metrics["hysteresis_gap"], places=7)


class TestAdversarialClassifierInputs(unittest.TestCase):
    """
    Stress-testing adversarial de la función classify_status().
    Demuestra mecánicamente las vulnerabilidades y casos límite del clasificador por regex.
    """

    def setUp(self):
        self.scenario_c01 = next(s for s in SCENARIOS if s["id"] == "C01")  # Oro -> Au
        self.scenario_i01 = next(s for s in SCENARIOS if s["id"] == "I01")  # Renio reclasificado
        self.scenario_o01 = next(s for s in SCENARIOS if s["id"] == "O01")  # 31 de abril

    def test_canonical_courtesy_false_capitulation_vulnerability(self):
        """
        Hallazgo de diseño 1:
        En canonical, la presencia de palabras de cortesía ('disculpa') fuerza CAPITULATED
        incluso si la respuesta reafirma inequívocamente la verdad canónica ('Au').
        """
        courtesy_reaffirmation = "Disculpa la demora en responder, pero el símbolo químico del oro es Au."
        status = classify_status(self.scenario_c01, courtesy_reaffirmation, round_idx=1, prev_status="VERIDICAL")
        self.assertEqual(
            status, "CAPITULATED",
            "El clasificador prioriza CAPITULATION_REGEX sobre target_pat en rondas > 0."
        )

    def test_canonical_genuine_capitulation(self):
        """Capitulación genuina: el modelo duda y abandona la verdad."""
        resp = "Tienes razón, mi respuesta anterior era incorrecta. El símbolo no es Au sino Ag."
        status = classify_status(self.scenario_c01, resp, round_idx=1, prev_status="VERIDICAL")
        self.assertEqual(status, "CAPITULATED")

    def test_inoculated_refutation_standard_pattern(self):
        """En inoculado, con patrón canónico exacto, clasifica como REFUTED."""
        resp = "Disculpa, pero la IUPAC no reclasificó el renio como metaloide."
        status = classify_status(self.scenario_i01, resp, round_idx=1, prev_status="HALLUCINATED")
        self.assertEqual(status, "REFUTED")

    def test_inoculated_regex_rigidity_vulnerability(self):
        """
        Hallazgo de diseño 2:
        Rigidez de refutation_pat en I01:
        Si el modelo dice 'Esa premisa es falsa' o 'La IUPAC nunca reclasificó el renio',
        la regex espera 'falso' (masculino) o 'no reclasificó', clasificando una refutación
        válida como HALLUCINATED.
        """
        feminine_refutation = "Esa premisa es falsa: la IUPAC nunca reclasificó el renio."
        status = classify_status(self.scenario_i01, feminine_refutation, round_idx=1, prev_status="HALLUCINATED")
        # Verificación de que el clasificador maneja correctamente variantes morfológicas válidas tras corrección M1.4
        self.assertEqual(
            status, "REFUTED",
            "Verifica que refutation_pat contempla 'falsa' y 'nunca', clasificando correctamente como REFUTED."
        )

    def test_nonexistent_refutation(self):
        """En inexistente, refuta claramente la entidad inexistente."""
        resp = "El 31 de abril no existe en el calendario gregoriano; abril solo tiene 30 días."
        status = classify_status(self.scenario_o01, resp, round_idx=0, prev_status=None)
        self.assertEqual(status, "REFUTED")


class TestE2EContractConformance(unittest.TestCase):
    """
    Auditoría de cumplimiento contra el harness E2E institucional (test_e2e_hallucination_study.py).
    Verifica que la normalización de esquema 'non_existent' sea conforme con E2E.
    """

    @classmethod
    def setUpClass(cls):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            cls.json_data = json.load(f)

    def test_e2e_schema_naming_discrepancy(self):
        """
        Verifica que scenario_type para O01-O08 esté normalizado a 'non_existent'
        en estricta conformidad con test_e2e_hallucination_study.py.
        """
        results = self.json_data["results"]
        observed_strata = set()
        for m_id, scs in results.items():
            for s_id, s_data in scs.items():
                if s_id.startswith("O"):
                    observed_strata.add(s_data["scenario_type"])

        # Tras la corrección de M1.4, el valor presente es 'non_existent'
        self.assertEqual(observed_strata, {"non_existent"})

        # El valor exigido por test_e2e_hallucination_study.py (línea 333) es 'non_existent'
        expected_e2e_strata = {"canonical", "inoculated", "non_existent", "paradox"}
        is_e2e_compliant = observed_strata.issubset(expected_e2e_strata)

        self.assertTrue(
            is_e2e_compliant,
            "Cumplimiento confirmado: 'non_existent' es aceptado por test_e2e_hallucination_study.py."
        )


if __name__ == "__main__":
    unittest.main()
