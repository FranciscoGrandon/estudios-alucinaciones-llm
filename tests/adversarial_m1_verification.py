# Creado por Francisco Grandón Vergara
"""
Script de Verificación Adversarial e Invariantes Empíricas: Hito M1
Valida rigurosamente cascade_results.json y cascade_comparative_multimodel.csv.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
CASCADE_JSON = WORKSPACE_DIR / "experimentos" / "cascade_results.json"
CASCADE_CSV = WORKSPACE_DIR / "experimentos" / "cascade_comparative_multimodel.csv"

EXPECTED_MODELS = [
    "gemma2:2b",
    "gemma3:4b",
    "gemma4:e4b",
    "gemma4:e4b_ablated",
]

EXPECTED_SCENARIOS = {
    "canonical": [f"C0{i}" for i in range(1, 9)],
    "inoculated": [f"I0{i}" for i in range(1, 9)],
    "nonexistent": [f"O0{i}" for i in range(1, 9)],
}
ALL_SCENARIO_IDS = [s for sub in EXPECTED_SCENARIOS.values() for s in sub]
VALID_STATUSES = {"VERIDICAL", "HALLUCINATED", "REFUTED", "CAPITULATED"}


def run_all_checks():
    report = {
        "summary": {},
        "invariants": {},
        "failures": [],
        "warnings": [],
    }

    print("=" * 80)
    print("INICIANDO VERIFICACIÓN ADVERSARIAL DE ARTEFACTOS DEL HITO M1")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Existencia de Archivos
    # --------------------------------------------------------------------------
    if not CASCADE_JSON.exists():
        print(f"[CRITICAL FAIL] No existe {CASCADE_JSON}")
        report["failures"].append(f"Archivo inexistente: {CASCADE_JSON}")
        return report
    if not CASCADE_CSV.exists():
        print(f"[CRITICAL FAIL] No existe {CASCADE_CSV}")
        report["failures"].append(f"Archivo inexistente: {CASCADE_CSV}")
        return report

    with open(CASCADE_JSON, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    with open(CASCADE_CSV, "r", encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))

    # --------------------------------------------------------------------------
    # 2. Invariante 1: Cardinalidad y Grid Factorial (384 registros exactos)
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 1] Verificando cardinalidad y diseño factorial (384 inferencias)...")
    inv1_passed = True
    inv1_details = {}

    json_records_count = 0
    results = json_data.get("results", {})
    for m in EXPECTED_MODELS:
        m_data = results.get(m, {})
        for s in ALL_SCENARIO_IDS:
            s_data = m_data.get(s, {})
            rounds = s_data.get("rounds", [])
            json_records_count += len(rounds)

    csv_records_count = len(csv_rows)

    inv1_details["json_count"] = json_records_count
    inv1_details["csv_count"] = csv_records_count
    inv1_details["expected"] = 384

    if json_records_count != 384:
        inv1_passed = False
        msg = f"JSON contiene {json_records_count} inferencias (esperado: 384)"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print(f"  PASSED: JSON tiene exactamente {json_records_count} inferencias.")

    if csv_records_count != 384:
        inv1_passed = False
        msg = f"CSV contiene {csv_records_count} filas de datos (esperado: 384)"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print(f"  PASSED: CSV tiene exactamente {csv_records_count} filas de datos.")

    report["invariants"]["cardinality_384"] = {
        "passed": inv1_passed,
        "details": inv1_details,
    }

    # --------------------------------------------------------------------------
    # 3. Invariante 2: Campos Válidos, Longitudes de Respuesta y No-Vaciedad
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 2] Verificando campos válidos, longitud de respuesta > 0 y consistencia...")
    inv2_passed = True
    empty_responses = []
    empty_prompts = []
    invalid_rounds = []
    invalid_statuses = []
    len_mismatches = []

    for m in EXPECTED_MODELS:
        m_dict = results.get(m, {})
        for s in ALL_SCENARIO_IDS:
            s_dict = m_dict.get(s, {})
            rounds = s_dict.get("rounds", [])
            if len(rounds) != 4:
                inv2_passed = False
                report["failures"].append(f"Escenario {m}/{s} tiene {len(rounds)} rondas (esperado: 4)")
            for idx, r in enumerate(rounds):
                r_num = r.get("round")
                if r_num != idx or r_num not in [0, 1, 2, 3]:
                    invalid_rounds.append(f"{m}/{s}/r={r_num}")

                p = r.get("prompt", "")
                if not p or len(p.strip()) == 0:
                    empty_prompts.append(f"{m}/{s}/r={r_num}")

                resp = r.get("response", "")
                if not resp or len(resp.strip()) == 0:
                    empty_responses.append(f"{m}/{s}/r={r_num}")

                st = r.get("status")
                if st not in VALID_STATUSES:
                    invalid_statuses.append(f"{m}/{s}/r={r_num}: {st}")

                resp_len = r.get("response_len", 0)
                if resp_len != len(resp):
                    len_mismatches.append(f"{m}/{s}/r={r_num}: reported {resp_len} != actual {len(resp)}")

                think = r.get("thinking", "")
                think_len = r.get("thinking_len", 0)
                if think_len != len(think):
                    len_mismatches.append(f"{m}/{s}/r={r_num}: reported think {think_len} != actual {len(think)}")

    if empty_responses:
        inv2_passed = False
        msg = f"Detectadas {len(empty_responses)} respuestas vacías: {empty_responses[:5]}"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: 100% de las 384 inferencias tienen respuestas no vacías (response_len > 0).")

    if empty_prompts:
        inv2_passed = False
        msg = f"Detectados {len(empty_prompts)} prompts vacíos: {empty_prompts[:5]}"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: 100% de los 384 prompts están definidos y no vacíos.")

    if invalid_rounds:
        inv2_passed = False
        msg = f"Detectadas rondas inválidas: {invalid_rounds[:5]}"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: Rondas estrictamente secuenciales [0, 1, 2, 3] en todas las inferencias.")

    if invalid_statuses:
        inv2_passed = False
        msg = f"Detectados statuses inválidos: {invalid_statuses[:5]}"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print(f"  PASSED: 100% de los tags de status pertenecen al conjunto cerrado: {VALID_STATUSES}.")

    if len_mismatches:
        inv2_passed = False
        msg = f"Discrepancias entre longitud declarada y real: {len_mismatches[:5]}"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: Longitudes declaradas (response_len, thinking_len) coinciden exactamente con len().")

    report["invariants"]["fields_and_lengths"] = {
        "passed": inv2_passed,
        "empty_responses": len(empty_responses),
        "empty_prompts": len(empty_prompts),
        "invalid_statuses": len(invalid_statuses),
    }

    # --------------------------------------------------------------------------
    # 4. Invariante 3: Coherencia de Flags Binarios y Mutex de Status
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 3] Verificando consistencia de flags binarios (is_veridical, is_refuted, is_capitulated)...")
    inv3_passed = True
    flag_inconsistencies = []

    for m in EXPECTED_MODELS:
        m_dict = results.get(m, {})
        for s in ALL_SCENARIO_IDS:
            rounds = m_dict.get(s, {}).get("rounds", [])
            for r in rounds:
                st = r.get("status")
                iv = r.get("is_veridical")
                ir = r.get("is_refuted")
                ic = r.get("is_capitulated")

                # Chequeo de valores binarios
                if iv not in [0, 1] or ir not in [0, 1] or ic not in [0, 1]:
                    flag_inconsistencies.append(f"{m}/{s}/r={r['round']}: Non-binary flags ({iv}, {ir}, {ic})")

                # Chequeo de exclusión mutua
                if iv + ir + ic > 1:
                    flag_inconsistencies.append(f"{m}/{s}/r={r['round']}: Multiple flags active ({iv}, {ir}, {ic})")

                # Coherencia con status
                if st == "VERIDICAL" and not (iv == 1 and ir == 0 and ic == 0):
                    flag_inconsistencies.append(f"{m}/{s}/r={r['round']}: VERIDICAL mismatch with flags")
                elif st == "REFUTED" and not (iv == 0 and ir == 1 and ic == 0):
                    flag_inconsistencies.append(f"{m}/{s}/r={r['round']}: REFUTED mismatch with flags")
                elif st == "CAPITULATED" and not (iv == 0 and ir == 0 and ic == 1):
                    flag_inconsistencies.append(f"{m}/{s}/r={r['round']}: CAPITULATED mismatch with flags")
                elif st == "HALLUCINATED" and not (iv == 0 and ir == 0 and ic == 0):
                    flag_inconsistencies.append(f"{m}/{s}/r={r['round']}: HALLUCINATED mismatch with flags")

    if flag_inconsistencies:
        inv3_passed = False
        msg = f"Inconsistencias en flags binarios detectadas: {len(flag_inconsistencies)} (muestra: {flag_inconsistencies[:5]})"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: 100% de coherencia lógica y exclusión mutua entre status y flags binarios.")

    report["invariants"]["flag_consistency"] = {
        "passed": inv3_passed,
        "violations": len(flag_inconsistencies),
    }

    # --------------------------------------------------------------------------
    # 5. Invariante 4: Alineación Celda a Celda entre JSON y CSV
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 4] Verificando alineación estricta celda a celda entre JSON y CSV...")
    inv4_passed = True
    alignment_errors = []

    # Mapear filas de CSV por (model_id, scenario_id, round)
    csv_map = {}
    for r in csv_rows:
        key = (r["model_id"], r["scenario_id"], int(r["round"]))
        csv_map[key] = r

    for m in EXPECTED_MODELS:
        m_dict = results.get(m, {})
        for s in ALL_SCENARIO_IDS:
            rounds = m_dict.get(s, {}).get("rounds", [])
            for r in rounds:
                key = (m, s, r["round"])
                if key not in csv_map:
                    alignment_errors.append(f"Fila {key} no encontrada en CSV")
                    continue
                c_row = csv_map[key]

                if int(c_row["thinking_length"]) != r["thinking_len"]:
                    alignment_errors.append(f"{key}: CSV thinking {c_row['thinking_length']} != JSON {r['thinking_len']}")
                if int(c_row["response_length"]) != r["response_len"]:
                    alignment_errors.append(f"{key}: CSV response {c_row['response_length']} != JSON {r['response_len']}")
                if c_row["status"] != r["status"]:
                    alignment_errors.append(f"{key}: CSV status {c_row['status']} != JSON {r['status']}")
                if int(c_row["is_veridical"]) != r["is_veridical"]:
                    alignment_errors.append(f"{key}: CSV is_veridical != JSON")
                if int(c_row["is_refuted"]) != r["is_refuted"]:
                    alignment_errors.append(f"{key}: CSV is_refuted != JSON")
                if int(c_row["is_capitulated"]) != r["is_capitulated"]:
                    alignment_errors.append(f"{key}: CSV is_capitulated != JSON")

    if alignment_errors:
        inv4_passed = False
        msg = f"Desalineaciones entre JSON y CSV detectadas: {len(alignment_errors)} (muestra: {alignment_errors[:5]})"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: 100% de alineación y consistencia exacta entre cascade_results.json y cascade_comparative_multimodel.csv.")

    report["invariants"]["json_csv_alignment"] = {
        "passed": inv4_passed,
        "mismatches": len(alignment_errors),
    }

    # --------------------------------------------------------------------------
    # 6. Invariante 5: Invariante Estricta de Ablación CoT (thinking_len)
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 5] Verificando invariante estricta de ablación CoT (thinking_len)...")
    inv5_passed = True

    # 1. gemma4:e4b_ablated -> 100% de las 96 inferencias DEBEN tener thinking_len == 0
    ablated_rounds = [
        r
        for s in results.get("gemma4:e4b_ablated", {}).values()
        for r in s.get("rounds", [])
    ]
    ablated_non_zero = [r for r in ablated_rounds if r.get("thinking_len", 0) > 0 or len(r.get("thinking", "")) > 0]
    print(f"  gemma4:e4b_ablated: Total={len(ablated_rounds)}, No-cero={len(ablated_non_zero)}")

    if len(ablated_rounds) != 96:
        inv5_passed = False
        report["failures"].append(f"gemma4:e4b_ablated tiene {len(ablated_rounds)} rondas (esperado: 96)")
    if len(ablated_non_zero) > 0:
        inv5_passed = False
        msg = f"Violación de ablación CoT en gemma4:e4b_ablated: {len(ablated_non_zero)} inferencias con thinking > 0"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: gemma4:e4b_ablated cumple 100% con thinking_len == 0 y thinking == '' (96/96).")

    # 2. gemma2:2b y gemma3:4b -> 100% de las 192 inferencias DEBEN tener thinking_len == 0
    for baseline_m in ["gemma2:2b", "gemma3:4b"]:
        b_rounds = [
            r
            for s in results.get(baseline_m, {}).values()
            for r in s.get("rounds", [])
        ]
        b_non_zero = [r for r in b_rounds if r.get("thinking_len", 0) > 0 or len(r.get("thinking", "")) > 0]
        print(f"  {baseline_m}: Total={len(b_rounds)}, No-cero={len(b_non_zero)}")
        if len(b_rounds) != 96:
            inv5_passed = False
            report["failures"].append(f"{baseline_m} tiene {len(b_rounds)} rondas (esperado: 96)")
        if len(b_non_zero) > 0:
            inv5_passed = False
            msg = f"Pensamiento inesperado en {baseline_m}: {len(b_non_zero)} inferencias con thinking > 0"
            report["failures"].append(msg)
            print(f"  FAILED: {msg}")
        else:
            print(f"  PASSED: {baseline_m} cumple 100% con thinking_len == 0 (96/96).")

    # 3. gemma4:e4b (CoT activo) -> debe tener thinking_len > 0 en >= 80% de las inferencias
    g4_rounds = [
        r
        for s in results.get("gemma4:e4b", {}).values()
        for r in s.get("rounds", [])
    ]
    g4_active = [r for r in g4_rounds if r.get("thinking_len", 0) > 0]
    g4_active_pct = len(g4_active) / len(g4_rounds) if g4_rounds else 0.0
    print(f"  gemma4:e4b: Total={len(g4_rounds)}, Activo={len(g4_active)} ({g4_active_pct * 100:.2f}%)")

    if g4_active_pct < 0.80:
        inv5_passed = False
        msg = f"gemma4:e4b tiene solo {g4_active_pct * 100:.2f}% inferencias con thinking activo (< 80%)"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print(f"  PASSED: gemma4:e4b exhibe thinking activo en {g4_active_pct * 100:.2f}% de inferencias (>= 80%).")

    # Estadísticas de longitud de thinking en gemma4:e4b
    g4_lens = [r["thinking_len"] for r in g4_rounds]
    g4_min, g4_max = min(g4_lens), max(g4_lens)
    g4_mean = sum(g4_lens) / len(g4_lens)
    print(f"  Estadísticas gemma4:e4b thinking_len -> Min: {g4_min}, Max: {g4_max}, Media: {g4_mean:.1f} caracteres.")

    report["invariants"]["cot_ablation"] = {
        "passed": inv5_passed,
        "ablated_zero_pct": 1.0 - (len(ablated_non_zero) / len(ablated_rounds)),
        "g4_active_pct": g4_active_pct,
        "g4_stats": {"min": g4_min, "max": g4_max, "mean": g4_mean},
    }

    # --------------------------------------------------------------------------
    # 7. Invariante 6: Matrices de Transición y Cómputo de Brecha de Histéresis
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 6] Verificando matrices de transición y cómputo de histéresis...")
    inv6_passed = True

    # Re-cómputo independiente sobre escenario canónico conforme al código del worker
    recomputed_v_total = 0
    recomputed_v_to_h = 0
    recomputed_h_total = 0
    recomputed_h_to_v = 0

    # También matriz global completa de 4 estados
    transitions_4x4 = defaultdict(lambda: defaultdict(int))
    transitions_stratum = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for m in EXPECTED_MODELS:
        m_dict = results.get(m, {})
        for s, s_data in m_dict.items():
            s_type = s_data.get("scenario_type")
            rounds = s_data.get("rounds", [])
            for r_idx in range(len(rounds) - 1):
                curr_st = rounds[r_idx]["status"]
                next_st = rounds[r_idx + 1]["status"]
                transitions_4x4[curr_st][next_st] += 1
                transitions_stratum[s_type][curr_st][next_st] += 1

                if s_type == "canonical":
                    if curr_st == "VERIDICAL":
                        recomputed_v_total += 1
                        if next_st in ["HALLUCINATED", "CAPITULATED"]:
                            recomputed_v_to_h += 1
                    elif curr_st in ["HALLUCINATED", "CAPITULATED"]:
                        recomputed_h_total += 1
                        if next_st == "VERIDICAL":
                            recomputed_h_to_v += 1

    reported_metrics = json_data.get("metrics", {})
    rep_tc = reported_metrics.get("transition_counts", {})
    rep_v_to_h = rep_tc.get("V_to_H")
    rep_v_total = rep_tc.get("V_total")
    rep_h_to_v = rep_tc.get("H_to_V")
    rep_h_total = rep_tc.get("H_total")

    print(f"  Worker counts   : V_to_H={rep_v_to_h}, V_total={rep_v_total}, H_to_V={rep_h_to_v}, H_total={rep_h_total}")
    print(f"  Recomputed counts: V_to_H={recomputed_v_to_h}, V_total={recomputed_v_total}, H_to_V={recomputed_h_to_v}, H_total={recomputed_h_total}")

    if (rep_v_to_h != recomputed_v_to_h or rep_v_total != recomputed_v_total or
            rep_h_to_v != recomputed_h_to_v or rep_h_total != recomputed_h_total):
        inv6_passed = False
        msg = "Discrepancia en recuento de transiciones entre JSON y cálculo independiente"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: Recuentos de transiciones coinciden con precisión matemática absoluta.")

    calc_p_v_to_h = recomputed_v_to_h / recomputed_v_total
    calc_p_h_to_v = recomputed_h_to_v / recomputed_h_total
    calc_gap = calc_p_v_to_h - calc_p_h_to_v

    rep_gap = reported_metrics.get("hysteresis_gap", 0.0)
    rep_p_v_to_h = reported_metrics.get("p_V_to_H", 0.0)
    rep_p_h_to_v = reported_metrics.get("p_H_to_V", 0.0)

    print(f"  P(V->H) calculado: {calc_p_v_to_h:.6f} vs Reportado: {rep_p_v_to_h:.6f}")
    print(f"  P(H->V) calculado: {calc_p_h_to_v:.6f} vs Reportado: {rep_p_h_to_v:.6f}")
    print(f"  Hysteresis Gap   : {calc_gap:.6f} vs Reportado: {rep_gap:.6f}")

    if abs(calc_gap - rep_gap) > 1e-9:
        inv6_passed = False
        msg = f"Discrepancia en hysteresis_gap ({calc_gap} vs {rep_gap})"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print(f"  PASSED: hysteresis_gap coincide exactamente ({rep_gap:.6f} = {rep_gap * 100:.2f}%).")

    # Verificación de propiedades estocásticas de la matriz de transición completa
    print("\n  [Matriz de Transición Empírica Global 4x4 (384 inferencias, 288 transiciones)]:")
    stochastic_matrix_valid = True
    for st_from in sorted(VALID_STATUSES):
        total_out = sum(transitions_4x4[st_from].values())
        if total_out > 0:
            probs = {st_to: transitions_4x4[st_from][st_to] / total_out for st_to in sorted(VALID_STATUSES)}
            p_sum = sum(probs.values())
            print(f"    From {st_from:12s} (N={total_out:3d}): " +
                  ", ".join([f"{to}: {p:.3f}" for to, p in probs.items()]) +
                  f" | Suma={p_sum:.4f}")
            if abs(p_sum - 1.0) > 1e-6:
                stochastic_matrix_valid = False
        else:
            print(f"    From {st_from:12s} (N=  0): [Estado absorbente o sin transiciones originadas]")

    if not stochastic_matrix_valid:
        inv6_passed = False
        msg = "Violación de propiedad estocástica: las filas de la matriz no suman 1.0"
        report["failures"].append(msg)
        print(f"  FAILED: {msg}")
    else:
        print("  PASSED: Todas las filas de la matriz de transición completa suman exactamente 1.0 (propiedad estocástica).")

    report["invariants"]["transition_and_hysteresis"] = {
        "passed": inv6_passed,
        "p_v_to_h": calc_p_v_to_h,
        "p_h_to_v": calc_p_h_to_v,
        "hysteresis_gap": calc_gap,
        "stochastic_matrix_valid": stochastic_matrix_valid,
        "transitions_4x4": {k: dict(v) for k, v in transitions_4x4.items()},
    }

    # --------------------------------------------------------------------------
    # 8. Invariante 7: Análisis de Coherencia de Esquema y Contratos de Interfaz
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 7] Verificando contratos de interfaz y nombres de estratos...")
    inv7_passed = True
    contract_issues = []

    # 1. Chequeo de scenario_type en JSON y CSV
    json_types = set()
    for m in EXPECTED_MODELS:
        for s, s_data in results.get(m, {}).items():
            json_types.add(s_data.get("scenario_type"))

    csv_types = set(r["scenario_type"] for r in csv_rows)

    print(f"  Tipos de escenario encontrados en JSON: {json_types}")
    print(f"  Tipos de escenario encontrados en CSV : {csv_types}")

    # Verificar si usa 'nonexistent' o 'non_existent'
    if "nonexistent" in json_types and "non_existent" not in json_types:
        msg = (
            "DIVERGENCIA DE CONTRATO DETECTADA: El worker generó 'nonexistent' en JSON y CSV, "
            "pero el test E2E (test_t2_01 en test_e2e_hallucination_study.py) y la especificación canónica "
            "esperan 'non_existent'. Esto ocasiona fallo inmediato en la suite E2E automatizada."
        )
        contract_issues.append(msg)
        print(f"  [HALLAZGO / ERROR DE CONTRATO]: {msg}")

    # 2. Chequeo de clave 'transition_matrix' en metrics
    if "transition_matrix" not in reported_metrics:
        msg = (
            "OMISIÓN EN METRICS: cascade_results.json contiene 'transition_counts', 'p_V_to_H', 'p_H_to_V' "
            "y 'hysteresis_gap', pero omite la clave 'transition_matrix' esperada por el test E2E test_t3_04."
        )
        report["warnings"].append(msg)
        print(f"  [WARNING]: {msg}")

    if contract_issues:
        inv7_passed = False
        report["failures"].extend(contract_issues)

    report["invariants"]["interface_contracts"] = {
        "passed": inv7_passed,
        "json_types": list(json_types),
        "csv_types": list(csv_types),
        "issues": contract_issues,
    }

    # --------------------------------------------------------------------------
    # 9. Invariante 8: Integridad Semántica del Clasificador y Detección de Falsos Positivos
    # --------------------------------------------------------------------------
    print("\n[INVARIANTE 8] Verificando integridad semántica del clasificador y falsos positivos...")
    inv8_passed = True
    semantic_false_positives = []

    # Detectar si alguna respuesta etiquetada como REFUTED en entidades inexistentes
    # en realidad defiende la existencia de la entidad diciendo "no existen alucinaciones"
    for m in EXPECTED_MODELS:
        m_dict = results.get(m, {})
        for s in ALL_SCENARIO_IDS:
            if s.startswith("O"):
                rounds = m_dict.get(s, {}).get("rounds", [])
                for r in rounds:
                    if r.get("status") == "REFUTED":
                        resp_lower = r.get("response", "").lower()
                        if "no existen alucinaciones" in resp_lower or "no hay errores factuales" in resp_lower:
                            # Verificar si la respuesta en realidad defiende la entidad inexistente
                            # Caso O03 (Comuna Ficticia Puerto Alto del Biobío)
                            if s == "O03" and "puerto alto" in resp_lower and "ine" in resp_lower:
                                msg = (
                                    f"FALSO POSITIVO CRÍTICO EN CLASIFICADOR: En {m}/{s}/r={r['round']}, "
                                    f"el modelo defendió la existencia de la comuna ficticia afirmando "
                                    f"'no existen alucinaciones ni errores factuales'. El patrón regex 'no existe' "
                                    f"hizo match por subcadena, clasificando una alucinación persistente como 'REFUTED'."
                                )
                                semantic_false_positives.append(msg)
                                print(f"  [ERROR SEMÁNTICO]: {msg}")

    if semantic_false_positives:
        inv8_passed = False
        report["failures"].extend(semantic_false_positives)
    else:
        print("  PASSED: No se detectaron falsos positivos flagrantes en la clasificación de refutaciones.")

    report["invariants"]["semantic_classification"] = {
        "passed": inv8_passed,
        "false_positives": semantic_false_positives,
    }

    # --------------------------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESUMEN DE VERIFICACIÓN ADVERSARIAL:")
    total_invariants = len(report["invariants"])
    passed_invariants = sum(1 for inv in report["invariants"].values() if inv["passed"])
    print(f"Invariantes Evaluadas: {total_invariants}")
    print(f"Invariantes Aprobadas: {passed_invariants}")
    print(f"Invariantes Fallidas : {total_invariants - passed_invariants}")

    if report["failures"]:
        print("\nFALLOS DETECTADOS:")
        for idx, f in enumerate(report["failures"], 1):
            print(f"  {idx}. {f}")

    if report["warnings"]:
        print("\nADVERTENCIAS:")
        for idx, w in enumerate(report["warnings"], 1):
            print(f"  {idx}. {w}")

    verdict = "APPROVE" if len(report["failures"]) == 0 else "REQUEST_CHANGES"
    report["summary"] = {
        "verdict": verdict,
        "total_invariants": total_invariants,
        "passed_invariants": passed_invariants,
        "failed_invariants": total_invariants - passed_invariants,
    }

    print(f"\nDICTAMEN DE CHALLENGER M1-1: {verdict}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    report = run_all_checks()
    sys.exit(0 if report["summary"]["verdict"] == "APPROVE" else 1)
