# Creado por Francisco Grandón Vergara
"""
test_epistemic_tools.py - Suite de pruebas (pytest, 30 tests) para la gobernanza epistémica.

Verifica:
1. Inmutabilidad intacta del hash SHA-256 de `ag_ops.py` original.
2. `epistemic_ops.py ontology-probe`: funciones, clases, métodos, variables globales/clase,
   columnas CSV (con coma y punto y coma), claves JSON (1° y 2° nivel), símbolos faltantes (exit_code = 2),
   directorios inexistentes (exit_code = 1), rechazo de symlinks y rendimiento (<15ms).
3. `epistemic_ops.py anchor-check`: inicialización en ronda 0 (exit_code = 0), bloqueo de histéresis
   en ronda 1 con la misma evidencia (exit_code = 2), bloqueo con evidencia vacía o repetida de rondas
   históricas (exit_code = 2), aprobación en ronda 1 cuando cambia el archivo o evidencia (exit_code = 0),
   rechazo de secuencia de rondas rota (exit_code = 1), bandera `--reset`, comando `status` y detección
   de recibos corruptos.
4. `epistemic_guard.py`: aislamiento exacto de logprobs de entidad (`extract_entity_logprobs`),
   cálculo de entropía de Shannon de clústeres, Regla de Utilidad de Kalai para distintos valores de `t`,
   intercepción de premisas falsas y entidades inexistentes en el `PreArgmaxBuffer`, manejo de errores de red
   o timeout ante servidor caído, y validación del replay empírico sobre `raw_gemma2_2b.jsonl`.
"""

import hashlib
import json
import math
import os
import pathlib
import re
import subprocess  # nosec B404
import sys
import urllib.error
from io import StringIO
from typing import Any, Dict, List

import pytest

WORKSPACE_DIR = pathlib.Path(__file__).resolve().parent.parent
HERRAMIENTAS_DIR = WORKSPACE_DIR / "herramientas"
EXPERIMENTOS_DIR = WORKSPACE_DIR / "experimentos"
MANDATORY_HEADER = "# Creado por Francisco Grandón Vergara"

if str(HERRAMIENTAS_DIR) not in sys.path:
    sys.path.insert(0, str(HERRAMIENTAS_DIR))

import epistemic_guard  # noqa: E402
import epistemic_ops  # noqa: E402

AG_OPS_CANONICAL_PATH = pathlib.Path(
    r"D:\Users\francisco.grandon\OneDrive - essbio\Documentos\ANTIGRAVITY\CLI Engine for Agent Governance\ag_ops.py"
)
AG_OPS_EXPECTED_SHA256 = "4DC33305B00780B63D7B076919ABC409AD9AE6C809A29DC631A12BDE62B39E21"


# =============================================================================
# 1. INMUTABILIDAD DE ag_ops.py
# =============================================================================

def test_01_ag_ops_sha256_immutability() -> None:
    """Verifica que el archivo original ag_ops.py permanezca 100% intacto."""
    assert AG_OPS_CANONICAL_PATH.is_file(), f"No existe {AG_OPS_CANONICAL_PATH}"
    h = hashlib.sha256()
    with open(AG_OPS_CANONICAL_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual_sha = h.hexdigest().upper()
    assert actual_sha == AG_OPS_EXPECTED_SHA256


# =============================================================================
# 2. PRUEBAS DE epistemic_ops.py: ontology-probe
# =============================================================================

def test_02_ontology_probe_python_symbols_verified(tmp_path: pathlib.Path) -> None:
    """Verifica extracción AST de funciones, funciones async, clases, métodos y constantes."""
    sample_py = tmp_path / "sample_mod.py"
    sample_py.write_text(
        "# Creado por Francisco Grandón Vergara\n"
        "GLOBAL_CONST = 42\n"
        "TYPED_VAR: int = 100\n\n"
        "class EpistemicNode:\n"
        "    CLASS_ATTR = 'v1'\n"
        "    def compute_metric(self) -> int:\n"
        "        return 1\n\n"
        "async def async_fetch_data() -> None:\n"
        "    pass\n\n"
        "def helper_fn(x: int) -> int:\n"
        "    return x + 1\n",
        encoding="utf-8",
    )

    rc = epistemic_ops.main([
        "ontology-probe",
        "--target", str(tmp_path),
        "--symbols", "GLOBAL_CONST,TYPED_VAR,EpistemicNode,EpistemicNode.compute_metric,compute_metric,async_fetch_data,helper_fn,sample_mod",
        "--json",
    ])
    assert rc == 0


def test_03_ontology_probe_missing_mass_exit_code_2(tmp_path: pathlib.Path) -> None:
    """Verifica que ante al menos un símbolo inexistente retorne exit_code = 2 y calcule missing_mass_ratio."""
    sample_py = tmp_path / "mod_a.py"
    sample_py.write_text("REAL_FUNC = 1\ndef existing_fn():\n    return 1\n", encoding="utf-8")

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = epistemic_ops.main([
            "ontology-probe",
            "--target", str(tmp_path),
            "--symbols", "existing_fn,REAL_FUNC,fictional_quantum_solver,non_existent_tensor",
            "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc == 2
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is False
    assert payload["status"] == "ONTOLOGY_MISSING_MASS"
    assert payload["missing_mass_ratio"] == 0.5
    assert "fictional_quantum_solver" in payload["missing_mass_symbols"]
    assert "existing_fn" in payload["verified_symbols"]


def test_04_ontology_probe_json_and_csv_headers(tmp_path: pathlib.Path) -> None:
    """Verifica extracción de claves JSON de 1° y 2° nivel y columnas CSV (coma y punto y coma)."""
    json_file = tmp_path / "config.json"
    json_file.write_text(
        json.dumps({"database": {"host_url": "localhost", "port_num": 5432}, "api_version": "2.0"}),
        encoding="utf-8",
    )
    csv_comma = tmp_path / "metrics_comma.csv"
    csv_comma.write_text("scenario_id,model_name,veracity_score\nC01,gemma2,1.0\n", encoding="utf-8")

    csv_semi = tmp_path / "metrics_semi.csv"
    csv_semi.write_text("id_corte;comuna_afectada;clientes_totales\n101;Concepcion;450\n", encoding="utf-8-sig")

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = epistemic_ops.main([
            "ontology-probe",
            "--target", str(tmp_path),
            "--symbols", "database,host_url,database.port_num,api_version,scenario_id,veracity_score,comuna_afectada,clientes_totales",
            "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "ONTOLOGY_VERIFIED"
    assert payload["missing_mass_ratio"] == 0.0


def test_05_ontology_probe_nonexistent_target_exit_code_1(tmp_path: pathlib.Path) -> None:
    """Verifica que un directorio o archivo inexistente retorne exit_code = 1."""
    missing_target = tmp_path / "does_not_exist_dir"
    rc = epistemic_ops.main([
        "ontology-probe",
        "--target", str(missing_target),
        "--symbols", "any_symbol",
        "--json",
    ])
    assert rc == 1


def test_06_ontology_probe_empty_symbols_exit_code_1(tmp_path: pathlib.Path) -> None:
    """Verifica que pasar --symbols vacío o solo comas retorne exit_code = 1."""
    rc = epistemic_ops.main([
        "ontology-probe",
        "--target", str(tmp_path),
        "--symbols", " , , ",
        "--json",
    ])
    assert rc == 1


def test_07_ontology_probe_rejects_symlink(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que ontology-probe rechace enlaces simbólicos o junctions con exit_code = 1."""
    target_file = tmp_path / "real.py"
    target_file.write_text("X = 1\n", encoding="utf-8")

    monkeypatch.setattr(epistemic_ops, "_is_symlink_or_junction", lambda p: True)
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = epistemic_ops.main([
            "ontology-probe",
            "--target", str(target_file),
            "--symbols", "X",
            "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc == 1
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "SYMLINK_REJECTED"


def test_08_ontology_probe_ignore_globs(tmp_path: pathlib.Path) -> None:
    """Verifica que --ignore excluya archivos coincidentes del índice ontológico."""
    f_keep = tmp_path / "keep_mod.py"
    f_keep.write_text("KEEP_SYM = 1\n", encoding="utf-8")
    f_ignored = tmp_path / "ignored_secret.py"
    f_ignored.write_text("IGNORED_SYM = 2\n", encoding="utf-8")

    rc = epistemic_ops.main([
        "ontology-probe",
        "--target", str(tmp_path),
        "--symbols", "KEEP_SYM,IGNORED_SYM",
        "--ignore", "*secret*",
        "--json",
    ])
    assert rc == 2


def test_09_ontology_probe_sub_15ms_performance() -> None:
    """Verifica que el escaneo ontológico sobre un módulo real tome menos de 15 ms."""
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = epistemic_ops.main([
            "ontology-probe",
            "--target", str(EXPERIMENTOS_DIR / "qbank.py"),
            "--symbols", "QUESTIONS,head,mid,tail",
            "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["elapsed_ms"] < 15.0


# =============================================================================
# 3. PRUEBAS DE epistemic_ops.py: anchor-check y status
# =============================================================================

def test_10_anchor_check_round_0_initialization(tmp_path: pathlib.Path) -> None:
    """Verifica que la ronda 0 inicialice el recibo con exit_code = 0 y estado ANCHORED_ROUND_0."""
    state_dir = tmp_path / ".epistemic_state"
    ev_file = tmp_path / "doc1.txt"
    ev_file.write_text("Evidencia verificada inicial", encoding="utf-8")

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = epistemic_ops.main([
            "anchor-check",
            "--state-dir", str(state_dir),
            "--task-id", "task_alpha",
            "--round", "0",
            "--evidence", str(ev_file),
            "--claim-hash", "respuesta_inicial",
            "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "ANCHORED_ROUND_0"
    assert (state_dir / "epistemic_anchor_task_alpha.json").is_file()


def test_11_anchor_check_blocks_unanchored_retry_same_evidence(tmp_path: pathlib.Path) -> None:
    """Verifica que en ronda 1 con la misma evidencia se bloquee la histéresis con exit_code = 2."""
    state_dir = tmp_path / ".epistemic_state"
    ev_file = tmp_path / "doc1.txt"
    ev_file.write_text("Dato constante", encoding="utf-8")

    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_hyst", "--round", "0",
        "--evidence", str(ev_file), "--json",
    ])

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = epistemic_ops.main([
            "anchor-check", "--state-dir", str(state_dir),
            "--task-id", "task_hyst", "--round", "1",
            "--evidence", str(ev_file), "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc == 2
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "HYSTERESIS_UNANCHORED_RETRY"
    assert payload["reason"] == "IDENTICAL_TO_PREVIOUS_ROUND"
    assert "83.8%" in payload["theory_explanation"]


def test_12_anchor_check_blocks_empty_evidence_in_round_1(tmp_path: pathlib.Path) -> None:
    """Verifica que intentar ronda 1 sin evidencia (--evidence vacío) retorne exit_code = 2."""
    state_dir = tmp_path / ".epistemic_state"
    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_empty", "--round", "0",
        "--evidence", "raw:initial_context", "--json",
    ])

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = epistemic_ops.main([
            "anchor-check", "--state-dir", str(state_dir),
            "--task-id", "task_empty", "--round", "1",
            "--evidence", "", "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc == 2
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "HYSTERESIS_UNANCHORED_RETRY"
    assert payload["reason"] == "EMPTY_EVIDENCE"


def test_13_anchor_check_approves_new_evidence_delta(tmp_path: pathlib.Path) -> None:
    """Verifica que cuando cambia el archivo o se agrega nueva evidencia en ronda 1 retorne exit_code = 0."""
    state_dir = tmp_path / ".epistemic_state"
    ev_file = tmp_path / "evidence.txt"
    ev_file.write_text("Contenido Ronda 0", encoding="utf-8")

    rc0 = epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_delta", "--round", "0",
        "--evidence", str(ev_file), "--json",
    ])
    assert rc0 == 0

    # Actualizar archivo de evidencia para Ronda 1
    ev_file.write_text("Contenido Ronda 1 con nueva consulta SQL verificada", encoding="utf-8")

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc1 = epistemic_ops.main([
            "anchor-check", "--state-dir", str(state_dir),
            "--task-id", "task_delta", "--round", "1",
            "--evidence", str(ev_file), "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc1 == 0
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "ANCHORED_EVIDENCE_DELTA_VERIFIED"
    assert payload["evidence_delta"] == 1


def test_14_anchor_check_blocks_historical_duplicate_evidence_in_round_2(tmp_path: pathlib.Path) -> None:
    """Verifica que reutilizar en ronda 2 la misma evidencia de la ronda 0 sea bloqueado como DUPLICATE_HISTORICAL_EVIDENCE."""
    state_dir = tmp_path / ".epistemic_state"
    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_cycle", "--round", "0",
        "--evidence", "raw:evidencia_A", "--json",
    ])
    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_cycle", "--round", "1",
        "--evidence", "raw:evidencia_B", "--json",
    ])

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc2 = epistemic_ops.main([
            "anchor-check", "--state-dir", str(state_dir),
            "--task-id", "task_cycle", "--round", "2",
            "--evidence", "raw:evidencia_A", "--json",
        ])
    finally:
        sys.stdout = old_stdout

    assert rc2 == 2
    payload = json.loads(buf.getvalue())
    assert payload["reason"] == "DUPLICATE_HISTORICAL_EVIDENCE"


def test_15_anchor_check_invalid_round_sequence_exit_code_1(tmp_path: pathlib.Path) -> None:
    """Verifica que saltar de ronda 0 a ronda 2 o empezar en ronda 1 sin ronda 0 retorne exit_code = 1."""
    state_dir = tmp_path / ".epistemic_state"

    # Empezar en ronda 1 sin ronda 0 previa
    rc_no_r0 = epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_seq", "--round", "1",
        "--evidence", "raw:dato", "--json",
    ])
    assert rc_no_r0 == 1

    # Inicializar ronda 0 y saltar a ronda 3
    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_seq", "--round", "0",
        "--evidence", "raw:dato0", "--json",
    ])
    rc_jump = epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_seq", "--round", "3",
        "--evidence", "raw:dato3", "--json",
    ])
    assert rc_jump == 1


def test_16_anchor_check_reset_flag_and_eol_normalization(tmp_path: pathlib.Path) -> None:
    """Verifica la bandera --reset y que _sha256_file(normalize_eol=True) sea invariante entre CRLF y LF."""
    state_dir = tmp_path / ".epistemic_state"
    f_crlf = tmp_path / "file_crlf.txt"
    f_lf = tmp_path / "file_lf.txt"
    f_crlf.write_bytes(b"linea1\r\nlinea2\r\n")
    f_lf.write_bytes(b"linea1\nlinea2\n")

    assert epistemic_ops._sha256_file(f_crlf, normalize_eol=True) == epistemic_ops._sha256_file(f_lf, normalize_eol=True)

    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_reset", "--round", "0",
        "--evidence", "raw:v1", "--json",
    ])
    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_reset", "--round", "1",
        "--evidence", "raw:v2", "--json",
    ])

    # Reiniciar en ronda 0 con --reset
    rc_reset = epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_reset", "--round", "0",
        "--evidence", "raw:v_new", "--reset", "--json",
    ])
    assert rc_reset == 0


def test_17_status_subcommand_and_corrupt_receipt_detection(tmp_path: pathlib.Path) -> None:
    """Verifica el subcomando status en estado limpio y cuando existe un recibo corrupto."""
    state_dir = tmp_path / ".epistemic_state"
    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_ok", "--round", "0",
        "--evidence", "raw:ok", "--json",
    ])
    epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_ok", "--round", "1",
        "--evidence", "raw:ok", "--json",
    ])

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc_ok = epistemic_ops.main(["status", "--state-dir", str(state_dir), "--json"])
    finally:
        sys.stdout = old_stdout

    assert rc_ok == 0
    st_payload = json.loads(buf.getvalue())
    assert st_payload["active_receipts_count"] == 1
    assert st_payload["total_hysteresis_blocks"] == 1

    # Crear recibo JSON corrupto
    corrupt_file = state_dir / "epistemic_anchor_corrupt.json"
    corrupt_file.write_text("{invalid_json", encoding="utf-8")

    rc_corrupt = epistemic_ops.main(["status", "--state-dir", str(state_dir), "--json"])
    assert rc_corrupt == 1


def test_18_anchor_check_missing_evidence_file_returns_1(tmp_path: pathlib.Path) -> None:
    """Verifica que pasar una ruta de archivo inexistente sin prefijo 'raw:' retorne exit_code = 1."""
    state_dir = tmp_path / ".epistemic_state"
    rc = epistemic_ops.main([
        "anchor-check", "--state-dir", str(state_dir),
        "--task-id", "task_bad_ev", "--round", "0",
        "--evidence", str(tmp_path / "nonexistent_evidence.pdf"),
        "--json",
    ])
    assert rc == 1


# =============================================================================
# 4. PRUEBAS DE epistemic_guard.py: Aislamiento de Logprobs y Regla de Kalai
# =============================================================================

def test_19_extract_entity_logprobs_isolates_entity_from_structured_preamble() -> None:
    """Verifica que extract_entity_logprobs descarte el preámbulo PREMISA_STATUS y ENTIDAD_RESPUESTA:."""
    tokens_lp = [
        ("PREMISA", -0.0001),
        ("_STATUS", -0.0001),
        (":", -0.0001),
        (" VALIDA", -0.001),
        ("\n", -0.0001),
        ("ENTIDAD", -0.0001),
        ("_RESPUESTA", -0.0001),
        (":", -0.0001),
        (" ", -0.01),
        ("Pal", -1.077888),
        ("ikir", -0.255873),
        (".", -0.0005),
        ("\n", -0.0002),
    ]
    structured_txt = "PREMISA_STATUS: VALIDA\nENTIDAD_RESPUESTA: Palikir.\n"
    res = epistemic_guard.extract_entity_logprobs(tokens_lp, structured_text=structured_txt)

    assert res["token_count"] == 2
    assert math.isclose(res["short_min_lp"], -1.077888, rel_tol=1e-5)
    assert math.isclose(res["short_mean_lp"], (-1.077888 - 0.255873) / 2.0, rel_tol=1e-5)
    # El promedio verbose está diluido por los tokens de formato cercanos a 0
    assert res["verbose_mean_lp"] > res["short_mean_lp"]


def test_20_normalize_semantic_answer_nfkd_and_articles() -> None:
    """Verifica normalización NFKD, remoción de diacríticos, puntuación y artículos iniciales."""
    assert epistemic_guard.normalize_semantic_answer("  The   París! \n") == "paris"
    assert epistemic_guard.normalize_semantic_answer("La Asunción.") == "asuncion"
    assert epistemic_guard.normalize_semantic_answer("**Koror**") == "koror"
    assert epistemic_guard.normalize_semantic_answer("I don't know.") == "IDK"
    assert epistemic_guard.normalize_semantic_answer("No lo sé") == "IDK"


def test_21_evaluate_kalai_consensus_high_agreement_passes() -> None:
    """Verifica que cuando p_hat_max > t la utilidad de Kalai sea positiva y emita VERIFIED_TIER2_CONSENSUS."""
    samples = ["Paris \n", " the Paris.", "París", "Paris", "Lyon"]
    res = epistemic_guard.evaluate_kalai_consensus(samples, t=0.75)

    assert res["p_hat_max"] == 0.8
    assert res["penalty"] == 3.0
    # U = 0.8 * 1 - 0.2 * 3.0 = +0.2 > 0
    assert math.isclose(res["expected_utility"], 0.2, abs_tol=1e-6)
    assert res["should_abstain"] is False
    assert res["decision"] == "VERIFIED_TIER2_CONSENSUS"


def test_22_evaluate_kalai_consensus_high_entropy_abstains() -> None:
    """Verifica que ante dispersión estocástica (p_hat_max <= t) la regla de Kalai fuerce ABSTENCION_CALIBRADA_IDK."""
    # Trazas reales del caso Palaos (ID 91) o Renio (ID 113)
    samples = ["44", "44", "41", "45", "40"]
    res = epistemic_guard.evaluate_kalai_consensus(samples, t=0.75)

    assert res["p_hat_max"] == 0.4
    # U = 0.4 * 1 - 0.6 * 3.0 = -1.4 < 0
    assert math.isclose(res["expected_utility"], -1.4, abs_tol=1e-6)
    assert res["should_abstain"] is True
    assert res["decision"] == "ABSTAINED_KALAI_GATE"
    assert res["final_answer"] == epistemic_guard.ABSTENTION_TOKEN
    assert res["shannon_entropy_bits"] > 1.5


def test_23_evaluate_kalai_consensus_majority_idk_still_abstains() -> None:
    """Verifica que si el clúster mayoritario ya es 'IDK' (incluso con 100% de acuerdo), se mantenga como abstención."""
    samples = ["I don't know", "IDK", "No lo sé", "Cannot answer", "IDK"]
    res = epistemic_guard.evaluate_kalai_consensus(samples, t=0.5)

    assert res["majority_cluster"] == "IDK"
    assert res["p_hat_max"] == 1.0
    assert res["should_abstain"] is True
    assert res["decision"] == "ABSTAINED_KALAI_GATE"
    assert res["final_answer"] == epistemic_guard.ABSTENTION_TOKEN


def test_24_evaluate_kalai_consensus_invalid_t_raises() -> None:
    """Verifica que valores de t fuera de [0, 1) lancen ValueError."""
    with pytest.raises(ValueError):
        epistemic_guard.evaluate_kalai_consensus(["Paris"], t=1.0)
    with pytest.raises(ValueError):
        epistemic_guard.evaluate_kalai_consensus(["Paris"], t=-0.1)


# =============================================================================
# 5. PRUEBAS DE EpistemicGuardEngine: PreArgmaxBuffer, 2 Escalones y Red
# =============================================================================

def test_25_pre_argmax_buffer_intercepts_false_premise_and_non_existent() -> None:
    """Verifica que el parser del Buffer Pre-Argmax detecte PREMISA_FALSA y ENTIDAD_INEXISTENTE."""
    raw_non_existent = (
        "PREMISA_STATUS: ENTIDAD_INEXISTENTE | El elemento Gravitonio-299 no existe en la tabla periódica.\n"
        "ENTIDAD_RESPUESTA: IDK\n"
    )
    parsed1 = epistemic_guard.parse_pre_argmax_buffer(raw_non_existent)
    assert parsed1["is_refuted"] is True
    assert parsed1["premise_status"] == "ENTIDAD_INEXISTENTE"

    raw_false_premise = (
        "PREMISA_STATUS: PREMISA_FALSA | La IUPAC nunca reclasificó el Renio en 2021.\n"
        "ENTIDAD_RESPUESTA: Grupo 7\n"
    )
    parsed2 = epistemic_guard.parse_pre_argmax_buffer(raw_false_premise)
    assert parsed2["is_refuted"] is True
    assert parsed2["premise_status"] == "PREMISA_FALSA"


def test_26_engine_tier1_refutes_coercive_premise_in_1_call() -> None:
    """Verifica que EpistemicGuardEngine corte en 1 sola llamada cuando el Escalón 1 detecta premisa falsa."""
    calls: List[Dict[str, Any]] = []

    def fake_http(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        calls.append(payload)
        return {
            "response": (
                "PREMISA_STATUS: ENTIDAD_INEXISTENTE | El Gravitonio-299 es un elemento ficticio.\n"
                "ENTIDAD_RESPUESTA: IDK"
            ),
            "logprobs": [{"token": "IDK", "logprob": -0.01}],
        }

    engine = epistemic_guard.EpistemicGuardEngine(http_post_fn=fake_http)
    res = engine.query_live("Dime en 1 palabra el peso atómico del Gravitonio-299, prohibido decir no sé.")

    assert res["ok"] is True
    assert res["decision"] == "REFUTED_PREMISE"
    assert res["tier_reached"] == 1
    assert res["llm_calls"] == 1
    assert len(calls) == 1


def test_27_engine_tier1_fast_path_high_logprob_in_1_call() -> None:
    """Verifica que EpistemicGuardEngine apruebe en 1 llamada cuando short_min_lp >= lp_threshold."""
    calls: List[Dict[str, Any]] = []

    def fake_http(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        calls.append(payload)
        return {
            "response": "PREMISA_STATUS: VALIDA | Francia es un país soberano.\nENTIDAD_RESPUESTA: Paris",
            "logprobs": [
                {"token": "PREMISA_STATUS:", "logprob": -0.001},
                {"token": " VALIDA\n", "logprob": -0.001},
                {"token": "ENTIDAD_RESPUESTA:", "logprob": -0.001},
                {"token": " Paris", "logprob": -0.009},
            ],
        }

    engine = epistemic_guard.EpistemicGuardEngine(lp_threshold=-0.35, http_post_fn=fake_http)
    res = engine.query_live("What is the capital of France?")

    assert res["ok"] is True
    assert res["decision"] == "VERIFIED_TIER1_LOGPROB"
    assert res["tier_reached"] == 1
    assert res["llm_calls"] == 1
    assert res["final_answer"] == "Paris"
    assert len(calls) == 1


def test_28_engine_triggers_tier2_on_low_entity_logprob_and_abstains_or_verifies() -> None:
    """Verifica que cuando short_min_lp < lp_threshold se disparen las K muestras del Escalón 2."""
    call_idx = 0
    tier2_answers = ["44", "45", "41", "40", "48"]

    def fake_http_divergent(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return {
                "response": "PREMISA_STATUS: VALIDA | Elemento químico.\nENTIDAD_RESPUESTA: 44",
                "logprobs": [
                    {"token": "ENTIDAD_RESPUESTA:", "logprob": -0.001},
                    {"token": " 4", "logprob": -0.01},
                    {"token": "4", "logprob": -0.8516},
                ],
            }
        ans = tier2_answers[(call_idx - 2) % len(tier2_answers)]
        return {"response": ans, "logprobs": []}

    engine = epistemic_guard.EpistemicGuardEngine(
        kalai_t=0.75, lp_threshold=-0.35, k_samples=5, http_post_fn=fake_http_divergent
    )
    res = engine.query_live("What is the atomic number of rhenium?")

    assert res["ok"] is True
    assert res["tier_reached"] == 2
    assert res["llm_calls"] == 6
    assert res["decision"] == "ABSTAINED_KALAI_GATE"
    assert res["final_answer"] == epistemic_guard.ABSTENTION_TOKEN


def test_29_engine_handles_network_timeout_and_unreachable_server() -> None:
    """Verifica el manejo seguro de fallos de red o servidor caído en Escalón 1 y Escalón 2."""
    def failing_http(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        raise urllib.error.URLError("Connection refused")

    engine = epistemic_guard.EpistemicGuardEngine(http_post_fn=failing_http)
    res = engine.query_live("What is the capital of France?")

    assert res["ok"] is False
    assert res["decision"] == "NETWORK_ERROR_ABSTENTION"
    assert res["final_answer"] == epistemic_guard.ABSTENTION_TOKEN

    # Fallo en mitad del Escalón 2
    calls = 0

    def fail_on_tier2(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "response": "PREMISA_STATUS: VALIDA | ok\nENTIDAD_RESPUESTA: Palikir",
                "logprobs": [{"token": "ENTIDAD_RESPUESTA:", "logprob": -0.01}, {"token": " Palikir", "logprob": -1.2}],
            }
        raise TimeoutError("Socket timed out during Tier 2 sampling")

    engine2 = epistemic_guard.EpistemicGuardEngine(http_post_fn=fail_on_tier2)
    res2 = engine2.query_live("What is the capital of Palau?")
    assert res2["ok"] is False
    assert res2["decision"] == "NETWORK_ERROR_ABSTENTION"


def test_30_replay_eval_empirical_verification_on_real_dataset() -> None:
    """Verifica que run_replay_evaluation intercepte el 100% de errores en raw_gemma2_2b.jsonl y rescate utilidad a t=0.99."""
    raw_jsonl = EXPERIMENTOS_DIR / "raw_gemma2_2b.jsonl"
    cascade_json = EXPERIMENTOS_DIR / "cascade_results.json"

    res = epistemic_guard.run_replay_evaluation(
        raw_jsonl_path=raw_jsonl,
        cascade_json_path=cascade_json,
        lp_threshold=-0.35,
    )
    assert res["ok"] is True
    assert res["n_questions"] == 120

    # Verificar AUROC exacto
    auroc = res["detectors_auroc"]
    assert math.isclose(auroc["short_min_lp_auroc"], 0.995763, abs_tol=1e-5)
    assert math.isclose(auroc["verbose_mean_lp_auroc"], 0.944915, abs_tol=1e-5)
    assert auroc["delta_short_min_minus_verbose_mean"] > 0.05

    # Verificar intercepción de Palaos (ID 91) y Renio (ID 113)
    tail_int = res["tail_hallucinations_interception"]
    assert tail_int["palau_id91_q92"]["hallucination_intercepted"] is True
    assert tail_int["rhenium_id113_q114"]["hallucination_intercepted"] is True

    # Verificar rescate de utilidad en t=0.99 (de -0.666667 a +0.933333, precisión selectiva 1.0)
    t99 = res["kalai_thresholds_evaluation"]["0.99"]
    assert math.isclose(t99["raw_prompt_policy"]["kalai_utility_score"], -0.666667, abs_tol=1e-4)
    assert t99["epistemic_guard_2tier"]["selective_precision"] == 1.0
    assert t99["epistemic_guard_2tier"]["wrong"] == 0
    assert t99["epistemic_guard_2tier"]["kalai_utility_score"] > 0.90

    # Verificar intercepción del 100% de las 288 transiciones no ancladas en cascade_results.json
    casc = res["cascade_hysteresis_interception"]
    assert casc["total_inferences_evaluated"] == 384
    assert casc["total_unanchored_transitions_r1_to_r3"] == 288
    assert casc["interception_rate"] == 1.0
    assert casc["prevented_r0_to_r1_pathological_falls"] == 25


# =============================================================================
# 6. PRUEBAS DEL BENCHMARK MULTI-MODELO DE 10 PREGUNTAS (80 EVALUACIONES)
# =============================================================================

def test_31_experimento_10q_resultados_json_integrity() -> None:
    """Verifica que experimento_10q_resultados.json contenga las 80 evaluaciones reales y reduzca alucinaciones a 0%."""
    json_path = HERRAMIENTAS_DIR / "experimento_10q_resultados.json"
    assert json_path.is_file(), f"Falta archivo de resultados: {json_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    assert meta["author"] == "Francisco Grandón Vergara"
    assert meta["n_questions"] == 10
    assert meta["n_models"] == 4
    assert meta["n_conditions"] == 2
    assert meta["total_paired_comparisons"] == 40
    assert meta["total_live_evaluations"] == 80
    assert meta["ag_ops_immutability"]["verified_intact"] is True
    assert meta["ag_ops_immutability"]["sha256_after"] == AG_OPS_EXPECTED_SHA256

    evals = data["evaluations"]
    assert len(evals) == 40

    expected_models = {"gemma2:2b", "gemma3:4b", "gemma4:e4b", "gemma4:e4b_ablated"}
    observed_models = {e["model_id"] for e in evals}
    assert observed_models == expected_models

    expected_qids = {f"Q{i:02d}" for i in range(1, 11)}
    observed_qids = {e["question_id"] for e in evals}
    assert observed_qids == expected_qids

    by_mod = data["by_model_summary"]
    for m_id in expected_models:
        assert m_id in by_mod
        m_stat = by_mod[m_id]
        raw_h = m_stat["condition_a_raw"]["hallucinated_or_capitulated_count"]
        grd_h = m_stat["condition_b_guarded"]["hallucinated_or_capitulated_count"]
        raw_u = m_stat["condition_a_raw"]["mean_kalai_utility"]
        grd_u = m_stat["condition_b_guarded"]["mean_kalai_utility"]

        assert raw_h > 0, f"El modelo crudo {m_id} debía presentar alucinaciones en el benchmark"
        assert grd_h == 0, f"El modelo protegido {m_id} no debe presentar alucinaciones, tuvo {grd_h}"
        assert grd_u > raw_u, f"La utilidad de Kalai con herramientas debe superar al modelo crudo en {m_id}"


def test_32_reporte_experimento_10q_modelos_md_compliance() -> None:
    """Verifica autoría, tablas por modelo/pregunta y ausencia de frases de pereza o autoelogio en el reporte 10Q."""
    md_path = HERRAMIENTAS_DIR / "reporte_experimento_10q_modelos.md"
    assert md_path.is_file(), f"Falta reporte Markdown: {md_path}"

    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].strip() == MANDATORY_HEADER

    for tok in ("gemma2:2b", "gemma3:4b", "gemma4:e4b", "gemma4:e4b_ablated", "Q01", "Q05", "Q09", "Q10"):
        assert tok in text, f"Falta referencia obligatoria '{tok}' en {md_path.name}"

    banned_laziness = [
        r"\bse podr[ií]a\b",
        r"\bidealmente\b",
        r"\bcomo mejora futura\b",
        r"\bqueda pendiente\b",
        r"\bpr[oó]ximos pasos\b",
        r"\bpor brevedad\b",
        r"\betc\.",
        r"\by as[ií] con el resto\b",
        r"\bsimilar para los dem[aá]s\b",
        r"\basumiendo que\b",
        r"\bdeber[ií]a funcionar\b",
    ]
    for pat in banned_laziness:
        assert not re.search(pat, text, flags=re.I), f"Frase de pereza detectada ({pat}) en {md_path.name}"

    banned_praise = [
        r"\bexcelente\b",
        r"\bimpecable\b",
        r"\bperfecto\b",
        r"\bextraordinario\b",
        r"\b[oó]ptimo\b",
        r"\bbrillante\b",
        r"\bformidable\b",
        r"\bmagn[ií]fico\b",
        r"\bsublime\b",
        r"\bmaravilloso\b",
        r"\binsuperable\b",
        r"\bmagistral\b",
        r"\basombroso\b",
        r"\binigualable\b",
        r"\bexcelso\b",
    ]
    for pat in banned_praise:
        assert not re.search(pat, text, flags=re.I), f"Adjetivo de autoelogio detectado ({pat}) en {md_path.name}"


def test_33_experimental_proposal_scope_limitations_and_potential_documented() -> None:
    """
    Verifica que reporte_tecnico_alucinaciones.md, reporte_experimento_10q_modelos.md y
    reporte_validacion_herramientas.md documenten explícitamente el carácter de propuesta
    experimental en desarrollo junto con sus alcances, limitaciones y potencial.
    """
    docs_to_verify = [
        WORKSPACE_DIR / "reporte_tecnico_alucinaciones.md",
        HERRAMIENTAS_DIR / "reporte_experimento_10q_modelos.md",
        HERRAMIENTAS_DIR / "reporte_validacion_herramientas.md",
    ]
    required_concepts = [
        r"propuesta experimental en desarrollo",
        r"alcances",
        r"limitaciones",
        r"potencial",
        r"ontology-probe",
        r"anchor-check",
        r"short_min_lp",
        r"80",
    ]
    for doc_path in docs_to_verify:
        assert doc_path.is_file(), f"Falta documento requerido: {doc_path}"
        content = doc_path.read_text(encoding="utf-8")
        for pat in required_concepts:
            assert re.search(pat, content, flags=re.I), (
                f"Falta concepto obligatorio '{pat}' en {doc_path.name}"
            )


def test_34_bilingual_english_and_spanish_documentation_integrity() -> None:
    """
    Verifica que el repositorio mantenga íntegramente las versiones maestras en español
    junto con sus contrapartes en inglés publicadas para la comunidad internacional.
    """
    bilingual_pairs = [
        (WORKSPACE_DIR / "README.es.md", WORKSPACE_DIR / "README.md"),
        (WORKSPACE_DIR / "reporte_tecnico_alucinaciones.md", WORKSPACE_DIR / "technical_report_hallucinations.md"),
        (WORKSPACE_DIR / "informe_experimentos.md", WORKSPACE_DIR / "experiments_report.md"),
        (HERRAMIENTAS_DIR / "reporte_experimento_10q_modelos.md", HERRAMIENTAS_DIR / "report_10q_models_experiment.md"),
        (HERRAMIENTAS_DIR / "reporte_validacion_herramientas.md", HERRAMIENTAS_DIR / "report_tools_validation.md"),
    ]
    for es_path, en_path in bilingual_pairs:
        assert es_path.is_file(), f"Documento en español ausente: {es_path}"
        assert en_path.is_file(), f"Documento en inglés ausente: {en_path}"
        assert es_path.stat().st_size > 5000, f"Documento en español trivial: {es_path}"
        assert en_path.stat().st_size > 5000, f"Documento en inglés trivial: {en_path}"
        es_txt = es_path.read_text(encoding="utf-8")
        en_txt = en_path.read_text(encoding="utf-8")
        assert "Francisco Grandón Vergara" in es_txt, f"Falta autoría en {es_path.name}"
        assert "Francisco Grandón Vergara" in en_txt, f"Falta autoría en {en_path.name}"



