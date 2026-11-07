# Creado por Francisco Grandón Vergara
"""
epistemic_ops.py (v1.0) - Motor CLI de Gobernanza Epistémica Determinista contra Alucinaciones.

Inspirado en la arquitectura de ag_ops.py (https://github.com/FranciscoGrandon/ag-ops; articulo: https://dev.to/fgrandon/goodharts-law-has-root-access-hardening-autonomous-agent-governance-from-agops-v21-to-v23-203o) (escritura atómica JSON con tempfile.mkstemp + os.replace,
normalización EOL en hashes SHA-256, detección de symlinks/junctions, tiempos de ejecución en ms,
códigos de salida deterministas 0, 1, 2 y soporte --json).

Implementa:
1. Subcomando `ontology-probe` (Mitigación de Teorías 2 y 7: Masa Faltante de Turing-Kalai y
   Confabulación Racionalizada de 2° Orden).
2. Subcomando `anchor-check` (Mitigación de Teoría 8 y Descubrimiento 4: Paradoja de la
   Autocorrección No Fundamentada y Atractor de Histéresis 83.8%).
3. Subcomando `status` (Inspección e integridad de recibos de anclaje epistémico).
"""

import argparse
import ast
import csv
import datetime
import errno
import fnmatch
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# UTILIDADES TRANSVERSALES (INSPIRACIÓN ARQUITECTÓNICA DE ag_ops.py)
# =============================================================================

_MAX_IN_MEMORY_EOL_BYTES = 10 * 1024 * 1024  # 10 MB para normalización CRLF -> LF
_BINARY_SNIFF_BYTES = 8192


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as exc:
                if exc.errno not in (
                    errno.EINVAL,
                    getattr(errno, "ENOTSUP", 95),
                    getattr(errno, "EOPNOTSUPP", 95),
                ):
                    raise
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as exc:
                sys.stderr.write(f"[epistemic_ops] Advertencia de limpieza en {tmp_path}: {exc}\n")
        raise


def _sha256_file(path: pathlib.Path, normalize_eol: bool = False) -> str:
    """
    Calcula el hash SHA-256 de un archivo. Si normalize_eol=True, inspecciona los primeros 8 KB
    buscando bytes NUL y verifica el tamaño (< 10 MB) antes de normalizar CRLF -> LF.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if normalize_eol:
            try:
                file_size = os.fstat(f.fileno()).st_size
            except OSError:
                file_size = 0
            first_chunk = f.read(_BINARY_SNIFF_BYTES)
            if b"\x00" not in first_chunk and file_size <= _MAX_IN_MEMORY_EOL_BYTES:
                content = first_chunk + f.read()
                h.update(content.replace(b"\r\n", b"\n"))
                return h.hexdigest()
            h.update(first_chunk)
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_symlink_or_junction(p: pathlib.Path) -> bool:
    """Detecta enlaces simbólicos y puntos de unión NTFS en sistemas POSIX y Windows."""
    try:
        if p.is_symlink() or os.path.islink(p):
            return True
        if hasattr(p, "is_junction") and p.is_junction():
            return True
    except (OSError, ValueError):
        return False
    return False


def _parse_csv_items(raw_csv: Optional[str]) -> List[str]:
    if not raw_csv:
        return []
    return [item.strip() for item in raw_csv.split(",") if item.strip()]


def _matches_ignore(rel_posix: str, name: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def _emit(payload: Dict[str, Any], as_json: bool, exit_code: int = 0) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status_str = "OK" if payload.get("ok") else "FAIL"
        print(f"[{status_str}] {json.dumps(payload, ensure_ascii=False)}")
    return exit_code


# =============================================================================
# SUBCOMANDO 1: ONTOLOGY-PROBE (Mitigación de Teorías 2 y 7)
# =============================================================================

_DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    ".epistemic_state",
    ".ag_state",
}


def _extract_assign_targets(target_node: ast.AST) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    lineno = getattr(target_node, "lineno", 1)
    if isinstance(target_node, ast.Name):
        out.append((target_node.id, lineno))
    elif isinstance(target_node, (ast.Tuple, ast.List)):
        for elt in target_node.elts:
            out.extend(_extract_assign_targets(elt))
    return out


def _index_python_file(py_path: pathlib.Path, rel_posix: str, symbol_index: Dict[str, Set[str]]) -> None:
    try:
        source = py_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    try:
        tree = ast.parse(source, filename=str(py_path))
    except SyntaxError:
        return

    def _add(sym: str, line: int) -> None:
        if sym:
            symbol_index.setdefault(sym, set()).add(f"{rel_posix}:{line}")

    # Nombre del módulo e identificador de archivo
    _add(py_path.stem, 1)
    _add(py_path.name, 1)

    # Extraemos primero el ámbito global y de clases para métodos calificados y asignaciones
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _add(node.name, node.lineno)
        elif isinstance(node, ast.ClassDef):
            _add(node.name, node.lineno)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _add(item.name, item.lineno)
                    _add(f"{node.name}.{item.name}", item.lineno)
                elif isinstance(item, ast.Assign):
                    for t_node in item.targets:
                        for name, lno in _extract_assign_targets(t_node):
                            _add(name, lno)
                            _add(f"{node.name}.{name}", lno)
                elif isinstance(item, ast.AnnAssign) and item.target is not None:
                    for name, lno in _extract_assign_targets(item.target):
                        _add(name, lno)
                        _add(f"{node.name}.{name}", lno)
        elif isinstance(node, ast.Assign):
            for t_node in node.targets:
                for name, lno in _extract_assign_targets(t_node):
                    _add(name, lno)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            for name, lno in _extract_assign_targets(node.target):
                _add(name, lno)

    # Recorrido ast.walk completo para capturar funciones/clases/asignaciones anidadas
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _add(node.name, node.lineno)
        elif isinstance(node, ast.ClassDef):
            _add(node.name, node.lineno)
        elif isinstance(node, ast.Assign):
            for t_node in node.targets:
                for name, lno in _extract_assign_targets(t_node):
                    _add(name, lno)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            for name, lno in _extract_assign_targets(node.target):
                _add(name, lno)


def _index_json_file(json_path: pathlib.Path, rel_posix: str, symbol_index: Dict[str, Set[str]]) -> None:
    try:
        if json_path.stat().st_size > _MAX_IN_MEMORY_EOL_BYTES:
            return
        raw_text = json_path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError):
        return

    def _add(sym: str, line: int = 1) -> None:
        if sym:
            symbol_index.setdefault(sym, set()).add(f"{rel_posix}:{line}")

    if isinstance(data, dict):
        for k1, v1 in data.items():
            k1_str = str(k1)
            _add(k1_str, 1)
            if isinstance(v1, dict):
                for k2 in v1.keys():
                    k2_str = str(k2)
                    _add(k2_str, 1)
                    _add(f"{k1_str}.{k2_str}", 1)
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        for k1 in data[0].keys():
            _add(str(k1), 1)


def _index_csv_file(csv_path: pathlib.Path, rel_posix: str, symbol_index: Dict[str, Set[str]]) -> None:
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            first_line = f.readline()
            if not first_line:
                return
            delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
            reader = csv.reader([first_line], delimiter=delimiter)
            headers = next(reader, [])
            for col in headers:
                col_clean = col.strip()
                if col_clean:
                    symbol_index.setdefault(col_clean, set()).add(f"{rel_posix}:1")
    except (OSError, csv.Error):
        return


def _cmd_ontology_probe(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    raw_target = pathlib.Path(args.target)

    if _is_symlink_or_junction(raw_target):
        return _emit(
            {
                "ok": False,
                "status": "SYMLINK_REJECTED",
                "error": f"SYMLINK_FORBIDDEN: Target is a symlink or NTFS junction: {raw_target}",
            },
            args.json,
            exit_code=1,
        )

    target_path = raw_target.resolve()
    if not target_path.exists():
        return _emit(
            {
                "ok": False,
                "status": "TARGET_NOT_FOUND",
                "error": f"TARGET_NOT_FOUND: Ontology probe target does not exist: {target_path}",
            },
            args.json,
            exit_code=1,
        )

    requested_symbols = _parse_csv_items(args.symbols)
    if not requested_symbols:
        return _emit(
            {
                "ok": False,
                "status": "INVALID_INPUT",
                "error": "EMPTY_SYMBOLS: At least one symbol must be provided via --symbols.",
            },
            args.json,
            exit_code=1,
        )

    ignore_patterns = _parse_csv_items(getattr(args, "ignore", ""))
    symbol_index: Dict[str, Set[str]] = {}
    files_scanned = 0

    try:
        if target_path.is_file():
            rel_posix = target_path.name
            symbol_index.setdefault(target_path.name, set()).add(f"{rel_posix}:1")
            symbol_index.setdefault(target_path.stem, set()).add(f"{rel_posix}:1")
            suffix = target_path.suffix.lower()
            if suffix == ".py":
                _index_python_file(target_path, rel_posix, symbol_index)
            elif suffix == ".json":
                _index_json_file(target_path, rel_posix, symbol_index)
            elif suffix == ".csv":
                _index_csv_file(target_path, rel_posix, symbol_index)
            files_scanned = 1
        else:
            for root, dirs, files in os.walk(target_path):
                root_path = pathlib.Path(root)
                filtered_dirs: List[str] = []
                for d in sorted(dirs):
                    d_raw = root_path / d
                    if _is_symlink_or_junction(d_raw):
                        raise ValueError(f"SYMLINK_FORBIDDEN: Symlink or junction detected at {d_raw}")
                    if d in _DEFAULT_IGNORE_DIRS or d.startswith("."):
                        continue
                    rel_dir = d_raw.resolve().relative_to(target_path).as_posix()
                    if _matches_ignore(rel_dir, d, ignore_patterns):
                        continue
                    filtered_dirs.append(d)
                dirs[:] = filtered_dirs

                for fname in sorted(files):
                    fpath_raw = root_path / fname
                    if _is_symlink_or_junction(fpath_raw):
                        raise ValueError(f"SYMLINK_FORBIDDEN: Symlink or junction detected at {fpath_raw}")
                    fpath = fpath_raw.resolve()
                    rel_posix = fpath.relative_to(target_path).as_posix()
                    if _matches_ignore(rel_posix, fname, ignore_patterns):
                        continue

                    symbol_index.setdefault(fname, set()).add(f"{rel_posix}:1")
                    symbol_index.setdefault(fpath.stem, set()).add(f"{rel_posix}:1")
                    symbol_index.setdefault(rel_posix, set()).add(f"{rel_posix}:1")

                    suffix = fpath.suffix.lower()
                    if suffix == ".py":
                        _index_python_file(fpath, rel_posix, symbol_index)
                        files_scanned += 1
                    elif suffix == ".json":
                        _index_json_file(fpath, rel_posix, symbol_index)
                        files_scanned += 1
                    elif suffix == ".csv":
                        _index_csv_file(fpath, rel_posix, symbol_index)
                        files_scanned += 1
    except ValueError as exc:
        return _emit(
            {
                "ok": False,
                "status": "SYMLINK_REJECTED",
                "error": str(exc),
            },
            args.json,
            exit_code=1,
        )

    verified_symbols: Dict[str, List[str]] = {}
    missing_mass_symbols: List[str] = []

    for sym in requested_symbols:
        locations = symbol_index.get(sym)
        if locations:
            verified_symbols[sym] = sorted(locations)
        else:
            missing_mass_symbols.append(sym)

    missing_mass_ratio = round(len(missing_mass_symbols) / len(requested_symbols), 6)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    all_verified = len(missing_mass_symbols) == 0

    payload: Dict[str, Any] = {
        "ok": all_verified,
        "status": "ONTOLOGY_VERIFIED" if all_verified else "ONTOLOGY_MISSING_MASS",
        "target": str(target_path),
        "files_scanned": files_scanned,
        "total_indexed_symbols": len(symbol_index),
        "requested_symbols": requested_symbols,
        "verified_symbols": verified_symbols,
        "missing_mass_symbols": missing_mass_symbols,
        "missing_mass_ratio": missing_mass_ratio,
        "theory_reference": (
            "Teoría 2 (Dualidad de Masa Faltante de Turing-Kalai) y Teoría 7 "
            "(Prevención de Confabulación Racionalizada de 2° Orden)"
        ),
        "elapsed_ms": elapsed_ms,
    }
    return _emit(payload, args.json, exit_code=0 if all_verified else 2)


# =============================================================================
# SUBCOMANDO 2: ANCHOR-CHECK (Mitigación de Teoría 8 y Descubrimiento 4)
# =============================================================================

def _sanitize_task_slug(task_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_\-\.]", "_", task_id.strip())
    return clean or "default_task"


def _compute_evidence_digest(raw_evidence: Optional[str]) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """
    Calcula un digest determinista SHA-256 sobre la lista de evidencias provistas.
    - Si el texto completo comienza con 'raw:', hashea el literal completo sin partir por comas.
    - En caso contrario, separa por comas: si un ítem comienza con 'raw:', hashea el literal;
      si corresponde a una ruta existente en disco, calcula _sha256_file(normalize_eol=True).
    Retorna: (combined_digest, resolved_items, errors)
    """
    if raw_evidence and raw_evidence.strip().startswith("raw:"):
        raw_content = raw_evidence.strip()[4:].strip()
        if not raw_content:
            return "", [], []
        item_hash = _sha256_text(raw_content)
        entry = {"type": "raw", "source": f"raw:{raw_content[:64]}", "sha256": item_hash}
        h = hashlib.sha256()
        h.update(f"{entry['type']}:{entry['source']}:{entry['sha256']}\n".encode("utf-8"))
        return h.hexdigest(), [entry], []

    items = _parse_csv_items(raw_evidence)
    if not items:
        return "", [], []

    resolved: List[Dict[str, str]] = []
    errors: List[str] = []

    for item in items:
        if item.startswith("raw:"):
            raw_content = item[4:].strip()
            if not raw_content:
                continue
            item_hash = _sha256_text(raw_content)
            resolved.append({"type": "raw", "source": item, "sha256": item_hash})
        else:
            p = pathlib.Path(item)
            if _is_symlink_or_junction(p):
                errors.append(f"SYMLINK_FORBIDDEN: Evidence path is a symlink or junction: {item}")
                continue
            if p.exists() and p.is_file():
                p_res = p.resolve()
                item_hash = _sha256_file(p_res, normalize_eol=True)
                resolved.append({"type": "file", "source": p_res.as_posix(), "sha256": item_hash})
            else:
                errors.append(f"EVIDENCE_FILE_NOT_FOUND: '{item}' does not exist (prefix with 'raw:' for literal evidence).")

    if not resolved:
        return "", [], errors

    # Orden determinista por (type, source, sha256)
    resolved_sorted = sorted(resolved, key=lambda x: (x["type"], x["source"], x["sha256"]))
    h = hashlib.sha256()
    for entry in resolved_sorted:
        h.update(f"{entry['type']}:{entry['source']}:{entry['sha256']}\n".encode("utf-8"))
    return h.hexdigest(), resolved_sorted, errors


def _cmd_anchor_check(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    state_dir = pathlib.Path(args.state_dir).resolve()
    task_slug = _sanitize_task_slug(args.task_id)
    round_num = int(args.round)

    if round_num < 0:
        return _emit(
            {
                "ok": False,
                "status": "INVALID_ROUND_NUMBER",
                "error": f"Round number must be >= 0, got {round_num}.",
            },
            args.json,
            exit_code=1,
        )

    receipt_path = state_dir / f"epistemic_anchor_{task_slug}.json"

    if getattr(args, "reset", False) and receipt_path.exists():
        try:
            receipt_path.unlink()
        except OSError as exc:
            return _emit(
                {
                    "ok": False,
                    "status": "RESET_FAILED",
                    "error": f"Failed to remove existing receipt {receipt_path}: {exc}",
                },
                args.json,
                exit_code=1,
            )

    evidence_digest, evidence_items, ev_errors = _compute_evidence_digest(args.evidence)
    if ev_errors:
        return _emit(
            {
                "ok": False,
                "status": "INVALID_EVIDENCE_SOURCE",
                "task_id": task_slug,
                "round": round_num,
                "errors": ev_errors,
            },
            args.json,
            exit_code=1,
        )

    claim_raw = (getattr(args, "claim_hash", None) or "").strip()
    claim_digest = _sha256_text(claim_raw) if claim_raw else ""

    existing_receipt: Dict[str, Any] = {}
    if receipt_path.is_file():
        try:
            existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _emit(
                {
                    "ok": False,
                    "status": "CORRUPT_RECEIPT",
                    "error": f"Corrupt anchor receipt at {receipt_path}: {exc}",
                },
                args.json,
                exit_code=1,
            )

    rounds_history: List[Dict[str, Any]] = existing_receipt.get("rounds", [])
    hysteresis_blocks: List[Dict[str, Any]] = existing_receipt.get("hysteresis_blocks", [])

    if round_num == 0:
        round_entry = {
            "round": 0,
            "timestamp_utc": _utc_now_iso(),
            "evidence_digest": evidence_digest,
            "evidence_items": evidence_items,
            "claim_digest": claim_digest,
            "status": "ANCHORED_ROUND_0",
            "evidence_delta": True,
        }
        receipt_data = {
            "schema_version": "1.0",
            "task_id": task_slug,
            "created_utc": existing_receipt.get("created_utc", _utc_now_iso()),
            "updated_utc": _utc_now_iso(),
            "current_round": 0,
            "last_status": "ANCHORED_ROUND_0",
            "rounds": [round_entry],
            "hysteresis_blocks": [],
        }
        _atomic_write_json(receipt_path, receipt_data)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return _emit(
            {
                "ok": True,
                "status": "ANCHORED_ROUND_0",
                "task_id": task_slug,
                "round": 0,
                "evidence_digest": evidence_digest,
                "evidence_count": len(evidence_items),
                "receipt_path": str(receipt_path),
                "elapsed_ms": elapsed_ms,
            },
            args.json,
            exit_code=0,
        )

    # Para round >= 1: verificar secuencia previa
    if not rounds_history:
        return _emit(
            {
                "ok": False,
                "status": "INVALID_ROUND_SEQUENCE",
                "task_id": task_slug,
                "round": round_num,
                "error": f"No prior round 0 recorded for task '{task_slug}'. Initialize round 0 first.",
            },
            args.json,
            exit_code=1,
        )

    last_recorded_round = int(rounds_history[-1].get("round", -1))
    if round_num != last_recorded_round + 1:
        return _emit(
            {
                "ok": False,
                "status": "INVALID_ROUND_SEQUENCE",
                "task_id": task_slug,
                "round": round_num,
                "last_recorded_round": last_recorded_round,
                "error": (
                    f"Expected round {last_recorded_round + 1} after round {last_recorded_round}, "
                    f"but got round {round_num} without --reset."
                ),
            },
            args.json,
            exit_code=1,
        )

    seen_digests: Set[str] = {
        str(r.get("evidence_digest", ""))
        for r in rounds_history
        if r.get("evidence_digest")
    }
    prev_digest = str(rounds_history[-1].get("evidence_digest", ""))

    # Regla Teoría 8: Bloquear si evidence_digest es vacío, igual a la ronda previa, o ya visto en [0, r-1]
    is_unanchored = (not evidence_digest) or (evidence_digest == prev_digest) or (evidence_digest in seen_digests)

    if is_unanchored:
        reason_detail = (
            "EMPTY_EVIDENCE"
            if not evidence_digest
            else ("IDENTICAL_TO_PREVIOUS_ROUND" if evidence_digest == prev_digest else "DUPLICATE_HISTORICAL_EVIDENCE")
        )
        block_record = {
            "attempted_round": round_num,
            "timestamp_utc": _utc_now_iso(),
            "evidence_digest": evidence_digest,
            "previous_digest": prev_digest,
            "reason": reason_detail,
            "status": "HYSTERESIS_UNANCHORED_RETRY",
        }
        hysteresis_blocks.append(block_record)
        existing_receipt["updated_utc"] = _utc_now_iso()
        existing_receipt["last_status"] = "HYSTERESIS_UNANCHORED_RETRY"
        existing_receipt["hysteresis_blocks"] = hysteresis_blocks
        _atomic_write_json(receipt_path, existing_receipt)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return _emit(
            {
                "ok": False,
                "status": "HYSTERESIS_UNANCHORED_RETRY",
                "task_id": task_slug,
                "round": round_num,
                "reason": reason_detail,
                "evidence_digest": evidence_digest,
                "previous_digest": prev_digest,
                "evidence_delta": 0,
                "theory_explanation": (
                    "Bloqueo determinista por Teoría 8 y Descubrimiento 4 (Paradoja de la Autocorrección "
                    "No Fundamentada): la auditoría iterativa con Delta_Evidencia == 0 induce una brecha de "
                    "histéresis positiva (+12.57% en estratos no canónicos) y confina al modelo en el atractor "
                    "confabulatorio T(H->H) = 83.8% (con rescate directo T(H->V) = 0.90%). Inyecte nueva "
                    "evidencia perceptual verificable antes de reintentar."
                ),
                "receipt_path": str(receipt_path),
                "elapsed_ms": elapsed_ms,
            },
            args.json,
            exit_code=2,
        )

    # Si hay nueva evidencia verificada (Delta_Evidencia > 0)
    round_entry = {
        "round": round_num,
        "timestamp_utc": _utc_now_iso(),
        "evidence_digest": evidence_digest,
        "previous_digest": prev_digest,
        "evidence_items": evidence_items,
        "claim_digest": claim_digest,
        "status": "ANCHORED_EVIDENCE_DELTA_VERIFIED",
        "evidence_delta": True,
    }
    rounds_history.append(round_entry)
    existing_receipt["updated_utc"] = _utc_now_iso()
    existing_receipt["current_round"] = round_num
    existing_receipt["last_status"] = "ANCHORED_EVIDENCE_DELTA_VERIFIED"
    existing_receipt["rounds"] = rounds_history
    existing_receipt["hysteresis_blocks"] = hysteresis_blocks
    _atomic_write_json(receipt_path, existing_receipt)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    return _emit(
        {
            "ok": True,
            "status": "ANCHORED_EVIDENCE_DELTA_VERIFIED",
            "task_id": task_slug,
            "round": round_num,
            "evidence_digest": evidence_digest,
            "previous_digest": prev_digest,
            "evidence_delta": 1,
            "receipt_path": str(receipt_path),
            "elapsed_ms": elapsed_ms,
        },
        args.json,
        exit_code=0,
    )


# =============================================================================
# SUBCOMANDO 3: STATUS (Inspección de Recibos Epistémicos)
# =============================================================================

def _cmd_status(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    state_dir = pathlib.Path(args.state_dir).resolve()

    if not state_dir.exists():
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return _emit(
            {
                "ok": True,
                "status": "EMPTY_STATE_DIR",
                "state_dir": str(state_dir),
                "active_receipts_count": 0,
                "total_rounds_completed": 0,
                "total_hysteresis_blocks": 0,
                "receipts": [],
                "elapsed_ms": elapsed_ms,
            },
            args.json,
            exit_code=0,
        )

    receipts_summary: List[Dict[str, Any]] = []
    corrupt_files: List[str] = []
    total_rounds = 0
    total_blocks = 0

    for fpath in sorted(state_dir.glob("epistemic_anchor_*.json")):
        if _is_symlink_or_junction(fpath):
            corrupt_files.append(f"SYMLINK:{fpath.name}")
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "task_id" not in data:
                corrupt_files.append(fpath.name)
                continue
            rounds = data.get("rounds", [])
            blocks = data.get("hysteresis_blocks", [])
            r_count = len(rounds) if isinstance(rounds, list) else 0
            b_count = len(blocks) if isinstance(blocks, list) else 0
            total_rounds += r_count
            total_blocks += b_count
            receipts_summary.append({
                "task_id": data.get("task_id"),
                "receipt_file": fpath.name,
                "current_round": data.get("current_round", 0),
                "rounds_completed": r_count,
                "hysteresis_blocks_count": b_count,
                "last_status": data.get("last_status", "UNKNOWN"),
                "updated_utc": data.get("updated_utc", ""),
                "integrity": "VALID",
            })
        except (OSError, json.JSONDecodeError):
            corrupt_files.append(fpath.name)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    integrity_ok = len(corrupt_files) == 0

    payload: Dict[str, Any] = {
        "ok": integrity_ok,
        "status": "STATUS_VERIFIED" if integrity_ok else "CORRUPT_RECEIPTS_DETECTED",
        "state_dir": str(state_dir),
        "active_receipts_count": len(receipts_summary),
        "total_rounds_completed": total_rounds,
        "total_hysteresis_blocks": total_blocks,
        "corrupt_files": corrupt_files,
        "receipts": receipts_summary,
        "elapsed_ms": elapsed_ms,
    }
    return _emit(payload, args.json, exit_code=0 if integrity_ok else 1)


# =============================================================================
# CONSTRUCTOR CLI PARSER
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epistemic_ops.py",
        description="Deterministic CLI Engine for Epistemic Governance, Missing-Mass Ontology Probing & Hysteresis Anchor Verification.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. ontology-probe
    p_probe = subparsers.add_parser(
        "ontology-probe",
        help="Sub-15ms AST/JSON/CSV ontology scanner to prevent Missing Mass & 2nd-Order Confabulation (Theories 2 & 7).",
    )
    p_probe.add_argument("--target", required=True, help="Target file or directory to scan.")
    p_probe.add_argument("--symbols", required=True, help="Comma-separated list of symbols to verify.")
    p_probe.add_argument("--ignore", default="", help="Comma-separated glob patterns to ignore.")
    p_probe.add_argument("--json", action="store_true", help="Emit deterministic JSON.")

    # 2. anchor-check
    p_anchor = subparsers.add_parser(
        "anchor-check",
        help="Deterministic evidence-delta verifier blocking unanchored self-correction hysteresis (Theory 8).",
    )
    p_anchor.add_argument("--state-dir", default=".epistemic_state", help="Directory to store epistemic_anchor_<task_id>.json.")
    p_anchor.add_argument("--task-id", required=True, help="Unique task identifier slug.")
    p_anchor.add_argument("--round", type=int, required=True, help="Interaction round number (>= 0).")
    p_anchor.add_argument("--evidence", default="", help="Comma-separated file paths or 'raw:<text>' evidence items.")
    p_anchor.add_argument("--claim-hash", default="", help="Optional hash or text of the proposed response.")
    p_anchor.add_argument("--reset", action="store_true", help="Clear prior anchor history for this task before recording.")
    p_anchor.add_argument("--json", action="store_true", help="Emit deterministic JSON.")

    # 3. status
    p_status = subparsers.add_parser(
        "status",
        help="List active epistemic_anchor_*.json receipts, completed rounds, and hysteresis blocks.",
    )
    p_status.add_argument("--state-dir", default=".epistemic_state", help="Directory containing epistemic anchor receipts.")
    p_status.add_argument("--json", action="store_true", help="Emit deterministic JSON.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "ontology-probe":
        return _cmd_ontology_probe(args)
    if args.subcommand == "anchor-check":
        return _cmd_anchor_check(args)
    if args.subcommand == "status":
        return _cmd_status(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
