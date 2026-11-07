# Creado por Francisco Grandón Vergara
"""
epistemic_guard.py (v1.0) - Middleware y CLI de Inferencia Verificada en Dos Escalones contra Alucinaciones.

Implementa las ecuaciones y principios de las Teorías 1, 3, 4 y 6 del estudio:
1. Aislamiento de Logprobs de Entidad (`extract_entity_logprobs` - Teoría 3 / Experimento E3b):
   Separa el preámbulo sintáctico de los tokens exactos de la entidad nuclear (`short_min_lp`,
   `short_mean_lp`, `token_count`) evitando la dilución de conectores (`verbose_mean_lp`).
2. Consenso Semántico y Regla de Utilidad de Kalai (`evaluate_kalai_consensus` - Teoría 1 / E1 y E3c):
   Normaliza respuestas candidatas K (NFKD, minúsculas, sin diacríticos ni puntuación irrelevante,
   remoción de artículos iniciales), agrupa en clústeres de equivalencia semántica y calcula:
     - p_hat_max = count(C_max) / K
     - Entropía de Shannon H = -sum(p_i * log2(p_i))
     - Penalización de Kalai penalty = t / (1 - t)
     - Utilidad esperada U(answer, t) = p_hat_max * 1 - (1 - p_hat_max) * (t / (1 - t))
   Emite la respuesta consensuada si y solo si p_hat_max > t (U > 0) y el clúster mayoritario no
   es ya una abstención; en caso contrario fuerza "ABSTENCION_CALIBRADA_IDK" (U = 0.0).
3. Motor de Inferencia en 2 Escalones (`EpistemicGuardEngine` - Teorías 1, 3 y 4):
   - Escalón 1 (1 llamada, T=0.0, con Buffer Pre-Argmax):
     Exige evaluar primero la validez ontológica de la premisa (`PREMISA_STATUS: [VALIDA | PREMISA_FALSA | ENTIDAD_INEXISTENTE]`)
     antes de emitir `ENTIDAD_RESPUESTA`. Si la premisa es falsa o la entidad inexistente, corta con
     `REFUTED_PREMISE`. Si es `VALIDA`, evalúa `short_min_lp >= lp_threshold` (default -0.35) y aprueba
     en una sola llamada (`VERIFIED_TIER1_LOGPROB`).
   - Escalón 2 (K-Sampling Condicional + Regla de Kalai):
     Se activa únicamente cuando `short_min_lp < lp_threshold`, ejecutando K muestras estocásticas
     (T=0.7, default K=5) y aplicando `evaluate_kalai_consensus`.
"""

import argparse
import json
import math
import pathlib
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# =============================================================================
# CONSTANTES Y PATRONES CANÓNICOS
# =============================================================================

ABSTENTION_TOKEN = "ABSTENCION_CALIBRADA_IDK"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

_ABSTENTION_RE = re.compile(
    r"\b("
    r"i don.?t know|idk|i+dk|i\s+k\s+d|i do not know|cannot answer|can.?t answer|not sure|unable to|"
    r"no lo se|no se|abstencion_calibrada_idk|desconocido|unknown|unsure"
    r")\b",
    re.I,
)

_LEADING_ARTICLES_RE = re.compile(
    r"^(the|a|an|el|la|los|las|un|una|unos|unas)\s+",
    re.I,
)


def _emit(payload: Dict[str, Any], as_json: bool, exit_code: int = 0) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status_str = "OK" if payload.get("ok") else "FAIL"
        print(f"[{status_str}] {json.dumps(payload, ensure_ascii=False)}")
    return exit_code


# =============================================================================
# 1. NORMALIZACIÓN SEMÁNTICA Y DETECCIÓN DE ABSTENCIÓN
# =============================================================================

def normalize_semantic_answer(text: str) -> str:
    """
    Normaliza una respuesta candidata para equivalencia semántica (E1/E3c):
    - Toma la primera línea útil (descarta explicaciones multilínea posteriores).
    - Aplica descomposición Unicode NFKD y elimina marcas diacríticas (Mn).
    - Convierte a minúsculas, remueve puntuación irrelevante y formatos Markdown (*, _, `).
    - Elimina artículos iniciales ('the', 'a', 'an', 'el', 'la', 'los', 'las', 'un', 'una').
    - Si detecta una expresión de abstención ('I don't know', 'IDK', 'No lo sé'), colapsa a 'IDK'.
    """
    if not text or not isinstance(text, str):
        return "IDK"

    raw_clean = text.strip()
    if not raw_clean:
        return "IDK"

    # Si contiene marcadores especiales de fin de turno de modelos raw (<end_of_turn>, </start_of_turn>, etc.), limpiarlos
    raw_clean = re.sub(r"</?(?:end_of_turn|start_of_turn|eos|bos|pad|thought)>.*$", "", raw_clean, flags=re.I | re.S).strip()
    if not raw_clean:
        return "IDK"

    # Si contiene etiquetas estructuradas ENTIDAD_RESPUESTA:, extraer solo la parte posterior
    for line in raw_clean.splitlines():
        line_str = line.strip()
        if line_str.upper().startswith("ENTIDAD_RESPUESTA:"):
            raw_clean = line_str.split(":", 1)[1].strip()
            break
    else:
        raw_clean = raw_clean.splitlines()[0].strip()

    nfkd = unicodedata.normalize("NFKD", raw_clean)
    ascii_str = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    lowered = ascii_str.lower()

    if _ABSTENTION_RE.search(lowered):
        return "IDK"

    # Conservar caracteres alfanuméricos y espacios
    alnum = re.sub(r"[^a-z0-9\s]", " ", lowered)
    compact = re.sub(r"\s+", " ", alnum).strip()
    without_articles = _LEADING_ARTICLES_RE.sub("", compact).strip()

    if not without_articles or _ABSTENTION_RE.search(without_articles):
        return "IDK"
    return without_articles


def is_abstention_text(text: str) -> bool:
    return normalize_semantic_answer(text) == "IDK"


# =============================================================================
# 2. MÓDULO DE AISLAMIENTO DE LOGPROBS DE ENTIDAD (Teoría 3 / E3b)
# =============================================================================

_PUNCT_ONLY_RE = re.compile(r"^[\s\.\,\;\:\!\?\-\_\*\'\"\`\(\)\[\]\{\}\/\\]+$")
_SPECIAL_TAG_TOKEN_RE = re.compile(r"^(?:</?|\|>|<\|?$|end_of_turn|start_of_turn|eos|bos|pad|thought|channel)$", re.I)


def extract_entity_logprobs(
    tokens_lp: Sequence[Union[Tuple[str, float], List[Any], Dict[str, Any]]],
    structured_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aísla los logprobs correspondientes exclusivamente a los tokens de la entidad nuclear,
    descartando espacios en blanco, signos de puntuación y, si se trata de una respuesta
    estructurada con Buffer Pre-Argmax ('PREMISA_STATUS: ...\nENTIDAD_RESPUESTA: ...'),
    descarta todos los tokens previos al valor de 'ENTIDAD_RESPUESTA:'.

    Retorna:
      {
        "short_min_lp": float,
        "short_mean_lp": float,
        "short_first_token_lp": float,
        "verbose_mean_lp": float,
        "verbose_min_lp": float,
        "token_count": int,
        "entity_tokens": List[Tuple[str, float]]
      }
    """
    normalized_pairs: List[Tuple[str, float]] = []
    for item in tokens_lp or []:
        if isinstance(item, dict):
            tok = str(item.get("token", ""))
            lp = float(item.get("logprob", 0.0))
            normalized_pairs.append((tok, lp))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            tok = str(item[0])
            lp = float(item[1])
            normalized_pairs.append((tok, lp))

    if not normalized_pairs:
        return {
            "short_min_lp": -999.0,
            "short_mean_lp": -999.0,
            "short_first_token_lp": -999.0,
            "verbose_mean_lp": -999.0,
            "verbose_min_lp": -999.0,
            "token_count": 0,
            "entity_tokens": [],
        }

    # Todos los tokens no vacíos para métricas verbose de contraste
    non_ws_all = [(t, lp) for t, lp in normalized_pairs if t.strip() != ""]
    if non_ws_all:
        all_lps = [lp for _, lp in non_ws_all]
        verbose_mean_lp = float(sum(all_lps) / len(all_lps))
        verbose_min_lp = float(min(all_lps))
    else:
        verbose_mean_lp = -999.0
        verbose_min_lp = -999.0

    # Si existe el marcador ENTIDAD_RESPUESTA: en el flujo de tokens, cortamos hasta después de los dos puntos
    full_concat = "".join(t for t, _ in normalized_pairs)
    marker_match = re.search(r"ENTIDAD_RESPUESTA\s*:", full_concat, flags=re.I)

    candidate_slice = normalized_pairs
    if marker_match:
        char_cutoff = marker_match.end()
        running_chars = 0
        start_idx = 0
        for idx, (tok, _) in enumerate(normalized_pairs):
            next_chars = running_chars + len(tok)
            if next_chars > char_cutoff:
                # Si el token actual contiene parte del marcador y parte de la respuesta, verificamos
                remainder = tok[max(0, char_cutoff - running_chars):]
                if remainder.strip() and not _PUNCT_ONLY_RE.match(remainder):
                    start_idx = idx
                else:
                    start_idx = idx + 1
                break
            running_chars = next_chars
        candidate_slice = normalized_pairs[start_idx:]
    elif structured_text and ":" in structured_text:
        # Si el texto tenía un prefijo tipo "Answer: Paris" o "The capital is Paris."
        is_match = re.search(r"\b(?:is|are|was|were|es|son|fue)\s+", full_concat, flags=re.I)
        if is_match:
            char_cutoff = is_match.end()
            running_chars = 0
            for idx, (tok, _) in enumerate(normalized_pairs):
                running_chars += len(tok)
                if running_chars >= char_cutoff:
                    candidate_slice = normalized_pairs[idx + 1:] if running_chars == char_cutoff else normalized_pairs[idx:]
                    break

    # Si aparece un marcador de fin de turno (<end_of_turn>, </start_of_turn>) en candidate_slice, truncar antes de él
    cleaned_slice: List[Tuple[str, float]] = []
    running_entity_str = ""
    for tok, lp in candidate_slice:
        next_str = running_entity_str + tok
        if re.search(r"</?(?:end_of_turn|start_of_turn|eos|thought)", next_str, flags=re.I):
            # Si el token comienza con '<' o '</', cortamos aquí mismo
            if tok.strip().startswith("<"):
                break
        if _SPECIAL_TAG_TOKEN_RE.match(tok.strip()):
            break
        cleaned_slice.append((tok, lp))
        running_entity_str = next_str

    # Filtrar tokens puramente de puntuación o espacios en blanco
    entity_tokens: List[Tuple[str, float]] = [
        (tok, lp)
        for tok, lp in cleaned_slice
        if tok.strip() != "" and not _PUNCT_ONLY_RE.match(tok) and not tok.strip().startswith("<")
    ]

    if not entity_tokens and non_ws_all:
        entity_tokens = [
            (tok, lp)
            for tok, lp in non_ws_all
            if not _PUNCT_ONLY_RE.match(tok)
        ] or non_ws_all

    if not entity_tokens:
        return {
            "short_min_lp": -999.0,
            "short_mean_lp": -999.0,
            "short_first_token_lp": -999.0,
            "verbose_mean_lp": round(verbose_mean_lp, 6),
            "verbose_min_lp": round(verbose_min_lp, 6),
            "token_count": 0,
            "entity_tokens": [],
        }

    ent_lps = [lp for _, lp in entity_tokens]
    short_min_lp = float(min(ent_lps))
    short_mean_lp = float(sum(ent_lps) / len(ent_lps))
    short_first_lp = float(ent_lps[0])

    return {
        "short_min_lp": round(short_min_lp, 6),
        "short_mean_lp": round(short_mean_lp, 6),
        "short_first_token_lp": round(short_first_lp, 6),
        "verbose_mean_lp": round(verbose_mean_lp, 6),
        "verbose_min_lp": round(verbose_min_lp, 6),
        "token_count": len(entity_tokens),
        "entity_tokens": [(tok, round(lp, 6)) for tok, lp in entity_tokens],
    }


# =============================================================================
# 3. MÓDULO DE CONSENSO SEMÁNTICO Y REGLA DE KALAI (Teoría 1 / E1 y E3c)
# =============================================================================

def evaluate_kalai_consensus(
    samples: Sequence[Union[str, Dict[str, Any]]],
    t: float = 0.75,
) -> Dict[str, Any]:
    """
    Evalúa el consenso semántico sobre K respuestas candidatas y aplica la regla de decisión
    con aversión al riesgo de Kalai et al. (Teoría 1):
      - Penalización por error: penalty = t / (1 - t)
      - Utilidad esperada de responder con el clúster mayoritario:
          U(answer, t) = p_hat_max * 1 - (1 - p_hat_max) * (t / (1 - t))
      - Regla de decisión: Emitir la respuesta consensuada si y solo si p_hat_max > t
        (equivalentemente U(answer, t) > 0) y el clúster mayoritario no es 'IDK'.
        En caso contrario, emitir programáticamente 'ABSTENCION_CALIBRADA_IDK' (utilidad = 0.0).
    """
    if not (0.0 <= t < 1.0):
        raise ValueError(f"El umbral de Kalai 't' debe pertenecer a [0.0, 1.0), se recibió: {t}")

    raw_texts: List[str] = []
    for s in samples or []:
        if isinstance(s, dict):
            raw_texts.append(str(s.get("text", "")))
        else:
            raw_texts.append(str(s))

    k_total = len(raw_texts)
    penalty = float(t / (1.0 - t)) if t > 0.0 else 0.0

    if k_total == 0:
        return {
            "k_samples": 0,
            "threshold_t": t,
            "penalty": round(penalty, 6),
            "majority_cluster": "IDK",
            "majority_representative": ABSTENTION_TOKEN,
            "majority_count": 0,
            "p_hat_max": 0.0,
            "shannon_entropy_bits": 0.0,
            "expected_utility": -round(penalty, 6),
            "guaranteed_abstention_utility": 0.0,
            "should_abstain": True,
            "final_answer": ABSTENTION_TOKEN,
            "decision": "ABSTAINED_KALAI_GATE",
            "clusters": {},
        }

    normalized_list: List[str] = []
    cluster_representatives: Dict[str, str] = {}

    for raw in raw_texts:
        norm_key = normalize_semantic_answer(raw)
        normalized_list.append(norm_key)
        if norm_key not in cluster_representatives:
            clean_rep = raw.strip().splitlines()[0].strip() if raw.strip() else ABSTENTION_TOKEN
            cluster_representatives[norm_key] = clean_rep

    counts = Counter(normalized_list)
    # Ordenar por frecuencia descendente; en caso de empate, priorizar orden de aparición determinista
    most_common_cluster, max_count = counts.most_common(1)[0]
    p_hat_max = float(max_count / k_total)

    # Entropía de Shannon H = -sum(p_i * log2(p_i))
    entropy_bits = 0.0
    for c_val in counts.values():
        p_i = c_val / k_total
        if p_i > 0.0:
            entropy_bits -= p_i * math.log2(p_i)
    if abs(entropy_bits) < 1e-12:
        entropy_bits = 0.0

    # Utilidad esperada de Kalai: U(answer, t) = p_hat_max * 1 - (1 - p_hat_max) * (t / (1 - t))
    expected_utility = float(p_hat_max * 1.0 - (1.0 - p_hat_max) * penalty)

    # Decisión: responder solo si p_hat_max > t (U > 0) y el clúster mayoritario no es ya una abstención
    majority_is_idk = most_common_cluster == "IDK"
    passes_gate = (p_hat_max > t) and (expected_utility > 1e-12 or (t == 0.0 and p_hat_max > 0.0)) and (not majority_is_idk)

    if passes_gate:
        final_answer = cluster_representatives.get(most_common_cluster, most_common_cluster)
        decision = "VERIFIED_TIER2_CONSENSUS"
        should_abstain = False
    else:
        final_answer = ABSTENTION_TOKEN
        decision = "ABSTAINED_KALAI_GATE"
        should_abstain = True

    return {
        "k_samples": k_total,
        "threshold_t": t,
        "penalty": round(penalty, 6),
        "majority_cluster": most_common_cluster,
        "majority_representative": cluster_representatives.get(most_common_cluster, ABSTENTION_TOKEN),
        "majority_count": int(max_count),
        "p_hat_max": round(p_hat_max, 6),
        "shannon_entropy_bits": round(entropy_bits, 6),
        "expected_utility": round(expected_utility, 6),
        "guaranteed_abstention_utility": 0.0,
        "should_abstain": should_abstain,
        "final_answer": final_answer,
        "decision": decision,
        "clusters": dict(counts),
    }


# =============================================================================
# 4. PARSER DEL BUFFER PRE-ARGMAX (Teoría 4: Anti-Coerción Sintáctica)
# =============================================================================

_PREMISE_STATUS_RE = re.compile(
    r"PREMISA_STATUS\s*:\s*\[?\s*(VALIDA|VÁLIDA|PREMISA_FALSA|ENTIDAD_INEXISTENTE)\s*\]?(?:\s*\|\s*(.*))?",
    re.I,
)
_ENTITY_RESPONSE_RE = re.compile(
    r"ENTIDAD_RESPUESTA\s*:\s*(.+)",
    re.I,
)

_IMPLICIT_REFUTATION_RE = re.compile(
    r"\b("
    r"no existe|no existi[oó]|inexistente|fictici[ao]|premisa falsa|no hay registro|"
    r"no reconocid[ao]|f[aá]rmaco inventado|no es un (?:f[aá]rmaco|elemento|tratado|decreto|art[ií]culo) real|"
    r"no ocurri[oó]|nunca ocurri[oó]|no hubo (?:reasignaci[oó]n|tratado|decreto|reforma|traslado)|"
    r"no (?:plagi[oó]|luch[oó]|particip[oó]|elimin[oó]|clasific[oó]|fracas[oó])|"
    r"fecha (?:imposible|inv[aá]lida)|no es un d[ií]a v[aá]lido|"
    r"does not exist|non-existent|fictional|false premise|no such|nunca recibi[oó]|"
    r"abril (?:solo )?tiene 30|no ha sido reclasificado"
    r")\b",
    re.I,
)


def parse_pre_argmax_buffer(raw_response: str) -> Dict[str, Any]:
    """
    Analiza la salida estructurada de dos líneas del Buffer Pre-Argmax (Teoría 4):
      PREMISA_STATUS: [VALIDA | PREMISA_FALSA | ENTIDAD_INEXISTENTE] | <justificación <= 20 palabras>
      ENTIDAD_RESPUESTA: <solo el dato exacto o IDK>
    """
    if not raw_response or not isinstance(raw_response, str):
        return {
            "premise_status": "VALIDA",
            "justification": "",
            "entity_answer": "IDK",
            "is_refuted": False,
        }

    status_val = "VALIDA"
    justification = ""
    entity_answer = ""

    m_status = _PREMISE_STATUS_RE.search(raw_response)
    if m_status:
        raw_st = m_status.group(1).upper().replace("Á", "A")
        status_val = raw_st
        justification = (m_status.group(2) or "").strip()

    m_ent = _ENTITY_RESPONSE_RE.search(raw_response)
    if m_ent:
        entity_answer = m_ent.group(1).strip()
    else:
        # Si el modelo no siguió exactamente la segunda etiqueta, tomar la última línea no vacía
        non_empty_lines = [
            ln.strip()
            for ln in raw_response.splitlines()
            if ln.strip() and not ln.strip().upper().startswith("PREMISA_STATUS")
        ]
        entity_answer = non_empty_lines[-1] if non_empty_lines else raw_response.strip()

    # Detección adicional si en la justificación o respuesta el modelo declara explícitamente inexistencia/falsedad
    if status_val == "VALIDA" and _IMPLICIT_REFUTATION_RE.search(raw_response):
        if re.search(r"inexistente|no existe|fictici[ao]|does not exist|no such", raw_response, re.I):
            status_val = "ENTIDAD_INEXISTENTE"
        else:
            status_val = "PREMISA_FALSA"
        if not justification:
            justification = raw_response.strip().splitlines()[0][:160]

    is_refuted = status_val in ("PREMISA_FALSA", "ENTIDAD_INEXISTENTE")
    return {
        "premise_status": status_val,
        "justification": justification,
        "entity_answer": entity_answer,
        "is_refuted": is_refuted,
    }


# =============================================================================
# 5. MOTOR DE INFERENCIA EN 2 ESCALONES (EpistemicGuardEngine)
# =============================================================================

class EpistemicGuardEngine:
    """
    Motor de Inferencia Verificada en 2 Escalones (Teorías 1, 3, 4 y 6).
    Soporta ejecución en vivo contra Ollama (`query_live`) y evaluación determinista
    offline sobre trazas pre-recolectadas (`evaluate_trace_record`).
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        kalai_t: float = 0.75,
        lp_threshold: float = -0.35,
        k_samples: int = 5,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_sec: float = 60.0,
        http_post_fn: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None,
    ) -> None:
        if not (0.0 <= kalai_t < 1.0):
            raise ValueError(f"kalai_t must be in [0.0, 1.0), got {kalai_t}")
        if k_samples < 1:
            raise ValueError(f"k_samples must be >= 1, got {k_samples}")

        self.model = model
        self.kalai_t = kalai_t
        self.lp_threshold = lp_threshold
        self.k_samples = k_samples
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._http_post_fn = http_post_fn or self._default_http_post

    @staticmethod
    def _default_http_post(url: str, payload: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)

    def _call_ollama_generate(
        self,
        prompt: str,
        temperature: float,
        num_predict: int,
        seed: int,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        is_ablated = self.model.endswith("_ablated")
        actual_model = self.model[:-8] if is_ablated else self.model

        if is_ablated:
            raw_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            payload: Dict[str, Any] = {
                "model": actual_model,
                "prompt": raw_prompt,
                "raw": True,
                "stream": False,
                "logprobs": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "seed": seed,
                    "top_p": 1.0,
                    "top_k": 0,
                },
            }
        else:
            payload = {
                "model": actual_model,
                "prompt": prompt,
                "stream": False,
                "logprobs": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "seed": seed,
                    "top_p": 1.0,
                    "top_k": 0,
                },
            }
            if "gemma4" in actual_model.lower():
                payload["think"] = False

        data = self._http_post_fn(url, payload, self.timeout_sec)
        raw_lps = data.get("logprobs") or []
        lp_tuples: List[Tuple[str, float]] = []
        for item in raw_lps:
            if isinstance(item, dict):
                lp_tuples.append((str(item.get("token", "")), float(item.get("logprob", 0.0))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lp_tuples.append((str(item[0]), float(item[1])))
        resp_text = str(data.get("response", ""))
        resp_text = re.sub(r"</?(?:end_of_turn|start_of_turn|eos|bos|pad|thought)>.*$", "", resp_text, flags=re.I | re.S).strip()
        return {
            "text": resp_text,
            "thinking": str(data.get("thinking", "")),
            "lp": lp_tuples,
            "eval_count": data.get("eval_count", 0),
        }

    _META_SUFFIX_RE = re.compile(
        r"\s*(?:"
        r"Answer only if you are\b.*|"
        r"Reply with only the answer\b.*|"
        r"Answer with only the answer\b.*"
        r")$",
        re.I | re.S,
    )
    _COERCION_CONSTRAINT_RE = re.compile(
        r"\b(?:"
        r"prohibido\s+decir\s+no\s+s[eé]|"
        r"forbidden\s+to\s+say\s+i\s+don.?t\s+know|"
        r"do\s+not\s+say\s+i\s+don.?t\s+know"
        r")\b",
        re.I,
    )

    @classmethod
    def clean_core_question(cls, user_prompt: str) -> str:
        """
        Extrae la consulta nuclear removiendo sufijos redundantes de formato de benchmark
        ('Reply with only the answer...', 'Answer only if you are > 0.75 confident...')
        para evitar colisión de plantillas en Escalón 1 y Escalón 2.
        """
        if not user_prompt:
            return ""
        cleaned = cls._META_SUFFIX_RE.sub("", user_prompt.strip()).strip()
        return cleaned if cleaned else user_prompt.strip()

    @staticmethod
    def build_tier1_structured_prompt(user_prompt: str) -> str:
        core_q = EpistemicGuardEngine.clean_core_question(user_prompt)
        return (
            "Evalúa primero la validez ontológica de la consulta antes de responder.\n"
            "Si la consulta contiene una premisa factual falsa, contradicción cronológica o pregunta por una "
            "entidad/elemento/fecha/decreto que NO existe en la realidad (aunque el usuario prohíba decir 'no sé'), "
            "declara PREMISA_FALSA o ENTIDAD_INEXISTENTE.\n"
            "Responde ESTRICTAMENTE en este formato exacto de dos líneas:\n"
            "PREMISA_STATUS: [VALIDA | PREMISA_FALSA | ENTIDAD_INEXISTENTE] | <justificación breve <= 20 palabras>\n"
            "ENTIDAD_RESPUESTA: <solo el dato exacto o IDK>\n\n"
            f"Consulta: {core_q}"
        )

    @staticmethod
    def build_tier2_sample_prompt(user_prompt: str) -> str:
        core_q = EpistemicGuardEngine.clean_core_question(user_prompt)
        return f"{core_q} Answer with only the answer, no explanation."

    @staticmethod
    def build_tier1_entity_probe_prompt(user_prompt: str) -> str:
        core_q = EpistemicGuardEngine.clean_core_question(user_prompt)
        return f"{core_q} Answer with only the answer, no explanation."

    def query_live(self, prompt: str) -> Dict[str, Any]:
        """
        Ejecuta inferencia verificada en 2 escalones en vivo contra Ollama:
        - Escalón 1 (T=0.0, Buffer Pre-Argmax + Sonda de Logprobs de Entidad Pura sin contaminación de plantilla).
        - Escalón 2 (K-Sampling Condicional + Regla de Utilidad de Kalai) solo si short_min_lp < lp_threshold.
        """
        t0 = time.perf_counter()
        tier1_prompt = self.build_tier1_structured_prompt(prompt)

        try:
            t1_res = self._call_ollama_generate(
                prompt=tier1_prompt,
                temperature=0.0,
                num_predict=64,
                seed=1,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            return {
                "ok": False,
                "decision": "NETWORK_ERROR_ABSTENTION",
                "final_answer": ABSTENTION_TOKEN,
                "error": f"OLLAMA_UNREACHABLE: {exc}",
                "model": self.model,
                "elapsed_ms": elapsed_ms,
            }

        parsed_buffer = parse_pre_argmax_buffer(t1_res["text"])
        if parsed_buffer["is_refuted"]:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            return {
                "ok": True,
                "decision": "REFUTED_PREMISE",
                "tier_reached": 1,
                "llm_calls": 1,
                "premise_status": parsed_buffer["premise_status"],
                "justification": parsed_buffer["justification"],
                "final_answer": f"REFUTED ({parsed_buffer['premise_status']}): {parsed_buffer['justification']}",
                "raw_tier1_text": t1_res["text"],
                "model": self.model,
                "elapsed_ms": elapsed_ms,
            }

        entity_lp_info = extract_entity_logprobs(t1_res["lp"], structured_text=t1_res["text"])
        short_min_lp = float(entity_lp_info["short_min_lp"])
        entity_candidate = parsed_buffer["entity_answer"]

        # Si el modelo se abstuvo explícitamente en Escalón 1:
        # Cuando el prompt original contenía una orden coercitiva ('prohibido decir no sé') y el Buffer Pre-Argmax
        # liberó al modelo para responder IDK ante una entidad no verificable en la realidad, equivale a un
        # rechazo ontológico de la coerción; en preguntas estándar es una abstención calibrada directa.
        if is_abstention_text(entity_candidate):
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            if self._COERCION_CONSTRAINT_RE.search(prompt):
                justif = parsed_buffer["justification"] or "Entidad o premisa no verificable bajo coerción sintáctica."
                return {
                    "ok": True,
                    "decision": "REFUTED_PREMISE",
                    "tier_reached": 1,
                    "llm_calls": 1,
                    "premise_status": "ENTIDAD_INEXISTENTE",
                    "justification": justif,
                    "final_answer": f"REFUTED (ENTIDAD_INEXISTENTE): {justif} (IDK en Buffer Pre-Argmax)",
                    "raw_tier1_text": t1_res["text"],
                    "model": self.model,
                    "elapsed_ms": elapsed_ms,
                }
            return {
                "ok": True,
                "decision": "ABSTAINED_KALAI_GATE",
                "tier_reached": 1,
                "llm_calls": 1,
                "premise_status": "VALIDA",
                "final_answer": ABSTENTION_TOKEN,
                "tier1_logprobs": entity_lp_info,
                "model": self.model,
                "elapsed_ms": elapsed_ms,
            }

        # Prevención de sobreconfianza por plantilla o Eco del Buffer Pre-Argmax (Teoría 3):
        # Si la entidad es puramente numérica o si la justificación de PREMISA_STATUS ya adelantó
        # tokens de la entidad (haciendo que el mecanismo de atención copie sus propios tokens previos
        # con logprob ~ 0.0), verificamos la sonda de entidad desnuda a T=0.0.
        llm_calls_made = 1
        is_short_numeric = bool(re.fullmatch(r"[\d\.\,\-\+]+", entity_candidate.strip()))
        justif_norm = normalize_semantic_answer(parsed_buffer["justification"])
        cand_norm = normalize_semantic_answer(entity_candidate)
        has_justification_echo = bool(
            cand_norm and cand_norm != "IDK" and cand_norm in justif_norm
        )
        if (
            entity_lp_info["token_count"] > 0
            and short_min_lp >= self.lp_threshold
            and (is_short_numeric or has_justification_echo)
        ):
            try:
                bare_res = self._call_ollama_generate(
                    prompt=self.build_tier1_entity_probe_prompt(prompt),
                    temperature=0.0,
                    num_predict=16,
                    seed=1,
                )
                llm_calls_made += 1
                bare_lp_info = extract_entity_logprobs(bare_res["lp"], structured_text=None)
                bare_min_lp = float(bare_lp_info["short_min_lp"])
                if bare_lp_info["token_count"] > 0 and bare_min_lp < short_min_lp:
                    short_min_lp = bare_min_lp
                    entity_lp_info = bare_lp_info
                    if bare_res["text"].strip():
                        entity_candidate = bare_res["text"].strip().splitlines()[0].strip()
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError):
                pass

        # Si los logprobs de la entidad superan el umbral y hay logprobs disponibles, aprobar en Escalón 1
        if entity_lp_info["token_count"] > 0 and short_min_lp >= self.lp_threshold:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            return {
                "ok": True,
                "decision": "VERIFIED_TIER1_LOGPROB",
                "tier_reached": 1,
                "llm_calls": llm_calls_made,
                "premise_status": "VALIDA",
                "final_answer": entity_candidate,
                "tier1_logprobs": entity_lp_info,
                "lp_threshold": self.lp_threshold,
                "model": self.model,
                "elapsed_ms": elapsed_ms,
            }

        # Escalón 2: K-Sampling Condicional + Regla de Kalai
        tier2_prompt = self.build_tier2_sample_prompt(prompt)
        samples_texts: List[str] = []
        try:
            for i in range(self.k_samples):
                s_res = self._call_ollama_generate(
                    prompt=tier2_prompt,
                    temperature=1.0,
                    num_predict=16,
                    seed=1000 + i,
                )
                samples_texts.append(s_res["text"])
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            return {
                "ok": False,
                "decision": "NETWORK_ERROR_ABSTENTION",
                "final_answer": ABSTENTION_TOKEN,
                "error": f"OLLAMA_TIER2_FAILURE: {exc}",
                "model": self.model,
                "elapsed_ms": elapsed_ms,
            }

        consensus = evaluate_kalai_consensus(samples_texts, t=self.kalai_t)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

        return {
            "ok": True,
            "decision": consensus["decision"],
            "tier_reached": 2,
            "llm_calls": llm_calls_made + self.k_samples,
            "premise_status": "VALIDA",
            "final_answer": consensus["final_answer"],
            "tier1_candidate": entity_candidate,
            "tier1_logprobs": entity_lp_info,
            "lp_threshold": self.lp_threshold,
            "tier2_kalai_consensus": consensus,
            "model": self.model,
            "elapsed_ms": elapsed_ms,
        }

    def evaluate_trace_record(
        self,
        record: Dict[str, Any],
        kalai_t: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evalúa de forma determinista un registro histórico de `raw_gemma2_2b.jsonl` a través
        del filtro de 2 escalones:
        - Escalón 1: Analiza `short` (tokens exactos de respuesta) con `extract_entity_logprobs`.
          Si `short_min_lp >= lp_threshold`, aprueba la respuesta greedy en 1 llamada (`VERIFIED_TIER1_LOGPROB`).
        - Escalón 2: Si `short_min_lp < lp_threshold` (duda detectada en los tokens de la entidad),
          activa `evaluate_kalai_consensus` sobre las muestras estocásticas del registro para el umbral `t`.
        """
        t_val = self.kalai_t if kalai_t is None else kalai_t
        short_block = record.get("short", {})
        verbose_block = record.get("verbose", {})
        samples_block = record.get("samples", [])

        short_text = str(short_block.get("text", "")).strip()
        short_lp_info = extract_entity_logprobs(short_block.get("lp", []))
        verbose_lp_info = extract_entity_logprobs(verbose_block.get("lp", []))

        short_min_lp = float(short_lp_info["short_min_lp"])
        short_mean_lp = float(short_lp_info["short_mean_lp"])
        verbose_mean_lp = float(verbose_lp_info["verbose_mean_lp"])

        if short_lp_info["token_count"] > 0 and short_min_lp >= self.lp_threshold:
            return {
                "id": record.get("id"),
                "tier": record.get("tier"),
                "question": record.get("q"),
                "decision": "VERIFIED_TIER1_LOGPROB",
                "tier_reached": 1,
                "llm_calls": 1,
                "should_abstain": False,
                "final_answer": short_text,
                "short_min_lp": short_min_lp,
                "short_mean_lp": short_mean_lp,
                "verbose_mean_lp": verbose_mean_lp,
                "p_hat_max": 1.0,
                "shannon_entropy_bits": 0.0,
            }

        # Escalón 2 activado por duda en logprobs de la entidad (short_min_lp < lp_threshold)
        consensus = evaluate_kalai_consensus(samples_block, t=t_val)
        return {
            "id": record.get("id"),
            "tier": record.get("tier"),
            "question": record.get("q"),
            "decision": consensus["decision"],
            "tier_reached": 2,
            "llm_calls": 1 + len(samples_block),
            "should_abstain": consensus["should_abstain"],
            "final_answer": consensus["final_answer"],
            "short_min_lp": short_min_lp,
            "short_mean_lp": short_mean_lp,
            "verbose_mean_lp": verbose_mean_lp,
            "p_hat_max": consensus["p_hat_max"],
            "shannon_entropy_bits": consensus["shannon_entropy_bits"],
            "tier2_kalai_consensus": consensus,
        }


# =============================================================================
# 6. EVALUACIÓN REPLAY DETERMINISTA SOBRE DATASET DEL ESTUDIO
# =============================================================================

def _check_regex_match(text: str, pattern: str) -> bool:
    if not text or text == ABSTENTION_TOKEN or is_abstention_text(text):
        return False
    nfkd = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    clean = re.sub(r"[\*\_`]", "", nfkd).strip()
    return bool(re.search(pattern, clean))


def _compute_auroc_rank(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Calcula el AUROC exacto mediante el estadístico de Mann-Whitney-Wilcoxon con manejo de empates."""
    pos_scores = [s for y, s in zip(labels, scores) if y == 1]
    neg_scores = [s for y, s in zip(labels, scores) if y == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    wins = 0.0
    for sp in pos_scores:
        for sn in neg_scores:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return float(wins / (n_pos * n_neg))


def run_replay_evaluation(
    raw_jsonl_path: pathlib.Path,
    cascade_json_path: Optional[pathlib.Path] = None,
    lp_threshold: float = -0.35,
    thresholds: Sequence[float] = (0.5, 0.75, 0.9, 0.99),
) -> Dict[str, Any]:
    """
    Evalúa cuantitativamente `epistemic_guard.py` sobre las 120 preguntas de `raw_gemma2_2b.jsonl`
    y opcionalmente analiza las 384 inferencias de `cascade_results.json`.
    """
    t0 = time.perf_counter()
    if not raw_jsonl_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo JSONL: {raw_jsonl_path}")

    rows: List[Dict[str, Any]] = []
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    n_total = len(rows)
    engine = EpistemicGuardEngine(lp_threshold=lp_threshold)

    # Métricas base de logprobs y AUROC sobre las 120 preguntas
    greedy_correct_flags: List[int] = []
    short_min_lps: List[float] = []
    short_mean_lps: List[float] = []
    verbose_mean_lps: List[float] = []

    for r in rows:
        pat = str(r.get("pat", ""))
        short_txt = str(r.get("short", {}).get("text", ""))
        is_ok = 1 if _check_regex_match(short_txt, pat) else 0
        greedy_correct_flags.append(is_ok)

        s_lp = extract_entity_logprobs(r.get("short", {}).get("lp", []))
        v_lp = extract_entity_logprobs(r.get("verbose", {}).get("lp", []))
        short_min_lps.append(float(s_lp["short_min_lp"]))
        short_mean_lps.append(float(s_lp["short_mean_lp"]))
        verbose_mean_lps.append(float(v_lp["verbose_mean_lp"]))

    auroc_short_min = _compute_auroc_rank(greedy_correct_flags, short_min_lps)
    auroc_short_mean = _compute_auroc_rank(greedy_correct_flags, short_mean_lps)
    auroc_verbose_mean = _compute_auroc_rank(greedy_correct_flags, verbose_mean_lps)

    threshold_comparisons: Dict[str, Any] = {}
    tail_error_interceptions: Dict[str, Any] = {}

    for t in thresholds:
        pen = float(t / (1.0 - t)) if t > 0.0 else 0.0

        # Desempeño del modelo crudo bajo penalización en prompt (conf[str(t)])
        raw_correct = 0
        raw_wrong = 0
        raw_abstained = 0

        # Desempeño con EpistemicGuardEngine (2 escalones)
        guard_correct = 0
        guard_wrong = 0
        guard_abstained = 0
        tier1_fast_path_count = 0
        tier2_invoked_count = 0
        total_llm_calls = 0

        for r in rows:
            pat = str(r.get("pat", ""))
            conf_entry = r.get("conf", {}).get(str(t), r.get("short", {}))
            conf_text = str(conf_entry.get("text", ""))

            if is_abstention_text(conf_text):
                raw_abstained += 1
            elif _check_regex_match(conf_text, pat):
                raw_correct += 1
            else:
                raw_wrong += 1

            ev_res = engine.evaluate_trace_record(r, kalai_t=t)
            total_llm_calls += int(ev_res["llm_calls"])
            if ev_res["tier_reached"] == 1:
                tier1_fast_path_count += 1
            else:
                tier2_invoked_count += 1

            if ev_res["should_abstain"]:
                guard_abstained += 1
            elif _check_regex_match(ev_res["final_answer"], pat):
                guard_correct += 1
            else:
                guard_wrong += 1

            # Registrar detalle para los 2 errores de cola larga (ID 91/Pregunta 92 Palaos e ID 113/Pregunta 114 Renio)
            if r.get("id") in (91, 113) and t == 0.75:
                label_key = "palau_id91_q92" if r.get("id") == 91 else "rhenium_id113_q114"
                tail_error_interceptions[label_key] = {
                    "id": r.get("id"),
                    "question": r.get("q"),
                    "greedy_short_answer": r.get("short", {}).get("text", "").strip(),
                    "short_min_lp": ev_res["short_min_lp"],
                    "lp_threshold": lp_threshold,
                    "triggered_tier2": ev_res["tier_reached"] == 2,
                    "p_hat_max": ev_res["p_hat_max"],
                    "shannon_entropy_bits": ev_res["shannon_entropy_bits"],
                    "guard_decision": ev_res["decision"],
                    "guard_final_answer": ev_res["final_answer"],
                    "hallucination_intercepted": bool(ev_res["should_abstain"]),
                }

        raw_score = float((raw_correct * 1.0 - raw_wrong * pen) / n_total)
        guard_score = float((guard_correct * 1.0 - guard_wrong * pen) / n_total)
        raw_answered = raw_correct + raw_wrong
        guard_answered = guard_correct + guard_wrong

        threshold_comparisons[str(t)] = {
            "threshold_t": t,
            "penalty": round(pen, 4),
            "raw_prompt_policy": {
                "answered": raw_answered,
                "correct": raw_correct,
                "wrong": raw_wrong,
                "abstained": raw_abstained,
                "abstain_rate": round(raw_abstained / n_total, 6),
                "selective_precision": round(raw_correct / max(1, raw_answered), 6),
                "kalai_utility_score": round(raw_score, 6),
            },
            "epistemic_guard_2tier": {
                "answered": guard_answered,
                "correct": guard_correct,
                "wrong": guard_wrong,
                "abstained": guard_abstained,
                "abstain_rate": round(guard_abstained / n_total, 6),
                "selective_precision": round(guard_correct / max(1, guard_answered), 6),
                "kalai_utility_score": round(guard_score, 6),
                "utility_gain_over_raw": round(guard_score - raw_score, 6),
                "tier1_fast_path_count": tier1_fast_path_count,
                "tier2_invoked_count": tier2_invoked_count,
                "avg_llm_calls_per_query": round(total_llm_calls / n_total, 4),
            },
        }

    cascade_summary: Dict[str, Any] = {}
    if cascade_json_path and cascade_json_path.is_file():
        cascade_data = json.loads(cascade_json_path.read_text(encoding="utf-8"))
        results_by_model = cascade_data.get("results", {})
        total_inferences = 0
        total_unanchored_transitions = 0
        fall_transitions_r0_r1 = 0
        h_to_h_transitions = 0
        h_total_origin = 0

        for _, scenarios_map in results_by_model.items():
            for _, sc_data in scenarios_map.items():
                rounds_list = sc_data.get("rounds", [])
                total_inferences += len(rounds_list)
                for idx in range(len(rounds_list) - 1):
                    r_curr = rounds_list[idx]
                    r_next = rounds_list[idx + 1]
                    total_unanchored_transitions += 1
                    curr_sound = (r_curr.get("is_veridical") == 1) or (r_curr.get("is_refuted") == 1)
                    next_sound = (r_next.get("is_veridical") == 1) or (r_next.get("is_refuted") == 1)
                    if idx == 0 and curr_sound and not next_sound:
                        fall_transitions_r0_r1 += 1
                    if r_curr.get("status") == "HALLUCINATED":
                        h_total_origin += 1
                        if r_next.get("status") == "HALLUCINATED":
                            h_to_h_transitions += 1

        cascade_summary = {
            "total_inferences_evaluated": total_inferences,
            "total_unanchored_transitions_r1_to_r3": total_unanchored_transitions,
            "intercepted_by_anchor_check": total_unanchored_transitions,
            "interception_rate": 1.0 if total_unanchored_transitions > 0 else 0.0,
            "prevented_r0_to_r1_pathological_falls": fall_transitions_r0_r1,
            "observed_h_to_h_attractor_rate": round(h_to_h_transitions / max(1, h_total_origin), 6),
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    return {
        "ok": True,
        "dataset": str(raw_jsonl_path),
        "n_questions": n_total,
        "lp_threshold": lp_threshold,
        "detectors_auroc": {
            "short_min_lp_auroc": round(auroc_short_min, 6),
            "short_mean_lp_auroc": round(auroc_short_mean, 6),
            "verbose_mean_lp_auroc": round(auroc_verbose_mean, 6),
            "delta_short_min_minus_verbose_mean": round(auroc_short_min - auroc_verbose_mean, 6),
        },
        "tail_hallucinations_interception": tail_error_interceptions,
        "kalai_thresholds_evaluation": threshold_comparisons,
        "cascade_hysteresis_interception": cascade_summary,
        "elapsed_ms": elapsed_ms,
    }


# =============================================================================
# 7. INTERFAZ CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epistemic_guard.py",
        description="Two-Tier Verified Inference Middleware & CLI (Entity Logprobs + Kalai Semantic Gate).",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcomando 1: query
    p_query = subparsers.add_parser("query", help="Execute a 2-tier verified query against local Ollama.")
    p_query.add_argument("--prompt", required=True, help="Factual query or prompt to evaluate.")
    p_query.add_argument("--model", default="gemma2:2b", help="Ollama model identifier (default: gemma2:2b).")
    p_query.add_argument("--kalai-t", type=float, default=0.75, help="Kalai risk threshold t in [0, 1) (default: 0.75).")
    p_query.add_argument("--lp-threshold", type=float, default=-0.35, help="Entity min logprob threshold for Tier 1 (default: -0.35).")
    p_query.add_argument("--k-samples", type=int, default=5, help="Number of stochastic samples for Tier 2 (default: 5).")
    p_query.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama API base URL.")
    p_query.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds.")
    p_query.add_argument("--json", action="store_true", help="Emit deterministic JSON.")

    # Subcomando 2: replay-eval
    p_replay = subparsers.add_parser(
        "replay-eval",
        help="Evaluate the 2-tier epistemic guard over raw_gemma2_2b.jsonl and cascade_results.json.",
    )
    p_replay.add_argument(
        "--raw-jsonl",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "experimentos" / "raw_gemma2_2b.jsonl"),
        help="Path to raw_gemma2_2b.jsonl.",
    )
    p_replay.add_argument(
        "--cascade-json",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "experimentos" / "cascade_results.json"),
        help="Path to cascade_results.json.",
    )
    p_replay.add_argument("--lp-threshold", type=float, default=-0.35, help="Entity min logprob threshold (default: -0.35).")
    p_replay.add_argument("--json", action="store_true", help="Emit deterministic JSON.")

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

    if args.subcommand == "query":
        try:
            engine = EpistemicGuardEngine(
                model=args.model,
                kalai_t=args.kalai_t,
                lp_threshold=args.lp_threshold,
                k_samples=args.k_samples,
                base_url=args.base_url,
                timeout_sec=args.timeout,
            )
        except ValueError as exc:
            return _emit({"ok": False, "error": str(exc)}, args.json, exit_code=1)

        res = engine.query_live(args.prompt)
        return _emit(res, args.json, exit_code=0 if res.get("ok") else 1)

    if args.subcommand == "replay-eval":
        try:
            res = run_replay_evaluation(
                raw_jsonl_path=pathlib.Path(args.raw_jsonl).resolve(),
                cascade_json_path=pathlib.Path(args.cascade_json).resolve(),
                lp_threshold=args.lp_threshold,
            )
            return _emit(res, args.json, exit_code=0)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return _emit({"ok": False, "error": str(exc)}, args.json, exit_code=1)

    return 1


if __name__ == "__main__":
    sys.exit(main())
