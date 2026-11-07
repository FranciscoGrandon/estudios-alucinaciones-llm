# Creado por Francisco Grandón Vergara
"""
Test Suite: Challenger M3-2 Adversarial Text & Protocolo de Superación Stress Tester
Empirically verifies:
1. Zero occurrences of 11 banned laziness phrases + passive hedges in deliverables.
2. Zero occurrences of 15 self-praise adjectives in deliverables.
3. Mandatory authorship attribution on Line 1 of all markdown deliverables and core code.
4. Sensitivity and robustness of the detector against synthetic adversarial injections (negative oracles).
5. Formal 4-point closing sweep log presence in M3 handoff artifacts.
"""

import os
import re
import unicodedata
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent

# Core deliverable markdown files for Milestone 3
DELIVERABLE_DOCS = [
    WORKSPACE_DIR / "reporte_tecnico_alucinaciones.md",
    WORKSPACE_DIR / "informe_experimentos.md",
    WORKSPACE_DIR / "README.md",
    WORKSPACE_DIR / "CHANGELOG.md",
]

# Canonical 11 banned laziness phrases
BANNED_LAZINESS_11 = [
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

# Passive hedges
PASSIVE_HEDGES = [
    "trabajo futuro",
    "en el futuro",
    "podríamos",
    "sería deseable",
]

# All 15 self-praise adjectives from Protocolo de Superación
FORBIDDEN_SELF_PRAISE_15 = [
    "excelente",
    "impecable",
    "perfecto",
    "extraordinario",
    "óptimo",
    "brillante",
    "formidable",
    "magnífico",
    "sublime",
    "maravilloso",
    "insuperable",
    "magistral",
    "asombroso",
    "inigualable",
    "excelso",
]

MANDATORY_AUTHORSHIP_HEADER = "# Creado por Francisco Grandón Vergara"


def strip_accents(text: str) -> str:
    """Removes diacritics / accents for robust normalization."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def scan_text_for_laziness(text: str, filename: str = "text") -> List[Tuple[int, str, str]]:
    """
    Scans a given text (line by line) for banned laziness phrases and passive hedges.
    Returns a list of (line_number, matched_phrase, line_content).
    """
    violations = []
    lines = text.splitlines()
    all_laziness = BANNED_LAZINESS_11 + PASSIVE_HEDGES

    for line_no, raw_line in enumerate(lines, start=1):
        clean = raw_line.lower()
        clean_no_acc = strip_accents(clean)

        # Allow explicit meta-discussions or rule definitions
        if any(term in clean or term in clean_no_acc for term in [
            "alarma de pereza", "frase prohibida", "protocolo de superacion",
            "banned_laziness", "passive_hedges"
        ]):
            continue

        for phrase in all_laziness:
            p_clean = phrase.lower()
            p_no_acc = strip_accents(p_clean)

            if p_clean in clean or p_no_acc in clean_no_acc:
                violations.append((line_no, phrase, raw_line.strip()))
                break

    return violations


def scan_text_for_self_praise(text: str, filename: str = "text") -> List[Tuple[int, str, str]]:
    """
    Scans a given text for forbidden self-praise adjectives.
    Checks both direct word boundaries and contextual self-praise patterns.
    Returns a list of (line_number, matched_adjective, line_content).
    """
    violations = []
    lines = text.splitlines()

    for line_no, raw_line in enumerate(lines, start=1):
        clean = raw_line.lower()
        clean_no_acc = strip_accents(clean)

        # Allow explicit meta-discussions or rule definitions
        if any(term in clean or term in clean_no_acc for term in [
            "autoelogio", "protocolo de superacion", "forbidden_self_praise",
            "self_praise", "regla 3"
        ]):
            continue

        for adj in FORBIDDEN_SELF_PRAISE_15:
            adj_clean = adj.lower()
            adj_no_acc = strip_accents(adj_clean)

            # Match exact word boundaries
            pat = rf"\b{re.escape(adj_no_acc)}\b"
            if re.search(pat, clean_no_acc):
                violations.append((line_no, adj, raw_line.strip()))
                break

    return violations


class TestBannedLazinessPhrasesInDeliverables(unittest.TestCase):
    """Verifies that all M3 deliverable markdown files contain zero banned laziness phrases."""

    def test_all_deliverable_docs_exist(self):
        """All M3 deliverable files must physically exist."""
        for doc in DELIVERABLE_DOCS:
            self.assertTrue(doc.exists(), f"Deliverable file {doc.name} does not exist at {doc}")

    def test_zero_canonical_11_laziness_phrases(self):
        """Scans for all 11 canonical laziness phrases in deliverable documents."""
        all_violations = []
        for doc in DELIVERABLE_DOCS:
            with open(doc, "r", encoding="utf-8") as f:
                content = f.read()
            violations = scan_text_for_laziness(content, doc.name)
            for line_no, phrase, line in violations:
                if phrase in BANNED_LAZINESS_11:
                    all_violations.append(f"{doc.name}:{line_no} -> Banned phrase '{phrase}': '{line}'")

        self.assertEqual(
            all_violations,
            [],
            f"Detected banned laziness phrases:\n" + "\n".join(all_violations)
        )

    def test_zero_passive_hedges(self):
        """Scans for passive hedge extensions ('trabajo futuro', 'en el futuro', 'podríamos', 'sería deseable')."""
        all_violations = []
        for doc in DELIVERABLE_DOCS:
            with open(doc, "r", encoding="utf-8") as f:
                content = f.read()
            violations = scan_text_for_laziness(content, doc.name)
            for line_no, phrase, line in violations:
                if phrase in PASSIVE_HEDGES:
                    all_violations.append(f"{doc.name}:{line_no} -> Passive hedge '{phrase}': '{line}'")

        self.assertEqual(
            all_violations,
            [],
            f"Detected passive hedges:\n" + "\n".join(all_violations)
        )


class TestForbiddenSelfPraiseInDeliverables(unittest.TestCase):
    """Verifies that all M3 deliverable files contain zero self-praise adjectives."""

    def test_zero_self_praise_adjectives(self):
        """Scans for all 15 self-praise adjectives in deliverable documents."""
        all_violations = []
        for doc in DELIVERABLE_DOCS:
            with open(doc, "r", encoding="utf-8") as f:
                content = f.read()
            violations = scan_text_for_self_praise(content, doc.name)
            for line_no, adj, line in violations:
                all_violations.append(f"{doc.name}:{line_no} -> Self-praise adjective '{adj}': '{line}'")

        self.assertEqual(
            all_violations,
            [],
            f"Detected self-praise adjectives in deliverables:\n" + "\n".join(all_violations)
        )


class TestMandatoryAuthorshipLine1(unittest.TestCase):
    """Verifies line 1 of all markdown deliverables and core code files contains official authorship."""

    def test_markdown_deliverables_line1(self):
        """Line 1 of every deliverable markdown file must exactly equal '# Creado por Francisco Grandón Vergara'."""
        for doc in DELIVERABLE_DOCS:
            self.assertTrue(doc.exists(), f"{doc.name} missing")
            with open(doc, "r", encoding="utf-8") as f:
                first_line = f.readline().rstrip("\r\n")
            self.assertEqual(
                first_line,
                MANDATORY_AUTHORSHIP_HEADER,
                f"{doc.name}: Line 1 must be '{MANDATORY_AUTHORSHIP_HEADER}', but got '{first_line}'"
            )

    def test_experimentos_python_files_line1(self):
        """Line 1 of key python scripts in experimentos/ must contain authorship."""
        experimentos_dir = WORKSPACE_DIR / "experimentos"
        core_scripts = [
            experimentos_dir / "run_cascade_experiments.py",
            experimentos_dir / "analyze_cascade.py",
        ]
        for script in core_scripts:
            self.assertTrue(script.exists(), f"{script.name} missing")
            with open(script, "r", encoding="utf-8") as f:
                first_line = f.readline().rstrip("\r\n")
            self.assertEqual(
                first_line,
                MANDATORY_AUTHORSHIP_HEADER,
                f"{script.name}: Line 1 must be '{MANDATORY_AUTHORSHIP_HEADER}', but got '{first_line}'"
            )


class TestDetectorNegativeOraclesAndSensitivity(unittest.TestCase):
    """
    Stress-tests the scanner logic itself using synthetic adversarial inputs.
    Confirms the scanner catches every banned phrase, hedge, and self-praise adjective without evasion.
    """

    def test_catches_all_11_banned_phrases_synthetically(self):
        """Ensures that every one of the 11 banned phrases is detected by scan_text_for_laziness."""
        for phrase in BANNED_LAZINESS_11:
            synthetic_text = f"En esta sección {phrase} observar la variabilidad."
            findings = scan_text_for_laziness(synthetic_text)
            self.assertGreaterEqual(
                len(findings), 1,
                f"Scanner failed to detect canonical phrase: '{phrase}'"
            )

    def test_catches_all_passive_hedges_synthetically(self):
        """Ensures that every passive hedge is detected."""
        for hedge in PASSIVE_HEDGES:
            synthetic_text = f"Esto se abordará como {hedge} en la investigación."
            findings = scan_text_for_laziness(synthetic_text)
            self.assertGreaterEqual(
                len(findings), 1,
                f"Scanner failed to detect passive hedge: '{hedge}'"
            )

    def test_catches_all_15_self_praise_adjectives_synthetically(self):
        """Ensures that all 15 self-praise adjectives are detected."""
        for adj in FORBIDDEN_SELF_PRAISE_15:
            synthetic_text = f"Obtuvimos un resultado {adj} en el benchmark."
            findings = scan_text_for_self_praise(synthetic_text)
            self.assertGreaterEqual(
                len(findings), 1,
                f"Scanner failed to detect self-praise adjective: '{adj}'"
            )

    def test_accent_insensitivity_and_casing(self):
        """Ensures that scanner detects unaccented or uppercase variants."""
        variations = [
            "SE PODRÍA verificar esto.",
            "se podria verificar esto.",
            "queda PENDIENTE realizar la prueba.",
            "Un modelo OPTIMO sin precedentes.",
            "un diseno extraordinario y PERFECTO.",
            "en el FUTURO resolveremos la cota.",
        ]
        for var in variations:
            findings_laziness = scan_text_for_laziness(var)
            findings_praise = scan_text_for_self_praise(var)
            total_findings = len(findings_laziness) + len(findings_praise)
            self.assertGreaterEqual(
                total_findings, 1,
                f"Scanner failed to detect variation: '{var}'"
            )


class TestClosingSweepLogCompliance(unittest.TestCase):
    """Verifies that the 4-point closing sweep log exists and is complete in Worker M3 handoff."""

    def test_closing_sweep_in_worker_m3_handoff(self):
        """Worker M3 handoff must contain all 4 sections of the closing sweep."""
        worker_handoff = WORKSPACE_DIR / ".agents" / "teamwork" / "worker_m3" / "handoff.md"
        self.assertTrue(worker_handoff.exists(), "Worker M3 handoff.md must exist")

        with open(worker_handoff, "r", encoding="utf-8") as f:
            content = f.read().lower()

        required_sections = [
            r"qu[eé]\s+busqu[eé]\s+en\s+el\s+barrido",
            r"qu[eé]\s+encontr[eé]\s+y\s+correg[ií]",
            r"qu[eé]\s+qued[oó][^a-zA-Z0-9]+no\s+verificado",
            r"no\s+completado",
        ]

        for pat in required_sections:
            self.assertIsNotNone(
                re.search(pat, content),
                f"Missing section in worker_m3 closing sweep matching: {pat}"
            )


if __name__ == "__main__":
    unittest.main()
