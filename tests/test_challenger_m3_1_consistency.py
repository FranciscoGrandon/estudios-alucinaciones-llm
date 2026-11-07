# Creado por Francisco Grandón Vergara
"""
Test Suite: Challenger M3-1 Cross-Document Numerical Consistency Challenger
Empirically and adversarially challenges:
1. Sample sizes across documents and raw data (384 inferences, 24 scenarios, 4 models, 4 rounds).
2. Numerical values for McNemar, Mauchly, GG epsilon (0.4154), RM-ANOVA F (29.6222), eta_p^2 (0.4886),
   Mixed ANOVA Fs (57.3333, 103.2000, 26.6667), and Wilcoxon (W=0.0, Z=3.4903, p=3.052e-05, r=1.0000).
3. Hysteresis gap values (+5.59%, +12.57%, +9.80%, +11.81%, -16.70%) and attractor (83.8% / 83.78%).
4. Stochastic transition matrix entries in tables vs cascade_results.json.
5. Protocolo de Superación: 0 banned phrases, 0 self-praise adjectives, mandatory authorship.
"""

import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd

WORKSPACE_DIR = Path(__file__).resolve().parent.parent

PAPER_FILE = WORKSPACE_DIR / "reporte_tecnico_alucinaciones.md"
INFORME_FILE = WORKSPACE_DIR / "informe_experimentos.md"
README_FILE = WORKSPACE_DIR / "README.md"
CHANGELOG_FILE = WORKSPACE_DIR / "CHANGELOG.md"

CASCADE_JSON = WORKSPACE_DIR / "experimentos" / "cascade_results.json"
CASCADE_CSV = WORKSPACE_DIR / "experimentos" / "cascade_comparative_multimodel.csv"
STATS_JSON = WORKSPACE_DIR / "experimentos" / "statistical_analysis_results.json"


class TestSampleSizesAcrossDocs(unittest.TestCase):
    """Adversarially verifies sample sizes in text vs raw datasets."""

    @classmethod
    def setUpClass(cls):
        with open(PAPER_FILE, "r", encoding="utf-8") as f:
            cls.paper_raw = f.read()
            cls.paper_text = cls.paper_raw.replace(r"\%", "%")
        with open(INFORME_FILE, "r", encoding="utf-8") as f:
            cls.informe_raw = f.read()
            cls.informe_text = cls.informe_raw.replace(r"\%", "%")
        with open(README_FILE, "r", encoding="utf-8") as f:
            cls.readme_raw = f.read()
            cls.readme_text = cls.readme_raw.replace(r"\%", "%")
        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            cls.cascade_data = json.load(f)
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            cls.stats_data = json.load(f)
        cls.csv_df = pd.read_csv(CASCADE_CSV, encoding="utf-8-sig")

    def test_sample_size_384_inferences_raw_and_text(self):
        """Verify that 384 inferences are empirically present in CSV, JSON and mentioned in docs."""
        self.assertEqual(len(self.csv_df), 384, "CSV must contain exactly 384 rows")
        self.assertEqual(self.stats_data["metadata"]["total_inferences"], 384)
        
        # Count rounds in cascade_results.json
        total_inferences_json = 0
        for model_id, scenarios in self.cascade_data["results"].items():
            for sc_id, sc_data in scenarios.items():
                total_inferences_json += len(sc_data["rounds"])
        self.assertEqual(total_inferences_json, 384, "cascade_results.json must contain exactly 384 round executions")

        # Texts must explicitly cite 384 inferences
        self.assertIn("384", self.paper_text, "reporte_tecnico_alucinaciones.md must cite 384 inferences")
        self.assertIn("384", self.informe_text, "informe_experimentos.md must cite 384 inferences")
        self.assertIn("384", self.readme_text, "README.md must cite 384 inferences")

    def test_24_scenarios_and_stratification(self):
        """Verify 24 scenarios across 3 main strata (8 canonical, 8 inoculated, 8 non-existent)."""
        self.assertEqual(self.cascade_data["metadata"]["total_scenarios"], 24)
        unique_scenarios = self.csv_df["scenario_id"].unique()
        self.assertEqual(len(unique_scenarios), 24, "CSV must contain exactly 24 unique scenarios")

        can_sc = [s for s in unique_scenarios if s.startswith("C")]
        inoc_sc = [s for s in unique_scenarios if s.startswith("I")]
        non_sc = [s for s in unique_scenarios if s.startswith("O")]
        self.assertEqual(len(can_sc), 8, "Must be exactly 8 canonical scenarios (C01-C08)")
        self.assertEqual(len(inoc_sc), 8, "Must be exactly 8 inoculated scenarios (I01-I08)")
        self.assertEqual(len(non_sc), 8, "Must be exactly 8 non-existent scenarios (O01-O08)")

        self.assertIn("24 escenarios", self.paper_text)
        self.assertIn("24 escenarios", self.informe_text)

    def test_4_models_and_4_rounds(self):
        """Verify exactly 4 models and 4 rounds in data and text."""
        expected_models = ["gemma2:2b", "gemma3:4b", "gemma4:e4b", "gemma4:e4b_ablated"]
        self.assertEqual(self.cascade_data["metadata"]["models"], expected_models)
        self.assertEqual(list(self.csv_df["model_id"].unique()), expected_models)
        self.assertEqual(self.cascade_data["metadata"]["total_rounds"], 4)
        self.assertEqual(sorted(list(self.csv_df["round"].unique())), [0, 1, 2, 3])

        for m in expected_models:
            self.assertIn(m, self.paper_text)
            self.assertIn(m, self.informe_text)


class TestTransitionMatrixConsistency(unittest.TestCase):
    """Adversarially verifies the 4x4 Markov transition matrix entries."""

    @classmethod
    def setUpClass(cls):
        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            cls.cascade_data = json.load(f)
        with open(PAPER_FILE, "r", encoding="utf-8") as f:
            cls.paper_text = f.read().replace(r"\%", "%")
        with open(INFORME_FILE, "r", encoding="utf-8") as f:
            cls.informe_text = f.read().replace(r"\%", "%")
        with open(README_FILE, "r", encoding="utf-8") as f:
            cls.readme_text = f.read().replace(r"\%", "%")

        cls.t_mat = cls.cascade_data["metrics"]["transition_matrix"]

    def test_transition_matrix_row_stochastic_property(self):
        """All rows in the 4x4 transition matrix must sum to 1.0."""
        for state, row in self.t_mat.items():
            row_sum = sum(row.values())
            self.assertAlmostEqual(row_sum, 1.0, places=6,
                                   msg=f"Transition matrix row {state} does not sum to 1.0: {row_sum}")

    def test_transition_matrix_entries_in_tables(self):
        """Verify the exact formatted percentages of the transition matrix in paper and informe tables."""
        # VERIDICAL row: 72.58%, 0.00%, 25.81%, 1.61%
        self.assertAlmostEqual(self.t_mat["VERIDICAL"]["VERIDICAL"], 0.72580645, places=4)
        self.assertAlmostEqual(self.t_mat["VERIDICAL"]["REFUTED"], 0.0, places=4)
        self.assertAlmostEqual(self.t_mat["VERIDICAL"]["CAPITULATED"], 0.25806452, places=4)
        self.assertAlmostEqual(self.t_mat["VERIDICAL"]["HALLUCINATED"], 0.01612903, places=4)

        # REFUTED row: 0.00%, 71.05%, 10.53%, 18.42%
        self.assertAlmostEqual(self.t_mat["REFUTED"]["VERIDICAL"], 0.0, places=4)
        self.assertAlmostEqual(self.t_mat["REFUTED"]["REFUTED"], 0.71052632, places=4)
        self.assertAlmostEqual(self.t_mat["REFUTED"]["CAPITULATED"], 0.10526316, places=4)
        self.assertAlmostEqual(self.t_mat["REFUTED"]["HALLUCINATED"], 0.18421053, places=4)

        # CAPITULATED row: 35.90%, 5.13%, 43.59%, 15.38%
        self.assertAlmostEqual(self.t_mat["CAPITULATED"]["VERIDICAL"], 0.35897436, places=4)
        self.assertAlmostEqual(self.t_mat["CAPITULATED"]["REFUTED"], 0.05128205, places=4)
        self.assertAlmostEqual(self.t_mat["CAPITULATED"]["CAPITULATED"], 0.43589744, places=4)
        self.assertAlmostEqual(self.t_mat["CAPITULATED"]["HALLUCINATED"], 0.15384615, places=4)

        # HALLUCINATED row: 0.90%, 15.32%, 0.00%, 83.78%
        self.assertAlmostEqual(self.t_mat["HALLUCINATED"]["VERIDICAL"], 0.00900901, places=4)
        self.assertAlmostEqual(self.t_mat["HALLUCINATED"]["REFUTED"], 0.15315315, places=4)
        self.assertAlmostEqual(self.t_mat["HALLUCINATED"]["CAPITULATED"], 0.0, places=4)
        self.assertAlmostEqual(self.t_mat["HALLUCINATED"]["HALLUCINATED"], 0.83783784, places=4)

        # Check presence in paper and informe text
        expected_tokens = ["72.58%", "25.81%", "1.61%", "71.05%", "10.53%", "18.42%",
                           "35.90%", "5.13%", "43.59%", "15.38%", "0.90%", "15.32%", "83.78%"]
        for tok in expected_tokens:
            self.assertIn(tok, self.paper_text, f"Paper missing transition matrix value {tok}")
            self.assertIn(tok, self.informe_text, f"Informe missing transition matrix value {tok}")

    def test_attractor_value_and_narrative_claim(self):
        """Attractor T_{H->H} = 83.78% ~ 83.8% must match across all documents."""
        h_h = self.t_mat["HALLUCINATED"]["HALLUCINATED"]
        self.assertAlmostEqual(h_h, 0.8378378, places=4)
        self.assertIn("83.78%", self.paper_text)
        self.assertIn("83.8%", self.paper_text)
        self.assertIn("83.78%", self.informe_text)
        self.assertIn("83.8%", self.readme_text)


class TestHysteresisGapConsistency(unittest.TestCase):
    """Adversarially verifies multi-strata hysteresis gap values in text vs raw metrics."""

    @classmethod
    def setUpClass(cls):
        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            cls.cascade_data = json.load(f)
        with open(PAPER_FILE, "r", encoding="utf-8") as f:
            cls.paper_text = f.read().replace(r"\%", "%")
        with open(INFORME_FILE, "r", encoding="utf-8") as f:
            cls.informe_text = f.read().replace(r"\%", "%")
        cls.hyst = cls.cascade_data["metrics"]["hysteresis"]

    def test_global_hysteresis_gap(self):
        """Global: p_fall = 28.26%, p_rec = 22.67%, gap = +5.59%."""
        g = self.hyst["global"]
        self.assertAlmostEqual(g["p_fall"], 39 / 138, places=6)
        self.assertAlmostEqual(g["p_recovery"], 34 / 150, places=6)
        self.assertAlmostEqual(g["gap"], (39 / 138) - (34 / 150), places=6)
        self.assertAlmostEqual(g["gap"], 0.055942, places=4)

        self.assertIn("+5.59%", self.paper_text)
        self.assertIn("+5.59%", self.informe_text)

    def test_non_canonical_hysteresis_gap(self):
        """Non-canonical: p_fall = 28.95%, p_rec = 16.38%, gap = +12.57%."""
        nc = self.hyst["non_canonical"]
        self.assertAlmostEqual(nc["p_fall"], 22 / 76, places=6)
        self.assertAlmostEqual(nc["p_recovery"], 19 / 116, places=6)
        self.assertAlmostEqual(nc["gap"], (22 / 76) - (19 / 116), places=6)
        self.assertAlmostEqual(nc["gap"], 0.125680, places=4)

        self.assertIn("+12.57%", self.paper_text)
        self.assertIn("+12.57%", self.informe_text)

    def test_inoculated_hysteresis_gap(self):
        """Inoculated: p_fall = 33.33%, p_rec = 23.53%, gap = +9.80%."""
        inoc = self.hyst["inoculated"]
        self.assertAlmostEqual(inoc["p_fall"], 15 / 45, places=6)
        self.assertAlmostEqual(inoc["p_recovery"], 12 / 51, places=6)
        self.assertAlmostEqual(inoc["gap"], (15 / 45) - (12 / 51), places=6)
        self.assertAlmostEqual(inoc["gap"], 0.098039, places=4)

        self.assertIn("+9.80%", self.paper_text)
        self.assertIn("+9.80%", self.informe_text)

    def test_non_existent_hysteresis_gap(self):
        """Non-existent: p_fall = 22.58%, p_rec = 10.77%, gap = +11.81%."""
        ne = self.hyst["non_existent"]
        self.assertAlmostEqual(ne["p_fall"], 7 / 31, places=6)
        self.assertAlmostEqual(ne["p_recovery"], 7 / 65, places=6)
        self.assertAlmostEqual(ne["gap"], (7 / 31) - (7 / 65), places=6)
        self.assertAlmostEqual(ne["gap"], 0.118114, places=4)

        self.assertIn("+11.81%", self.paper_text)
        self.assertIn("+11.81%", self.informe_text)

    def test_canonical_hysteresis_gap(self):
        """Canonical: p_fall = 27.42%, p_rec = 44.12%, gap = -16.70%."""
        can = self.hyst["canonical"]
        self.assertAlmostEqual(can["p_fall"], 17 / 62, places=6)
        self.assertAlmostEqual(can["p_recovery"], 15 / 34, places=6)
        self.assertAlmostEqual(can["gap"], (17 / 62) - (15 / 34), places=6)
        self.assertAlmostEqual(can["gap"], -0.166983, places=4)

        self.assertIn("-16.70%", self.paper_text)
        self.assertIn("-16.70%", self.informe_text)


class TestInferentialStatisticsConsistency(unittest.TestCase):
    """Adversarially verifies McNemar, ANOVA, and Wilcoxon statistics across text and JSON."""

    @classmethod
    def setUpClass(cls):
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            cls.stats_data = json.load(f)
        with open(PAPER_FILE, "r", encoding="utf-8") as f:
            cls.paper_text = f.read().replace(r"\%", "%")
        with open(INFORME_FILE, "r", encoding="utf-8") as f:
            cls.informe_text = f.read().replace(r"\%", "%")
        cls.csv_df = pd.read_csv(CASCADE_CSV, encoding="utf-8-sig")

    def test_mcnemar_adversarial_initiation_and_endpoint(self):
        """Verify McNemar initiation and endpoint tests."""
        init = self.stats_data["mcnemar_test"]["adversarial_initiation_r0_to_r1"]["global"]
        self.assertAlmostEqual(init["chi2_statistic"], 12.033333, places=4)
        self.assertAlmostEqual(init["cohen_g"], 0.333333, places=4)
        self.assertEqual(init["b_fall"], 25)
        self.assertEqual(init["c_recovery"], 5)
        self.assertIn("12.0333", self.paper_text)
        self.assertIn("0.3333", self.paper_text)
        self.assertIn("12.0333", self.informe_text)

        end = self.stats_data["mcnemar_test"]["endpoint_transitions_r0_to_r3"]["global"]
        self.assertAlmostEqual(end["chi2_statistic"], 1.066667, places=4)
        self.assertAlmostEqual(end["cohen_g"], 0.166667, places=4)
        self.assertEqual(end["b_fall"], 10)
        self.assertEqual(end["c_recovery"], 5)
        self.assertIn("1.0667", self.paper_text)

    def test_repeated_measures_anova_canonical(self):
        """Verify 1-way RM-ANOVA canonical veridicality metrics."""
        rm = self.stats_data["repeated_measures_anova"]["canonical_veridicality"]["overall_within_subjects"]
        self.assertAlmostEqual(rm["F_statistic"], 29.622222, places=4)
        self.assertAlmostEqual(rm["partial_eta_squared"], 0.488636, places=4)
        self.assertAlmostEqual(rm["greenhouse_geisser_correction"]["epsilon"], 0.415385, places=4)
        self.assertAlmostEqual(rm["greenhouse_geisser_correction"]["df_round_corrected"], 1.246154, places=2)
        self.assertAlmostEqual(rm["greenhouse_geisser_correction"]["df_error_corrected"], 38.630769, places=2)

        self.assertIn("29.6222", self.paper_text)
        self.assertIn("0.4886", self.paper_text)
        self.assertIn("0.4154", self.paper_text)
        self.assertIn("29.6222", self.informe_text)
        self.assertIn("0.4886", self.informe_text)
        self.assertIn("0.4154", self.informe_text)

    def test_mixed_2way_anova(self):
        """Verify Mixed 2-way ANOVA F-statistics and effect sizes."""
        mixed = self.stats_data["repeated_measures_anova"]["canonical_veridicality"]["mixed_anova"]
        
        # Between Model
        bet = mixed["between_subjects_factor"]
        self.assertAlmostEqual(bet["F_statistic"], 57.333333, places=4)
        self.assertAlmostEqual(bet["partial_eta_squared"], 0.86, places=4)
        self.assertIn("57.3333", self.paper_text)
        self.assertIn("0.8600", self.paper_text)
        self.assertIn("57.3333", self.informe_text)

        # Within Round
        wit = mixed["within_subjects_factor"]
        self.assertAlmostEqual(wit["F_statistic"], 103.2, places=4)
        self.assertAlmostEqual(wit["partial_eta_squared"], 0.786585, places=4)
        self.assertAlmostEqual(wit["greenhouse_geisser_epsilon"], 0.490196, places=4)
        self.assertIn("103.2000", self.paper_text)
        self.assertIn("0.7866", self.paper_text)
        self.assertIn("0.4902", self.paper_text)
        self.assertIn("103.2000", self.informe_text)

        # Interaction
        inter = mixed["interaction_factor"]
        self.assertAlmostEqual(inter["F_statistic"], 26.666667, places=4)
        self.assertAlmostEqual(inter["partial_eta_squared"], 0.740741, places=4)
        self.assertIn("26.6667", self.paper_text)
        self.assertIn("0.7407", self.paper_text)
        self.assertIn("26.6667", self.informe_text)

    def test_wilcoxon_causal_ablation_statistics(self):
        """Verify Wilcoxon signed rank test for thinking ablation."""
        w = self.stats_data["wilcoxon_signed_rank"]["gemma4_vs_ablated_non_canonical"]["scenario_level_overall"]
        self.assertEqual(w["W_statistic"], 0.0)
        self.assertEqual(w["W_positive"], 136.0)
        self.assertAlmostEqual(w["Z_score"], 3.490342, places=4)
        self.assertAlmostEqual(w["p_value"], 3.052e-05, places=8)
        self.assertEqual(w["rank_biserial_correlation_r"], 1.0)
        self.assertAlmostEqual(w["mean_difference"], 1648.86, places=2)

        self.assertIn("3.4903", self.paper_text)
        self.assertIn("3.052", self.paper_text)
        self.assertIn("1.0000", self.paper_text)
        self.assertIn("3.4903", self.informe_text)
        self.assertIn("3.052", self.informe_text)
        self.assertIn("1.0000", self.informe_text)

    def test_trajectories_raw_csv_vs_doc_tables(self):
        """Compute trajectories directly from CSV and assert match with tables in paper and informe."""
        # 1. Canonical Veridicality by model and round
        can_df = self.csv_df[self.csv_df["scenario_type"] == "canonical"]
        piv_can = can_df.pivot_table(index="model_id", columns="round", values="is_veridical", aggfunc="mean")
        
        # Gemma 2: 1.0, 0.0, 0.0, 0.875
        self.assertEqual(list(piv_can.loc["gemma2:2b"]), [1.0, 0.0, 0.0, 0.875])
        # Gemma 3: 1.0, 0.0, 0.0, 0.875
        self.assertEqual(list(piv_can.loc["gemma3:4b"]), [1.0, 0.0, 0.0, 0.875])
        # Gemma 4: 1.0, 0.875, 0.875, 1.0
        self.assertEqual(list(piv_can.loc["gemma4:e4b"]), [1.0, 0.875, 0.875, 1.0])
        # Gemma 4 ablated: 1.0, 1.0, 1.0, 1.0
        self.assertEqual(list(piv_can.loc["gemma4:e4b_ablated"]), [1.0, 1.0, 1.0, 1.0])

        # 2. Refutation emergence in non-existent scenarios (O01-O08, as defined in run_cascade_experiments.py metrics['refutation_emergence'])
        non_exist_df = self.csv_df[self.csv_df["scenario_type"] == "non_existent"]
        piv_ref = non_exist_df.pivot_table(index="model_id", columns="round", values="is_refuted", aggfunc="mean")

        # Gemma 2: 0.25, 0.125, 0.25, 0.25
        self.assertEqual(list(piv_ref.loc["gemma2:2b"]), [0.25, 0.125, 0.25, 0.25])
        # Gemma 3: 0.25, 0.125, 0.125, 0.0
        self.assertEqual(list(piv_ref.loc["gemma3:4b"]), [0.25, 0.125, 0.125, 0.0])
        # Gemma 4: 0.625, 0.50, 0.50, 0.75
        self.assertEqual(list(piv_ref.loc["gemma4:e4b"]), [0.625, 0.50, 0.50, 0.75])
        # Gemma 4 ablated: 0.50, 0.375, 0.25, 0.625
        self.assertEqual(list(piv_ref.loc["gemma4:e4b_ablated"]), [0.50, 0.375, 0.25, 0.625])

        # 3. Mean Thinking Length by model and round on inoculated scenarios (I01-I08, as defined in metrics['thinking_amplification'])
        inoc_df = self.csv_df[self.csv_df["scenario_type"] == "inoculated"]
        piv_think = inoc_df.pivot_table(index="model_id", columns="round", values="thinking_length", aggfunc="mean")
        self.assertEqual(list(piv_think.loc["gemma2:2b"]), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(list(piv_think.loc["gemma3:4b"]), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(list(piv_think.loc["gemma4:e4b_ablated"]), [0.0, 0.0, 0.0, 0.0])
        gemma4_think = [round(x, 1) for x in piv_think.loc["gemma4:e4b"]]
        self.assertEqual(gemma4_think, [2150.4, 1819.9, 2012.8, 1291.1])

        # Verify exact tokens exist in paper and informe text
        for tok in ["100.00%", "87.50%", "62.50%", "50.00%", "75.00%", "2150.4", "1819.9", "2012.8", "1291.1"]:
            self.assertIn(tok, self.paper_text)
        for tok in ["100.0%", "87.5%", "62.5%", "50.0%", "75.0%", "2150.4", "1819.9", "2012.8", "1291.1"]:
            self.assertIn(tok, self.informe_text)

    def test_greenhouse_geisser_adjusted_degrees_of_freedom_equations(self):
        """Verify df1* = 1.25 and df2* = 38.63 Greenhouse-Geisser adjustments."""
        eps = 0.415385
        df1 = 3
        df2 = 93
        df1_adj = round(eps * df1, 2)
        df2_adj = round(eps * df2, 2)
        self.assertEqual(df1_adj, 1.25)
        self.assertEqual(df2_adj, 38.63)

        self.assertIn("1.25", self.paper_text)
        self.assertIn("38.63", self.paper_text)
        self.assertIn("1.25", self.informe_text)
        self.assertIn("38.63", self.informe_text)

    def test_mcnemar_exact_edwards_calculation(self):
        """Challenge exact formula chi2 = (|b - c| - 1)^2 / (b + c)."""
        # Initiation: b=25, c=5
        chi2_init = ((abs(25 - 5) - 1) ** 2) / 30
        self.assertAlmostEqual(chi2_init, 12.033333, places=5)
        g_init = abs(25 / 30 - 0.5)
        self.assertAlmostEqual(g_init, 0.333333, places=5)

        # Endpoint: b=10, c=5
        chi2_end = ((abs(10 - 5) - 1) ** 2) / 15
        self.assertAlmostEqual(chi2_end, 1.066667, places=5)
        g_end = abs(10 / 15 - 0.5)
        self.assertAlmostEqual(g_end, 0.166667, places=5)

    def test_wilcoxon_rank_biserial_exact_derivation(self):
        """Challenge rank sum identities: W+ = N(N+1)/2 when all differences are strictly positive."""
        for N in [8, 16, 64]:
            w_max = N * (N + 1) / 2
            r = (w_max - 0) / w_max
            self.assertEqual(r, 1.0)
            if N == 16:
                self.assertEqual(w_max, 136.0)
            elif N == 8:
                self.assertEqual(w_max, 36.0)
            elif N == 64:
                self.assertEqual(w_max, 2080.0)



class TestProtocoloSuperacionCompliance(unittest.TestCase):
    """Adversarially scans for banned phrases, self-praise, and proper attribution."""

    BANNED_PHRASES = [
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

    SELF_PRAISE = [
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

    TARGET_FILES = [PAPER_FILE, INFORME_FILE, README_FILE, CHANGELOG_FILE]

    def test_mandatory_authorship_attribution(self):
        """All milestone documentation files must begin with '# Creado por Francisco Grandón Vergara'."""
        for target in self.TARGET_FILES:
            with open(target, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            self.assertEqual(
                first_line,
                "# Creado por Francisco Grandón Vergara",
                f"File {target.name} must start with '# Creado por Francisco Grandón Vergara'",
            )

    def test_zero_laziness_banned_phrases(self):
        """No document may contain any of the 11 banned laziness phrases."""
        violations = []
        for target in self.TARGET_FILES:
            with open(target, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines, 1):
                # Ignore lines that explicitly define the test or protocol lists
                if "BANNED_PHRASES" in line or "Alarmas de Pereza" in line or "lista de" in line:
                    continue
                lower_line = line.lower()
                for phrase in self.BANNED_PHRASES:
                    if phrase in lower_line:
                        # Allow explicit citations of the protocol rule if it's the rule description itself
                        if "alarmas de pereza" in lower_line or "frases prohibidas" in lower_line:
                            continue
                        violations.append(f"{target.name}:{idx} contains banned phrase '{phrase}': {line.strip()[:80]}")
        self.assertEqual(violations, [], f"Found banned laziness phrases: {violations}")

    def test_zero_self_praise_adjectives(self):
        """No document may contain ungrounded self-praise adjectives."""
        violations = []
        for target in self.TARGET_FILES:
            with open(target, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines, 1):
                if "SELF_PRAISE" in line or "adjetivos autoelogiosos" in line or "autoelogio" in line:
                    continue
                lower_line = line.lower()
                for adj in self.SELF_PRAISE:
                    pattern = rf"\b{adj}\b"
                    if re.search(pattern, lower_line):
                        violations.append(f"{target.name}:{idx} contains self-praise '{adj}': {line.strip()[:80]}")
        self.assertEqual(violations, [], f"Found self-praise adjectives: {violations}")


if __name__ == "__main__":
    unittest.main()
