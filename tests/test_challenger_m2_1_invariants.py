# Creado por Francisco Grandón Vergara
"""
Test Suite: Challenger M2-1 Mathematical & Statistical Invariant Challenger
Empirically and mathematically challenges:
1. Sum of Squares Partition & Non-Negativity (1-way RM-ANOVA & 2-way Mixed ANOVA).
2. Exact F-statistic and p-value relations across all models and conditions.
3. Partial eta-squared bounds [0, 1] and mathematical definitions.
4. Mauchly Sphericity & Greenhouse-Geisser Epsilon bounds [1/(k-1), 1.0].
5. Degrees of freedom and corrected degrees of freedom identities.
6. McNemar chi2 >= 0, Edwards continuity correction, Cohen's g in [-0.5, 0.5].
7. Wilcoxon W >= 0, Z >= 0, rank-biserial r in [-1.0, 1.0].
8. Total absence of NaN and Inf across all JSON outputs.
9. Cross-artifact consistency: statistical_analysis_results.json vs cascade_results.json.
10. Protocolo de Superación: banned phrases, self-praise, mandatory attribution.
"""

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import scipy.stats as stats

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR / "experimentos"))

CASCADE_JSON = WORKSPACE_DIR / "experimentos" / "cascade_results.json"
CASCADE_CSV = WORKSPACE_DIR / "experimentos" / "cascade_comparative_multimodel.csv"
STATS_JSON = WORKSPACE_DIR / "experimentos" / "statistical_analysis_results.json"
ANALYZE_SCRIPT = WORKSPACE_DIR / "experimentos" / "analyze_cascade.py"


def check_no_nan_or_inf(obj: Any, path: str = "") -> List[str]:
    """Recursively checks that no NaN, Inf or -Inf exists in nested structures."""
    violations = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            violations.extend(check_no_nan_or_inf(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            violations.extend(check_no_nan_or_inf(item, f"{path}[{idx}]"))
    elif isinstance(obj, float):
        if math.isnan(obj):
            violations.append(f"NaN at {path}")
        if math.isinf(obj):
            violations.append(f"Inf at {path}")
    return violations


class TestANOVAInvariants(unittest.TestCase):
    """Adversarial stress-testing of Sum of Squares, variances, and ANOVA formulas."""

    @classmethod
    def setUpClass(cls):
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            cls.stats_data = json.load(f)
        cls.csv_df = pd.read_csv(CASCADE_CSV, encoding="utf-8-sig")

    def test_rm_anova_1way_sum_of_squares_partition_canonical(self):
        """
        Challenge SS Partition in 1-Way RM-ANOVA (Canonical Veridicality):
        SS_Total = SS_Subject + SS_Within = SS_Subject + SS_Round + SS_Error.
        """
        can_df = self.csv_df[self.csv_df["scenario_type"] == "canonical"].copy()
        piv = can_df.pivot(index=["model_id", "scenario_id"], columns="round", values="is_veridical").values
        N, k = piv.shape  # 32 x 4

        grand_mean = float(np.mean(piv))
        sub_means = np.mean(piv, axis=1)
        round_means = np.mean(piv, axis=0)

        ss_total = float(np.sum((piv - grand_mean) ** 2))
        ss_subjects = float(k * np.sum((sub_means - grand_mean) ** 2))
        ss_within = float(np.sum((piv - sub_means[:, None]) ** 2))
        ss_round = float(N * np.sum((round_means - grand_mean) ** 2))
        ss_error = float(np.sum((piv - sub_means[:, None] - round_means[None, :] + grand_mean) ** 2))

        # Check exact mathematical identities
        self.assertAlmostEqual(ss_total, ss_subjects + ss_within, places=10,
                               msg="SS_Total must equal SS_Subjects + SS_Within")
        self.assertAlmostEqual(ss_within, ss_round + ss_error, places=10,
                               msg="SS_Within must equal SS_Round + SS_Error")
        self.assertAlmostEqual(ss_total, ss_subjects + ss_round + ss_error, places=10,
                               msg="SS_Total must equal SS_Subjects + SS_Round + SS_Error")

        # Check values reported in statistical_analysis_results.json
        res = self.stats_data["repeated_measures_anova"]["canonical_veridicality"]["overall_within_subjects"]
        self.assertAlmostEqual(ss_round, res["SS_round"], places=5,
                               msg=f"SS_round mismatch: computed {ss_round} vs reported {res['SS_round']}")
        self.assertAlmostEqual(ss_error, res["SS_error"], places=5,
                               msg=f"SS_error mismatch: computed {ss_error} vs reported {res['SS_error']}")

        # Non-negativity
        self.assertGreaterEqual(ss_round, 0.0)
        self.assertGreaterEqual(ss_error, 0.0)
        self.assertGreaterEqual(res["MS_round"], 0.0)
        self.assertGreaterEqual(res["MS_error"], 0.0)
        self.assertGreaterEqual(res["F_statistic"], 0.0)

        # Exact F statistic from unrounded sums of squares: F = (SS_round / df_round) / (SS_error / df_error)
        exact_f = (ss_round / (k - 1)) / (ss_error / ((N - 1) * (k - 1)))
        self.assertAlmostEqual(exact_f, res["F_statistic"], places=5)
        # Check that dividing reported MS matches within rounding tolerance (delta=1e-3)
        self.assertAlmostEqual(res["MS_round"] / res["MS_error"], res["F_statistic"], delta=1e-3)

        # Partial eta-squared bounds: eta_p^2 = SS_round / (SS_round + SS_error)
        expected_eta_p2 = ss_round / (ss_round + ss_error)
        self.assertAlmostEqual(expected_eta_p2, res["partial_eta_squared"], places=5)
        self.assertGreaterEqual(res["partial_eta_squared"], 0.0)
        self.assertLessEqual(res["partial_eta_squared"], 1.0)

        # Degrees of freedom check
        self.assertEqual(res["df_round"], k - 1)
        self.assertEqual(res["df_error"], (N - 1) * (k - 1))

    def test_rm_anova_1way_sum_of_squares_partition_global_soundness(self):
        """
        Challenge SS Partition in 1-Way RM-ANOVA (Global Soundness, N=96):
        SS_Total = SS_Subject + SS_Round + SS_Error.
        """
        df_global = self.csv_df.copy()
        df_global["is_sound"] = ((df_global["is_veridical"] == 1) | (df_global["is_refuted"] == 1)).astype(int)
        piv = df_global.pivot(index=["model_id", "scenario_id"], columns="round", values="is_sound").values
        N, k = piv.shape  # 96 x 4

        grand_mean = float(np.mean(piv))
        sub_means = np.mean(piv, axis=1)
        round_means = np.mean(piv, axis=0)

        ss_total = float(np.sum((piv - grand_mean) ** 2))
        ss_subjects = float(k * np.sum((sub_means - grand_mean) ** 2))
        ss_within = float(np.sum((piv - sub_means[:, None]) ** 2))
        ss_round = float(N * np.sum((round_means - grand_mean) ** 2))
        ss_error = float(np.sum((piv - sub_means[:, None] - round_means[None, :] + grand_mean) ** 2))

        self.assertAlmostEqual(ss_total, ss_subjects + ss_within, places=10)
        self.assertAlmostEqual(ss_within, ss_round + ss_error, places=10)
        self.assertAlmostEqual(ss_total, ss_subjects + ss_round + ss_error, places=10)

        res = self.stats_data["repeated_measures_anova"]["global_soundness"]["overall_within_subjects"]
        self.assertAlmostEqual(ss_round, res["SS_round"], places=5)
        self.assertAlmostEqual(ss_error, res["SS_error"], places=5)
        self.assertGreaterEqual(res["partial_eta_squared"], 0.0)
        self.assertLessEqual(res["partial_eta_squared"], 1.0)
        self.assertEqual(res["df_round"], 3)
        self.assertEqual(res["df_error"], 95 * 3)  # 285

        exact_f = (ss_round / 3) / (ss_error / 285)
        self.assertAlmostEqual(exact_f, res["F_statistic"], places=5)

    def test_mixed_anova_2way_sum_of_squares_decomposition(self):
        """
        Challenge 5-component SS decomposition in 2-Way Mixed ANOVA:
        SS_Total = SS_Group + SS_SubWithinGroup + SS_Round + SS_Interaction + SS_Error2.
        """
        models = ["gemma2:2b", "gemma3:4b", "gemma4:e4b", "gemma4:e4b_ablated"]
        can_df = self.csv_df[self.csv_df["scenario_type"] == "canonical"].copy()

        Y_3d = []
        for m in models:
            sub = can_df[can_df["model_id"] == m]
            piv = sub.pivot(index="scenario_id", columns="round", values="is_veridical").values
            Y_3d.append(piv)
        Y_3d = np.array(Y_3d)  # (4, 8, 4)
        G, n, k = Y_3d.shape

        grand_mean = float(np.mean(Y_3d))
        group_means = np.mean(Y_3d, axis=(1, 2))
        round_means = np.mean(Y_3d, axis=(0, 1))
        group_round_means = np.mean(Y_3d, axis=1)
        sub_means = np.mean(Y_3d, axis=2)

        ss_total = float(np.sum((Y_3d - grand_mean) ** 2))
        ss_between_sub = float(k * np.sum((sub_means - grand_mean) ** 2))
        ss_group = float(n * k * np.sum((group_means - grand_mean) ** 2))
        ss_sub_within_group = float(ss_between_sub - ss_group)

        ss_within_sub = float(ss_total - ss_between_sub)
        ss_round = float(G * n * np.sum((round_means - grand_mean) ** 2))
        ss_interaction = float(n * np.sum((group_round_means - group_means[:, None] - round_means[None, :] + grand_mean) ** 2))
        ss_error2 = float(ss_within_sub - ss_round - ss_interaction)

        # Invariant: 5-way additive decomposition
        sum_of_5 = ss_group + ss_sub_within_group + ss_round + ss_interaction + ss_error2
        self.assertAlmostEqual(ss_total, sum_of_5, places=10,
                               msg=f"SS_Total ({ss_total}) must equal sum of 5 components ({sum_of_5})")

        # Invariant: Degrees of Freedom sum to Total df
        df_group = G - 1  # 3
        df_sub_within_group = G * (n - 1)  # 28
        df_round = k - 1  # 3
        df_interaction = (G - 1) * (k - 1)  # 9
        df_error2 = G * (n - 1) * (k - 1)  # 84
        df_total = G * n * k - 1  # 127
        self.assertEqual(df_total, df_group + df_sub_within_group + df_round + df_interaction + df_error2)

        # Compare with reported values in JSON
        mixed = self.stats_data["repeated_measures_anova"]["canonical_veridicality"]["mixed_anova"]
        self.assertAlmostEqual(ss_group, mixed["between_subjects_factor"]["SS"], places=5)
        self.assertAlmostEqual(ss_sub_within_group, mixed["between_subjects_factor"]["SS_error"], places=5)
        self.assertAlmostEqual(ss_round, mixed["within_subjects_factor"]["SS"], places=5)
        self.assertAlmostEqual(ss_interaction, mixed["interaction_factor"]["SS"], places=5)
        self.assertAlmostEqual(ss_error2, mixed["within_subjects_factor"]["SS_error"], places=5)

        # Partial eta-squared bounds
        for factor in [mixed["between_subjects_factor"], mixed["within_subjects_factor"], mixed["interaction_factor"]]:
            eta = factor["partial_eta_squared"]
            self.assertGreaterEqual(eta, 0.0)
            self.assertLessEqual(eta, 1.0)

    def test_per_model_rm_anova_subsets(self):
        """Verify per-model RM-ANOVA computations and edge cases (e.g. ablated zero variance)."""
        per_model = self.stats_data["repeated_measures_anova"]["canonical_veridicality"]["per_model_anova"]
        for m_id, res in per_model.items():
            self.assertEqual(res["n_subjects"], 8)
            self.assertEqual(res["k_conditions"], 4)
            self.assertEqual(res["df_round"], 3)
            self.assertEqual(res["df_error"], 21)
            self.assertGreaterEqual(res["SS_round"], 0.0)
            self.assertGreaterEqual(res["SS_error"], 0.0)
            self.assertGreaterEqual(res["partial_eta_squared"], 0.0)
            self.assertLessEqual(res["partial_eta_squared"], 1.0)

        # Specific check for gemma4:e4b_ablated (zero variance: veridical across all rounds)
        abl = per_model["gemma4:e4b_ablated"]
        self.assertEqual(abl["SS_round"], 0.0)
        self.assertEqual(abl["SS_error"], 0.0)
        self.assertEqual(abl["F_statistic"], 0.0)
        self.assertEqual(abl["p_value_uncorrected"], 1.0)
        self.assertEqual(abl["partial_eta_squared"], 0.0)


class TestSphericityAndGreenhouseGeisserBounds(unittest.TestCase):
    """Adversarial stress-testing of Mauchly's Sphericity and Greenhouse-Geisser Epsilon bounds."""

    @classmethod
    def setUpClass(cls):
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            cls.stats_data = json.load(f)
        cls.csv_df = pd.read_csv(CASCADE_CSV, encoding="utf-8-sig")

    def test_greenhouse_geisser_theoretical_bounds(self):
        """
        Greenhouse-Geisser epsilon must satisfy:
        1 / (k - 1) <= epsilon <= 1.0
        For k = 4 conditions: 1 / 3 = 0.333333... <= epsilon <= 1.0.
        """
        k = 4
        lower_bound = 1.0 / (k - 1)
        upper_bound = 1.0

        def traverse_and_check_epsilons(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for k_key, v in obj.items():
                    if k_key in ("epsilon", "greenhouse_geisser_epsilon"):
                        eps = float(v)
                        self.assertGreaterEqual(
                            eps, lower_bound - 1e-6,
                            f"GG epsilon {eps} at {path}.{k_key} violates theoretical lower bound {lower_bound}!"
                        )
                        self.assertLessEqual(
                            eps, upper_bound + 1e-6,
                            f"GG epsilon {eps} at {path}.{k_key} violates theoretical upper bound {upper_bound}!"
                        )
                    traverse_and_check_epsilons(v, f"{path}.{k_key}" if path else str(k_key))
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    traverse_and_check_epsilons(item, f"{path}[{idx}]")

        traverse_and_check_epsilons(self.stats_data["repeated_measures_anova"])

    def test_greenhouse_geisser_adjusted_degrees_of_freedom(self):
        """
        Adjusted df must satisfy:
        df_corrected = epsilon * df_original.
        Recomputes raw double-precision epsilon from data to verify exact match.
        """
        can_df = self.csv_df[self.csv_df["scenario_type"] == "canonical"].copy()
        piv = can_df.pivot(index=["model_id", "scenario_id"], columns="round", values="is_veridical").values
        S = np.cov(piv, rowvar=False)
        k = 4
        p = k - 1
        C = np.zeros((p, k), dtype=float)
        for i in range(p):
            C[i, : i + 1] = 1.0
            C[i, i + 1] = -(i + 1)
        row_norms = np.linalg.norm(C, axis=1, keepdims=True)
        M = (C / row_norms).T
        Sigma_M = M.T @ S @ M
        tr_sigma = float(np.trace(Sigma_M))
        tr_sq = float(np.trace(Sigma_M @ Sigma_M))
        raw_eps = float((tr_sigma ** 2) / (p * tr_sq))

        can_within = self.stats_data["repeated_measures_anova"]["canonical_veridicality"]["overall_within_subjects"]
        df_round = can_within["df_round"]  # 3
        df_error = can_within["df_error"]  # 93

        # Exact unrounded calculation matches reported JSON to 6 decimal places
        self.assertAlmostEqual(raw_eps, can_within["greenhouse_geisser_correction"]["epsilon"], places=6)
        self.assertAlmostEqual(raw_eps * df_round, can_within["greenhouse_geisser_correction"]["df_round_corrected"], places=6)
        self.assertAlmostEqual(raw_eps * df_error, can_within["greenhouse_geisser_correction"]["df_error_corrected"], places=6)

        # Check p-value calculation with adjusted df
        f_stat = can_within["F_statistic"]
        expected_p_gg = float(stats.f.sf(f_stat, raw_eps * df_round, raw_eps * df_error))
        reported_p_gg = can_within["greenhouse_geisser_correction"]["p_value_corrected"]
        self.assertAlmostEqual(expected_p_gg, reported_p_gg, places=6)


class TestMcNemarInvariants(unittest.TestCase):
    """Adversarial stress-testing of McNemar contrast and Cohen's g effect size."""

    @classmethod
    def setUpClass(cls):
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            cls.stats_data = json.load(f)

    def test_mcnemar_chi2_non_negativity_and_edwards_formula(self):
        """
        Verify:
        chi2 = (|b - c| - 1)^2 / (b + c) >= 0.
        When |b - c| <= 1 or b + c == 0, chi2 == 0.0.
        """
        mcnemar = self.stats_data["mcnemar_test"]
        for category in ["step_by_step_transitions", "endpoint_transitions_r0_to_r3", "adversarial_initiation_r0_to_r1"]:
            for stratum, res in mcnemar[category].items():
                b = res["b_fall"]
                c = res["c_recovery"]
                chi2 = res["chi2_statistic"]
                p_val = res["p_value"]
                df = res["degrees_of_freedom"]

                self.assertGreaterEqual(chi2, 0.0, f"Chi2 negative in {category}/{stratum}")
                self.assertEqual(df, 1, f"df != 1 in {category}/{stratum}")
                self.assertGreaterEqual(p_val, 0.0)
                self.assertLessEqual(p_val, 1.0)

                diff = abs(b - c)
                total_disc = b + c
                if total_disc > 0 and diff > 1:
                    expected_chi2 = ((diff - 1.0) ** 2) / total_disc
                else:
                    expected_chi2 = 0.0

                self.assertAlmostEqual(expected_chi2, chi2, places=5,
                                       msg=f"Edwards chi2 mismatch in {category}/{stratum}")

                # Verify survival function of chi2 distribution with 1 df
                expected_p = float(stats.chi2.sf(expected_chi2, 1))
                self.assertAlmostEqual(expected_p, p_val, places=6,
                                       msg=f"p-value mismatch in {category}/{stratum}")

    def test_cohen_g_effect_size_bounds(self):
        """
        Cohen's g = b / (b + c) - 0.5 must strictly lie in [-0.5, +0.5].
        """
        mcnemar = self.stats_data["mcnemar_test"]
        for category in ["step_by_step_transitions", "endpoint_transitions_r0_to_r3", "adversarial_initiation_r0_to_r1"]:
            for stratum, res in mcnemar[category].items():
                b = res["b_fall"]
                c = res["c_recovery"]
                g = res["cohen_g"]

                self.assertGreaterEqual(g, -0.5 - 1e-6, f"Cohen g < -0.5 in {category}/{stratum}")
                self.assertLessEqual(g, +0.5 + 1e-6, f"Cohen g > +0.5 in {category}/{stratum}")

                total_disc = b + c
                expected_g = (b / total_disc - 0.5) if total_disc > 0 else 0.0
                self.assertAlmostEqual(expected_g, g, places=5)

    def test_contingency_table_conservation(self):
        """Verify contingency table sums match total paired observations."""
        mcnemar = self.stats_data["mcnemar_test"]
        # In step_by_step_transitions:
        # Global: 288 transitions (4 models x 24 scenarios x 3 steps)
        global_step = mcnemar["step_by_step_transitions"]["global"]
        total_obs = sum(sum(row) for row in global_step["contingency_table"])
        self.assertEqual(total_obs, 288, f"Global step transitions should sum to 288, got {total_obs}")

        # Endpoint transitions: 96 transitions (4 models x 24 scenarios)
        global_endpoint = mcnemar["endpoint_transitions_r0_to_r3"]["global"]
        total_endpoint = sum(sum(row) for row in global_endpoint["contingency_table"])
        self.assertEqual(total_endpoint, 96, f"Global endpoint transitions should sum to 96, got {total_endpoint}")


class TestWilcoxonInvariants(unittest.TestCase):
    """Adversarial stress-testing of Wilcoxon signed-rank test and rank-biserial correlation."""

    @classmethod
    def setUpClass(cls):
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            cls.stats_data = json.load(f)

    def test_wilcoxon_statistic_and_rank_biserial_bounds(self):
        """
        Verify:
        W >= 0, W+ >= 0, W- >= 0.
        rank-biserial r = (W+ - W-) / (W+ + W-) in [-1.0, +1.0].
        Z >= 0, p-value in [0, 1].
        """
        wilcoxon = self.stats_data["wilcoxon_signed_rank"]
        for section in ["gemma4_vs_ablated_non_canonical", "adversarial_expansion_within_gemma4"]:
            for pair_key, res in wilcoxon[section].items():
                w = res["W_statistic"]
                w_pos = res["W_positive"]
                w_neg = res["W_negative"]
                z = res["Z_score"]
                p_val = res["p_value"]
                r = res["rank_biserial_correlation_r"]

                self.assertGreaterEqual(w, 0.0, f"W < 0 in {pair_key}")
                self.assertGreaterEqual(w_pos, 0.0, f"W+ < 0 in {pair_key}")
                self.assertGreaterEqual(w_neg, 0.0, f"W- < 0 in {pair_key}")
                self.assertEqual(w, min(w_pos, w_neg), f"W != min(W+, W-) in {pair_key}")
                self.assertGreaterEqual(z, 0.0, f"Z < 0 in {pair_key}")
                self.assertGreaterEqual(p_val, 0.0, f"p < 0 in {pair_key}")
                self.assertLessEqual(p_val, 1.0, f"p > 1 in {pair_key}")

                # Rank-biserial bounds
                self.assertGreaterEqual(r, -1.0 - 1e-6, f"r < -1.0 in {pair_key}")
                self.assertLessEqual(r, +1.0 + 1e-6, f"r > +1.0 in {pair_key}")

                tot = w_pos + w_neg
                expected_r = (w_pos - w_neg) / tot if tot > 0 else 0.0
                self.assertAlmostEqual(expected_r, r, places=5)


class TestCrossArtifactConsistency(unittest.TestCase):
    """Challenge consistency between statistical_analysis_results.json and cascade_results.json."""

    def test_injected_metrics_in_cascade_json_match_stats_json(self):
        """Verify metrics.statistical_tests in cascade_results.json matches statistical_analysis_results.json."""
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            stats_data = json.load(f)
        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            cascade_data = json.load(f)

        self.assertIn("metrics", cascade_data)
        self.assertIn("statistical_tests", cascade_data["metrics"])
        injected = cascade_data["metrics"]["statistical_tests"]

        self.assertEqual(injected["mcnemar_test"], stats_data["mcnemar_test"])
        self.assertEqual(injected["repeated_measures_anova"], stats_data["repeated_measures_anova"])
        self.assertEqual(injected["wilcoxon_signed_rank"], stats_data["wilcoxon_signed_rank"])


class TestArtifactIntegrityAndSuperacion(unittest.TestCase):
    """Adversarial testing of NaN/Inf, banned phrases, and attribution."""

    def test_absence_of_nan_and_inf_in_statistical_analysis_results(self):
        """Verify zero NaN or Inf anywhere in statistical_analysis_results.json."""
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        violations = check_no_nan_or_inf(data)
        self.assertEqual(len(violations), 0, f"Found NaN/Inf in statistical_analysis_results.json: {violations}")

    def test_absence_of_nan_and_inf_in_cascade_results(self):
        """Verify zero NaN or Inf anywhere in cascade_results.json."""
        with open(CASCADE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        violations = check_no_nan_or_inf(data)
        self.assertEqual(len(violations), 0, f"Found NaN/Inf in cascade_results.json: {violations}")

    def test_mandatory_authorship_attribution(self):
        """Verify line 1 attribution in analyze_cascade.py."""
        with open(ANALYZE_SCRIPT, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        self.assertEqual(first_line, "# Creado por Francisco Grandón Vergara")

    def test_zero_laziness_banned_phrases(self):
        """Verify absence of laziness phrases in script and results."""
        banned = [
            "se podría",
            "idealmente",
            "como mejora futura",
            "queda pendiente",
            "trabajo futuro",
            "en el futuro",
            "podríamos",
            "sería deseable",
        ]
        for path in [ANALYZE_SCRIPT, STATS_JSON]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().lower()
            for phrase in banned:
                self.assertNotIn(phrase, content, f"Banned phrase '{phrase}' found in {path.name}!")

    def test_zero_self_praise(self):
        """Verify absence of self-praise adjectives in analyze_cascade.py."""
        self_praise = [
            "impecable",
            "perfecto",
            "magistral",
            "asombroso",
            "inigualable",
            "extraordinario",
            "excelso",
        ]
        with open(ANALYZE_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read().lower()
        for word in self_praise:
            self.assertNotIn(word, content, f"Self-praise '{word}' found in analyze_cascade.py!")


class TestSyntheticAdversarialCornerCases(unittest.TestCase):
    """
    Adversarial stress-testing of analyze_cascade.py functions under synthetic extreme edge cases:
    - Constant matrices (zero variance)
    - Perfect sphericity (identity covariance) vs Maximum sphericity violation (rank 1 covariance)
    - Zero discordant pairs in McNemar, boundary |b-c| = 1
    - Identical samples in Wilcoxon, extensive tie structures
    """

    def test_mcnemar_zero_discordants_and_boundary(self):
        from analyze_cascade import compute_mcnemar_edwards
        # Edge Case 1: b=0, c=0 (zero discordant pairs)
        res = compute_mcnemar_edwards(b=0, c=0, a=10, d=10)
        self.assertEqual(res["chi2_statistic"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["cohen_g"], 0.0)
        self.assertFalse(math.isnan(res["chi2_statistic"]))
        self.assertFalse(math.isnan(res["cohen_g"]))

        # Edge Case 2: diff = 1 (b=1, c=0)
        res2 = compute_mcnemar_edwards(b=1, c=0, a=10, d=10)
        self.assertEqual(res2["chi2_statistic"], 0.0)
        self.assertEqual(res2["p_value"], 1.0)
        self.assertEqual(res2["cohen_g"], 0.5)

        # Edge Case 3: all falls (b=100, c=0) -> cohen_g = +0.5
        res3 = compute_mcnemar_edwards(b=100, c=0)
        self.assertEqual(res3["cohen_g"], 0.5)
        self.assertGreater(res3["chi2_statistic"], 0.0)

        # Edge Case 4: all recoveries (b=0, c=100) -> cohen_g = -0.5
        res4 = compute_mcnemar_edwards(b=0, c=100)
        self.assertEqual(res4["cohen_g"], -0.5)

    def test_greenhouse_geisser_extremes_identity_and_singular(self):
        from analyze_cascade import compute_mauchly_and_greenhouse_geisser
        k = 4
        # Case 1: Perfect Sphericity S = 2.5 * I_k
        S_perfect = 2.5 * np.eye(k)
        w, chi2, df, p, eps = compute_mauchly_and_greenhouse_geisser(S_perfect, n_effective=30, k_conditions=k)
        self.assertAlmostEqual(w, 1.0, places=5)
        self.assertAlmostEqual(eps, 1.0, places=5)
        self.assertEqual(df, 5)

        # Case 2: Rank-1 covariance (maximum possible violation of sphericity)
        # S = v @ v.T where v is a non-constant vector
        v = np.array([[1.0], [2.0], [3.0], [4.0]])
        S_rank1 = v @ v.T
        w2, chi2_2, df2, p2, eps2 = compute_mauchly_and_greenhouse_geisser(S_rank1, n_effective=30, k_conditions=k)
        # Under rank-1 covariance, theoretical lower bound is 1 / (k-1) = 1/3
        self.assertAlmostEqual(eps2, 1.0 / (k - 1), places=5)
        self.assertGreaterEqual(eps2, 1.0 / (k - 1))
        self.assertLessEqual(eps2, 1.0)

        # Case 3: Zero matrix
        S_zero = np.zeros((k, k))
        w3, chi2_3, df3, p3, eps3 = compute_mauchly_and_greenhouse_geisser(S_zero, n_effective=30, k_conditions=k)
        self.assertFalse(math.isnan(eps3))
        self.assertFalse(math.isnan(w3))
        self.assertGreaterEqual(eps3, 1.0 / (k - 1))
        self.assertLessEqual(eps3, 1.0)

    def test_rm_anova_constant_data_and_zero_variance(self):
        from analyze_cascade import compute_repeated_measures_anova_1way
        # Constant data: all 1s
        Y_const = np.ones((10, 4))
        res = compute_repeated_measures_anova_1way(Y_const)
        self.assertEqual(res["SS_round"], 0.0)
        self.assertEqual(res["SS_error"], 0.0)
        self.assertEqual(res["MS_round"], 0.0)
        self.assertEqual(res["MS_error"], 0.0)
        self.assertEqual(res["F_statistic"], 0.0)
        self.assertEqual(res["p_value_uncorrected"], 1.0)
        self.assertEqual(res["partial_eta_squared"], 0.0)
        self.assertFalse(math.isnan(res["F_statistic"]))
        self.assertFalse(math.isnan(res["partial_eta_squared"]))

    def test_wilcoxon_identical_samples_and_ties(self):
        from analyze_cascade import compute_wilcoxon_signed_rank
        # Case 1: Identical arrays (zero differences)
        x = np.array([5.0, 10.0, 15.0])
        y = np.array([5.0, 10.0, 15.0])
        res = compute_wilcoxon_signed_rank(x, y)
        self.assertEqual(res["N_non_zero_differences"], 0)
        self.assertEqual(res["W_statistic"], 0.0)
        self.assertEqual(res["Z_score"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["rank_biserial_correlation_r"], 0.0)

        # Case 2: Extensive ties in differences
        x2 = np.array([10.0, 20.0, 20.0, 20.0, 30.0])
        y2 = np.array([5.0, 15.0, 15.0, 15.0, 20.0])
        # Diffs: [5, 5, 5, 5, 10] -> four ties of 5
        res2 = compute_wilcoxon_signed_rank(x2, y2)
        self.assertGreaterEqual(res2["W_statistic"], 0.0)
        self.assertGreaterEqual(res2["Z_score"], 0.0)
        self.assertEqual(res2["rank_biserial_correlation_r"], 1.0)
        self.assertFalse(math.isnan(res2["Z_score"]))


if __name__ == "__main__":
    unittest.main()

