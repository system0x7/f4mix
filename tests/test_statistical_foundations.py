from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from f4mix.core._admixpy.genotypes import get_block_lengths, read_snp
from f4mix.core.profiles import (
    F4ProfileData,
    _vectorized_count_jackknife,
    _vectorized_influence_covariance,
    fit_f4_profiles,
)


class BlockCovarianceTests(unittest.TestCase):
    def test_complete_equal_blocks_match_covariance_of_the_mean(self):
        values = np.random.default_rng(7).normal(size=(4, 80))
        _, _, influence, contributes = _vectorized_count_jackknife(
            values, np.ones_like(values),
        )
        covariance = _vectorized_influence_covariance(influence, contributes)
        np.testing.assert_allclose(covariance, np.cov(values) / 80, atol=1e-14)

    def test_nested_coverage_matches_sampling_covariance(self):
        # Known sampling covariance: means of 50 and 100 independent unit-
        # variance blocks sharing the first 50 have covariance 1/100.
        rng = np.random.default_rng(17)
        counts = np.ones((2, 100))
        counts[0, 50:] = 0
        estimates = []
        for _ in range(2000):
            values = np.tile(rng.normal(size=100), (2, 1))
            _, _, influence, contributes = _vectorized_count_jackknife(values, counts)
            estimates.append(_vectorized_influence_covariance(influence, contributes))
        np.testing.assert_allclose(
            np.mean(estimates, axis=0),
            [[0.02, 0.01], [0.01, 0.01]],
            rtol=0.025, atol=0,
        )

    def test_uneven_masks_are_psd_and_disjoint_blocks_have_zero_covariance(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(12, 60))
        counts = rng.integers(1, 100, size=values.shape).astype(float)
        counts[rng.random(values.shape) < 0.5] = 0
        counts[0, :30] = 10
        counts[0, 30:] = 0
        counts[1, :30] = 0
        counts[1, 30:] = 10
        _, _, influence, contributes = _vectorized_count_jackknife(values, counts)
        covariance = _vectorized_influence_covariance(influence, contributes)
        self.assertTrue(np.isfinite(covariance).all())
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)
        self.assertEqual(covariance[0, 1], 0)

    def test_insufficient_blocks_remain_unavailable(self):
        values = np.array([[1., 2., 3.], [1., np.nan, np.nan]])
        counts = np.array([[1., 1., 1.], [1., 0., 0.]])
        _, _, influence, contributes = _vectorized_count_jackknife(values, counts)
        covariance = _vectorized_influence_covariance(influence, contributes)
        self.assertTrue(np.isfinite(covariance[0, 0]))
        self.assertTrue(np.isnan(covariance[1]).all())
        self.assertTrue(np.isnan(covariance[:, 1]).all())


class WeightObjectiveTests(unittest.TestCase):
    def test_asymptotic_fit_pvalue_uses_full_contrast_dimension(self):
        profile = F4ProfileData(
            sources=("A", "B"), features=("f1", "f2"),
            matrix=np.array([[2., 0.], [2., 0.]]),
            covariance=np.eye(4) * .25,
        )
        fit = fit_f4_profiles(profile, jackknife=False)
        self.assertEqual(fit.fit_dof, 2)
        self.assertEqual(fit.pvalue_method, "chi2_d_conservative_asymptotic")
        self.assertEqual(fit.fit_status, "rejected")
        self.assertAlmostEqual(fit.fit_pvalue, np.exp(-fit.fit_statistic / 2))
        np.testing.assert_allclose(fit.fit_statistic, fit.chi_square, atol=1e-7)

    def test_identical_profiles_do_not_get_a_precision_driven_split(self):
        profile = F4ProfileData(
            sources=("A", "B"), features=("f1", "f2"),
            matrix=np.zeros((2, 2)),
            covariance=np.diag([.01, .01, .09, .09]),
            fit_covariance=np.eye(2) * .025,
            loo=np.zeros((2, 2, 4)),
        )
        fit = fit_f4_profiles(profile)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.weights, [.5, .5], atol=1e-8)
        self.assertFalse(fit.weights_identifiable)
        self.assertEqual(fit.source_contrast_rank, 0)
        self.assertEqual(fit.free_weight_parameters, 1)
        self.assertTrue(np.isnan(fit.standard_errors).all())
        self.assertAlmostEqual(fit.chi_square, 0)

    def test_known_mixture_recovered_despite_unequal_source_precision(self):
        profile = F4ProfileData(
            sources=("A", "B"), features=("f1", "f2"),
            matrix=np.array([[-.7, -1.4], [.3, .6]]),
            covariance=np.diag([.01, .01, .09, .09]),
            fit_covariance=np.eye(2) * .025,
        )
        for fit_covariance in (profile.fit_covariance, None):
            fit = fit_f4_profiles(replace(profile, fit_covariance=fit_covariance), jackknife=False)
            self.assertTrue(fit.success)
            self.assertTrue(fit.weights_identifiable)
            np.testing.assert_allclose(fit.weights, [.3, .7], atol=1e-6)
            self.assertAlmostEqual(fit.chi_square, fit.fit_statistic, places=7)

    def test_jackknife_uses_same_minimum_q_estimator(self):
        # A deletion changes the exact zero-residual mixture. Its jackknife
        # variance is available analytically, including unequal precision.
        mixtures = np.array([.27, .29, .31, .33])
        matrices = np.array([[[w - 1, 2 * (w - 1)], [w, 2 * w]] for w in mixtures])
        profile = F4ProfileData(
            sources=("A", "B"), features=("f1", "f2"),
            matrix=matrices.mean(axis=0), loo=matrices.transpose(1, 2, 0),
            covariance=np.diag([.01, .01, .09, .09]),
        )
        fit = fit_f4_profiles(profile)
        expected_se = np.sqrt(3 / 4 * np.sum((mixtures - mixtures.mean()) ** 2))
        self.assertTrue(fit.success)
        self.assertEqual(fit.jackknife_replicates_used, 4)
        np.testing.assert_allclose(fit.standard_errors, expected_se, atol=1e-6)

    def test_group_constraints_are_preserved(self):
        profile = F4ProfileData(
            sources=("A", "B", "C"), features=("f1", "f2"),
            matrix=np.array([[-1., 0.], [1., 0.], [0., 0.]]),
            covariance=np.eye(6) * .01,
        )
        fit = fit_f4_profiles(
            profile, group_sums={"AB": (("A", "B"), .6), "C": (("C",), .4)},
            jackknife=False,
        )
        self.assertTrue(fit.success)
        self.assertTrue(fit.weights_identifiable)
        self.assertEqual(fit.free_weight_parameters, 1)
        np.testing.assert_allclose(fit.weights, [.3, .3, .4], atol=1e-6)

class GeneticMapUnitsTests(unittest.TestCase):
    def test_equivalent_plink_and_eigenstrat_maps_give_same_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            snp = Path(directory) / "map.snp"
            bim = Path(directory) / "map.bim"
            snp.write_text("s1 1 0.00 100 A G\ns2 1 0.04 200 A G\ns3 1 0.06 300 A G\ns4 2 0.00 100 A G\n")
            bim.write_text("1 s1 0 100 A G\n1 s2 4 200 A G\n1 s3 6 300 A G\n2 s4 0 100 A G\n")
            eigenstrat = read_snp(snp)
            plink = read_snp(bim, plink=True)
            np.testing.assert_allclose(eigenstrat["cm"], plink["cm"])
            np.testing.assert_array_equal(get_block_lengths(eigenstrat), [2, 1, 1])
            np.testing.assert_array_equal(get_block_lengths(plink), [2, 1, 1])


if __name__ == "__main__":
    unittest.main()
