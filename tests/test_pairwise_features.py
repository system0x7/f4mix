from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from f4mix.core.profiles import (
    _helmert_basis,
    _pairwise_incidence,
    _pairwise_f4_block_stats,
    _project_pairwise_profiles,
    _vectorized_count_jackknife,
    _vectorized_influence_covariance,
    build_f4_profiles_for_targets,
    fit_f4_profiles,
)
from f4mix.core._admixpy.fstats import _count_jackknife, f4_stats
from f4mix.core._admixpy.genotypes import AfData


class PairwiseFeatureProjectionTests(unittest.TestCase):
    def test_helmert_basis_is_centered_and_orthonormal(self):
        basis = _helmert_basis(6)

        np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-14)

    def test_gls_projection_recovers_exact_pairwise_system_and_loo(self):
        nright = 4
        pairs = tuple(combinations(range(nright), 2))
        basis = _helmert_basis(nright)
        edge_design = _pairwise_incidence(nright, pairs) @ basis
        coordinates = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]])
        pairwise = coordinates @ edge_design.T

        raw_size = pairwise.size
        generator = np.arange(1, raw_size * raw_size + 1, dtype=float).reshape(
            raw_size, raw_size
        )
        covariance = generator @ generator.T + np.eye(raw_size)
        loo_coordinates = np.stack(
            [coordinates, coordinates + 0.01, coordinates - 0.02], axis=2
        )
        pairwise_loo = np.einsum("scb,ec->seb", loo_coordinates, edge_design)

        projected, projected_covariance, projected_loo = _project_pairwise_profiles(
            pairwise,
            covariance,
            pairwise_loo,
            edge_design,
            covariance_ridge=1e-8,
        )

        np.testing.assert_allclose(projected, coordinates, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            projected_loo, loo_coordinates, rtol=1e-9, atol=1e-9
        )
        self.assertEqual(projected_covariance.shape, (6, 6))
        np.testing.assert_allclose(
            projected_covariance, projected_covariance.T, atol=1e-12
        )

    def test_reordering_right_populations_only_rotates_coordinates(self):
        nright = 5
        pairs = tuple(combinations(range(nright), 2))
        basis = _helmert_basis(nright)
        incidence = _pairwise_incidence(nright, pairs)
        edge_design = incidence @ basis
        rng = np.random.default_rng(17)
        pairwise = rng.normal(size=(2, len(pairs)))
        raw_size = pairwise.size
        generator = rng.normal(size=(raw_size, raw_size))
        covariance = generator @ generator.T + np.eye(raw_size)
        projected, _, _ = _project_pairwise_profiles(
            pairwise,
            covariance,
            None,
            edge_design,
            covariance_ridge=1e-8,
        )
        centered_scores = projected @ basis.T

        permutation = np.array([3, 0, 4, 1, 2])
        old_edge = {pair: edge for edge, pair in enumerate(pairs)}
        edge_transform = np.zeros((len(pairs), len(pairs)))
        for new_edge, (first, second) in enumerate(pairs):
            original_first = int(permutation[first])
            original_second = int(permutation[second])
            sign = 1.0 if original_first < original_second else -1.0
            original_pair = tuple(sorted((original_first, original_second)))
            edge_transform[new_edge, old_edge[original_pair]] = sign
        full_transform = np.kron(np.eye(pairwise.shape[0]), edge_transform)
        permuted_pairwise = pairwise @ edge_transform.T
        permuted_covariance = full_transform @ covariance @ full_transform.T
        permuted_projected, _, _ = _project_pairwise_profiles(
            permuted_pairwise,
            permuted_covariance,
            None,
            edge_design,
            covariance_ridge=1e-8,
        )

        np.testing.assert_allclose(
            permuted_projected @ basis.T,
            centered_scores[:, permutation],
            rtol=1e-10,
            atol=1e-10,
        )


class VectorizedPairwiseKernelTests(unittest.TestCase):
    def test_vectorized_jackknife_matches_scalar_edge_cases(self):
        estimates = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [np.nan, 0.2, np.nan, 0.5],
                [np.nan, np.nan, np.nan, np.nan],
                [0.3, np.nan, np.nan, np.nan],
            ]
        )
        counts = np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [0.0, 12.0, 0.0, 18.0],
                [0.0, 0.0, 0.0, 0.0],
                [15.0, 0.0, 0.0, 0.0],
            ]
        )
        totals, loo, influence, contributes = _vectorized_count_jackknife(
            estimates, counts
        )

        for statistic in range(len(estimates)):
            reference = _count_jackknife(estimates[statistic], counts[statistic])
            np.testing.assert_allclose(
                totals[statistic], reference.total, equal_nan=True
            )
            np.testing.assert_allclose(
                loo[statistic], reference.loo, equal_nan=True
            )
            np.testing.assert_allclose(
                influence[statistic], reference.influence, equal_nan=True
            )
            np.testing.assert_array_equal(
                contributes[statistic], reference.contributes
            )

    def test_matches_generic_f4_engine_with_per_population_missingness(self):
        rng = np.random.default_rng(91)
        nsnps = 240
        targets = ("T1", "T2")
        sources = ("S1", "S2")
        right = ("R1", "R2", "R3", "R4")
        columns = (*targets, *sources, *right)
        values = rng.uniform(0.05, 0.95, size=(nsnps, len(columns)))
        values[rng.random(values.shape) < 0.08] = np.nan
        afs = pd.DataFrame(values, columns=columns)
        counts = pd.DataFrame(
            np.where(np.isfinite(values), 2.0, 0.0), columns=columns
        )
        snpfile = pd.DataFrame(
            {
                "CHR": np.repeat(np.arange(1, 25), 10),
                "cm": np.tile(np.linspace(0.0, 0.04, 10), 24),
                "POS": np.arange(nsnps) * 1_000,
            }
        )
        data = AfData(afs=afs, counts=counts, snpfile=snpfile)
        pairs = tuple(combinations(range(len(right)), 2))

        vectorized = _pairwise_f4_block_stats(
            data,
            targets,
            sources,
            right,
            pairs,
            blgsize=0.05,
            verbose=False,
        )
        combos = pd.DataFrame(
            [
                {
                    "pop1": source,
                    "pop2": target,
                    "pop3": right[first],
                    "pop4": right[second],
                }
                for target in targets
                for source in sources
                for first, second in pairs
            ]
        )
        reference = f4_stats(
            data,
            combos,
            unique_only=False,
            allsnps=True,
            blgsize=0.05,
            keep_blocks=False,
            keep_loo=True,
            covariance=True,
            verbose=False,
        )

        np.testing.assert_allclose(vectorized.est, reference.est, atol=1e-14)
        np.testing.assert_allclose(
            vectorized.loo, reference.loo, atol=1e-14, equal_nan=True
        )
        np.testing.assert_array_equal(vectorized.snp_counts, reference.snp_counts)
        np.testing.assert_allclose(
            vectorized.influence,
            reference.influence,
            atol=1e-14,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            vectorized.contributes, reference.contributes
        )
        per_target = len(sources) * len(pairs)
        for target_i in range(len(targets)):
            target_slice = slice(
                target_i * per_target, (target_i + 1) * per_target
            )
            covariance = _vectorized_influence_covariance(
                vectorized.influence[target_slice],
                vectorized.contributes[target_slice],
            )
            np.testing.assert_allclose(
                covariance,
                reference.cov[target_slice, target_slice],
                atol=1e-14,
                equal_nan=True,
            )


class PairwiseFeatureBuilderTests(unittest.TestCase):
    def test_builder_batches_all_targets_in_one_kernel_call(self):
        sources = ("S1", "S2")
        right = ("R1", "R2", "R3")
        pairs = tuple(combinations(range(len(right)), 2))
        basis = _helmert_basis(len(right))
        edge_design = _pairwise_incidence(len(right), pairs) @ basis
        coordinates = np.array([[0.2, -0.1], [-0.3, 0.4]])
        target_coordinates = np.stack([coordinates, coordinates * 2.0])
        estimates = np.einsum(
            "tsc,ec->tse", target_coordinates, edge_design
        ).reshape(-1)
        blocks = 8
        rng = np.random.default_rng(5)
        fake_stats = SimpleNamespace(
            est=estimates,
            loo=np.repeat(estimates[:, None], blocks, axis=1),
            snp_counts=np.full((len(estimates), blocks), 100.0),
            influence=rng.normal(size=(len(estimates), blocks)),
            contributes=np.ones((len(estimates), blocks), dtype=bool),
        )
        data = SimpleNamespace(
            afs=pd.DataFrame(
                {
                    "T": [0.1, 0.2, np.nan, 0.3],
                    "T2": [0.2, 0.3, 0.4, 0.5],
                }
            ),
            counts=pd.DataFrame(
                {
                    "T": [2.0, 2.0, 0.0, 2.0],
                    "T2": [2.0, 2.0, 2.0, 2.0],
                }
            ),
        )

        with patch(
            "f4mix.core.profiles._pairwise_f4_block_stats",
            return_value=fake_stats,
        ) as mocked:
            profiles = build_f4_profiles_for_targets(
                data,
                targets=["T", "T2"],
                sources=sources,
                right=right,
                outgroup="not_loaded_or_used",
                verbose=False,
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args.args[1], ("T", "T2"))
        self.assertEqual(mocked.call_args.args[2], sources)
        self.assertEqual(mocked.call_args.args[3], right)
        self.assertEqual(mocked.call_args.args[4], pairs)
        profile = profiles["T"]
        self.assertEqual(profile.features, ("right_contrast_1", "right_contrast_2"))
        np.testing.assert_allclose(profile.matrix, coordinates, atol=1e-10)
        np.testing.assert_allclose(profiles["T2"].matrix, coordinates * 2.0, atol=1e-10)
        self.assertEqual(profile.target_callable_snps, 3)
        np.testing.assert_allclose(profile.effective_f4_snps, [800.0, 800.0])

    def test_real_f4_builder_preserves_mixture_linearity_without_a_base(self):
        rng = np.random.default_rng(20260820)
        nsnps = 120
        source1 = rng.uniform(0.05, 0.95, nsnps)
        source2 = rng.uniform(0.05, 0.95, nsnps)
        target = 0.3 * source1 + 0.7 * source2
        afs = pd.DataFrame(
            {
                "T": target,
                "S1": source1,
                "S2": source2,
                "R1": rng.uniform(0.05, 0.95, nsnps),
                "R2": rng.uniform(0.05, 0.95, nsnps),
                "R3": rng.uniform(0.05, 0.95, nsnps),
                "R4": rng.uniform(0.05, 0.95, nsnps),
            }
        )
        counts = pd.DataFrame(2.0, index=afs.index, columns=afs.columns)
        snpfile = pd.DataFrame(
            {
                "CHR": np.repeat(np.arange(1, 13), 10),
                "cm": np.tile(np.linspace(0.0, 0.04, 10), 12),
                "POS": np.arange(nsnps) * 1_000,
            }
        )
        data = AfData(afs=afs, counts=counts, snpfile=snpfile)

        first = build_f4_profiles_for_targets(
            data,
            targets=["T"],
            sources=["S1", "S2"],
            right=["R1", "R2", "R3", "R4"],
            verbose=False,
        )["T"]
        reordered = build_f4_profiles_for_targets(
            data,
            targets=["T"],
            sources=["S1", "S2"],
            right=["R3", "R1", "R4", "R2"],
            verbose=False,
        )["T"]

        np.testing.assert_allclose(
            np.array([0.3, 0.7]) @ first.matrix, 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.array([0.3, 0.7]) @ reordered.matrix, 0.0, atol=1e-12
        )
        first_fit = fit_f4_profiles(first, jackknife=False)
        reordered_fit = fit_f4_profiles(reordered, jackknife=False)
        np.testing.assert_allclose(first_fit.weights, [0.3, 0.7], atol=1e-7)
        np.testing.assert_allclose(
            reordered_fit.weights, first_fit.weights, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
