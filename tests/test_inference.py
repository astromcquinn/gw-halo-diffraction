"""Tests of statistical identities, not a detector-population forecast."""

import unittest

import numpy as np

from gw_diffraction.inference import (
    covariance_eigenvalues,
    covariance_metrics,
    difference_weight,
    effective_snr,
    eigenvalue_metrics,
    fold_pair_sensitivity,
    nuisance_projection,
    projected_covariance,
    stack_kl_separation,
)


class ImageDifferenceTests(unittest.TestCase):
    def test_equal_and_unequal_snr(self):
        self.assertAlmostEqual(effective_snr(100, 100), 100 / np.sqrt(2))
        self.assertAlmostEqual(effective_snr(300, 400), 240)
        self.assertAlmostEqual(effective_snr(1e10, 30), 30)
        self.assertEqual(effective_snr(0, 10), 0)
        self.assertEqual(effective_snr(0, 0), 0)

    def test_harmonic_frequency_weight_and_disjoint_support(self):
        np.testing.assert_allclose(difference_weight([4, 0, 9], [4, 5, 1]), [2, 0, 0.9])
        # Total image SNRs alone can overstate differential information when
        # their spectral support differs.
        self.assertEqual(np.sum(difference_weight([100, 0], [0, 100])), 0)
        self.assertGreater(effective_snr(10, 10), 0)

    def test_common_source_elimination_matches_harmonic_weight(self):
        w1, w2 = 3.0, 11.0
        x1, x2 = 1.2 + 0.3j, -0.4 + 1.8j
        source_fit = (w1 * x1 + w2 * x2) / (w1 + w2)
        minimized_cost = w1 * abs(x1 - source_fit) ** 2 + w2 * abs(x2 - source_fit) ** 2
        differential_cost = difference_weight(w1, w2) * abs(x1 - x2) ** 2
        self.assertAlmostEqual(minimized_cost, differential_cost)

    def test_fold_independent_fluctuations_cancel_sqrt_two_penalty(self):
        value = fold_pair_sensitivity(50, 100, 1e-3)
        self.assertAlmostEqual(value["rho_image"], 500)
        self.assertAlmostEqual(value["epsilon_delta"], np.sqrt(2) * 1e-3)
        self.assertAlmostEqual(value["deterministic_rms_snr"], 0.5)
        common = fold_pair_sensitivity(50, 100, 1e-3, correlation=1)
        self.assertEqual(common["epsilon_delta"], 0)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.frequencies = np.array([30.0, 60.0, 100.0, 200.0, 300.0])
        self.weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        self.n = self.frequencies.size

    def test_remove_only_real_amplitude_imaginary_phase_and_delay(self):
        projection = nuisance_projection(self.frequencies, self.weights)
        n = self.n
        amplitude = np.r_[np.ones(n), np.zeros(n)]
        phase = np.r_[np.zeros(n), np.ones(n)]
        delay = np.r_[np.zeros(n), self.frequencies]
        for nuisance in [amplitude, phase, delay]:
            np.testing.assert_allclose(projection @ nuisance, 0, atol=2e-12)
        real_slope = np.r_[self.frequencies, np.zeros(n)]
        expected = np.r_[
            self.frequencies - np.average(self.frequencies, weights=self.weights), np.zeros(n)
        ]
        np.testing.assert_allclose(projection @ real_slope, expected, atol=2e-12)
        np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)

    def test_nuisance_only_covariance_leaves_zero_signal(self):
        n = self.n
        amplitude = np.r_[np.ones(n), np.zeros(n)]
        phase = np.r_[np.zeros(n), np.ones(n)]
        covariance = np.outer(amplitude, amplitude) + np.outer(phase, phase)
        result = projected_covariance(covariance, self.frequencies, self.weights)
        np.testing.assert_allclose(result, 0, atol=1e-13)

    def test_weighted_self_adjoint_projection(self):
        p = nuisance_projection(self.frequencies, self.weights)
        w = np.diag(np.r_[self.weights, self.weights])
        np.testing.assert_allclose(w @ p, p.T @ w, atol=1e-12)


class CovarianceLikelihoodTests(unittest.TestCase):
    def test_whitening_and_deterministic_rms_normalization(self):
        # Two frequencies with isotropic real/imaginary variance 0.5*eps^2.
        eps, rho = 0.01, 100.0
        covariance = 0.5 * eps**2 * np.eye(4)
        metrics = covariance_metrics(covariance, [1, 1], rho)
        np.testing.assert_allclose(metrics["eigenvalues"], [0.25] * 4)
        self.assertAlmostEqual(metrics["fractional_rms"], eps)
        self.assertAlmostEqual(metrics["deterministic_rms_snr"], rho * eps)

    def test_kl_exact_and_weak_signal_limit(self):
        eta = np.array([0.2, 1.0, 3.0])
        metrics = eigenvalue_metrics(eta)
        expected = np.sqrt(np.sum(eta - np.log1p(eta)))
        self.assertAlmostEqual(metrics["kl_separation"], expected)
        weak = eigenvalue_metrics(np.array([1e-12, 3e-12]))
        self.assertGreater(weak["kl_separation"], 0)
        self.assertAlmostEqual(weak["kl_separation"] / weak["weak_z"], 1, places=10)

    def test_weak_stochastic_metric_scales_as_amplitude_squared(self):
        first = eigenvalue_metrics([1e-10, 2e-10])
        # Doubling fractional amplitude multiplies covariance by four.
        second = eigenvalue_metrics([4e-10, 8e-10])
        self.assertAlmostEqual(second["deterministic_rms_snr"] / first["deterministic_rms_snr"], 2)
        self.assertAlmostEqual(second["weak_z"] / first["weak_z"], 4)
        self.assertAlmostEqual(second["kl_separation"] / first["kl_separation"], 4, places=8)
        self.assertEqual(stack_kl_separation(0.2, 100), 2.0)

    def test_reject_materially_indefinite_or_complex_covariance(self):
        with self.assertRaises(ValueError):
            covariance_eigenvalues(np.diag([1.0, -0.1]), [1])
        with self.assertRaises(ValueError):
            covariance_metrics(np.eye(2, dtype=complex), [1])

    def test_zero_signal(self):
        metrics = covariance_metrics(np.zeros((4, 4)), [1, 2], 200)
        self.assertEqual(metrics["eigenvalues"], [])
        self.assertEqual(metrics["deterministic_rms_snr"], 0)
        self.assertEqual(metrics["kl_separation"], 0)
        self.assertEqual(metrics["weak_z"], 0)


if __name__ == "__main__":
    unittest.main()
