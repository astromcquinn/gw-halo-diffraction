"""Independent physical and analytic checks for the diffraction calculation."""

import unittest

import numpy as np

from gw_diffraction.waveoptics import (
    project_modes,
    projected_covariance,
    response,
    scale_free_gain,
)


class WaveOpticsTests(unittest.TestCase):
    def setUp(self):
        self.f = np.geomspace(30.0, 300.0, 31)
        self.w = np.ones(self.f.size) / self.f.size

    def test_fresnel_response_and_geometric_limit(self):
        # A=I must reproduce the independently written Fourier Born kernel.
        q = np.array([0.1, 1.0, 3.0])[:, None]
        theta = np.array([0.0, 0.4, 1.7, 2.8])[None, :]
        actual = response(self.f, q, theta, (1.0, 1.0))
        s = q[..., None] ** 2 * 100.0 / (2.0 * self.f)
        expected = np.expm1(-1j * s) / (-1j * s)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=2e-15)
        geom = response(self.f, 0.0, 0.0, (0.01, 1.0))
        np.testing.assert_allclose(geom, 100.0, atol=2e-14)

    def test_nuisance_projection_preserves_amplitude_slope(self):
        # Exact nuisance directions must vanish, but a real amplitude slope
        # is not an amplitude/phase/time nuisance and must survive.
        nuisance = 4.0 + 1j * (0.3 + 2.4 * self.f)
        np.testing.assert_allclose(project_modes(nuisance, self.f, self.w), 0, atol=3e-13)
        real_slope = project_modes(self.f, self.f, self.w)
        np.testing.assert_allclose(real_slope.real, self.f - np.mean(self.f))
        self.assertGreater(np.sum(self.w * np.abs(real_slope) ** 2), 100.0)
        H = response(self.f, np.geomspace(0.01, 10.0, 50), 0.3, (0.2, 1.0))
        QH = project_modes(H, self.f, self.w)
        np.testing.assert_allclose(np.sum(QH * self.w, axis=-1), 0, atol=3e-15)
        np.testing.assert_allclose(np.sum(QH.imag * self.f * self.w, axis=-1), 0, atol=2e-13)
        np.testing.assert_allclose(project_modes(QH, self.f, self.w), QH, atol=2e-15)

    def test_scale_free_angular_gain_matches_radial_calculation(self):
        # This checks a full radial+angular covariance against an independent
        # analytic integral.  Delta^2=1 corresponds to P proportional q^-2.
        f = np.geomspace(30.0, 300.0, 13)
        w = np.ones(f.size) / f.size
        logq = np.linspace(np.log(1e-4), np.log(1e3), 2201)
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        delta2 = np.ones(logq.size)
        diagonal_weight = np.concatenate((w, w))
        C0 = projected_covariance(f, w, (1.0, 1.0), logq, delta2, angles[:1])
        v0 = diagonal_weight @ np.diag(C0)
        for eigenvalues in [(0.1, 1.0), (-0.1, 1.0), (0.5, 0.5)]:
            C = projected_covariance(f, w, eigenvalues, logq, delta2, angles)
            measured = np.sqrt((diagonal_weight @ np.diag(C)) / v0)
            self.assertAlmostEqual(measured / scale_free_gain(-2, eigenvalues), 1.0, places=5)

    def test_covariance_psd_gauge_nulls_and_parity(self):
        logq = np.linspace(-5.0, 4.0, 301)
        angles = np.linspace(0, np.pi, 12, endpoint=False)
        C = projected_covariance(self.f, self.w, (0.2, 1.0), logq, np.ones(logq.size), angles)
        eig = np.linalg.eigvalsh(C)
        self.assertGreaterEqual(eig.min(), -1e-12 * eig.max())
        n = self.f.size
        amplitude = np.concatenate((self.w, np.zeros(n)))
        phase = np.concatenate((np.zeros(n), self.w))
        time = np.concatenate((np.zeros(n), self.w * self.f))
        for direction in (amplitude, phase, time):
            np.testing.assert_allclose(C @ direction, 0, atol=2e-12)
        positive = response(self.f, 1.3, 0.4, (0.2, 1.0))
        negative = response(self.f, 1.3, 0.4, (-0.2, -1.0))
        np.testing.assert_allclose(negative, -positive.conj(), atol=2e-15)

    def test_general_slope_and_input_guards(self):
        # For isotropic A=aI the analytic RMS is |a|^((n-2)/4).
        for n in [-4.0, -1.0, 0.0, 1.0]:
            self.assertAlmostEqual(scale_free_gain(n, (0.2, 0.2)), 0.2 ** ((n - 2) / 4), places=10)
        with self.assertRaises(ValueError):
            response(self.f, 1.0, 0.0, (0.0, 1.0))
        with self.assertRaises(ValueError):
            response([0.0, 1.0], 1.0, 0.0, (1.0, 1.0))
        with self.assertRaises(ValueError):
            scale_free_gain(-6.0, (1.0, 1.0))

    def test_unprojected_amplitude_phase_consistency(self):
        # A lens-mode identity, independent of any power spectrum:
        # |H(f)|^2 = [Re H(2f)]^2.  It must include negative-parity images.
        # This is an ensemble-variance constraint, not K^2+S^2=K(2f)^2
        # for the sum of Fourier modes in one observed lens realization.
        q = np.concatenate(([0.0], np.geomspace(1e-4, 20.0, 70)))[:, None]
        angles = np.linspace(0.0, np.pi, 19, endpoint=False)[None, :]
        for eigenvalues in [(1.0, 1.0), (0.01, 1.0), (-0.01, 1.0), (-0.5, -1.0)]:
            h = response(self.f, q, angles, eigenvalues)
            h2 = response(2.0 * self.f, q, angles, eigenvalues)
            np.testing.assert_allclose(np.abs(h) ** 2, h2.real ** 2, rtol=2e-12, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
