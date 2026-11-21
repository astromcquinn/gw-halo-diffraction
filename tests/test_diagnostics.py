"""Scientific identities and explicit blind spots of systematics diagnostics."""

import unittest

import numpy as np

from gw_diffraction.diagnostics import (
    align_macroimage,
    image_detector_cross_difference,
    morse_factor,
    network_null_projector,
)


class MorsePhaseTests(unittest.TestCase):
    def test_minimum_saddle_and_maximum_are_distinct(self):
        f = np.array([-100.0, -20.0, 20.0, 100.0])
        np.testing.assert_array_equal(morse_factor(f, 0), np.ones(4))
        np.testing.assert_array_equal(morse_factor(f, 0.5), [1j, 1j, -1j, -1j])
        np.testing.assert_array_equal(morse_factor(f, 1), -np.ones(4))
        for index in [0, 0.5, 1]:
            np.testing.assert_array_equal(morse_factor(-f, index), morse_factor(f, index).conj())

    def test_morse_operator_preserves_real_time_domain_signal(self):
        # Odd sample count avoids a self-conjugate Nyquist bin. The explicit
        # M(0)=1 convention preserves an arbitrary real DC component.
        rng = np.random.default_rng(3814)
        strain = rng.normal(size=129)
        frequencies = np.fft.fftfreq(strain.size)
        spectrum = np.fft.fft(strain)
        for index in [0, 0.5, 1]:
            transformed = np.fft.ifft(spectrum * morse_factor(frequencies, index))
            np.testing.assert_allclose(transformed.imag, 0, atol=1e-14)

    def test_complete_macro_alignment_recovers_arbitrary_waveform(self):
        f = np.array([-100.0, -40.0, 40.0, 100.0])
        h = np.array([0.1 + 0.9j, -0.8 - 0.3j, -0.8 + 0.3j, 0.1 - 0.9j])
        for index in [0, 0.5, 1]:
            image = np.sqrt(30) * np.exp(2j * np.pi * f * 0.0123) * morse_factor(f, index) * h
            result = align_macroimage(image, f, 30, 0.0123, index)
            np.testing.assert_allclose(result, h, atol=1e-14)


class SystematicsNullTests(unittest.TestCase):
    def setUp(self):
        f = np.linspace(10, 100, 35)
        self.h = (f / 10) ** (-7 / 6) * np.exp(1j * f ** 0.2)
        self.f1 = 1 + 0.02 * np.sin(f / 8) + 0.01j * np.cos(f / 11)
        self.f2 = 1 - 0.01 * np.sin(f / 17) + 0.02j * np.cos(f / 5)
        self.g1 = 1 + 0.03j * (f / 100) ** 2
        self.g2 = 1 + 0.04 * np.sin(f / 30)

    def channels(self):
        return [self.g1 * self.f1 * self.h, self.g1 * self.f2 * self.h,
                self.g2 * self.f1 * self.h, self.g2 * self.f2 * self.h]

    def test_stable_calibration_and_arbitrary_diffraction_pass_cross_difference(self):
        value = image_detector_cross_difference(*self.channels())
        np.testing.assert_allclose(value, 0, atol=1e-14)

    def test_detector_specific_calibration_drift_fails_cross_difference(self):
        channels = self.channels()
        channels[1] *= 1 + 0.01j
        value = image_detector_cross_difference(*channels)
        self.assertGreater(np.linalg.norm(value), 1e-4)

    def test_common_epoch_error_is_exactly_degenerate_with_lensing(self):
        channels = self.channels()
        epoch_error = np.exp(0.2j * np.linspace(0, 1, self.h.size))
        channels[1] *= epoch_error
        channels[3] *= epoch_error
        np.testing.assert_allclose(image_detector_cross_difference(*channels), 0, atol=1e-14)
        # The image ratio has changed, even though the detector cross test passes.
        ratio_before = self.f1 / self.f2
        ratio_after = channels[0] / channels[1]
        self.assertGreater(np.linalg.norm(ratio_after - ratio_before), 0.1)

    def test_network_null_annihilates_scalar_lensing_but_not_generic_detector_error(self):
        response = np.array([[1, 0], [0, 1], [1, 1]], dtype=complex)
        projector = network_null_projector(response)
        source = np.array([1 + 0.3j, -0.4 + 0.8j])
        lensed = response @ source * (1.02 + 0.03j)
        np.testing.assert_allclose(projector @ lensed, 0, atol=1e-14)
        wrong_detector = lensed * np.array([1.01, 1, 1])
        self.assertGreater(np.linalg.norm(projector @ wrong_detector), 1e-4)
        # A common calibration transfer is an explicit null-stream blind spot.
        np.testing.assert_allclose(projector @ (lensed * (1.1 - 0.2j)), 0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
