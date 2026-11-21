"""Morse alignment and idealized detector/image consistency identities.

These utilities verify algebraic diagnostics. They do not provide detection
significances or the noise covariance of nonlinear cross products. Detector
channels must already have consistent antenna/polarization response corrections;
raw strains recorded at different image arrival times need a network likelihood.
"""

from __future__ import annotations

import numpy as np


def morse_factor(frequencies, morse_index):
    """Return exp[-i*pi*n*sign(f)] for n=0, 1/2, or 1.

    Minimum: +1. Saddle: -i at positive f and +i at negative f. Maximum:
    -1 at all nonzero frequencies. This preserves conjugate symmetry and a
    real time-domain waveform. Geometric optics does not fix the f=0 limit;
    this routine explicitly uses the algebraic sign(0)=0 convention, M(0)=1.
    Gravitational-wave observations here use nonzero, band-passed frequencies.
    """
    frequency = np.asarray(frequencies, dtype=float)
    if not np.all(np.isfinite(frequency)):
        raise ValueError("frequencies must be finite")
    if morse_index not in (0, 0.5, 1):
        raise ValueError("morse_index must be 0, 0.5, or 1")
    if morse_index == 0:
        return np.ones_like(frequency, dtype=complex)
    if morse_index == 0.5:
        return np.where(frequency > 0, -1j, np.where(frequency < 0, 1j, 1.0 + 0j))
    return np.where(frequency == 0, 1.0, -1.0).astype(complex)


def align_macroimage(spectrum, frequencies, magnification=1.0, time_delay=0.0, morse_index=0):
    """Undo macro amplitude, a +2*pi*f*t Fourier delay, and Morse phase.

    ``magnification`` is the nonnegative absolute intensity magnification.
    It must be strictly positive. Source/polarization detector responses are
    outside this helper; undo those through a consistent network model.
    """
    if not np.isfinite(magnification) or magnification <= 0:
        raise ValueError("magnification must be finite and strictly positive")
    if not np.isfinite(time_delay):
        raise ValueError("time_delay must be finite")
    frequency = np.asarray(frequencies, dtype=float)
    signal = np.asarray(spectrum, dtype=complex)
    if signal.shape != frequency.shape or not np.all(np.isfinite(signal)):
        raise ValueError("spectrum and frequencies must have matching finite shapes")
    return (
        signal
        * np.exp(-2j * np.pi * frequency * time_delay)
        * np.conjugate(morse_factor(frequency, morse_index))
        / np.sqrt(magnification)
    )


def image_detector_cross_difference(x_ii, x_ij, x_ji, x_jj):
    """Return x[I,i]*x[J,j] - x[I,j]*x[J,i] for corrected scalar channels.

    For x[I,i]=g[I,i]*F[i]*h, this vanishes with time-stable detector gains
    g[I,i]=g[I,j], regardless of the image-dependent diffraction F[i]. It
    responds to detector-dependent calibration evolution. It is *also* blind
    to an arbitrary image/epoch-dependent multiplicative error common to all
    detectors; that error is exactly degenerate with a changed F[i].

    This is a noiseless identity and not a normalized detection statistic.
    Arrays broadcast using NumPy's usual rules. The division-free form avoids
    ill-defined raw ratios near waveform zeros but still needs a noise model.
    """
    a, b, c, d = np.broadcast_arrays(
        np.asarray(x_ii, dtype=complex), np.asarray(x_ij, dtype=complex),
        np.asarray(x_ji, dtype=complex), np.asarray(x_jj, dtype=complex),
    )
    if not all(np.all(np.isfinite(x)) for x in (a, b, c, d)):
        raise ValueError("channel values must be finite")
    return a * d - b * c


def network_null_projector(response):
    """Orthogonal projector off a known two-polarization network response.

    ``response`` has shape (number_of_detectors, 2) at a fixed frequency.
    Both response and detector data should first be noise-whitened if a
    noise-weighted null stream is desired. An arbitrary astrophysical scalar
    lensing transfer stays in the response subspace and is annihilated, while
    a detector-specific multiplicative error generally produces leakage.
    A common transfer to every detector remains invisible to this diagnostic.
    """
    matrix = np.asarray(response, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or matrix.shape[0] == 0:
        raise ValueError("response must have shape (number_of_detectors, 2)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("response must be finite")
    projector = np.eye(matrix.shape[0]) - matrix @ np.linalg.pinv(matrix)
    return (projector + projector.conj().T) / 2
