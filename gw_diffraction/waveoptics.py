"""Perturbative diffraction about one nonsingular macroimage.

The response is Eq. (21) of Oguri & Takahashi (2022),
https://arxiv.org/abs/2204.00814, in dimensionless lens-plane wavenumbers
q = k * r_F(reference_hz).  The two ``eigenvalues`` are eigenvalues of
the lens mapping A, NOT directional magnifications.  Negative eigenvalues
are allowed (saddles/maxima); zero eigenvalues are not.

All frequency arrays are positive observed frequencies.  A response array
has frequency on its LAST axis.  Covariances are for the real vector
[Re(delta F(f_0)), ..., Re(delta F(f_N)), Im(delta F(f_0)), ...].
They are unweighted: detector weights enter only the nuisance projection.

The calculation assumes weak perturbations on the SAME thin lens plane
as a locally quadratic macro potential.  It does not implement a general
multiplane propagation calculation or nonlinear stellar microlensing.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def _frequencies(frequency_hz):
    frequency = np.asarray(frequency_hz, dtype=float)
    if frequency.ndim != 1 or frequency.size == 0:
        raise ValueError("frequency_hz must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("frequencies must be finite and strictly positive")
    return frequency


def _eigenvalues(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.shape != (2,):
        raise ValueError("eigenvalues must contain the two eigenvalues of A")
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues == 0):
        raise ValueError("macroimage eigenvalues must be finite and nonzero")
    return eigenvalues


def _weights(weights, number):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (number,):
        raise ValueError("weights must have one entry per frequency")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("frequency weights must be finite and strictly positive")
    return weights / np.sum(weights)


def response(frequency_hz, q, angle, eigenvalues, reference_hz=100.0):
    """Return the complex fractional Born response per convergence mode.

    ``q`` and ``angle`` follow NumPy broadcasting rules.  The output shape
    is their common broadcast shape followed by ``(len(frequency_hz),)``.
    For example, radial and angular grids can be passed as ``q[:, None]``
    and ``angles[None, :]``.  ``q`` is dimensionless and nonnegative.

    With b = cos(angle)^2/lambda_1 + sin(angle)^2/lambda_2 and
    s = q^2 b reference_hz/(2 f),

        H_A = 2 i f/(reference_hz q^2) [exp(-i s) - 1]
            = b sinc(s/(2 pi)) exp(-i s/2).

    The second expression is stable at q=0 and b=0 and correctly returns
    the real, frequency-independent geometric response H_A(q=0)=b.
    A perturbing potential evaluated at the macroimage has already been
    removed as an unobservable arrival-time shift.
    """
    frequency = _frequencies(frequency_hz)
    lam = _eigenvalues(eigenvalues)
    reference_hz = float(reference_hz)
    if not np.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError("reference_hz must be finite and strictly positive")
    q, angle = np.broadcast_arrays(
        np.asarray(q, dtype=float), np.asarray(angle, dtype=float)
    )
    if np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("q must be finite and nonnegative")
    if np.any(~np.isfinite(angle)):
        raise ValueError("angle must be finite")
    b = np.cos(angle) ** 2 / lam[0] + np.sin(angle) ** 2 / lam[1]
    s = q[..., None] ** 2 * b[..., None] * reference_hz / (2.0 * frequency)
    return b[..., None] * np.sinc(s / (2.0 * np.pi)) * np.exp(-0.5j * s)


def project_modes(H, frequency_hz, weights):
    """Remove fitted amplitude, constant phase, and arrival time.

    This is the orthogonal projection, in the real detector inner
    product Re(sum(weights * u.conj() * v)), off {1, i, i*f}.
    Real and imaginary parts therefore have DIFFERENT projections:
    a real constant is removed from Re(H), while a real linear function
    of f is removed from Im(H).  Complex regression on {1, f} would
    incorrectly remove a frequency-dependent amplitude mode.

    The last H axis must index frequencies.  Weights are normalized
    internally.  At least two distinct frequencies are required.
    """
    frequency = _frequencies(frequency_hz)
    weights = _weights(weights, frequency.size)
    H = np.asarray(H, dtype=complex)
    if H.ndim == 0 or H.shape[-1] != frequency.size:
        raise ValueError("the last H axis must match the frequency array")
    if np.any(~np.isfinite(H)):
        raise ValueError("H must be finite")
    centered_frequency = frequency - np.sum(weights * frequency)
    variance = np.sum(weights * centered_frequency**2)
    if variance <= 0:
        raise ValueError("projection needs at least two distinct frequencies")
    real_part = H.real - np.sum(weights * H.real, axis=-1, keepdims=True)
    imag_part = H.imag - np.sum(weights * H.imag, axis=-1, keepdims=True)
    slope = np.sum(weights * centered_frequency * imag_part, axis=-1) / variance
    imag_part = imag_part - slope[..., None] * centered_frequency
    return real_part + 1j * imag_part


def _trapezoid_weights(x):
    result = np.empty_like(x)
    result[0] = 0.5 * (x[1] - x[0])
    result[-1] = 0.5 * (x[-1] - x[-2])
    result[1:-1] = 0.5 * (x[2:] - x[:-2])
    return result


def projected_covariance(
    frequency_hz,
    weights,
    eigenvalues,
    logq,
    delta2,
    angles,
    reference_hz=100.0,
):
    """Integrate the real joint covariance of a homogeneous screen.

    Parameters
    ----------
    logq : 1-D array
        Strictly increasing natural logarithms of dimensionless q.
    delta2 : 1-D array
        Nonnegative convergence power per natural-log interval,
        Delta_kappa^2(q) = q^2 P_kappa(q)/(2 pi), on ``logq``.
        Changing q units requires changing P consistently; Delta^2 is
        invariant under that change.
    angles : 1-D array
        Equally weighted, uniformly sampled Fourier directions in
        radians, normally over [0, 2 pi) or [0, pi).

    Returns
    -------
    C : (2*Nf, 2*Nf) real array
        C = integral dlnq Delta^2(q) <x(q,theta) x(q,theta)^T>_theta,
        where x is [Re(Q H), Im(Q H)].  Radial integration uses trapezoid
        weights, and angular integration uses an equal-weight mean.

    The real convergence field makes the complex process noncircular.
    Accumulating the joint real/imaginary outer product preserves its
    pseudo-covariance; using only E[delta F delta F*] would lose it.
    Each radial block has at most 2048 rows to bound temporary memory.
    """
    frequency = _frequencies(frequency_hz)
    weights = _weights(weights, frequency.size)
    lam = _eigenvalues(eigenvalues)
    logq = np.asarray(logq, dtype=float)
    delta2 = np.asarray(delta2, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if (
        logq.ndim != 1
        or logq.size < 2
        or np.any(~np.isfinite(logq))
        or np.any(np.diff(logq) <= 0)
    ):
        raise ValueError("logq must be finite, one-dimensional, and increasing")
    if delta2.shape != logq.shape or np.any(~np.isfinite(delta2)) or np.any(delta2 < 0):
        raise ValueError("delta2 must be finite, nonnegative, and match logq")
    if angles.ndim != 1 or angles.size == 0 or np.any(~np.isfinite(angles)):
        raise ValueError("angles must be a nonempty finite one-dimensional array")
    q = np.exp(logq)
    radial_weights = _trapezoid_weights(logq) * delta2 / angles.size
    C = np.zeros((2 * frequency.size, 2 * frequency.size), dtype=float)
    for angle in angles:
        for start in range(0, logq.size, 2048):
            stop = min(start + 2048, logq.size)
            H = response(frequency, q[start:stop], angle, lam, reference_hz)
            H = project_modes(H, frequency, weights)
            joint = np.concatenate((H.real, H.imag), axis=-1)
            weighted = joint * np.sqrt(radial_weights[start:stop, None])
            C += weighted.T @ weighted
    return 0.5 * (C + C.T)


def scale_free_gain(slope, eigenvalues):
    """Return the projected RMS macro gain for P_kappa(q) proportional q^n.

    ``slope`` is n (the slope of P, NOT of Delta^2).  The result is

        G^2 = <|cos(theta)^2/lambda_1 + sin(theta)^2/lambda_2|^((2-n)/2)>.

    It assumes an unlimited power law, a common frequency band and
    weights, and the same projection as :func:`project_modes`.  The
    projected radial integral converges for -6 < n < 2.  Finite cutoffs
    and changes of spectral slope invalidate the exact scaling.

    For n=-2 the angular result is analytic:

        G^2 = 3/8 (lambda_1^-2 + lambda_2^-2)
              + 1/(4 lambda_1 lambda_2).
    """
    slope = float(slope)
    lam = _eigenvalues(eigenvalues)
    if not np.isfinite(slope) or not (-6.0 < slope < 2.0):
        raise ValueError("the projected scale-free integral needs -6 < slope < 2")
    a1, a2 = 1.0 / lam
    if slope == -2.0:
        return np.sqrt(0.375 * (a1 * a1 + a2 * a2) + 0.25 * a1 * a2)
    exponent = 0.5 * (2.0 - slope)
    turning_points = []
    if a1 * a2 < 0:
        turning_points = [np.arctan(np.sqrt(-a1 / a2))]
    integral, _ = quad(
        lambda theta: abs(a1 * np.cos(theta) ** 2 + a2 * np.sin(theta) ** 2)
        ** exponent,
        0.0,
        0.5 * np.pi,
        points=turning_points,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    return np.sqrt(2.0 * integral / np.pi)
