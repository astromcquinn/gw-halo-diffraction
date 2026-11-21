"""Differential-image sensitivity and Gaussian covariance likelihoods.

The scalar-image model assumes the two observations have already been corrected
for their detector/network response.  It is an idealized sensitivity calculation,
not a detector population or discovery-rate forecast.

Real covariance matrices use the order ``[Re(z_1), ..., Re(z_n),
Im(z_1), ..., Im(z_n)]``.  Frequency weights include their integration-bin widths.
They are normalized to unit sum internally; total matched-filter SNR is supplied
separately.  A complex covariance alone is generally insufficient because the
wave-optics perturbation need not be a proper/circular complex random process.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _nonnegative_finite(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def effective_snr(rho1: Any, rho2: Any) -> Any:
    """Return ``rho1*rho2/hypot(rho1,rho2)`` for identical spectral shapes.

    This scalar relation assumes that both image SNR densities have the same
    normalized frequency dependence. Otherwise integrate :func:`difference_weight`
    pointwise before taking the square root. A zero-SNR image gives zero.
    Scalars and broadcastable NumPy arrays are supported.
    """
    first, second = np.broadcast_arrays(
        _nonnegative_finite(rho1, "rho1"),
        _nonnegative_finite(rho2, "rho2"),
    )
    lower, upper = np.minimum(first, second), np.maximum(first, second)
    ratio = np.divide(lower, upper, out=np.zeros_like(lower), where=upper > 0)
    answer = lower / np.sqrt(1.0 + ratio * ratio)
    return float(answer) if answer.ndim == 0 else answer


def difference_weight(w1: Any, w2: Any) -> Any:
    """Pointwise harmonic/parallel weight ``1/(1/w1 + 1/w2)``.

    If ``w_i=4 |h_i|**2/S_i``, the returned density weights the squared fractional
    difference between macro-aligned images. If either weight is zero, no
    differential information is available at that frequency.
    """
    first, second = np.broadcast_arrays(
        _nonnegative_finite(w1, "w1"), _nonnegative_finite(w2, "w2")
    )
    lower, upper = np.minimum(first, second), np.maximum(first, second)
    ratio = np.divide(lower, upper, out=np.zeros_like(lower), where=upper > 0)
    answer = lower / (1.0 + ratio)
    return float(answer) if answer.ndim == 0 else answer


def normalized_weights(weights: Any) -> np.ndarray:
    """Validate and normalize one-dimensional frequency-bin weights."""
    value = _nonnegative_finite(weights, "weights")
    if value.ndim != 1 or value.size == 0 or not np.any(value > 0):
        raise ValueError("weights must be a nonempty vector with positive total")
    # Scaling first avoids overflow if a caller passes large unnormalized weights.
    scaled = value / value.max()
    return scaled / scaled.sum()


def _real_covariance(covariance: Any, n: int) -> np.ndarray:
    original = np.asarray(covariance)
    if np.iscomplexobj(original):
        raise ValueError("supply the stacked real covariance, not a complex covariance")
    value = np.asarray(original, dtype=float)
    if value.shape != (2 * n, 2 * n) or not np.all(np.isfinite(value)):
        raise ValueError(f"covariance must be finite with shape {(2*n, 2*n)}")
    scale = float(np.max(np.abs(value))) if value.size else 0.0
    if scale and np.max(np.abs(value - value.T)) > 1e-10 * scale:
        raise ValueError("covariance must be symmetric")
    return (value + value.T) * 0.5


def nuisance_projection(frequencies: Any, weights: Any) -> np.ndarray:
    """Weighted real projector removing amplitude, phase, and relative delay.

    The three real nuisance columns represent ``1``, ``i``, and ``i*f``. A real
    frequency-dependent amplitude slope is deliberately *not* removed. The
    returned matrix acts on an unwhitened stacked real fractional waveform.
    """
    freq = _nonnegative_finite(frequencies, "frequencies")
    weight = normalized_weights(weights)
    if freq.ndim != 1 or freq.shape != weight.shape:
        raise ValueError("frequencies and weights must be vectors of equal length")
    n = freq.size
    centered = freq - np.dot(weight, freq)
    frequency_scale = float(np.sqrt(np.dot(weight, centered * centered)))
    time_column = centered / frequency_scale if frequency_scale > 0 else centered
    basis = np.zeros((2 * n, 3))
    basis[:n, 0] = 1.0
    basis[n:, 1] = 1.0
    basis[n:, 2] = time_column
    real_weight = np.concatenate((weight, weight))
    gram = basis.T @ (real_weight[:, None] * basis)
    return np.eye(2 * n) - basis @ np.linalg.pinv(gram, rcond=1e-12) @ (
        basis.T * real_weight[None, :]
    )


def projected_covariance(covariance: Any, frequencies: Any, weights: Any) -> np.ndarray:
    """Remove fitted macro amplitude/phase/delay from a real covariance."""
    weight = normalized_weights(weights)
    real_cov = _real_covariance(covariance, weight.size)
    projection = nuisance_projection(frequencies, weight)
    result = projection @ real_cov @ projection.T
    return (result + result.T) * 0.5


def whitened_covariance(covariance: Any, weights: Any, rho_eff: float = 1.0) -> np.ndarray:
    """Convert a fractional residual covariance to unit-noise coordinates.

    ``rho_eff`` is the differential SNR (not the quadrature-summed image SNR).
    Apply :func:`projected_covariance` first when macro nuisance parameters have
    not yet been removed. Null nuisance directions have zero signal eigenvalues
    and can be left in the returned full matrix.
    """
    rho = _nonnegative_finite(rho_eff, "rho_eff")
    if rho.ndim != 0:
        raise ValueError("rho_eff must be scalar")
    weight = normalized_weights(weights)
    real_cov = _real_covariance(covariance, weight.size)
    root_weight = float(rho) * np.sqrt(np.concatenate((weight, weight)))
    return root_weight[:, None] * real_cov * root_weight[None, :]


def covariance_eigenvalues(covariance: Any, weights: Any, rho_eff: float = 1.0) -> np.ndarray:
    """Return positive eigenvalues of the noise-whitened signal covariance.

    Tiny negative numerical eigenvalues, relative to the largest absolute
    eigenvalue, are clipped. A materially indefinite covariance raises an error.
    """
    matrix = whitened_covariance(covariance, weights, rho_eff)
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    if scale and eigenvalues.min() < -1e-10 * scale:
        raise ValueError("signal covariance must be positive semidefinite")
    return eigenvalues[eigenvalues > 0]


def _lambda_minus_log1p(eigenvalues: np.ndarray) -> np.ndarray:
    """Stable ``lambda-log(1+lambda)`` even far below machine precision."""
    value = np.asarray(eigenvalues, dtype=float)
    small = value < 1e-3
    answer = np.empty_like(value)
    x = value[small]
    answer[small] = x * x * (
        0.5 + x * (-1.0 / 3.0 + x * (0.25 + x * (-0.2 + x * (1.0 / 6.0 - x / 7.0))))
    )
    x = value[~small]
    answer[~small] = x - np.log1p(x)
    return answer


def eigenvalue_metrics(eigenvalues: Any) -> dict[str, Any]:
    """Summarize known-realization and covariance-only information.

    ``deterministic_rms_snr=sqrt(sum(eta))`` is the ensemble RMS residual SNR
    *if the realization were known*. ``kl_separation=sqrt(sum(eta-log1p(eta)))``
    is ``sqrt(2 D_KL(signal+noise || noise))``. It is an information measure,
    **not a calibrated Gaussian discovery significance**. ``weak_z`` is the
    weak-covariance optimal-score approximation and requires all ``eta << 1``.
    """
    eta = _nonnegative_finite(eigenvalues, "eigenvalues")
    if eta.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    total = float(eta.sum())
    squared = float(np.dot(eta, eta))
    return {
        "eigenvalues": eta.tolist(),
        "deterministic_rms_snr": math.sqrt(total),
        "kl_separation": math.sqrt(float(_lambda_minus_log1p(eta).sum())),
        "weak_z": math.sqrt(0.5 * squared),
        "effective_modes": total * total / squared if squared else 0.0,
    }


def covariance_metrics(covariance: Any, weights: Any, rho_eff: float = 1.0) -> dict[str, Any]:
    """Return JSON-serializable sensitivity metrics for a real covariance."""
    result = eigenvalue_metrics(covariance_eigenvalues(covariance, weights, rho_eff))
    # Calculate fractional RMS directly so rho_eff=0 is defined without 0/0.
    weight = normalized_weights(weights)
    real_cov = _real_covariance(covariance, weight.size)
    variance = float(np.dot(np.concatenate((weight, weight)), np.diag(real_cov)))
    if variance < 0:
        raise ValueError("fractional covariance has negative weighted variance")
    result["fractional_rms"] = math.sqrt(variance)
    return result


def stack_kl_separation(kl_separation: float, n_events: int) -> float:
    """Information-equivalent separation for independent identical events.

    Kullback-Leibler divergences add, so their square-root measure grows as
    ``sqrt(n_events)``. For weak stochastic residuals this corresponds to an
    RMS-amplitude threshold scaling as ``n_events**(-1/4)``, not ``N**(-1/2)``.
    """
    separation = _nonnegative_finite(kl_separation, "kl_separation")
    if separation.ndim != 0:
        raise ValueError("kl_separation must be scalar")
    if isinstance(n_events, bool) or int(n_events) != n_events or n_events < 0:
        raise ValueError("n_events must be a nonnegative integer")
    return float(separation) * math.sqrt(n_events)


def fold_pair_sensitivity(
    rho0: float,
    mu: float,
    epsilon0: float,
    gain_minimum: float = 1.0,
    gain_saddle: float = 1.0,
    correlation: float = 0.0,
) -> dict[str, float]:
    """Sensitivity of an ideal equal-magnification minimum/saddle image pair.

    ``mu`` is each image's absolute intensity magnification, not their sum.
    ``epsilon0`` is a *per-image* baseline projected fractional RMS. The two
    gains multiply that fractional RMS, beyond the macro amplitude boost.
    ``correlation`` is the weighted real correlation of the projected residuals.
    This is a sensitivity requirement, not a claim about its event rate or a
    prediction of the gains from a halo model.
    """
    vals = [rho0, mu, epsilon0, gain_minimum, gain_saddle]
    for name, value in zip(
        ["rho0", "mu", "epsilon0", "gain_minimum", "gain_saddle"], vals
    ):
        if _nonnegative_finite(value, name).ndim != 0:
            raise ValueError(f"{name} must be scalar")
    if not np.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be finite and in [-1, 1]")
    rho_image = float(rho0) * math.sqrt(float(mu))
    rho_delta = effective_snr(rho_image, rho_image)
    # Sum of squares form remains nonnegative close to perfect correlation.
    delta_variance = (gain_minimum - gain_saddle) ** 2 + (
        2.0 * (1.0 - correlation) * gain_minimum * gain_saddle
    )
    epsilon_delta = float(epsilon0) * math.sqrt(delta_variance)
    return {
        "rho_image": rho_image,
        "rho_delta": rho_delta,
        "epsilon_delta": epsilon_delta,
        "deterministic_rms_snr": rho_delta * epsilon_delta,
    }
