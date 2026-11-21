#!/usr/bin/env python3
"""Targeted integration checks; does not regenerate figures.

Run from the repository root with ``python tests/check_convergence.py``.
The saved JSON compares the production resolution with doubled radial,
angular, and frequency sampling for the decisive magnification cases.
It also checks the low-frequency plot's infrared cutoff separately.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from gw_diffraction.inference import covariance_metrics, effective_snr
from gw_diffraction.physical import Geometry, SubhaloModel, delta2_from_physical_power, stellar_power
from gw_diffraction.waveoptics import projected_covariance
from scripts.run_calculations import band_grid, covariance_foreground_metrics, radial_response


def spectra(logq):
    geometry = Geometry()
    rf = float(geometry.fresnel_kpc(100.0))
    q = np.exp(logq)
    k = q / rf
    return {
        "dm": delta2_from_physical_power(q, SubhaloModel().power(k, geometry), rf),
        "stars": delta2_from_physical_power(q, np.full_like(k, stellar_power(geometry)), rf),
    }


def calculate(mu, resolution):
    f, w = band_grid(10.0, 100.0, n=64 * resolution)
    logq = np.linspace(-12.0, 7.5, 1500 * resolution)
    angles = (np.arange(56 * resolution) + 0.5) * np.pi / (56 * resolution)
    power = spectra(logq)
    rplus = radial_response(logq, angles, f, w, (1.0 / mu, 1.0))
    rminus = radial_response(logq, angles, f, w, (-1.0 / mu, 1.0))
    cov = {}
    rho_eff = effective_snr(50.0 * np.sqrt(mu), 50.0 * np.sqrt(mu))
    result = {}
    for key in ("dm", "stars"):
        cov[key] = projected_covariance(f, w, (1.0 / mu, 1.0), logq, power[key], angles)
        cov[key] += projected_covariance(f, w, (-1.0 / mu, 1.0), logq, power[key], angles)
        measured = covariance_metrics(cov[key], w, rho_eff)
        variance_radial = float(np.trapezoid(power[key] * (rplus + rminus), logq))
        result[key] = {
            k: measured[k]
            for k in ("fractional_rms", "deterministic_rms_snr", "kl_separation", "effective_modes")
        }
        result[key]["radial_to_covariance_variance_relative_error"] = (
            variance_radial / measured["fractional_rms"] ** 2 - 1.0
        )
    fg = covariance_foreground_metrics(cov["dm"], cov["stars"], w, rho_eff)
    result["foreground"] = {
        k: fg[k]
        for k in ("kl_separation", "weak_dm_fisher_fixed_star", "weak_dm_fisher_free_star_amplitude", "fisher_shape_cosine")
    }
    return result


def infrared_check(center, mu, minimum):
    # Preserve the original 1100-point grid's dlnq spacing while extending it.
    nq = round((7.5 - minimum) / (14.5 / 1099)) + 1
    logq = np.linspace(minimum, 7.5, nq)
    angles = (np.arange(56) + 0.5) * np.pi / 56
    f, w = band_grid(center / np.sqrt(10.0), center * np.sqrt(10.0), n=64)
    rr = radial_response(logq, angles, f, w, (1.0 / mu, 1.0))
    return {
        key: float(np.sqrt(np.trapezoid(power * rr, logq)))
        for key, power in spectra(logq).items()
    }


def main():
    results = {
        "description": "Targeted same-plane numerical convergence checks; no physical-model calibration",
        "base_resolution": {"nq": 1500, "nangle": 56, "nf": 64, "logq_min": -12.0, "logq_max": 7.5},
        "refined_resolution": {"nq": 3000, "nangle": 112, "nf": 128, "logq_min": -12.0, "logq_max": 7.5},
        "rho0": 50.0,
        "cases": [],
        "infrared_checks": [],
    }
    for mu in (1.0, 30.0, 100.0):
        base = calculate(mu, 1)
        refined = calculate(mu, 2)
        change = {
            name: {
                key: float(refined[name][key] / value - 1.0) if value else 0.0
                for key, value in values.items()
                if key != "radial_to_covariance_variance_relative_error"
            }
            for name, values in base.items()
        }
        results["cases"].append({"mu": mu, "base": base, "refined": refined, "relative_change": change})
        print(json.dumps({"mu": mu, "relative_change": change}), flush=True)
    for center, mu in ((0.003, 1.0), (0.003, 30.0), (0.01, 30.0), (np.sqrt(1000.0), 100.0)):
        vals = {str(minimum): infrared_check(center, mu, minimum) for minimum in (-7.0, -10.0, -12.0)}
        results["infrared_checks"].append({"center_hz": center, "mu": mu, "rms_by_logq_min": vals})
        print(json.dumps(results["infrared_checks"][-1]), flush=True)
    destination = ROOT / "results" / "convergence.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(str(destination), flush=True)


if __name__ == "__main__":
    main()
