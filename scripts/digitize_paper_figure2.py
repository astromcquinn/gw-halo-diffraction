#!/usr/bin/env python3
"""Digitize the *plotted* lowest-cutoff curve of Choi et al. (2026), Fig. 2.

Optional dependency: pip install pymupdf

This reads the vector polyline embedded in the inspected arXiv v1 PDF. It is
not the authors' numerical spectrum. Matplotlib has simplified the plotted
polyline; interpolated samples do not have independent precision.

The PDF is downloaded to a temporary directory unless --source-pdf is given.
It is never copied into the repository or output directory. The calibrations
below are deliberately version-locked by SHA256 and independently checked
against all visible major ticks on all three axes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np

try:
    import fitz
except ImportError as exc:
    raise SystemExit("Optional figure audit requires PyMuPDF: pip install pymupdf") from exc


SOURCE_URL = "https://pdfs.assets.alphaxiv.org/2607.11887v1.pdf"
SOURCE_SHA256 = "f1265b024f43ec56b07b03168df24cf720a451b455741d454bfadf6344209584"
SOURCE_VERSION = "arXiv:2607.11887v1, 13 July 2026"
PAGE_INDEX = 2


def is_black(drawing):
    color = drawing.get("color")
    return color is not None and max(abs(float(c)) for c in color) < 1e-7


def calibrate(drawings):
    """Fit logarithmic axes from independently identified major tick marks."""
    f_ticks, k_ticks, p_ticks = [], [], []
    for drawing in drawings:
        if not is_black(drawing):
            continue
        r = drawing["rect"]
        if r.width < 1e-3 and 2.3 < r.height < 2.5 and 77 < r.x0 < 265:
            if abs(r.y1 - 69.05146789550781) < 1e-3:
                f_ticks.append(r.x0)
            if abs(r.y0 - 228.6669158935547) < 1e-3:
                k_ticks.append(r.x0)
        if r.height < 1e-3 and 2.3 < r.width < 2.5 and abs(r.x1 - 77.46931457519531) < 1e-3:
            if 70 < r.y0 < 165:
                p_ticks.append(r.y0)
    f_ticks, k_ticks, p_ticks = (np.sort(np.asarray(x)) for x in (f_ticks, k_ticks, p_ticks))
    if not (len(f_ticks) == len(k_ticks) == len(p_ticks) == 8):
        raise ValueError("Major tick extraction failed; this script must be manually recalibrated")
    # These labels were independently read from a rendered page. Only their
    # locations are extracted from the PDF vector drawing primitives.
    f_logs = np.arange(-10, 5, 2, dtype=float)
    k_logs = np.arange(-2, 6, dtype=float)
    p_logs = np.arange(-5, -27, -3, dtype=float)
    f_fit = np.polyfit(f_ticks, f_logs, 1)
    k_fit = np.polyfit(k_ticks, k_logs, 1)
    p_fit = np.polyfit(p_ticks, p_logs, 1)
    residuals = {
        "frequency_tick_max_error_dex": float(np.max(abs(np.polyval(f_fit, f_ticks) - f_logs))),
        "wavenumber_tick_max_error_dex": float(np.max(abs(np.polyval(k_fit, k_ticks) - k_logs))),
        "power_tick_max_error_dex": float(np.max(abs(np.polyval(p_fit, p_ticks) - p_logs))),
        "d_log_frequency_d_log_wavenumber": float(f_fit[0] / k_fit[0]),
    }
    if max(residuals[k] for k in residuals if k.endswith("dex")) > 1e-4:
        raise ValueError("Axis tick calibration inaccurate")
    if abs(residuals["d_log_frequency_d_log_wavenumber"] - 2) > 1e-5:
        raise ValueError("Equivalent-frequency axis must scale as k_phys squared")
    return f_fit, k_fit, p_fit, residuals


def line_vertices(drawing):
    items = drawing["items"]
    if not items or any(item[0] != "l" for item in items):
        raise ValueError("Expected a straight-segment plotting polyline")
    points = [[items[0][1].x, items[0][1].y]]
    for item in items:
        if np.max(np.abs(np.asarray(points[-1]) - np.asarray([item[1].x, item[1].y]))) > 1e-4:
            raise ValueError("Discontinuous vector path")
        points.append([item[2].x, item[2].y])
    xy = np.asarray(points)
    if not np.all(np.diff(xy[:, 0]) > 0):
        raise ValueError("Expected increasing logarithmic wavenumber")
    return xy


def run(source_pdf, output_dir):
    digest = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"PDF hash differs from inspected version: {digest}. Do not reuse axis calibration blindly.")
    with fitz.open(source_pdf) as doc:
        drawings = doc[PAGE_INDEX].get_drawings()
    f_fit, k_fit, p_fit, validation = calibrate(drawings)
    candidates = []
    for index, drawing in enumerate(drawings):
        r, color = drawing["rect"], drawing.get("color")
        if (color is not None and max(color) - min(color) > 0.1
                and r.width > 160 and 70 < r.y0 < 80 and r.y1 < 170
                and len(drawing["items"]) > 10):
            xy = line_vertices(drawing)
            candidates.append((float(xy[-1, 1]), index, drawing, xy))
    if len(candidates) != 12:
        raise ValueError("Expected twelve colored cutoff curves in Fig. 2 upper panel")
    candidates.sort(key=lambda item: item[0])
    _, index, drawing, xy = candidates[0]
    # The topmost right endpoint is the least-suppressed, lowest mass-cutoff
    # curve; its dark-purple color is also the bottom of the figure's colorbar.
    expected_color = (0.18995, 0.07176, 0.23217)
    if np.max(abs(np.asarray(drawing["color"]) - expected_color)) > 1e-5:
        raise ValueError("Lowest-cutoff curve color does not match inspected figure")

    def interpolate(frequency, points=xy):
        x = (np.log10(frequency) - f_fit[1]) / f_fit[0]
        if np.any(x < points[0, 0]) or np.any(x > points[-1, 0]):
            raise ValueError("Requested frequency outside vector curve; no extrapolation allowed")
        y = np.interp(x, points[:, 0], points[:, 1])
        return 10**np.polyval(k_fit, x), 10**np.polyval(p_fit, y)

    f_samples = np.array([1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 30, 100, 300, 1000])
    k_samples, p_samples = interpolate(f_samples)
    endpoints = np.array([0.01, 100.0])
    _, p_end = interpolate(endpoints)
    ratio_fp = float(endpoints[1]*p_end[1]/(endpoints[0]*p_end[0]))
    n_fit = float(2*np.log(p_end[1]/p_end[0])/np.log(endpoints[1]/endpoints[0]))

    segment_slopes = []
    for start, stop in [(xy[-3], xy[-2]), (xy[-2], xy[-1])]:
        segment_slopes.append({
            "equivalent_f_range_hz": (10**np.polyval(f_fit, [start[0], stop[0]])).tolist(),
            "n_dlogP_dlogk": float((np.polyval(p_fit, stop[1]) - np.polyval(p_fit, start[1]))
                                    /(np.polyval(k_fit, stop[0]) - np.polyval(k_fit, start[0]))),
        })
    _, p_second = interpolate(endpoints, candidates[1][3])
    validation["next_cutoff_curve_relative_difference_at_0p01_100_hz"] = (p_second/p_end - 1).tolist()
    validation["independent_adjacent_segment_slopes"] = segment_slopes

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paper_figure2_digitization.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["equivalent_frequency_hz_zlens1", "physical_wavenumber_kpc_inverse", "P_kappa_dimensionless_as_plotted"])
        for f, k, p in zip(f_samples, k_samples, p_samples):
            writer.writerow([f"{f:.8g}", f"{k:.8g}", f"{p:.8g}"])
    metadata = {
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "source_sha256": digest,
        "figure": "Figure 2 upper panel, PDF page 3",
        "source_redshift": 2,
        "equivalent_frequency_axis_reference_lens_redshift": 1,
        "curve_selection": "Topmost right endpoint among 12 colored polylines; colorbar minimum log10(M_min/Msun)=-6",
        "selected_drawing_index": index,
        "selected_rgb": list(drawing["color"]),
        "vector_vertex_count": len(xy),
        "axis_fits": {"log10_f_from_x_pdf_points": f_fit.tolist(),
                      "log10_k_phys_from_x_pdf_points": k_fit.tolist(),
                      "log10_P_from_y_pdf_points": p_fit.tolist()},
        "validation": validation,
        "P_at_0p01_100_hz": p_end.tolist(),
        "high_k_slope_between_0p01_100_hz": n_fit,
        "ratio_fP_100_over_0p01": ratio_fp,
        "conditional_scale_free_RMS_ratio": float(np.sqrt(ratio_fp)),
        "limitations": [
            "Vector digitization of a simplified plotted curve, not original numerical data.",
            "Equivalent frequency labels map spatial k to f for a reference lens plane; P_kappa includes a line-of-sight integral.",
            "Dimensionless plotted P_kappa normalization is not converted into a measured fractional residual.",
            "RMS ratio assumes a power law over kernel support, identical scaled weighting/projection, and fixed geometry; it is not a complete LIGO/LISA forecast.",
        ],
    }
    (output_dir / "paper_figure2_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"csv": str(csv_path), "n": n_fit, "ratio_fP": ratio_fp,
                      "conditional_rms_ratio": float(np.sqrt(ratio_fp)), "validation": validation}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, help="Previously downloaded inspected PDF (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]/"results")
    args = parser.parse_args()
    if args.source_pdf is not None:
        run(args.source_pdf, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="gw_halo_figure_audit_") as tmp:
            target = Path(tmp)/"2607.11887v1.pdf"
            # Some public mirrors reject Python's default HTTP client while
            # serving the same public PDF to curl. Keep a normal CLI fallback.
            try:
                with urlopen(SOURCE_URL, timeout=30) as response:
                    target.write_bytes(response.read())
            except URLError:
                if shutil.which("curl") is None:
                    raise SystemExit("PDF download failed. Download the public PDF and supply --source-pdf.")
                subprocess.run(["curl", "--location", "--fail", "--silent", "--show-error",
                                "--max-time", "30", SOURCE_URL, "--output", str(target)], check=True)
            run(target, args.output_dir)


if __name__ == "__main__":
    main()
