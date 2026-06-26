"""
GNSS Out File Processor and Blunder Rejection

Handles parsing of gLAB output files, implements Ghilani §3.6 iterative Gaussian
outlier rejection, and computes PDOP-weighted coordinate means.
"""

import math
import pandas as pd
import numpy as np
from math_helpers import shapiro_wilk_p

def parse_out_file(filepath):
    """
    Parse a gLAB .out file to extract epochs from lines starting with 'OUTPUT'.
    Expected index mapping:
      parts[10]: PDOP
      parts[20]: Latitude (WGS84)
      parts[21]: Longitude (WGS84)
      parts[22]: Height (Ellipsoidal)
    """
    records = []
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            if not line.startswith("OUTPUT"):
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            try:
                records.append({
                    "PDOP":   float(parts[10]),
                    "Lat":    float(parts[20]),
                    "Lon":    float(parts[21]),
                    "Height": float(parts[22]),
                    # Using Longitude as X and Latitude as Y in planar coordinate operations
                    "X":      float(parts[21]),
                    "Y":      float(parts[20]),
                })
            except (ValueError, IndexError):
                continue
    if not records:
        raise ValueError(
            "No valid OUTPUT lines with Lat/Lon columns found in this file.\n"
            "Expected at least 23 whitespace-separated fields per OUTPUT line.")
    return pd.DataFrame(records)

def gaussian_iterative_rejection(df, log_cb=None):
    """
    Perform Ghilani §3.2-3.6 iterative blunder rejection.
    Rejects epochs with residues exceeding 2.9680 * standard deviation (99.7% confidence).
    """
    def _log(msg, tag="info"):
        if log_cb:
            log_cb(msg, tag)

    clean = df.copy()
    iterations = []

    for iteration in range(1, 10_000):
        n = len(clean)
        mx = float(clean["X"].mean())
        my = float(clean["Y"].mean())
        sx = float(clean["X"].std(ddof=1))
        sy = float(clean["Y"].std(ddof=1))

        e50_x  = 0.6745 * sx
        e50_y  = 0.6745 * sy
        e95_x  = 1.9600 * sx
        e95_y  = 1.9600 * sy
        e997_x = 2.9680 * sx
        e997_y = 2.9680 * sy

        vx = (clean["X"] - mx).abs()
        vy = (clean["Y"] - my).abs()
        flagged = (vx > e997_x) | (vy > e997_y)
        n_flagged = int(flagged.sum())

        _log(f"   Iter {iteration}: n={n}  "
             f"S\u1d39={sx:.7f}\u00b0  S\u1d38={sy:.7f}\u00b0  "
             f"E99.7\u03bb={e997_x:.7f}\u00b0  E99.7\u03c6={e997_y:.7f}\u00b0  "
             f"flagged={n_flagged}\n", "info")

        def zones(v, s):
            z = (v / s) if s > 0 else v * 0
            return (int((z < 1).sum()),
                    int(((z >= 1) & (z < 2)).sum()),
                    int(((z >= 2) & (z < 3)).sum()),
                    int((z >= 3).sum()))

        z1x, z2x, z3x, z4x = zones(vx, sx)
        z1y, z2y, z3y, z4y = zones(vy, sy)

        iterations.append({
            "iteration": iteration, "n_start": n, "n_flagged": n_flagged,
            "mean_lon": mx,   "mean_lat": my,
            "S_lon":    sx,   "S_lat":    sy,
            "E50_lon":  e50_x,  "E50_lat":  e50_y,
            "E95_lon":  e95_x,  "E95_lat":  e95_y,
            "E997_lon": e997_x, "E997_lat": e997_y,
            "mean_x": mx, "mean_y": my,
            "S_x": sx, "S_y": sy,
            "E50_x": e50_x, "E50_y": e50_y,
            "E95_x": e95_x, "E95_y": e95_y,
            "E997_x": e997_x, "E997_y": e997_y,
            "z1_x": z1x, "z2_x": z2x, "z3_x": z3x, "z4_x": z4x,
            "z1_y": z1y, "z2_y": z2y, "z3_y": z3y, "z4_y": z4y,
            "sw_p_x": shapiro_wilk_p(clean["X"].values),
            "sw_p_y": shapiro_wilk_p(clean["Y"].values),
        })

        if n_flagged == 0:
            _log(f"   Converged after {iteration} iteration(s).\n", "ok")
            break
        clean = clean[~flagged].copy()

    return clean, iterations

def weighted_mean(df):
    """Compute the coordinate weighted mean using 1 / PDOP^2 weight coefficient."""
    pdop = df["PDOP"].replace(0, float(df["PDOP"].median()))
    weights = 1.0 / (pdop ** 2)
    total = weights.sum()
    n = len(df)
    sx = float(df["X"].std(ddof=1))
    sy = float(df["Y"].std(ddof=1))
    mean_h = float(df["Height"].mean()) if "Height" in df.columns else 0.0
    return {
        "Lon":       float((df["X"] * weights).sum() / total),
        "Lat":       float((df["Y"] * weights).sum() / total),
        "Height":    mean_h,
        "X":         float((df["X"] * weights).sum() / total),
        "Y":         float((df["Y"] * weights).sum() / total),
        "X_std":     sx,
        "Y_std":     sy,
        "Lon_std":   sx,
        "Lat_std":   sy,
        "n_epochs":  n,
        "PDOP_mean": float(df["PDOP"].mean()),
        "S_mean_x":  sx / math.sqrt(n),
        "S_mean_y":  sy / math.sqrt(n),
        "S_mean_lon": sx / math.sqrt(n),
        "S_mean_lat": sy / math.sqrt(n),
    }
