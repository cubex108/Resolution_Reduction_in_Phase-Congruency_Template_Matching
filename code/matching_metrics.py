"""Fixed-quantization, two-dimensional template matching and pixel-based metrics.

CCRE direction is always template X (the cumulative variable) -> search Y
(the categorical variable).  Its discrete approximation is
    (1 / bins) * sum_ij G[i,j] log(G[i,j] / (G_X[i] p_Y[j])),
where G[i,j] = P(X > i, Y = j).  The bin width 1/bins defines the
cumulative coordinate on [0, 1], rather than on arbitrary intensity units.
MI uses the same joint histogram and natural logarithms.

Each input is quantized once.  Candidate windows NEVER change bin edges.
For experiments using crops of precomputed feature maps, supply the ranges
of the complete optical and other-modality work maps through template_range
and search_range.  Defaults use the complete arrays passed to this function.
No candidate score or similarity map is min-max/max normalized.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Small-test fallback; large benchmarks should require Numba."""
        def decorate(function):
            return function
        return decorate


def _array2d(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _range(array, supplied, name):
    if supplied is None:
        return float(array.min()), float(array.max())
    if len(supplied) != 2:
        raise ValueError(f"{name} must contain (minimum, maximum)")
    low, high = float(supplied[0]), float(supplied[1])
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError(f"invalid {name}")
    return low, high


def quantize_fixed(image, bins=32, value_range=None):
    """Equal-width integer bins, with both endpoints clipped into [0,bins-1].

    The maximum belongs to the last bin.  Values beyond an explicitly supplied
    range are clipped, not used to alter that range.  A constant range maps all
    entries to zero.  Missing/nonfinite input is an error, never an empty bin.
    """
    image = _array2d(image, "image")
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    low, high = _range(image, value_range, "value_range")
    if high == low:
        return np.zeros(image.shape, dtype=np.int32)
    unit = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.ascontiguousarray(np.minimum(np.floor(unit * bins), bins - 1), dtype=np.int32)


@njit(cache=True, nogil=True)
def _maps_quantized(template, search, rows, cols, bins):
    height, width = template.shape
    n = height * width
    n_float = float(n)
    template_counts = np.zeros(bins, dtype=np.int64)
    for r in range(height):
        for c in range(width):
            template_counts[template[r, c]] += 1
    survival_counts = np.zeros(bins, dtype=np.int64)
    running = 0
    for i in range(bins - 1, -1, -1):
        survival_counts[i] = running
        running += template_counts[i]

    mi_map = np.zeros((len(rows), len(cols)), dtype=np.float64)
    ccre_map = np.zeros_like(mi_map)
    joint = np.zeros((bins, bins), dtype=np.int64)
    search_counts = np.zeros(bins, dtype=np.int64)
    for ri in range(len(rows)):
        row = rows[ri]
        for ci in range(len(cols)):
            col = cols[ci]
            joint[:] = 0
            search_counts[:] = 0
            for r in range(height):
                for c in range(width):
                    x = template[r, c]
                    y = search[row + r, col + c]
                    joint[x, y] += 1
                    search_counts[y] += 1
            mi = 0.0
            ccre = 0.0
            for j in range(bins):
                if search_counts[j] == 0:
                    continue
                cumulative = 0
                for i in range(bins - 1, -1, -1):
                    count = joint[i, j]
                    if count > 0:
                        mi += (count / n_float) * math.log(
                            count * n_float / (template_counts[i] * search_counts[j])
                        )
                    if cumulative > 0:
                        ccre += (cumulative / n_float) * math.log(
                            cumulative * n_float / (survival_counts[i] * search_counts[j])
                        )
                    cumulative += count
            # Mathematical nonnegativity; remove floating cancellation only.
            mi_map[ri, ci] = max(0.0, mi)
            ccre_map[ri, ci] = max(0.0, ccre / bins)
    return mi_map, ccre_map


def similarity_maps(template2D, search2D, step=2, bins=32, *,
                    template_range=None, search_range=None):
    """Return MI/CCRE maps and candidate top-left coordinates in image pixels.

    A Numba-compiled kernel computes both metrics in one pass over each
    candidate.  NUMBA_AVAILABLE is exposed so the runner can reject accidental
    slow fallback execution.  ``rows`` and ``cols`` are evenly spaced candidate
    top-left coordinates; an off-grid final boundary is not silently appended.
    """
    template = _array2d(template2D, "template2D")
    search = _array2d(search2D, "search2D")
    if not isinstance(step, (int, np.integer)) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
        raise ValueError("template is larger than search")
    trange = _range(template, template_range, "template_range")
    srange = _range(search, search_range, "search_range")
    tq = quantize_fixed(template, bins, trange)
    sq = quantize_fixed(search, bins, srange)
    rows = np.arange(0, search.shape[0] - template.shape[0] + 1, step, dtype=np.int64)
    cols = np.arange(0, search.shape[1] - template.shape[1] + 1, step, dtype=np.int64)
    mi, ccre = _maps_quantized(tq, sq, rows, cols, int(bins))
    return {
        "MI": mi, "CCRE": ccre, "rows": rows, "cols": cols,
        "quantization": {
            "bins": int(bins), "template_range": list(trange),
            "search_range": list(srange), "rule": "fixed_equal_width_clipped",
            "ccre_bin_width": 1.0 / int(bins), "log_base": "e",
        },
        "ccre_direction": "template_cumulative_to_search_categorical",
    }


def _bilinear(score, rows, cols, row, col):
    if row < rows[0] or row > rows[-1] or col < cols[0] or col > cols[-1]:
        return None
    if len(rows) == 1 or len(cols) == 1:
        return None
    i = min(max(int(np.searchsorted(rows, row, side="right")) - 1, 0), len(rows) - 2)
    j = min(max(int(np.searchsorted(cols, col, side="right")) - 1, 0), len(cols) - 2)
    wy = (row - rows[i]) / (rows[i + 1] - rows[i])
    wx = (col - cols[j]) / (cols[j + 1] - cols[j])
    return float((1 - wy) * ((1 - wx) * score[i, j] + wx * score[i, j + 1])
                 + wy * ((1 - wx) * score[i + 1, j] + wx * score[i + 1, j + 1]))


def summarize_map(score_map, rows, cols, true_top_left, template_shape, *,
                  peak_radius_px=10.0, ring_samples=16):
    """Decode a 2-D global maximum and summarize its geometry and prominence.

    ``rows``, ``cols`` and ``true_top_left`` are top-left pixel coordinates in
    the search image.  Ties use the first row-major maximum and are reported.
    Error is Euclidean pixels; success3 requires error < 3 AND nonflat data.
    Even a flat map retains the deterministic decoder's error, but its
    success3 is False and its invalid peak is explicit.

    Baseline-corrected peak drop = (peak - mean of a 16-point circular ring
    at radius 10 px) / (peak - median sidelobe).  Ring values are bilinearly
    interpolated.  Sidelobes exclude the disk of that same radius.  The drop
    is unavailable (None) when the whole ring is not observable or the
    denominator is degenerate.  It is never computed from clipped edge
    positions or a varying index radius.  PSR = (peak - mean sidelobe) /
    sample SD(sidelobe); unavailable values remain None, never zero.
    """
    score = _array2d(score_map, "score_map")
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    if rows.ndim != 1 or cols.ndim != 1 or score.shape != (len(rows), len(cols)):
        raise ValueError("coordinate vectors do not match score_map")
    if not np.isfinite(rows).all() or not np.isfinite(cols).all():
        raise ValueError("coordinates must be finite")
    if (np.diff(rows) <= 0).any() or (np.diff(cols) <= 0).any():
        raise ValueError("coordinates must be strictly increasing")
    truth = np.asarray(true_top_left, dtype=np.float64)
    shape = np.asarray(template_shape, dtype=np.float64)
    if truth.shape != (2,) or not np.isfinite(truth).all():
        raise ValueError("true_top_left must be two finite pixel coordinates")
    if shape.shape != (2,) or not np.isfinite(shape).all() or (shape <= 0).any():
        raise ValueError("template_shape must be two positive dimensions")
    if not math.isfinite(peak_radius_px) or peak_radius_px <= 0:
        raise ValueError("peak_radius_px must be positive")
    if not isinstance(ring_samples, int) or ring_samples < 4:
        raise ValueError("ring_samples must be an integer >= 4")

    pi, pj = np.unravel_index(np.argmax(score), score.shape)
    estimate = np.array([rows[pi], cols[pj]])
    error = float(np.linalg.norm(estimate - truth))
    peak = float(score[pi, pj])
    spread = float(peak - score.min())
    tolerance = max(abs(peak), abs(float(score.min())), 1e-300) * 1e-12
    degenerate = spread <= tolerance
    distances2 = (rows[:, None] - estimate[0]) ** 2 + (cols[None, :] - estimate[1]) ** 2
    sidelobes = score[distances2 > peak_radius_px ** 2]
    background = float(np.median(sidelobes)) if sidelobes.size else None
    psr = None
    if sidelobes.size > 1:
        standard_deviation = float(np.std(sidelobes, ddof=1))
        if standard_deviation > tolerance:
            psr = float((peak - np.mean(sidelobes)) / standard_deviation)
    ring = []
    for angle in np.arange(ring_samples) * (2.0 * math.pi / ring_samples):
        value = _bilinear(score, rows, cols,
                          estimate[0] + peak_radius_px * math.sin(angle),
                          estimate[1] + peak_radius_px * math.cos(angle))
        if value is not None:
            ring.append(value)
    drop = None
    if not degenerate and len(ring) == ring_samples and background is not None:
        denominator = peak - background
        if denominator > tolerance:
            drop = float((peak - np.mean(ring)) / denominator)
    return {
        "err": error, "success3": bool(error < 3.0 and not degenerate),
        "map_range": spread, "peak_value": peak,
        "estimated_top_left": estimate.tolist(), "true_top_left": truth.tolist(),
        "estimated_center": (estimate + (shape - 1.0) / 2.0).tolist(),
        "true_center": (truth + (shape - 1.0) / 2.0).tolist(),
        "truth_in_sampled_domain": bool(rows[0] <= truth[0] <= rows[-1]
                                        and cols[0] <= truth[1] <= cols[-1]),
        "degenerate": bool(degenerate), "peak_tie_count": int(np.sum(score == peak)),
        "baseline_corrected_peak_drop": drop, "PSR": psr,
        "peak_radius_px": float(peak_radius_px), "ring_samples": ring_samples,
        "ring_samples_available": len(ring), "sidelobe_n": int(sidelobes.size),
        "sidelobe_baseline": background,
    }
