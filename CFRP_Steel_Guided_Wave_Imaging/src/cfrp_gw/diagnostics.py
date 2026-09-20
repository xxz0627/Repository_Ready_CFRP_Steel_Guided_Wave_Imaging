from __future__ import annotations

import numpy as np
from scipy import sparse


def ray_support_density(A: sparse.csr_matrix, shape: tuple[int, int]) -> np.ndarray:
    """Count the number of measured paths intersecting each reconstruction cell."""
    count = np.asarray((A > 0).sum(axis=0)).ravel()
    if count.size != shape[0] * shape[1]:
        raise ValueError("shape is inconsistent with the ray matrix")
    return count.reshape(shape)


def _segment_intersects_rectangle(x1, y1, x2, y2, rectangle) -> bool:
    xmin, xmax, ymin, ymax = rectangle
    # Liang-Barsky clipping.
    dx = x2 - x1
    dy = y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - xmin, xmax - x1, y1 - ymin, ymax - y1)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1.0e-12:
            if qi < 0.0:
                return False
            continue
        t = qi / pi
        if pi < 0.0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return False
    return True


def affected_path_mask(path_table, rectangles_mm) -> np.ndarray:
    """Return paths crossing at least one supplied defect rectangle.

    Each rectangle is `(xmin, xmax, ymin, ymax)` in mm. Reference geometry is
    used only for retrospective diagnostics, not for image reconstruction.
    """
    rectangles = list(rectangles_mm)
    mask = np.zeros(len(path_table), dtype=bool)
    for index, row in enumerate(path_table.itertuples(index=False)):
        for rectangle in rectangles:
            if _segment_intersects_rectangle(row.Tx_X, row.Tx_Y, row.Rx_X, row.Rx_Y, rectangle):
                mask[index] = True
                break
    return mask


def affected_path_ratio(path_table, rectangles_mm) -> float:
    mask = affected_path_mask(path_table, rectangles_mm)
    return float(np.mean(mask)) if mask.size else 0.0


def angular_coverage_entropy(path_table, rectangles_mm, bins: int = 12) -> float:
    """Normalized Shannon entropy of path orientations crossing the supplied region."""
    mask = affected_path_mask(path_table, rectangles_mm)
    selected = path_table.loc[mask]
    if selected.empty:
        return 0.0
    dx = selected["Rx_X"].to_numpy(float) - selected["Tx_X"].to_numpy(float)
    dy = selected["Rx_Y"].to_numpy(float) - selected["Tx_Y"].to_numpy(float)
    angle = np.mod(np.arctan2(dy, dx), np.pi)
    count, _ = np.histogram(angle, bins=bins, range=(0.0, np.pi))
    probability = count[count > 0] / np.sum(count)
    entropy = -np.sum(probability * np.log(probability))
    return float(entropy / np.log(bins))
