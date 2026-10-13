"""Numerical helpers for future LRO Diviner observation comparisons."""

from __future__ import annotations

import numpy as np

DIVINER_MISSING_TB = -9999.0


def temperature_metrics(
    observed_k: np.ndarray, modeled_k: np.ndarray
) -> dict[str, float | int]:
    """Compute paired metrics after removing documented missing values.

    This function does not perform spatial, temporal, footprint, channel, or
    emission-angle matching.  Those operations must occur before calling it.
    """
    observed = np.asarray(observed_k, dtype=np.float64)
    modeled = np.asarray(modeled_k, dtype=np.float64)
    if observed.shape != modeled.shape:
        raise ValueError("Observed and modeled temperature arrays must have equal shape.")
    valid = (
        np.isfinite(observed)
        & np.isfinite(modeled)
        & (observed != DIVINER_MISSING_TB)
    )
    if not np.any(valid):
        raise ValueError("No valid paired Diviner observations remain.")
    residual = modeled[valid] - observed[valid]
    return {
        "n": int(residual.size),
        "bias_k": float(np.mean(residual)),
        "mae_k": float(np.mean(np.abs(residual))),
        "rmse_k": float(np.sqrt(np.mean(residual**2))),
    }
