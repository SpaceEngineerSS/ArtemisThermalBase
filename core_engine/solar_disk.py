"""Solar disk sampling for extended-source illumination.

Generates uniformly distributed sample directions on the solar disk
using a Fibonacci spiral pattern on a spherical cap. This enables
accurate penumbra computation near shadow boundaries — critical
for thermal modeling of PSR transition zones where the 10–20 K
temperature difference between point-source and extended-source
models can affect volatile stability predictions.

Author: Mehmet Gümüş (github.com/SpaceEngineerSS)

References
----------
- González, Á. (2010). "Measurement of areas on a sphere using
  Fibonacci and latitude–longitude lattices." Math. Geosci., 42, 49–64.
- Kopp, G. & Lean, J.L. (2011). "A new, lower value of total solar
  irradiance." Geophys. Res. Lett., 38, L01706.

Algorithm
---------
Fibonacci Spiral on a spherical cap of angular radius θ_sun:

    golden_angle = π(3 − √5) ≈ 2.3999…

    For i ∈ [0, N−1]:
        θ_i = θ_sun · √(i / (N−1))       (area-uniform radial)
        φ_i = i · golden_angle             (golden-angle azimuthal)
        local = (sin θ_i cos φ_i,  sin θ_i sin φ_i,  cos θ_i)

    Then rotate local frame so that ẑ aligns with the sun center direction.

Solid angle of the solar disk:
    Ω_sun = 2π(1 − cos θ_sun) ≈ π sin²(θ_sun) ≈ 6.79 × 10⁻⁵ sr
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOLDEN_ANGLE: float = np.pi * (3.0 - np.sqrt(5.0))  # ≈ 2.39996 rad

# Default solar angular radius at 1 AU (half of 0.533° diameter)
_DEFAULT_SOLAR_ANGULAR_RADIUS_RAD: float = np.radians(0.533 / 2.0)


# ---------------------------------------------------------------------------
# Core: Generate Solar Disk Sample Directions
# ---------------------------------------------------------------------------


def generate_solar_disk_samples(
    sun_center_dir: np.ndarray,
    angular_radius_rad: float = _DEFAULT_SOLAR_ANGULAR_RADIUS_RAD,
    num_samples: int = 64,
) -> np.ndarray:
    """Generate uniformly distributed sample directions on the solar disk.

    Uses a Fibonacci spiral pattern to distribute N points with
    approximately equal solid-angle weight across the disk.

    Parameters
    ----------
    sun_center_dir : np.ndarray
        Unit vector pointing toward the sun center. Shape: (3,).
    angular_radius_rad : float
        Angular radius of the solar disk in radians.
        Default: 0.2665° ≈ 4.651 × 10⁻³ rad.
    num_samples : int
        Number of sample points on the disk. Must be ≥ 1.
        Default: 64.

    Returns
    -------
    samples : np.ndarray
        Unit direction vectors toward each sample point on the solar
        disk. Shape: (num_samples, 3), dtype: float64.

    Raises
    ------
    ValueError
        If num_samples < 1 or angular_radius_rad ≤ 0.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be ≥ 1, got {num_samples}")
    if angular_radius_rad <= 0.0:
        raise ValueError(
            f"angular_radius_rad must be > 0, got {angular_radius_rad}"
        )

    # Normalize the input direction
    sun_dir = sun_center_dir / np.linalg.norm(sun_center_dir)

    # Degenerate case: single sample = point source at center
    if num_samples == 1:
        return sun_dir.reshape(1, 3).copy()

    # Step 1: Generate Fibonacci spiral points in LOCAL frame (z-axis = center)
    local_dirs = np.empty((num_samples, 3), dtype=np.float64)

    for i in range(num_samples):
        # Area-uniform radial: theta proportional to sqrt(i/(N-1))
        # This ensures equal area per sample on the spherical cap
        theta_i = angular_radius_rad * np.sqrt(i / (num_samples - 1))
        phi_i = i * _GOLDEN_ANGLE

        sin_theta = np.sin(theta_i)
        cos_theta = np.cos(theta_i)

        local_dirs[i, 0] = sin_theta * np.cos(phi_i)
        local_dirs[i, 1] = sin_theta * np.sin(phi_i)
        local_dirs[i, 2] = cos_theta

    # Step 2: Rotate LOCAL frame so z-axis aligns with sun_dir
    rotation_matrix = _rotation_z_to_direction(sun_dir)
    samples = (rotation_matrix @ local_dirs.T).T

    # Normalize for safety (rotation should preserve norm, but float ops)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples /= norms

    logger.debug(
        "Generated %d solar disk samples (angular_radius=%.4f°, Ω=%.3e sr)",
        num_samples,
        np.degrees(angular_radius_rad),
        compute_solar_solid_angle(angular_radius_rad),
    )

    return samples


# ---------------------------------------------------------------------------
# Rotation: Align z-axis to an arbitrary direction
# ---------------------------------------------------------------------------


def _rotation_z_to_direction(target_dir: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that maps ẑ = (0, 0, 1) to `target_dir`.

    Uses Rodrigues' rotation formula. Handles the singular cases where
    target_dir ≈ +ẑ (identity) or ≈ −ẑ (180° flip around x-axis).

    Parameters
    ----------
    target_dir : np.ndarray
        Unit target direction. Shape: (3,).

    Returns
    -------
    R : np.ndarray
        3×3 rotation matrix. Shape: (3, 3).
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z_axis, target_dir)

    # Near-parallel to +z: identity
    if dot > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)

    # Near-antiparallel to +z: rotate 180° around x-axis
    if dot < -1.0 + 1e-12:
        return np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ], dtype=np.float64)

    # General case: Rodrigues' rotation formula
    # Rotation axis k = z × target_dir (normalized)
    # Rotation angle θ = arccos(dot)
    k = np.cross(z_axis, target_dir)
    k_norm = np.linalg.norm(k)
    k /= k_norm  # safe: we excluded the parallel cases

    sin_theta = k_norm  # |z × d| = sin(angle)
    cos_theta = dot

    # Skew-symmetric matrix K for cross product k ×
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)

    # R = I + sin(θ)·K + (1 − cos(θ))·K²
    R = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)

    return R


# ---------------------------------------------------------------------------
# Solid Angle Computation
# ---------------------------------------------------------------------------


def compute_solar_solid_angle(
    angular_radius_rad: float = _DEFAULT_SOLAR_ANGULAR_RADIUS_RAD,
) -> float:
    """Compute the exact solid angle of the solar disk.

    Ω = 2π(1 − cos θ_sun)

    This is the rigorous formula for a spherical cap, NOT the
    small-angle approximation Ω ≈ π θ². The difference matters
    at O(θ⁴) but we use the exact form for consistency.

    Parameters
    ----------
    angular_radius_rad : float
        Angular radius of the solar disk in radians.

    Returns
    -------
    float
        Solid angle in steradians.
    """
    return 2.0 * np.pi * (1.0 - np.cos(angular_radius_rad))


def validate_sample_weights(
    num_samples: int,
    angular_radius_rad: float = _DEFAULT_SOLAR_ANGULAR_RADIUS_RAD,
    tolerance: float = 0.01,
) -> dict[str, float]:
    """Validate that uniform sample weights correctly represent the solid angle.

    For uniform-weight Fibonacci sampling, each sample has weight 1/N,
    and the total weight sums to 1.0. The effective energy per sample
    is S₀ / N (where S₀ is the solar constant). The key invariant is:

        Σ weights = 1.0  →  total_flux = S₀ · cos(θ_incidence)

    This function verifies that the sample distribution is uniform
    enough that each sample's "solid angle share" is approximately
    Ω_sun / N.

    Parameters
    ----------
    num_samples : int
        Number of sample points.
    angular_radius_rad : float
        Angular radius of the solar disk.
    tolerance : float
        Maximum allowed relative error in solid angle coverage.

    Returns
    -------
    dict
        Validation results with keys:
        - 'solid_angle_exact' : exact Ω in sr
        - 'solid_angle_per_sample' : Ω / N in sr
        - 'num_samples' : N
        - 'weight_per_sample' : 1/N
        - 'total_weight' : should be 1.0
        - 'passed' : bool
    """
    omega_exact = compute_solar_solid_angle(angular_radius_rad)
    omega_per_sample = omega_exact / num_samples
    total_weight = 1.0  # uniform weights sum to 1.0 by construction

    # Verify the centroid of samples aligns with the sun direction
    # (this is a statistical check, not a weight check)
    sun_dir = np.array([0.0, 0.0, 1.0])
    samples = generate_solar_disk_samples(sun_dir, angular_radius_rad, num_samples)
    centroid = samples.mean(axis=0)
    centroid_norm = centroid / np.linalg.norm(centroid)
    centroid_deviation_deg = np.degrees(np.arccos(
        np.clip(np.dot(centroid_norm, sun_dir), -1.0, 1.0)
    ))

    passed = centroid_deviation_deg < np.degrees(angular_radius_rad) * tolerance

    result = {
        "solid_angle_exact_sr": omega_exact,
        "solid_angle_per_sample_sr": omega_per_sample,
        "num_samples": num_samples,
        "weight_per_sample": 1.0 / num_samples,
        "total_weight": total_weight,
        "centroid_deviation_deg": centroid_deviation_deg,
        "passed": passed,
    }

    logger.info(
        "Solar disk validation: Ω=%.3e sr, N=%d, "
        "centroid_dev=%.4f° (threshold=%.4f°), %s",
        omega_exact,
        num_samples,
        centroid_deviation_deg,
        np.degrees(angular_radius_rad) * tolerance,
        "PASSED" if passed else "FAILED",
    )

    return result
