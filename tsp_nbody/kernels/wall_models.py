"""Wall-force device functions.

Each entry defines ``__device__ float compute_wall_force_mag(...)`` with a
unified signature.  The geometry template calls this function and applies the
result along the appropriate direction vector.

Parameters
----------
dist_to_wall : float
    Signed distance from the particle to the wall surface.  Positive means the
    particle is safely inside the boundary; negative (or zero) means it has
    penetrated the wall.
wall_range : float
    Width of the force zone measured inward from the wall surface.
strength : float
    Scalar wall-force strength (pre-multiplied by the Python engine).

Returns
-------
float
    Magnitude of the repulsive wall force (always >= 0).  The geometry
    template multiplies by the appropriate direction.
"""

WALL_MODELS = {

    # ------------------------------------------------------------------
    # Linear spring wall (original planar engine)
    #
    # Force is zero when the particle is inside the boundary (dist > 0).
    # When the particle penetrates (dist < 0), force is proportional to
    # penetration depth.
    # ------------------------------------------------------------------
    "linear": r"""
__device__ float compute_wall_force_mag(float dist_to_wall, float wall_range, float strength) {
    if (dist_to_wall >= 0.0f) return 0.0f;
    return strength * (-dist_to_wall);
}
""",

    # ------------------------------------------------------------------
    # Inverse-square wall (torus / annular / planar-inverse_square)
    #
    # Force ramps up as the particle approaches the wall surface.
    # Active only when 0 < dist_to_wall < wall_range.
    # ------------------------------------------------------------------
    "inverse_square": r"""
__device__ float compute_wall_force_mag(float dist_to_wall, float wall_range, float strength) {
    if (dist_to_wall >= wall_range || dist_to_wall <= 0.0f) return 0.0f;
    float t = fmaxf(0.001f, dist_to_wall / wall_range);
    return strength / (t * t);
}
""",
}
