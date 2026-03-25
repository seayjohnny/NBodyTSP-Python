"""Pairwise inter-particle force device functions.

Each entry defines a ``__device__ float compute_force_mag(...)`` function.
The kernel builder concatenates the chosen force model source before the
geometry template so that the template can call ``compute_force_mag`` by name.

The *piecewise_lj* and *smooth_lj* models expect additional ``#define``
constants to be prepended by the kernel builder:

* piecewise_lj: ``SLOPE_REPULSION``, ``MAG_ATTRACTION``, ``FORCE_CUTOFF_EXTRA``
* smooth_lj:    ``SMOOTH_P``, ``SMOOTH_Q``, ``SMOOTH_M``
"""

FORCE_MODELS = {

    # ------------------------------------------------------------------
    # Piecewise LJ approximation (original planar engine)
    # ------------------------------------------------------------------
    "piecewise_lj": r"""
__device__ float compute_force_mag(float dist, float eq_param, float strength) {
    /* Sign convention: positive = repulsive, negative = attractive.
       diff vector points AWAY from the other particle (pos[i] - pos[j]). */
    float eq_dist = eq_param;
    if (dist <= eq_dist) {
        /* Compressed: repel (positive) */
        return (eq_dist - dist) * SLOPE_REPULSION;
    } else if (dist < FORCE_CUTOFF_EXTRA) {
        /* Extended: attract (negative) */
        return -MAG_ATTRACTION / eq_dist;
    }
    return 0.0f;
}
""",

    # ------------------------------------------------------------------
    # Smooth (generalized) LJ with tunable exponents p, q
    # ------------------------------------------------------------------
    "smooth_lj": r"""
__device__ float compute_force_mag(float dist, float eq_param, float strength) {
    /* Sign convention: positive = repulsive, negative = attractive.
       diff vector points AWAY from the other particle (pos[i] - pos[j]). */
    float L = eq_param;
    float h = SMOOTH_M * powf(powf(SMOOTH_Q / SMOOTH_P, 1.0f / (SMOOTH_Q - SMOOTH_P)) * L, SMOOTH_P)
              / (1.0f - SMOOTH_P / SMOOTH_Q);
    float g = h * powf(L, SMOOTH_Q - SMOOTH_P);
    /* Negate: original formula assumed diff toward other particle */
    return -(g / powf(dist, SMOOTH_Q) - h / powf(dist, SMOOTH_P));
}
""",

    # ------------------------------------------------------------------
    # True Lennard-Jones 12-6 with per-pair sigma
    # ------------------------------------------------------------------
    "true_lj": r"""
__device__ float compute_force_mag(float dist, float eq_param, float strength) {
    float sigma = eq_param;
    float cutoff = sigma * 2.5f;
    if (dist > cutoff || dist < 0.005f) return 0.0f;
    float r_inv = 1.0f / fmaxf(dist, 0.01f);
    float sr = fminf(sigma * r_inv, 100.0f);
    float sr6 = sr * sr * sr * sr * sr * sr;
    float fmag = 24.0f * strength * r_inv * (2.0f * sr6 * sr6 - sr6);
    return fmaxf(-1000.0f, fminf(fmag, 1000.0f));
}
""",
}
