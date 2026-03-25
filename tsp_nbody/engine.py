"""Composable PhysicsEngine — single unified engine replacing the three old
geometry-specific engines (planar, torus, annular).

Configured at init time with geometry, force model, wall model, and compute
backend.  Uses the kernel_builder module to compile CUDA kernels on-demand
and provides NumPy-vectorised CPU fallbacks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tsp_nbody.base_engine import BasePhysicsEngine

# ---------------------------------------------------------------------------
# CuPy (soft dependency)
# ---------------------------------------------------------------------------
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

TAU = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Deterministic PRNG (port of JS mulberry32 — critical for reproducibility)
# ---------------------------------------------------------------------------

def _mulberry32(seed: int):
    """Seeded PRNG matching the JS mulberry32 implementation."""
    state = [seed & 0xFFFFFFFF]

    def next_val():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t = (t ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296.0

    return next_val


class PhysicsEngine(BasePhysicsEngine):
    """Composable N-Body TSP physics engine.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Original city coordinates.
    geometry : str
        ``"planar"`` | ``"torus"`` | ``"annular"``.
    force_model : str
        ``"piecewise_lj"`` | ``"smooth_lj"`` | ``"true_lj"``.
    wall_model : str
        ``"linear"`` | ``"inverse_square"``.
    backend : str
        ``"gpu"`` (auto-fallback to CPU if CuPy unavailable) | ``"cpu"``.
    options : dict, optional
        Geometry / model-specific parameters (see module docstring).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        coords: np.ndarray,
        geometry: str = "planar",
        force_model: str = "true_lj",
        wall_model: str = "inverse_square",
        backend: str = "gpu",
        options: Optional[dict] = None,
    ):
        opts = options or {}

        # Validate enums
        if geometry not in ("planar", "torus", "annular"):
            raise ValueError(f"Unknown geometry {geometry!r}")
        if force_model not in ("piecewise_lj", "smooth_lj", "true_lj"):
            raise ValueError(f"Unknown force_model {force_model!r}")
        if wall_model not in ("linear", "inverse_square"):
            raise ValueError(f"Unknown wall_model {wall_model!r}")

        self.geometry = geometry
        self.force_model = force_model
        self.wall_model = wall_model

        # Backend selection (auto-fallback)
        self.use_gpu = (backend == "gpu") and _CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

        # Store original coordinates (always numpy on CPU)
        self.original_coords = np.asarray(coords, dtype=np.float32)
        self.n_cities = len(coords)

        # ---- Common physics parameters ----
        self.dt: float = opts.get("dt", 0.004)
        self.damping: float = opts.get("damping", 0.97)
        self.max_speed: float = opts.get("max_speed", 2.5)
        self.lj_strength: float = opts.get("lj_strength", 1.0)
        self.wall_range_frac: float = opts.get("wall_range_frac", 0.05)
        self.collapse_rate: float = opts.get("collapse_rate",
                                              opts.get("shrink_rate",
                                                       opts.get("dr", 0.01)))
        self.epsilon: float = opts.get("epsilon", 0.08)
        self.wall_strength: float = opts.get("wall_strength", 20000.0)
        self.wall_force_scale: float = opts.get("wall_force_scale", 8.0)

        # ---- Per-particle mass (scalar broadcasts to array in initialize_physics) ----
        self._mass_scalar: float = opts.get("mass", 1.0)
        self._mass_per_particle: Optional[np.ndarray] = None  # set if array provided
        mass_opt = opts.get("mass", 1.0)
        if isinstance(mass_opt, (list, np.ndarray)):
            self._mass_per_particle = np.asarray(mass_opt, dtype=np.float32)
            self._mass_scalar = 1.0  # unused when per-particle is set
        else:
            self._mass_scalar = float(mass_opt)
        self.mass_arr = None  # (N,) array, set in initialize_physics

        # ---- Force-model-specific parameters ----
        self.slope_repulsion: float = opts.get("slope_repulsion", 50.0)
        self.mag_attraction: float = opts.get("mag_attraction", 0.5)
        self.force_cutoff_extra: float = opts.get("force_cutoff_extra", 0.1)
        self.p_exp: float = opts.get("p", 6.0)
        self.q_exp: float = opts.get("q", 12.0)
        self.m_coeff: float = opts.get("m", -0.05)

        # ---- Torus-specific ----
        self.perturbation: float = opts.get("perturbation", 0.50)
        self.embed_mode: str = opts.get("embed_mode", "flat")
        self.sigma_arc_scale: float = opts.get("sigma_arc_scale", 0.35)
        self.circle_lj_scale: float = opts.get("circle_lj_scale", 0.3)
        self.circle_damping: float = opts.get("circle_damping", 0.95)
        self.max_omega: float = opts.get("max_omega", 3.0)

        # ---- Geometry radii ----
        if geometry == "planar":
            self.inner_radius: float = 0.0
            self.outer_radius: float = opts.get("outer_radius", 0.0)
            # 0.0 means auto-compute in initialize_physics
        else:
            # Torus / Annular: R + r = OUTER (constant)
            self.OUTER: float = opts.get("outer_radius", 3.0)
            self.R: float = 0.0
            self.r: float = 0.0
            self.r0: float = 0.0

        # ---- Particle state (set in initialize_physics) ----
        self.pos = None
        self.vel = None
        self.pair_eq = None  # (N, N) — sigma or raw distances

        # ---- Simulation state flags ----
        self.collapsing: bool = False
        self.collapsed: bool = False
        self.circle_phase: bool = False
        self.finished: bool = False
        self.time: float = 0.0
        self.found_tour: Optional[list[int]] = None

        # ---- GPU kernels ----
        self._kernel = None
        self._circle_kernel = None
        if self.use_gpu:
            self._compile_kernels(opts)

    # ------------------------------------------------------------------
    # Kernel compilation (GPU only)
    # ------------------------------------------------------------------

    def _compile_kernels(self, opts: dict) -> None:
        """Compile CUDA kernels via kernel_builder."""
        from tsp_nbody.kernel_builder import build_kernel, build_circle_kernel

        # Build force-model-specific #define overrides
        force_params: dict = {}
        if self.force_model == "piecewise_lj":
            for k in ("slope_repulsion", "mag_attraction", "force_cutoff_extra"):
                if k in opts:
                    force_params[k] = opts[k]
        elif self.force_model == "smooth_lj":
            for k in ("p", "q", "m"):
                if k in opts:
                    force_params[k] = opts[k]

        self._kernel = build_kernel(
            self.geometry,
            self.force_model,
            self.wall_model,
            force_params or None,
        )

        if self.geometry == "torus":
            self._circle_kernel = build_circle_kernel()

    # ------------------------------------------------------------------
    # Tube-radius helper (torus / annular)
    # ------------------------------------------------------------------

    def _set_tube_radius(self, r: float) -> None:
        """Set tube radius and update major radius so R + r = OUTER."""
        self.r = max(0.01, r)
        self.R = self.OUTER - self.r

    # ------------------------------------------------------------------
    # initialize_physics
    # ------------------------------------------------------------------

    def initialize_physics(self) -> None:  # noqa: C901  (complexity unavoidable)
        """Reset particle state to initial configuration."""
        self.collapsing = False
        self.collapsed = False
        self.circle_phase = False
        self.finished = False
        self.time = 0.0
        self.found_tour = None

        if self.geometry == "planar":
            self._init_planar()
        elif self.geometry == "torus":
            self._init_torus()
        else:
            self._init_annular()

        # Per-particle mass array
        xp = self.xp
        if self._mass_per_particle is not None:
            self.mass_arr = xp.asarray(self._mass_per_particle, dtype=xp.float32)
        else:
            self.mass_arr = xp.full(self.n_cities, self._mass_scalar, dtype=xp.float32)

    # ---- Planar init ----

    def _init_planar(self) -> None:
        coords = self.original_coords.copy()
        n = self.n_cities
        xp = self.xp

        self.inner_radius = 0.0

        # Auto-compute outer_radius if not set
        if self.outer_radius <= 0.0:
            radii = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
            self.outer_radius = float(np.max(radii)) * 1.05

        # Move particles too close to origin
        radii = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        too_close = radii < 0.001
        coords[too_close, 0] = 0.001
        coords[too_close, 1] = 0.001

        self.pos = xp.asarray(coords, dtype=xp.float32)
        self.vel = xp.zeros((n, 2), dtype=xp.float32)

        # Compute pair_eq
        self._compute_pair_eq_planar()

    def _compute_pair_eq_planar(self) -> None:
        """Compute per-pair equilibrium parameter for the planar geometry."""
        xp = self.xp

        # For all force models, we need initial pairwise distances
        # stored on whatever device is active.
        coords_dev = xp.asarray(self.original_coords, dtype=xp.float32)
        diff = coords_dev[xp.newaxis, :, :] - coords_dev[:, xp.newaxis, :]
        dist = xp.sqrt(xp.sum(diff * diff, axis=2))  # (N, N)

        if self.force_model == "true_lj":
            sig_scale = 1.0 / (2.0 ** (1.0 / 6.0))
            self.pair_eq = (dist * sig_scale).astype(xp.float32)
        else:
            # piecewise_lj / smooth_lj: pair_eq = raw initial distances
            self.pair_eq = dist.astype(xp.float32)

    # ---- Torus init ----

    def _init_torus(self, seed: int = 42) -> None:
        rng = _mulberry32(seed)
        n = self.n_cities
        xp = self.xp

        start_r = self.OUTER * 0.48
        self._set_tube_radius(start_r)
        self.r0 = self.r

        coords = self.original_coords
        cx = float(np.mean(coords[:, 0]))
        cy = float(np.mean(coords[:, 1]))
        min_x, max_x = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
        min_y, max_y = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
        range_x = max(max_x - min_x, 1.0)
        range_y = max(max_y - min_y, 1.0)

        positions = np.zeros((n, 3), dtype=np.float32)

        if self.embed_mode == "linear":
            for i in range(n):
                raw_x, raw_y = coords[i, 0], coords[i, 1]
                nx_val = (raw_x - min_x) / range_x
                ny_val = (raw_y - min_y) / range_y
                u = nx_val * TAU
                cross_dist = (ny_val - 0.5) * 2.0
                r_frac = abs(cross_dist) * 0.75
                v_angle = 0.0 if cross_dist >= 0 else np.pi
                pert_scale = self.perturbation
                pert_angle = (rng() - 0.5) * np.pi * pert_scale
                final_v = v_angle + pert_angle
                final_r_frac = r_frac + (rng() - 0.5) * 0.15 * pert_scale
                rr = max(0, min(0.85, abs(final_r_frac))) * self.r
                cu, su = np.cos(u), np.sin(u)
                cv, sv = np.cos(final_v), np.sin(final_v)
                positions[i] = [
                    (self.R + rr * cv) * cu,
                    (self.R + rr * cv) * su,
                    rr * sv,
                ]

        elif self.embed_mode == "centroid":
            angles = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)
            dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
            max_dist = max(float(np.max(dists)), 1e-10)

            for i in range(n):
                u = angles[i]
                r_frac = (dists[i] / max_dist) * 0.80
                v_angle = 0.0
                pert_scale = self.perturbation
                pert_angle = (rng() - 0.5) * np.pi * pert_scale
                final_v = v_angle + pert_angle
                rr = max(0, min(0.85, r_frac + (rng() - 0.5) * 0.1 * pert_scale)) * self.r
                cu, su = np.cos(u), np.sin(u)
                cv, sv = np.cos(final_v), np.sin(final_v)
                positions[i] = [
                    (self.R + rr * cv) * cu,
                    (self.R + rr * cv) * su,
                    rr * sv,
                ]

        elif self.embed_mode == "flat":
            for i in range(n):
                lx = coords[i, 0] - cx
                ly = coords[i, 1] - cy
                pert_scale = self.perturbation
                pz = (rng() - 0.5) * self.r * 0.1 * pert_scale
                positions[i] = [lx, ly, pz]

        # Centre of mass to origin
        com = np.mean(positions, axis=0)
        positions -= com

        # For flat mode: rescale so widest extent spans torus diameter
        if self.embed_mode == "flat":
            max_extent = float(np.max(np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)))
            if max_extent > 0:
                torus_diameter = self.R + self.r
                scale = torus_diameter / max_extent
                positions *= scale

        self.pos = xp.asarray(positions, dtype=xp.float32)
        self.vel = xp.zeros((n, 3), dtype=xp.float32)

        self._compute_pair_sigma_3d()

    def _compute_pair_sigma_3d(self) -> None:
        """Compute per-pair sigma = distance * 2^(-1/6) for torus geometry."""
        sig_scale = 1.0 / (2.0 ** (1.0 / 6.0))
        pos_cpu = self.pos if not self.use_gpu else cp.asnumpy(self.pos)
        diff = pos_cpu[:, np.newaxis, :] - pos_cpu[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        pair_sigma = (dist * sig_scale).astype(np.float32)
        self.pair_eq = self.xp.asarray(pair_sigma) if self.use_gpu else pair_sigma

    # ---- Annular init ----

    def _init_annular(self) -> None:
        n = self.n_cities
        xp = self.xp

        start_r = self.OUTER * 0.48
        self._set_tube_radius(start_r)
        self.r0 = self.r

        coords = self.original_coords.copy()
        cx = float(np.mean(coords[:, 0]))
        cy = float(np.mean(coords[:, 1]))
        coords[:, 0] -= cx
        coords[:, 1] -= cy

        max_extent = float(np.max(np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)))
        if max_extent > 0:
            scale = (self.R + self.r * 0.8) / max_extent
            coords *= scale

        self.pos = xp.asarray(coords, dtype=xp.float32)
        self.vel = xp.zeros((n, 2), dtype=xp.float32)

        self._compute_pair_sigma_2d()

    def _compute_pair_sigma_2d(self) -> None:
        """Compute per-pair sigma for 2-D annular geometry."""
        sig_scale = 1.0 / (2.0 ** (1.0 / 6.0))
        pos_cpu = self.pos if not self.use_gpu else cp.asnumpy(self.pos)
        diff = pos_cpu[:, np.newaxis, :] - pos_cpu[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        pair_sigma = (dist * sig_scale).astype(np.float32)
        self.pair_eq = self.xp.asarray(pair_sigma) if self.use_gpu else pair_sigma

    # ------------------------------------------------------------------
    # start_collapse / integrate_step
    # ------------------------------------------------------------------

    def start_collapse(self) -> None:
        self.collapsing = True

    def integrate_step(self) -> None:
        if not self.collapsing:
            self.time += self.dt
            return

        # Dispatch to GPU or CPU physics step
        if self.use_gpu:
            self._step_gpu()
        else:
            self._step_cpu()

        # Collapse progression
        if self.geometry == "planar":
            self.inner_radius += self.collapse_rate
            if self.inner_radius + self.collapse_rate * 10 >= self.outer_radius:
                self.finished = True
                self._extract_tour()
        elif self.geometry == "torus":
            if not self.circle_phase:
                self._set_tube_radius(self.r - self.collapse_rate * self.dt)
                if self.r <= self.epsilon:
                    self._set_tube_radius(self.epsilon)
                    self.collapsed = True
                    self._trigger_circle_phase()
        else:  # annular
            self._set_tube_radius(self.r - self.collapse_rate * self.dt)
            if self.r <= self.epsilon:
                self._set_tube_radius(self.epsilon)
                self.collapsed = True
                self._extract_tour()

        self.time += self.dt

    # ------------------------------------------------------------------
    # GPU step dispatch
    # ------------------------------------------------------------------

    def _step_gpu(self) -> None:
        n = self.n_cities
        threads = 256
        blocks = (n + threads - 1) // threads

        if self.geometry == "torus" and self.circle_phase:
            self._step_gpu_circle(blocks, threads)
            return

        if self.geometry == "planar":
            self._step_gpu_planar(blocks, threads)
        elif self.geometry == "torus":
            self._step_gpu_torus(blocks, threads)
        else:
            self._step_gpu_annular(blocks, threads)

    def _step_gpu_planar(self, blocks: int, threads: int) -> None:
        if self.wall_model == "inverse_square":
            effective_wall_strength = self.lj_strength * self.wall_force_scale
        else:
            effective_wall_strength = self.wall_strength

        self._kernel(
            (blocks,), (threads,),
            (
                self.pos, self.vel, self.pair_eq, self.mass_arr,
                cp.float32(self.lj_strength),
                cp.float32(self.inner_radius),
                cp.float32(self.outer_radius),
                cp.float32(effective_wall_strength),
                cp.float32(self.wall_range_frac),
                cp.float32(self.dt),
                cp.float32(self.damping),
                cp.float32(self.max_speed),
                cp.int32(self.n_cities),
            ),
        )

    def _step_gpu_torus(self, blocks: int, threads: int) -> None:
        effective_wall_strength = self.lj_strength * self.wall_force_scale
        self._kernel(
            (blocks,), (threads,),
            (
                self.pos, self.vel, self.pair_eq, self.mass_arr,
                cp.float32(self.lj_strength),
                cp.float32(self.R),
                cp.float32(self.r),
                cp.float32(effective_wall_strength),
                cp.float32(self.wall_range_frac),
                cp.float32(self.dt),
                cp.float32(self.damping),
                cp.float32(self.max_speed),
                cp.int32(self.n_cities),
            ),
        )

    def _step_gpu_annular(self, blocks: int, threads: int) -> None:
        effective_wall_strength = self.lj_strength * self.wall_force_scale
        self._kernel(
            (blocks,), (threads,),
            (
                self.pos, self.vel, self.pair_eq, self.mass_arr,
                cp.float32(self.lj_strength),
                cp.float32(self.R),
                cp.float32(self.r),
                cp.float32(effective_wall_strength),
                cp.float32(self.wall_range_frac),
                cp.float32(self.dt),
                cp.float32(self.damping),
                cp.float32(self.max_speed),
                cp.int32(self.n_cities),
            ),
        )

    def _step_gpu_circle(self, blocks: int, threads: int) -> None:
        self._circle_kernel(
            (blocks,), (threads,),
            (
                self.pos, self.vel, self.pair_eq,
                cp.float32(self.lj_strength),
                cp.float32(self.circle_lj_scale),
                cp.float32(self.R),
                cp.float32(self.dt),
                cp.float32(self.circle_damping),
                cp.float32(self.max_omega),
                cp.float32(self.sigma_arc_scale),
                cp.int32(self.n_cities),
            ),
        )

    # ------------------------------------------------------------------
    # CPU step dispatch
    # ------------------------------------------------------------------

    def _step_cpu(self) -> None:
        if self.geometry == "torus" and self.circle_phase:
            self._step_cpu_circle()
            return

        if self.geometry == "planar":
            self._step_cpu_planar()
        elif self.geometry == "torus":
            self._step_cpu_torus()
        else:
            self._step_cpu_annular()

    # ---- Planar CPU ----

    def _step_cpu_planar(self) -> None:
        xp = self.xp
        n = self.n_cities

        # Pairwise differences: diff[i,j] = pos[i] - pos[j] (away from j)
        diff = self.pos[:, xp.newaxis, :] - self.pos[xp.newaxis, :, :]  # (N,N,2)
        dist = xp.sqrt(xp.sum(diff * diff, axis=2))  # (N,N)
        dist_safe = xp.maximum(dist, 1e-10)

        # Compute force magnitudes
        force_mag = self._compute_cpu_force_mag(dist, dist_safe, xp)

        # Zero diagonal
        xp.fill_diagonal(force_mag, 0.0)

        # Direction unit vectors and sum
        direction = diff / dist_safe[:, :, xp.newaxis]  # (N,N,2)
        forces = xp.sum(force_mag[:, :, xp.newaxis] * direction, axis=1)  # (N,2)

        # Wall forces
        forces += self._compute_cpu_wall_forces_planar(xp)

        # Integration
        forces = xp.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        self.vel[:] = (self.vel + forces * self.dt / self.mass_arr[:, xp.newaxis]) * self.damping
        self.vel[:] = xp.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)

        speed = xp.sqrt(xp.sum(self.vel * self.vel, axis=1))
        too_fast = speed > self.max_speed
        if xp.any(too_fast):
            self.vel[too_fast] *= (self.max_speed / speed[too_fast])[:, xp.newaxis]

        self.pos[:] = self.pos + self.vel * self.dt

    def _compute_cpu_force_mag(self, dist, dist_safe, xp):
        """Compute pairwise force magnitudes based on force_model."""
        n = self.n_cities

        if self.force_model == "true_lj":
            sigma = self.pair_eq
            cutoff = sigma * 2.5
            sr = xp.minimum(sigma / dist_safe, 100.0)
            sr6 = sr ** 6
            fmag = 24.0 * self.lj_strength / dist_safe * (2.0 * sr6 * sr6 - sr6)
            fmag = xp.clip(fmag, -1000.0, 1000.0)
            mask = (dist > 0.005) & (dist < cutoff)
            xp.fill_diagonal(mask, False)
            return xp.where(mask, fmag, 0.0)

        elif self.force_model == "piecewise_lj":
            # Sign convention: positive = repulsive, negative = attractive
            # (diff points away from j, so positive pushes apart)
            eq_len = self.pair_eq  # raw initial distances
            eq_len_safe = xp.maximum(eq_len, 1e-10)
            fmag = xp.zeros((n, n), dtype=xp.float32)
            compressed = dist_safe <= eq_len
            fmag[compressed] = (eq_len[compressed] - dist_safe[compressed]) * self.slope_repulsion
            extended = (dist_safe > eq_len) & (dist_safe < self.force_cutoff_extra)
            fmag[extended] = -self.mag_attraction / eq_len_safe[extended]
            return fmag

        else:  # smooth_lj
            # Sign convention: positive = repulsive, negative = attractive
            eq_len = self.pair_eq
            eq_len_safe = xp.maximum(eq_len, 1e-10)
            p, q, m = self.p_exp, self.q_exp, self.m_coeff
            h = m * ((q / p) ** (1.0 / (q - p)) * eq_len_safe) ** p / (1.0 - p / q)
            g = h * eq_len_safe ** (q - p)
            fmag = -(g / dist_safe ** q - h / dist_safe ** p)
            return fmag

    def _compute_cpu_wall_forces_planar(self, xp):
        """Compute concentric-circle wall forces for planar geometry."""
        forces = xp.zeros_like(self.pos)
        dx = self.pos[:, 0]
        dy = self.pos[:, 1]
        radius = xp.sqrt(dx * dx + dy * dy)
        safe_r = xp.maximum(radius, 1e-10)
        nx_dir = dx / safe_r
        ny_dir = dy / safe_r

        wall_gap = self.outer_radius - self.inner_radius
        wall_range = wall_gap * self.wall_range_frac

        if self.wall_model == "inverse_square":
            effective_strength = self.lj_strength * self.wall_force_scale

            # Inner wall
            if self.inner_radius > 0.001:
                dist_inner = radius - self.inner_radius  # positive = safe
                near_inner = (dist_inner < wall_range) & (dist_inner > 0) & (radius > 1e-10)
                if xp.any(near_inner):
                    t = xp.maximum(0.001, dist_inner[near_inner] / wall_range)
                    f_mag = effective_strength / (t * t)
                    forces[near_inner, 0] += f_mag * nx_dir[near_inner]
                    forces[near_inner, 1] += f_mag * ny_dir[near_inner]

                # Hard constraint: clamp inside inner wall
                inside = radius < self.inner_radius
                if xp.any(inside):
                    sr = xp.maximum(radius[inside], 1e-10)
                    n_x = dx[inside] / sr
                    n_y = dy[inside] / sr
                    self.pos[inside, 0] = n_x * (self.inner_radius + 0.01)
                    self.pos[inside, 1] = n_y * (self.inner_radius + 0.01)
                    vn = self.vel[inside, 0] * n_x + self.vel[inside, 1] * n_y
                    self.vel[inside, 0] = (self.vel[inside, 0] - 2.0 * vn * n_x) * 0.7
                    self.vel[inside, 1] = (self.vel[inside, 1] - 2.0 * vn * n_y) * 0.7

            # Outer wall
            dist_outer = self.outer_radius - radius  # positive = safe
            near_outer = (dist_outer < wall_range) & (dist_outer > 0) & (radius > 1e-10)
            if xp.any(near_outer):
                t = xp.maximum(0.001, dist_outer[near_outer] / wall_range)
                f_mag = -effective_strength / (t * t)
                forces[near_outer, 0] += f_mag * nx_dir[near_outer]
                forces[near_outer, 1] += f_mag * ny_dir[near_outer]

            # Hard constraint: clamp outside outer wall
            outside = radius > self.outer_radius
            if xp.any(outside):
                sr = xp.maximum(radius[outside], 1e-10)
                n_x = dx[outside] / sr
                n_y = dy[outside] / sr
                self.pos[outside, 0] = n_x * (self.outer_radius - 0.01)
                self.pos[outside, 1] = n_y * (self.outer_radius - 0.01)
                vn = self.vel[outside, 0] * n_x + self.vel[outside, 1] * n_y
                self.vel[outside, 0] = (self.vel[outside, 0] - 2.0 * vn * n_x) * 0.7
                self.vel[outside, 1] = (self.vel[outside, 1] - 2.0 * vn * n_y) * 0.7

        else:  # linear
            # Inner wall: particles inside inner radius
            if self.inner_radius > 0.001:
                inside_inner = radius < self.inner_radius
                if xp.any(inside_inner):
                    f_mag = self.wall_strength * (self.inner_radius - radius[inside_inner])
                    sr = xp.maximum(radius[inside_inner], 1e-10)
                    forces[inside_inner, 0] = f_mag * dx[inside_inner] / sr
                    forces[inside_inner, 1] = f_mag * dy[inside_inner] / sr

            # Outer wall: particles outside outer radius
            outside_outer = radius > self.outer_radius
            if xp.any(outside_outer):
                f_mag = self.wall_strength * (self.outer_radius - radius[outside_outer])
                sr = xp.maximum(radius[outside_outer], 1e-10)
                forces[outside_outer, 0] = f_mag * dx[outside_outer] / sr
                forces[outside_outer, 1] = f_mag * dy[outside_outer] / sr

        return forces

    # ---- Torus CPU ----

    def _step_cpu_torus(self) -> None:
        xp = self.xp
        n = self.n_cities
        pos = self.pos
        vel = self.vel

        # Pairwise LJ forces (vectorised)
        diff = pos[:, xp.newaxis, :] - pos[xp.newaxis, :, :]  # (N,N,3)
        dist = xp.sqrt(xp.sum(diff * diff, axis=2))  # (N,N)
        dist_safe = xp.maximum(dist, 0.01)

        sigma = self.pair_eq
        cutoff = sigma * 2.5
        sr = sigma / dist_safe
        sr6 = sr ** 6
        fmag = 24.0 * self.lj_strength / dist_safe * (2.0 * sr6 * sr6 - sr6)

        mask = (dist > 0.005) & (dist < cutoff)
        xp.fill_diagonal(mask, False)
        fmag = xp.where(mask, fmag, 0.0)

        direction = diff / xp.maximum(dist, 1e-8)[:, :, xp.newaxis]
        forces = xp.sum(fmag[:, :, xp.newaxis] * direction, axis=1)  # (N,3)

        # Wall forces (tube wall)
        theta = xp.arctan2(pos[:, 1], pos[:, 0])
        centers = xp.stack([
            self.R * xp.cos(theta),
            self.R * xp.sin(theta),
            xp.zeros(n, dtype=xp.float32),
        ], axis=1)

        w_diff = pos - centers
        w_dist = xp.sqrt(xp.sum(w_diff * w_diff, axis=1))
        wall_range = self.r * self.wall_range_frac
        wall_dist = self.r - w_dist
        w_active = (wall_dist < wall_range) & (w_dist > 0.001)

        if xp.any(w_active):
            t = xp.maximum(0.001, wall_dist[w_active] / wall_range)
            w_direction = w_diff[w_active] / w_dist[w_active, xp.newaxis]
            w_fmag = -self.lj_strength * self.wall_force_scale / (t * t)
            forces[w_active] += w_fmag[:, xp.newaxis] * w_direction

        # Integration with damping
        vel[:] = (vel + forces * self.dt / self.mass_arr[:, xp.newaxis]) * self.damping
        speed = xp.sqrt(xp.sum(vel * vel, axis=1))
        too_fast = speed > self.max_speed
        if xp.any(too_fast):
            vel[too_fast] *= (self.max_speed / speed[too_fast])[:, xp.newaxis]

        pos[:] = pos + vel * self.dt

        # Hard torus constraint
        theta2 = xp.arctan2(pos[:, 1], pos[:, 0])
        centers2 = xp.stack([
            self.R * xp.cos(theta2),
            self.R * xp.sin(theta2),
            xp.zeros(n, dtype=xp.float32),
        ], axis=1)
        c_diff = pos - centers2
        c_dist = xp.sqrt(xp.sum(c_diff * c_diff, axis=1))

        violated = c_dist > self.r * 0.99
        if xp.any(violated):
            c_n = c_diff[violated] / xp.maximum(c_dist[violated], 1e-8)[:, xp.newaxis]
            pos[violated] = centers2[violated] + c_n * (self.r * 0.98)
            vn = xp.sum(vel[violated] * c_n, axis=1)
            vel[violated] = (vel[violated] - 2.0 * vn[:, xp.newaxis] * c_n) * 0.7

    # ---- Annular CPU ----

    def _step_cpu_annular(self) -> None:
        xp = self.xp
        n = self.n_cities
        pos = self.pos
        vel = self.vel

        # Pairwise LJ forces
        diff = pos[:, xp.newaxis, :] - pos[xp.newaxis, :, :]  # (N,N,2)
        dist = xp.sqrt(xp.sum(diff * diff, axis=2))
        dist_safe = xp.maximum(dist, 0.01)

        sigma = self.pair_eq
        cutoff = sigma * 2.5
        sr = xp.minimum(sigma / dist_safe, 100.0)
        sr6 = sr ** 6
        fmag = 24.0 * self.lj_strength / dist_safe * (2.0 * sr6 * sr6 - sr6)
        fmag = xp.clip(fmag, -1000.0, 1000.0)

        mask = (dist > 0.01) & (dist < cutoff)
        xp.fill_diagonal(mask, False)
        fmag = xp.where(mask, fmag, 0.0)

        direction = diff / dist_safe[:, :, xp.newaxis]
        forces = xp.sum(fmag[:, :, xp.newaxis] * direction, axis=1)  # (N,2)

        # Wall force: distance from major circle at radius R
        theta = xp.arctan2(pos[:, 1], pos[:, 0])
        centers = xp.stack([self.R * xp.cos(theta), self.R * xp.sin(theta)], axis=1)
        w_diff = pos - centers
        w_dist = xp.sqrt(xp.sum(w_diff * w_diff, axis=1))
        wall_dist = self.r - w_dist
        wall_range = self.r * self.wall_range_frac

        w_active = (wall_dist < wall_range) & (w_dist > 0.001)
        if xp.any(w_active):
            t = xp.maximum(0.001, wall_dist[w_active] / wall_range)
            w_dir = w_diff[w_active] / w_dist[w_active, xp.newaxis]
            w_fmag = xp.clip(
                -self.lj_strength * self.wall_force_scale / (t * t),
                -1000.0, 1000.0,
            )
            forces[w_active] += w_fmag[:, xp.newaxis] * w_dir

        # Hard constraint: clamp to tube
        radius = xp.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        dist_from_R = xp.abs(radius - self.R)
        violated = (dist_from_R > self.r * 0.99) & (radius > 0.001)
        if xp.any(violated):
            sign = xp.sign(radius[violated] - self.R)
            target_r = self.R + sign * self.r * 0.98
            inv_r = 1.0 / xp.maximum(radius[violated], 1e-8)
            pos[violated, 0] = pos[violated, 0] * inv_r * target_r
            pos[violated, 1] = pos[violated, 1] * inv_r * target_r
            n_x = pos[violated, 0] * inv_r
            n_y = pos[violated, 1] * inv_r
            vn = vel[violated, 0] * n_x + vel[violated, 1] * n_y
            vel[violated, 0] = (vel[violated, 0] - 2.0 * vn * n_x) * 0.7
            vel[violated, 1] = (vel[violated, 1] - 2.0 * vn * n_y) * 0.7

        # Integration
        forces = xp.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        vel[:] = (vel + forces * self.dt / self.mass_arr[:, xp.newaxis]) * self.damping
        vel[:] = xp.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        speed = xp.sqrt(xp.sum(vel * vel, axis=1))
        too_fast = speed > self.max_speed
        if xp.any(too_fast):
            vel[too_fast] *= (self.max_speed / speed[too_fast])[:, xp.newaxis]
        pos[:] = pos + vel * self.dt

    # ---- Circle-phase CPU (torus only) ----

    def _step_cpu_circle(self) -> None:
        xp = self.xp
        n = self.n_cities
        pos = self.pos
        vel = self.vel

        thetas = xp.arctan2(pos[:, 1], pos[:, 0])

        # Pairwise angular differences
        dtheta = thetas[xp.newaxis, :] - thetas[:, xp.newaxis]  # (N,N)
        dtheta = (dtheta + np.pi) % TAU - np.pi

        arc_dist = xp.abs(dtheta) * self.R

        sigma_arc = self.pair_eq * self.sigma_arc_scale
        cutoff = sigma_arc * 3.0

        arc_safe = xp.maximum(arc_dist, 0.01)
        sr = sigma_arc / arc_safe
        sr6 = sr ** 6
        eps = self.lj_strength * self.circle_lj_scale
        fmag = 24.0 * eps / arc_safe * (2.0 * sr6 * sr6 - sr6)

        mask = (arc_dist > 0.01) & (arc_dist < cutoff)
        xp.fill_diagonal(mask, False)
        fmag = xp.where(mask, fmag, 0.0)

        sign = xp.where(dtheta > 0, 1.0, -1.0)
        angular_forces = xp.sum(fmag * sign, axis=1)  # (N,)

        cos_t = xp.cos(thetas)
        sin_t = xp.sin(thetas)
        tangents = xp.stack([-sin_t, cos_t, xp.zeros(n, dtype=xp.float32)], axis=1)

        omega = xp.sum(vel * tangents, axis=1) / self.R
        omega = (omega + angular_forces * self.dt / self.R) * self.circle_damping
        omega = xp.clip(omega, -self.max_omega, self.max_omega)

        thetas = thetas + omega * self.dt
        cos_t = xp.cos(thetas)
        sin_t = xp.sin(thetas)

        pos[:, 0] = self.R * cos_t
        pos[:, 1] = self.R * sin_t
        pos[:, 2] = 0.0
        vel[:, 0] = -sin_t * omega * self.R
        vel[:, 1] = cos_t * omega * self.R
        vel[:, 2] = 0.0

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _trigger_circle_phase(self) -> None:
        """Transition from torus collapse to circle phase."""
        self.circle_phase = True
        pos_cpu = self.get_positions_cpu()
        vel_cpu = self.get_velocities_cpu()

        thetas = np.arctan2(pos_cpu[:, 1], pos_cpu[:, 0])
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Tangent vectors
        tangents = np.stack(
            [-sin_t, cos_t, np.zeros(self.n_cities)], axis=1
        ).astype(np.float32)

        # Project velocity onto tangent
        v_tang = np.sum(vel_cpu * tangents, axis=1)

        pos_cpu[:, 0] = self.R * cos_t
        pos_cpu[:, 1] = self.R * sin_t
        pos_cpu[:, 2] = 0.0
        vel_cpu[:] = tangents * v_tang[:, np.newaxis]

        # Extract initial tour (0-indexed)
        sorted_indices = np.argsort(thetas)
        self.found_tour = sorted_indices.tolist()

        self.pos = self.xp.asarray(pos_cpu)
        self.vel = self.xp.asarray(vel_cpu)

    def _extract_tour(self) -> None:
        """Extract tour by angle-sorting current positions (0-indexed)."""
        pos_cpu = self.get_positions_cpu()
        thetas = np.arctan2(pos_cpu[:, 1], pos_cpu[:, 0])
        sorted_indices = np.argsort(thetas)
        self.found_tour = sorted_indices.tolist()

    # ------------------------------------------------------------------
    # Accessors (BasePhysicsEngine interface)
    # ------------------------------------------------------------------

    def get_positions_cpu(self) -> np.ndarray:
        if self.use_gpu:
            return cp.asnumpy(self.pos)
        return self.pos.copy()

    def get_velocities_cpu(self) -> np.ndarray:
        if self.use_gpu:
            return cp.asnumpy(self.vel)
        return self.vel.copy()

    def get_final_tour(self) -> np.ndarray:
        """Return 0-indexed tour from current particle positions."""
        pos = self.get_positions_cpu()
        return np.argsort(np.arctan2(pos[:, 1], pos[:, 0]))

    def get_found_tour(self) -> Optional[list[int]]:
        """Return 0-indexed tour extracted during collapse, or None."""
        return self.found_tour

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        if self.geometry == "torus":
            if self.circle_phase:
                return "CIRCLE"
            elif self.collapsing:
                return "COLLAPSING"
            return "READY"
        elif self.geometry == "annular":
            if self.collapsed:
                return "COLLAPSED"
            elif self.collapsing:
                return "COLLAPSING"
            return "READY"
        else:  # planar
            if self.finished:
                return "FINISHED"
            elif self.collapsing:
                return "COLLAPSING"
            return "READY"

    @property
    def is_complete(self) -> bool:
        if self.geometry == "torus":
            return self.circle_phase
        elif self.geometry == "annular":
            return self.collapsed
        return self.finished

    @property
    def ndim(self) -> int:
        return 3 if self.geometry == "torus" else 2
