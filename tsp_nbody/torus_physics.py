"""
Torus Physics Engine for N-Body TSP Simulator.

Implements 3D torus geometry with tube collapse to circle (S¹).
Ported from the JavaScript NBodyTSP-Torus project.

Key concepts:
- Particles live inside a 3D torus defined by major radius R and tube radius r
- R + r = OUTER (constant)
- Tube collapses (r shrinks) until particles are forced onto the major circle
- Tour is extracted by sorting particles by angle on the circle
"""

import numpy as np
from typing import Optional, TypedDict

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

TAU = 2.0 * np.pi


class TorusPhysicsOptions(TypedDict, total=False):
    use_gpu: bool
    outer_radius: float      # R + r constant (default 3.0)
    shrink_rate: float        # Rate of tube collapse per timestep
    epsilon: float            # Tube radius threshold for collapse completion
    lj_strength: float        # Lennard-Jones force scale
    perturbation: float       # Initial z-perturbation magnitude
    dt: float                 # Timestep
    damping: float            # Velocity damping factor per step
    max_speed: float          # Velocity clamp
    embed_mode: str           # 'flat', 'linear', or 'centroid'
    wall_range_frac: float    # Wall force zone as fraction of tube radius
    wall_force_scale: float   # Wall force strength multiplier
    sigma_arc_scale: float    # Circle phase sigma scaling factor
    circle_lj_scale: float    # Circle phase LJ strength multiplier
    circle_damping: float     # Circle phase angular velocity damping
    max_omega: float          # Circle phase angular velocity clamp


default_torus_options: TorusPhysicsOptions = {
    'use_gpu': False,
    'outer_radius': 3.0,
    'shrink_rate': 0.10,
    'epsilon': 0.08,
    'lj_strength': 1.0,
    'perturbation': 0.50,
    'dt': 0.004,
    'damping': 0.97,
    'max_speed': 2.5,
    'embed_mode': 'flat',
    'wall_range_frac': 0.05,
    'wall_force_scale': 8.0,
    'sigma_arc_scale': 0.35,
    'circle_lj_scale': 0.3,
    'circle_damping': 0.95,
    'max_omega': 3.0,
}


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


class TorusPhysicsEngine:
    """3D torus-based N-body physics for TSP."""

    def __init__(self, coords: np.ndarray, options: Optional[TorusPhysicsOptions] = None):
        opts = default_torus_options.copy()
        if options:
            opts.update(options)

        self.use_gpu = opts.get('use_gpu', False) and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.original_coords = np.asarray(coords, dtype=np.float32)
        self.n_cities = len(coords)

        # Torus geometry
        self.OUTER = opts['outer_radius']
        self.R = 0.0  # Major circle radius
        self.r = 0.0  # Tube radius
        self.r0 = 0.0  # Initial tube radius

        # Physics parameters
        self.shrink_rate = opts['shrink_rate']
        self.epsilon = opts['epsilon']
        self.lj_strength = opts['lj_strength']
        self.perturbation = opts['perturbation']
        self.dt = opts['dt']
        self.damping = opts['damping']
        self.max_speed = opts['max_speed']
        self.embed_mode = opts['embed_mode']
        self.wall_range_frac = opts['wall_range_frac']
        self.wall_force_scale = opts['wall_force_scale']
        self.sigma_arc_scale = opts['sigma_arc_scale']
        self.circle_lj_scale = opts['circle_lj_scale']
        self.circle_damping = opts['circle_damping']
        self.max_omega = opts['max_omega']

        # 3D particle state
        self.pos = None  # (N, 3) positions
        self.vel = None  # (N, 3) velocities
        self.pair_sigma = None  # (N, N) equilibrium parameters

        # Simulation state
        self.collapsing = False
        self.collapsed = False
        self.circle_phase = False
        self.time = 0.0
        self.found_tour = None

        # CUDA kernels
        if self.use_gpu:
            self._compile_kernels()

        print(f"Torus physics engine initialized ({'GPU' if self.use_gpu else 'CPU'})")
        print(f"  Cities: {self.n_cities}")
        print(f"  Embed mode: {self.embed_mode}")
        print(f"  OUTER radius: {self.OUTER}")

    def _set_tube_radius(self, r: float):
        """Set tube radius and update major radius to maintain R + r = OUTER."""
        self.r = max(0.01, r)
        self.R = self.OUTER - self.r

    def initialize_physics(self, seed: int = 42):
        """Initialize particle positions using the selected embedding mode."""
        rng = _mulberry32(seed)
        self.collapsed = False
        self.circle_phase = False
        self.collapsing = False
        self.time = 0.0
        self.found_tour = None

        start_r = self.OUTER * 0.48
        self._set_tube_radius(start_r)
        self.r0 = self.r

        coords = self.original_coords
        n = self.n_cities

        # Compute centroid and bounds
        cx = np.mean(coords[:, 0])
        cy = np.mean(coords[:, 1])
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        range_x = max(max_x - min_x, 1.0)
        range_y = max(max_y - min_y, 1.0)

        positions = np.zeros((n, 3), dtype=np.float32)

        if self.embed_mode == 'linear':
            for i in range(n):
                raw_x, raw_y = coords[i, 0], coords[i, 1]
                nx = (raw_x - min_x) / range_x
                ny = (raw_y - min_y) / range_y
                u = nx * TAU
                cross_dist = (ny - 0.5) * 2.0
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

        elif self.embed_mode == 'centroid':
            angles = np.arctan2(coords[:, 1] - cy, coords[:, 0] - cx)
            dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
            max_dist = max(np.max(dists), 1e-10)

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

        elif self.embed_mode == 'flat':
            for i in range(n):
                lx = coords[i, 0] - cx
                ly = coords[i, 1] - cy
                pert_scale = self.perturbation
                pz = (rng() - 0.5) * self.r * 0.1 * pert_scale
                positions[i] = [lx, ly, pz]

        # Center of mass to origin
        com = np.mean(positions, axis=0)
        positions -= com

        # For flat mode: rescale so widest extent spans torus diameter
        if self.embed_mode == 'flat':
            max_extent = np.max(np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2))
            if max_extent > 0:
                torus_diameter = self.R + self.r
                scale = torus_diameter / max_extent
                positions *= scale

        # Initialize state
        self.pos = self.xp.asarray(positions, dtype=self.xp.float32)
        self.vel = self.xp.zeros((n, 3), dtype=self.xp.float32)

        # Compute pairwise equilibrium sigmas
        self._compute_pair_sigma()

        print(f"  Torus initialized: R={self.R:.3f}, r={self.r:.3f}")

    def _compute_pair_sigma(self):
        """Compute per-pair sigma values: sigma = distance * 2^(-1/6)."""
        sig_scale = 1.0 / (2.0 ** (1.0 / 6.0))
        pos_cpu = self.pos if not self.use_gpu else cp.asnumpy(self.pos)

        # Vectorized pairwise distances: (N,1,3) - (1,N,3) -> (N,N,3) -> (N,N)
        diff = pos_cpu[:, np.newaxis, :] - pos_cpu[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        pair_sigma = (dist * sig_scale).astype(np.float32)

        if self.use_gpu:
            self.pair_sigma = cp.asarray(pair_sigma)
        else:
            self.pair_sigma = pair_sigma

    def simulate_step_cpu(self):
        """One torus physics step on CPU — fully vectorized with NumPy."""
        if not self.collapsing:
            self.time += self.dt
            return

        n = self.n_cities
        pos = self.pos
        vel = self.vel

        # === Pairwise LJ forces (vectorized) ===
        # diff[i,j] = pos[i] - pos[j], shape (N,N,3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N,N)
        dist_safe = np.maximum(dist, 0.01)

        sigma = self.pair_sigma  # (N,N)
        cutoff = sigma * 2.5

        # LJ force magnitude: 24*eps/r * (2*(sigma/r)^12 - (sigma/r)^6)
        sr = sigma / dist_safe
        sr6 = sr ** 6
        fmag = 24.0 * self.lj_strength / dist_safe * (2.0 * sr6 * sr6 - sr6)

        # Mask: only apply where dist < cutoff and dist > 0.005 and i != j
        mask = (dist > 0.005) & (dist < cutoff)
        np.fill_diagonal(mask, False)
        fmag = np.where(mask, fmag, 0.0)

        # Direction unit vectors: diff / dist, shape (N,N,3)
        direction = diff / np.maximum(dist, 1e-8)[:, :, np.newaxis]

        # Force vectors: fmag * direction, summed over j for each i
        forces = np.sum(fmag[:, :, np.newaxis] * direction, axis=1)  # (N,3)

        # === Wall forces (vectorized) ===
        # Closest point on major circle for each particle
        theta = np.arctan2(pos[:, 1], pos[:, 0])  # (N,)
        centers = np.stack([
            self.R * np.cos(theta),
            self.R * np.sin(theta),
            np.zeros(n, dtype=np.float32),
        ], axis=1)  # (N,3)

        w_diff = pos - centers  # (N,3)
        w_dist = np.sqrt(np.sum(w_diff * w_diff, axis=1))  # (N,)

        wall_range = self.r * self.wall_range_frac
        wall_dist = self.r - w_dist
        w_active = (wall_dist < wall_range) & (w_dist > 0.001)

        if np.any(w_active):
            t = np.maximum(0.001, wall_dist[w_active] / wall_range)
            w_direction = w_diff[w_active] / w_dist[w_active, np.newaxis]
            w_fmag = -self.lj_strength * self.wall_force_scale / (t * t)
            forces[w_active] += w_fmag[:, np.newaxis] * w_direction

        # === Integration with damping ===
        vel[:] = (vel + forces * self.dt) * self.damping
        speed = np.sqrt(np.sum(vel * vel, axis=1))  # (N,)
        too_fast = speed > self.max_speed
        if np.any(too_fast):
            vel[too_fast] *= (self.max_speed / speed[too_fast])[:, np.newaxis]

        pos[:] = pos + vel * self.dt

        # === Hard torus constraint (vectorized) ===
        theta2 = np.arctan2(pos[:, 1], pos[:, 0])
        centers2 = np.stack([
            self.R * np.cos(theta2),
            self.R * np.sin(theta2),
            np.zeros(n, dtype=np.float32),
        ], axis=1)
        c_diff = pos - centers2
        c_dist = np.sqrt(np.sum(c_diff * c_diff, axis=1))

        violated = c_dist > self.r * 0.99
        if np.any(violated):
            c_n = c_diff[violated] / np.maximum(c_dist[violated], 1e-8)[:, np.newaxis]
            pos[violated] = centers2[violated] + c_n * (self.r * 0.98)
            vn = np.sum(vel[violated] * c_n, axis=1)
            vel[violated] = (vel[violated] - 2.0 * vn[:, np.newaxis] * c_n) * 0.7

        # === Tube collapse ===
        if self.collapsing and not self.collapsed:
            self._set_tube_radius(self.r - self.shrink_rate * self.dt)
            if self.r <= self.epsilon:
                self._set_tube_radius(self.epsilon)
                self.collapsed = True
                self._trigger_circle_phase()

        self.time += self.dt

    def simulate_step_gpu(self):
        """One torus physics step on GPU."""
        if not self.collapsing:
            self.time += self.dt
            return

        n = self.n_cities
        threads = 256
        blocks = (n + threads - 1) // threads

        self.torus_step_kernel(
            (blocks,), (threads,),
            (
                self.pos, self.vel, self.pair_sigma,
                cp.float32(self.lj_strength),
                cp.float32(self.R), cp.float32(self.r),
                cp.float32(self.dt), cp.float32(self.damping),
                cp.float32(self.max_speed),
                cp.float32(self.wall_range_frac),
                cp.float32(self.wall_force_scale),
                cp.int32(n),
            )
        )

        # Tube collapse (on CPU side)
        if self.collapsing and not self.collapsed:
            self._set_tube_radius(self.r - self.shrink_rate * self.dt)
            if self.r <= self.epsilon:
                self._set_tube_radius(self.epsilon)
                self.collapsed = True
                self._trigger_circle_phase()

        self.time += self.dt

    def _trigger_circle_phase(self):
        """Transition from torus to circle phase."""
        self.circle_phase = True
        pos_cpu = self.get_positions_cpu()
        vel_cpu = self.get_velocities_cpu()

        # Vectorized projection to major circle
        thetas = np.arctan2(pos_cpu[:, 1], pos_cpu[:, 0])
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Tangent vectors
        tangents = np.stack([-sin_t, cos_t, np.zeros(self.n_cities)], axis=1).astype(np.float32)
        # Project velocity onto tangent
        v_tang = np.sum(vel_cpu * tangents, axis=1)

        pos_cpu[:, 0] = self.R * cos_t
        pos_cpu[:, 1] = self.R * sin_t
        pos_cpu[:, 2] = 0.0
        vel_cpu[:] = tangents * v_tang[:, np.newaxis]

        # Extract initial tour
        sorted_indices = np.argsort(thetas)
        self.found_tour = [int(idx + 1) for idx in sorted_indices]

        self.pos = self.xp.asarray(pos_cpu)
        self.vel = self.xp.asarray(vel_cpu)

        print(f"  Circle phase triggered at R={self.R:.3f}, r={self.r:.3f}")

    def simulate_circle_step_cpu(self):
        """One circle phase step on CPU — vectorized angular LJ forces on S¹."""
        n = self.n_cities
        pos = self.pos
        vel = self.vel

        thetas = np.arctan2(pos[:, 1], pos[:, 0])  # (N,)

        # Pairwise angular differences: dtheta[i,j] = theta[j] - theta[i]
        dtheta = thetas[np.newaxis, :] - thetas[:, np.newaxis]  # (N,N)
        # Wrap to (-pi, pi]
        dtheta = (dtheta + np.pi) % TAU - np.pi

        arc_dist = np.abs(dtheta) * self.R  # (N,N)

        sigma_arc = self.pair_sigma * self.sigma_arc_scale  # (N,N)
        cutoff = sigma_arc * 3.0

        # LJ force magnitude
        arc_safe = np.maximum(arc_dist, 0.01)
        sr = sigma_arc / arc_safe
        sr6 = sr ** 6
        eps = self.lj_strength * self.circle_lj_scale
        fmag = 24.0 * eps / arc_safe * (2.0 * sr6 * sr6 - sr6)

        # Mask
        mask = (arc_dist > 0.01) & (arc_dist < cutoff)
        np.fill_diagonal(mask, False)
        fmag = np.where(mask, fmag, 0.0)

        # Sign: positive dtheta -> positive force on i
        sign = np.where(dtheta > 0, 1.0, -1.0)

        # Sum angular forces on each particle
        forces = np.sum(fmag * sign, axis=1)  # (N,)

        # Integration
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        tangents = np.stack([-sin_t, cos_t, np.zeros(n)], axis=1).astype(np.float32)

        omega = np.sum(vel * tangents, axis=1) / self.R  # (N,)
        omega = (omega + forces * self.dt / self.R) * self.circle_damping
        omega = np.clip(omega, -self.max_omega, self.max_omega)

        thetas = thetas + omega * self.dt
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        pos[:, 0] = self.R * cos_t
        pos[:, 1] = self.R * sin_t
        pos[:, 2] = 0.0
        vel[:, 0] = -sin_t * omega * self.R
        vel[:, 1] = cos_t * omega * self.R
        vel[:, 2] = 0.0

        self.time += self.dt

    def integrate_step(self):
        """Perform one physics step (dispatches to correct phase and backend)."""
        if self.circle_phase:
            if self.use_gpu:
                self.simulate_circle_step_gpu()
            else:
                self.simulate_circle_step_cpu()
        else:
            if self.use_gpu:
                self.simulate_step_gpu()
            else:
                self.simulate_step_cpu()

    def run_substeps(self, k: int):
        """Run K physics substeps."""
        for _ in range(k):
            self.integrate_step()

    def start_collapse(self):
        """Begin the torus collapse process."""
        self.collapsing = True

    def get_positions_cpu(self) -> np.ndarray:
        if self.use_gpu:
            return cp.asnumpy(self.pos)
        return self.pos.copy()

    def get_velocities_cpu(self) -> np.ndarray:
        if self.use_gpu:
            return cp.asnumpy(self.vel)
        return self.vel.copy()

    def get_positions_2d(self) -> np.ndarray:
        """Get positions projected to 2D (x, y only) for rendering."""
        pos = self.get_positions_cpu()
        return pos[:, :2]

    def get_final_tour(self) -> np.ndarray:
        """Get tour order based on angles (0-indexed)."""
        pos = self.get_positions_cpu()
        angles = np.arctan2(pos[:, 1], pos[:, 0])
        return np.argsort(angles)

    def get_found_tour(self) -> Optional[list[int]]:
        """Get the 1-indexed tour found during circle phase transition."""
        return self.found_tour

    def compute_kinetic_energy(self) -> float:
        xp = self.xp
        v_sq = xp.sum(self.vel ** 2, axis=1)
        ke = 0.5 * xp.sum(v_sq)
        if self.use_gpu:
            return float(cp.asnumpy(ke))
        return float(ke)

    @property
    def phase(self) -> str:
        if self.circle_phase:
            return "CIRCLE"
        elif self.collapsing:
            return "COLLAPSING"
        else:
            return "TORUS"

    # =========================================================
    # GPU Kernels
    # =========================================================
    def _compile_kernels(self):
        """Compile CUDA kernels for torus physics."""
        self.torus_step_kernel = cp.RawKernel(r"""
        extern "C" __global__
        void torusStep(float* pos, float* vel, const float* pairSigma,
                       float ljStrength, float majorR, float tubeR,
                       float dt, float damping, float maxSpeed,
                       float wallRangeFrac, float wallForceScale,
                       int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N) return;

            float px = pos[idx*3], py = pos[idx*3+1], pz = pos[idx*3+2];
            float vx = vel[idx*3], vy = vel[idx*3+1], vz = vel[idx*3+2];

            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            // Pairwise LJ forces
            for (int j = 0; j < N; j++) {
                if (j == idx) continue;
                float dx = px - pos[j*3];
                float dy = py - pos[j*3+1];
                float dz = pz - pos[j*3+2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                float sigma = pairSigma[idx*N + j];
                float cutoff = sigma * 2.5f;

                if (dist < cutoff && dist > 0.005f) {
                    float r_inv = 1.0f / fmaxf(dist, 0.01f);
                    float sr = sigma * r_inv;
                    float sr6 = sr * sr * sr * sr * sr * sr;
                    float fmag = 24.0f * ljStrength * r_inv * (2.0f * sr6 * sr6 - sr6);
                    float inv_d = 1.0f / dist;
                    fx += fmag * dx * inv_d;
                    fy += fmag * dy * inv_d;
                    fz += fmag * dz * inv_d;
                }
            }

            // Wall force: closest point on major circle
            float theta = atan2f(py, px);
            float cx = majorR * cosf(theta);
            float cy = majorR * sinf(theta);
            float cz = 0.0f;
            float wx = px - cx, wy = py - cy, wz = pz - cz;
            float wdist = sqrtf(wx*wx + wy*wy + wz*wz);
            float wallDist = tubeR - wdist;
            float wallRange = tubeR * wallRangeFrac;

            if (wallDist < wallRange && wdist > 0.001f) {
                float t = fmaxf(0.001f, wallDist / wallRange);
                float wfmag = -ljStrength * wallForceScale / (t * t);
                float inv_w = 1.0f / wdist;
                fx += wfmag * wx * inv_w;
                fy += wfmag * wy * inv_w;
                fz += wfmag * wz * inv_w;
            }

            // Integration with damping
            vx = (vx + fx * dt) * damping;
            vy = (vy + fy * dt) * damping;
            vz = (vz + fz * dt) * damping;

            float speed = sqrtf(vx*vx + vy*vy + vz*vz);
            if (speed > maxSpeed) {
                float s = maxSpeed / speed;
                vx *= s; vy *= s; vz *= s;
            }

            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;

            // Hard torus constraint
            theta = atan2f(py, px);
            cx = majorR * cosf(theta);
            cy = majorR * sinf(theta);
            wx = px - cx; wy = py - cy; wz = pz;
            wdist = sqrtf(wx*wx + wy*wy + wz*wz);
            if (wdist > tubeR * 0.99f) {
                float inv_w = 1.0f / fmaxf(wdist, 1e-8f);
                float nx = wx * inv_w, ny = wy * inv_w, nz = wz * inv_w;
                px = cx + nx * tubeR * 0.98f;
                py = cy + ny * tubeR * 0.98f;
                pz = nz * tubeR * 0.98f;
                float vn = vx*nx + vy*ny + vz*nz;
                vx = (vx - 2.0f*vn*nx) * 0.7f;
                vy = (vy - 2.0f*vn*ny) * 0.7f;
                vz = (vz - 2.0f*vn*nz) * 0.7f;
            }

            pos[idx*3] = px; pos[idx*3+1] = py; pos[idx*3+2] = pz;
            vel[idx*3] = vx; vel[idx*3+1] = vy; vel[idx*3+2] = vz;
        }
        """, "torusStep")

        self.circle_step_kernel = cp.RawKernel(r"""
        extern "C" __global__
        void circleStep(float* pos, float* vel, const float* pairSigma,
                        float ljStrength, float circleLjScale,
                        float majorR, float dt, float circleDamping,
                        float maxOmega, float sigmaArcScale,
                        int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N) return;

            float px = pos[idx*3], py = pos[idx*3+1];
            float vx = vel[idx*3], vy = vel[idx*3+1];
            float myTheta = atan2f(py, px);

            float force = 0.0f;

            for (int j = 0; j < N; j++) {
                if (j == idx) continue;
                float otherTheta = atan2f(pos[j*3+1], pos[j*3]);
                float dtheta = otherTheta - myTheta;
                while (dtheta > 3.14159265f) dtheta -= 6.28318530f;
                while (dtheta < -3.14159265f) dtheta += 6.28318530f;

                float arcDist = fabsf(dtheta) * majorR;
                float sigma3d = pairSigma[idx*N + j];
                float sigmaArc = sigma3d * sigmaArcScale;
                float cutoff = sigmaArc * 3.0f;

                if (arcDist > 0.01f && arcDist < cutoff) {
                    float r = fmaxf(arcDist, 0.01f);
                    float sr = sigmaArc / r;
                    float sr6 = sr * sr * sr * sr * sr * sr;
                    float fmag = 24.0f * ljStrength * circleLjScale / r * (2.0f * sr6 * sr6 - sr6);
                    float sign = dtheta > 0.0f ? 1.0f : -1.0f;
                    force += fmag * sign;
                }
            }

            float tangentX = -sinf(myTheta), tangentY = cosf(myTheta);
            float omega = (vx * tangentX + vy * tangentY) / majorR;
            omega = (omega + force * dt / majorR) * circleDamping;
            if (fabsf(omega) > maxOmega) omega = copysignf(maxOmega, omega);

            myTheta += omega * dt;
            pos[idx*3] = majorR * cosf(myTheta);
            pos[idx*3+1] = majorR * sinf(myTheta);
            pos[idx*3+2] = 0.0f;
            vel[idx*3] = -sinf(myTheta) * omega * majorR;
            vel[idx*3+1] = cosf(myTheta) * omega * majorR;
            vel[idx*3+2] = 0.0f;
        }
        """, "circleStep")

    def simulate_circle_step_gpu(self):
        """One circle phase step on GPU."""
        n = self.n_cities
        threads = 256
        blocks = (n + threads - 1) // threads

        self.circle_step_kernel(
            (blocks,), (threads,),
            (
                self.pos, self.vel, self.pair_sigma,
                cp.float32(self.lj_strength),
                cp.float32(self.circle_lj_scale),
                cp.float32(self.R), cp.float32(self.dt),
                cp.float32(self.circle_damping),
                cp.float32(self.max_omega),
                cp.float32(self.sigma_arc_scale),
                cp.int32(n),
            )
        )
        self.time += self.dt
