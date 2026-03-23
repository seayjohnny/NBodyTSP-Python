"""
2D Annular Torus Physics Engine.

Same physics as the 3D torus but in 2D:
- R + r = OUTER (constant)
- Tube collapse: r shrinks, R grows
- Wall force based on distance from the major circle at radius R
  (not from origin like the old 2D engine)
- Per-pair sigma LJ: sigma = initial_distance * 2^(-1/6)
- True LJ 12-6 force with cutoff at 2.5*sigma
- Multiplicative damping + velocity clamp
- GPU (CUDA) is the primary compute path, CPU is fallback

Projected to 2D, the torus annulus is the ring between R-r and R+r.
The wall force pushes particles toward the circle at radius R.
"""

import numpy as np
from typing import Optional

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

TAU = 2.0 * np.pi


class AnnularPhysicsEngine:
    """2D annular torus physics — same model as 3D torus, projected to 2D."""

    def __init__(self, coords: np.ndarray, options: Optional[dict] = None):
        opts = options or {}

        self.use_gpu = opts.get('use_gpu', True) and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.original_coords = np.asarray(coords, dtype=np.float32)
        self.n_cities = len(coords)

        # Torus geometry (same as 3D)
        self.OUTER = opts.get('outer_radius', 3.0)
        self.R = 0.0
        self.r = 0.0
        self.r0 = 0.0

        # Physics (same params as torus)
        self.shrink_rate = opts.get('shrink_rate', 0.10)
        self.epsilon = opts.get('epsilon', 0.08)
        self.lj_strength = opts.get('lj_strength', 1.0)
        self.dt = opts.get('dt', 0.004)
        self.damping = opts.get('damping', 0.97)
        self.max_speed = opts.get('max_speed', 2.5)
        self.wall_range_frac = opts.get('wall_range_frac', 0.05)
        self.wall_force_scale = opts.get('wall_force_scale', 8.0)

        # 2D state
        self.pos = None   # (N, 2)
        self.vel = None   # (N, 2)
        self.pair_sigma = None  # (N, N)

        # Simulation state
        self.collapsing = False
        self.collapsed = False
        self.time = 0.0
        self.found_tour = None

        if self.use_gpu:
            self._compile_kernels()

        print(f"Annular physics engine initialized ({'GPU' if self.use_gpu else 'CPU'})")
        print(f"  Cities: {self.n_cities}, OUTER: {self.OUTER}")

    def _set_tube_radius(self, r: float):
        self.r = max(0.01, r)
        self.R = self.OUTER - self.r

    def initialize_physics(self):
        """Place particles at original coords, centered, scaled to fit annulus."""
        self.collapsed = False
        self.collapsing = False
        self.time = 0.0
        self.found_tour = None

        start_r = self.OUTER * 0.48
        self._set_tube_radius(start_r)
        self.r0 = self.r

        coords = self.original_coords
        n = self.n_cities

        # Center on centroid
        cx, cy = np.mean(coords[:, 0]), np.mean(coords[:, 1])
        positions = coords.copy()
        positions[:, 0] -= cx
        positions[:, 1] -= cy

        # Scale so widest extent fits the annulus (R+r diameter)
        max_extent = np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2))
        if max_extent > 0:
            scale = (self.R + self.r * 0.8) / max_extent
            positions *= scale

        self.pos = self.xp.asarray(positions, dtype=self.xp.float32)
        self.vel = self.xp.zeros((n, 2), dtype=self.xp.float32)

        # Per-pair sigma (same as torus)
        self._compute_pair_sigma()

        print(f"  Annular initialized: R={self.R:.3f}, r={self.r:.3f}")

    def _compute_pair_sigma(self):
        sig_scale = 1.0 / (2.0 ** (1.0 / 6.0))
        pos_cpu = self.pos if not self.use_gpu else cp.asnumpy(self.pos)
        diff = pos_cpu[:, np.newaxis, :] - pos_cpu[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        sigma = (dist * sig_scale).astype(np.float32)
        self.pair_sigma = self.xp.asarray(sigma) if self.use_gpu else sigma

    def start_collapse(self):
        self.collapsing = True

    def integrate_step(self):
        if self.use_gpu:
            self._step_gpu()
        else:
            self._step_cpu()

    def run_substeps(self, k: int):
        for _ in range(k):
            self.integrate_step()

    # =================================================================
    # GPU PATH (primary)
    # =================================================================
    def _step_gpu(self):
        if not self.collapsing:
            self.time += self.dt
            return

        n = self.n_cities
        threads = 256
        blocks = (n + threads - 1) // threads

        self.annular_kernel(
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

        if self.collapsing and not self.collapsed:
            self._set_tube_radius(self.r - self.shrink_rate * self.dt)
            if self.r <= self.epsilon:
                self._set_tube_radius(self.epsilon)
                self.collapsed = True
                self._extract_tour()

        self.time += self.dt

    def _compile_kernels(self):
        self.annular_kernel = cp.RawKernel(r"""
        extern "C" __global__
        void annularStep(float* pos, float* vel, const float* pairSigma,
                         float ljStrength, float majorR, float tubeR,
                         float dt, float damping, float maxSpeed,
                         float wallRangeFrac, float wallForceScale,
                         int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N) return;

            float px = pos[idx*2], py = pos[idx*2+1];
            float vx = vel[idx*2], vy = vel[idx*2+1];
            float fx = 0.0f, fy = 0.0f;

            // Pairwise LJ forces (true LJ 12-6 with per-pair sigma)
            for (int j = 0; j < N; j++) {
                if (j == idx) continue;
                float dx = px - pos[j*2];
                float dy = py - pos[j*2+1];
                float d = sqrtf(dx*dx + dy*dy);

                float sigma = pairSigma[idx*N + j];
                float cutoff = sigma * 2.5f;

                if (d < cutoff && d > 0.01f) {
                    float r_inv = 1.0f / fmaxf(d, 0.01f);
                    float sr = fminf(sigma * r_inv, 100.0f);
                    float sr6 = sr * sr * sr * sr * sr * sr;
                    float fmag = 24.0f * ljStrength * r_inv * (2.0f * sr6 * sr6 - sr6);
                    fmag = fmaxf(-1000.0f, fminf(fmag, 1000.0f));
                    float inv_d = 1.0f / d;
                    fx += fmag * dx * inv_d;
                    fy += fmag * dy * inv_d;
                }
            }

            // Wall force: distance from major circle at radius R
            // In 2D, closest point on major circle = (R*cos(theta), R*sin(theta))
            // where theta = atan2(py, px)
            float theta = atan2f(py, px);
            float cx = majorR * cosf(theta);
            float cy = majorR * sinf(theta);
            float wx = px - cx, wy = py - cy;
            float wdist = sqrtf(wx*wx + wy*wy);  // distance from major circle
            float wallDist = tubeR - wdist;        // distance to tube wall
            float wallRange = tubeR * wallRangeFrac;

            if (wallDist < wallRange && wdist > 0.001f) {
                float t = fmaxf(0.001f, wallDist / wallRange);
                float wfmag = -ljStrength * wallForceScale / (t * t);
                wfmag = fmaxf(-1000.0f, fminf(wfmag, 1000.0f));
                float inv_w = 1.0f / wdist;
                fx += wfmag * wx * inv_w;
                fy += wfmag * wy * inv_w;
            }

            // Hard constraint: clamp to tube
            float radius = sqrtf(px*px + py*py);
            float distFromR = fabsf(radius - majorR);
            if (distFromR > tubeR * 0.99f && radius > 0.001f) {
                // Project back to 98% of tube
                float targetR = majorR + copysignf(tubeR * 0.98f, radius - majorR);
                float inv_r = 1.0f / radius;
                px = px * inv_r * targetR;
                py = py * inv_r * targetR;
                // Reflect radial velocity component
                float nx = px * inv_r, ny = py * inv_r;
                float vn = vx*nx + vy*ny;
                vx = (vx - 2.0f*vn*nx) * 0.7f;
                vy = (vy - 2.0f*vn*ny) * 0.7f;
            }

            // Multiplicative damping integration
            vx = (vx + fx * dt) * damping;
            vy = (vy + fy * dt) * damping;

            // Velocity clamp
            float speed = sqrtf(vx*vx + vy*vy);
            if (speed > maxSpeed) {
                float s = maxSpeed / speed;
                vx *= s; vy *= s;
            }

            px += vx * dt;
            py += vy * dt;

            // Final hard clamp (post-integration)
            radius = sqrtf(px*px + py*py);
            distFromR = fabsf(radius - majorR);
            if (distFromR > tubeR * 0.99f && radius > 0.001f) {
                float targetR = majorR + copysignf(tubeR * 0.98f, radius - majorR);
                float inv_r = 1.0f / radius;
                px = px * inv_r * targetR;
                py = py * inv_r * targetR;
                float nx = px * inv_r, ny = py * inv_r;
                float vn = vx*nx + vy*ny;
                vx = (vx - 2.0f*vn*nx) * 0.7f;
                vy = (vy - 2.0f*vn*ny) * 0.7f;
            }

            pos[idx*2] = px; pos[idx*2+1] = py;
            vel[idx*2] = vx; vel[idx*2+1] = vy;
        }
        """, "annularStep")

    # =================================================================
    # CPU PATH (fallback)
    # =================================================================
    def _step_cpu(self):
        if not self.collapsing:
            self.time += self.dt
            return

        xp = self.xp
        n = self.n_cities
        pos = self.pos
        vel = self.vel

        # Pairwise LJ forces (vectorized)
        diff = pos[:, xp.newaxis, :] - pos[xp.newaxis, :, :]  # (N,N,2)
        dist = xp.sqrt(xp.sum(diff * diff, axis=2))  # (N,N)
        dist_safe = xp.maximum(dist, 0.01)

        sigma = self.pair_sigma
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

        # Wall force: distance from major circle
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
            w_fmag = xp.clip(-self.lj_strength * self.wall_force_scale / (t * t), -1000.0, 1000.0)
            forces[w_active] += w_fmag[:, xp.newaxis] * w_dir

        # Hard constraint: clamp to tube
        radius = xp.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        dist_from_R = xp.abs(radius - self.R)
        violated = (dist_from_R > self.r * 0.99) & (radius > 0.001)
        if xp.any(violated):
            sign = xp.sign(radius[violated] - self.R)
            target_r = self.R + sign * self.r * 0.98
            inv_r = 1.0 / xp.maximum(radius[violated], 1e-8)
            pos[violated, 0] = pos[violated, 0] * inv_r * target_r
            pos[violated, 1] = pos[violated, 1] * inv_r * target_r
            nx = pos[violated, 0] * inv_r
            ny = pos[violated, 1] * inv_r
            vn = vel[violated, 0] * nx + vel[violated, 1] * ny
            vel[violated, 0] = (vel[violated, 0] - 2.0 * vn * nx) * 0.7
            vel[violated, 1] = (vel[violated, 1] - 2.0 * vn * ny) * 0.7

        # Multiplicative damping + velocity clamp
        forces = xp.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        vel[:] = (vel + forces * self.dt) * self.damping
        vel[:] = xp.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        speed = xp.sqrt(xp.sum(vel * vel, axis=1))
        too_fast = speed > self.max_speed
        if xp.any(too_fast):
            vel[too_fast] *= (self.max_speed / speed[too_fast])[:, xp.newaxis]
        pos[:] = pos + vel * self.dt

        # Tube collapse
        if self.collapsing and not self.collapsed:
            self._set_tube_radius(self.r - self.shrink_rate * self.dt)
            if self.r <= self.epsilon:
                self._set_tube_radius(self.epsilon)
                self.collapsed = True
                self._extract_tour()

        self.time += self.dt

    # =================================================================
    # Tour extraction
    # =================================================================
    def _extract_tour(self):
        pos_cpu = self.get_positions_cpu()
        thetas = np.arctan2(pos_cpu[:, 1], pos_cpu[:, 0])
        sorted_indices = np.argsort(thetas)
        self.found_tour = [int(idx + 1) for idx in sorted_indices]
        print(f"  Annular collapse complete at R={self.R:.3f}, r={self.r:.3f}")

    def get_positions_cpu(self) -> np.ndarray:
        if self.use_gpu:
            return cp.asnumpy(self.pos)
        return self.pos.copy()

    def get_velocities_cpu(self) -> np.ndarray:
        if self.use_gpu:
            return cp.asnumpy(self.vel)
        return self.vel.copy()

    def get_final_tour(self) -> np.ndarray:
        pos = self.get_positions_cpu()
        return np.argsort(np.arctan2(pos[:, 1], pos[:, 0]))

    def get_found_tour(self) -> Optional[list]:
        return self.found_tour

    @property
    def phase(self) -> str:
        if self.collapsed:
            return "COLLAPSED"
        elif self.collapsing:
            return "COLLAPSING"
        return "READY"
