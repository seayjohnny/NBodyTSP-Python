"""
GPU Physics Engine for N-Body TSP Simulator

Handles force calculations and physics integration using GPU acceleration.
"""

import numpy as np
from typing import Optional, Tuple, TypedDict

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    print("Warning: CuPy not available. Falling back to CPU (NumPy)")
    GPU_AVAILABLE = False

from tsp_nbody.advanced_features import ImprovedWallForces, AdaptiveBubbles


class NBodyPhysicsOptions(TypedDict, total=False):
    """Options for configuring the N-body physics engine."""
    use_gpu: bool
    DAMP: float
    MASS: float
    WALL_STRENGTH: float
    FORCE_CUTOFF: float
    DT: float
    DR: float
    force_mode: str  # 'piecewise' or 'smooth'
    slope_repulsion: float
    mag_attraction: float
    force_cutoff_extra: float
    p: float
    q: float
    m: float
    lower_pressure_limit: float
    upper_pressure_limit: float
    use_improved_walls: bool
    use_adaptive_bubbles: bool

default_nbody_options: NBodyPhysicsOptions = {
    'use_gpu': GPU_AVAILABLE,
    'DAMP': 20.0,
    'MASS': 80.0,
    'WALL_STRENGTH': 20000.0,
    'FORCE_CUTOFF': 100000.0,
    'DT': 0.01,
    'DR': 0.01,
    'force_mode': 'piecewise',
    'slope_repulsion': 50.0,
    'mag_attraction': 0.5,
    'force_cutoff_extra': 0.10,
    'p': 6,
    'q': 12,
    'm': -0.05,
    "lower_pressure_limit": 1.0,
    "upper_pressure_limit": 10.0,
    'use_improved_walls': False,
    'use_adaptive_bubbles': False,
}


class NBodyPhysicsEngine:
    """GPU-accelerated N-body physics simulation for TSP."""

    def __init__(
        self,
        coords: np.ndarray,
        options: NBodyPhysicsOptions = default_nbody_options,
    ):
        """
        Initialize the physics engine.

        Args:
            coords: Original city coordinates (n, 2)
            use_gpu: Whether to use GPU acceleration
        """

        self.use_gpu = options.get('use_gpu', GPU_AVAILABLE) and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        # Convert to GPU if needed
        self.coords = self.xp.asarray(coords, dtype=self.xp.float32)
        self.n_cities = len(coords)

        # Physics state
        self.pos = None  # Current positions
        self.vel = None  # Velocities
        self.acc = None  # Accelerations

        # Physics parameters (from original C++ code)
        self.DAMP = options.get('DAMP', 20.0)
        self.MASS = options.get('MASS', 80.0)
        self.WALL_STRENGTH = options.get('WALL_STRENGTH', 20000.0)
        self.FORCE_CUTOFF = options.get('FORCE_CUTOFF', 100000.0)
        self.DT = options.get('DT', 0.01)
        self.DR = options.get('DR', 0.01)

        self.force_mode = options.get('force_mode', "piecewise")

        # TSP piecewise LJ parameters
        self.slope_repulsion = options.get('slope_repulsion', 50.0)
        self.mag_attraction = options.get('mag_attraction', 0.5)
        self.force_cutoff_extra = options.get('force_cutoff_extra', 0.10)

        # TSP smooth LJ parameters
        self.p = options.get('p', 6)
        self.q = options.get('q', 12)
        self.m = options.get('m', -0.05)

        # Advanced features
        self.use_improved_walls = options.get('use_improved_walls', False)  # Recommended: prevents bleeding
        self.use_adaptive_bubbles = options.get('use_adaptive_bubbles', False)  # Enable for dense datasets

        # Initialize improved wall forces
        if self.use_improved_walls:
            self.improved_walls = ImprovedWallForces(
                base_strength=self.WALL_STRENGTH, use_gpu=self.use_gpu
            )
        else:
            self.improved_walls = None

        # Initialize adaptive bubbles
        if self.use_adaptive_bubbles:
            self.adaptive_bubbles = AdaptiveBubbles(
                grid_size=10,
                density_threshold=3.5,
                bubble_strength=20000.0,
                use_gpu=self.use_gpu,
            )
        else:
            self.adaptive_bubbles = None

        # Pressure tracking (for visualization)
        self.current_pressure = 0.0

        if self.use_gpu:
            self._compile_kernels()

        print(f"Physics engine initialized ({'GPU' if self.use_gpu else 'CPU'})")
        print(f"  Cities: {self.n_cities}")
        print(f"  Force mode: {self.force_mode}")
        print(
            f"  Advanced features: improved_walls={'ON' if self.use_improved_walls else 'OFF'}, "
            f"adaptive_bubbles={'ON' if self.use_adaptive_bubbles else 'OFF'}"
        )
        print("\n  N-Body parameters")
        print("-" * 40)

        if self.force_mode == "piecewise":
            print(f"  Slope Repulsion: {self.slope_repulsion}")
            print(f"  Magnitude Attraction: {self.mag_attraction}")
            print(f"  Force Cutoff Extra: {self.force_cutoff_extra}")
        else:
            print(f"  P: {self.p}")
            print(f"  Q: {self.q}")
            print(f"  M: {self.m}")

        print("-" * 40)


    def _compile_kernels(self):
        """Compile the CUDA kernels for GPU acceleration."""
        # Piecewise Lennard-Jones kernel
        # Uses piecewise linear/inverse forces
        self.nbody_piecewise_lj_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void nBodyStepPiecewiseLJ(const float2* shInitPos, float2* shPos, float2* vel, float2* acc,
                              float slopeRepulsion, float magAttraction, float forceCutoffDist,
                              float iR, float oR, int N, float WALL_STRENGTH, float DAMP,
                              float FORCE_CUTOFF, float MASS)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= N) return;

            float2 currentPos = shPos[idx];
            float2 currentVel = vel[idx];
            float2 initPos = shInitPos[idx];

            float2 force = make_float2(0.0f, 0.0f);
            float d, edgeLength, radius, forceMag;

            // N-body interactions (piecewise Lennard-Jones approximation)
            for(int i = 0; i < N; i++)
            {
                if(i != idx)
                {
                    // Current distance
                    float2 diff = make_float2(shPos[i].x - currentPos.x,
                                              shPos[i].y - currentPos.y);
 
                    d = sqrtf(diff.x*diff.x + diff.y*diff.y);
                    if(d < 1e-10f) d = 1e-10f;  // Avoid singularity

                    // Initial distance (equilibrium length for this pair)
                    float2 initDiff = make_float2(shInitPos[i].x - initPos.x,
                                                  shInitPos[i].y - initPos.y);
                    edgeLength = sqrtf(initDiff.x*initDiff.x + initDiff.y*initDiff.y);

                    // Piecewise force calculation
                    // Mimics Lennard-Jones: repulsive when compressed, attractive when extended
                    if(d <= edgeLength)
                    {
                        // Compressed: linear repulsive force (like spring)
                        forceMag = -(edgeLength - d) * slopeRepulsion;
                    }
                    else if(d < forceCutoffDist)
                    {
                        // Extended: weak attractive force (inverse with equilibrium length)
                        // This approximates the attractive tail of LJ
                        forceMag = magAttraction / edgeLength;
                    }
                    else
                    {
                        // Beyond cutoff: no force
                        forceMag = 0.0f;
                    }

                    force.x += forceMag * diff.x / d;
                    force.y += forceMag * diff.y / d;
                }
            }

            // Wall forces
            radius = sqrtf(currentPos.x*currentPos.x + currentPos.y*currentPos.y);
            if(radius < iR && radius > 1e-10f)
            {
                forceMag = WALL_STRENGTH * (iR - radius);
                force.x += forceMag * currentPos.x / radius;
                force.y += forceMag * currentPos.y / radius;
            }
            else if(radius > oR && radius > 1e-10f)
            {
                forceMag = WALL_STRENGTH * (oR - radius);
                force.x += forceMag * currentPos.x / radius;
                force.y += forceMag * currentPos.y / radius;
            }

            // Damping
            force.x -= DAMP * currentVel.x;
            force.y -= DAMP * currentVel.y;

            // Force cutoff (for stability)
            if(fabsf(force.x) > FORCE_CUTOFF) force.x = 0.0f;
            if(fabsf(force.y) > FORCE_CUTOFF) force.y = 0.0f;

            // Update acceleration
            acc[idx] = make_float2(force.x / MASS, force.y / MASS);
        }
        """,
            "nBodyStepPiecewiseLJ",
        )

        self.nbody_smooth_lj_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void nBodyStepSmoothLJ(const float2* shInitPos, float2* shPos, float2* vel, float2* acc,
                              float p, float q, float m,
                              float iR, float oR, int N, float WALL_STRENGTH, float DAMP,
                              float FORCE_CUTOFF, float MASS)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= N) return;

            float2 currentPos = shPos[idx];
            float2 currentVel = vel[idx];
            float2 initPos = shInitPos[idx];

            float2 force = make_float2(0.0f, 0.0f);
            float d, edgeLength, radius, forceMag, h, c;

            // N-body interactions (piecewise Lennard-Jones approximation)
            for(int i = 0; i < N; i++)
            {
                if(i != idx)
                {
                    // Current distance
                    float2 diff = make_float2(shPos[i].x - currentPos.x,
                                              shPos[i].y - currentPos.y);

                    d = sqrtf(diff.x*diff.x + diff.y*diff.y);
                    if(d < 1e-10f) d = 1e-10f;  // Avoid singularity

                    // Initial distance (equilibrium length for this pair)
                    float2 initDiff = make_float2(shInitPos[i].x - initPos.x,
                                                  shInitPos[i].y - initPos.y);
                    edgeLength = sqrtf(initDiff.x*initDiff.x + initDiff.y*initDiff.y);

                    // Smooth Lennard-Jones force calculation
                    h = m * (powf(powf(q/p, 1.0/(q - p))*edgeLength, p))/(1.0 - p/q);
                    c = powf(edgeLength/d, q-p);
                    forceMag = (c - 1) * h / powf(d, p);

                    force.x += forceMag * diff.x / d;
                    force.y += forceMag * diff.y / d;
                }
            }

            // Wall forces
            radius = sqrtf(currentPos.x*currentPos.x + currentPos.y*currentPos.y);
            if(radius < iR && radius > 1e-10f)
            {
                forceMag = WALL_STRENGTH * (iR - radius);
                force.x += forceMag * currentPos.x / radius;
                force.y += forceMag * currentPos.y / radius;
            }
            else if(radius > oR && radius > 1e-10f)
            {
                forceMag = WALL_STRENGTH * (oR - radius);
                force.x += forceMag * currentPos.x / radius;
                force.y += forceMag * currentPos.y / radius;
            }

            // Damping
            force.x -= DAMP * currentVel.x;
            force.y -= DAMP * currentVel.y;

            // Force cutoff (for stability)
            if(fabsf(force.x) > FORCE_CUTOFF) force.x = 0.0f;
            if(fabsf(force.y) > FORCE_CUTOFF) force.y = 0.0f;

            // Update acceleration
            acc[idx] = make_float2(force.x / MASS, force.y / MASS);
        }
        """,
            "nBodyStepSmoothLJ",
        )

    def initialize_physics(self):
        """Set initial conditions for the simulation."""
        # Positions start at original coordinates
        self.pos = self.coords.copy()

        # Velocities start at zero
        self.vel = self.xp.zeros_like(self.pos)

        # Accelerations start at zero
        self.acc = self.xp.zeros_like(self.pos)

        # Move any cities at exact origin slightly off center
        radii = self.xp.sqrt(self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2)
        too_close = radii < 0.001
        if self.xp.any(too_close):
            self.pos[too_close, 0] = 0.001
            self.pos[too_close, 1] = 0.001

        print("Physics initialized: positions, velocities set")

    def compute_wall_forces_cpu(
        self, inner_radius: float, outer_radius: float
    ) -> np.ndarray:
        """Calculate wall forces using CPU."""
        xp = self.xp

        dx = self.pos[:, 0]
        dy = self.pos[:, 1]
        radius = xp.sqrt(dx * dx + dy * dy)

        force = xp.zeros_like(self.pos)

        # Inner wall
        inside_inner = radius < inner_radius
        if xp.any(inside_inner):
            force_mag = self.WALL_STRENGTH * (inner_radius - radius[inside_inner])
            safe_radius = xp.maximum(radius[inside_inner], 1e-10)
            force[inside_inner, 0] = force_mag * dx[inside_inner] / safe_radius
            force[inside_inner, 1] = force_mag * dy[inside_inner] / safe_radius

        # Outer wall
        outside_outer = radius > outer_radius
        if xp.any(outside_outer):
            force_mag = self.WALL_STRENGTH * (outer_radius - radius[outside_outer])
            safe_radius = xp.maximum(radius[outside_outer], 1e-10)
            force[outside_outer, 0] = force_mag * dx[outside_outer] / safe_radius
            force[outside_outer, 1] = force_mag * dy[outside_outer] / safe_radius

        return force

    def compute_bubble_forces_gpu(self, bubbles: Optional[np.ndarray]) -> cp.ndarray:
        """Calculate bubble forces using GPU kernel."""
        force = cp.zeros_like(self.pos)

        if bubbles is None or len(bubbles) == 0:
            return force

        bubbles_xp = cp.asarray(bubbles, dtype=cp.float32)

        float2_dtype = cp.dtype([("x", cp.float32), ("y", cp.float32)])
        float4_dtype = cp.dtype(
            [("x", cp.float32), ("y", cp.float32), ("z", cp.float32), ("w", cp.float32)]
        )

        pos_f2 = self.pos.view(float2_dtype).reshape(-1)
        force_f2 = force.view(float2_dtype).reshape(-1)
        bubbles_f4 = bubbles_xp.view(float4_dtype).reshape(-1)

        threads_per_block = 256
        blocks = (self.n_cities + threads_per_block - 1) // threads_per_block

        self.bubble_kernel(
            (blocks,),
            (threads_per_block,),
            (
                pos_f2,
                force_f2,
                bubbles_f4,
                cp.int32(len(bubbles)),
                cp.float32(self.WALL_STRENGTH),
                cp.int32(self.n_cities),
            ),
        )

        return force

    def compute_bubble_forces_cpu(self, bubbles: Optional[np.ndarray]) -> np.ndarray:
        """Calculate bubble forces using CPU."""
        xp = self.xp
        force = xp.zeros_like(self.pos)

        if bubbles is None or len(bubbles) == 0:
            return force

        bubbles_xp = xp.asarray(bubbles)

        for b in range(len(bubbles_xp)):
            if bubbles_xp[b, 3] < 0.5:  # Check if active
                continue

            bx, by, br = bubbles_xp[b, 0], bubbles_xp[b, 1], bubbles_xp[b, 2]

            dx = bx - self.pos[:, 0]
            dy = by - self.pos[:, 1]
            dist = xp.sqrt(dx * dx + dy * dy)

            inside = dist < br
            if xp.any(inside):
                force_mag = -self.WALL_STRENGTH * (br - dist[inside])
                safe_dist = xp.maximum(dist[inside], 1e-10)
                force[inside, 0] += force_mag * dx[inside] / safe_dist
                force[inside, 1] += force_mag * dy[inside] / safe_dist

        return force

    def compute_damping_forces(self) -> Tuple:
        """Calculate damping forces."""
        force_x = -self.DAMP * self.vel[:, 0]
        force_y = -self.DAMP * self.vel[:, 1]
        return self.xp.stack([force_x, force_y], axis=1)

    def integrate_step_tsp_gpu(self, inner_radius: float, outer_radius: float):
        """Perform TSP integration using GPU kernel."""
        # Create structured dtype for float2
        float2_dtype = cp.dtype([("x", cp.float32), ("y", cp.float32)])

        # Create views as float2 structured arrays
        pos_f2 = self.pos.view(float2_dtype).reshape(-1)
        vel_f2 = self.vel.view(float2_dtype).reshape(-1)
        acc_f2 = self.acc.view(float2_dtype).reshape(-1)
        init_pos_f2 = self.coords.view(float2_dtype).reshape(-1)

        threads_per_block = 256
        blocks = (self.n_cities + threads_per_block - 1) // threads_per_block

        # Use reduced wall strength if improved walls are enabled
        wall_strength = 0.0 if self.use_improved_walls else self.WALL_STRENGTH

        if self.force_mode == "smooth":
            # Smooth Lennard-Jones kernel
            self.nbody_smooth_lj_kernel(
                (blocks,),
                (threads_per_block,),
                (
                    init_pos_f2,
                    pos_f2,
                    vel_f2,
                    acc_f2,
                    cp.float32(self.p),
                    cp.float32(self.q),
                    cp.float32(self.m),
                    cp.float32(inner_radius),
                    cp.float32(outer_radius),
                    cp.int32(self.n_cities),
                    cp.float32(wall_strength),
                    cp.float32(self.DAMP),
                    cp.float32(self.FORCE_CUTOFF),
                    cp.float32(self.MASS),
                ),
            )
        else:
            # Piecewise Lennard-Jones kernel
            self.nbody_piecewise_lj_kernel(
                (blocks,),
                (threads_per_block,),
                (
                    init_pos_f2,
                    pos_f2,
                    vel_f2,
                    acc_f2,
                    cp.float32(self.slope_repulsion),
                    cp.float32(self.mag_attraction),
                    cp.float32(self.force_cutoff_extra),
                    cp.float32(inner_radius),
                    cp.float32(outer_radius),
                    cp.int32(self.n_cities),
                    cp.float32(wall_strength),
                    cp.float32(self.DAMP),
                    cp.float32(self.FORCE_CUTOFF),
                    cp.float32(self.MASS),
                ),
            )

        # Add improved wall forces if enabled
        if self.use_improved_walls:
            wall_force = self.improved_walls.compute_forces(
                self.pos, inner_radius, outer_radius
            )
            self.acc += wall_force / self.MASS

        # Add adaptive bubble forces if enabled
        if self.use_adaptive_bubbles:
            bubble_force = self.adaptive_bubbles.compute_forces(self.pos)
            self.acc += bubble_force / self.MASS

        # Update velocities and positions
        self.vel += self.acc * self.DT
        self.pos += self.vel * self.DT

    def integrate_step_tsp_cpu(self, inner_radius: float, outer_radius: float):
        """Perform TSP integration using CPU."""
        xp = self.xp
        n = self.n_cities

        force = xp.zeros_like(self.pos)

        if self.force_mode == "smooth":
            # Smooth Lennard-Jones implementation
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    # Current distance
                    diff = self.pos[j] - self.pos[i]
                    d = xp.linalg.norm(diff)
                    d = max(d, 1e-10)

                    # Initial distance (equilibrium length)
                    init_diff = self.coords[j] - self.coords[i]
                    l = xp.linalg.norm(init_diff)

                    # Smooth Lennard-Jones force calculation
                    h = self.m * ( ( (self.q / self.p) ** (1.0 / (self.q - self.p)) * l ) ** self.p ) / (1.0 - self.p / self.q)
                    c = (l / d) ** (self.q - self.p)
                    force_mag = (c - 1) * h / (d ** self.p)

                    force[i] += force_mag * diff / d
        else:
            # Piecewise Lennard-Jones implementation
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    # Current distance
                    diff = self.pos[j] - self.pos[i]
                    d = xp.linalg.norm(diff)
                    d = max(d, 1e-10)

                    # Initial distance (equilibrium length)
                    init_diff = self.coords[j] - self.coords[i]
                    l = xp.linalg.norm(init_diff)

                    # Piecewise force calculation
                    if d <= l:
                        # Compressed: linear repulsive force
                        force_mag = -(l - d) * self.slope_repulsion
                    elif d < self.force_cutoff_extra:
                        # Extended: weak attractive force
                        force_mag = self.mag_attraction / l
                    else:
                        # Beyond cutoff
                        force_mag = 0.0

                    force[i] += force_mag * diff / d

        # Wall forces (improved or standard)
        if self.use_improved_walls:
            force += self.improved_walls.compute_forces(
                self.pos, inner_radius, outer_radius
            )
        else:
            force += self.compute_wall_forces_cpu(inner_radius, outer_radius)

        # Adaptive bubble forces
        if self.use_adaptive_bubbles:
            force += self.adaptive_bubbles.compute_forces(self.pos)

        # Damping
        force += self.compute_damping_forces()

        # Force cutoff
        force[force > self.FORCE_CUTOFF] = 0.0

        # Compute acceleration
        self.acc = force / self.MASS

        # Velocity Verlet integration
        self.vel += self.acc * self.DT
        self.pos += self.vel * self.DT

    def integrate_step(
        self,
        inner_radius: float,
        outer_radius: float,
        bubbles: Optional[np.ndarray] = None,
    ):
        """
        Perform one physics integration step.

        Args:
            inner_radius: Inner wall radius
            outer_radius: Outer wall radius
            bubbles: Optional bubble array
        """
        if self.use_gpu:
            self.integrate_step_tsp_gpu(inner_radius, outer_radius)
        else:
            self.integrate_step_tsp_cpu(inner_radius, outer_radius)

    def get_positions_cpu(self) -> np.ndarray:
        """Get current positions as a CPU numpy array."""
        if self.use_gpu:
            return cp.asnumpy(self.pos)
        return self.pos.copy()

    def get_velocities_cpu(self) -> np.ndarray:
        """Get current velocities as a CPU numpy array."""
        if self.use_gpu:
            return cp.asnumpy(self.vel)
        return self.vel.copy()

    def compute_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of the system."""
        xp = self.xp
        v_squared = xp.sum(self.vel**2, axis=1)
        ke = 0.5 * self.MASS * xp.sum(v_squared)

        if self.use_gpu:
            return float(cp.asnumpy(ke))
        return float(ke)

    def compute_pressure_on_outer_wall(self, outer_radius: float) -> float:
        """Calculate pressure exerted on the outer wall."""
        xp = self.xp

        radii = xp.sqrt(self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2)

        beyond = radii - outer_radius
        beyond = xp.maximum(beyond, 0.0)
        total_pressure = xp.sum(beyond)

        circumference = 2.0 * np.pi * outer_radius
        pressure = float(total_pressure) / circumference

        if self.use_gpu:
            pressure = float(cp.asnumpy(pressure))

        # Store for visualization
        self.current_pressure = pressure

        return pressure

    def update_bubble_density(self, outer_radius: float):
        """
        Update adaptive bubble density analysis.

        Call this periodically (e.g., every 10 steps) when using adaptive bubbles.

        Args:
            outer_radius: Current outer wall radius (used for bounds)
        """
        if not self.use_adaptive_bubbles:
            return

        # Get current positions
        pos_cpu = self.get_positions_cpu()

        # Set bounds based on outer radius
        bounds_min = np.array([-outer_radius, -outer_radius], dtype=np.float32)
        bounds_max = np.array([outer_radius, outer_radius], dtype=np.float32)

        # Update bubble system
        self.adaptive_bubbles.update_density(pos_cpu, bounds_min, bounds_max)

    def get_bubble_data_for_renderer(self) -> Optional[np.ndarray]:
        """
        Get bubble data in format expected by renderer.

        Returns:
            Array of shape (n, 4) with (x, y, radius, active) for each bubble,
            or None if no bubbles
        """
        if not self.use_adaptive_bubbles:
            return None

        bubbles = self.adaptive_bubbles.get_bubble_info()
        if len(bubbles) == 0:
            return None

        # Convert from (cx, cy, r, strength) to (x, y, radius, active)
        bubble_array = np.zeros((len(bubbles), 4), dtype=np.float32)
        for i, (cx, cy, r, strength) in enumerate(bubbles):
            bubble_array[i] = [cx, cy, r, 1.0 if strength > 1.0 else 0.0]

        return bubble_array

    def get_final_tour(self) -> np.ndarray:
        """Get the final tour order based on angles."""
        pos = self.get_positions_cpu()
        angles = np.arctan2(pos[:, 1], pos[:, 0])
        return np.argsort(angles)

    def compute_tour_cost(self) -> float:
        """Compute the cost of the current tour."""
        tour = self.get_final_tour()
        init_pos = self.coords if not self.use_gpu else cp.asnumpy(self.coords)

        cost = 0.0
        for i in range(len(tour)):
            curr_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            diff = init_pos[next_city] - init_pos[curr_city]
            cost += np.sqrt(np.sum(diff**2))

        return cost


# Example usage
if __name__ == "__main__":
    # Test TSP mode
    print("=" * 60)
    print("Testing TSP Mode")
    print("=" * 60)

    n_cities = 48
    np.random.seed(42)
    coords = np.random.randn(n_cities, 2).astype(np.float32) * 10.0

    engine = NBodyPhysicsEngine(coords, use_gpu=True)

    p = 3.5
    q = p + 2.5
    h = 1.5
    engine.set_parameters(p, q, h)

    engine.initialize_physics()

    init_pos = engine.get_positions_cpu()
    max_dist = np.max(np.sqrt(np.sum(init_pos**2, axis=1)))

    inner_radius = 0.0
    outer_radius = max_dist

    print(f"\nSimulation parameters:")
    print(f"  p={p}, q={q}, h={h}")
    print(f"  Initial outer radius: {outer_radius:.4f}")

    print("\nRunning TSP simulation...")
    step = 0
    while inner_radius < outer_radius - engine.DR:
        t = 0.0
        while t < 1.0:
            engine.integrate_step(inner_radius, outer_radius)
            t += engine.DT
            step += 1

        inner_radius += engine.DR

        if int(inner_radius * 100) % 10 == 0:
            ke = engine.compute_kinetic_energy()
            print(f"  Inner radius: {inner_radius:.2f}, KE: {ke:.4f}")

    tour = engine.get_final_tour()
    cost = engine.compute_tour_cost()

    print(f"\nTSP simulation complete:")
    print(f"  Total steps: {step}")
    print(f"  Final tour cost: {cost:.4f}")
