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

from tsp_nbody.advanced_features import AdaptiveBubbles


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
    grid_bins: int
    min_bin_density: int
    use_pressure: bool
    use_density_grid: bool
    use_bubbles: bool
    num_bubbles: int

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
    "grid_bins": 8,
    "min_bin_density": 3,
    'use_pressure': False,
    'use_density_grid': False,
    'use_bubbles': False,
    'num_bubbles': 3,
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

        # Pressure parameters
        self.lower_pressure_limit = options.get('lower_pressure_limit', 500)
        self.upper_pressure_limit = options.get('upper_pressure_limit', 1000)
        self.use_pressure = options.get('use_pressure', True)

        # Bubble and density grid options
        self.use_density_grid = options.get('use_density_grid', False)
        self.use_bubbles = options.get('use_bubbles', False)  # Enable for dense datasets
        self.num_bubbles = options.get('num_bubbles', 3)

        # Initialize adaptive bubbles
        if self.use_bubbles:
            self.adaptive_bubbles = AdaptiveBubbles(
                self.coords,
                grid_size=10,
                density_threshold=0,
                num_bubbles=self.num_bubbles,
                use_gpu=self.use_gpu,
            )
        else:
            self.adaptive_bubbles = None

        # Pressure tracking (for visualization)
        self.current_pressure = 0.0

        # Density grid tracking (for visualization)
        self.grid_bins = options.get('grid_bins', 8)  # 8x8 grid like CUDA version
        self.grid_bounds = np.linspace(-1.0, 1.0, self.grid_bins + 1)
        self.density_grid = np.zeros(self.grid_bins * self.grid_bins, dtype=np.int32)
        self.density_centers = np.zeros((self.grid_bins * self.grid_bins, 2), dtype=np.float32)

        # Bubble management (like CUDA version)
        self.min_bin_density = options.get('min_bin_density', 3)  # Minimum density to spawn a bubble
        self.max_bubbles = self.num_bubbles  # Maximum number of bubbles (spawn at top B densest cells)
        self.bubbles = np.zeros((self.grid_bins * self.grid_bins, 4), dtype=np.float32)  # (x, y, radius, active)
        self.bubbles_enabled = False  # Will be enabled when pressure is right

        if self.use_gpu:
            self._compile_kernels()

        print(f"Physics engine initialized ({'GPU' if self.use_gpu else 'CPU'})")
        print(f"  Cities: {self.n_cities}")
        print(f"  Force mode: {self.force_mode}")
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
                              float4* bubbles, int numBubbles,
                              float slopeRepulsion, float magAttraction, float forceCutoffDist,
                              float iR, float oR, int N, float WALL_STRENGTH, float DAMP,
                              float FORCE_CUTOFF, float MASS, float DT)
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

            __syncthreads();

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

            __syncthreads();

            // Bubble forces
            for(int b = 0; b < numBubbles; b++)
            {
                float4 bubble = bubbles[b];

                // Only apply force if bubble is active
                if(bubble.w > 0.5f)
                {
                    float2 bubbleCenter = make_float2(bubble.x, bubble.y);
                    float bubbleRadius = bubble.z;

                    float2 diff = make_float2(currentPos.x - bubbleCenter.x,
                                              currentPos.y - bubbleCenter.y);
                    float dist = sqrtf(diff.x*diff.x + diff.y*diff.y);

                    if(dist < bubbleRadius && dist > 1e-10f)
                    {
                        float penetration = bubbleRadius - dist;
                        float forceMag = WALL_STRENGTH * penetration;

                        force.x += forceMag * diff.x / dist;
                        force.y += forceMag * diff.y / dist;
                    }
                }
            }

            __syncthreads();

            // Damping
            force.x -= DAMP * currentVel.x;
            force.y -= DAMP * currentVel.y;

            // Force cutoff (for stability)
            if(fabsf(force.x) > FORCE_CUTOFF) force.x = 0.0f;
            if(fabsf(force.y) > FORCE_CUTOFF) force.y = 0.0f;

            // Update acceleration
            acc[idx] = make_float2(force.x / MASS, force.y / MASS);

            __syncthreads();

            // Update positions and velocities. We update position first to utilize
            // leap-frog integration.
            vel[idx].x += acc[idx].x * DT;
            vel[idx].y += acc[idx].y * DT;

            shPos[idx].x += currentVel.x * DT;
            shPos[idx].y += currentVel.y * DT;
        }
        """,
            "nBodyStepPiecewiseLJ",
        )

        # Density grid kernel (for visualization)
        self.density_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void computeDensity(const float2* pos, int* densityGrid, float2* densityCenters,
                            const float* bounds, int bins, int N)
        {
            int x = blockIdx.x;
            int y = blockIdx.y;
            int id = x + y * bins;

            // Get bounds for this cell
            float xr_min = bounds[x];
            float xr_max = bounds[x + 1];
            float yr_min = bounds[bins - 1 - y];
            float yr_max = bounds[bins - y];

            float xBar = 0.0f;
            float yBar = 0.0f;
            int count = 0;

            // Count particles in this cell
            for(int i = 0; i < N; i++)
            {
                if(pos[i].x >= xr_min && pos[i].x < xr_max &&
                   pos[i].y >= yr_min && pos[i].y < yr_max)
                {
                    count++;
                    xBar += pos[i].x;
                    yBar += pos[i].y;
                }
            }

            densityGrid[id] = count;

            // Compute center if density > 0
            if(count > 0)
            {
                densityCenters[id].x = xBar / count;
                densityCenters[id].y = yBar / count;
            }
            else
            {
                densityCenters[id].x = 0.0f;
                densityCenters[id].y = 0.0f;
            }
        }
        ''', 'computeDensity')

        self.nbody_smooth_lj_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void nBodyStepSmoothLJ(const float2* shInitPos, float2* shPos, float2* vel, float2* acc,
                              float p, float q, float h,
                              float iR, float oR, int N, float WALL_STRENGTH, float DAMP,
                              float FORCE_CUTOFF, float MASS, float DT)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx >= N) return;

            float2 currentPos = shPos[idx];
            float2 currentVel = vel[idx];
            float2 initPos = shInitPos[idx];

            float2 force = make_float2(0.0f, 0.0f);
            float d, edgeLength, radius, forceMag, g;

            // N-body interactions (piecewise Lennard-Jones approximation)
            for(int i = 0; i < N; i++)
            {
                if(i != idx)
                {
                    // Current distance
                    float2 diff = make_float2(shPos[i].x - currentPos.x,
                                              shPos[i].y - currentPos.y);

                    d = sqrtf(diff.x*diff.x + diff.y*diff.y);
                    if(d < 1e-10f) d = 1e-5f;  // Avoid singularity

                    // Initial distance (equilibrium length for this pair)
                    float2 initDiff = make_float2(shInitPos[i].x - initPos.x,
                                                  shInitPos[i].y - initPos.y);
                    edgeLength = sqrtf(initDiff.x*initDiff.x + initDiff.y*initDiff.y);

                    // Smooth Lennard-Jones force calculation
                    g = h * powf(edgeLength, q-p);
                    forceMag = g/powf(d, q) - h/powf(d, p);

                    force.x += forceMag * diff.x / d;
                    force.y += forceMag * diff.y / d;
                }
            }

            __syncthreads();

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

            __syncthreads();

            // Damping
            force.x -= DAMP * currentVel.x;
            force.y -= DAMP * currentVel.y;

            // Force cutoff (for stability)
            if(fabsf(force.x) > FORCE_CUTOFF) force.x = 0.0f;
            if(fabsf(force.y) > FORCE_CUTOFF) force.y = 0.0f;

            // Update acceleration
            acc[idx] = make_float2(force.x / MASS, force.y / MASS);

            __syncthreads();

            // Update positions and velocities. We update position first to utilize
            // leap-frog integration.
            shPos[idx].x += currentVel.x * DT;
            shPos[idx].y += currentVel.y * DT;
            vel[idx].x += acc[idx].x * DT;
            vel[idx].y += acc[idx].y * DT;
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

        # Move all cities at least a small distance away from origin to avoid singularities
        radii = self.xp.sqrt(self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2)
        too_close = radii < 0.001
        if self.xp.any(too_close):
            self.pos[too_close, 0] = 0.001
            self.pos[too_close, 1] = 0.001

        # Add a tiny perturbation to avoid exact overlaps
        # perturbation = 1e-5 * self.xp.random.randn(self.n_cities, 2).astype(self.xp.float32)
        # self.pos += perturbation

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
        float4_dtype = cp.dtype([("x", cp.float32), ("y", cp.float32), ("z", cp.float32), ("w", cp.float32)])

        # Create views as float2 structured arrays
        pos_f2 = self.pos.view(float2_dtype).reshape(-1)
        vel_f2 = self.vel.view(float2_dtype).reshape(-1)
        acc_f2 = self.acc.view(float2_dtype).reshape(-1)
        init_pos_f2 = self.coords.view(float2_dtype).reshape(-1)

        # Get active bubbles for GPU
        if self.bubbles_enabled:
            # Convert bubbles from normalized space back to world space for kernel
            bubbles_cpu = self.bubbles.copy()
            bubbles_cpu[:, 0:2] *= outer_radius  # Denormalize x, y positions
            bubbles_cpu[:, 2] *= outer_radius    # Denormalize radius

            bubbles_gpu = cp.asarray(bubbles_cpu, dtype=cp.float32)
            bubbles_f4 = bubbles_gpu.view(float4_dtype).reshape(-1)
            num_bubbles = len(bubbles_f4)
        else:
            bubbles_f4 = cp.zeros(1, dtype=float4_dtype)
            num_bubbles = 0

        threads_per_block = 256
        blocks = (self.n_cities + threads_per_block - 1) // threads_per_block

        # Use reduced wall strength if improved walls are enabled
        wall_strength = self.WALL_STRENGTH

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
                    cp.float32(self.DT)
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
                    bubbles_f4,
                    cp.int32(num_bubbles),
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
                    cp.float32(self.DT)
                ),
            )

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

        force += self.compute_wall_forces_cpu(inner_radius, outer_radius)

        # Adaptive bubble forces
        if self.use_bubbles:
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

    def get_accelerations_cpu(self) -> np.ndarray:
        """Get current accelerations as a CPU numpy array."""
        if self.use_gpu:
            return cp.asnumpy(self.acc)
        return self.acc.copy()

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
        pressure = float(total_pressure) * float(self.WALL_STRENGTH) / circumference

        if self.use_gpu:
            pressure = float(cp.asnumpy(pressure))

        # Store for visualization
        self.current_pressure = pressure

        return pressure

    def compute_density_grid_gpu(self, outer_radius: float = 1.0):
        """
        Compute density grid on GPU.

        Args:
            outer_radius: Current outer wall radius for normalization
        """
        # Normalize positions to [-1, 1] space (grid is in normalized coords)
        pos_normalized = self.pos / outer_radius

        # Create float2 views
        float2_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32)])
        pos_f2 = pos_normalized.view(float2_dtype).reshape(-1)

        # Allocate GPU arrays
        density_gpu = cp.zeros(self.grid_bins * self.grid_bins, dtype=cp.int32)
        centers_gpu = cp.zeros(self.grid_bins * self.grid_bins, dtype=float2_dtype)
        bounds_gpu = cp.asarray(self.grid_bounds, dtype=cp.float32)

        # Launch kernel with 2D grid
        grid_dim = (self.grid_bins, self.grid_bins)

        self.density_kernel(
            grid_dim, (1,),
            (pos_f2, density_gpu, centers_gpu, bounds_gpu,
             cp.int32(self.grid_bins), cp.int32(self.n_cities))
        )

        # Copy results back
        self.density_grid = cp.asnumpy(density_gpu)
        centers_2d = centers_gpu.view(cp.float32).reshape(-1, 2)
        self.density_centers = cp.asnumpy(centers_2d)

    def compute_density_grid_cpu(self, outer_radius: float = 1.0):
        """
        Compute density grid on CPU.

        Args:
            outer_radius: Current outer wall radius for normalization
        """
        # Reset grid
        self.density_grid.fill(0)
        self.density_centers.fill(0)

        # Normalize positions to [-1, 1] space
        pos_normalized = self.pos / outer_radius if outer_radius > 0 else self.pos

        # Temporary arrays for computing centers
        x_sums = np.zeros(self.grid_bins * self.grid_bins, dtype=np.float32)
        y_sums = np.zeros(self.grid_bins * self.grid_bins, dtype=np.float32)

        # Bin particles
        for i in range(self.n_cities):
            x, y = pos_normalized[i, 0], pos_normalized[i, 1]

            # Find grid cell
            x_idx = np.searchsorted(self.grid_bounds[1:], x)
            y_idx = self.grid_bins - 1 - np.searchsorted(self.grid_bounds[1:], y)

            # Clamp to grid
            x_idx = max(0, min(x_idx, self.grid_bins - 1))
            y_idx = max(0, min(y_idx, self.grid_bins - 1))

            cell_id = x_idx + y_idx * self.grid_bins

            self.density_grid[cell_id] += 1
            x_sums[cell_id] += x
            y_sums[cell_id] += y

        # Compute centers
        for i in range(self.grid_bins * self.grid_bins):
            if self.density_grid[i] > 0:
                self.density_centers[i, 0] = x_sums[i] / self.density_grid[i]
                self.density_centers[i, 1] = y_sums[i] / self.density_grid[i]

    def update_density_grid(self, outer_radius: float = 1.0):
        """
        Update the density grid based on current particle positions.

        Args:
            outer_radius: Current outer wall radius for normalization
        """
        if not self.use_density_grid:
            return

        if self.use_gpu:
            self.compute_density_grid_gpu(outer_radius)
        else:
            self.compute_density_grid_cpu(outer_radius)

    def get_density_grid_for_renderer(self) -> Optional[np.ndarray]:
        """
        Get density grid in format expected by renderer.

        Returns:
            Array of density values for each grid cell
        """
        if not self.use_density_grid:
            return None

        return self.density_grid.copy()

    def initialize_bubbles(self, inner_radius: float, outer_radius: float,
                          manual_positions: Optional[list] = None):
        """
        Initialize bubbles at manual positions or at the top B most dense grid cells.

        Args:
            inner_radius: Starting radius for bubbles (in world space)
            outer_radius: Current outer radius (for normalization)
            manual_positions: Optional list of (x, y) tuples for manual bubble placement
        """
        if not self.use_bubbles:
            return


        # Reset all bubbles
        self.bubbles[:] = 0.0

        # Normalize inner radius to [-1, 1] space
        inner_r_norm = inner_radius / outer_radius if outer_radius > 0 else inner_radius

        # If manual positions provided, use them instead of density-based placement
        if manual_positions is not None and len(manual_positions) > 0:
            print(f"  Initializing {len(manual_positions)} manual bubbles at inner_radius={inner_radius:.4f} (norm={inner_r_norm:.4f})")

            # Place bubbles at manual positions (already in normalized [-1, 1] space)
            for i, (x, y) in enumerate(manual_positions):
                if i >= len(self.bubbles):
                    print(f"  Warning: Too many manual bubbles ({len(manual_positions)}), only using first {len(self.bubbles)}")
                    break

                self.bubbles[i, 0] = x  # x position (normalized)
                self.bubbles[i, 1] = y  # y position (normalized)
                self.bubbles[i, 2] = inner_r_norm  # radius (normalized)
                self.bubbles[i, 3] = 1.0  # active
        else:
            # Use density-based placement (original logic)
            # Find cells that meet minimum density threshold
            dense_cells = np.where(self.density_grid >= self.min_bin_density)[0]

            if len(dense_cells) > 0:
                # Get densities of those cells
                dense_cell_densities = self.density_grid[dense_cells]

                # Sort by density (descending) and take top MAX_BUBBLES
                sorted_indices = np.argsort(dense_cell_densities)[::-1]
                top_cells = dense_cells[sorted_indices[:self.max_bubbles]]

                # Create bubbles at top B densest cells
                for cell_idx in top_cells:
                    self.bubbles[cell_idx, 0] = self.density_centers[cell_idx, 0]  # x (center of mass)
                    self.bubbles[cell_idx, 1] = self.density_centers[cell_idx, 1]  # y (center of mass)
                    self.bubbles[cell_idx, 2] = inner_r_norm  # radius (normalized)
                    self.bubbles[cell_idx, 3] = 1.0  # active

        self.bubbles_enabled = True

        # Count active bubbles
        active_count = int(np.sum(self.bubbles[:, 3] > 0.5))
        if active_count > 0 and (manual_positions is None or len(manual_positions) == 0):
            print(f"  Initialized {active_count} bubbles (top {self.max_bubbles} densest cells) at inner_radius={inner_radius:.4f} (norm={inner_r_norm:.4f})")

    def update_bubbles(self, dr: float, outer_radius: float):
        """
        Update bubble radii and deactivate those that hit the outer wall.

        Args:
            dr: Change in radius (in world space)
            outer_radius: Current outer wall radius (in world space)
        """
        if not self.use_bubbles:
            return

        if not self.bubbles_enabled:
            return

        # Normalize dr to [-1, 1] space
        dr_norm = dr / outer_radius if outer_radius > 0 else dr

        for i in range(len(self.bubbles)):
            if self.bubbles[i, 3] > 0.5:  # If active
                # Grow bubble (in normalized space)
                self.bubbles[i, 2] += dr_norm

                # Deactivate if bubble hits outer wall (outer wall is at 1.0 in normalized space)
                cx, cy, r = self.bubbles[i, 0], self.bubbles[i, 1], self.bubbles[i, 2]
                dist_from_origin = np.sqrt(cx*cx + cy*cy)

                if dist_from_origin + r >= 1.0:  # Outer wall at 1.0 in normalized space
                    self.bubbles[i, 3] = 0.0  # Deactivate

    def get_bubble_data_for_renderer(self) -> Optional[np.ndarray]:
        """
        Get bubble data in format expected by renderer.

        Returns:
            Array of shape (n, 4) with (x, y, radius, active) for each bubble,
            or None if no bubbles
        """
        if not self.use_bubbles:
            return None

        if not self.bubbles_enabled:
            return None

        # Return only active bubbles
        active_mask = self.bubbles[:, 3] > 0.5
        if not np.any(active_mask):
            return None

        return self.bubbles[active_mask].copy()

    def normalize_positions(self, norm_factor: float):
        """
        Normalize all positions by dividing by norm_factor.

        This keeps the outer radius at 1.0 during the simulation.

        Args:
            norm_factor: Factor to divide positions by
        """
        self.pos /= norm_factor

    def get_final_tour(self) -> np.ndarray:
        """Get the final tour order based on angles."""
        pos = self.get_positions_cpu()
        angles = np.arctan2(pos[:, 1], pos[:, 0])
        return np.argsort(angles)

    def get_debug_data(self) -> dict:
        """
        Bundle all debug data for visualization.

        Returns:
            dict: Contains positions, velocities, accelerations, and distances
        """
        pos = self.get_positions_cpu()
        vel = self.get_velocities_cpu()
        acc = self.get_accelerations_cpu()

        # Compute distance from origin
        distances = np.linalg.norm(pos, axis=1)

        return {
            'positions': pos,
            'velocities': vel,
            'accelerations': acc,
            'distances': distances,
        }
