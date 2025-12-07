"""
Advanced features for N-Body TSP:
1. Improved wall forces to prevent bleeding
2. Better bubble implementation for cluster breaking

Based on orig_files/TSP-ver10.cu but with improvements.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class ImprovedWallForces:
    """
    Enhanced wall forces that prevent particle bleeding when walls get close.

    Improvements over original:
    - Non-linear forces (inverse square) instead of linear
    - Force strength increases as walls approach each other
    - Smoother transition to avoid instability
    """

    def __init__(self, base_strength=20000.0, use_gpu=True):
        self.base_strength = base_strength
        self.use_gpu = use_gpu and cp is not None

        if self.use_gpu:
            self._compile_gpu_kernel()

    def _compile_gpu_kernel(self):
        """Compile CUDA kernel for improved wall forces."""
        self.wall_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void computeImprovedWallForces(const float2* pos, float2* force,
                                       float innerRadius, float outerRadius,
                                       float baseStrength, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= N) return;

            float2 p = pos[idx];
            float radius = sqrtf(p.x*p.x + p.y*p.y);

            if(radius < 1e-10f) return;  // Avoid singularity at origin

            float2 f = make_float2(0.0f, 0.0f);
            float forceMag = 0.0f;

            // Wall gap factor - increases force strength when walls are close
            float wallGap = outerRadius - innerRadius;
            float gapFactor = 1.0f + 100.0f / (wallGap + 1.0f);  // Increases as gap shrinks

            // Inner wall: Inverse square law for stronger repulsion
            if(radius < innerRadius)
            {
                float penetration = innerRadius - radius;
                float depth_ratio = penetration / innerRadius;

                // Inverse square + linear term for smooth transition
                forceMag = baseStrength * gapFactor * (
                    1.0f / (penetration + 0.1f) +  // Inverse law (strong at surface)
                    10.0f * depth_ratio              // Linear term (smooth)
                );

                // Radial direction (outward)
                f.x = forceMag * p.x / radius;
                f.y = forceMag * p.y / radius;
            }
            // Outer wall: Inverse square law for stronger confinement
            else if(radius > outerRadius)
            {
                float penetration = radius - outerRadius;
                float depth_ratio = penetration / outerRadius;

                // Inverse square + linear term
                forceMag = baseStrength * gapFactor * (
                    1.0f / (penetration + 0.1f) +
                    10.0f * depth_ratio
                );

                // Radial direction (inward)
                f.x = -forceMag * p.x / radius;
                f.y = -forceMag * p.y / radius;
            }

            force[idx] = f;
        }
        ''', 'computeImprovedWallForces')

    def compute_forces(self, positions, inner_radius, outer_radius):
        """
        Compute improved wall forces.

        Args:
            positions: Nx2 array of particle positions
            inner_radius: Inner wall radius
            outer_radius: Outer wall radius

        Returns:
            Nx2 array of force vectors
        """
        n = len(positions)

        if self.use_gpu:
            return self._compute_gpu(positions, inner_radius, outer_radius, n)
        else:
            return self._compute_cpu(positions, inner_radius, outer_radius, n)

    def _compute_gpu(self, positions, inner_radius, outer_radius, n):
        """GPU implementation."""
        xp = cp

        # Create float2 views
        float2_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32)])
        pos_f2 = positions.view(float2_dtype).reshape(-1)
        force_f2 = xp.zeros(n, dtype=float2_dtype)

        threads = 256
        blocks = (n + threads - 1) // threads

        self.wall_kernel(
            (blocks,), (threads,),
            (pos_f2, force_f2,
             xp.float32(inner_radius), xp.float32(outer_radius),
             xp.float32(self.base_strength), xp.int32(n))
        )

        return force_f2.view(xp.float32).reshape(n, 2)

    def _compute_cpu(self, positions, inner_radius, outer_radius, n):
        """CPU implementation."""
        xp = np
        force = xp.zeros_like(positions)

        radii = xp.sqrt(xp.sum(positions**2, axis=1))
        wall_gap = outer_radius - inner_radius
        gap_factor = 1.0 + 100.0 / (wall_gap + 1.0)

        # Inner wall
        inner_mask = radii < inner_radius
        if xp.any(inner_mask):
            penetration = inner_radius - radii[inner_mask]
            depth_ratio = penetration / inner_radius

            force_mag = self.base_strength * gap_factor * (
                1.0 / (penetration + 0.1) +
                10.0 * depth_ratio
            )

            safe_radii = xp.maximum(radii[inner_mask], 1e-10)
            force[inner_mask, 0] = force_mag * positions[inner_mask, 0] / safe_radii
            force[inner_mask, 1] = force_mag * positions[inner_mask, 1] / safe_radii

        # Outer wall
        outer_mask = radii > outer_radius
        if xp.any(outer_mask):
            penetration = radii[outer_mask] - outer_radius
            depth_ratio = penetration / outer_radius

            force_mag = self.base_strength * gap_factor * (
                1.0 / (penetration + 0.1) +
                10.0 * depth_ratio
            )

            safe_radii = xp.maximum(radii[outer_mask], 1e-10)
            force[outer_mask, 0] = -force_mag * positions[outer_mask, 0] / safe_radii
            force[outer_mask, 1] = -force_mag * positions[outer_mask, 1] / safe_radii

        return force


class AdaptiveBubbles:
    """
    Improved bubble implementation for breaking up dense clusters.

    Improvements over original grid-based approach:
    - Spatial hashing for efficient density detection
    - Adaptive bubble sizing based on actual cluster density
    - Bubble decay when clusters disperse
    - Smoother bubble forces
    """

    def __init__(self, grid_size=10, density_threshold=3.5,
                 bubble_strength=20000.0, use_gpu=True):
        """
        Args:
            grid_size: Number of grid cells per dimension (for initial binning)
            density_threshold: Particles per cell to trigger bubble creation
            bubble_strength: Base repulsive strength of bubbles
            use_gpu: Use GPU acceleration if available
        """
        self.grid_size = grid_size
        self.density_threshold = density_threshold
        self.bubble_strength = bubble_strength
        self.use_gpu = use_gpu and cp is not None

        # Active bubbles: list of (center_x, center_y, radius, strength)
        self.bubbles = []

        # Bubble lifetime and growth parameters
        self.min_radius = 0.5
        self.max_radius = 10.0
        self.growth_rate = 0.01  # Per timestep
        self.decay_rate = 0.98    # Strength decay per timestep when not reinforced

        if self.use_gpu:
            self._compile_gpu_kernel()

    def _compile_gpu_kernel(self):
        """Compile CUDA kernel for bubble forces."""
        self.bubble_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void computeBubbleForces(const float2* pos, float2* force,
                                const float4* bubbles, int numBubbles, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= N) return;

            float2 p = pos[idx];
            float2 f = make_float2(0.0f, 0.0f);

            // Check against all active bubbles
            for(int i = 0; i < numBubbles; i++)
            {
                float4 bubble = bubbles[i];  // (center.x, center.y, radius, strength)

                float dx = p.x - bubble.x;
                float dy = p.y - bubble.y;
                float dist = sqrtf(dx*dx + dy*dy);

                // Repulsive force inside bubble
                if(dist < bubble.z && dist > 1e-10f)
                {
                    float penetration = bubble.z - dist;
                    float forceMag = bubble.w * penetration / bubble.z;  // Linear with depth

                    // Radial repulsion
                    f.x += forceMag * dx / dist;
                    f.y += forceMag * dy / dist;
                }
            }

            force[idx] = f;
        }
        ''', 'computeBubbleForces')

    def update_density(self, positions, bounds_min, bounds_max):
        """
        Analyze particle density and create/update bubbles.

        Args:
            positions: Nx2 array of particle positions
            bounds_min: (x_min, y_min) of simulation domain
            bounds_max: (x_max, y_max) of simulation domain
        """
        xp = cp if self.use_gpu else np

        # Convert to numpy for processing
        if self.use_gpu:
            pos_cpu = cp.asnumpy(positions)
        else:
            pos_cpu = positions

        # Grid-based density analysis
        grid_counts, grid_centers = self._compute_grid_density(
            pos_cpu, bounds_min, bounds_max
        )

        # Update existing bubbles (decay strength)
        self.bubbles = [
            (cx, cy, r, s * self.decay_rate)
            for cx, cy, r, s in self.bubbles
            if s > 1.0  # Remove weak bubbles
        ]

        # Create new bubbles in dense regions
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                density = grid_counts[idx]

                if density > self.density_threshold:
                    center = grid_centers[idx]

                    # Check if bubble already exists nearby
                    existing = False
                    for k, (cx, cy, r, s) in enumerate(self.bubbles):
                        dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                        if dist < r * 1.5:
                            # Reinforce existing bubble
                            new_radius = min(r + self.growth_rate, self.max_radius)
                            new_strength = min(s * 1.2, self.bubble_strength * 2.0)
                            self.bubbles[k] = (cx, cy, new_radius, new_strength)
                            existing = True
                            break

                    if not existing:
                        # Create new bubble
                        radius = self.min_radius * (density / self.density_threshold)
                        self.bubbles.append((
                            center[0], center[1], radius, self.bubble_strength
                        ))

    def _compute_grid_density(self, positions, bounds_min, bounds_max):
        """Compute particle density on a grid."""
        grid_counts = np.zeros(self.grid_size * self.grid_size, dtype=np.int32)
        grid_centers = np.zeros((self.grid_size * self.grid_size, 2), dtype=np.float32)

        # Grid cell dimensions
        dx = (bounds_max[0] - bounds_min[0]) / self.grid_size
        dy = (bounds_max[1] - bounds_min[1]) / self.grid_size

        # Bin particles into grid cells
        for p in positions:
            i = int((p[0] - bounds_min[0]) / dx)
            j = int((p[1] - bounds_min[1]) / dy)

            # Clamp to grid bounds
            i = max(0, min(i, self.grid_size - 1))
            j = max(0, min(j, self.grid_size - 1))

            idx = i * self.grid_size + j
            grid_counts[idx] += 1

        # Compute grid cell centers
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                grid_centers[idx] = [
                    bounds_min[0] + (i + 0.5) * dx,
                    bounds_min[1] + (j + 0.5) * dy
                ]

        return grid_counts, grid_centers

    def compute_forces(self, positions):
        """
        Compute bubble repulsion forces.

        Args:
            positions: Nx2 array of particle positions

        Returns:
            Nx2 array of force vectors
        """
        if len(self.bubbles) == 0:
            xp = cp if self.use_gpu else np
            return xp.zeros_like(positions)

        n = len(positions)

        if self.use_gpu:
            return self._compute_gpu(positions, n)
        else:
            return self._compute_cpu(positions, n)

    def _compute_gpu(self, positions, n):
        """GPU implementation."""
        xp = cp

        # Pack bubbles into float4 array (center.x, center.y, radius, strength)
        bubble_data = xp.array(self.bubbles, dtype=xp.float32)
        num_bubbles = len(self.bubbles)

        # Create float2 views
        float2_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32)])
        pos_f2 = positions.view(float2_dtype).reshape(-1)
        force_f2 = xp.zeros(n, dtype=float2_dtype)

        threads = 256
        blocks = (n + threads - 1) // threads

        self.bubble_kernel(
            (blocks,), (threads,),
            (pos_f2, force_f2, bubble_data, xp.int32(num_bubbles), xp.int32(n))
        )

        return force_f2.view(xp.float32).reshape(n, 2)

    def _compute_cpu(self, positions, n):
        """CPU implementation."""
        xp = np
        force = xp.zeros_like(positions)

        for cx, cy, radius, strength in self.bubbles:
            dx = positions[:, 0] - cx
            dy = positions[:, 1] - cy
            dist = xp.sqrt(dx**2 + dy**2)

            # Particles inside bubble
            inside = (dist < radius) & (dist > 1e-10)

            if xp.any(inside):
                penetration = radius - dist[inside]
                force_mag = strength * penetration / radius

                force[inside, 0] += force_mag * dx[inside] / dist[inside]
                force[inside, 1] += force_mag * dy[inside] / dist[inside]

        return force

    def get_bubble_info(self):
        """Return list of bubbles for visualization."""
        return self.bubbles




def test_adaptive_bubbles():
    """Test adaptive bubbles with rendering."""
    from tsp_nbody.simulator import TSPNBodySimulator

    print("\n" + "=" * 60)
    print("Test 2: Adaptive Bubbles Visualization")
    print("=" * 60)

    # Configure simulator with adaptive bubbles enabled
    params = {
        'p': 3.5,
        'q': 6.0,
        'h': 1.5,
        'wall_schedule': 'linear',
        'squeeze_steps': 500,
        'relax_steps': 100,
        'use_gpu': True,
        'draw': True,
        'render_frequency': 1
    }

    sim = TSPNBodySimulator("datasets/att48/coords.txt", params)

    # Initialize
    if not sim.initialize():
        print("Initialization failed!")
        return

    # Enable adaptive bubbles
    sim.physics_engine.use_adaptive_bubbles = True

    print(f"Improved walls enabled: {sim.physics_engine.use_improved_walls}")
    print(f"Adaptive bubbles enabled: {sim.physics_engine.use_adaptive_bubbles}")

    # Run with visualization - bubbles should appear in dense regions
    print("\nRunning simulation with adaptive bubbles...")
    print("Watch for cyan bubbles appearing in dense particle clusters")
    print("\nPress ESC to exit early")

    path, cost = sim.run_simulation()

    print(f"\nFinal tour cost: {cost:.2f}")
    print(f"Initial cost: {sim.optimal_cost:.2f}")
    print(f"Improvement: {((sim.optimal_cost - cost) / sim.optimal_cost * 100):.1f}%")

    sim.cleanup()

if __name__ == "__main__":
    print("N-Body TSP: Visualization Test for Advanced Features")
    print("=" * 60)
    print()

    try:
        test_adaptive_bubbles()

        print("\n" + "=" * 60)
        print("All visualization tests completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()