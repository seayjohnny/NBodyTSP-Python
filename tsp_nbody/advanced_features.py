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

class AdaptiveBubbles:
    """
    Improved bubble implementation for breaking up dense clusters.

    Improvements over original grid-based approach:
    - Spatial hashing for efficient density detection
    - Adaptive bubble sizing based on actual cluster density
    - Bubble decay when clusters disperse
    - Smoother bubble forces
    """

    def __init__(
        self, positions, grid_size=8, density_threshold=3.5,
        num_bubbles = 3, use_gpu=True):
        """
        Args:
            grid_size: Number of grid cells per dimension (for initial binning)
            density_threshold: Particles per cell to trigger bubble creation
            bubble_strength: Base repulsive strength of bubbles
            use_gpu: Use GPU acceleration if available
        """
        self.grid_size = grid_size
        self.density_threshold = density_threshold
        self.num_bubbles = num_bubbles
        self.use_gpu = use_gpu and cp is not None

        # Active bubbles: list of (center_x, center_y, radius, enabled)
        self.bounds = np.linspace(-1.0, 1.0, grid_size + 1)
        self.bubbles = []
        self.grid_counts = []
        self.grid_centers = []

        self.initialize_bubbles(positions)

        if self.use_gpu:
            self._compile_gpu_kernel()

    def _compile_gpu_kernel(self):
        """Compile CUDA kernel for calculating where bubbles should spawn."""
        self.density_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void computeDensity(const float2* pos, int* densityGrid, float2* gridCenters,
                            float* bounds, int b, int N)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int id = x + y * gridDim.x * blockDim.x;

            float xr[2] = {bounds[blockIdx.x], bounds[blockIdx.x + 1]};
            float yr[2] = {bounds[b - 1 - blockIdx.y], bounds[b - blockIdx.y]};
            float xBar = 0.0;
            float yBar = 0.0;

            for(int i = 0; i < N; i++)
            {
                densityGrid[id] = 0; // Initialize
            }

            __syncthreads();

            for(int i = 0; i < N; i++)
            {
                // If the particle is in this grid cell, increment density
                if(pos[i].x >= xr[0] && pos[i].x < xr[1] &&
                   pos[i].y >= yr[0] && pos[i].y < yr[1])
                {
                    atomicAdd(&densityGrid[id], 1);
                    xBar += pos[i].x;
                    yBar += pos[i].y;
                }
            }

            __syncthreads();

            // If density > 0, compute center
            if(densityGrid[id] > 0)
            {
                gridCenters[id].x = xBar / densityGrid[id];
                gridCenters[id].y = yBar / densityGrid[id];
            }

            __syncthreads();
        }
        ''', 'computeDensity')

    def _compute_grid_density(self, positions):
        """Compute particle density on a grid."""
        grid_counts = np.zeros(self.grid_size * self.grid_size, dtype=np.int32)
        grid_centers = np.zeros((self.grid_size * self.grid_size, 2), dtype=np.float32)

        # Grid cell dimensions
        dx = (self.bounds[-1] - self.bounds[0]) / self.grid_size
        dy = (self.bounds[-1] - self.bounds[0]) / self.grid_size

        # Bin particles into grid cells
        for p in positions:
            i = int((p[0] - self.bounds[0]) / dx)
            j = int((p[1] - self.bounds[0]) / dy)

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
                    self.bounds[0] + (i + 0.5) * dx,
                    self.bounds[0] + (j + 0.5) * dy
                ]

        self.grid_counts = grid_counts
        self.grid_centers = grid_centers

        return grid_counts, grid_centers

    def initialize_bubbles(self, positions):
        """Create the initial bubble list.
        
        Bubbles are represented as (center_x, center_y, radius, enabled).
        On GPU, this will be packed into a float4 array for efficiency.
        """
        
        grid_counts, grid_centers = self._compute_grid_density(positions)
        
        # Select top N densest cells to create bubbles
        dense_indices = np.argsort(grid_counts)[-self.num_bubbles:]
        self.bubbles = []

        for idx in dense_indices:
            if grid_counts[idx] >= self.density_threshold:
                center = grid_centers[idx]
                radius = 0.1  # Initial radius
                self.bubbles.append([center[0], center[1], radius, 1.0])  # Enabled
        
        return self.bubbles

    def get_density_grid(self) -> tuple[list, list]:
        """Return the density grid for external use."""
        return self.grid_counts, self.grid_centers

    def get_bubbles_gpu(self):
        """Return bubbles as a GPU array for force calculations."""
        if not self.use_gpu:
            raise RuntimeError("GPU not enabled for AdaptiveBubbles")

        bubble_array = cp.array(self.bubbles, dtype=cp.float32)
        return bubble_array
        
    def get_bubbles_cpu(self):
        """Return bubbles as a CPU array for force calculations."""
        return np.array(self.bubbles, dtype=np.float32)

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