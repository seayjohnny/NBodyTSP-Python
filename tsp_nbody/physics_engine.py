"""
GPU Physics Engine for N-Body TSP Simulator

Handles force calculations and physics integration using GPU acceleration.
"""

import numpy as np
from typing import Optional, Tuple
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: CuPy not available. Falling back to CPU (NumPy)")
    GPU_AVAILABLE = False


class NBodyPhysicsEngine:
    """GPU-accelerated N-body physics simulation for TSP."""
    
    def __init__(self, coords: np.ndarray, use_gpu: bool = True):
        """
        Initialize the physics engine.
        
        Args:
            coords: Original city coordinates (n, 2)
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Convert to GPU if needed
        self.coords = self.xp.asarray(coords, dtype=self.xp.float32)
        self.n_cities = len(coords)
        
        # Physics state
        self.pos = None  # Current positions
        self.vel = None  # Velocities
        self.acc = None  # Accelerations
        
        # Physics parameters (from original C++ code)
        self.DAMP = 20.0
        self.MASS = 80.0
        self.WALL_STRENGTH = 20000.0
        self.FORCE_CUTOFF = 100000.0
        self.DT = 0.01
        self.DR = 0.01
        
        # Additional parameters for extra features
        self.slope_repulsion = 50.0
        self.mag_attraction = 0.5
        self.force_cutoff_extra = 0.10
        
        # TSP parameters (p, q, h) - will be set later
        self.p = 2.0
        self.q = 3.0
        self.h = 1.0
        
        # Mode: 'tsp' for TSP forces, 'intercity' for your original implementation
        self.force_mode = 'tsp'
        
        if self.use_gpu:
            self._compile_kernels()
        
        print(f"Physics engine initialized ({'GPU' if self.use_gpu else 'CPU'})")
        print(f"  Cities: {self.n_cities}")
    
    def _compile_kernels(self):
        """Compile the CUDA kernels for GPU acceleration."""
        # Original TSP kernel
        self.nbody_tsp_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void nBodyStepTSP(const float2* shInitPos, float2* shPos, float2* vel, float2* acc,
                      float p, float q, float h, float iR, float oR, 
                      int N, float WALL_STRENGTH, float DAMP, 
                      float FORCE_CUTOFF, float MASS)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if(idx >= N) return;
            
            float2 currentPos = shPos[idx];
            float2 currentVel = vel[idx];
            float2 initPos = shInitPos[idx];

            float2 force = make_float2(0.0f, 0.0f);
            float d, d_temp, l;
            float forceMag, g, radius;

            // N-body interactions
            for(int i = 0; i < N; i++)
            {
                if(i != idx)
                {
                    // Current distance
                    float2 diff = make_float2(shPos[i].x - currentPos.x, 
                                              shPos[i].y - currentPos.y);
                    d_temp = sqrtf(diff.x*diff.x + diff.y*diff.y);
                    d = d_temp > 0.0f ? d_temp : 1.0f;
                    
                    // Initial distance
                    float2 initDiff = make_float2(shInitPos[i].x - initPos.x,
                                                  shInitPos[i].y - initPos.y);
                    l = sqrtf(initDiff.x*initDiff.x + initDiff.y*initDiff.y);

                    // Force calculation
                    g = h * powf(l, q - p);
                    forceMag = g / powf(d, q) - h / powf(d, p);

                    force.x += forceMag * diff.x / d;
                    force.y += forceMag * diff.y / d;
                }
            }

            // Wall forces
            radius = sqrtf(currentPos.x*currentPos.x + currentPos.y*currentPos.y);
            if(radius < iR)
            {
                forceMag = WALL_STRENGTH * (iR - radius);
                force.x += forceMag * currentPos.x / radius;
                force.y += forceMag * currentPos.y / radius;
            }
            else if(radius > oR)
            {
                forceMag = WALL_STRENGTH * (oR - radius);
                force.x += forceMag * currentPos.x / radius;
                force.y += forceMag * currentPos.y / radius;
            }

            // Damping
            force.x -= DAMP * currentVel.x;
            force.y -= DAMP * currentVel.y;
            
            // Force cutoff
            if(force.x > FORCE_CUTOFF) force.x = 0.0f;
            if(force.y > FORCE_CUTOFF) force.y = 0.0f;
            
            // Update acceleration
            acc[idx] = make_float2(force.x / MASS, force.y / MASS);
        }
        ''', 'nBodyStepTSP')
        
        # Intercity forces kernel (for extra features mode)
        self.intercity_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void computeIntercityForces(const float2* initPos, const float2* pos, float2* force,
                                   int N, float slope_repulsion, float mag_attraction, 
                                   float force_cutoff_extra)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= N) return;
            
            float2 total_force = make_float2(0.0f, 0.0f);
            float2 my_pos = pos[idx];
            float2 my_init_pos = initPos[idx];
            
            for(int j = 0; j < N; j++)
            {
                if(j == idx) continue;
                
                // Current distance
                float2 diff = make_float2(pos[j].x - my_pos.x, pos[j].y - my_pos.y);
                float dist = sqrtf(diff.x*diff.x + diff.y*diff.y);
                
                // Original edge length
                float2 edge_diff = make_float2(initPos[j].x - my_init_pos.x, 
                                              initPos[j].y - my_init_pos.y);
                float edge_length = sqrtf(edge_diff.x*edge_diff.x + edge_diff.y*edge_diff.y);
                
                float force_mag = 0.0f;
                
                // Calculate force magnitude based on distance vs edge length
                if(dist <= edge_length)
                {
                    // Repulsion when too close
                    force_mag = -(edge_length - dist) * slope_repulsion;
                }
                else if(edge_length < dist && dist < force_cutoff_extra)
                {
                    // Attraction at medium range
                    force_mag = mag_attraction / edge_length;
                }
                
                // Add force components
                if(dist > 1e-10f)
                {
                    total_force.x += force_mag * diff.x / dist;
                    total_force.y += force_mag * diff.y / dist;
                }
            }
            
            force[idx] = total_force;
        }
        ''', 'computeIntercityForces')
        
        # Wall forces kernel
        self.wall_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void computeWallForces(const float2* pos, float2* force, 
                              float inner_radius, float outer_radius,
                              float wall_strength, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= N) return;
            
            float2 my_pos = pos[idx];
            float radius = sqrtf(my_pos.x*my_pos.x + my_pos.y*my_pos.y);
            float2 wall_force = make_float2(0.0f, 0.0f);
            
            // Inner wall: push outward if inside
            if(radius < inner_radius && radius > 1e-10f)
            {
                float force_mag = wall_strength * (inner_radius - radius);
                wall_force.x = force_mag * my_pos.x / radius;
                wall_force.y = force_mag * my_pos.y / radius;
            }
            // Outer wall: push inward if outside
            else if(radius > outer_radius && radius > 1e-10f)
            {
                float force_mag = wall_strength * (outer_radius - radius);
                wall_force.x = force_mag * my_pos.x / radius;
                wall_force.y = force_mag * my_pos.y / radius;
            }
            
            force[idx] = wall_force;
        }
        ''', 'computeWallForces')
        
        # Bubble forces kernel
        self.bubble_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void computeBubbleForces(const float2* pos, float2* force,
                                const float4* bubbles, int n_bubbles,
                                float wall_strength, int N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= N) return;
            
            float2 my_pos = pos[idx];
            float2 bubble_force = make_float2(0.0f, 0.0f);
            
            // For each active bubble
            for(int b = 0; b < n_bubbles; b++)
            {
                float4 bubble = bubbles[b];
                
                // Check if active (w field)
                if(bubble.w < 0.5f) continue;
                
                float bx = bubble.x;
                float by = bubble.y;
                float br = bubble.z;
                
                // Distance from city to bubble center
                float dx = bx - my_pos.x;
                float dy = by - my_pos.y;
                float dist = sqrtf(dx*dx + dy*dy);
                
                // Push cities out if inside bubble
                if(dist < br && dist > 1e-10f)
                {
                    float force_mag = -wall_strength * (br - dist);
                    bubble_force.x += force_mag * dx / dist;
                    bubble_force.y += force_mag * dy / dist;
                }
            }
            
            force[idx] = bubble_force;
        }
        ''', 'computeBubbleForces')
        
    def set_parameters(self, p: float, q: float, h: float):
        """
        Set the TSP force parameters.
        
        Args:
            p: Repulsion exponent
            q: Attraction exponent  
            h: Force scaling constant
        """
        self.p = p
        self.q = q
        self.h = h
        
    def set_force_mode(self, mode: str):
        """
        Set the force calculation mode.
        
        Args:
            mode: Either 'tsp' for TSP forces or 'intercity' for original implementation
        """
        if mode not in ['tsp', 'intercity']:
            raise ValueError("Mode must be 'tsp' or 'intercity'")
        self.force_mode = mode
        print(f"Force mode set to: {mode}")
        
    def initialize_physics(self):
        """Set initial conditions for the simulation."""
        # Positions start at original coordinates
        self.pos = self.coords.copy()
        
        # Velocities start at zero
        self.vel = self.xp.zeros_like(self.pos)
        
        # Accelerations start at zero
        self.acc = self.xp.zeros_like(self.pos)
        
        # Move any cities at exact origin slightly off center
        radii = self.xp.sqrt(self.pos[:, 0]**2 + self.pos[:, 1]**2)
        too_close = radii < 0.001
        if self.xp.any(too_close):
            self.pos[too_close, 0] = 0.001
            self.pos[too_close, 1] = 0.001
        
        print("Physics initialized: positions, velocities set")
    
    def compute_intercity_forces_gpu(self) -> cp.ndarray:
        """Calculate intercity forces using GPU kernel."""
        force = cp.zeros_like(self.pos)
        
        float2_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32)])
        
        pos_f2 = self.pos.view(float2_dtype).reshape(-1)
        init_pos_f2 = self.coords.view(float2_dtype).reshape(-1)
        force_f2 = force.view(float2_dtype).reshape(-1)
        
        threads_per_block = 256
        blocks = (self.n_cities + threads_per_block - 1) // threads_per_block
        
        self.intercity_kernel(
            (blocks,), (threads_per_block,),
            (init_pos_f2, pos_f2, force_f2,
            cp.int32(self.n_cities), cp.float32(self.slope_repulsion),
            cp.float32(self.mag_attraction), cp.float32(self.force_cutoff_extra))
        )
        
        return force
    
    def compute_intercity_forces_cpu(self) -> np.ndarray:
        """Calculate intercity forces using CPU."""
        xp = self.xp
        n = self.n_cities
        force = xp.zeros_like(self.pos)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Current distance
                dx = self.pos[j, 0] - self.pos[i, 0]
                dy = self.pos[j, 1] - self.pos[i, 1]
                dist = xp.sqrt(dx*dx + dy*dy)
                
                # Original edge length
                edx = self.coords[j, 0] - self.coords[i, 0]
                edy = self.coords[j, 1] - self.coords[i, 1]
                edge_length = xp.sqrt(edx*edx + edy*edy)
                
                # Calculate force magnitude
                if dist <= edge_length:
                    force_mag = -(edge_length - dist) * self.slope_repulsion
                elif edge_length < dist < self.force_cutoff_extra:
                    force_mag = self.mag_attraction / edge_length
                else:
                    force_mag = 0.0
                
                # Add force components
                if dist > 1e-10:
                    force[i, 0] += force_mag * dx / dist
                    force[i, 1] += force_mag * dy / dist
        
        return force
    
    def compute_wall_forces_gpu(self, inner_radius: float, outer_radius: float) -> cp.ndarray:
        """Calculate wall forces using GPU kernel."""
        force = cp.zeros_like(self.pos)
        
        float2_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32)])
        
        pos_f2 = self.pos.view(float2_dtype).reshape(-1)
        force_f2 = force.view(float2_dtype).reshape(-1)
        
        threads_per_block = 256
        blocks = (self.n_cities + threads_per_block - 1) // threads_per_block
        
        self.wall_kernel(
            (blocks,), (threads_per_block,),
            (pos_f2, force_f2, cp.float32(inner_radius), cp.float32(outer_radius),
            cp.float32(self.WALL_STRENGTH), cp.int32(self.n_cities))
        )
        
        return force
    
    def compute_wall_forces_cpu(self, inner_radius: float, outer_radius: float) -> np.ndarray:
        """Calculate wall forces using CPU."""
        xp = self.xp
        
        dx = self.pos[:, 0]
        dy = self.pos[:, 1]
        radius = xp.sqrt(dx*dx + dy*dy)
        
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
        
        float2_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32)])
        float4_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32), 
                                ('z', cp.float32), ('w', cp.float32)])
        
        pos_f2 = self.pos.view(float2_dtype).reshape(-1)
        force_f2 = force.view(float2_dtype).reshape(-1)
        bubbles_f4 = bubbles_xp.view(float4_dtype).reshape(-1)
        
        threads_per_block = 256
        blocks = (self.n_cities + threads_per_block - 1) // threads_per_block
        
        self.bubble_kernel(
            (blocks,), (threads_per_block,),
            (pos_f2, force_f2, bubbles_f4, cp.int32(len(bubbles)),
            cp.float32(self.WALL_STRENGTH), cp.int32(self.n_cities))
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
            dist = xp.sqrt(dx*dx + dy*dy)
            
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
        float2_dtype = cp.dtype([('x', cp.float32), ('y', cp.float32)])
        
        # Create views as float2 structured arrays
        pos_f2 = self.pos.view(float2_dtype).reshape(-1)
        vel_f2 = self.vel.view(float2_dtype).reshape(-1)
        acc_f2 = self.acc.view(float2_dtype).reshape(-1)
        init_pos_f2 = self.coords.view(float2_dtype).reshape(-1)
        
        threads_per_block = 256
        blocks = (self.n_cities + threads_per_block - 1) // threads_per_block
        
        self.nbody_tsp_kernel(
            (blocks,), (threads_per_block,),
            (init_pos_f2, pos_f2, vel_f2, acc_f2,
            cp.float32(self.p), cp.float32(self.q), cp.float32(self.h),
            cp.float32(inner_radius), cp.float32(outer_radius),
            cp.int32(self.n_cities), cp.float32(self.WALL_STRENGTH),
            cp.float32(self.DAMP), cp.float32(self.FORCE_CUTOFF), 
            cp.float32(self.MASS))
        )
        
        # Update velocities and positions
        self.vel += self.acc * self.DT
        self.pos += self.vel * self.DT
    
    def integrate_step_intercity_gpu(self, inner_radius: float, outer_radius: float,
                                     bubbles: Optional[np.ndarray] = None):
        """Perform intercity integration using GPU kernels."""
        # Compute all forces
        force = cp.zeros_like(self.pos)
        
        force += self.compute_intercity_forces_gpu()
        force += self.compute_wall_forces_gpu(inner_radius, outer_radius)
        if bubbles is not None:
            force += self.compute_bubble_forces_gpu(bubbles)
        force += self.compute_damping_forces()
        
        # Compute acceleration
        self.acc = force / self.MASS
        
        # Update velocities and positions
        self.vel += self.acc * self.DT
        self.pos += self.vel * self.DT
    
    def integrate_step_tsp_cpu(self, inner_radius: float, outer_radius: float):
        """Perform TSP integration using CPU."""
        xp = self.xp
        n = self.n_cities
        
        force = xp.zeros_like(self.pos)
        
        # N-body forces
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Current distance
                diff = self.pos[j] - self.pos[i]
                d = xp.linalg.norm(diff)
                d = d if d > 0 else 1.0
                
                # Initial distance
                init_diff = self.coords[j] - self.coords[i]
                l = xp.linalg.norm(init_diff)
                
                # Force magnitude
                g = self.h * (l ** (self.q - self.p))
                force_mag = g / (d ** self.q) - self.h / (d ** self.p)
                
                force[i] += force_mag * diff / d
        
        # Wall forces
        force += self.compute_wall_forces_cpu(inner_radius, outer_radius)
        
        # Damping
        force += self.compute_damping_forces()
        
        # Force cutoff
        force[force > self.FORCE_CUTOFF] = 0.0
        
        # Compute acceleration
        self.acc = force / self.MASS
        
        # Velocity Verlet integration
        self.vel += self.acc * self.DT
        self.pos += self.vel * self.DT
    
    def integrate_step_intercity_cpu(self, inner_radius: float, outer_radius: float,
                                     bubbles: Optional[np.ndarray] = None):
        """Perform intercity integration using CPU."""
        force = self.xp.zeros_like(self.pos)
        
        force += self.compute_intercity_forces_cpu()
        force += self.compute_wall_forces_cpu(inner_radius, outer_radius)
        if bubbles is not None:
            force += self.compute_bubble_forces_cpu(bubbles)
        force += self.compute_damping_forces()
        
        # Compute acceleration
        self.acc = force / self.MASS
        
        # Update velocities and positions
        self.vel += self.acc * self.DT
        self.pos += self.vel * self.DT
    
    def integrate_step(self, inner_radius: float, outer_radius: float,
                      bubbles: Optional[np.ndarray] = None):
        """
        Perform one physics integration step.
        
        Args:
            inner_radius: Inner wall radius
            outer_radius: Outer wall radius
            bubbles: Optional bubble array (only for intercity mode)
        """
        if self.force_mode == 'tsp':
            if self.use_gpu:
                self.integrate_step_tsp_gpu(inner_radius, outer_radius)
            else:
                self.integrate_step_tsp_cpu(inner_radius, outer_radius)
        else:  # intercity mode
            if self.use_gpu:
                self.integrate_step_intercity_gpu(inner_radius, outer_radius, bubbles)
            else:
                self.integrate_step_intercity_cpu(inner_radius, outer_radius, bubbles)
    
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
        v_squared = xp.sum(self.vel ** 2, axis=1)
        ke = 0.5 * self.MASS * xp.sum(v_squared)
        
        if self.use_gpu:
            return float(cp.asnumpy(ke))
        return float(ke)
    
    def compute_pressure_on_outer_wall(self, outer_radius: float) -> float:
        """Calculate pressure exerted on the outer wall."""
        xp = self.xp
        
        radii = xp.sqrt(self.pos[:, 0]**2 + self.pos[:, 1]**2)
        
        beyond = radii - outer_radius
        beyond = xp.maximum(beyond, 0.0)
        total_pressure = xp.sum(beyond)
        
        circumference = 2.0 * np.pi * outer_radius
        pressure = float(total_pressure) * self.WALL_STRENGTH / circumference
        
        if self.use_gpu:
            return float(cp.asnumpy(pressure))
        return float(pressure)
    
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
            cost += np.sqrt(np.sum(diff ** 2))
        
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
    engine.set_force_mode('tsp')
    
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
    
    # Test intercity mode with bubbles
    print("\n" + "=" * 60)
    print("Testing Intercity Mode with Bubbles")
    print("=" * 60)
    
    engine2 = NBodyPhysicsEngine(coords, use_gpu=True)
    engine2.set_force_mode('intercity')
    engine2.initialize_physics()
    
    # Create some test bubbles (x, y, radius, active)
    bubbles = np.array([
        [5.0, 5.0, 3.0, 1.0],
        [-5.0, -5.0, 2.0, 1.0],
        [0.0, 8.0, 2.5, 0.0],  # inactive
    ], dtype=np.float32)
    
    print("\nRunning intercity simulation with bubbles...")
    for step in range(100):
        engine2.integrate_step(0.0, max_dist, bubbles)
        
        if step % 20 == 0:
            ke = engine2.compute_kinetic_energy()
            pressure = engine2.compute_pressure_on_outer_wall(max_dist)
            print(f"Step {step}: KE={ke:.4f}, Pressure={pressure:.4f}")
    
    print("\nIntercity simulation complete!")