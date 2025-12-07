"""
N-Body TSP Simulator - Main Controller

Orchestrates the complete N-body physics simulation for solving TSP.
"""
import sys
import time
import logging
import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Dict


# Import our modules
from tsp_nbody.dataio import TSPDataLoader, load_optimal_path, load_optimal_cost
from tsp_nbody.physics_engine import NBodyPhysicsEngine, NBodyPhysicsOptions, default_nbody_options
from tsp_nbody.path_extraction import PathExtractor, nearest_neighbor_tsp
from tsp_nbody.renderer import TSPRenderer, RendererOptions, default_renderer_options, OPENGL_AVAILABLE


class TSPNBodySimulator:
    """Complete N-body TSP simulator with visualization."""

    def __init__(
        self,
        coord_file: str,
        params: Optional[Dict] = None,
        nbody_options: Optional[NBodyPhysicsOptions] = None,
        renderer_options: Optional[RendererOptions] = None,
    ):
        """
        Initialize the simulator.

        Args:
            coord_file: Path to coordinate file
            params: Optional parameter dictionary
        """
        self.coord_file = Path(coord_file)
        self.params = self.default_params()
        if params is not None:
            self.params.update(params)

        self.nbody_options = default_nbody_options.copy()
        if nbody_options is not None:
            self.nbody_options.update(nbody_options)

        self.renderer_options = default_renderer_options.copy()
        if renderer_options is not None:
            self.renderer_options.update(renderer_options)

        # Components
        self.data_loader = TSPDataLoader(str(coord_file))
        self.physics_engine = None
        self.path_extractor = PathExtractor()
        self.renderer = None

        # Simulation state
        self.is_initialized = False
        self.coords = None
        self.n_cities = 0
        self.normalizing_factor = 1.0

        # Wall state
        self.inner_radius = 0.0
        self.outer_radius = 0.0
        self.inner_direction = 1
        self.outer_direction = 0

        # Results
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_time = 0.0
        self.final_path = None
        self.final_cost = 0.0
        self.optimal_cost = None

        print(f"Simulator created for: {self.coord_file.name}")


    def default_params(self) -> Dict:
        """Get default simulation parameters."""
        return {
            # Integration
            "steps_per_wall_move": 1,  # Match C++ inner while loop (1.0 / 0.01 = 100)

            # Rendering
            "draw": True,
            "render_frequency": 10,  # Render every N physics steps
            "pause_initial": True,

            # GPU
            "use_gpu": True,
        }

    def initialize(self) -> bool:
        """
        Initialize all simulator components.

        Returns:
            True if successful
        """
        try:
            print("\n" + "=" * 60)
            print("INITIALIZING N-BODY TSP SIMULATOR")
            print("=" * 60)

            # Load and preprocess data
            print("\n[1/4] Loading and preprocessing data...")
            stats = self.data_loader.preprocess(normalize_method="minimum")
            self.coords = self.data_loader.coords
            self.n_cities = self.data_loader.n_cities
            self.normalizing_factor = self.data_loader.normalizing_factor

            # Initialize physics engine
            print("\n[2/4] Initializing physics engine...")
            self.physics_engine = NBodyPhysicsEngine(
                self.coords,
                options=self.nbody_options
            )

            # Initialize physics state
            self.physics_engine.initialize_physics()

            # Initialize wall radii
            self.inner_radius = 0.0
            self.outer_radius = self.data_loader.get_bounding_circle_radius()
            self.inner_direction = 1
            self.outer_direction = 0

            print(f"Initial outer radius: {self.outer_radius:.4f}")
            print(f"DR (wall step): {self.nbody_options['DR']:.4f}")
            print(f"DT (time step): {self.nbody_options['DT']:.4f}")

            # Initialize renderer if drawing enabled
            print("\n[3/4] Initializing renderer...")
            if self.params["draw"] and OPENGL_AVAILABLE:
                self.renderer = TSPRenderer(options=self.renderer_options)
                if not self.renderer.initialize():
                    print(
                        "Warning: Renderer initialization failed, continuing without visualization"
                    )
                    self.renderer = None
            else:
                print("Rendering disabled or OpenGL not available")

            # Load optimal solution if available
            print("\n[4/4] Loading optimal solution (if available)...")
            dataset_dir = self.coord_file.parent
            opt_cost_file = dataset_dir / "tour_len.txt"
            self.optimal_cost = load_optimal_cost(str(opt_cost_file))
            if self.optimal_cost:
                # Adjust for normalization
                self.optimal_cost /= self.normalizing_factor
                print(f"Optimal cost (normalized): {self.optimal_cost:.4f}")

            self.is_initialized = True
            print("\n" + "=" * 60)
            print("INITIALIZATION COMPLETE")
            print("=" * 60 + "\n")
            return True

        except Exception as e:
            print(f"Initialization failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def compute_pressure(self) -> float:
        """Calculate pressure on outer wall."""
        return self.physics_engine.compute_pressure_on_outer_wall(self.outer_radius)

    def update_wall_directions(self, pressure: float):
        """
        Update wall movement directions based on pressure.

        Args:
            pressure: Current pressure on outer wall
        """
        if pressure < self.nbody_options["lower_pressure_limit"]:
            # Low pressure: shrink outer wall
            self.inner_direction = 0
            self.outer_direction = -1
        elif pressure < self.nbody_options["upper_pressure_limit"]:
            # Medium pressure: expand inner wall
            self.inner_direction = 1
            self.outer_direction = 0
        else:
            # High pressure: expand outer wall
            self.inner_direction = 0
            self.outer_direction = 1

    def step_walls(self, dr: float):
        """
        Move walls by one step.

        Args:
            dr: Step size for wall movement
        """
        if self.nbody_options["use_improved_walls"]:
            # Use pressure-based control
            self.inner_radius += dr * self.inner_direction
            self.outer_radius += dr * self.outer_direction
        else:
            # Simple expansion of inner wall (TSP mode)
            self.inner_radius += dr

        # Ensure radii stay valid
        self.inner_radius = max(0.0, self.inner_radius)
        self.outer_radius = max(self.inner_radius + 0.01, self.outer_radius)

    def render_frame(self, step: int = 0):
        """
        Render current simulation state.

        Args:
            step: Current step number (for display)
        """
        if self.renderer is None or not self.renderer.is_initialized:
            return

        # Get current positions from GPU
        positions = self.physics_engine.get_positions_cpu()

        # Get bubble data if available
        bubbles = self.physics_engine.get_bubble_data_for_renderer()

        # Get current pressure
        pressure = self.physics_engine.current_pressure

        # Normalize positions for rendering (to outer radius)
        if self.outer_radius > 0:
            positions_normalized = positions / self.outer_radius
            inner_r_normalized = self.inner_radius / self.outer_radius
            outer_r_normalized = 1.0

            # Normalize bubble positions too
            if bubbles is not None:
                bubbles_normalized = bubbles.copy()
                bubbles_normalized[:, 0:2] /= self.outer_radius  # Normalize x, y
                bubbles_normalized[:, 2] /= self.outer_radius  # Normalize radius
            else:
                bubbles_normalized = None
        else:
            positions_normalized = positions
            inner_r_normalized = self.inner_radius
            outer_r_normalized = self.outer_radius
            bubbles_normalized = bubbles

        # Draw frame
        self.renderer.draw_frame_complete(
            positions_normalized,
            inner_r_normalized,
            outer_r_normalized,
            self.inner_direction,
            self.outer_direction,
            bubbles=bubbles_normalized,
            pressure=pressure,
        )

    def run_simulation(self) -> Tuple[np.ndarray, float]:
        """
        Run the complete N-body TSP simulation.

        Returns:
            Tuple of (path, cost)
        """
        if not self.is_initialized:
            raise RuntimeError("Simulator not initialized. Call initialize() first.")

        print("\n" + "=" * 60)
        print("RUNNING N-BODY SIMULATION")
        print("=" * 60)

        self.start_time = time.perf_counter()

        # Get DR (wall step size)
        dr = self.nbody_options["DR"]

        # Show initial configuration
        if self.params["draw"] and self.renderer:
            print("\nShowing initial configuration...")
            self.render_frame()
            if self.params["pause_initial"]:
                self.renderer.wait_for_key()

        # Main simulation loop (matches C++ structure)
        print("\nRunning N-body extrusion...")

        draw_count = 0
        total_steps = 0
        last_print_radius = 0.0

        # Main loop: while inner < outer - DR
        while self.inner_radius < self.outer_radius - dr:
            for _ in range(self.params["steps_per_wall_move"]):
                # Integrate one step
                self.physics_engine.integrate_step(
                    self.inner_radius,
                    self.outer_radius,
                    bubbles=None,  # No bubbles in MVP
                )

                # Render periodically
                if (
                    self.params["draw"]
                    and draw_count % self.params["render_frequency"] == 0
                ):
                    self.render_frame(total_steps)

                    # Handle window events
                    if self.renderer and not self.renderer.handle_events():
                        print("\nSimulation interrupted by user")
                        break

                draw_count += 1
                total_steps += 1

            # Move walls after integration phase
            if self.nbody_options["use_improved_walls"]:
                pressure = self.compute_pressure()
                self.update_wall_directions(pressure)
            else:
                # Simple mode: always expand inner wall
                self.inner_direction = 1
                self.outer_direction = 0

            self.step_walls(dr)

            # Update bubble density periodically (if adaptive bubbles enabled)
            if self.physics_engine.use_adaptive_bubbles:
                self.physics_engine.update_bubble_density(self.outer_radius)

            # Progress indicator (print every 0.1 units of inner radius)
            if self.inner_radius - last_print_radius >= 0.1:
                pressure = (
                    self.compute_pressure()
                    if self.nbody_options["use_improved_walls"]
                    else 0.0
                )
                ke = self.physics_engine.compute_kinetic_energy()
                # print(f"  Inner R: {self.inner_radius:.2f} | "
                #       f"Outer R: {self.outer_radius:.2f} | "
                #       f"KE: {ke:.4f} | "
                #       f"Pressure: {pressure:.1f}")
                last_print_radius = self.inner_radius

        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

        print(f"\nSimulation complete in {self.elapsed_time:.2f}s")
        print(f"Total physics steps: {total_steps}")

        # Extract path from final configuration
        print("\nExtracting TSP path from ring configuration...")
        final_positions = self.physics_engine.get_positions_cpu()

        # Use the physics engine's built-in tour extraction
        self.final_path = self.physics_engine.get_final_tour()

        # Calculate path cost (use original unnormalized coordinates)
        original_coords = self.data_loader.original_coords
        self.final_cost = self.path_extractor.calculate_path_cost(
            original_coords, self.final_path
        )

        # Display final path
        if self.params["draw"] and self.renderer:
            print("\nShowing final path...")
            positions_normalized = final_positions / self.outer_radius
            coords_normalized = self.coords / self.outer_radius

            self.renderer.clear()
            self.renderer.draw_path(
                coords_normalized, self.final_path,
                width=self.renderer.path_width,
                color=self.renderer.color_path
            )
            self.renderer.draw_cities(coords_normalized, size=self.renderer.city_size, color=self.renderer.color_city)
            self.renderer.swap_buffers()

            if self.params["pause_initial"]:
                self.print_results()
                self.renderer.wait_for_key()
        else:
            self.print_results()

        return self.final_path, self.final_cost

    def print_results(self):
        """Print simulation results."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)

        print(f"\nDataset: {self.coord_file.name}")
        print(f"Cities: {self.n_cities}")
        print(f"N-body path cost: {self.final_cost:.4f}")

        if self.optimal_cost is not None:
            comparison = self.path_extractor.compare_with_optimal(
                self.final_cost, self.optimal_cost * self.normalizing_factor
            )
            print(f"Optimal cost: {self.optimal_cost * self.normalizing_factor:.4f}")
            print(f"Percent difference: {comparison['percent_difference']:.2f}%")

            if comparison["is_better"]:
                print("! N-body solution is BETTER than recorded optimal!")
            elif comparison["percent_difference"] < 10.0:
                print("+ Solution is within 10% of optimal")
            else:
                print("- Solution is >10% from optimal")

        # Calculate nearest neighbor for comparison
        print("\nComparing with Nearest Neighbor heuristic...")
        nn_path = nearest_neighbor_tsp(self.data_loader.original_coords)
        nn_cost = self.path_extractor.calculate_path_cost(
            self.data_loader.original_coords, nn_path
        )

        print(f"Nearest Neighbor cost: {nn_cost:.4f}")
        improvement = 100.0 * (nn_cost - self.final_cost) / nn_cost
        print(
            f"N-body vs NN: {improvement:+.2f}% "
            f"({'better' if improvement > 0 else 'worse'})"
        )

        print("\n" + "=" * 60 + "\n")

    def cleanup(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()


def main():
    """Main entry point for running the simulator."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        coord_file = sys.argv[1]
    else:
        # Default dataset
        coord_file = "datasets/grid4x4/coords.txt"

    print("N-Body TSP Simulator")
    print(f"Using dataset: {coord_file}\n")

    # Create simulator with custom parameters (optional)
    nbody_options = {
        "p": 9.073103,
        "q": 13.695556,
        "m": -1.5,
        "slope_repulsion": 50.0,
        "mag_attraction": 10.0,
        "force_cutoff_extra": 100,
        "force_mode": "piecewise",
        "DT": 0.01,
        "use_gpu": True,
        "use_improved_walls": True,
    }

    renderer_options = {
        "city_size": 16.0,
        "path_width": 4.0,
        "wall_width": 2.0,
    }

    params = {
        "draw": True,
        "use_gpu": True,
        "render_frequency": 1,
    }

    simulator = TSPNBodySimulator(
        coord_file, params=params,
        nbody_options=nbody_options, renderer_options=renderer_options
    )

    # Initialize
    if not simulator.initialize():
        print("Initialization failed. Exiting.")
        return 1

    # Run simulation
    try:
        path, cost = simulator.run_simulation()

        # Print results
        # simulator.print_results()

        # Cleanup
        simulator.cleanup()

        return 0

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        simulator.cleanup()
        return 1
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        import traceback

        traceback.print_exc()
        simulator.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())
