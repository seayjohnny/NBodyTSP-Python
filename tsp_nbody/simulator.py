"""
N-Body TSP Simulator - Main Controller

Orchestrates the complete N-body physics simulation for solving TSP.
"""
import sys
import time
import logging
import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Dict, TypedDict
from datetime import datetime

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pygame
except ImportError:
    pygame = None

# Import our modules
from tsp_nbody.dataio import TSPDataLoader, load_optimal_path, load_optimal_cost
from tsp_nbody.physics_engine import NBodyPhysicsEngine, NBodyPhysicsOptions, default_nbody_options
from tsp_nbody.path_extraction import PathExtractor, random_nearest_neighbor_tsp
from tsp_nbody.renderer import TSPRenderer, RendererOptions, default_renderer_options, OPENGL_AVAILABLE
from tsp_nbody.best_results import att48, ch150


class SimulatorOptions(TypedDict, total=False):
    """Typed dictionary for simulator options."""
    # Common options
    steps_per_wall_move: int
    use_gpu: bool
    use_pressure: bool
    use_density_grid: bool
    use_bubbles: bool  # Requires density grid to be enabled

    # Rendering options
    draw: bool
    render_frequency: int
    pause_initial: bool
    step_mode: str  # "continuous" or "step"

    # Video recording options
    record_video: bool
    video_output_path: Optional[str]
    video_fps: int
    video_record_frequency: int  # Record every Nth frame

    # Debug options
    debug_window: bool
    debug_update_frequency: int


default_simulator_options: SimulatorOptions = {
    "steps_per_wall_move": 1,
    "use_gpu": True,
    "use_pressure": False,
    "use_density_grid": False,
    "use_bubbles": False,
    "draw": True,
    "render_frequency": 10,
    "pause_initial": True,
    "step_mode": "continuous",
    "record_video": False,
    "video_output_path": None,
    "video_fps": 30,
    "video_record_frequency": 1,  # Record every frame
    "debug_window": False,
    "debug_update_frequency": 10,
}


class TSPNBodySimulator:
    """Complete N-body TSP simulator with visualization."""

    def __init__(
        self,
        coord_file: str,
        options: Optional[SimulatorOptions] = None,
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

        # N-body and rendering parameters
        self.nbody_options: NBodyPhysicsEngine = default_nbody_options.copy()
        self.nbody_options.update(nbody_options or {})

        self.renderer_options: RendererOptions = default_renderer_options.copy()
        self.renderer_options.update(renderer_options or {})
        

        self.options = default_simulator_options.copy()
        if options is not None:
            self.options.update(options)

            self.nbody_options["use_gpu"] = self.options.get("use_gpu", True)
            self.nbody_options["use_pressure"] = self.options.get("use_pressure", False)
            self.nbody_options["use_density_grid"] = self.options.get("use_density_grid", False)
            self.nbody_options["use_bubbles"] = self.options.get("use_bubbles", False)

            self.renderer_options["show_grid"] = self.options.get("use_density_grid", False)
            self.renderer_options["show_density"] = self.options.get("use_density_grid", False)
            self.renderer_options["show_bubbles"] = self.options.get("use_bubbles", False)

        # Components
        self.data_loader = TSPDataLoader(str(coord_file))
        self.physics_engine = None
        self.path_extractor = PathExtractor()
        self.renderer = None
        self.debug_window = None
        self.video_writer = None

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

        # Bubble mode selection
        self.use_dynamic_bubbles = False

        print(f"Simulator created for: {self.coord_file.name}")


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
            if self.options["draw"] and OPENGL_AVAILABLE:
                self.renderer = TSPRenderer(options=self.renderer_options)
                if not self.renderer.initialize():
                    print(
                        "Warning: Renderer initialization failed, continuing without visualization"
                    )
                    self.renderer = None
                else:
                    # Initialize video recording if enabled
                    if self.options.get("record_video", False):
                        self._initialize_video_writer()
            else:
                print("Rendering disabled or OpenGL not available")

            # Initialize debug window if enabled
            if self.options.get("debug_window", False):
                try:
                    from tsp_nbody.debug_window import DebugWindow
                    self.debug_window = DebugWindow(
                        title=f"Debug: {self.coord_file.name}",
                        n_particles=self.n_cities
                    )
                    print("Debug window created")
                except Exception as e:
                    print(f"Warning: Could not create debug window: {e}")
                    self.debug_window = None

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

    def _initialize_video_writer(self):
        """Initialize video writer for recording simulation."""
        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available. Video recording disabled.")
            self.options["record_video"] = False
            return

        if not self.renderer or not self.renderer.is_initialized:
            print("Warning: Renderer not available. Video recording disabled.")
            self.options["record_video"] = False
            return

        # Generate output path if not provided
        if self.options.get("video_output_path") is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = self.coord_file.stem
            output_dir = Path("videos")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{dataset_name}_{timestamp}.mp4"
            self.options["video_output_path"] = str(output_path)

        # Get video parameters
        width, height = self.renderer.window_size
        fps = self.options.get("video_fps", 30)
        output_path = self.options["video_output_path"]

        # Initialize VideoWriter with H.264 codec (best for YouTube)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for better compatibility
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not self.video_writer.isOpened():
            print(f"Warning: Failed to open video writer for {output_path}")
            self.video_writer = None
            self.options["record_video"] = False
        else:
            print(f"Video recording initialized: {output_path}")
            print(f"  Resolution: {width}x{height}, FPS: {fps}")

    def get_smallest_distance(self) -> float:
        """Calculate smallest inter-city distance."""
        min_dist = float('inf')
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                if dist < min_dist:
                    min_dist = dist
        return min_dist

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
        if self.nbody_options["use_pressure"]:
            # Use pressure-based control
            self.inner_radius += dr * self.inner_direction
            self.outer_radius += dr * self.outer_direction
        else:
            # Simple expansion of inner wall (TSP mode)
            self.inner_radius += dr

        # Ensure radii stay valid
        # self.inner_radius = max(0.0, self.inner_radius)
        # self.outer_radius = max(self.inner_radius + 0.001, self.outer_radius)

    def _draw_frame_no_swap(self, positions: np.ndarray, inner_radius: float,
                           outer_radius: float, inner_dir: int, outer_dir: int,
                           bubbles: Optional[np.ndarray] = None,
                           density_grid: Optional[np.ndarray] = None,
                           grid_bins: int = 8,
                           pressure: Optional[float] = None):
        """
        Draw a frame without swapping buffers (for video capture).

        This is the same as renderer.draw_frame_complete but without the swap_buffers() call.
        """
        if not self.renderer or not self.renderer.is_initialized:
            return

        self.renderer.clear()

        # Draw density grid
        self.renderer.draw_density_grid(density_grid, grid_bins)

        # Draw walls
        self.renderer.draw_walls(inner_radius, outer_radius, inner_dir, outer_dir, width=self.renderer.wall_width)
        if bubbles is not None:
            self.renderer.draw_bubbles(bubbles)

        # Draw cities
        self.renderer.draw_cities(positions, size=self.renderer.city_size, color=self.renderer.color_city)

        # Draw pressure overlay
        if pressure is not None:
            wall_gap = outer_radius - inner_radius
            self.renderer.draw_pressure_overlay(pressure, wall_gap)

        # Draw pause indicator if paused
        self.renderer.draw_pause_indicator()

    def render_frame_with_manual_bubbles(self):
        """Render initial frame with manual bubble placement overlays."""
        if not self.options["draw"]:
            return

        if self.renderer is None or not self.renderer.is_initialized:
            return

        # Get current positions from GPU
        positions = self.physics_engine.get_positions_cpu()

        # Normalize positions for rendering
        if self.outer_radius > 0:
            positions_normalized = positions / self.outer_radius
            inner_r_normalized = self.inner_radius / self.outer_radius
            outer_r_normalized = 1.0
        else:
            positions_normalized = positions
            inner_r_normalized = self.inner_radius
            outer_r_normalized = self.outer_radius

        # Draw frame
        self.renderer.clear()
        self.renderer.draw_walls(inner_r_normalized, outer_r_normalized,
                                self.inner_direction, self.outer_direction,
                                width=self.renderer.wall_width)
        self.renderer.draw_cities(positions_normalized, size=self.renderer.city_size,
                                 color=self.renderer.color_city)

        # Draw manual bubbles
        self.renderer.draw_manual_bubbles()

        # Draw bubble placement indicator
        self.renderer.draw_bubble_placement_indicator()

        self.renderer.swap_buffers()

    def render_frame(self, step: int = 0):
        """
        Render current simulation state.

        Args:
            step: Current step number (for display)
        """
        if not self.options["draw"]:
            return

        if self.renderer is None or not self.renderer.is_initialized:
            return

        # Get current positions from GPU
        positions = self.physics_engine.get_positions_cpu()

        # Update and get density grid (pass outer_radius for normalization), if available
        # If density grid is not used, this will be a no-op
        self.physics_engine.update_density_grid(self.outer_radius)
        density_grid = self.physics_engine.get_density_grid_for_renderer()

        # Get bubble data if available
        # If bubbles are not used, this will be a no-op
        bubbles = self.physics_engine.get_bubble_data_for_renderer()

        # Get current pressure
        pressure = self.physics_engine.current_pressure

        # Normalize positions for rendering (to current outer radius)
        # This keeps the outer wall always at 1.0 visually
        if self.outer_radius > 0:
            positions_normalized = positions / self.outer_radius
            inner_r_normalized = self.inner_radius / self.outer_radius
            outer_r_normalized = 1.0  # Outer wall is always at 1.0

            # Bubbles are already in normalized [-1, 1] space, just copy them
            bubbles_normalized = bubbles.copy() if bubbles is not None else None
        else:
            positions_normalized = positions
            inner_r_normalized = self.inner_radius
            outer_r_normalized = self.outer_radius
            bubbles_normalized = bubbles

        # Draw frame (but don't swap buffers yet if recording video)
        should_record = (self.options.get("record_video", False) and
                        self.video_writer is not None and
                        step % self.options.get("video_record_frequency", 1) == 0)

        if should_record:
            # Draw without swapping so we can capture the back buffer
            self._draw_frame_no_swap(
                positions_normalized,
                inner_r_normalized,
                outer_r_normalized,
                self.inner_direction,
                self.outer_direction,
                bubbles=bubbles_normalized,
                density_grid=density_grid,
                grid_bins=self.physics_engine.grid_bins,
                pressure=pressure,
            )
            # Capture frame from back buffer before swapping
            frame = self.renderer.capture_frame(from_back_buffer=True)
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame_bgr)
            # Now swap buffers
            self.renderer.swap_buffers()
        else:
            # Normal drawing with swap
            self.renderer.draw_frame_complete(
                positions_normalized,
                inner_r_normalized,
                outer_r_normalized,
                self.inner_direction,
                self.outer_direction,
                bubbles=bubbles_normalized,
                density_grid=density_grid,
                grid_bins=self.physics_engine.grid_bins,
                pressure=pressure,
            )

        # Update debug window if enabled
        if self.debug_window and step % self.options.get("debug_update_frequency", 10) == 0:
            debug_data = self.physics_engine.get_debug_data()
            debug_data['step'] = step
            self.debug_window.update(debug_data)

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

        # stop_separation = self.get_smallest_distance() * 0.25
        stop_separation = dr * 10.0 #Stop with a buffer gap to avoid singularities near the end

        # Show initial configuration with bubble placement mode if bubbles enabled
        if self.options["draw"] and self.renderer:
            print("\nShowing initial configuration...")

            # Enable bubble placement mode if bubbles are enabled
            if self.nbody_options["use_bubbles"]:
                self.renderer.bubble_placement_mode = True
                print("\n" + "=" * 60)
                print("BUBBLE PLACEMENT MODE")
                print("=" * 60)
                print("Left Click: Place bubble spawn point")
                print("Right Click: Remove nearest bubble")
                print("Press SPACE to start with manual bubbles")
                print("Press 'D' to use dynamic (density-based) bubbles instead")
                print("=" * 60 + "\n")

            # Render loop for bubble placement
            waiting = self.options["pause_initial"]
            use_dynamic_bubbles = False

            while waiting:
                # Render frame with manual bubbles
                self.render_frame_with_manual_bubbles()

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nSimulation interrupted by user")
                        raise KeyboardInterrupt

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\nSimulation interrupted by user")
                            raise KeyboardInterrupt
                        elif event.key == pygame.K_d:
                            # Use dynamic bubbles instead
                            print("Switching to dynamic (density-based) bubble placement")
                            self.renderer.bubble_placement_mode = False
                            self.renderer.manual_bubbles = []  # Clear manual bubbles
                            use_dynamic_bubbles = True
                            waiting = False
                        elif event.key == pygame.K_SPACE or (not self.nbody_options["use_bubbles"]):
                            # Start simulation with current bubbles
                            print(f"Starting simulation with {len(self.renderer.manual_bubbles)} manual bubbles")
                            self.renderer.bubble_placement_mode = False
                            waiting = False

                    if event.type == pygame.MOUSEBUTTONDOWN and self.renderer.bubble_placement_mode:
                        # Handle bubble placement/removal
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        world_x, world_y = self.renderer.screen_to_world(mouse_x, mouse_y)

                        if event.button == 1:  # Left mouse button - add bubble
                            self.renderer.manual_bubbles.append((world_x, world_y))
                            print(f"Bubble placed at ({world_x:.2f}, {world_y:.2f}). Total: {len(self.renderer.manual_bubbles)}")
                        elif event.button == 3:  # Right mouse button - remove nearest bubble
                            if self.renderer.manual_bubbles:
                                # Find and remove nearest bubble
                                min_dist = float('inf')
                                nearest_idx = -1
                                for i, (bx, by) in enumerate(self.renderer.manual_bubbles):
                                    dist = np.sqrt((bx - world_x)**2 + (by - world_y)**2)
                                    if dist < min_dist:
                                        min_dist = dist
                                        nearest_idx = i

                                if nearest_idx >= 0 and min_dist < 0.2:  # Only remove if close enough
                                    removed = self.renderer.manual_bubbles.pop(nearest_idx)
                                    print(f"Bubble removed at ({removed[0]:.2f}, {removed[1]:.2f}). Total: {len(self.renderer.manual_bubbles)}")
                                else:
                                    print("No bubble close enough to remove")

                pygame.time.wait(10)

            # Store whether to use dynamic bubbles
            self.use_dynamic_bubbles = use_dynamic_bubbles

            if not self.options["pause_initial"]:
                # If not pausing initially, just render once
                self.render_frame()

        # Main simulation loop (matches C++ structure)
        print("\nRunning N-body extrusion...")

        draw_count = 0
        total_steps = 0
        last_print_radius = 0.0

        # Main loop: while inner < outer - DR
        while self.inner_radius + stop_separation < self.outer_radius:
            if self.options["step_mode"] == "step":
                    self.renderer.wait_for_key()

            if self.renderer and self.renderer.is_paused:
                self.renderer.wait_for_unpause()

            self.step_walls(dr)

            for _ in range(self.options["steps_per_wall_move"]):
                # Integrate one step
                self.physics_engine.integrate_step(
                    self.inner_radius,
                    self.outer_radius,
                    bubbles=None,  # Bubbles handled in kernel
                )

                # Render periodically
                if (
                    self.options["draw"]
                    and draw_count % self.options["render_frequency"] == 0
                ):
                    self.render_frame(total_steps)

                    # Handle window events
                    if self.renderer and not self.renderer.handle_events():
                        print("\nSimulation interrupted by user")
                        break

                draw_count += 1
                total_steps += 1

            # Update bubbles based on density grid
            # Check if we should create/update bubbles (when inner wall is expanding)
            if self.inner_direction > 0:
                
                if self.nbody_options["use_density_grid"]:
                    # Update density grid to find clusters
                    self.physics_engine.update_density_grid(self.outer_radius)

                # Initialize bubbles if not already done
                if self.nbody_options["use_bubbles"] and self.nbody_options["use_density_grid"]:
                    if not self.physics_engine.bubbles_enabled:
                        # Get manual bubble positions from renderer if available and not using dynamic mode
                        manual_positions = None
                        if (not hasattr(self, 'use_dynamic_bubbles') or not self.use_dynamic_bubbles):
                            if self.renderer and hasattr(self.renderer, 'manual_bubbles') and len(self.renderer.manual_bubbles) > 0:
                                manual_positions = self.renderer.manual_bubbles

                        self.physics_engine.initialize_bubbles(self.inner_radius, self.outer_radius,
                                                              manual_positions=manual_positions)

                    # Grow existing bubbles
                    self.physics_engine.update_bubbles(dr, self.outer_radius)

            # Move walls after integration phase
            if self.nbody_options["use_pressure"]:
                pressure = self.compute_pressure()
                self.update_wall_directions(pressure)
            else:
                # Simple mode: always expand inner wall
                self.inner_direction = 1
                self.outer_direction = 0

            # Progress indicator (print every 0.1 units of inner radius)
            if self.inner_radius - last_print_radius >= 0.1:
                pressure = (
                    self.compute_pressure()
                    if self.nbody_options["use_pressure"]
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
        if self.options["draw"] and self.renderer:
            print("\nShowing final path...")
            positions_normalized = final_positions / self.outer_radius
            coords_normalized = self.coords / self.outer_radius

            # Ensure projection with padding is active for final path display
            # self.renderer.setup_padded_projection()

            self.renderer.clear()
            self.renderer.draw_path(coords_normalized, self.final_path)
            self.renderer.draw_cities(coords_normalized)

            # Record final path to video from back buffer (hold for 2 seconds worth of frames)
            if self.options.get("record_video", False) and self.video_writer is not None:
                fps = self.options.get("video_fps", 30)
                final_frames = fps * 2  # 2 seconds
                frame = self.renderer.capture_frame(from_back_buffer=True)
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    for _ in range(final_frames):
                        self.video_writer.write(frame_bgr)
                    print(f"Recorded final path to video ({final_frames} frames)")

            self.renderer.swap_buffers()

            if self.options["pause_initial"]:
                self.print_results()
                self.renderer.wait_for_key()
        else:
            self.print_results()

        # Clean up debug window
        if self.debug_window:
            self.debug_window.close()
            self.debug_window = None

        return self.final_path, self.final_cost

    def print_results(self):
        """Print simulation results."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)

        print(f"\nDataset: {self.coord_file.name}")
        print(f"Cities: {self.n_cities}")
        print(f"Elapsed time: {self.elapsed_time:.2f} seconds")
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
        print("\nComparing with Nearest Neighbor heuristic: ")
        print("Getting the best of 10 random starts...")
        nn_path, nn_cost, nn_duration = random_nearest_neighbor_tsp(self.data_loader.original_coords, num_samples=10)
        print(f"Nearest Neighbor cost: {nn_cost:.4f}")
        print(f"Nearest Neighbor elapsed time: {nn_duration:.2f} seconds")
        improvement = 100.0 * (nn_cost - self.final_cost) / nn_cost
        print(
            f"N-body vs NN: {improvement:+.2f}% "
            f"({'better' if improvement > 0 else 'worse'})"
        )

        print("\n" + "=" * 60 + "\n")

    def cleanup(self):
        """Clean up resources."""
        # Release video writer
        if self.video_writer is not None:
            if self.video_writer.isOpened():
                self.video_writer.release()
                print(f"\nVideo saved to: {self.options.get('video_output_path')}")
                self.video_writer = None

        # Close renderer
        if self.renderer:
            self.renderer.close()


def main():
    """Main entry point for running the simulator."""
    best_options = {}
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        coord_file = sys.argv[1]
    else:
        # Default dataset
        coord_file = "datasets/att48/coords.txt"
        best_options = att48

    print("N-Body TSP Simulator")
    print(f"Using dataset: {coord_file}\n")

    nbody_options = best_options.get("nbody_options", {})
    renderer_options = best_options.get("renderer_options", {})
    simulator_options = best_options.get("simulator_options", {})

    nbody_options.update({
        # "p": 2.0,
        # "q": 1.0,
        # "m": .10,
        "slope_repulsion": 50,
        "mag_attraction": 25,
        "force_cutoff_extra": 100,
        "WALL_STRENGTH": 2000.0,
        "MASS": 80,
        "DAMP": 200.0,
        # "force_mode": "piecewise",
        # "DT": 0.01,
        # "num_bubbles": 20,
        # "lower_pressure_limit": 1.0,
        # "upper_pressure_limit": 10.0,
    })

    simulator_options.update({
        # "use_pressure": True,
        "use_density_grid": True,
        "use_bubbles": True,
        # "draw": True,
        # "use_gpu": True,
        # "render_frequency": 1,
        # "debug_window": True,
        # "step_mode": "step",
        # "record_video": True,
        # "video_output_path": None,  # Auto-generate if None
        # "video_fps": 60,
        # "video_record_frequency": 1,  # Record every N frames (1 = every frame)
    })

    renderer_options = {
        "color_background": (1, 1, 1),
        "color_density": (0.5, 0.2, 1.0),
        "city_size": 16.0,
        "path_width": 4.0,
        "wall_width": 4.0,
        "padding": 0.5,
        "use_random_city_colors": True,
    }

    simulator = TSPNBodySimulator(
        coord_file, options=simulator_options,
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
