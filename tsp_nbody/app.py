"""
Application Controller for N-Body TSP Simulator.

Manages the app lifecycle: setup screen -> simulation -> back to setup.
Handles mode switching, dataset switching, and theme changes without restarting.
"""

import sys
import time
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from tsp_nbody.setup_screen import show_setup_screen
from tsp_nbody.themes import get_theme
from tsp_nbody.ui_controls import ControlPanel, PANEL_WIDTH
from tsp_nbody.dataio import TSPDataLoader, load_optimal_cost
from tsp_nbody.path_extraction import PathExtractor, random_nearest_neighbor_tsp
from tsp_nbody.local_search import improve as local_search_improve, make_euclidean_dist_fn, tour_distance
from tsp_nbody.torus_physics import TorusPhysicsEngine
from tsp_nbody.physics_engine import NBodyPhysicsEngine, default_nbody_options

from pathlib import Path


# Exit reasons from simulation loop
EXIT_QUIT = "quit"
EXIT_MENU = "menu"
EXIT_FINISHED = "finished"


class App:
    """Main application controller."""

    def run(self):
        """Main app loop: setup screen -> simulation -> repeat."""
        while True:
            # Show setup screen (owns its own pygame window)
            config = show_setup_screen()
            if config is None:
                # User closed setup
                return 0

            # Run simulation with the chosen config
            exit_reason = self._run_simulation(config)

            if exit_reason == EXIT_QUIT:
                return 0
            # EXIT_MENU or EXIT_FINISHED -> loop back to setup

    def _run_simulation(self, config: dict) -> str:
        """
        Run a simulation with the given config. Returns exit reason.

        Config keys: dataset_path, dataset_name, mode, wall_force_mode, theme
        """
        mode = config["mode"]
        theme = get_theme(config["theme"])

        if mode == "torus":
            return self._run_torus(config, theme)
        else:
            return self._run_2d(config, theme)

    # =================================================================
    # TORUS MODE
    # =================================================================
    def _run_torus(self, config: dict, theme: dict) -> str:
        from tsp_nbody.torus_renderer import TorusRenderer

        # Load data
        data_loader = TSPDataLoader(config["dataset_path"])
        data_loader.preprocess(normalize_method="minimum")
        original_coords = data_loader.original_coords
        n_cities = data_loader.n_cities

        # Load optimal cost
        optimal_cost = None
        dataset_dir = Path(config["dataset_path"]).parent
        opt_file = dataset_dir / "tour_len.txt"
        if opt_file.exists():
            optimal_cost = load_optimal_cost(str(opt_file))

        # Create engine
        engine = TorusPhysicsEngine(original_coords, options={
            'use_gpu': False,
            'shrink_rate': 0.10,
            'epsilon': 0.08,
            'lj_strength': 1.0,
            'perturbation': 0.50,
            'embed_mode': 'flat',
        })
        engine.initialize_physics()

        # Create renderer
        renderer = TorusRenderer(
            sim_size=(700, 700),
            panel_width=PANEL_WIDTH,
            title=f"N-Body TSP · {config['dataset_name']} · Torus",
        )
        if not renderer.initialize():
            print("Renderer failed to initialize")
            return EXIT_QUIT

        # Set background from theme
        from OpenGL.GL import glClearColor
        bg = theme["background"]
        glClearColor(bg[0], bg[1], bg[2], 1.0)

        # Create control panel
        panel = ControlPanel(PANEL_WIDTH, mode="torus", theme_name=config["theme"], panel_height=700)
        renderer.control_panel = panel

        # Sync slider values
        panel.set_slider_value('shrink_rate', engine.shrink_rate)
        panel.set_slider_value('epsilon', engine.epsilon)
        panel.set_slider_value('lj_strength', engine.lj_strength)
        panel.set_slider_value('perturbation', engine.perturbation)
        panel.set_slider_value('speed', 4)

        substeps = 4
        particle_ids = list(range(1, n_cities + 1))
        dist_fn = make_euclidean_dist_fn(original_coords)

        fps_timer = time.perf_counter()
        fps_count = 0
        current_fps = 0
        exit_reason = EXIT_QUIT

        # Main loop
        running = True
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_reason = EXIT_QUIT
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        exit_reason = EXIT_MENU
                        running = False
                        break
                    if event.key == pygame.K_SPACE:
                        if not engine.collapsing and not engine.collapsed:
                            engine.start_collapse()
                        else:
                            renderer.is_paused = not renderer.is_paused
                    if event.key == pygame.K_l:
                        renderer.show_labels = not renderer.show_labels

                # Camera
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if mx < renderer.sim_size[0]:
                        if event.button == 1:
                            renderer.is_dragging = True
                            renderer.last_mx = mx
                            renderer.last_my = my
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    renderer.is_dragging = False
                if event.type == pygame.MOUSEMOTION and renderer.is_dragging:
                    mx, my = event.pos
                    renderer.cam_angle += (mx - renderer.last_mx) * 0.005
                    renderer.cam_pitch += (my - renderer.last_my) * 0.005
                    renderer.cam_pitch = max(-1.2, min(1.2, renderer.cam_pitch))
                    renderer.last_mx = mx
                    renderer.last_my = my
                if event.type == pygame.MOUSEWHEEL:
                    renderer.cam_dist = max(3.0, min(18.0, renderer.cam_dist - event.y * 0.5))

                # Panel
                changes = panel.handle_event(event, renderer.sim_size[0])
                for key, value in changes.items():
                    if key == 'shrink_rate': engine.shrink_rate = value
                    elif key == 'epsilon': engine.epsilon = value
                    elif key == 'lj_strength': engine.lj_strength = value
                    elif key == 'perturbation': engine.perturbation = value
                    elif key == 'speed': substeps = int(value)
                    elif key == 'collapse':
                        if not engine.collapsed:
                            engine.start_collapse()
                    elif key == 'reset':
                        engine.initialize_physics()
                    elif key == 'embed_mode':
                        engine.embed_mode = value
                        if not engine.collapsing:
                            engine.initialize_physics()
                    elif key == 'menu':
                        exit_reason = EXIT_MENU
                        running = False

            if not running:
                break

            # Update pygame-gui manager
            panel.update(1.0 / 60.0)

            if renderer.is_paused:
                # Still render but don't advance physics
                self._render_torus(renderer, engine, particle_ids, theme)
                pygame.time.wait(16)
                continue

            # Physics
            engine.run_substeps(substeps)

            # Render
            self._render_torus(renderer, engine, particle_ids, theme)

            # FPS
            fps_count += 1
            now = time.perf_counter()
            if now - fps_timer > 0.5:
                current_fps = int(fps_count / (now - fps_timer))
                fps_timer = now
                fps_count = 0

            # Update stats
            extra = {'tube_r': f"{engine.r:.3f}", 'major_r': f"{engine.R:.3f}"}
            if engine.found_tour:
                td = tour_distance(engine.found_tour, dist_fn)
                extra['tour_dist'] = f"{td:,.0f}"
                if optimal_cost:
                    gap = (td - optimal_cost) / optimal_cost * 100
                    extra['gap'] = f"{gap:.1f}%"
            panel.update_stats(phase=engine.phase, fps=current_fps, **extra)

        renderer.close()
        return exit_reason

    def _render_torus(self, renderer, engine, particle_ids, theme):
        pos = engine.get_positions_cpu()
        vel = engine.get_velocities_cpu()
        renderer.render(
            particles_pos=pos, particles_vel=vel,
            R=engine.R, r=engine.r, r0=engine.r0,
            circle_phase=engine.circle_phase,
            found_tour=engine.found_tour,
            particle_ids=particle_ids,
        )

    # =================================================================
    # 2D MODE
    # =================================================================
    def _run_2d(self, config: dict, theme: dict) -> str:
        from tsp_nbody.renderer import TSPRenderer, OPENGL_AVAILABLE
        if not OPENGL_AVAILABLE:
            print("OpenGL not available for 2D mode")
            return EXIT_QUIT

        # Load data
        data_loader = TSPDataLoader(config["dataset_path"])
        data_loader.preprocess(normalize_method="minimum")
        coords = data_loader.coords
        original_coords = data_loader.original_coords
        n_cities = data_loader.n_cities

        # Load optimal cost
        optimal_cost = None
        dataset_dir = Path(config["dataset_path"]).parent
        opt_file = dataset_dir / "tour_len.txt"
        if opt_file.exists():
            optimal_cost = load_optimal_cost(str(opt_file))

        # Physics engine
        nbody_opts = default_nbody_options.copy()
        nbody_opts.update({
            'use_gpu': False,
            'wall_force_mode': config['wall_force_mode'],
            'WALL_STRENGTH': 20000.0 if config['wall_force_mode'] == 'linear' else 8.0,
        })
        physics = NBodyPhysicsEngine(coords, options=nbody_opts)
        physics.initialize_physics()

        # Renderer
        renderer_opts = {
            "window_size": (700, 700),
            "panel_width": PANEL_WIDTH,
            "color_background": theme["background"],
            "color_city": theme["city"],
            "color_path": theme["path"],
            "color_wall_contract": theme["wall_contract"],
            "color_wall_static": theme["wall_static"],
            "color_wall_expand": theme["wall_expand"],
            "city_size": 6.0,
            "path_width": 2.0,
            "wall_width": 2.0,
            "padding": 0.15,
        }
        renderer = TSPRenderer(options=renderer_opts)
        if not renderer.initialize():
            print("Renderer failed")
            return EXIT_QUIT

        # Control panel
        panel = ControlPanel(PANEL_WIDTH, mode="2d", theme_name=config["theme"], panel_height=700)
        renderer.control_panel = panel

        # Sync sliders
        panel.set_slider_value('wall_strength', physics.WALL_STRENGTH)
        panel.set_slider_value('damp', physics.DAMP)
        panel.set_slider_value('dt', physics.DT)
        panel.set_slider_value('dr', physics.DR)
        panel.set_slider_value('slope_repulsion', physics.slope_repulsion)
        panel.set_slider_value('mag_attraction', physics.mag_attraction)

        # Wall force dropdown
        if 'wall_force' in panel.dropdowns:
            wf_idx = 0 if config['wall_force_mode'] == 'linear' else 1
            panel.dropdowns['wall_force'].selected = wf_idx

        # Simulation state
        inner_radius = 0.0
        outer_radius = data_loader.get_bounding_circle_radius()
        dr = physics.DR
        stop_separation = dr * 10.0
        sim_running = False
        sim_finished = False

        path_extractor = PathExtractor()
        final_path = None
        final_cost = None

        fps_timer = time.perf_counter()
        fps_count = 0
        current_fps = 0
        exit_reason = EXIT_QUIT

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit_reason = EXIT_QUIT
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        exit_reason = EXIT_MENU
                        running = False
                        break
                    if event.key == pygame.K_SPACE:
                        if not sim_running and not sim_finished:
                            sim_running = True
                        else:
                            renderer.is_paused = not renderer.is_paused

                # Panel
                changes = panel.handle_event(event, renderer.gl_viewport_size[0])
                for key, value in changes.items():
                    if key == 'wall_strength': physics.WALL_STRENGTH = value
                    elif key == 'damp': physics.DAMP = value
                    elif key == 'dt': physics.DT = value
                    elif key == 'dr':
                        physics.DR = value
                        dr = value
                        stop_separation = dr * 10.0
                    elif key == 'slope_repulsion': physics.slope_repulsion = value
                    elif key == 'mag_attraction': physics.mag_attraction = value
                    elif key == 'start':
                        sim_running = True
                    elif key == 'reset':
                        physics.initialize_physics()
                        inner_radius = 0.0
                        outer_radius = data_loader.get_bounding_circle_radius()
                        sim_running = False
                        sim_finished = False
                        final_path = None
                        final_cost = None
                    elif key == 'wall_force':
                        physics.wall_force_mode = value
                        if value == 'inverse_square':
                            physics.WALL_STRENGTH = 8.0
                            panel.set_slider_value('wall_strength', 8.0)
                        else:
                            physics.WALL_STRENGTH = 20000.0
                            panel.set_slider_value('wall_strength', 20000.0)
                    elif key == 'menu':
                        exit_reason = EXIT_MENU
                        running = False

            if not running:
                break

            # Update pygame-gui manager
            panel.update(1.0 / 60.0)

            # Physics step
            if sim_running and not sim_finished and not renderer.is_paused:
                inner_radius += dr
                physics.integrate_step(inner_radius, outer_radius)

                if inner_radius + stop_separation >= outer_radius:
                    sim_running = False
                    sim_finished = True
                    # Extract path
                    final_path = physics.get_final_tour()
                    final_cost = path_extractor.calculate_path_cost(original_coords, final_path)

            # Render
            positions = physics.get_positions_cpu()
            if outer_radius > 0:
                pos_norm = positions / outer_radius
                ir_norm = inner_radius / outer_radius
                or_norm = 1.0
            else:
                pos_norm = positions
                ir_norm = inner_radius
                or_norm = outer_radius

            inner_dir = 1 if sim_running else 0
            outer_dir = 0

            if sim_finished and final_path is not None:
                coords_norm = coords / outer_radius
                renderer.draw_frame_complete(
                    pos_norm, ir_norm, or_norm, inner_dir, outer_dir,
                    path=final_path, coords=coords_norm,
                )
            else:
                renderer.draw_frame_complete(pos_norm, ir_norm, or_norm, inner_dir, outer_dir)

            # FPS
            fps_count += 1
            now = time.perf_counter()
            if now - fps_timer > 0.5:
                current_fps = int(fps_count / (now - fps_timer))
                fps_timer = now
                fps_count = 0

            # Stats
            phase = "FINISHED" if sim_finished else ("RUNNING" if sim_running else "READY")
            extra = {
                'inner_r': f"{inner_radius:.3f}",
                'outer_r': f"{outer_radius:.3f}",
                'wall_gap': f"{outer_radius - inner_radius:.3f}",
            }
            if final_cost is not None:
                extra['tour_cost'] = f"{final_cost:,.0f}"
                if optimal_cost:
                    gap = (final_cost - optimal_cost) / optimal_cost * 100
                    extra['gap'] = f"{gap:.1f}%"
            panel.update_stats(phase=phase, fps=current_fps, **extra)

        renderer.close()
        return exit_reason
