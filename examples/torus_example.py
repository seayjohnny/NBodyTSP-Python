"""
Example: Torus Mode N-Body TSP Simulator

Demonstrates the torus collapse approach ported from the JavaScript
NBodyTSP-Torus project. Features:
- 3D torus geometry with tube collapse
- Interactive UI controls
- 2-opt/3-opt local search
- Bayesian optimization

Usage:
    python examples/torus_example.py                      # Basic torus run
    python examples/torus_example.py --optimize           # With optimization
    python examples/torus_example.py datasets/att48/coords.txt  # Custom dataset
"""

import sys
from pathlib import Path
from tsp_nbody.simulator import TSPNBodySimulator


def run_torus_basic(coord_file: str = "datasets/grid4x4/coords.txt"):
    """Basic torus collapse simulation with UI controls."""
    print("\n" + "=" * 60)
    print("TORUS MODE: Basic Simulation")
    print("=" * 60)

    simulator = TSPNBodySimulator(
        coord_file,
        options={
            "mode": "torus",
            "draw": True,
            "render_frequency": 1,
            "substeps": 4,
            "use_local_search": True,
            "local_search_mode": "2-opt",
            "show_ui_panel": True,
            "pause_initial": True,
            "compare_nearest_neighbor": True,
            "use_gpu": False,
        },
        nbody_options={
            "shrink_rate": 0.10,
            "epsilon": 0.08,
            "lj_strength": 1.0,
            "perturbation": 0.50,
            "embed_mode": "flat",
        },
        renderer_options={
            "window_size": (600, 600),
            "color_background": (0.14, 0.13, 0.11),
            "city_size": 6.0,
            "path_width": 2.0,
            "wall_width": 2.0,
            "padding": 0.15,
            "use_random_city_colors": True,
        },
    )

    if not simulator.initialize_torus():
        print("Initialization failed!")
        return

    try:
        path, cost = simulator.run_torus_simulation()
        print(f"\nFinal tour cost: {cost:.2f}")

        output_path = Path(coord_file).with_suffix('.torus.tour.txt')
        simulator.save_results(str(output_path))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        simulator.cleanup()


def run_torus_optimize(coord_file: str = "datasets/grid4x4/coords.txt"):
    """Torus simulation with Bayesian parameter optimization."""
    print("\n" + "=" * 60)
    print("TORUS MODE: Bayesian Optimization")
    print("=" * 60)

    simulator = TSPNBodySimulator(
        coord_file,
        options={
            "mode": "torus",
            "draw": False,  # No rendering during optimization
            "substeps": 4,
            "use_local_search": True,
            "local_search_mode": "2-opt",
            "compare_nearest_neighbor": True,
            "use_gpu": False,
        },
        nbody_options={
            "embed_mode": "flat",
        },
    )

    if not simulator.initialize_torus():
        print("Initialization failed!")
        return

    try:
        # Run optimization
        result = simulator.run_optimization(max_trials=30)
        print(f"\nBest parameters: {result['best_params']}")
        print(f"Best cost: {result['best_cost']:,.0f}")

        # Run final simulation with best params (with visualization)
        simulator.options["draw"] = True
        simulator.options["show_ui_panel"] = True
        if simulator.renderer is None:
            from tsp_nbody.renderer import TSPRenderer, OPENGL_AVAILABLE
            if OPENGL_AVAILABLE:
                simulator.renderer_options["panel_width"] = 260
                simulator.renderer = TSPRenderer(options=simulator.renderer_options)
                simulator.renderer.initialize()

        simulator.torus_engine.initialize_physics()
        path, cost = simulator.run_torus_simulation()
        print(f"\nFinal tour cost with best params: {cost:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    coord_file = "datasets/grid4x4/coords.txt"
    do_optimize = False

    for arg in sys.argv[1:]:
        if arg == "--optimize":
            do_optimize = True
        elif not arg.startswith("--"):
            coord_file = arg

    if do_optimize:
        run_torus_optimize(coord_file)
    else:
        run_torus_basic(coord_file)
