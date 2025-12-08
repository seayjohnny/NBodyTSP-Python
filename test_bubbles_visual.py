"""Quick visual test for bubble rendering."""
from tsp_nbody.simulator import TSPNBodySimulator

# Configure for visual bubble test
nbody_options = {
    'slope_repulsion': 50.0,
    'mag_attraction': 25,
    'force_cutoff_extra': 100,
    'WALL_STRENGTH': 200.0,
    'MASS': 80,
    'DAMP': 20.0,
    'force_mode': 'piecewise',
    'DT': 0.01,
    'use_gpu': True,
    'use_pressure': False,  # Simple mode - always expand inner wall
}

renderer_options = {
    'city_size': 8.0,
    'path_width': 2.0,
    'wall_width': 2.0,
    'padding': 0.2,
}

params = {
    'draw': True,
    'render_frequency': 1,
    'steps_per_wall_move': 1,
    'pause_initial': False,
}

simulator = TSPNBodySimulator(
    'datasets/att48/coords.txt',
    params=params,
    nbody_options=nbody_options,
    renderer_options=renderer_options
)

if simulator.initialize():
    print("\n" + "=" * 60)
    print("BUBBLE VISUALIZATION TEST")
    print("=" * 60)
    print("\nYou should see:")
    print("  1. Density grid (8x8) with colored cells")
    print("  2. Cyan bubbles appearing at dense clusters")
    print("  3. Bubbles growing as the inner wall expands")
    print("\nPress ESC to exit\n")

    # Force inner wall to expand (triggers bubble creation)
    simulator.inner_direction = 1
    simulator.outer_direction = 0

    # Run simulation
    try:
        path, cost = simulator.run_simulation()
        print(f"\nSimulation complete! Final cost: {cost:.4f}")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        simulator.cleanup()
else:
    print("Failed to initialize simulator")
