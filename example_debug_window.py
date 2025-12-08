"""
Example: Running TSP N-Body Simulator with Debug Window

This script demonstrates how to enable the debug window feature
that displays real-time information about all particles.
"""

from pathlib import Path
from tsp_nbody.simulator import TSPNBodySimulator


def main():
    """Run TSP simulation with debug window enabled."""

    # Path to your TSP data file
    # Adjust this to point to your actual data file
    data_dir = Path("data")
    coord_file = data_dir / "berlin52.tsp"

    # Check if file exists
    if not coord_file.exists():
        print(f"Error: Data file not found: {coord_file}")
        print("Please update the path to your TSP data file.")
        return

    # Configure simulation parameters
    params = {
        "debug_window": True,  # Enable debug window
        "debug_update_frequency": 10,  # Update every 10 render frames
        "draw": True,
        "render_frequency": 10,
        "pause_initial": True,
        "step_mode": "continuous",
    }

    # Optional: Configure renderer options
    renderer_options = {
        "city_size": 8.0,
        "path_width": 2.0,
        "wall_width": 2.0,
    }

    # Create and run simulator
    print("Creating TSP N-Body Simulator with Debug Window...")
    simulator = TSPNBodySimulator(
        coord_file=str(coord_file),
        params=params,
        renderer_options=renderer_options,
    )

    # Initialize simulator
    if not simulator.initialize():
        print("Failed to initialize simulator")
        return

    # Run simulation
    print("\nStarting simulation...")
    print("The debug window will show real-time particle data:")
    print("  - Position (X, Y)")
    print("  - Velocity (Vx, Vy)")
    print("  - Acceleration (Ax, Ay)")
    print("  - Distance from origin")
    print()

    final_path, final_cost = simulator.run_simulation()

    print(f"\nSimulation complete!")
    print(f"Final tour cost: {final_cost:.4f}")


if __name__ == "__main__":
    main()
