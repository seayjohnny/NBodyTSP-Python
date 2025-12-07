"""
Example Usage of N-Body TSP Simulator

This script demonstrates various ways to use the simulator.
"""

from tsp_nbody.simulator import TSPNBodySimulator
from tsp_nbody.path_extraction import nearest_neighbor_tsp
import sys
import numpy as np


def example_basic():
    """Example 1: Basic usage with default parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    # Create simulator with default parameters
    simulator = TSPNBodySimulator("datasets/att48/coords.txt")

    # Initialize
    if not simulator.initialize():
        print("Initialization failed!")
        return

    # Run simulation
    path, cost = simulator.run_simulation()

    # Print results
    simulator.print_results()

    # Cleanup
    simulator.cleanup()


def example_custom_params():
    """Example 2: Running with custom parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom TSP Kernel Parameters")
    print("=" * 70)

    # Use TSP mode with custom Lennard-Jones-like parameters
    custom_params = {
        # TSP kernel parameters (Lennard-Jones-like)
        "p": 3.5,  # Repulsive exponent
        "q": 6.0,  # Attractive exponent
        "h": 1.5,  # Force scaling constant
        # Common physics parameters
        "wall_strength": 20000.0,
        "damping": 20.0,
        "mass": 80.0,
        "dt": 0.01,
        "dr": 0.01,
        "draw": False,
    }

    print("\nRunning with smooth power-law Lennard-Jones forces...")
    print(
        f"  Parameters: p={custom_params['p']}, q={custom_params['q']}, h={custom_params['h']}"
    )

    simulator = TSPNBodySimulator("datasets/att48/coords.txt", custom_params)

    if not simulator.initialize():
        print("Initialization failed!")
        return

    path, cost = simulator.run_simulation()
    simulator.print_results()
    simulator.cleanup()


def example_no_visualization():
    """Example 3: Running without visualization (headless mode)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Headless Mode (No Visualization)")
    print("=" * 70)

    params = {
        "draw": False,  # Disable visualization
        "use_gpu": True,  # Still use GPU for physics
        "wall_moves": 1000,
    }

    simulator = TSPNBodySimulator("datasets/att48/coords.txt", params)

    if not simulator.initialize():
        print("Initialization failed!")
        return

    # This will run much faster without rendering
    path, cost = simulator.run_simulation()
    simulator.print_results()
    simulator.cleanup()


def example_compare_methods():
    """Example 4: Compare N-body with nearest neighbor."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Method Comparison")
    print("=" * 70)

    dataset = "datasets/att48/coords.txt"

    # Run N-body
    print("\nRunning N-body method...")
    params = {"draw": False, "use_gpu": True}
    simulator = TSPNBodySimulator(dataset, params)
    simulator.initialize()
    nbody_path, nbody_cost = simulator.run_simulation()
    simulator.cleanup()

    # Run nearest neighbor
    print("\nRunning Nearest Neighbor method...")
    from tsp_nbody.dataio import TSPDataLoader

    loader = TSPDataLoader(dataset)
    loader.load_coordinates()
    nn_path = nearest_neighbor_tsp(loader.coords)
    nn_cost = simulator.path_extractor.calculate_path_cost(loader.coords, nn_path)

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON RESULTS")
    print("-" * 70)
    print(f"N-body cost:         {nbody_cost:.4f}")
    print(f"Nearest Neighbor:    {nn_cost:.4f}")
    print(f"Difference:          {nbody_cost - nn_cost:+.4f}")
    print(f"Percent difference:  {100 * (nbody_cost - nn_cost) / nn_cost:+.2f}%")

    if nbody_cost < nn_cost:
        print("\n✓ N-body solution is BETTER!")
    elif abs(nbody_cost - nn_cost) / nn_cost < 0.01:
        print("\n≈ Solutions are very similar")
    else:
        print("\n✗ Nearest Neighbor is better")


def example_batch_processing():
    """Example 5: Process multiple datasets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Batch Processing")
    print("=" * 70)

    datasets = [
        "datasets/att48/coords.txt",
        "datasets/bay29/coords.txt",
    ]

    params = {
        "draw": False,  # No visualization for batch
        "use_gpu": True,
        "wall_moves": 800,  # Faster for testing
    }

    results = []

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"Processing: {dataset}")
        print("=" * 70)

        simulator = TSPNBodySimulator(dataset, params)

        if not simulator.initialize():
            print(f"Failed to initialize {dataset}")
            continue

        path, cost = simulator.run_simulation()

        # Store results
        results.append(
            {
                "dataset": dataset,
                "cost": cost,
                "n_cities": simulator.n_cities,
                "optimal_cost": simulator.optimal_cost * simulator.normalizing_factor
                if simulator.optimal_cost
                else None,
            }
        )

        simulator.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<30} {'Cities':<8} {'Cost':<12} {'Optimal':<12} {'% Diff':<8}")
    print("-" * 70)

    for result in results:
        dataset_name = result["dataset"].split("/")[-2]
        cities = result["n_cities"]
        cost = result["cost"]
        opt = result["optimal_cost"]

        if opt:
            pct_diff = 100 * (cost - opt) / opt
            print(
                f"{dataset_name:<30} {cities:<8} {cost:<12.2f} {opt:<12.2f} {pct_diff:+.2f}%"
            )
        else:
            print(
                f"{dataset_name:<30} {cities:<8} {cost:<12.2f} {'N/A':<12} {'N/A':<8}"
            )


def example_custom_coordinates():
    """Example 6: Using custom coordinate data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Custom Coordinates")
    print("=" * 70)

    import numpy as np
    import tempfile
    import os

    # Generate random cities
    n_cities = 20
    np.random.seed(42)
    coords = np.random.randn(n_cities, 2) * 10.0

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    for x, y in coords:
        temp_file.write(f"{x} {y}\n")
    temp_file.close()

    print(f"Created temporary file with {n_cities} random cities")

    # Run simulator
    params = {"draw": True, "wall_moves": 500, "pause_initial": False}
    simulator = TSPNBodySimulator(temp_file.name, params)

    if simulator.initialize():
        path, cost = simulator.run_simulation()
        simulator.print_results()
        simulator.cleanup()

    # Clean up temporary file
    os.unlink(temp_file.name)
    print(f"Removed temporary file")


def example_param_sweep():
    """Example 7: Parameter sweep to analyze effects."""

    print("\n" + "=" * 70)
    print("EXAMPLE 7: Parameter Sweep")
    print("=" * 70)

    dataset = "datasets/att48/coords.txt"
    attraction_values = np.linspace(5, 50, 2)  # Attraction from 5 to 50
    force_cutoff_values = np.linspace(5, 55, 2)  # Force cutoff from 5 to 50
    repulsion_values = np.linspace(5, 50, 2)  # Repulsion from 5 to 50
    damping_values = np.linspace(20, 200, 2)  # Damping from 20 to 200

    results = []

    for attraction in attraction_values:
        for force_cutoff in force_cutoff_values:
            for repulsion in repulsion_values:
                for damping in damping_values:
                    print(
                        f"\nTesting attraction={attraction}, force_cutoff={force_cutoff}, repulsion={repulsion}, damping={damping}"
                    )
                    params = {
                        "draw": True,
                        "use_gpu": True,
                        "mag_attraction": attraction,
                        "force_cutoff_extra": force_cutoff,
                        "slope_repulsion": repulsion,
                        "damping": damping,
                        "pause_initial": False,
                    }

                simulator = TSPNBodySimulator(dataset, params)
                if not simulator.initialize():
                    print("Initialization failed!")
                    continue

                path, cost = simulator.run_simulation()
                percent_diff = simulator.path_extractor.compare_with_optimal(
                    cost,
                    simulator.optimal_cost * simulator.normalizing_factor
                    if simulator.optimal_cost
                    else None,
                )["percent_difference"]
                results.append(
                    (attraction, force_cutoff, repulsion, damping, cost, percent_diff)
                )
                simulator.cleanup()

    # Print summary
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 70)
    print(
        f"{'Attraction':<20} {'Force Cutoff':<12} {'Repulsion':<12} {'Damping':<12} {'Cost':<12} {'% Diff':<8}"
    )
    print("-" * 70)
    for attraction, force_cutoff, repulsion, damping, cost, percent_diff in results:
        print(
            f"{attraction:<20} {force_cutoff:<12} {repulsion:<12} {damping:<12} {cost:<12.2f} {percent_diff:+.2f}%"
        )


def main():
    """Run examples based on command line argument."""

    examples = {
        "1": ("Basic Usage", example_basic),
        "2": ("Power-Law TSP Kernel", example_custom_params),
        "3": ("Headless Mode", example_no_visualization),
        "4": ("Method Comparison", example_compare_methods),
        "5": ("Batch Processing", example_batch_processing),
        "6": ("Custom Coordinates", example_custom_coordinates),
        "7": ("Parameter Sweep", example_param_sweep),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nN-Body TSP Simulator - Examples")
        print("=" * 70)
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print("\nUsage: python examples.py [1-7]")
        print("       python examples.py all    # Run all examples\n")

        choice = input("Select example (1-7, or 'all'): ").strip()

    if choice.lower() == "all":
        for key, (name, func) in examples.items():
            try:
                func()
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\n\nExample {key} failed: {e}")
                import traceback

                traceback.print_exc()
    elif choice in examples:
        name, func = examples[choice]
        try:
            func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n\nExample failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"Invalid choice: {choice}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
