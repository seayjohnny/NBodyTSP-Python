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
    print(f"{'Dataset':<30} {'Cities':<8} {'Cost':<12} {'Optimal':<12} {'% error':<8}")
    print("-" * 70)

    for result in results:
        dataset_name = result["dataset"].split("/")[-2]
        cities = result["n_cities"]
        cost = result["cost"]
        opt = result["optimal_cost"]

        if opt:
            pct_error = 100 * (cost - opt) / opt
            print(
                f"{dataset_name:<30} {cities:<8} {cost:<12.4f} {opt:<12.4f} {pct_error:+.4f}%"
            )
        else:
            print(
                f"{dataset_name:<30} {cities:<8} {cost:<12.4f} {'N/A':<12} {'N/A':<8}"
            )


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
                percent_error = simulator.path_extractor.compare_with_optimal(
                    cost,
                    simulator.optimal_cost * simulator.normalizing_factor
                    if simulator.optimal_cost
                    else None,
                )["percent_error"]
                results.append(
                    (attraction, force_cutoff, repulsion, damping, cost, percent_error)
                )
                simulator.cleanup()

    # Print summary
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 70)
    print(
        f"{'Attraction':<20} {'Force Cutoff':<12} {'Repulsion':<12} {'Damping':<12} {'Cost':<12} {'% error':<8}"
    )
    print("-" * 70)
    for attraction, force_cutoff, repulsion, damping, cost, percent_error in results:
        print(
            f"{attraction:<20} {force_cutoff:<12} {repulsion:<12} {damping:<12} {cost:<12.4f} {percent_error:+.4f}%"
        )


def example_random_dataset():
    """Example 5: Generate and solve a random dataset."""

    print("\n" + "=" * 70)
    print("EXAMPLE 5: Random Dataset")
    print("=" * 70) # Scale to 100x100 area 

    options = {
        'draw': False,
        'run_brute_force': True,
    }

    n_cities = 9
    num_runs = 100
    total_percent_error = 0.0
    total_best_nn_percent_error = 0.0
    total_first_nn_percent_error = 0.0

    # Create results CSV file
    with open(f"random_dataset_results_{n_cities}_cities.csv", "w") as f:
        header = "Run,Optimal Cost,N-Body Cost,N-Body Error,"
        for i in range(n_cities):
            header += f"NN Run {i+1} Cost,NN Run {i+1} Error,"
        
        header = header.rstrip(",")
        f.write(header + "\n")

    for run in range(num_runs):
        np.random.seed(run)
        random_coords = np.random.rand(n_cities, 2)
        simulator = TSPNBodySimulator(
            coords=random_coords,
            options=options
        )

        if not simulator.initialize():
            print("Initialization failed!")
            return

        results = simulator.run_simulation()
        simulator.cleanup()

        nn_runs = simulator.results['nn_results']['runs']
        run_results = []
        for nn_run in nn_runs:
            cost = nn_run['cost'] * simulator.normalizing_factor
            run_comparison = simulator.path_extractor.compare_with_optimal(
                cost, simulator.optimal_cost
            )
            run_results.append((cost, run_comparison['percent_error']))


        # Write to CSV
        with open(f"random_dataset_results_{n_cities}_cities.csv", "a") as f:
            line = f"{run+1},{simulator.optimal_cost:.4f},{simulator.results['final_cost']:.4f},{simulator.results['optimal_comparison']['percent_error']:.4f},"
            for cost, error in run_results:
                line += f"{cost:.4f},{error:.4f},"
            f.write(line.rstrip(",") + "\n")

        total_percent_error += simulator.optimal_comparison['percent_error']
        total_best_nn_percent_error += simulator.best_nn_comparison['percent_error']
        total_first_nn_percent_error += simulator.first_nn_comparison['percent_error']

    avg_percent_error = total_percent_error / num_runs
    avg_best_nn_percent_error = total_best_nn_percent_error / num_runs
    avg_first_nn_percent_error = total_first_nn_percent_error / num_runs


    print(f"\nAverage Percent Error from Optimal over {num_runs} runs: {avg_percent_error:.4f}%")
    print(f"Average Percent Error for Best Nearest Neighbor over {num_runs} runs: {avg_best_nn_percent_error:.4f}%")
    print(f"Average Percent Error for First Nearest Neighbor over {num_runs} runs: {avg_first_nn_percent_error:.4f}%")


def main():
    """Run examples based on command line argument."""

    examples = {
        "1": ("Basic Usage", example_basic),
        "2": ("Headless Mode", example_no_visualization),
        "3": ("Batch Processing", example_batch_processing),
        "4": ("Parameter Sweep", example_param_sweep),
        "5": ("Random Dataset", example_random_dataset),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nN-Body TSP Simulator - Examples")
        print("=" * 70)
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print("\nUsage: python examples.py [1-4]")
        print("       python examples.py all    # Run all examples\n")

        choice = input("Select example (1-4, or 'all'): ").strip()

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
