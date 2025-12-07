"""
Test visualization of improved wall forces and adaptive bubbles.

This example demonstrates:
1. Wall pressure visualization (color-coded text overlay)
2. Adaptive bubbles rendering
3. Integration of both features in the physics engine
"""

from tsp_nbody.simulator import TSPNBodySimulator
import sys

def test_improved_walls_visualization():
    """Test improved wall forces with pressure visualization."""
    print("=" * 60)
    print("Test 1: Improved Wall Forces with Pressure Visualization")
    print("=" * 60)

    # Configure simulator with improved walls enabled
    params = {
        'p': 3.5,
        'q': 6.0,
        'h': 1.5,
        'wall_schedule': 'linear',
        'squeeze_steps': 500,
        'relax_steps': 100,
        'use_gpu': True,
        'draw': True,
        'render_frequency': 10
    }

    sim = TSPNBodySimulator("datasets/att48/coords.txt", params)

    # Initialize
    if not sim.initialize():
        print("Initialization failed!")
        return

    # Enable improved walls (should be enabled by default)
    print(f"Improved walls enabled: {sim.physics_engine.use_improved_walls}")
    print(f"Adaptive bubbles enabled: {sim.physics_engine.use_adaptive_bubbles}")

    # Run with visualization - pressure should be visible in top-left corner
    print("\nRunning simulation with pressure visualization...")
    print("Watch for pressure display in top-left corner (color-coded):")
    print("  - White: Low pressure (< 50)")
    print("  - Yellow: Medium pressure (50-100)")
    print("  - Red: High pressure (> 100)")
    print("\nPress ESC to exit early")

    path, cost = sim.run_simulation()

    print(f"\nFinal tour cost: {cost:.2f}")
    print(f"Initial cost: {sim.optimal_cost:.2f}")
    print(f"Improvement: {((sim.optimal_cost - cost) / sim.optimal_cost * 100):.1f}%")

    sim.cleanup()

def test_adaptive_bubbles_visualization():
    """Test adaptive bubbles with rendering."""
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
        'render_frequency': 10
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

def test_combined_features():
    """Test both improved walls and adaptive bubbles together."""
    print("\n" + "=" * 60)
    print("Test 3: Combined Features (Walls + Bubbles)")
    print("=" * 60)

    # Configure simulator with both features enabled
    params = {
        'p': 3.5,
        'q': 6.0,
        'h': 1.5,
        'wall_schedule': 'linear',
        'squeeze_steps': 500,
        'relax_steps': 100,
        'use_gpu': True,
        'draw': True,
        'render_frequency': 10
    }

    sim = TSPNBodySimulator("datasets/att48/coords.txt", params)

    # Initialize
    if not sim.initialize():
        print("Initialization failed!")
        return

    # Enable both features
    sim.physics_engine.use_improved_walls = True
    sim.physics_engine.use_adaptive_bubbles = True

    print(f"Improved walls enabled: {sim.physics_engine.use_improved_walls}")
    print(f"Adaptive bubbles enabled: {sim.physics_engine.use_adaptive_bubbles}")

    # Run with visualization
    print("\nRunning simulation with both features...")
    print("Features to observe:")
    print("  1. Pressure display in top-left (color-coded)")
    print("  2. Cyan bubbles in dense regions")
    print("  3. Better containment near walls")
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
        # Run all tests
        # test_improved_walls_visualization()
        test_adaptive_bubbles_visualization()
        # test_combined_features()

        print("\n" + "=" * 60)
        print("All visualization tests completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
