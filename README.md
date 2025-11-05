# TSP N-Body Simulator

A physics-based approach to solving the Traveling Salesman Problem (TSP) using N-body simulation. This implementation models TSP cities as particles in a constrained physical system, using gravitational attraction and boundary forces to discover near-optimal tour paths.

## Overview

The TSP N-Body Simulator treats the Traveling Salesman Problem as a physical system where:
- Cities are represented as particles with mass
- Particles are confined within concentric circular walls
- Physical forces guide particles into an optimal ordering along the walls
- The final particle arrangement corresponds to a tour solution

This approach provides an alternative to traditional optimization algorithms by leveraging physical simulation dynamics.

## Key Features

- **Physics-based optimization**: Uses N-body gravitational simulation with wall constraints
- **GPU acceleration**: Optional CUDA/CuPy support for faster computation
- **Multiple solving methods**: Supports pressure-based and bubble (local density) methods
- **Real-time visualization**: Optional Pygame-based rendering of the simulation
- **Batch processing**: Run multiple simulations with different parameters
- **Flexible input**: Supports various TSP instance formats
- **Configurable parameters**: Extensive control over physics and simulation settings

## Installation

### Requirements

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management (recommended)
- CUDA Toolkit (optional, for GPU acceleration)

### Setup with uv (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd tsp-nbody-python

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install base dependencies (CPU-only)
uv sync

# OR install with GPU support for your CUDA version:
uv sync --extra cuda13  # For CUDA 13.x
uv sync --extra cuda12  # For CUDA 12.x
uv sync --extra cuda11  # For CUDA 11.x

# Install the package in editable mode
uv pip install -e .
```

### Alternative Setup with pip
```bash
# Clone the repository
git clone <repository-url>
cd tsp-nbody-python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base dependencies
pip install -e .

# OR install with GPU support:
pip install -e ".[cuda13]"  # For CUDA 13.x
pip install -e ".[cuda12]"  # For CUDA 12.x
pip install -e ".[cuda11]"  # For CUDA 11.x
```

### Verifying GPU Support

After installation, verify GPU acceleration is available:
```python
from tsp_nbody.physics_engine import GPU_AVAILABLE
print(f"GPU Available: {GPU_AVAILABLE}")
```

## Quick Start

### Basic Usage
```python
from tsp_nbody import TSPNBodySimulator

# Create simulator with default parameters
simulator = TSPNBodySimulator('datasets/bay29/coords.txt')

# Initialize the simulation
if simulator.initialize():
    # Run the simulation
    path, cost = simulator.run_simulation()
    
    # Print results
    simulator.print_results()
    
    # Cleanup
    simulator.cleanup()
```

### Using GPU Acceleration
```python
from tsp_nbody import TSPNBodySimulator

# Enable GPU acceleration in parameters
params = {
    'use_gpu': True,  # Enable GPU
    'draw': True,
    'wall_moves': 1000,
}

simulator = TSPNBodySimulator('datasets/bay29/coords.txt', params)
# ... rest of your code
```

### Command Line Interface

Run a simulation directly:
```bash
python main.py datasets/bay29/coords.txt
```

### Examples

The `examples/basic.py` file contains six comprehensive examples:

1. **Basic Usage**: Simple simulation with default parameters
2. **Custom Parameters**: Adjust physics and simulation settings
3. **Headless Mode**: Run without visualization for faster execution
4. **Method Comparison**: Compare pressure and bubble methods
5. **Batch Processing**: Process multiple datasets automatically
6. **Custom Coordinates**: Generate and solve random TSP instances

Run examples:
```bash
python examples/basic.py 1    # Run example 1
python examples/basic.py all  # Run all examples
```

## Project Structure
```
tsp-nbody-python/
├── tsp_nbody/              # Main package
│   ├── __init__.py         # Package initialization
│   ├── simulator.py        # Main simulator class
│   ├── physics_engine.py   # N-body physics calculations (GPU/CPU)
│   ├── path_extraction.py  # Tour extraction from particle positions
│   ├── renderer.py         # Pygame visualization
│   └── dataio.py          # Dataset I/O utilities
├── datasets/               # TSP benchmark instances
│   ├── bay29/             # 29-city instance
│   ├── att48/             # 48-city instance
│   ├── ch150/             # 150-city instance
│   └── ...
├── examples/              # Example scripts
│   └── basic.py           # Comprehensive examples
├── main.py                # Main entry point
├── pyproject.toml         # Project configuration
└── .python-version        # Python version specification
```

## Configuration

### Simulation Parameters

The simulator accepts a dictionary of parameters:
```python
params = {
    # Visualization
    'draw': True,              # Enable/disable rendering
    'pause_initial': True,     # Pause before simulation starts
    
    # Performance
    'use_gpu': True,           # Enable GPU acceleration (requires CuPy)
    
    # Physics
    'wall_moves': 1000,        # Number of wall compression steps
    'inner_rad': 0.05,         # Initial inner wall radius
    'outer_rad': 0.80,         # Initial outer wall radius
    'grav_const': 0.25,        # Gravitational constant
    
    # Bubbles (local density method)
    'use_bubbles': False,      # Enable bubble method
    'bubble_density': 0.5,     # Density threshold for bubble insertion
    
    # Advanced
    'particle_mass': 1.0,      # Mass of each particle
    'dt': 0.01,               # Simulation timestep
}

simulator = TSPNBodySimulator('coords.txt', params)
```

### Key Parameters

- **use_gpu**: Enable GPU acceleration (requires CuPy installation with matching CUDA version)
- **wall_moves**: Controls simulation duration (more = potentially better solutions, but slower)
- **inner_rad/outer_rad**: Initial wall positions (affects convergence behavior)
- **grav_const**: Strength of gravitational attraction between particles
- **use_bubbles**: Enable the bubble method for non-uniform city distributions

## Methods

### Pressure Method

The default method uses two circular walls that gradually compress inward, forcing particles to organize along the walls through gravitational attraction and boundary forces. This works well for uniformly distributed cities.

### Bubble Method

An enhanced method that addresses local clustering issues in non-uniform distributions by:
1. Creating a density map of particle positions
2. Inserting additional "bubble" forces in high-density regions
3. Breaking apart local clusters for better global ordering

Enable with `use_bubbles=True` in parameters.

## Datasets

The repository includes several standard TSP benchmark instances:

- **bay29**: 29 cities (optimal: 9,291.35)
- **att48**: 48 cities (optimal: 33,523.71)
- **ch150**: 150 cities (optimal: 6,532.28)
- **pres8**: 8 cities (simple test case)
- **rand8, rand128**: Random generated instances

Each dataset directory contains:
- `coords.txt`: City coordinates (x, y per line)
- `path.txt`: Optimal or reference tour
- `tour_len.txt`: Optimal or reference tour length

## Performance

Results vary based on parameters and instance characteristics. Example results:

| Instance | Optimal Cost | N-Body Cost (CPU) | N-Body Cost (GPU) | Error % | Runtime (CPU) | Runtime (GPU) |
|----------|--------------|-------------------|-------------------|---------|---------------|---------------|
| bay29    | 9,291.35     | ~9,400            | ~9,400            | ~1.2%   | ~5s           | ~2s           |
| att48    | 33,523.71    | ~35,700           | ~35,700           | ~6.5%   | ~7s           | ~3s           |
| ch150    | 6,532.28     | ~7,100            | ~7,100            | ~8.7%   | ~25s          | ~10s          |

Note: GPU acceleration provides significant speedups for larger instances. The bubble method generally provides better results for non-uniform distributions but may increase runtime.

## API Reference

### TSPNBodySimulator

Main simulator class for running N-body TSP simulations.

**Methods:**

- `__init__(coord_file, params=None)`: Initialize simulator
- `initialize()`: Set up simulation components
- `run_simulation()`: Execute the simulation
- `print_results()`: Display results summary
- `cleanup()`: Clean up resources

### PhysicsEngine

Handles N-body physics calculations and wall interactions. Automatically selects GPU (CuPy) or CPU (NumPy) backend based on availability and configuration.

**Key Features:**
- Automatic GPU/CPU backend selection
- N-body force calculations
- Wall collision detection and response
- Bubble force implementation

### PathExtraction

Extracts tour paths from final particle configurations using various algorithms.

### Renderer

Provides real-time Pygame visualization of the simulation.

### DataIO

Utilities for loading and saving TSP instance data.

## Visualization

When `draw=True`, the simulation displays:
- Red circles: City particles
- Pink circles: Outer and inner walls
- Green circles: Bubbles (if enabled)
- Real-time statistics overlay

Press `Ctrl+C` to stop the simulation early.

## Development

### Using uv for Development
```bash
# Install with all extras for development
uv sync --all-extras

# Add a new dependency
uv add <package-name>

# Add a CUDA-specific dependency
# Edit pyproject.toml to add to the appropriate [project.optional-dependencies] section

# Run tests (if available)
pytest

# Format code
black tsp_nbody/

# Lint code
ruff check tsp_nbody/
```

### CUDA Versions

The project supports multiple CUDA versions through optional dependencies:

- **cuda13**: For CUDA 13.x (installs `cupy-cuda13x`)
- **cuda12**: For CUDA 12.x (installs `cupy-cuda12x`)
- **cuda11**: For CUDA 11.x (installs `cupy-cuda11x`)

Choose the version that matches your installed CUDA Toolkit. You can check your CUDA version with:
```bash
nvcc --version
```

## Contributing

Contributions are welcome. Areas for improvement:
- Additional TSP benchmark instances
- Parameter optimization algorithms
- Alternative path extraction methods
- Performance optimizations
- Extended visualization features
- Support for additional CUDA versions

## License

This project is provided as-is for educational and research purposes.

## References

This implementation is based on the N-body approach to TSP solving as described in the accompanying paper. The method demonstrates how physical simulation can be applied to combinatorial optimization problems.

## Troubleshooting

**Simulation not converging**: Try adjusting `wall_moves`, `grav_const`, or wall radii.

**Poor results on clustered data**: Enable bubbles with `use_bubbles=True` and adjust `bubble_density`.

**Slow performance**: 
- Disable visualization with `draw=False`
- Reduce `wall_moves`
- Enable GPU acceleration with `use_gpu=True` (requires CuPy)

**GPU not being used**: 
- Verify CuPy is installed: `python -c "import cupy; print(cupy.__version__)"`
- Check CUDA installation: `nvcc --version`
- Ensure you installed the correct CUDA extra: `uv sync --extra cuda13` (or cuda12/cuda11)
- Verify in code: `from tsp_nbody.physics_engine import GPU_AVAILABLE; print(GPU_AVAILABLE)`

**Import errors**: 
- Ensure all dependencies are installed via `uv sync`
- For GPU support: `uv sync --extra cuda13` (replace with your CUDA version)

**CUDA version mismatch**: Install the CuPy package matching your CUDA Toolkit version using the appropriate extra flag.

**uv installation issues**: Visit [uv documentation](https://github.com/astral-sh/uv) for platform-specific installation instructions.

## Contact

For questions, issues, or contributions, please refer to the project repository or accompanying documentation.