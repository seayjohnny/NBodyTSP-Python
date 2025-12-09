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
├── tsp_nbody/                  # Main package
│   ├── __init__.py             # Package initialization
│   ├── advanced_features.py    # Additional features
│   ├── best_results.py         # Distionaries of best options found so far
│   ├── debug_window.py         # Debug window class
│   ├── simulator.py            # Main simulator class
│   ├── physics_engine.py       # N-body physics calculations (GPU/CPU)
│   ├── path_extraction.py      # Tour extraction from particle positions
│   ├── renderer.py             # Pygame visualization
│   └── dataio.py               # Dataset I/O utilities
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

### Simulation Options

The simulator accepts a dictionary of parameters:
```python
simulator_options = {
    # Common options
   "steps_per_wall_move": 1,
    "use_gpu": True,
    "use_pressure": False,
    "use_density_grid": False,
    "use_bubbles": False,

    # Rendering options
    "draw": True,
    "render_frequency": 10,
    "pause_initial": True,
    "step_mode": "continuous",

    # Video recording options
    "record_video": False,
    "video_output_path": None,
    "video_fps": 30,
    "video_record_frequency": 1,  # Record every frame

    # Debug options
    "debug_window": False,
    "debug_update_frequency": 10,
}

simulator = TSPNBodySimulator('coords.txt', options=simulator_options)
```

### Additional Options
The simulator also accepts additional options for the n-body engine and rendering:
```python
nbody_options = {
    'use_gpu': GPU_AVAILABLE,
    'DAMP': 20.0,
    'MASS': 80.0,
    'WALL_STRENGTH': 20000.0,
    'FORCE_CUTOFF': 100000.0,
    'DT': 0.01,
    'DR': 0.01,
    'force_mode': 'piecewise',
    'slope_repulsion': 50.0,
    'mag_attraction': 0.5,
    'force_cutoff_extra': 0.10,
    'p': 6,
    'q': 12,
    'm': -0.05,
    "lower_pressure_limit": 1.0,
    "upper_pressure_limit": 10.0,
    'use_pressure': True,
    'use_density_grid': False,
    'use_bubbles': False,
}

rendering_options = {
    "window_size": (800, 800),
    "title": "N-Body TSP Simulator",
    "city_size": 5.0,
    "path_width": 2.0,
    "wall_width": 2.0,
    "padding": 0.1,  # 10% padding around edges
    "show_grid": False,
    "show_density": False,
    "show_bubbles": False,
    "use_random_city_colors": False,
    "color_background": (0.9, 0.9, 1.0),
    "color_city": (0.2, 0.0, 1.0),
    "color_path": (0.0, 0.5, 0.0),
    "color_wall_contract": (1.0, 0.0, 0.0),
    "color_wall_static": (0.3, 0.3, 0.0),
    "color_wall_expand": (0.0, 0.0, 1.0),
    "color_bubble": (0.2, 0.8, 1.0),
    "color_density": (0.5, 0.5, 0.5),
    "color_text": (1.0, 1.0, 1.0),
    "record_video": False,
    "video_fps": 30,
}

simulator = TSPNBodySimulator(
    'coords.txt', options=simulator_options,
    nbody_options=nbody_options, renderer_options=renderer_options
)
```

### Video Recording

You can record simulations as MP4 videos (ideal for YouTube):

```python
params = {
    'draw': True,
    'record_video': True,          # Enable video recording
    'video_output_path': None,     # Auto-generate filename if None
    'video_fps': 30,               # Frame rate (30 recommended for YouTube)
    'video_record_frequency': 1,   # Record every N frames (1 = every frame)
}

simulator = TSPNBodySimulator('datasets/bay29/coords.txt', params)
```

**Video Recording Parameters:**

- **record_video**: Enable/disable video recording
- **video_output_path**: Output file path (auto-generated in `videos/` folder if None)
- **video_fps**: Video frame rate (default: 30 FPS, good for YouTube)
- **video_record_frequency**: Record every Nth rendered frame (1 = every frame, higher = faster processing)

Videos are saved as MP4 files using the H.264 codec, which is optimized for YouTube uploads. The final path visualization is held for 2 seconds at the end of each video.

**Requirements:** Video recording requires OpenCV to be installed:
```bash
uv pip install opencv-python
# or
pip install opencv-python
```

## Methods

### General Method

The default method uses two circular walls that gradually compress inward, forcing particles to organize along the walls through gravitational attraction and boundary forces. This works well for uniformly distributed cities.

### Pressure Method

The pressure method allows the outer wall grow based on the amount of force the particles are enacting on the outer wall. This is intended to help avoid singularities and explosions of particles that occur when many particles are pressed together in narrow gaps between the 2 walls. 

Enable with `use_pressure=True` in parameters.



### Bubble Method

An enhanced method that addresses local clustering issues in non-uniform distributions by:
1. Creating a density map of particle positions
2. Inserting additional "bubble" forces in high-density regions
3. Breaking apart local clusters for better global ordering

Enable with `use_density_grid=Truw` and `use_bubbles=True` in parameters.

## Datasets

The repository includes several standard TSP benchmark instances:

- **bay29**: 29 cities (optimal: 9,291.35)
- **att48**: 48 cities (optimal: 33,523.71)
- **ch150**: 150 cities (optimal: 6,532.28)
- **pres8**: 8 cities (simple test case)
- **rand8, rand128**: Random generated instances
- **grid4x4**: A 4x4 grid of points

Each dataset directory contains:
- `coords.txt`: City coordinates (x, y per line)
- `path.txt`: Optimal or reference tour (if known)
- `tour_len.txt`: Optimal or reference tour length (if known)

## Development

### Using uv for Development
```bash
# Install with all extras for development
uv sync --all-extras

# Add a new dependency
uv add <package-name>

# Add a CUDA-specific dependency
# Edit pyproject.toml to add to the appropriate [project.optional-dependencies] section
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
or
```bash
nvidia-smi
```

## License

This project is provided as-is for educational and research purposes.

## References

This implementation is based on the N-body approach to TSP solving as described in the accompanying paper. The method demonstrates how physical simulation can be applied to combinatorial optimization problems.


## Contact

For questions, issues, or contributions, please refer to the project repository or accompanying documentation.