"""
N-Body TSP Simulator — Main Entry Point

Launch the interactive application with setup screen.
For CLI usage, run: python -m tsp_nbody.simulator [dataset] [--torus] [--optimize]
"""

import sys

def main():
    from tsp_nbody.app import App
    app = App()
    return app.run()

if __name__ == "__main__":
    sys.exit(main() or 0)
