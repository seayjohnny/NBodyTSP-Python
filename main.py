"""
N-Body TSP Simulator — Main Entry Point

Launches the web application server.
"""

import sys


def main():
    from tsp_nbody.server import main as serve
    serve()


if __name__ == "__main__":
    sys.exit(main() or 0)
