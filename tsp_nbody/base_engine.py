"""Abstract base class for N-Body TSP physics engines."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BasePhysicsEngine(ABC):
    """Uniform interface for all N-Body TSP physics engines.

    Engines manage their own collapse state internally. Callers use
    start_collapse() to begin and check is_complete to detect termination.
    """

    n_cities: int
    original_coords: np.ndarray

    @abstractmethod
    def initialize_physics(self) -> None:
        """Reset particle state to initial configuration (phase -> READY)."""
        ...

    @abstractmethod
    def start_collapse(self) -> None:
        """Begin the wall/tube collapse process (phase -> COLLAPSING)."""
        ...

    @abstractmethod
    def integrate_step(self) -> None:
        """Perform one physics timestep.

        Handles collapse progression internally — no external arguments.
        """
        ...

    def run_substeps(self, k: int) -> None:
        """Run k physics substeps."""
        for _ in range(k):
            self.integrate_step()

    @abstractmethod
    def get_positions_cpu(self) -> np.ndarray:
        """Current positions as (N, ndim) float32 numpy array on CPU."""
        ...

    @abstractmethod
    def get_velocities_cpu(self) -> np.ndarray:
        """Current velocities as (N, ndim) float32 numpy array on CPU."""
        ...

    @abstractmethod
    def get_final_tour(self) -> np.ndarray:
        """Return 0-indexed tour from current particle positions."""
        ...

    def get_found_tour(self) -> Optional[list[int]]:
        """Return 0-indexed tour extracted during collapse, or None."""
        return None

    @property
    @abstractmethod
    def phase(self) -> str:
        """Current simulation phase: 'READY', 'COLLAPSING', or terminal."""
        ...

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        """True when simulation has reached its terminal phase."""
        ...

    @property
    def ndim(self) -> int:
        """Spatial dimensionality (2 or 3)."""
        return 2
