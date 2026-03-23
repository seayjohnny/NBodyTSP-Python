"""
Data I/O Module for N-Body TSP Simulator

Handles loading, preprocessing, and normalization of TSP coordinate data.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class TSPDataLoader:
    """Loads and preprocesses TSP coordinate data from files."""
    
    def __init__(self, filepath: str | None = None, coords: np.ndarray | None = None):
        """
        Initialize the data loader.
        
        Args:
            filepath: Path to the coordinate file
            coords: Optional numpy array of coordinates
        """
        self.filepath = Path(filepath) if filepath is not None else None
        self.coords = coords
        self.original_coords = coords.copy() if coords is not None else None
        self.n_cities = len(coords) if coords is not None else 0
        self.geometric_center = None
        self.normalizing_factor = 1.0
        self.bounding_box = None
        
    def load_coordinates(self) -> np.ndarray:
        """
        Load city coordinates from file.
        
        Expected format: Each line contains "x y" (space or tab separated).
        
        Returns:
            numpy array of shape (n, 2) with coordinates
        """
        if self.filepath and not self.filepath.exists():
            raise FileNotFoundError(f"Coordinate file not found: {self.filepath}")
        
        coords_list = []
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        coords_list.append([x, y])
        
        if not coords_list:
            raise ValueError(f"No valid coordinates found in {self.filepath}")
        
        self.coords = np.array(coords_list, dtype=np.float32)
        self.original_coords = self.coords.copy()
        self.n_cities = len(self.coords)
        
        print(f"Loaded {self.n_cities} cities from {self.filepath.name}")
        return self.coords
    
    def center_at_origin(self) -> Tuple[float, float]:
        """
        Center coordinates at the geometric center (origin).
        
        Returns:
            Tuple of (center_x, center_y) that was subtracted
        """
        if self.coords is None:
            raise ValueError("Must load coordinates first")
        
        center = np.mean(self.coords, axis=0)
        self.coords -= center
        self.geometric_center = tuple(center)
        
        print(f"Geometric center: ({center[0]:.4f}, {center[1]:.4f})")
        return self.geometric_center
    
    def normalize_by_minimum_separation(self) -> float:
        """
        Normalize coordinates so minimum city separation is 1.0.
        
        Returns:
            The normalizing factor applied
        """
        if self.coords is None:
            raise ValueError("Must load coordinates first")
        
        # Calculate all pairwise distances
        n = len(self.coords)
        min_dist = float('inf')
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.coords[i, 0] - self.coords[j, 0]
                dy = self.coords[i, 1] - self.coords[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
        
        if min_dist < 1e-10:
            raise ValueError("Cities are too close together (minimum separation near zero)")
        
        # Normalize
        self.coords /= min_dist
        self.normalizing_factor = min_dist
        
        print(f"Normalizing factor (min separation): {min_dist:.6f}")
        return self.normalizing_factor
    
    def normalize_by_average_separation(self) -> float:
        """
        Normalize coordinates so average city separation is 1.0.
        
        Returns:
            The normalizing factor applied
        """
        if self.coords is None:
            raise ValueError("Must load coordinates first")
        
        # Calculate all pairwise distances
        n = len(self.coords)
        total_dist = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.coords[i, 0] - self.coords[j, 0]
                dy = self.coords[i, 1] - self.coords[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                total_dist += dist
                count += 1
        
        avg_dist = total_dist / count if count > 0 else 1.0
        
        if avg_dist < 1e-10:
            raise ValueError("Average separation near zero")
        
        # Normalize
        self.coords /= avg_dist
        self.normalizing_factor = avg_dist
        
        print(f"Normalizing factor (avg separation): {avg_dist:.6f}")
        return self.normalizing_factor
    
    def get_bounding_circle_radius(self) -> float:
        """
        Calculate the radius of the bounding circle (distance to farthest city).
        
        Returns:
            Radius of bounding circle
        """
        if self.coords is None:
            raise ValueError("Must load coordinates first")
        
        # Calculate distance from origin for each city
        distances = np.sqrt(np.sum(self.coords ** 2, axis=1))
        max_radius = np.max(distances)
        
        # print(f"Bounding circle radius: {max_radius:.4f}")
        return max_radius
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of all cities.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if self.coords is None:
            raise ValueError("Must load coordinates first")
        
        min_x = np.min(self.coords[:, 0])
        min_y = np.min(self.coords[:, 1])
        max_x = np.max(self.coords[:, 0])
        max_y = np.max(self.coords[:, 1])
        
        self.bounding_box = (min_x, min_y, max_x, max_y)
        return self.bounding_box
    
    def get_smallest_separation(self) -> float:
        """
        Get the minimum distance between any two cities.
        
        Returns:
            Minimum separation distance
        """
        if self.coords is None:
            raise ValueError("Must load coordinates first")
        
        n = len(self.coords)
        min_dist = float('inf')
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.coords[i, 0] - self.coords[j, 0]
                dy = self.coords[i, 1] - self.coords[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
        
        return min_dist
    
    def preprocess(self, normalize_method: str = 'minimum') -> dict:
        """
        Complete preprocessing pipeline: load, center, and normalize.
        
        Args:
            normalize_method: 'minimum' or 'average' separation
            
        Returns:
            Dictionary with preprocessing statistics
        """
        # Load data
        if self.coords is None and self.filepath is not None:
            self.load_coordinates()
        
        # Get initial bounding box
        bbox_before = self.get_bounding_box()
        radius_before = self.get_bounding_circle_radius()
        
        # Center at origin
        center = self.center_at_origin()
        
        # Normalize
        if normalize_method == 'minimum':
            norm_factor = self.normalize_by_minimum_separation()
        elif normalize_method == 'average':
            norm_factor = self.normalize_by_average_separation()
        else:
            raise ValueError(f"Unknown normalization method: {normalize_method}")
        
        # Get final statistics
        bbox_after = self.get_bounding_box()
        radius_after = self.get_bounding_circle_radius()
        min_sep = self.get_smallest_separation()
        
        stats = {
            'n_cities': self.n_cities,
            'geometric_center': center,
            'normalizing_factor': norm_factor,
            'bounding_box_before': bbox_before,
            'bounding_box_after': bbox_after,
            'radius_before': radius_before,
            'radius_after': radius_after,
            'min_separation': min_sep,
        }
        
        print(f"\nPreprocessing complete!")
        print(f"  Cities: {self.n_cities}")
        print(f"  Bounding radius: {radius_after:.4f}")
        print(f"  Min separation: {min_sep:.4f}")
        
        return stats


def load_optimal_path(filepath: str) -> Optional[np.ndarray]:
    """
    Load optimal path from file (if available).
    
    Args:
        filepath: Path to the optimal path file
        
    Returns:
        Array of city indices representing the optimal path, or None if not found
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Optimal path file not found: {filepath}")
        return None
    
    path = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Path file might have just indices or index pairs
                parts = line.split()
                if parts:
                    path.append(int(parts[0]))
    
    if path:
        return np.array(path, dtype=np.int32)
    return None


def load_optimal_cost(filepath: str) -> Optional[float]:
    """
    Load optimal tour length from file (if available).
    
    Args:
        filepath: Path to the optimal cost file
        
    Returns:
        Optimal tour cost, or None if not found
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Optimal cost file not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    return float(line)
                except ValueError:
                    continue
    
    return None


def generate_grid_dataset(
    n_rows: int,
    n_cols: int,
) -> None:
    """
    Generate a grid dataset of city coordinates.
    
    Args:
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        
    Returns:
        numpy array of shape (n_rows * n_cols, 2) with grid coordinates
    """
    coords = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j
            y = i
            coords.append([x, y])
    
    # The optimal cost for a grid is simply the number of cities
    optimal_cost = n_rows * n_cols

    # The optimal path can be a simple snake pattern
    optimal_path = []
    for i in range(n_rows):
        row_indices = list(range(i * n_cols, (i + 1) * n_cols))
        if i % 2 == 1:
            row_indices.reverse()
        optimal_path.extend(row_indices)

    # Write to files
    with open(f"grid_{n_rows}x{n_cols}_coords.txt", 'w') as f:
        for coord in coords:
            f.write(f"{coord[0]} {coord[1]}\n")

    with open(f"grid_{n_rows}x{n_cols}_tour_len.txt", 'w') as f:
        f.write(f"{optimal_cost}\n")


def generate_random_dataset(
    n_cities: int,
    x_range: Tuple[float, float] = (0.0, 100.0),
    y_range: Tuple[float, float] = (0.0, 100.0),
) -> np.ndarray:
    """
    Generate a random dataset of city coordinates.
    
    Args:
        n_cities: Number of cities to generate
        x_range: Tuple specifying the (min, max) range for x-coordinates
        y_range: Tuple specifying the (min, max) range for y-coordinates
        
    Returns:
        numpy array of shape (n_cities, 2) with random coordinates
    """
    x_coords = np.random.uniform(x_range[0], x_range[1], n_cities)
    y_coords = np.random.uniform(y_range[0], y_range[1], n_cities)
    
    coords = np.column_stack((x_coords, y_coords))
    
    # Write to file
    with open(f"random_{n_cities}_coords.txt", 'w') as f:
        for coord in coords:
            f.write(f"{coord[0]} {coord[1]}\n")
    
    return coords


def discover_datasets(datasets_dir: str = "datasets") -> list[dict]:
    """
    Scan a directory for TSP dataset subdirectories containing coords.txt.

    Returns:
        List of dicts with keys: name, path, n_cities, has_optimal
    """
    results = []
    datasets_path = Path(datasets_dir)
    if not datasets_path.exists():
        return results

    for d in sorted(datasets_path.iterdir()):
        if not d.is_dir():
            continue
        coords_file = d / "coords.txt"
        if not coords_file.exists():
            continue

        # Count cities
        n = 0
        try:
            with open(coords_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        n += 1
        except Exception:
            continue

        has_optimal = (d / "tour_len.txt").exists()

        results.append({
            "name": d.name,
            "path": str(coords_file),
            "n_cities": n,
            "has_optimal": has_optimal,
        })

    return results


# Example usage
if __name__ == "__main__":
    # Test with a dataset
    loader = TSPDataLoader("datasets/att48/coords.txt")
    stats = loader.preprocess(normalize_method='minimum')
    
    print(f"\nCoordinates shape: {loader.coords.shape}")
    print(f"First 5 cities:")
    print(loader.coords[:5])
    
    # Try loading optimal path
    opt_path = load_optimal_path("datasets/att48/path.txt")
    if opt_path is not None:
        print(f"\nOptimal path length: {len(opt_path)}")
    
    opt_cost = load_optimal_cost("datasets/att48/tour_len.txt")
    if opt_cost is not None:
        print(f"Optimal tour cost: {opt_cost}")
