"""
Path Extraction Module for N-Body TSP Simulator

Extracts TSP paths from ring configurations using angular sorting.
"""

import numpy as np
from typing import Tuple, List


class PathExtractor:
    """Extracts and analyzes TSP paths from N-body simulation results."""
    
    def __init__(self):
        """Initialize the path extractor."""
        pass
    
    def extract_path_from_ring(self, positions: np.ndarray) -> np.ndarray:
        """
        Extract TSP path by sorting cities by polar angle.
        
        The N-body simulation pushes cities into a ring. We extract the path
        by visiting cities in angular order around the ring.
        
        Args:
            positions: City positions array of shape (n, 2)
            
        Returns:
            Array of city indices representing the path order
        """
        n = len(positions)
        
        # Calculate polar angle for each city
        angles = np.arctan2(positions[:, 1], positions[:, 0])
        
        # Sort cities by angle
        path = np.argsort(angles)
        
        return path.astype(np.int32)
    
    def calculate_path_cost(self, coords: np.ndarray, path: np.ndarray, 
                           closed: bool = True) -> float:
        """
        Calculate the total cost (length) of a TSP path.
        
        Args:
            coords: Original city coordinates (n, 2)
            path: Array of city indices in visit order
            closed: Whether the path returns to start (closed tour)
            
        Returns:
            Total path length
        """
        n = len(path)
        total_cost = 0.0
        
        # Sum distances between consecutive cities in path
        for i in range(n - 1):
            city_a = path[i]
            city_b = path[i + 1]
            
            dx = coords[city_a, 0] - coords[city_b, 0]
            dy = coords[city_a, 1] - coords[city_b, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            
            total_cost += dist
        
        # Add return to start if closed tour
        if closed and n > 0:
            city_a = path[-1]
            city_b = path[0]
            
            dx = coords[city_a, 0] - coords[city_b, 0]
            dy = coords[city_a, 1] - coords[city_b, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            
            total_cost += dist
        
        return total_cost
    
    def validate_path(self, path: np.ndarray, n_cities: int) -> bool:
        """
        Validate that a path is a valid TSP tour.
        
        Args:
            path: Array of city indices
            n_cities: Expected number of cities
            
        Returns:
            True if valid, False otherwise
        """
        # Check length
        if len(path) != n_cities:
            print(f"Invalid path length: {len(path)} != {n_cities}")
            return False
        
        # Check for valid indices
        if np.any(path < 0) or np.any(path >= n_cities):
            print("Path contains out-of-range indices")
            return False
        
        # Check for duplicates
        if len(np.unique(path)) != n_cities:
            print("Path contains duplicate cities")
            return False
        
        return True
    
    def calculate_path_statistics(self, coords: np.ndarray, path: np.ndarray) -> dict:
        """
        Calculate detailed statistics about a path.
        
        Args:
            coords: Original city coordinates (n, 2)
            path: Array of city indices in visit order
            
        Returns:
            Dictionary with path statistics
        """
        n = len(path)
        
        # Calculate edge lengths
        edge_lengths = []
        for i in range(n):
            city_a = path[i]
            city_b = path[(i + 1) % n]  # Wrap around to start
            
            dx = coords[city_a, 0] - coords[city_b, 0]
            dy = coords[city_a, 1] - coords[city_b, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            edge_lengths.append(dist)
        
        edge_lengths = np.array(edge_lengths)
        
        stats = {
            'total_cost': np.sum(edge_lengths),
            'mean_edge_length': np.mean(edge_lengths),
            'std_edge_length': np.std(edge_lengths),
            'min_edge_length': np.min(edge_lengths),
            'max_edge_length': np.max(edge_lengths),
            'n_edges': len(edge_lengths),
        }
        
        return stats
    
    def compare_with_optimal(self, nbody_cost: float, optimal_cost: float) -> dict:
        """
        Compare N-body solution with optimal solution.
        
        Args:
            nbody_cost: Cost of N-body path
            optimal_cost: Known optimal cost
            
        Returns:
            Dictionary with comparison metrics
        """
        if optimal_cost <= 0:
            raise ValueError("Optimal cost must be positive")
        
        absolute_diff = nbody_cost - optimal_cost
        percent_diff = 100.0 * absolute_diff / optimal_cost
        quality_ratio = nbody_cost / optimal_cost
        
        comparison = {
            'nbody_cost': nbody_cost,
            'optimal_cost': optimal_cost,
            'absolute_difference': absolute_diff,
            'percent_difference': percent_diff,
            'quality_ratio': quality_ratio,
            'is_better': nbody_cost < optimal_cost,
        }
        
        return comparison
    
    def compare_paths(self, coords: np.ndarray, path_a: np.ndarray, 
                     path_b: np.ndarray, labels: Tuple[str, str] = ('Path A', 'Path B')) -> dict:
        """
        Compare two TSP paths.
        
        Args:
            coords: Original city coordinates
            path_a: First path
            path_b: Second path
            labels: Names for the two paths
            
        Returns:
            Dictionary with comparison results
        """
        cost_a = self.calculate_path_cost(coords, path_a)
        cost_b = self.calculate_path_cost(coords, path_b)
        
        stats_a = self.calculate_path_statistics(coords, path_a)
        stats_b = self.calculate_path_statistics(coords, path_b)
        
        comparison = {
            labels[0]: {
                'cost': cost_a,
                'stats': stats_a,
            },
            labels[1]: {
                'cost': cost_b,
                'stats': stats_b,
            },
            'cost_difference': cost_a - cost_b,
            'percent_difference': 100.0 * (cost_a - cost_b) / cost_b if cost_b > 0 else 0.0,
            'better_path': labels[0] if cost_a < cost_b else labels[1],
        }
        
        return comparison
    
    def print_path(self, path: np.ndarray, max_cities: int = 20):
        """
        Print path in a readable format.
        
        Args:
            path: Array of city indices
            max_cities: Maximum number of cities to print
        """
        if len(path) <= max_cities:
            path_str = ' -> '.join(map(str, path)) + f' -> {path[0]}'
            print(f"Path: {path_str}")
        else:
            start = ' -> '.join(map(str, path[:10]))
            end = ' -> '.join(map(str, path[-10:]))
            print(f"Path: {start} ... {end} -> {path[0]}")
    
    def analyze_path_quality(self, coords: np.ndarray, path: np.ndarray, 
                           optimal_cost: float = None) -> dict:
        """
        Comprehensive path quality analysis.
        
        Args:
            coords: Original city coordinates
            path: TSP path to analyze
            optimal_cost: Optional known optimal cost for comparison
            
        Returns:
            Dictionary with complete analysis
        """
        # Basic validation
        valid = self.validate_path(path, len(coords))
        
        # Calculate statistics
        stats = self.calculate_path_statistics(coords, path)
        cost = stats['total_cost']
        
        analysis = {
            'valid': valid,
            'cost': cost,
            'statistics': stats,
        }
        
        # Compare with optimal if available
        if optimal_cost is not None and optimal_cost > 0:
            comparison = self.compare_with_optimal(cost, optimal_cost)
            analysis['optimal_comparison'] = comparison
        
        return analysis


def nearest_neighbor_tsp(coords: np.ndarray, start_city: int = 0) -> np.ndarray:
    """
    Simple nearest neighbor heuristic for TSP (for comparison).
    
    Args:
        coords: City coordinates (n, 2)
        start_city: Index of starting city
        
    Returns:
        Path as array of city indices
    """
    n = len(coords)
    unvisited = set(range(n))
    path = [start_city]
    unvisited.remove(start_city)
    
    current = start_city
    
    while unvisited:
        # Find nearest unvisited city
        nearest = None
        nearest_dist = float('inf')
        
        for city in unvisited:
            dx = coords[city, 0] - coords[current, 0]
            dy = coords[city, 1] - coords[current, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = city
        
        path.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return np.array(path, dtype=np.int32)


# Example usage
if __name__ == "__main__":
    # Create test data
    np.random.seed(42)
    n_cities = 20
    
    # Random cities in a rough circle (simulating post-simulation positions)
    angles = np.linspace(0, 2*np.pi, n_cities, endpoint=False)
    noise = np.random.randn(n_cities) * 0.1
    positions = np.column_stack([
        np.cos(angles + noise),
        np.sin(angles + noise)
    ])
    
    # Original coordinates (different from ring positions)
    coords = np.random.randn(n_cities, 2) * 2.0
    
    # Extract path
    extractor = PathExtractor()
    path = extractor.extract_path_from_ring(positions)
    
    print("Extracted path:")
    extractor.print_path(path)
    
    # Calculate cost
    cost = extractor.calculate_path_cost(coords, path)
    print(f"\nPath cost: {cost:.4f}")
    
    # Path statistics
    stats = extractor.calculate_path_statistics(coords, path)
    print("\nPath statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Compare with nearest neighbor
    nn_path = nearest_neighbor_tsp(coords)
    nn_cost = extractor.calculate_path_cost(coords, nn_path)
    
    print(f"\nNearest neighbor cost: {nn_cost:.4f}")
    print(f"N-body vs NN: {((cost - nn_cost) / nn_cost * 100):.2f}% difference")
    
    # Full analysis
    analysis = extractor.analyze_path_quality(coords, path, optimal_cost=nn_cost * 0.9)
    print("\nPath analysis:")
    print(f"  Valid: {analysis['valid']}")
    if 'optimal_comparison' in analysis:
        comp = analysis['optimal_comparison']
        print(f"  Percent from optimal: {comp['percent_difference']:.2f}%")