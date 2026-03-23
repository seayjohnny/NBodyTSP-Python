"""
Local Search: 2-opt & 3-opt post-processing for TSP tours.

Applied after physics produces a tour to improve it via
classical edge-swap local search.
  2-opt: O(N^2) per pass - very fast
  3-opt: O(N^3) per pass - still fast for N < 150
"""

import numpy as np
from typing import Callable, Optional


def tour_distance(tour: list[int], dist_fn: Callable[[int, int], float]) -> float:
    """Calculate total tour distance using 1-indexed city IDs."""
    total = 0.0
    n = len(tour)
    for k in range(n):
        total += dist_fn(tour[k] - 1, tour[(k + 1) % n] - 1)
    return total


def two_opt(tour: list[int], dist_fn: Callable[[int, int], float]) -> tuple[list[int], float]:
    """
    2-opt local search: repeatedly reverse segments to reduce tour cost.

    Args:
        tour: List of 1-indexed city IDs
        dist_fn: Function(i, j) returning distance between 0-indexed cities

    Returns:
        (improved_tour, distance)
    """
    tour = list(tour)
    n = len(tour)
    best_dist = tour_distance(tour, dist_fn)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue

                a = tour[i] - 1
                b = tour[i + 1] - 1
                c = tour[j] - 1
                d = tour[(j + 1) % n] - 1

                old_cost = dist_fn(a, b) + dist_fn(c, d)
                new_cost = dist_fn(a, c) + dist_fn(b, d)

                if new_cost < old_cost:
                    # Reverse segment [i+1..j]
                    lo, hi = i + 1, j
                    while lo < hi:
                        tour[lo], tour[hi] = tour[hi], tour[lo]
                        lo += 1
                        hi -= 1
                    best_dist = tour_distance(tour, dist_fn)
                    improved = True

    return tour, best_dist


def three_opt(tour: list[int], dist_fn: Callable[[int, int], float]) -> tuple[list[int], float]:
    """
    3-opt local search: try all triple edge removals with reconnections.

    Args:
        tour: List of 1-indexed city IDs
        dist_fn: Function(i, j) returning distance between 0-indexed cities

    Returns:
        (improved_tour, distance)
    """
    tour = list(tour)
    n = len(tour)
    best_dist = tour_distance(tour, dist_fn)
    improved = True

    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                for k in range(j + 2, n + (0 if i > 0 else -1)):
                    seg_a = tour[:i + 1]
                    seg_b = tour[i + 1:j + 1]
                    seg_c = tour[j + 1:k + 1]
                    seg_d = tour[k + 1:] if k + 1 < n else []

                    seg_b_r = seg_b[::-1]
                    seg_c_r = seg_c[::-1]

                    candidates = [
                        seg_a + seg_b_r + seg_c + seg_d,
                        seg_a + seg_b + seg_c_r + seg_d,
                        seg_a + seg_b_r + seg_c_r + seg_d,
                        seg_a + seg_c + seg_b + seg_d,
                        seg_a + seg_c_r + seg_b + seg_d,
                        seg_a + seg_c + seg_b_r + seg_d,
                        seg_a + seg_c_r + seg_b_r + seg_d,
                    ]

                    best_candidate = None
                    best_candidate_dist = best_dist

                    for cand in candidates:
                        d = tour_distance(cand, dist_fn)
                        if d < best_candidate_dist:
                            best_candidate_dist = d
                            best_candidate = cand

                    if best_candidate is not None and best_candidate_dist < best_dist:
                        tour = best_candidate
                        best_dist = best_candidate_dist
                        improved = True

    return tour, best_dist


def improve(
    tour: list[int],
    dist_fn: Callable[[int, int], float],
    mode: str = "2-opt",
) -> dict:
    """
    Apply local search to improve a tour.

    Args:
        tour: List of 1-indexed city IDs
        dist_fn: Function(i, j) returning distance between 0-indexed cities
        mode: "2-opt", "3-opt", or "both"

    Returns:
        dict with keys: tour, distance, before, after, saved, pct_improved, method
    """
    if not tour or len(tour) < 4:
        return {
            "tour": tour,
            "distance": tour_distance(tour, dist_fn) if tour else 0,
            "before": 0, "after": 0, "saved": 0,
            "pct_improved": 0, "method": "none",
        }

    before_dist = tour_distance(tour, dist_fn)
    working_tour = list(tour)
    method = ""

    if mode in ("2-opt", "both"):
        working_tour, _ = two_opt(working_tour, dist_fn)
        method = "2-opt"

    if mode in ("3-opt", "both"):
        working_tour, _ = three_opt(working_tour, dist_fn)
        method = "2-opt + 3-opt" if mode == "both" else "3-opt"

    after_dist = tour_distance(working_tour, dist_fn)

    return {
        "tour": working_tour,
        "distance": after_dist,
        "before": before_dist,
        "after": after_dist,
        "saved": before_dist - after_dist,
        "pct_improved": (before_dist - after_dist) / before_dist * 100 if before_dist > 0 else 0,
        "method": method,
    }


def make_euclidean_dist_fn(coords: np.ndarray) -> Callable[[int, int], float]:
    """Create a Euclidean distance function from coordinates array."""
    def dist_fn(i: int, j: int) -> float:
        diff = coords[i] - coords[j]
        return float(np.sqrt(np.sum(diff * diff)))
    return dist_fn
