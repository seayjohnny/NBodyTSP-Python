"""
Bayesian Optimization for N-Body TSP Parameters.

Engine-agnostic optimizer that searches a configurable parameter space
to minimize tour distance. Uses Latin Hypercube initial sampling followed by
Gaussian Process surrogate with Expected Improvement acquisition.

Ported from the JavaScript NBodyTSP-Torus optimizer.js.
"""

import math
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Preset parameter bounds for each geometry
# ---------------------------------------------------------------------------

TORUS_PARAM_BOUNDS = [
    (0.02, 0.45),   # collapse_rate
    (0.2, 4.5),     # lj_strength
    (0.0, 1.0),     # perturbation
    (0.02, 0.25),   # epsilon
]
TORUS_PARAM_NAMES = ['collapse_rate', 'lj_strength', 'perturbation', 'epsilon']

ANNULAR_PARAM_BOUNDS = [
    (0.02, 0.45),   # collapse_rate
    (0.2, 4.5),     # lj_strength
    (0.02, 0.25),   # epsilon
]
ANNULAR_PARAM_NAMES = ['collapse_rate', 'lj_strength', 'epsilon']

PLANAR_PARAM_BOUNDS = [
    (0.001, 0.05),  # collapse_rate
    (0.1, 50.0),    # lj_strength
    (0.5, 1.0),     # damping
]
PLANAR_PARAM_NAMES = ['collapse_rate', 'lj_strength', 'damping']


@dataclass
class OptTrial:
    params: list[float]
    distance: float


@dataclass
class OptimizerResult:
    best_params: list[float]
    best_distance: float
    best_tour: Optional[list[int]]
    trials: list[OptTrial]
    param_names: list[str] = field(default_factory=list)


class BayesianOptimizer:
    """Bayesian optimization for N-body collapse parameters."""

    def __init__(
        self,
        param_bounds: list[tuple[float, float]],
        param_names: list[str],
        max_trials: int = 60,
        n_initial: int = 15,
        n_candidates: int = 200,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.PARAM_BOUNDS = param_bounds
        self.PARAM_NAMES = param_names
        self.max_trials = max_trials
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.seed = seed
        self.verbose = verbose

        self.trials: list[OptTrial] = []
        self.best_dist = float('inf')
        self.best_params: Optional[list[float]] = None
        self.best_tour: Optional[list[int]] = None

        self._rng = np.random.RandomState(seed)

    def _sample_initial(self, idx: int) -> list[float]:
        """Latin Hypercube-style stratified random sampling."""
        params = []
        for d in range(len(self.PARAM_BOUNDS)):
            lo, hi = self.PARAM_BOUNDS[d]
            strata = self.n_initial
            band = (hi - lo) / strata
            strat_idx = (idx + d * 3) % strata
            params.append(lo + band * (strat_idx + self._rng.random()))
        return params

    def _sample_bayesian(self) -> list[float]:
        """GP surrogate + Expected Improvement acquisition."""
        X = [t.params for t in self.trials]
        Y = [t.distance for t in self.trials]
        n = len(X)
        D = len(self.PARAM_BOUNDS)

        # Normalize observations
        y_mean = sum(Y) / n
        y_std = math.sqrt(sum((y - y_mean) ** 2 for y in Y) / n) or 1.0
        Yn = [(y - y_mean) / y_std for y in Y]
        f_best = min(Yn)

        # Length scales per dimension (30% of range)
        ls = [(hi - lo) * 0.3 for lo, hi in self.PARAM_BOUNDS]

        def kernel(a, b):
            sq = sum(((a[d] - b[d]) / ls[d]) ** 2 for d in range(D))
            return math.exp(-0.5 * sq)

        # K matrix + regularization
        K = [[kernel(X[i], X[j]) + (0.01 if i == j else 0.0) for j in range(n)] for i in range(n)]

        # Cholesky decomposition
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = K[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                L[i][j] = math.sqrt(max(s, 1e-10)) if i == j else s / L[j][j]

        # Solve L * alpha_temp = Yn
        alpha_temp = [0.0] * n
        for i in range(n):
            s = Yn[i]
            for j in range(i):
                s -= L[i][j] * alpha_temp[j]
            alpha_temp[i] = s / L[i][i]

        # Solve L^T * alpha = alpha_temp
        alpha = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = alpha_temp[i]
            for j in range(i + 1, n):
                s -= L[j][i] * alpha[j]
            alpha[i] = s / L[i][i]

        best_ei = -float('inf')
        best_candidate = None

        for _ in range(self.n_candidates):
            candidate = [lo + self._rng.random() * (hi - lo) for lo, hi in self.PARAM_BOUNDS]

            # GP predict mean
            kstar = [kernel(candidate, X[i]) for i in range(n)]
            mu = sum(kstar[i] * alpha[i] for i in range(n))

            # Solve L * v = kstar for variance
            v = [0.0] * n
            for i in range(n):
                s = kstar[i]
                for j in range(i):
                    s -= L[i][j] * v[j]
                v[i] = s / L[i][i]

            var = kernel(candidate, candidate) - sum(vi * vi for vi in v)
            var = max(var, 1e-8)
            sigma = math.sqrt(var)

            # Expected Improvement
            z = (f_best - mu) / sigma
            ei = (f_best - mu) * _normcdf(z) + sigma * _normpdf(z)

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        return best_candidate

    def suggest_params(self) -> list[float]:
        """Return the next parameter set to evaluate."""
        idx = len(self.trials)
        if idx < self.n_initial:
            return self._sample_initial(idx)
        return self._sample_bayesian()

    def report_result(
        self,
        params: list[float],
        distance: float,
        tour: Optional[list[int]] = None,
    ) -> dict:
        """Record a completed trial and return a summary dict."""
        trial = OptTrial(params=params, distance=distance)
        self.trials.append(trial)

        if distance < self.best_dist:
            self.best_dist = distance
            self.best_params = list(params)
            self.best_tour = list(tour) if tour else None

        param_dict = {name: val for name, val in zip(self.PARAM_NAMES, params)}
        best_param_dict = {
            name: val for name, val in zip(self.PARAM_NAMES, self.best_params)
        } if self.best_params else {}

        return {
            "trial": len(self.trials),
            "max_trials": self.max_trials,
            "params": param_dict,
            "distance": distance,
            "best_distance": self.best_dist,
            "best_params": best_param_dict,
        }

    @property
    def is_complete(self) -> bool:
        return len(self.trials) >= self.max_trials

    def get_result(self) -> OptimizerResult:
        return OptimizerResult(
            best_params=self.best_params or [0.0] * len(self.PARAM_BOUNDS),
            best_distance=self.best_dist,
            best_tour=self.best_tour,
            trials=self.trials,
            param_names=list(self.PARAM_NAMES),
        )

    def optimize(self, run_trial, on_trial_complete=None):
        self.trials = []
        self.best_dist = float('inf')
        self.best_params = None
        self.best_tour = None

        if self.verbose:
            print("\n" + "=" * 60)
            print("BAYESIAN OPTIMIZATION")
            print(f"  Max trials: {self.max_trials}")
            print(f"  Initial samples: {self.n_initial}")
            print("=" * 60)

        while not self.is_complete:
            params = self.suggest_params()
            trial_config = {name: val for name, val in zip(self.PARAM_NAMES, params)}
            trial_config['seed'] = self.seed + len(self.trials) * 7

            distance, tour = run_trial(trial_config)
            method = "LHS" if len(self.trials) < self.n_initial else "GP-EI"
            self.report_result(params, distance, tour)

            if self.verbose:
                param_str = "  ".join(f"{name}:{val:.3f}" for name, val in zip(self.PARAM_NAMES, params))
                print(f"  Trial {len(self.trials):3d}/{self.max_trials} [{method:5s}] dist={distance:,.0f}  best={self.best_dist:,.0f}  {param_str}")

            if on_trial_complete:
                on_trial_complete(len(self.trials) - 1, self.trials[-1], self.best_dist)

        if self.verbose:
            print(f"\nBEST: dist={self.best_dist:,.0f}")
            if self.best_params:
                ps = "  ".join(f"{n}={v:.3f}" for n, v in zip(self.PARAM_NAMES, self.best_params))
                print(f"  {ps}")
            print("=" * 60 + "\n")

        return self.get_result()


def make_engine_objective(
    coords: np.ndarray,
    geometry: str = "torus",
    force_model: str = "true_lj",
    wall_model: str = "inverse_square",
    backend: str = "gpu",
    base_options: dict = None,
    param_names: list[str] = None,
    substeps_per_batch: int = 4,
    max_batches: int = 10000,
) -> Callable:
    """Create an objective function that runs a headless simulation to completion.

    Returns a callable: trial_params (list[float]) -> (distance, tour_0indexed)
    """
    from tsp_nbody.engine import PhysicsEngine
    from tsp_nbody.local_search import make_euclidean_dist_fn, tour_distance

    def objective(params: list[float]) -> tuple[float, list[int]]:
        opts = dict(base_options or {})
        for name, val in zip(param_names, params):
            opts[name] = val

        engine = PhysicsEngine(
            coords=coords,
            geometry=geometry,
            force_model=force_model,
            wall_model=wall_model,
            backend=backend,
            options=opts,
        )
        engine.initialize_physics()
        engine.start_collapse()

        for _ in range(max_batches):
            engine.run_substeps(substeps_per_batch)
            if engine.is_complete:
                break

        tour = engine.get_found_tour()
        if tour is None:
            tour = engine.get_final_tour().tolist()

        # tour is 0-indexed; distance computation needs 1-indexed
        tour_1indexed = [i + 1 for i in tour]
        dist_fn = make_euclidean_dist_fn(coords)
        distance = tour_distance(tour_1indexed, dist_fn)

        return distance, tour

    return objective


def _normcdf(x: float) -> float:
    """Approximation to normal CDF."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327
    p = d * math.exp(-x * x / 2) * (
        t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    )
    return 1.0 - p if x > 0 else p


def _normpdf(x: float) -> float:
    """Normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
