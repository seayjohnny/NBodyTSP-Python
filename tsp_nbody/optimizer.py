"""
Bayesian Optimization for Torus TSP Parameters.

Searches a 4D parameter space (shrinkRate, ljStrength, perturbation, epsilon)
to minimize tour distance. Uses Latin Hypercube initial sampling followed by
Gaussian Process surrogate with Expected Improvement acquisition.

Ported from the JavaScript NBodyTSP-Torus optimizer.js.
"""

import math
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field


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
    param_names: list[str] = field(default_factory=lambda: ['shrink_rate', 'lj_strength', 'perturbation', 'epsilon'])


class BayesianOptimizer:
    """Bayesian optimization for torus collapse parameters."""

    # Parameter space: [shrinkRate, ljStrength, perturbation, epsilon]
    PARAM_BOUNDS = [
        (0.02, 0.45),   # shrink_rate
        (0.2, 4.5),     # lj_strength
        (0.0, 1.0),     # perturbation
        (0.02, 0.25),   # epsilon
    ]
    PARAM_NAMES = ['shrink_rate', 'lj_strength', 'perturbation', 'epsilon']

    def __init__(
        self,
        max_trials: int = 60,
        n_initial: int = 15,
        n_candidates: int = 200,
        seed: int = 42,
        verbose: bool = True,
    ):
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

    def optimize(
        self,
        run_trial: Callable[[dict], tuple[float, Optional[list[int]]]],
        on_trial_complete: Optional[Callable[[int, OptTrial, float], None]] = None,
    ) -> OptimizerResult:
        """
        Run the full optimization loop.

        Args:
            run_trial: Function that takes a dict of parameters
                       {shrink_rate, lj_strength, perturbation, epsilon, seed}
                       and returns (tour_distance, tour_or_None)
            on_trial_complete: Optional callback(trial_index, trial, best_dist_so_far)

        Returns:
            OptimizerResult with best parameters, distance, and all trials
        """
        self.trials = []
        self.best_dist = float('inf')
        self.best_params = None
        self.best_tour = None

        if self.verbose:
            print("\n" + "=" * 60)
            print("BAYESIAN OPTIMIZATION")
            print(f"  Max trials: {self.max_trials}")
            print(f"  Initial samples: {self.n_initial}")
            print(f"  Bayesian candidates: {self.n_candidates}")
            print("=" * 60)

        for trial_idx in range(self.max_trials):
            # Pick parameters
            if trial_idx < self.n_initial:
                params = self._sample_initial(trial_idx)
                method = "LHS"
            else:
                params = self._sample_bayesian()
                method = "GP-EI"

            # Build trial config
            trial_config = {
                'shrink_rate': params[0],
                'lj_strength': params[1],
                'perturbation': params[2],
                'epsilon': params[3],
                'seed': self.seed + trial_idx * 7,
            }

            # Run trial
            distance, tour = run_trial(trial_config)
            trial = OptTrial(params=params, distance=distance)
            self.trials.append(trial)

            if distance < self.best_dist:
                self.best_dist = distance
                self.best_params = list(params)
                self.best_tour = list(tour) if tour else None

            if self.verbose:
                gap_str = ""
                print(
                    f"  Trial {trial_idx + 1:3d}/{self.max_trials} [{method:5s}] "
                    f"dist={distance:,.0f}  best={self.best_dist:,.0f}  "
                    f"S:{params[0]:.2f} LJ:{params[1]:.1f} P:{params[2]:.2f} E:{params[3]:.2f}"
                )

            if on_trial_complete:
                on_trial_complete(trial_idx, trial, self.best_dist)

        if self.verbose:
            print("\n" + "-" * 60)
            print(f"BEST: dist={self.best_dist:,.0f}")
            if self.best_params:
                print(
                    f"  shrink_rate={self.best_params[0]:.3f}  "
                    f"lj_strength={self.best_params[1]:.3f}  "
                    f"perturbation={self.best_params[2]:.3f}  "
                    f"epsilon={self.best_params[3]:.3f}"
                )
            print("=" * 60 + "\n")

        return OptimizerResult(
            best_params=self.best_params or [0] * 4,
            best_distance=self.best_dist,
            best_tour=self.best_tour,
            trials=self.trials,
        )


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
