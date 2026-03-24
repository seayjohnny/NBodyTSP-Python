"""
SimulationRunner — manages PhysicsEngine lifecycle, tour extraction,
and metadata generation for the web UI.

Extracted from the old SimulationState class in server.py.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from tsp_nbody.engine import PhysicsEngine
from tsp_nbody.dataio import discover_datasets, TSPDataLoader, load_optimal_cost
from tsp_nbody.optimizer import BayesianOptimizer, make_engine_objective
from tsp_nbody.local_search import (
    improve as local_search_improve,
    make_euclidean_dist_fn,
    tour_distance,
)
from tsp_nbody.path_extraction import PathExtractor, random_nearest_neighbor_tsp


# ---------------------------------------------------------------------------
# Helper: sensible defaults for 2D / planar mode
# ---------------------------------------------------------------------------

def _get_2d_defaults(force_mode: str, wall_mode: str) -> dict:
    """Return sensible default physics parameters for planar geometry."""
    if force_mode == "true_lj":
        if wall_mode == "inverse_square":
            return dict(
                lj_strength=0.3, wall_strength=8.0, damp=4.0, mass=1.0,
                dt=0.003, dr=0.003, force_cutoff=1000.0,
                slope_repulsion=50.0, mag_attraction=0.5, force_cutoff_extra=0.1,
            )
        else:
            return dict(
                lj_strength=0.3, wall_strength=500.0, damp=4.0, mass=1.0,
                dt=0.003, dr=0.003, force_cutoff=1000.0,
                slope_repulsion=50.0, mag_attraction=0.5, force_cutoff_extra=0.1,
            )
    elif force_mode == "smooth":
        if wall_mode == "inverse_square":
            return dict(
                lj_strength=1.0, wall_strength=500.0, damp=10.0, mass=80.0,
                dt=0.01, dr=0.01, force_cutoff=100000.0,
                slope_repulsion=50.0, mag_attraction=0.5, force_cutoff_extra=0.1,
            )
        else:
            return dict(
                lj_strength=1.0, wall_strength=20000.0, damp=20.0, mass=80.0,
                dt=0.01, dr=0.01, force_cutoff=100000.0,
                slope_repulsion=50.0, mag_attraction=0.5, force_cutoff_extra=0.1,
            )
    else:  # piecewise
        if wall_mode == "inverse_square":
            return dict(
                lj_strength=1.0, wall_strength=500.0, damp=10.0, mass=80.0,
                dt=0.01, dr=0.01, force_cutoff=100000.0,
                slope_repulsion=50.0, mag_attraction=0.5, force_cutoff_extra=100.0,
            )
        else:
            return dict(
                lj_strength=1.0, wall_strength=20000.0, damp=20.0, mass=80.0,
                dt=0.01, dr=0.01, force_cutoff=100000.0,
                slope_repulsion=50.0, mag_attraction=25.0, force_cutoff_extra=100.0,
            )


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """Manages a PhysicsEngine instance, handles lifecycle, tour extraction,
    and metadata generation for the web UI."""

    def __init__(self):
        self.engine: Optional[PhysicsEngine] = None
        self.mode: str = "torus"  # geometry name
        self.config: dict = {}
        self.original_coords: Optional[np.ndarray] = None
        self.n_cities: int = 0
        self.optimal_cost: Optional[float] = None
        self.substeps: int = 4
        self.running: bool = False
        self.data_loader: Optional[TSPDataLoader] = None

        # Tour results
        self.final_tour: Optional[list[int]] = None  # 0-indexed internally
        self.final_cost: Optional[float] = None
        self.raw_cost: Optional[float] = None
        self.ls_result: Optional[dict] = None
        self.nn_cost: Optional[float] = None
        self.use_local_search: bool = False
        self.local_search_mode: str = "2-opt"
        self._tour_extracted: bool = False

        # Optimization state
        self.optimizing: bool = False
        self.opt_mode: str = "fast"  # "fast" or "visual"
        self._optimizer: Optional["BayesianOptimizer"] = None
        self._opt_param_names: list[str] = []
        self._opt_visual_steps: int = 0
        self._opt_visual_max_steps: int = 50000
        self._current_opt_params: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_simulation(self, config: dict):
        """Create a PhysicsEngine from *config* and initialise physics.

        Config keys
        -----------
        dataset_path : str
            Path to the coordinates file.
        mode : str
            ``"torus"`` | ``"annular"`` | ``"2d"`` (mapped to geometry ``"planar"``).
        use_gpu : bool
        force_mode : str
            Force model name (for planar geometry).
        wall_force_mode : str
            Wall model name (for planar geometry).

        Plus geometry-specific options such as *shrink_rate*, *epsilon*,
        *lj_strength*, etc.
        """
        self.config = config

        # -- Load dataset --------------------------------------------------
        dataset_path = config["dataset_path"]
        self.data_loader = TSPDataLoader(filepath=dataset_path)
        self.data_loader.preprocess(normalize_method="minimum")
        self.original_coords = self.data_loader.original_coords
        self.n_cities = self.data_loader.n_cities

        # -- Optimal cost (if available) -----------------------------------
        dataset_dir = os.path.dirname(dataset_path)
        tour_len_path = os.path.join(dataset_dir, "tour_len.txt")
        self.optimal_cost = load_optimal_cost(tour_len_path)

        # -- Map config to PhysicsEngine args ------------------------------
        mode = config.get("mode", "torus")
        self.mode = mode
        use_gpu = config.get("use_gpu", True)
        backend = "gpu" if use_gpu else "cpu"

        # Map frontend force/wall mode names to engine names
        _FORCE_MODEL_MAP = {
            "true_lj": "true_lj",
            "smooth": "smooth_lj",
            "piecewise": "piecewise_lj",
        }
        raw_force = config.get("force_mode", "true_lj")
        force_model = _FORCE_MODEL_MAP.get(raw_force, raw_force)
        wall_model = config.get("wall_force_mode", "inverse_square")

        if mode == "2d":
            geometry = "planar"

            # Start from sensible defaults, then overlay caller overrides
            defaults = _get_2d_defaults(raw_force, wall_model)
            options: dict = {**defaults}
            for key in defaults:
                if key in config:
                    options[key] = config[key]
            for key in ("inner_radius", "outer_radius"):
                if key in config:
                    options[key] = config[key]
        else:
            geometry = mode  # "torus" or "annular"
            options = {}
            for key in (
                "shrink_rate", "epsilon", "lj_strength", "perturbation",
                "R", "r", "r0", "embed_mode",
            ):
                if key in config:
                    options[key] = config[key]

        coords = self.data_loader.coords
        self.engine = PhysicsEngine(
            coords=coords,
            geometry=geometry,
            force_model=force_model,
            wall_model=wall_model,
            backend=backend,
            options=options if options else None,
        )
        self.engine.initialize_physics()

        # Reset any prior tour state
        self.reset_tour()
        self.running = False

    def start(self):
        """Begin the wall / tube collapse process."""
        self.engine.start_collapse()
        self.running = True

    def stop(self):
        """Pause the simulation loop."""
        self.running = False

    def reset(self):
        """Re-initialise physics and clear tour state."""
        self.engine.initialize_physics()
        self.running = False
        self.reset_tour()

    def step(self) -> bool:
        """Advance the simulation by *substeps* physics steps.

        Returns ``True`` when the simulation has just completed (tour
        extracted), ``False`` otherwise.
        """
        if not self.running or self.engine is None:
            return False
        self.engine.run_substeps(self.substeps)
        if self.engine.is_complete and not self._tour_extracted:
            self.extract_tour()
            return True
        return False

    # ------------------------------------------------------------------
    # Tour extraction & statistics
    # ------------------------------------------------------------------

    def extract_tour(self):
        """Extract the tour from the engine, optionally improve it with
        local search, and compute comparison statistics."""
        if self._tour_extracted:
            return

        dist_fn = make_euclidean_dist_fn(self.original_coords)

        # Try engine's found_tour first (torus circle phase), fall back to
        # angular extraction via get_final_tour.
        tour_0indexed = self.engine.get_found_tour()
        if tour_0indexed is None:
            tour_0indexed = self.engine.get_final_tour().tolist()

        # local_search expects 1-indexed tours
        tour_1indexed = [i + 1 for i in tour_0indexed]

        self.raw_cost = tour_distance(tour_1indexed, dist_fn)
        self.final_tour = tour_0indexed  # store 0-indexed internally
        self.final_cost = self.raw_cost

        if self.use_local_search:
            self.ls_result = local_search_improve(
                tour_1indexed, dist_fn, mode=self.local_search_mode,
            )
            # Convert improved tour back to 0-indexed
            self.final_tour = [i - 1 for i in self.ls_result["tour"]]
            self.final_cost = self.ls_result["distance"]

        # Nearest-neighbour comparison
        try:
            nn = random_nearest_neighbor_tsp(
                self.original_coords, num_samples=min(self.n_cities, 20),
            )
            self.nn_cost = nn["best"]["cost"]
        except Exception:
            self.nn_cost = None

        self._tour_extracted = True

    def reset_tour(self):
        """Clear all stored tour results."""
        self.final_tour = None
        self.final_cost = None
        self.raw_cost = None
        self.ls_result = None
        self.nn_cost = None
        self._tour_extracted = False

    # ------------------------------------------------------------------
    # Metadata for web UI
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:
        """Generate a JSON-serialisable metadata dict describing the
        current simulation state, sent to the web UI."""
        meta: dict = {
            "mode": self.mode,
            "n_cities": self.n_cities,
            "optimal_cost": self.optimal_cost,
            "phase": self.engine.phase if self.engine else "READY",
        }

        # Engine parameters (unified)
        if self.engine:
            e = self.engine
            meta.update({
                "lj_strength": e.lj_strength,
                "damping": e.damping,
                "dt": e.dt,
                "collapse_rate": e.collapse_rate,
                "epsilon": e.epsilon,
                "force_mode": e.force_model,
                "wall_model": e.wall_model,
            })
            if self.mode in ("torus", "annular"):
                meta.update({
                    "R": e.R,
                    "r": e.r,
                    "r0": e.r0,
                })
                if self.mode == "torus":
                    meta["perturbation"] = e.perturbation
                if self.mode == "annular":
                    meta["inner_radius"] = max(0, e.R - e.r)
                    meta["outer_radius"] = e.R + e.r
            else:  # planar / 2d
                meta.update({
                    "inner_radius": e.inner_radius,
                    "outer_radius": e.outer_radius,
                    "wall_strength": e.wall_strength,
                })

        # Tour results (convert to 1-indexed for presentation)
        if self.final_tour is not None:
            tour_1indexed = [i + 1 for i in self.final_tour]
            meta["tour"] = tour_1indexed
            meta["tour_distance"] = self.final_cost
            meta["raw_distance"] = self.raw_cost
            if self.optimal_cost:
                meta["gap_pct"] = (
                    (self.final_cost - self.optimal_cost)
                    / self.optimal_cost
                    * 100
                )
            if self.nn_cost is not None:
                meta["nn_cost"] = self.nn_cost
                meta["nn_improvement_pct"] = (
                    (self.nn_cost - self.final_cost) / self.nn_cost * 100
                )
            if self.ls_result and self.ls_result.get("saved", 0) > 0:
                meta["ls_method"] = self.ls_result["method"]
                meta["ls_before"] = self.ls_result["before"]
                meta["ls_after"] = self.ls_result["after"]
                meta["ls_pct"] = self.ls_result["pct_improved"]

            # Tour vertex positions (original coords, for rendering the path)
            tour_positions = []
            for city_idx in self.final_tour:  # 0-indexed
                tour_positions.append(self.original_coords[city_idx].tolist())
            meta["tour_positions"] = tour_positions

        return meta

    # ------------------------------------------------------------------
    # Runtime parameter updates
    # ------------------------------------------------------------------

    def set_param(self, key: str, value):
        """Handle a runtime parameter update from the web UI."""
        if self.engine is None:
            return
        e = self.engine
        # Simulation-level params
        if key == "substeps":
            self.substeps = int(value)
        elif key == "use_local_search":
            self.use_local_search = bool(value)
        elif key == "local_search_mode":
            self.local_search_mode = str(value)
        # Common engine params
        elif key == "lj_strength":
            e.lj_strength = float(value)
        elif key in ("collapse_rate", "shrink_rate", "dr"):
            e.collapse_rate = float(value)
        elif key == "epsilon":
            e.epsilon = float(value)
        elif key in ("damping", "damp"):
            e.damping = float(value)
        elif key == "dt":
            e.dt = float(value)
        # Geometry-specific
        elif key == "wall_strength":
            e.wall_strength = float(value)
        elif key == "perturbation":
            e.perturbation = float(value)
        elif key == "embed_mode":
            e.embed_mode = value
            if not e.collapsing:
                e.initialize_physics()

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def start_optimization(self, config: dict):
        param_names = config["params"]
        param_bounds = [tuple(b) for b in config["bounds"]]
        max_trials = config.get("max_trials", 60)

        self._optimizer = BayesianOptimizer(
            param_bounds=param_bounds,
            param_names=param_names,
            max_trials=max_trials,
            verbose=True,
        )
        self._opt_param_names = param_names
        self.opt_mode = config.get("mode", "fast")
        self.optimizing = True
        self.running = False
        self._opt_visual_steps = 0

    def step_optimization_fast(self) -> Optional[dict]:
        if self._optimizer is None or self._optimizer.is_complete:
            return None

        params = self._optimizer.suggest_params()
        trial_config = {name: val for name, val in zip(self._opt_param_names, params)}

        from tsp_nbody.engine import PhysicsEngine
        from tsp_nbody.local_search import make_euclidean_dist_fn, tour_distance

        opts = dict(self.config)
        opts.update(trial_config)
        geometry = "planar" if self.mode == "2d" else self.mode
        force_model_map = {"true_lj": "true_lj", "smooth": "smooth_lj", "piecewise": "piecewise_lj"}
        raw_force = self.config.get("force_mode", "true_lj")
        force_model = force_model_map.get(raw_force, raw_force)
        wall_model = self.config.get("wall_force_mode", "inverse_square")

        engine = PhysicsEngine(
            coords=self.data_loader.coords,
            geometry=geometry, force_model=force_model, wall_model=wall_model,
            backend="gpu" if self.config.get("use_gpu", True) else "cpu",
            options=opts,
        )
        engine.initialize_physics()
        engine.start_collapse()

        for _ in range(10000):
            engine.run_substeps(4)
            if engine.is_complete:
                break

        tour = engine.get_found_tour()
        if tour is None:
            tour = engine.get_final_tour().tolist()

        tour_1indexed = [i + 1 for i in tour]
        dist_fn = make_euclidean_dist_fn(self.original_coords)
        distance = tour_distance(tour_1indexed, dist_fn)

        if not np.isfinite(distance):
            distance = 1e9

        result = self._optimizer.report_result(params, distance, tour)
        result["optimal_cost"] = self.optimal_cost
        if self.optimal_cost and self.optimal_cost > 0:
            result["best_gap_pct"] = (result["best_distance"] - self.optimal_cost) / self.optimal_cost * 100
        else:
            result["best_gap_pct"] = None

        return result

    def step_optimization_visual(self) -> Optional[dict]:
        if self._optimizer is None:
            return None

        # If engine not running, start next trial
        if not self.running and not self._optimizer.is_complete:
            params = self._optimizer.suggest_params()
            trial_config = {name: val for name, val in zip(self._opt_param_names, params)}
            for key, val in trial_config.items():
                self.set_param(key, val)

            self.engine.initialize_physics()
            self.engine.start_collapse()
            self.running = True
            self._opt_visual_steps = 0
            self._current_opt_params = list(params)
            self.reset_tour()
            return None

        # Step the simulation
        self.engine.run_substeps(self.substeps)
        self._opt_visual_steps += self.substeps

        # Check if trial is done
        timeout = self._opt_visual_steps >= self._opt_visual_max_steps
        if self.engine.is_complete or timeout:
            self.running = False

            from tsp_nbody.local_search import make_euclidean_dist_fn, tour_distance
            tour = self.engine.get_found_tour()
            if tour is None:
                tour = self.engine.get_final_tour().tolist()
            tour_1indexed = [i + 1 for i in tour]
            dist_fn = make_euclidean_dist_fn(self.original_coords)
            distance = tour_distance(tour_1indexed, dist_fn)

            if not np.isfinite(distance):
                distance = 1e9

            result = self._optimizer.report_result(self._current_opt_params, distance, tour)
            result["optimal_cost"] = self.optimal_cost
            if self.optimal_cost and self.optimal_cost > 0:
                result["best_gap_pct"] = (result["best_distance"] - self.optimal_cost) / self.optimal_cost * 100
            else:
                result["best_gap_pct"] = None

            self.reset_tour()
            return result

        return None

    def stop_optimization(self) -> dict:
        self.optimizing = False
        self.running = False

        best = {}
        if self._optimizer and self._optimizer.best_params:
            opt = self._optimizer
            best = {
                "best_params": {name: val for name, val in zip(opt.PARAM_NAMES, opt.best_params)},
                "best_distance": opt.best_dist,
                "best_tour": [i + 1 for i in opt.best_tour] if opt.best_tour else None,
            }
            for key, val in best["best_params"].items():
                self.set_param(key, val)

        self._optimizer = None
        return best
