"""
WebSocket server for N-Body TSP Simulator.

Streams particle state to a web frontend and receives control commands.
Run with: python -m tsp_nbody.server
"""

import asyncio
import json
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from tsp_nbody.dataio import discover_datasets, TSPDataLoader, load_optimal_cost
from tsp_nbody.torus_physics import TorusPhysicsEngine
from tsp_nbody.annular_physics import AnnularPhysicsEngine
from tsp_nbody.physics_engine import NBodyPhysicsEngine, default_nbody_options
from tsp_nbody.local_search import improve as local_search_improve, make_euclidean_dist_fn, tour_distance
from tsp_nbody.path_extraction import PathExtractor, random_nearest_neighbor_tsp

app = FastAPI()

# Serve static files from web-ui/dist/ (built React app) or web-ui/public/
WEB_DIR = Path(__file__).parent.parent / "web-ui" / "dist"
if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")


@app.get("/")
async def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/datasets")
async def list_datasets():
    datasets = discover_datasets()
    return JSONResponse(datasets)


@app.get("/api/info")
async def server_info():
    try:
        import cupy
        gpu = True
    except ImportError:
        gpu = False
    return JSONResponse({"gpu_available": gpu})


def _get_2d_defaults(force_mode: str, wall_mode: str) -> dict:
    """Get sensible default parameters for a 2D force+wall mode combination."""
    if force_mode == 'true_lj':
        # True LJ in 2D needs weak attraction to avoid clustering.
        # The LJ attractive tail pulls distant particles together; in 2D
        # (unlike the torus) particles aren't all confined together, so
        # the attraction causes clumping. Use low lj_strength.
        if wall_mode == 'inverse_square':
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
    elif force_mode == 'smooth':
        if wall_mode == 'inverse_square':
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
        if wall_mode == 'inverse_square':
            return dict(
                lj_strength=1.0, wall_strength=500.0, damp=10.0, mass=80.0,
                dt=0.01, dr=0.01, force_cutoff=100000.0,
                slope_repulsion=50.0, mag_attraction=0.5, force_cutoff_extra=0.1,
            )
        else:
            return dict(
                lj_strength=1.0, wall_strength=20000.0, damp=20.0, mass=80.0,
                dt=0.01, dr=0.01, force_cutoff=100000.0,
                slope_repulsion=50.0, mag_attraction=25.0, force_cutoff_extra=100.0,
            )


class SimulationState:
    """Holds the current simulation state for a WebSocket session."""

    def __init__(self):
        self.engine = None
        self.mode = "torus"
        self.config = {}
        self.original_coords = None
        self.n_cities = 0
        self.optimal_cost = None
        self.substeps = 4
        self.running = False  # physics loop running
        self.data_loader = None

        # Tour results
        self.final_tour = None        # 1-indexed tour
        self.final_cost = None
        self.raw_cost = None          # before local search
        self.ls_result = None         # local search improvement stats
        self.nn_cost = None           # nearest-neighbor comparison
        self.use_local_search = False
        self.local_search_mode = "2-opt"
        self._tour_extracted = False

    def init_simulation(self, config: dict):
        """Initialize a simulation from config."""
        self.config = config
        self.mode = config.get("mode", "torus")

        # Load data
        self.data_loader = TSPDataLoader(config["dataset_path"])
        self.data_loader.preprocess(normalize_method="minimum")
        self.original_coords = self.data_loader.original_coords
        self.n_cities = self.data_loader.n_cities

        # Load optimal cost
        dataset_dir = Path(config["dataset_path"]).parent
        opt_file = dataset_dir / "tour_len.txt"
        self.optimal_cost = load_optimal_cost(str(opt_file)) if opt_file.exists() else None

        if self.mode == "torus":
            self._init_torus(config)
        elif self.mode == "annular":
            self._init_annular(config)
        else:
            self._init_2d(config)

    def _init_annular(self, config: dict):
        self.engine = AnnularPhysicsEngine(self.original_coords, options={
            'use_gpu': config.get('use_gpu', True),
            'shrink_rate': config.get('shrink_rate', 0.10),
            'epsilon': config.get('epsilon', 0.08),
            'lj_strength': config.get('lj_strength', 1.0),
            'dt': config.get('dt', 0.004),
        })
        self.engine.initialize_physics()

    def _init_torus(self, config: dict):
        self.engine = TorusPhysicsEngine(self.original_coords, options={
            'use_gpu': config.get('use_gpu', True),
            'shrink_rate': config.get('shrink_rate', 0.10),
            'epsilon': config.get('epsilon', 0.08),
            'lj_strength': config.get('lj_strength', 1.0),
            'perturbation': config.get('perturbation', 0.50),
            'embed_mode': config.get('embed_mode', 'flat'),
        })
        self.engine.initialize_physics()

    def _init_2d(self, config: dict):
        coords = self.data_loader.coords
        opts = default_nbody_options.copy()
        force_mode = config.get('force_mode', 'piecewise')
        wall_mode = config.get('wall_force_mode', 'linear')

        # Pick sensible defaults based on the combination of force + wall mode
        defaults = _get_2d_defaults(force_mode, wall_mode)
        opts.update({
            'use_gpu': config.get('use_gpu', True),
            'force_mode': force_mode,
            'lj_strength': config.get('lj_strength', defaults['lj_strength']),
            'wall_force_mode': wall_mode,
            'WALL_STRENGTH': config.get('wall_strength', defaults['wall_strength']),
            'DAMP': config.get('damp', defaults['damp']),
            'MASS': defaults['mass'],
            'DT': config.get('dt', defaults['dt']),
            'DR': config.get('dr', defaults['dr']),
            'FORCE_CUTOFF': defaults['force_cutoff'],
            'slope_repulsion': defaults['slope_repulsion'],
            'mag_attraction': defaults['mag_attraction'],
            'force_cutoff_extra': defaults['force_cutoff_extra'],
        })
        self.engine = NBodyPhysicsEngine(coords, options=opts)
        self.engine.initialize_physics()
        # Store wall state
        self.inner_radius = 0.0
        self.outer_radius = self.data_loader.get_bounding_circle_radius()
        self.dr = opts['DT']
        self.sim_finished = False

    def extract_tour(self):
        """Extract tour from current positions, apply local search, compute stats."""
        if self._tour_extracted:
            return

        dist_fn = make_euclidean_dist_fn(self.original_coords)
        path_extractor = PathExtractor()

        if self.mode == "torus":
            tour = self.engine.get_found_tour()
            if tour is None:
                tour_0 = self.engine.get_final_tour()
                tour = [int(i + 1) for i in tour_0]
        else:
            tour_0 = self.engine.get_final_tour()
            tour = [int(i + 1) for i in tour_0]

        self.raw_cost = tour_distance(tour, dist_fn)
        self.final_tour = tour
        self.final_cost = self.raw_cost

        # Local search
        if self.use_local_search:
            self.ls_result = local_search_improve(tour, dist_fn, mode=self.local_search_mode)
            self.final_tour = self.ls_result["tour"]
            self.final_cost = self.ls_result["distance"]

        # Nearest-neighbor comparison
        try:
            nn = random_nearest_neighbor_tsp(self.original_coords, num_samples=min(self.n_cities, 20))
            self.nn_cost = nn['best']['cost']
        except Exception:
            self.nn_cost = None

        self._tour_extracted = True

    def reset_tour(self):
        """Clear extracted tour state."""
        self.final_tour = None
        self.final_cost = None
        self.raw_cost = None
        self.ls_result = None
        self.nn_cost = None
        self._tour_extracted = False

    def get_state_binary(self) -> bytes:
        """Pack current state into binary for fast WebSocket transfer."""
        if self.mode == "torus":
            return self._get_torus_state()
        elif self.mode == "annular":
            return self._get_annular_state()
        else:
            return self._get_2d_state()

    def _get_annular_state(self) -> bytes:
        engine = self.engine
        pos = engine.get_positions_cpu()  # (N, 2)
        vel = engine.get_velocities_cpu()  # (N, 2)

        # Use same format as 2D: mode=2, innerR=R-r, outerR=R+r
        inner_r = max(0, engine.R - engine.r)
        outer_r = engine.R + engine.r
        phase_int = 2 if engine.collapsed else (1 if engine.collapsing else 0)
        header = struct.pack('<IIffI',
                             2,  # mode=annular
                             engine.n_cities,
                             inner_r,
                             outer_r,
                             phase_int)
        return header + pos.astype(np.float32).tobytes() + vel.astype(np.float32).tobytes()

    def _get_torus_state(self) -> bytes:
        engine = self.engine
        pos = engine.get_positions_cpu()  # (N, 3) float32
        vel = engine.get_velocities_cpu()  # (N, 3) float32

        # Header: 24 bytes (all 4-byte aligned fields)
        # mode(I=4) + N(I=4) + R(f=4) + r(f=4) + r0(f=4) + phase(I=4) = 24
        phase_int = 0 if not engine.collapsing else (2 if engine.circle_phase else 1)
        header = struct.pack('<IIfffI',
                             0,  # mode=torus
                             engine.n_cities,
                             engine.R,
                             engine.r,
                             engine.r0,
                             phase_int)

        return header + pos.astype(np.float32).tobytes() + vel.astype(np.float32).tobytes()

    def _get_2d_state(self) -> bytes:
        engine = self.engine
        pos = engine.get_positions_cpu()  # (N, 2) float32
        vel = engine.get_velocities_cpu()  # (N, 2) float32

        # Header: 20 bytes (all 4-byte aligned fields)
        # mode(I=4) + N(I=4) + innerR(f=4) + outerR(f=4) + phase(I=4) = 20
        phase_int = 2 if self.sim_finished else (1 if self.running else 0)
        header = struct.pack('<IIffI',
                             1,  # mode=2d
                             engine.n_cities,
                             self.inner_radius,
                             self.outer_radius,
                             phase_int)

        return header + pos.astype(np.float32).tobytes() + vel.astype(np.float32).tobytes()

    def get_metadata(self) -> dict:
        """Get JSON metadata about current state."""
        meta = {
            "mode": self.mode,
            "n_cities": self.n_cities,
            "optimal_cost": self.optimal_cost,
        }

        if self.mode == "torus":
            engine = self.engine
            meta.update({
                "R": engine.R,
                "r": engine.r,
                "r0": engine.r0,
                "phase": engine.phase,
                "shrink_rate": engine.shrink_rate,
                "epsilon": engine.epsilon,
                "lj_strength": engine.lj_strength,
                "perturbation": engine.perturbation,
            })
        elif self.mode == "annular":
            engine = self.engine
            meta.update({
                "R": engine.R,
                "r": engine.r,
                "r0": engine.r0,
                "phase": engine.phase,
                "inner_radius": max(0, engine.R - engine.r),
                "outer_radius": engine.R + engine.r,
                "shrink_rate": engine.shrink_rate,
                "epsilon": engine.epsilon,
                "lj_strength": engine.lj_strength,
            })
        else:
            engine = self.engine
            meta.update({
                "inner_radius": self.inner_radius,
                "outer_radius": self.outer_radius,
                "phase": "FINISHED" if self.sim_finished else ("RUNNING" if self.running else "READY"),
                "wall_strength": engine.WALL_STRENGTH if engine else 20000,
                "damp": engine.DAMP if engine else 20,
                "dt": engine.DT if engine else 0.01,
                "dr": engine.DR if engine else 0.01,
                "lj_strength": engine.lj_strength if engine else 1.0,
                "force_mode": engine.force_mode if engine else "piecewise",
            })

        # Tour results (both modes)
        if self.final_tour is not None:
            meta["tour"] = self.final_tour
            meta["tour_distance"] = self.final_cost
            meta["raw_distance"] = self.raw_cost
            if self.optimal_cost:
                meta["gap_pct"] = (self.final_cost - self.optimal_cost) / self.optimal_cost * 100
            if self.nn_cost is not None:
                meta["nn_cost"] = self.nn_cost
                nn_improvement = (self.nn_cost - self.final_cost) / self.nn_cost * 100
                meta["nn_improvement_pct"] = nn_improvement
            if self.ls_result and self.ls_result["saved"] > 0:
                meta["ls_method"] = self.ls_result["method"]
                meta["ls_before"] = self.ls_result["before"]
                meta["ls_after"] = self.ls_result["after"]
                meta["ls_pct"] = self.ls_result["pct_improved"]

            # Tour vertex positions (original coords, for rendering the path)
            tour_positions = []
            for city_id in self.final_tour:
                idx = city_id - 1
                tour_positions.append(self.original_coords[idx].tolist())
            meta["tour_positions"] = tour_positions

        return meta


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sim = SimulationState()
    physics_task = None

    try:
        while True:
            # Check for incoming messages (non-blocking with timeout)
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                msg = json.loads(raw)
                await handle_message(ws, sim, msg)
            except asyncio.TimeoutError:
                pass

            # If physics is running, step and send state
            if sim.running and sim.engine is not None:
                # Run physics substeps
                if sim.mode == "torus":
                    sim.engine.run_substeps(sim.substeps)
                    if sim.engine.circle_phase and not sim._tour_extracted:
                        sim.extract_tour()
                        await ws.send_text(json.dumps({"type": "metadata", "data": sim.get_metadata()}, cls=_NumpyEncoder))
                elif sim.mode == "annular":
                    sim.engine.run_substeps(sim.substeps)
                    if sim.engine.collapsed and not sim._tour_extracted:
                        sim.extract_tour()
                        await ws.send_text(json.dumps({"type": "metadata", "data": sim.get_metadata()}, cls=_NumpyEncoder))
                else:
                    if not sim.sim_finished:
                        for _ in range(sim.substeps):
                            sim.inner_radius += sim.dr
                            sim.engine.integrate_step(sim.inner_radius, sim.outer_radius)
                        stop_sep = sim.dr * 10.0
                        if sim.inner_radius + stop_sep >= sim.outer_radius:
                            sim.sim_finished = True
                            sim.running = False
                            sim.extract_tour()
                            await ws.send_text(json.dumps({"type": "metadata", "data": sim.get_metadata()}, cls=_NumpyEncoder))

                # Send binary state
                state_bytes = sim.get_state_binary()
                await ws.send_bytes(state_bytes)

                # Send metadata periodically (every ~30 frames worth)
                # We rely on the client to request metadata when needed

                # Small yield to prevent blocking
                await asyncio.sleep(0.001)
            else:
                # Not running — just wait for messages
                await asyncio.sleep(0.016)  # ~60fps idle

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


async def handle_message(ws: WebSocket, sim: SimulationState, msg: dict):
    """Handle a JSON message from the client."""
    cmd = msg.get("cmd")

    if cmd == "init":
        config = msg.get("config", {})
        sim.use_local_search = config.get("use_local_search", False)
        sim.local_search_mode = config.get("local_search_mode", "2-opt")
        sim.init_simulation(config)
        sim.running = False
        # Send initial state + metadata
        await ws.send_text(json.dumps({"type": "metadata", "data": sim.get_metadata()}, cls=_NumpyEncoder))
        await ws.send_bytes(sim.get_state_binary())

    elif cmd == "start":
        if sim.mode in ("torus", "annular"):
            sim.engine.start_collapse()
        sim.running = True

    elif cmd == "stop":
        sim.running = False

    elif cmd == "reset":
        if sim.mode in ("torus", "annular"):
            sim.engine.initialize_physics()
        else:
            sim.engine.initialize_physics()
            sim.inner_radius = 0.0
            sim.outer_radius = sim.data_loader.get_bounding_circle_radius()
            sim.sim_finished = False
        sim.running = False
        sim.reset_tour()
        await ws.send_bytes(sim.get_state_binary())
        await ws.send_text(json.dumps({"type": "metadata", "data": sim.get_metadata()}, cls=_NumpyEncoder))

    elif cmd == "set_param":
        key = msg.get("key")
        value = msg.get("value")
        if sim.mode == "torus" and sim.engine:
            if key == "shrink_rate": sim.engine.shrink_rate = float(value)
            elif key == "epsilon": sim.engine.epsilon = float(value)
            elif key == "lj_strength": sim.engine.lj_strength = float(value)
            elif key == "perturbation": sim.engine.perturbation = float(value)
            elif key == "substeps": sim.substeps = int(value)
            elif key == "embed_mode":
                sim.engine.embed_mode = value
                if not sim.engine.collapsing:
                    sim.engine.initialize_physics()
                    await ws.send_bytes(sim.get_state_binary())
        elif sim.mode == "annular" and sim.engine:
            if key == "shrink_rate": sim.engine.shrink_rate = float(value)
            elif key == "epsilon": sim.engine.epsilon = float(value)
            elif key == "lj_strength": sim.engine.lj_strength = float(value)
            elif key == "substeps": sim.substeps = int(value)
        elif sim.mode == "2d" and sim.engine:
            if key == "wall_strength": sim.engine.WALL_STRENGTH = float(value)
            elif key == "damp": sim.engine.DAMP = float(value)
            elif key == "dt": sim.engine.DT = float(value)
            elif key == "dr":
                sim.engine.DR = float(value)
                sim.dr = float(value)
            elif key == "substeps": sim.substeps = int(value)
            elif key == "wall_force_mode": sim.engine.wall_force_mode = value
            elif key == "lj_strength": sim.engine.lj_strength = float(value)
        # Common params
        if key == "use_local_search": sim.use_local_search = bool(value)
        elif key == "local_search_mode": sim.local_search_mode = str(value)

    elif cmd == "get_metadata":
        await ws.send_text(json.dumps({"type": "metadata", "data": sim.get_metadata()}, cls=_NumpyEncoder))

    elif cmd == "get_state":
        if sim.engine:
            await ws.send_bytes(sim.get_state_binary())


def main():
    import uvicorn
    print(f"Serving web UI from: {WEB_DIR}")
    print(f"Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
