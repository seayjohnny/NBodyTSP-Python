"""
WebSocket server for N-Body TSP Simulator.

Thin transport layer — streams particle state to a web frontend
and receives control commands. Simulation logic lives in simulation.py.

Run with: python -m tsp_nbody.server
"""

import asyncio
import json
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from tsp_nbody.dataio import discover_datasets
from tsp_nbody.simulation import SimulationRunner


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


app = FastAPI()

# Serve built React app from web-ui/dist/
WEB_DIR = Path(__file__).parent.parent / "web-ui" / "dist"
if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")


@app.get("/")
async def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/datasets")
async def list_datasets():
    return JSONResponse(discover_datasets())


@app.get("/api/info")
async def server_info():
    try:
        import cupy  # noqa: F401
        gpu = True
    except ImportError:
        gpu = False
    return JSONResponse({"gpu_available": gpu})


# ---------------------------------------------------------------------------
# Binary state packing (transport concern — format must match web-ui parser)
# ---------------------------------------------------------------------------

def _pack_state_binary(sim: SimulationRunner) -> bytes:
    """Pack current engine state into binary for fast WebSocket transfer."""
    engine = sim.engine
    pos = engine.get_positions_cpu()
    vel = engine.get_velocities_cpu()

    phase_map = {"READY": 0, "COLLAPSING": 1}
    phase_str = engine.phase
    phase_int = phase_map.get(phase_str, 2)  # terminal phases → 2

    if sim.mode == "torus":
        # Header: mode(I) + N(I) + R(f) + r(f) + r0(f) + phase(I) = 24 bytes
        header = struct.pack('<IIfffI',
                             0,  # mode=torus
                             engine.n_cities,
                             engine.R,
                             engine.r,
                             engine.r0,
                             phase_int)
    elif sim.mode == "annular":
        inner_r = max(0, engine.R - engine.r)
        outer_r = engine.R + engine.r
        header = struct.pack('<IIffI',
                             2,  # mode=annular
                             engine.n_cities,
                             inner_r,
                             outer_r,
                             phase_int)
    else:  # planar / 2d
        header = struct.pack('<IIffI',
                             1,  # mode=2d
                             engine.n_cities,
                             engine.inner_radius,
                             engine.outer_radius,
                             phase_int)

    return header + pos.astype(np.float32).tobytes() + vel.astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

async def _send_metadata(ws: WebSocket, sim: SimulationRunner):
    """Send JSON metadata to the client."""
    await ws.send_text(json.dumps(
        {"type": "metadata", "data": sim.get_metadata()},
        cls=_NumpyEncoder,
    ))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sim = SimulationRunner()

    try:
        while True:
            # Check for incoming messages (non-blocking)
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                msg = json.loads(raw)
                await _handle_message(ws, sim, msg)
            except asyncio.TimeoutError:
                pass

            # Optimization loop
            if sim.optimizing and sim.engine is not None:
                if sim.opt_mode == "fast":
                    result = sim.step_optimization_fast()
                    if result:
                        await ws.send_text(json.dumps(
                            {"type": "opt_trial", "data": result},
                            cls=_NumpyEncoder,
                        ))
                    if sim._optimizer and sim._optimizer.is_complete:
                        complete = sim.stop_optimization()
                        await ws.send_text(json.dumps(
                            {"type": "opt_complete", "data": complete},
                            cls=_NumpyEncoder,
                        ))
                        sim.reset()
                        sim.start()
                    await asyncio.sleep(0)
                else:  # visual mode
                    result = sim.step_optimization_visual()
                    if result:
                        await ws.send_text(json.dumps(
                            {"type": "opt_trial", "data": result},
                            cls=_NumpyEncoder,
                        ))
                        if sim._optimizer and sim._optimizer.is_complete:
                            complete = sim.stop_optimization()
                            await ws.send_text(json.dumps(
                                {"type": "opt_complete", "data": complete},
                                cls=_NumpyEncoder,
                            ))
                            sim.reset()
                            sim.start()
                        await asyncio.sleep(0.08)
                    else:
                        await ws.send_bytes(_pack_state_binary(sim))
                        await asyncio.sleep(0.001)

            # Normal simulation loop
            elif sim.running and sim.engine is not None:
                just_finished = sim.step()
                if just_finished:
                    await _send_metadata(ws, sim)

                await ws.send_bytes(_pack_state_binary(sim))
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0.016)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


async def _handle_message(ws: WebSocket, sim: SimulationRunner, msg: dict):
    """Handle a JSON message from the client."""
    cmd = msg.get("cmd")

    if cmd == "init":
        config = msg.get("config", {})
        sim.use_local_search = config.get("use_local_search", False)
        sim.local_search_mode = config.get("local_search_mode", "2-opt")
        sim.init_simulation(config)
        sim.running = False
        await _send_metadata(ws, sim)
        await ws.send_bytes(_pack_state_binary(sim))

    elif cmd == "start":
        if not sim.optimizing:
            sim.start()

    elif cmd == "stop":
        sim.stop()

    elif cmd == "reset":
        if not sim.optimizing:
            sim.reset()
            await ws.send_bytes(_pack_state_binary(sim))
            await _send_metadata(ws, sim)

    elif cmd == "set_param":
        key = msg.get("key")
        value = msg.get("value")
        sim.set_param(key, value)
        # If embed_mode changed and physics was re-initialized, send new state
        if key == "embed_mode" and sim.engine and not sim.engine.collapsing:
            await ws.send_bytes(_pack_state_binary(sim))

    elif cmd == "optimize_start":
        if not sim.engine:
            return
        config = msg.get("config", {})
        sim.init_simulation(sim.config)  # re-init with current dataset
        sim.start_optimization(config)

    elif cmd == "optimize_stop":
        if sim.optimizing:
            result = sim.stop_optimization()
            await ws.send_text(json.dumps(
                {"type": "opt_complete", "data": result},
                cls=_NumpyEncoder,
            ))
            sim.reset()
            sim.start()

    elif cmd == "get_metadata":
        await _send_metadata(ws, sim)

    elif cmd == "get_state":
        if sim.engine:
            await ws.send_bytes(_pack_state_binary(sim))


def main():
    import uvicorn
    print(f"Serving web UI from: {WEB_DIR}")
    print(f"Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
