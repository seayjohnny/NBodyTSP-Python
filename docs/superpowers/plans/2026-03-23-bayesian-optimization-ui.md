# Bayesian Optimization Web UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Bayesian hyperparameter optimization to the web UI with fast/visual modes, real-time convergence chart, and parameter selection.

**Architecture:** Server-side optimizer with step-wise API. WebSocket streams per-trial results (`opt_trial`) and completion (`opt_complete`). Frontend shows setup modal, progress overlay with convergence canvas, and auto-plays final "best run". Two modes: fast (headless trials) and visual (streamed particle state per trial).

**Tech Stack:** Python (FastAPI, NumPy), React (Vite), Canvas 2D (convergence chart), WebSocket (JSON + binary)

**Spec:** `docs/superpowers/specs/2026-03-23-bayesian-optimization-ui-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tsp_nbody/optimizer.py` | Modify | Add step-wise API (`suggest_params`, `report_result`), update preset param names/bounds |
| `tsp_nbody/simulation.py` | Modify | Add optimization lifecycle methods to `SimulationRunner` |
| `tsp_nbody/server.py` | Modify | Handle `optimize_start`/`optimize_stop`, step optimization in WebSocket loop |
| `web-ui/src/hooks/useSimulation.js` | Modify | Handle `opt_trial`/`opt_complete` messages, expose optimization state |
| `web-ui/src/components/ControlPanel.jsx` | Modify | Add OPTIMIZE button |
| `web-ui/src/components/OptimizeModal.jsx` | Create | Setup dialog with parameter checkboxes, trial count, fast/visual toggle |
| `web-ui/src/components/OptimizeOverlay.jsx` | Create | Progress overlay with convergence chart canvas |
| `web-ui/src/App.jsx` | Modify | Wire up modal + overlay components |

---

### Task 1: Refactor BayesianOptimizer to step-wise API

**Files:**
- Modify: `tsp_nbody/optimizer.py`

The existing `optimize()` method is a blocking loop. We need to expose individual trial stepping so the server can interleave trials with WebSocket frame streaming.

- [ ] **Step 1: Add `suggest_params()` method**

Add this method to `BayesianOptimizer` (after `_sample_bayesian`, around line 176):

```python
def suggest_params(self) -> list[float]:
    """Return the next parameter set to evaluate.

    Uses Latin Hypercube sampling for the first ``n_initial`` trials,
    then GP-EI acquisition for subsequent trials.
    """
    idx = len(self.trials)
    if idx < self.n_initial:
        return self._sample_initial(idx)
    return self._sample_bayesian()
```

- [ ] **Step 2: Add `report_result()` method**

Add after `suggest_params`:

```python
def report_result(
    self,
    params: list[float],
    distance: float,
    tour: Optional[list[int]] = None,
) -> dict:
    """Record a completed trial and return a summary dict.

    Returns a dict with keys: trial, max_trials, params (dict),
    distance, best_distance, best_params (dict), best_gap_pct (optional).
    """
    trial = OptTrial(params=params, distance=distance)
    self.trials.append(trial)

    if distance < self.best_dist:
        self.best_dist = distance
        self.best_params = list(params)
        self.best_tour = list(tour) if tour else None

    param_dict = {
        name: val for name, val in zip(self.PARAM_NAMES, params)
    }
    best_param_dict = {
        name: val
        for name, val in zip(self.PARAM_NAMES, self.best_params)
    } if self.best_params else {}

    return {
        "trial": len(self.trials),
        "max_trials": self.max_trials,
        "params": param_dict,
        "distance": distance,
        "best_distance": self.best_dist,
        "best_params": best_param_dict,
    }
```

- [ ] **Step 3: Add `is_complete` property and `get_result()` method**

```python
@property
def is_complete(self) -> bool:
    """True when all trials have been run."""
    return len(self.trials) >= self.max_trials

def get_result(self) -> OptimizerResult:
    """Return the final optimization result."""
    return OptimizerResult(
        best_params=self.best_params or [0.0] * len(self.PARAM_BOUNDS),
        best_distance=self.best_dist,
        best_tour=self.best_tour,
        trials=self.trials,
        param_names=list(self.PARAM_NAMES),
    )
```

- [ ] **Step 4: Reimplement `optimize()` on top of step-wise API**

Replace the body of the existing `optimize()` method (lines 195-261) with:

```python
def optimize(
    self,
    run_trial: Callable[[dict], tuple[float, Optional[list[int]]]],
    on_trial_complete: Optional[Callable[[int, OptTrial, float], None]] = None,
) -> OptimizerResult:
    """Run the full optimization loop (blocking).

    Reimplemented on top of suggest_params/report_result for
    backward compatibility.
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
        print("=" * 60)

    while not self.is_complete:
        params = self.suggest_params()
        trial_config = {
            name: val for name, val in zip(self.PARAM_NAMES, params)
        }
        trial_config['seed'] = self.seed + len(self.trials) * 7

        distance, tour = run_trial(trial_config)
        method = "LHS" if len(self.trials) < self.n_initial else "GP-EI"
        self.report_result(params, distance, tour)

        if self.verbose:
            param_str = "  ".join(
                f"{name}:{val:.3f}"
                for name, val in zip(self.PARAM_NAMES, params)
            )
            print(
                f"  Trial {len(self.trials):3d}/{self.max_trials} "
                f"[{method:5s}] dist={distance:,.0f}  "
                f"best={self.best_dist:,.0f}  {param_str}"
            )

        if on_trial_complete:
            on_trial_complete(
                len(self.trials) - 1,
                self.trials[-1],
                self.best_dist,
            )

    if self.verbose:
        print(f"\nBEST: dist={self.best_dist:,.0f}")
        if self.best_params:
            ps = "  ".join(
                f"{n}={v:.3f}"
                for n, v in zip(self.PARAM_NAMES, self.best_params)
            )
            print(f"  {ps}")
        print("=" * 60 + "\n")

    return self.get_result()
```

- [ ] **Step 5: Update preset param names and bounds**

Replace the preset constants at the top of the file (lines 21-41):

```python
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
```

- [ ] **Step 6: Verify optimizer still works**

Run: `.venv/Scripts/python.exe -c "from tsp_nbody.optimizer import BayesianOptimizer, TORUS_PARAM_NAMES; print('OK', TORUS_PARAM_NAMES)"`

Expected: `OK ['collapse_rate', 'lj_strength', 'perturbation', 'epsilon']`

- [ ] **Step 7: Commit**

```bash
git add tsp_nbody/optimizer.py
git commit -m "refactor: add step-wise API to BayesianOptimizer (suggest_params/report_result)"
```

---

### Task 2: Add optimization methods to SimulationRunner

**Files:**
- Modify: `tsp_nbody/simulation.py`

- [ ] **Step 1: Add optimization state fields to `__init__`**

Add these fields at the end of `__init__` (after `self._tour_extracted = False`):

```python
        # Optimization state
        self.optimizing: bool = False
        self.opt_mode: str = "fast"  # "fast" or "visual"
        self._optimizer: Optional["BayesianOptimizer"] = None
        self._opt_param_names: list[str] = []
        self._opt_visual_steps: int = 0
        self._opt_visual_max_steps: int = 50000
        self._current_opt_params: list[float] = []
```

Add the import at the top of the file (with the other imports):

```python
from tsp_nbody.optimizer import BayesianOptimizer, make_engine_objective
```

- [ ] **Step 2: Add `start_optimization()` method**

Add after `set_param`:

```python
    def start_optimization(self, config: dict):
        """Initialize and start a Bayesian optimization run.

        Config keys: params (list[str]), bounds (list[list[float, float]]),
        max_trials (int), mode ("fast"|"visual").
        """
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
```

- [ ] **Step 3: Add `step_optimization_fast()` method**

```python
    def step_optimization_fast(self) -> Optional[dict]:
        """Run one complete headless trial. Returns trial result dict."""
        if self._optimizer is None or self._optimizer.is_complete:
            return None

        params = self._optimizer.suggest_params()
        trial_config = {
            name: val
            for name, val in zip(self._opt_param_names, params)
        }
        seed = 42 + len(self._optimizer.trials) * 7

        # Build a headless engine and run to completion
        from tsp_nbody.engine import PhysicsEngine
        from tsp_nbody.local_search import make_euclidean_dist_fn, tour_distance

        opts = dict(self.config)
        opts.update(trial_config)
        # Map mode to geometry
        geometry = "planar" if self.mode == "2d" else self.mode
        force_model_map = {
            "true_lj": "true_lj", "smooth": "smooth_lj",
            "piecewise": "piecewise_lj",
        }
        raw_force = self.config.get("force_mode", "true_lj")
        force_model = force_model_map.get(raw_force, raw_force)
        wall_model = self.config.get("wall_force_mode", "inverse_square")

        engine = PhysicsEngine(
            coords=self.data_loader.coords,
            geometry=geometry,
            force_model=force_model,
            wall_model=wall_model,
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

        # Clamp degenerate results
        if not np.isfinite(distance):
            distance = 1e9

        result = self._optimizer.report_result(params, distance, tour)
        result["optimal_cost"] = self.optimal_cost
        if self.optimal_cost and self.optimal_cost > 0:
            result["best_gap_pct"] = (
                (result["best_distance"] - self.optimal_cost)
                / self.optimal_cost * 100
            )
        else:
            result["best_gap_pct"] = None

        return result
```

- [ ] **Step 4: Add `step_optimization_visual()` method**

```python
    def step_optimization_visual(self) -> Optional[dict]:
        """Step one visual-mode trial. Returns trial result dict when
        a trial completes, None while mid-collapse."""
        if self._optimizer is None:
            return None

        # If engine not running, start next trial
        if not self.running and not self._optimizer.is_complete:
            params = self._optimizer.suggest_params()
            trial_config = {
                name: val
                for name, val in zip(self._opt_param_names, params)
            }
            # Apply trial params to the engine
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

            # Compute distance directly (skip expensive NN comparison)
            from tsp_nbody.local_search import make_euclidean_dist_fn, tour_distance
            tour = self.engine.get_found_tour()
            if tour is None:
                tour = self.engine.get_final_tour().tolist()
            tour_1indexed = [i + 1 for i in tour]
            dist_fn = make_euclidean_dist_fn(self.original_coords)
            distance = tour_distance(tour_1indexed, dist_fn)

            if not np.isfinite(distance):
                distance = 1e9

            result = self._optimizer.report_result(
                self._current_opt_params, distance, tour,
            )
            result["optimal_cost"] = self.optimal_cost
            if self.optimal_cost and self.optimal_cost > 0:
                result["best_gap_pct"] = (
                    (result["best_distance"] - self.optimal_cost)
                    / self.optimal_cost * 100
                )
            else:
                result["best_gap_pct"] = None

            self.reset_tour()
            return result

        return None
```

- [ ] **Step 5: Add `stop_optimization()` method**

```python
    def stop_optimization(self) -> dict:
        """Stop optimization and return the best result."""
        self.optimizing = False
        self.running = False

        best = {}
        if self._optimizer and self._optimizer.best_params:
            opt = self._optimizer
            best = {
                "best_params": {
                    name: val
                    for name, val in zip(opt.PARAM_NAMES, opt.best_params)
                },
                "best_distance": opt.best_dist,
                "best_tour": (
                    [i + 1 for i in opt.best_tour]
                    if opt.best_tour else None
                ),
            }

            # Load best params into engine for the final visual run
            for key, val in best["best_params"].items():
                self.set_param(key, val)

        self._optimizer = None
        return best
```

- [ ] **Step 6: Verify simulation module imports**

Run: `.venv/Scripts/python.exe -c "from tsp_nbody.simulation import SimulationRunner; s = SimulationRunner(); print('optimizing:', s.optimizing)"`

Expected: `optimizing: False`

- [ ] **Step 7: Commit**

```bash
git add tsp_nbody/simulation.py
git commit -m "feat: add optimization lifecycle methods to SimulationRunner"
```

---

### Task 3: Add optimization commands to server.py

**Files:**
- Modify: `tsp_nbody/server.py`

- [ ] **Step 1: Add optimization handlers to `_handle_message`**

Add these cases at the end of the `_handle_message` function (before the final `elif cmd == "get_state"` block):

```python
    elif cmd == "optimize_start":
        if not sim.engine:
            return  # no simulation initialized yet
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
            # Start final "best run"
            sim.reset()
            sim.start()
```

- [ ] **Step 2: Add optimization stepping to the main WebSocket loop**

Replace the `if sim.running` block in `websocket_endpoint` (lines 134-143) with:

```python
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
```

- [ ] **Step 3: Block normal commands during optimization**

Add a guard at the top of `start` and `reset` handlers:

```python
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
```

- [ ] **Step 4: Verify server loads**

Run: `.venv/Scripts/python.exe -c "from tsp_nbody.server import app; print('server OK')"`

Expected: `server OK`

- [ ] **Step 5: Commit**

```bash
git add tsp_nbody/server.py
git commit -m "feat: add optimize_start/stop WebSocket commands and optimization loop"
```

---

### Task 4: Update useSimulation.js for optimization state

**Files:**
- Modify: `web-ui/src/hooks/useSimulation.js`

- [ ] **Step 1: Add optimization state and handlers**

Replace the entire file with:

```javascript
import { useRef, useState, useCallback, useEffect } from 'react';

export function useSimulation() {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [metadata, setMetadata] = useState({});
  const [optimizing, setOptimizing] = useState(false);
  const [optTrials, setOptTrials] = useState([]);
  const [bestOptResult, setBestOptResult] = useState(null);

  const stateRef = useRef({
    positions: null, velocities: null, N: 0,
    R: 0, r: 0, r0: 0, phase: 0,
    innerR: 0, outerR: 1, mode: 'torus',
  });
  const frameCallbackRef = useRef(null);

  const connect = useCallback((config, onFrame) => {
    frameCallbackRef.current = onFrame;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${location.host}/ws`);
    wsRef.current = ws;
    stateRef.current.mode = config.mode;

    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({ cmd: 'init', config }));
    };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        const msg = JSON.parse(e.data);
        if (msg.type === 'metadata') setMetadata(msg.data);
        else if (msg.type === 'opt_trial') {
          setOptTrials(prev => [...prev, msg.data]);
          setBestOptResult(msg.data);
        }
        else if (msg.type === 'opt_complete') {
          setOptimizing(false);
          setBestOptResult(msg.data);
        }
      } else {
        parseState(e.data, stateRef.current);
        if (frameCallbackRef.current) frameCallbackRef.current(stateRef.current);
      }
    };
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    setConnected(false);
    setMetadata({});
    setOptimizing(false);
    setOptTrials([]);
    setBestOptResult(null);
  }, []);

  const sendCmd = useCallback((cmd) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ cmd }));
  }, []);

  const sendParam = useCallback((key, value) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ cmd: 'set_param', key, value }));
  }, []);

  const sendOptStart = useCallback((config) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setOptimizing(true);
      setOptTrials([]);
      setBestOptResult(null);
      wsRef.current.send(JSON.stringify({ cmd: 'optimize_start', config }));
    }
  }, []);

  const sendOptStop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ cmd: 'optimize_stop' }));
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return {
    connect, disconnect, sendCmd, sendParam,
    sendOptStart, sendOptStop,
    connected, metadata, stateRef,
    optimizing, optTrials, bestOptResult,
  };
}

function parseState(ab, state) {
  const buf = new DataView(ab);
  const mode = buf.getUint32(0, true);

  if (mode === 0) {
    state.mode = 'torus';
    state.N = buf.getUint32(4, true);
    state.R = buf.getFloat32(8, true);
    state.r = buf.getFloat32(12, true);
    state.r0 = buf.getFloat32(16, true);
    state.phase = buf.getUint32(20, true);
    const h = 24, n3 = state.N * 3;
    state.positions = new Float32Array(ab, h, n3);
    state.velocities = new Float32Array(ab, h + n3 * 4, n3);
  } else {
    state.mode = (mode === 2) ? 'annular' : '2d';
    state.N = buf.getUint32(4, true);
    state.innerR = buf.getFloat32(8, true);
    state.outerR = buf.getFloat32(12, true);
    state.phase = buf.getUint32(16, true);
    const h = 20, n2 = state.N * 2;
    state.positions = new Float32Array(ab, h, n2);
    state.velocities = new Float32Array(ab, h + n2 * 4, n2);
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add web-ui/src/hooks/useSimulation.js
git commit -m "feat: add optimization state and helpers to useSimulation hook"
```

---

### Task 5: Create OptimizeModal component

**Files:**
- Create: `web-ui/src/components/OptimizeModal.jsx`

- [ ] **Step 1: Create the modal component**

Create `web-ui/src/components/OptimizeModal.jsx` with parameter checkboxes per geometry, trial count, fast/visual toggle, and start/cancel buttons. The param presets are hardcoded to match `optimizer.py`.

The component receives props: `mode` (string), `theme` (object), `onStart` (callback with config), `onClose` (callback).

On Start, it sends a config object: `{ params: [...selected param names], bounds: [...matching bounds], max_trials: number, mode: "fast"|"visual" }`.

Write this as a self-contained functional component with local state for checkbox selections, trial count, and fast/visual toggle. Style it to match the existing dark theme panel aesthetic (see `ControlPanel.jsx` for reference patterns: monospace font, `#1c1e24` backgrounds, `#c0582a` accent color, `#2a2d36` borders).

The modal backdrop should be a semi-transparent dark overlay that covers the full screen. The modal itself is centered with `max-width: 400px`.

- [ ] **Step 2: Commit**

```bash
git add web-ui/src/components/OptimizeModal.jsx
git commit -m "feat: add OptimizeModal component for optimization setup"
```

---

### Task 6: Create OptimizeOverlay component

**Files:**
- Create: `web-ui/src/components/OptimizeOverlay.jsx`

- [ ] **Step 1: Create the overlay component**

Create `web-ui/src/components/OptimizeOverlay.jsx` with:
- Fixed position centered at top of screen
- Semi-transparent backdrop-blur panel (`rgba(28,30,36,0.95)`, `backdrop-filter: blur(12px)`)
- Trial counter display
- Current trial params row (abbreviated: `CR:0.12 LJ:2.3`)
- Current distance + Best distance + Best gap %
- Canvas element (440x100 CSS px, 2x DPI) for convergence chart
- Best params row
- "STOP & USE BEST" button

Props: `optTrials` (array), `bestOptResult` (object), `onStop` (callback), `theme` (object), `optimalCost` (number|null).

The convergence chart should be drawn using a `useEffect` that re-renders the Canvas 2D context whenever `optTrials` changes:
- Orange dots (opacity 0.4) for each trial distance
- Green line connecting the best-so-far distances (monotonic decrease)
- Dashed green line at `optimalCost` (if available)
- Y-axis: auto-scaled from min to max distance across all trials
- X-axis: trial index 0 to max_trials

Reference the JS project's `OPT.drawChart()` method for the exact rendering logic (detailed in the spec exploration notes).

- [ ] **Step 2: Commit**

```bash
git add web-ui/src/components/OptimizeOverlay.jsx
git commit -m "feat: add OptimizeOverlay component with convergence chart"
```

---

### Task 7: Wire up optimization in App.jsx and ControlPanel.jsx

**Files:**
- Modify: `web-ui/src/App.jsx`
- Modify: `web-ui/src/components/ControlPanel.jsx`

- [ ] **Step 1: Update App.jsx to manage optimization state and render new components**

Add imports for `OptimizeModal` and `OptimizeOverlay`. Add `showOptModal` state. Destructure `sendOptStart`, `sendOptStop`, `optimizing`, `optTrials`, `bestOptResult` from `useSimulation()`. Pass these through to child components. Render `OptimizeModal` when `showOptModal` is true. Render `OptimizeOverlay` when `optimizing` is true.

Key changes to `App.jsx`:

```jsx
// Add to imports:
import OptimizeModal from './components/OptimizeModal';
import OptimizeOverlay from './components/OptimizeOverlay';

// In App():
const [showOptModal, setShowOptModal] = useState(false);
const {
  connect, disconnect, sendCmd, sendParam,
  sendOptStart, sendOptStop,
  metadata, stateRef,
  optimizing, optTrials, bestOptResult,
} = useSimulation();

// Add handler:
const handleOptStart = useCallback((config) => {
  setShowOptModal(false);
  sendOptStart(config);
}, [sendOptStart]);

// In the render, after <ControlPanel ... />:
// Add OptimizeModal and OptimizeOverlay
```

- [ ] **Step 2: Add OPTIMIZE button to ControlPanel.jsx**

Add an `onOptimize` prop. Add an OPTIMIZE button in the button row (after COLLAPSE/RESET). Disable all buttons when `optimizing` is true.

The button row should become three buttons: COLLAPSE/START, RESET, OPTIMIZE.

Add the `optimizing` prop and use it to disable buttons:

```jsx
// In the button row:
<div className="flex gap-2 pt-2">
  <button onClick={() => sendCmd('start')} disabled={optimizing} ...>
    {isTorus ? 'COLLAPSE' : 'START'}
  </button>
  <button onClick={() => sendCmd('reset')} disabled={optimizing} ...>
    RESET
  </button>
</div>
<button onClick={onOptimize} disabled={optimizing}
  className="w-full text-xs font-mono font-semibold py-2 rounded-md border cursor-pointer mt-2"
  style={{ background: t.panelLight, borderColor: t.panelBorder, color: t.accent }}>
  OPTIMIZE
</button>
```

When `bestOptResult` is received via `opt_complete`, update the slider params state to match the best params. Add a `useEffect` watching `bestOptResult`:

```jsx
useEffect(() => {
  if (!optimizing && bestOptResult?.best_params) {
    setParams(p => ({ ...p, ...bestOptResult.best_params }));
  }
}, [bestOptResult, optimizing]);
```

- [ ] **Step 3: Build and verify**

Run: `cd web-ui && npm run build`

Expected: Build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add web-ui/src/App.jsx web-ui/src/components/ControlPanel.jsx
git commit -m "feat: wire up optimization modal and overlay in App and ControlPanel"
```

---

### Task 8: End-to-end test

- [ ] **Step 1: Start the server**

Run: `.venv/Scripts/python.exe main.py`

- [ ] **Step 2: Test fast mode optimization**

1. Open http://localhost:8000
2. Select a dataset (e.g., grid4x4) + Torus mode + True LJ
3. Click Start Simulation → should see particles
4. Click OPTIMIZE → modal appears
5. Leave all params checked, 60 trials, FAST mode
6. Click START OPTIMIZATION
7. Verify: overlay appears, trial counter increments rapidly, chart draws dots + best line
8. After completion: overlay disappears, sliders update, final collapse plays

- [ ] **Step 3: Test visual mode optimization**

Repeat step 2 but select VISUAL mode. Verify each trial shows particle collapse in the canvas, trials take longer, chart updates after each collapse completes.

- [ ] **Step 4: Test early stopping**

Start an optimization (either mode), click "STOP & USE BEST" mid-run. Verify: optimization stops, best params loaded, final run plays.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: end-to-end optimization fixes"
```
