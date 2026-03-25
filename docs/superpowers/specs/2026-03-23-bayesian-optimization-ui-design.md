# Bayesian Optimization Web UI

## Overview

Add Bayesian hyperparameter optimization to the web UI. Users configure which parameters to search, choose fast (headless) or visual (streamed) mode, and watch a convergence chart update in real-time. After optimization completes, the best parameters are loaded and a final visual simulation plays.

## Requirements

- User selects which engine parameters to optimize via checkboxes with preset bounds
- Toggle between **fast** (headless, milliseconds per trial) and **visual** (streamed to canvas, ~100-500ms per trial)
- Real-time convergence chart showing trial results and best-so-far line
- "Stop & Use Best" to halt early
- On completion, load best params into sliders and auto-run a final visual simulation

## Architecture

### WebSocket Protocol

**Client → Server:**

| Command | Payload | Description |
|---|---|---|
| `optimize_start` | `{ params: [str], bounds: [[lo,hi]], max_trials: int, mode: "fast"\|"visual" }` | Launch optimization |
| `optimize_stop` | (none) | Stop early, keep best result |

Note: `optimize_start` requires a new `sendOptStart(config)` helper in `useSimulation.js` since the existing `sendCmd` only sends `{ cmd }` with no payload. Similarly, `sendOptStop()` wraps `{ cmd: "optimize_stop" }`.

**Server → Client (new message types):**

| Type | Payload | When |
|---|---|---|
| `opt_trial` | `{ trial, max_trials, params, distance, best_distance, best_params, best_gap_pct, optimal_cost }` | After each trial |
| `opt_complete` | `{ best_params, best_distance, best_tour }` | All trials done or stopped |

- `best_gap_pct` is `null` when no optimal cost is available (dataset has no `tour_len.txt`)
- `best_tour` is **1-indexed** (matching the metadata tour format the frontend already expects)
- In **visual** mode, binary state packets are also streamed during each trial (same format as normal simulation)
- In **fast** mode, only `opt_trial` JSON is sent per trial

### Backend

#### BayesianOptimizer refactor: step-wise API

The existing `BayesianOptimizer.optimize()` is a blocking loop incompatible with visual mode (needs to yield control for WebSocket frame streaming). Refactor to expose a step-wise interface:

- `suggest_params() -> list[float]` — returns the next parameter set to try (Latin Hypercube for initial trials, GP-EI for subsequent)
- `report_result(params, distance, tour)` — records a trial result, updates best
- `is_complete -> bool` — True when `trial_count >= max_trials`
- `get_result() -> OptimizerResult` — returns the final result

The existing `optimize(run_trial)` method can be reimplemented on top of these primitives for backward compatibility. The `_sample_initial` and `_sample_bayesian` methods are unchanged internally.

#### simulation.py changes

New `SimulationRunner` methods:

- `start_optimization(config)` — creates `BayesianOptimizer` with user-selected params/bounds, sets `self.optimizing = True`, `self.opt_mode = "fast"|"visual"`
- `step_optimization_fast() -> dict|None` — runs one complete headless trial: calls `optimizer.suggest_params()`, runs engine to completion, calls `optimizer.report_result()`, returns trial result dict
- `step_optimization_visual() -> dict|None` — in visual mode, manages the current trial's engine stepping. Returns `None` while the trial is mid-collapse (server streams binary state). Returns trial result dict when the trial completes (engine.is_complete), then auto-configures the next trial.
- `stop_optimization() -> dict` — sets `self.optimizing = False`, returns `opt_complete` payload with best params/tour. Loads best params into engine for the final "best run".

**Degenerate trial handling:** If a headless trial produces `NaN`/`Inf` distance (particles exploded), clamp to a large penalty value (e.g., `1e9`). The GP surrogate will learn to avoid that region.

**Visual mode trial timeout:** Each visual-mode trial has a max step count (e.g., 50000 substeps). If the engine doesn't reach `is_complete` within that limit, extract tour from current positions and report whatever distance results.

#### server.py changes

- `_handle_message`: add `optimize_start` and `optimize_stop` handlers
- `optimize_start` calls `sim.start_optimization(config)`, blocks normal `start`/`reset` while `sim.optimizing`
- Main WebSocket loop: when `sim.optimizing`:
  - **Fast mode:** call `sim.step_optimization_fast()` per loop iteration, send `opt_trial` on result, `asyncio.sleep(0)` to yield
  - **Visual mode:** call `sim.step_optimization_visual()`, stream binary state via `_pack_state_binary()`, send `opt_trial` when trial completes
- On completion: send `opt_complete`, then initiate "best run" by calling `sim.reset()` with best params loaded, `sim.start()`, let normal sim loop take over

#### Final "best run" mechanism

After `opt_complete` is sent:
1. Server loads best params into engine: `sim.engine.collapse_rate = best_params["collapse_rate"]`, etc.
2. Server calls `sim.reset()` then `sim.start()` — this re-initializes physics and begins a normal visual collapse
3. The WebSocket loop resumes normal `sim.step()` + binary streaming behavior
4. Frontend sees binary state + metadata as usual, shows the collapse playing out
5. When simulation completes, normal tour extraction + results display

#### Cancellation

If the user clicks "STOP & USE BEST" mid-trial in visual mode:
- The current trial's partial result is **discarded** (not recorded in the optimizer)
- Engine is fully re-initialized with best params via `reset()` + `start()`
- Any accumulated optimizer results up to the last completed trial are preserved

### Frontend

**New components:**

**`OptimizeModal.jsx`** — setup dialog triggered by OPTIMIZE button:
- Checkboxes for available parameters, **hardcoded per geometry** in the frontend (matching optimizer.py presets):
  - Torus: collapse_rate, lj_strength, perturbation, epsilon
  - Annular: collapse_rate, lj_strength, epsilon
  - Planar: collapse_rate, lj_strength, damping
- Each row: checkbox + param name + bounds display (lo — hi)
- Trial count input (default 60)
- Fast / Visual toggle (two-segment button)
- Start + Cancel buttons
- On Start: sends `optimize_start` with selected params, their bounds, trial count, and mode

**`OptimizeOverlay.jsx`** — fixed overlay panel during optimization:
- Centered at top of screen, semi-transparent backdrop-blur background
- Trial counter: "TRIAL 23 / 60"
- Current trial params row (abbreviated: `CR:0.12 LJ:2.3 P:0.45 E:0.08`)
- Current distance + Best distance + Best gap % (gap only shown when optimal_cost available)
- Convergence chart (canvas):
  - Orange dots for each trial result (40% opacity)
  - Green line for best-so-far (monotonic)
  - Dashed green line for optimal distance (only when available)
- Best params display
- "STOP & USE BEST" button

**`useSimulation.js`** changes:
- Handle `opt_trial` and `opt_complete` message types in `ws.onmessage`
- New state: `optimizing` (bool), `optTrials` (array of `{trial, distance, params}`), `bestOptResult` (object)
- New helpers: `sendOptStart(config)` sends `{ cmd: "optimize_start", config }`, `sendOptStop()` sends `{ cmd: "optimize_stop" }`
- On `opt_complete`: set `optimizing = false`, clear overlay

**`ControlPanel.jsx`** changes:
- Add OPTIMIZE button (next to COLLAPSE/RESET row)
- All buttons disabled while `optimizing` is true
- On `opt_complete`: update slider state values to best params

**`App.jsx`** changes:
- Render `OptimizeModal` (controlled by `showOptModal` state)
- Render `OptimizeOverlay` when `optimizing` is true
- Pass optimization state + handlers through to components

### Preset Parameter Bounds

Updated to match the normalized engine parameter names. These are defined in `optimizer.py` and hardcoded in the frontend modal:

**Torus:** collapse_rate [0.02, 0.45], lj_strength [0.2, 4.5], perturbation [0.0, 1.0], epsilon [0.02, 0.25]

**Annular:** collapse_rate [0.02, 0.45], lj_strength [0.2, 4.5], epsilon [0.02, 0.25]

**Planar:** collapse_rate [0.001, 0.05], lj_strength [0.1, 50.0], damping [0.5, 1.0]

Note: `optimizer.py` preset constants must be updated to use `collapse_rate` instead of `shrink_rate`/`dr`, and planar `damping` bounds corrected from `[1.0, 50.0]` to `[0.5, 1.0]` (damping is now a multiplicative coefficient, not a friction constant).

Users can uncheck parameters they don't want optimized.

### Data Flow

```
User clicks OPTIMIZE → Modal opens → Configure → START
  → { cmd: "optimize_start", config: { params, bounds, max_trials, mode } }
  → Server: sim.start_optimization(config)
  → sim.optimizing = True

  For each trial:
    FAST: optimizer.suggest_params() → headless engine run → optimizer.report_result()
          → { type: "opt_trial", data: { trial, distance, best_distance, ... } }
          → asyncio.sleep(0)

    VISUAL: optimizer.suggest_params() → configure engine → start_collapse()
            → stream binary state (normal sim loop)
            → on engine.is_complete: optimizer.report_result()
            → { type: "opt_trial", data: { ... } }
            → 80ms delay → next trial

  Frontend: OptimizeOverlay updates chart + stats per opt_trial

  On complete (all trials, or STOP & USE BEST):
    → { type: "opt_complete", data: { best_params, best_distance, best_tour } }
    → Server: load best params, sim.reset(), sim.start()
    → Overlay disappears, sliders update
    → Final "best run" streams as normal simulation
    → Tour results shown when best run finishes
```

## UI Layout

- **Setup:** Modal dialog (centered, dark theme, checkbox list + config inputs)
- **Progress:** Fixed overlay at top center of screen, overlaying the canvas
- **Completion:** Overlay disappears, control panel updates, final visual run plays

## Files Changed

| File | Change |
|---|---|
| `tsp_nbody/optimizer.py` | Refactor BayesianOptimizer to step-wise API (suggest_params/report_result), update preset param names/bounds |
| `tsp_nbody/simulation.py` | Add start_optimization, step_optimization_fast, step_optimization_visual, stop_optimization |
| `tsp_nbody/server.py` | Handle optimize_start/stop commands, step optimization in main WebSocket loop |
| `web-ui/src/hooks/useSimulation.js` | Handle opt_trial/opt_complete messages, add sendOptStart/sendOptStop, expose optimization state |
| `web-ui/src/components/ControlPanel.jsx` | Add OPTIMIZE button |
| `web-ui/src/components/OptimizeModal.jsx` | New: setup dialog with param checkboxes + config |
| `web-ui/src/components/OptimizeOverlay.jsx` | New: progress overlay with convergence chart |
| `web-ui/src/App.jsx` | Wire up OptimizeModal + OptimizeOverlay |
