---
name: Next steps - Bayesian optimizer integration
description: Bayesian optimizer exists in Python but needs to be wired into the web UI
type: project
---

`tsp_nbody/optimizer.py` has a working `BayesianOptimizer` class:
- 4D parameter search: shrinkRate, ljStrength, perturbation, epsilon
- 15 Latin Hypercube initial samples + 45 GP-guided trials (60 total)
- Takes a `run_trial(config) -> (distance, tour)` callable
- Returns `OptimizerResult` with best params, distance, tour, all trials

**What needs to happen:**
1. Server: Add WebSocket commands for `optimize_start`, `optimize_stop`. Run trials in an async loop, streaming progress (trial index, params, distance, best-so-far) as JSON metadata messages.
2. Frontend: Add OPTIMIZE button to ControlPanel. Show optimizer panel with trial progress, current/best distance, parameter chart. Load best params into sliders when done.
3. The optimizer currently creates a fresh `TorusPhysicsEngine` per trial. Needs to support `AnnularPhysicsEngine` too.
4. Consider running optimizer trials headless (no binary state streaming during optimization) for speed, only streaming JSON progress updates.

**How to apply:** Start a new conversation for this — it's a significant feature that touches server, frontend, and optimizer module.
