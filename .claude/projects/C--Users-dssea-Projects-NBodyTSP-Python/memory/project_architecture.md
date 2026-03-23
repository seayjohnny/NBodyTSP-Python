---
name: Project architecture overview
description: Current architecture of the NBodyTSP-Python project after web UI migration
type: project
---

The project has a Python physics backend + React/Vite web frontend connected via WebSocket.

**Python backend** (`tsp_nbody/`):
- `server.py` — FastAPI + WebSocket server. Streams binary particle state, receives JSON commands. Handles init/start/stop/reset/set_param. Auto-extracts tour on completion with local search + NN comparison.
- `torus_physics.py` — 3D torus collapse (R+r=OUTER, LJ 12-6, per-pair sigma, GPU+CPU)
- `annular_physics.py` — 2D version of torus (same physics, no z-axis). Wall force from major circle, not origin. GPU primary, CPU fallback.
- `physics_engine.py` — Original 2D concentric walls (piecewise/smooth/true_lj force modes, linear/inverse_square wall modes, GPU+CPU)
- `local_search.py` — 2-opt/3-opt post-processing
- `optimizer.py` — Bayesian optimization (GP surrogate + EI acquisition). NOT yet wired into the web UI.
- `dataio.py` — Data loading, preprocessing, dataset discovery

**React frontend** (`web-ui/`):
- Vite + React, Tailwind via `@tailwindcss/vite`
- `SimCanvas.jsx` — Raw WebGL canvas, supports 3D torus (perspective+orbit) and 2D (orthographic) rendering
- `ControlPanel.jsx` — Theme-driven side panel with sliders, buttons, dropdowns, results, tour map
- `FullTourMap.jsx` — Full-size Canvas 2D tour visualization with city labels and directional arrows
- `TourMap.jsx` — Compact tour map in the panel
- `SetupScreen.jsx` — Dataset/mode selection
- `useSimulation.js` — WebSocket hook (binary state parsing, command sending)
- `themes.js` — 5 themes (dark, warm, cool, neon, light) with GL colors + panel UI colors

**Binary protocol** (WebSocket):
- Torus (mode=0): 24-byte header (mode, N, R, r, r0, phase) + float32 positions(N×3) + velocities(N×3)
- 2D/Annular (mode=1/2): 20-byte header (mode, N, innerR, outerR, phase) + float32 positions(N×2) + velocities(N×2)

**How to apply:** When adding features, update server.py for the backend, then React components for the frontend. GPU kernels are first-class — always update CUDA alongside CPU.
