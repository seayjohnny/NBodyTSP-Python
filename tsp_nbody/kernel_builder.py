"""Compose CUDA kernel source strings from device-function snippets and
geometry templates, then compile them with CuPy.

The kernel source is assembled by concatenating:
  1. ``#define`` statements for force-model-specific parameters
  2. Common helpers            (``kernels.common``)
  3. Force-model device func   (``kernels.force_models``)
  4. Wall-model device func    (``kernels.wall_models``)
  5. Geometry kernel template  (``kernels.geometries``)

Compiled kernels are cached so repeated calls with the same arguments
return the same ``cp.RawKernel`` without recompilation.
"""

from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# CuPy import (soft dependency)
# ---------------------------------------------------------------------------
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False


def _require_cupy() -> None:
    """Raise a clear error when CuPy is needed but not installed."""
    if not _CUPY_AVAILABLE:
        raise RuntimeError(
            "CuPy is required to compile CUDA kernels but is not installed. "
            "Install it with:  pip install cupy-cuda12x  (adjust for your CUDA version)."
        )


# ---------------------------------------------------------------------------
# Snippet imports (string constants from the kernels sub-package)
# ---------------------------------------------------------------------------
from tsp_nbody.kernels.common import COMMON_HEADER
from tsp_nbody.kernels.force_models import FORCE_MODELS as _FORCE_SNIPPETS
from tsp_nbody.kernels.wall_models import WALL_MODELS as _WALL_SNIPPETS
from tsp_nbody.kernels.geometries import GEOMETRY_KERNELS as _ALL_GEOMETRY

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
_GEOMETRY_TEMPLATES: Dict[str, str] = {
    k: v for k, v in _ALL_GEOMETRY.items() if k != "torus_circle"
}
_TORUS_CIRCLE_KERNEL: str = _ALL_GEOMETRY["torus_circle"]

# Kernel entry-point names embedded in the geometry templates.
_KERNEL_NAMES: Dict[str, str] = {
    "planar": "nbody_step",
    "torus": "nbody_step",
    "annular": "nbody_step",
    "torus_circle": "circle_step",
}

# ---------------------------------------------------------------------------
# Default #define values per force model
# ---------------------------------------------------------------------------
_PIECEWISE_DEFAULTS: Dict[str, float] = {
    "slope_repulsion": 50.0,
    "mag_attraction": 0.5,
    "force_cutoff_extra": 0.1,
}

_SMOOTH_DEFAULTS: Dict[str, float] = {
    "p": 6.0,
    "q": 12.0,
    "m": -0.05,
}

# true_lj needs no extra defines.

_FORCE_DEFINES: Dict[str, Dict[str, float]] = {
    "piecewise_lj": _PIECEWISE_DEFAULTS,
    "smooth_lj": _SMOOTH_DEFAULTS,
    "true_lj": {},
}

# Maps (force_param_key -> CUDA #define name)
_DEFINE_NAMES: Dict[str, str] = {
    "slope_repulsion": "SLOPE_REPULSION",
    "mag_attraction": "MAG_ATTRACTION",
    "force_cutoff_extra": "FORCE_CUTOFF_EXTRA",
    "p": "SMOOTH_P",
    "q": "SMOOTH_Q",
    "m": "SMOOTH_M",
}

# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------
_kernel_cache: Dict[tuple, "cp.RawKernel"] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_defines(force_model: str, force_params: Optional[dict]) -> str:
    """Return a string of ``#define`` directives for *force_model*.

    User-supplied *force_params* override the built-in defaults.
    """
    defaults = _FORCE_DEFINES.get(force_model, {})
    if not defaults:
        return ""

    merged = {**defaults, **(force_params or {})}
    lines: list[str] = []
    for key, value in merged.items():
        define_name = _DEFINE_NAMES.get(key)
        if define_name is None:
            continue
        lines.append(f"#define {define_name} {float(value)}f")
    return "\n".join(lines) + "\n"


def _cache_key(
    geometry: str,
    force_model: str,
    wall_model: str,
    force_params: Optional[dict],
) -> tuple:
    """Build a hashable cache key from the kernel configuration."""
    frozen_params = frozenset((force_params or {}).items())
    return (geometry, force_model, wall_model, frozen_params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_kernel(
    geometry: str,
    force_model: str,
    wall_model: str,
    force_params: dict = None,
) -> "cp.RawKernel":
    """Compose and compile a CUDA kernel from device functions + geometry template.

    Parameters
    ----------
    geometry : str
        One of ``"planar"``, ``"torus"``, ``"annular"``.
    force_model : str
        One of ``"piecewise_lj"``, ``"smooth_lj"``, ``"true_lj"``.
    wall_model : str
        One of ``"linear"``, ``"inverse_square"``.
    force_params : dict, optional
        Extra ``#define`` values that override the built-in defaults for the
        chosen *force_model* (e.g. ``{"slope_repulsion": 80.0}``).

    Returns
    -------
    cp.RawKernel
        A compiled CuPy RawKernel ready to launch.
    """
    _require_cupy()

    key = _cache_key(geometry, force_model, wall_model, force_params)
    if key in _kernel_cache:
        return _kernel_cache[key]

    # Validate selections
    if geometry not in _GEOMETRY_TEMPLATES:
        raise ValueError(
            f"Unknown geometry {geometry!r}. "
            f"Choose from {list(_GEOMETRY_TEMPLATES)}."
        )
    if force_model not in _FORCE_SNIPPETS:
        raise ValueError(
            f"Unknown force_model {force_model!r}. "
            f"Choose from {list(_FORCE_SNIPPETS)}."
        )
    if wall_model not in _WALL_SNIPPETS:
        raise ValueError(
            f"Unknown wall_model {wall_model!r}. "
            f"Choose from {list(_WALL_SNIPPETS)}."
        )

    # Assemble source
    defines = _build_defines(force_model, force_params)
    source = "\n".join([
        defines,
        COMMON_HEADER,
        _FORCE_SNIPPETS[force_model],
        _WALL_SNIPPETS[wall_model],
        _GEOMETRY_TEMPLATES[geometry],
    ])

    kernel_name = _KERNEL_NAMES[geometry]
    kernel = cp.RawKernel(source, kernel_name)

    _kernel_cache[key] = kernel
    return kernel


def build_circle_kernel() -> "cp.RawKernel":
    """Build the torus circle-phase kernel (standalone, not composed).

    This kernel is self-contained and does not use the force/wall snippet
    composition — it has its own angular LJ logic baked in.

    Returns
    -------
    cp.RawKernel
        A compiled CuPy RawKernel for the circle collapse step.
    """
    _require_cupy()

    key = ("torus_circle",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    source = "\n".join([
        COMMON_HEADER,
        _TORUS_CIRCLE_KERNEL,
    ])

    kernel_name = _KERNEL_NAMES["torus_circle"]
    kernel = cp.RawKernel(source, kernel_name)

    _kernel_cache[key] = kernel
    return kernel
