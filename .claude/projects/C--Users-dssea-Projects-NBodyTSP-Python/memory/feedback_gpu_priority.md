---
name: GPU is first-class and default
description: GPU compute should be the default, not CPU. GPU is the priority compute target.
type: feedback
---

GPU compute should be default and first-class. When GPU (CuPy/CUDA) is available, always default to it. CPU is the fallback.

**Why:** The user considers GPU the primary compute path. CPU-only testing is insufficient — GPU kernels must be tested and kept in sync with CPU stability fixes.

**How to apply:** When adding physics features, always update CUDA kernels first or simultaneously with CPU code. Default `use_gpu=True` when CUDA is available. Test GPU paths explicitly, not just CPU.
