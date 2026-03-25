"""Shared CUDA helper snippets prepended to every compiled kernel."""

# Math constants and small utility functions available to all kernels.
# The kernel builder prepends this block before force/wall/geometry code.

COMMON_HEADER = r"""
#ifndef M_PI_F
#define M_PI_F 3.14159265f
#endif

#ifndef TAU_F
#define TAU_F 6.28318530f
#endif
"""
