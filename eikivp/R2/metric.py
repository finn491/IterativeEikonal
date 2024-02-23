"""
    metric
    ======

    Provides tools to deal with metrics R^2. The primary methods are:
      1. `invert_metric`: compute the matrix defining the dual metric from the
      matrix defining the primal metric.
      2. `normalise`: normalise a vector to norm 1 with respect to some
      data-driven metric.
      3. `norm`: compute the norm of a vector with respect to some data-driven
      metric.
    
    Additionally, we have numerous functions to reorder arrays to align either
    with the standard array indexing conventions or with the real axes.
"""

import taichi as ti
import numpy as np

# Metric Inversion

def invert_metric(G):
    """
    Invert the metric tensor defined by the matrix `G`. If `G` is semidefinite,
    e.g. when we are dealing with a sub-Riemannian metric, the metric is first
    made definite by adding `1e-8` to the diagonal.

    Args:
        `G`: np.ndarray(shape=(2, 2), dtype=[float]) of matrix of left invariant
          metric tensor field with respect to standard basis.
    """
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.inv(G + np.identity(2) * 1e-8)
    return G_inv


# Normalisation

@ti.func
def normalise(
    vec: ti.types.vector(2, ti.f32),
    G: ti.types.matrix(2, 2, ti.f32),
    cost: ti.f32
) -> ti.types.vector(2, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in standard coordinates, to 1 with respect to 
    the datadriven left invariant metric tensor defined by `G` and `cost`.

    Args:
        `vec`: ti.types.vector(n=2, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=2, m=2, dtype=[float]) of constants of metric 
          tensor with respect to standard basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm(vec, G, cost)

@ti.func
def norm(
    vec: ti.types.vector(2, ti.f32),
    G: ti.types.matrix(2, 2, ti.f32),
    cost: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in standard coordinates with respect
    to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=2, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=2, m=2, dtype=[float]) of constants of metric 
          tensor with respect to standard basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        Norm of `vec`.
    """
    c_1, c_2 = vec[0], vec[1]
    return ti.math.sqrt(
            1 * G[0, 0] * c_1 * c_1 +
            2 * G[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            1 * G[1, 1] * c_2 * c_2
    ) * cost