"""
    metric
    ======

    Provides tools to deal with Riemannian metrics on SO(3). The primary methods
    are:
      1. `invert_metric`: compute the matrix defining the dual metric from the
      matrix defining the primal metric.
      2. `normalise_LI`: normalise a vector, given with respect to the left
      invariant frame, to norm 1 with respect to some data-driven Riemannian
      metric.
      3. `norm_LI`: compute the norm of a vector, given with respect to the left
      invariant frame, with respect to some data-driven Riemannian metric.
      4. `normalise_static`: normalise a vector, given with respect to the
      static frame, to norm 1 with respect to some data-driven Riemannian
      metric.
      5. `norm_static`: compute the norm of a vector, given with respect to the
      static frame, with respect to some data-driven Riemannian metric.
"""

import taichi as ti

# Metric Inversion

def invert_metric(G):
    """
    Invert the diagonal metric tensor defined by the matrix `G`.

    Args:
        `G`: np.ndarray(shape=(3,), dtype=[float]) of matrix of left invariant
          metric tensor field with respect to left invariant basis.
    """
    G_inv = 1 / G
    return G_inv


# Normalisation

@ti.func
def normalise_LI(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.vector(3, ti.f32),
    cost: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in left invariant coordinates, to 1 with 
    respect to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.vector(n=3, dtype=[float]) of constants of diagonal metric
          tensor with respect to left invariant basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm_LI(vec, G, cost)

@ti.func
def norm_LI(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.vector(3, ti.f32),
    cost: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in left invariant coordinates with
    respect to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.vector(n=3, dtype=[float]) of constants of diagonal metric
          tensor with respect to left invariant basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        Norm of `vec`.
    """
    c_1, c_2, c_3 = vec[0], vec[1], vec[2]
    return ti.math.sqrt(
            G[0] * c_1**2 +
            G[1] * c_2**2 +
            G[2] * c_3**2
    ) * cost

@ti.func
def normalise_static(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.vector(3, ti.f32),
    cost: ti.f32,
    α: ti.f32,
    φ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in static coordinates, to 1 with respect to the 
    left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.vector(n=3, dtype=[float]) of constants of diagonal metric
          tensor with respect to left invariant basis.
        `cost`: cost function at point, taking values between 0 and 1.
        `α`: α-coordinate of corresponding point on the manifold.
        `φ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm_static(vec, G, cost, α, φ)

@ti.func
def norm_static(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.vector(3, ti.f32),
    cost: ti.f32,
    α: ti.f32,
    φ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in static coordinates with respect to 
    the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.vector(n=3, dtype=[float]) of constants of diagonal metric
          tensor with respect to left invariant basis.
        `cost`: cost function at point, taking values between 0 and 1.
        `α`: α-coordinate of corresponding point on the manifold.
        `φ`: angle coordinate of corresponding point on the manifold.

    Returns:
        Norm of `vec`.
    """

    cosα = ti.math.cos(α)
    sinα = ti.math.sin(α)
    cosφ = ti.math.cos(φ)
    sinφ = ti.math.sin(φ)

    a_1, a_2, a_3 = vec[0], vec[1], vec[2]
    c_1 = a_1 * cosφ + a_2 * sinφ * cosα
    c_2 = -a_1 * sinφ + a_2 * cosφ * cosα
    c_3 = -a_2 * sinα + a_3
    return ti.math.sqrt(
            G[0] * c_1**2 +
            G[1] * c_2**2 +
            G[2] * c_3**2
    ) * cost