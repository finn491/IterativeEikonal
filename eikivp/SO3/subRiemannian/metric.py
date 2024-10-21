"""
    metric
    ======

    Provides tools to deal with sub-Riemannian metrics on SO(3). The primary
    methods are:
      1. `normalise_LI`: normalise a vector, given with respect to the left
      invariant frame, to norm 1 with respect to some data-driven sub-Riemannian
      metric.
      2. `norm_LI`: compute the norm of a vector, given with respect to the left
      invariant frame, with respect to some data-driven sub-Riemannian metric.
      3. `normalise_static`: normalise a vector, given with respect to the
      static frame, to norm 1 with respect to some data-driven sub-Riemannian
      metric.
      4. `norm_static`: compute the norm of a vector, given with respect to the
      static frame, with respect to some data-driven sub-Riemannian metric.
"""

import taichi as ti

# Normalisation

@ti.func
def normalise_LI(
    vec: ti.types.vector(3, ti.f32),
    ξ: ti.f32,
    cost: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in the left invariant frame, to 1 with respect
    to the left invariant sub-Riemannian metric tensor defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm_LI(vec, ξ, cost)

@ti.func
def norm_LI(
    vec: ti.types.vector(3, ti.f32),
    ξ: ti.f32,
    cost: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in the left invariant frame with
    respect to the left invariant sub-Riemannian metric tensor defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        Norm of `vec`.
    """
    return ti.math.sqrt(
            vec[0]**2 * ξ**2 +
            vec[2]**2
    ) * cost

@ti.func
def normalise_static(
    vec: ti.types.vector(3, ti.f32),
    ξ: ti.f32,
    cost: ti.f32,
    α: ti.f32,
    φ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in the static frame, to 1 with respect to the 
    left invariant sub-Riemannian metric tensor defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0.
        `cost`: cost function at point, taking values between 0 and 1.
        `α`: α-coordinate of corresponding point on the manifold.
        `φ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm_static(vec, ξ, cost, α, φ)

@ti.func
def norm_static(
    vec: ti.types.vector(3, ti.f32),
    ξ: ti.f32,
    cost: ti.f32,
    α: ti.f32,
    φ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in the static frame with respect to 
    the left invariant sub-Riemannian metric tensor defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0.
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
    c_3 = -a_2 * sinα + a_3
    return ti.math.sqrt(
            c_1**2 * ξ**2 +
            c_3**2
    ) * cost