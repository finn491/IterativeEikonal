"""
    metric
    ======

    Provides tools to deal with Finsler functions on SE(2). The primary methods
    are:
      1. `normalise_LI`: normalise a vector, given with respect to the left
      invariant frame, to norm 1 with respect to some data-driven Finsler
      function.
      2. `norm_LI`: compute the norm of a vector, given with respect to the left
      invariant frame, with respect to some data-driven Finsler function.
      3. `normalise_static`: normalise a vector, given with respect to the
      static frame, to norm 1 with respect to some data-driven Finsler function.
      4. `norm_static`: compute the norm of a vector, given with respect to the
      static frame, with respect to some data-driven Finsler function.
      5. `vector_static_to_LI`: compute the components of a vector, given with
      respect to the static frame, in the left invariant frame.
      6. `vectorfield_static_to_LI`: compute the components of a vectorfield,
      given with respect to the static frame, in the left invariant frame.
      7. `vector_LI_to_static`: compute the components of a vector, given with
      respect to the left invariant frame, in the left static frame.
      8. `vectorfield_LI_to_static`: compute the components of a vectorfield,
      given with respect to the left invariant, in the static frame.
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

    Normalise `vec`, represented in left invariant coordinates, to 1 with 
    respect to the left invariant Finsler function defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
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

    Compute the norm of `vec` represented in left invariant coordinates with
    respect to the left invariant Finsler function defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
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
    θ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in static coordinates, to 1 with respect to the 
    left invariant Finsler function defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `cost`: cost function at point, taking values between 0 and 1.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm_static(vec, ξ, cost, θ)

@ti.func
def norm_static(
    vec: ti.types.vector(3, ti.f32),
    ξ: ti.f32,
    cost: ti.f32,
    θ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in static coordinates with respect to 
    the left invariant Finsler function defined by `ξ`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `cost`: cost function at point, taking values between 0 and 1.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        Norm of `vec`.
    """
    a_1, a_2, a_3 = vec[0], vec[1], vec[2]
    c_1 = a_1 * ti.math.cos(θ) + a_2 * ti.math.sin(θ)
    c_3 = a_3
    return ti.math.sqrt(
            c_1**2 * ξ**2 +
            c_3**2
    ) * cost