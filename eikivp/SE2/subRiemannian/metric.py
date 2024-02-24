"""
    metric
    ======

    Provides tools to deal with metrics SE(2). The primary methods are:
      1. `invert_metric`: compute the matrix defining the dual metric from the
      matrix defining the primal metric.
      2. `normalise_LI`: normalise a vector, given with respect to the left
      invariant frame, to norm 1 with respect to some data-driven metric.
      3. `norm_LI`: compute the norm of a vector, given with respect to the left
      invariant frame, with respect to some data-driven metric.
      4. `normalise_static`: normalise a vector, given with respect to the
      static frame, to norm 1 with respect to some data-driven metric.
      5. `norm_static`: compute the norm of a vector, given with respect to the
      static frame, with respect to some data-driven metric.
      6. `vector_static_to_LI`: compute the components of a vector, given with
      respect to the static frame, in the left invariant frame.
      7. `vectorfield_static_to_LI`: compute the components of a vectorfield,
      given with respect to the static frame, in the left invariant frame.
      8. `vector_LI_to_static`: compute the components of a vector, given with
      respect to the left invariant frame, in the left static frame.
      9. `vectorfield_LI_to_static`: compute the components of a vectorfield,
      given with respect to the left invariant, in the static frame.
    
    Additionally, we have numerous functions to reorder arrays to align either
    with the standard array indexing conventions or with the real axes.
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
    respect to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.
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
    respect to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        Norm of `vec`.
    """
    c_1, c_2, c_3 = vec[0], vec[1], vec[2]
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
    left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.
        `cost`: cost function at point, taking values between 0 and 1.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    # Can do this but it's not necessary
    # vec_LI = vector_LI_to_static(vec, θ)
    # vec_normalised_LI = normalise_LI(vec_LI, G_inv)
    # vec_normalised = vector_static_to_LI(vec_normalised_LI, θ)
    # return vec_normalised
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
    the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.
        `cost`: cost function at point, taking values between 0 and 1.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        Norm of `vec`.
    """
    a_1, a_2, a_3 = vec[0], vec[1], vec[2]
    c_1 = a_1 * ti.math.cos(θ) + a_2 * ti.math.sin(θ)
    c_2 = -a_1 * ti.math.sin(θ) + a_2 * ti.math.cos(θ)
    c_3 = a_3
    return ti.math.sqrt(
            c_1**2 * ξ**2 +
            c_3**2
    ) * cost