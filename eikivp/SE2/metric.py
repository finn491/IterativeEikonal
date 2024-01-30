# metric.py

import taichi as ti
import numpy as np

# Metric Inversion

def invert_metric(G):
    """
    Invert the metric tensor defined by the matrix `G`. If `G` is semidefinite,
    e.g. when we are dealing with a sub-Riemannian metric, the metric is first
    made definite by adding `1e-8` to the diagonal.

    Args:
        `G`: np.ndarray(shape=(3, 3), dtype=[float]) of matrix of left invariant
          metric tensor field with respect to left invariant basis.
    """
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.inv(G + np.identity(3) * 1e-8)
    return G_inv


# Normalisation

@ti.func
def normalise_LI(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.matrix(3, 3, ti.f32)
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in left invariant coordinates, to 1 with 
    respect to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm_LI(vec, G)

@ti.func
def norm_LI(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.matrix(3, 3, ti.f32)
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in left invariant coordinates with
    respect to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.

    Returns:
        Norm of `vec`.
    """
    c_1, c_2, c_3 = vec[0], vec[1], vec[2]
    return ti.math.sqrt(
            1 * G[0, 0] * c_1 * c_1 +
            2 * G[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            2 * G[0, 2] * c_1 * c_3 +
            1 * G[1, 1] * c_2 * c_2 +
            2 * G[1, 2] * c_2 * c_3 +
            1 * G[2, 2] * c_3 * c_3
    )

@ti.func
def normalise_static(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.matrix(3, 3, ti.f32),
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
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    # Can do this but it's not necessary
    # vec_LI = vector_LI_to_static(vec, θ)
    # vec_normalised_LI = normalise_LI(vec_LI, G_inv)
    # vec_normalised = vector_static_to_LI(vec_normalised_LI, θ)
    # return vec_normalised
    return vec / norm_static(vec, G, θ)

@ti.func
def norm_static(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.matrix(3, 3, ti.f32),
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
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        Norm of `vec`.
    """
    a_1, a_2, a_3 = vec[0], vec[1], vec[2]
    c_1 = a_1 * ti.math.cos(θ) + a_2 * ti.math.sin(θ)
    c_2 = -a_1 * ti.math.sin(θ) + a_2 * ti.math.cos(θ)
    c_3 = a_3
    return ti.math.sqrt(
            1 * G[0, 0] * c_1 * c_1 +
            2 * G[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            2 * G[0, 2] * c_1 * c_3 +
            1 * G[1, 1] * c_2 * c_2 +
            2 * G[1, 2] * c_2 * c_3 +
            1 * G[2, 2] * c_3 * c_3
    )


# Coordinate Transforms

@ti.func
def vectorfield_LI_to_static(
    vectorfield_LI: ti.template(),
    θs: ti.template(),
    vectorfield_static: ti.template()
):
    """
    @taichi.func

    Change the coordinates of the vectorfield represented by `vectorfield_LI`
    from the left invariant to the static frame.

    Args:
      Static:
        `vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in LI
          coordinates.
        `θs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in
          static coordinates.
    """
    for I in ti.grouped(vectorfield_LI):
        vectorfield_static[I] = vector_LI_to_static(vectorfield_LI[I], θs[I])

@ti.func
def vector_LI_to_static(
    vector_LI: ti.types.vector(3, ti.f32),
    θ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Change the coordinates of the vector represented by `vector_LI` from the 
    left invariant to the static frame, given that the angle coordinate of the 
    point on the manifold corresponding to this vector is θ.

    Args:
      Static:
        `vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in LI
          coordinates.
        `θ`: angle coordinate of corresponding point on the manifold.
    """
    
    # A1 = [cos(θ),sin(θ),0]
    # A2 = [-sin(θ),cos(θ),0]
    # A3 = [0,0,1]

    return ti.Vector([
        ti.math.cos(θ) * vector_LI[0] - ti.math.sin(θ) * vector_LI[1],
        ti.math.sin(θ) * vector_LI[0] + ti.math.cos(θ) * vector_LI[1],
        vector_LI[2]
    ], dt=ti.f32)

@ti.func
def vectorfield_static_to_LI(
    vectorfield_static: ti.template(),
    θs: ti.template(),
    vectorfield_LI: ti.template()
):
    """
    @taichi.func

    Change the coordinates of the vectorfield represented by 
    `vectorfield_static` from the static to the left invariant frame.

    Args:
      Static:
        `vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in
          static coordinates.
        `θs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in
          LI coordinates.
    """
    for I in ti.grouped(vectorfield_static):
        vectorfield_static[I] = vector_LI_to_static(vectorfield_LI[I], θs[I])

@ti.func
def vector_static_to_LI(
    vector_static: ti.types.vector(3, ti.f32),
    θ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Change the coordinates of the vector represented by `vector_static` from the 
    left invariant to the static frame, given that the angle coordinate of the 
    point on the manifold corresponding to this vector is θ.

    Args:
      Static:
        `vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in 
          static coordinates.
        `θ`: angle coordinate of corresponding point on the manifold.
    """
    return ti.Vector([
        ti.math.cos(θ) * vector_static[0] + ti.math.sin(θ) * vector_static[1],
        -ti.math.sin(θ) * vector_static[0] + ti.math.cos(θ) * vector_static[1],
        vector_static[2]
    ], dt=ti.f32)