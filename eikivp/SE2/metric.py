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
    G: ti.types.matrix(3, 3, ti.f32),
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
    return vec / norm_LI(vec, G, cost)

@ti.func
def norm_LI(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.matrix(3, 3, ti.f32),
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
            1 * G[0, 0] * c_1 * c_1 +
            2 * G[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            2 * G[0, 2] * c_1 * c_3 +
            1 * G[1, 1] * c_2 * c_2 +
            2 * G[1, 2] * c_2 * c_3 +
            1 * G[2, 2] * c_3 * c_3
    ) * cost

@ti.func
def normalise_static(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.matrix(3, 3, ti.f32),
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
    return vec / norm_static(vec, G, cost, θ)

@ti.func
def norm_static(
    vec: ti.types.vector(3, ti.f32),
    G: ti.types.matrix(3, 3, ti.f32),
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
            1 * G[0, 0] * c_1 * c_1 +
            2 * G[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            2 * G[0, 2] * c_1 * c_3 +
            1 * G[1, 1] * c_2 * c_2 +
            2 * G[1, 2] * c_2 * c_3 +
            1 * G[2, 2] * c_3 * c_3
    ) * cost


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


def coordinate_real_to_array(x, y, θ, x_min, y_max, θ_min, dxy, dθ):
    """
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    (`x`, `y`, `θ`). Can broadcast over entire arrays of real coordinates.

    Args:
        `x`: x-coordinate of the point.
        `y`: y-coordinate of the point.
        `θ`: θ-coordinate of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_max`: maximum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    J = np.rint((x - x_min) / dxy).astype(int)
    I = np.rint((y_max - y) / dxy).astype(int)
    K = np.rint((θ - θ_min) / dθ).astype(int)
    return I, J, K


def coordinate_array_to_real(I, J, K, x_min, y_max, θ_min, dxy, dθ):
    """
    Compute the real coordinates (x, y, θ) of the point defined by array indices 
    (`I`, `J`, `K`). Can broadcast over entire arrays of array indices.

    Args:
        `I`: I index of the point.
        `J`: J index of the point.
        `K`: K index of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    x = x_min + J * dxy
    y = y_max - I * dxy
    θ = θ_min + K * dθ
    return x, y, θ

def align_to_real_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to standard array convention,
    in terms of indices with respect to arrays aligned with real axes (see Notes
    for more explanation). Here, `shape` gives the shape of the array in which
    we index _after_ aligning with real axes.

    Args:
        `point`: Tuple[int, int, int] describing point with respect to standard
          array indexing convention.
        `shape`: shape of array, aligned to real axes, in which we want to
          index. Note that `0 <= point[0] <= shape[1] - 1`, 
          `0 <= point[1] <= shape[0]`, and `0 <= point[2] <= shape[2]`.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0], point[2]

def align_to_real_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `field`: np.ndarray of scalar field given with respect to standard array
          convention.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    field_flipped = np.flip(field, axis=0)
    # field_aligned = np.transpose(field_flipped, axes=(1, 0, 2))
    field_aligned = field_flipped.swapaxes(1, 0)
    return field_aligned

def align_to_real_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in indices with respect to standard array 
    convention, with real axes (see Notes for more explanation).
    
    Args:
        `vector_field`: np.ndarray of vector field given with respect to 
          standard array convention.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    vector_field_flipped = np.flip(vector_field, axis=0)
    # vector_field_aligned = np.transpose(vector_field_flipped, axes=(1, 0, 2, 3))
    vector_field_aligned = vector_field_flipped.swapaxes(1, 0)
    return vector_field_aligned

def align_to_standard_array_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to arrays aligned with real 
    axes, in terms of indices with respect to standard array convention, (see 
    Notes for more explanation). Here, `shape` gives the shape of the array in 
    which we index _after_ aligning with standard array convention.

    Args:
        `point`: Tuple[int, int] describing point with respect to arrays aligned
          with real axes.
        `shape`: shape of array, with respect to standard array convention, in 
          which we want to index. Note that `0 <= point[0] <= shape[1] - 1`, 
          `0 <= point[1] <= shape[0]`, and `0 <= point[2] <= shape[2]`.

    Notes:
        Alignment is achieved by first transposing and subsequently flipping the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0], point[2]

def align_to_standard_array_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to arrays aligned with real
    axes, with respect to standard array convention (see Notes for more 
    explanation).

    Args:
        `field`: np.ndarray of scalar field given in indices with respect to
          arrays aligned with real axes.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    # field_transposed = np.transpose(field, axes=(1, 0, 2))
    field_transposed = field.swapaxes(1, 0)
    field_aligned = np.flip(field_transposed, axis=0)
    return field_aligned

def align_to_standard_array_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `vector_field`: np.ndarray of vector field given in indices with respect
          to arrays aligned with real axes.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    # vector_field_transposed = np.transpose(vector_field, axes=(1, 0, 2, 3))
    vector_field_transposed = vector_field.swapaxes(1, 0)
    vector_field_aligned = np.flip(vector_field_transposed, axis=0)
    return vector_field_aligned

# Apparently shouldn't do this...
# @ti.func
# def vector_standard_to_array(
#     vector_standard: ti.types.vector(3, ti.f32),
#     dxy: ti.f32,
#     dθ: ti.f32
# ) -> ti.types.vector(3, ti.f32):
#     """
#     @taichi.func

#     Change the coordinates of the vector represented by `vector_standard` from 
#     the standard frame to the frame of array indices, given that the spatial and
#     angular resolutions are `dxy` and `dθ`, respectively.

#     Args:
#       Static:
#         `vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in LI
#           coordinates.
#         `dxy`: Spatial resolution, taking values greater than 0.
#         `dθ`: Angular resolution, taking values greater than 0.
#     """
#     return ti.Vector([
#         vector_standard[0] / dxy,
#         vector_standard[1] / dxy,
#         vector_standard[2] / dθ
#     ])