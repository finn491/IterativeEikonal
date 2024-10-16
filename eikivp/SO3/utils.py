"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used with all
    controllers on SO(3).
"""

import numpy as np
import taichi as ti
from eikivp.utils import linear_interpolate

# Safe Indexing

@ti.func
def sanitize_index(
    index: ti.types.vector(3, ti.i32),
    input: ti.template()
) -> ti.types.vector(3, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Copied from Gijs
    Bellaard.

    Args:
        `index`: ti.types.vector(n=3, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=3, dtype=ti.i32) of index that is within `input`.
    """
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
        ti.math.mod(index[2], shape[2])
    ], dt=ti.i32)

# Interpolate

@ti.func
def trilinear_interpolate(
    v000: ti.f32, 
    v001: ti.f32, 
    v010: ti.f32, 
    v011: ti.f32, 
    v100: ti.f32, 
    v101: ti.f32, 
    v110: ti.f32, 
    v111: ti.f32, 
    r: ti.types.vector(3, ti.i32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v***` depending on the distance `r`, via 
    repeated linear interpolation (α, β, φ). Adapted from Gijs Bellaard.

    Args:
        `v***`: values at points between which we want to interpolate, taking 
          real values.
        `r`: ti.types.vector(n=3, dtype=ti.f32) defining the distance to the
          points between which we to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
        Interpolated value.
    """
    v00 = linear_interpolate(v000, v100, r[0])
    v01 = linear_interpolate(v001, v101, r[0])
    v10 = linear_interpolate(v010, v110, r[0])
    v11 = linear_interpolate(v011, v111, r[0])

    v0 = linear_interpolate(v00, v10, r[1])
    v1 = linear_interpolate(v01, v11, r[1])

    v = linear_interpolate(v0, v1, r[2])

    return v

@ti.func
def scalar_trilinear_interpolate(
    input: ti.template(), 
    index: ti.types.vector(3, ti.f32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of `input` at continuous `index` trilinearly, via repeated
    linear interpolation (α, β, φ). Copied from Gijs Bellaard.

    Args:
        `input`: ti.field(dtype=[float]) in which we want to interpolate.
        `index`: ti.types.vector(n=3, dtype=ti.f32) continuous index at which we 
          want to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
    """
    r = ti.math.fract(index)

    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, input)

    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, input)
    
    v000 = input[f[0], f[1], f[2]]
    v001 = input[f[0], f[1], c[2]]
    v010 = input[f[0], c[1], f[2]]
    v011 = input[f[0], c[1], c[2]]
    v100 = input[c[0], f[1], f[2]]
    v101 = input[c[0], f[1], c[2]]
    v110 = input[c[0], c[1], f[2]]
    v111 = input[c[0], c[1], c[2]]

    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)

    return v

# Distancemap

def get_boundary_conditions(source_point):
    """
    Determine the boundary conditions from `source_point`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    i_0, j_0, k_0 = source_point
    boundarypoints_np = np.array([[i_0 + 1, j_0 + 1, k_0]], dtype=int) # Account for padding.
    boundaryvalues_np = np.array([0.], dtype=float)
    boundarypoints = ti.Vector.field(n=3, dtype=ti.i32, shape=1)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=1, dtype=ti.f32)
    boundaryvalues.from_numpy(boundaryvalues_np)
    return boundarypoints, boundaryvalues

@ti.kernel
def field_abs_max(
    scalar_field: ti.template()
) -> ti.f32:
    """
    @taichi.kernel

    Find the largest absolute value in `scalar_field`.

    Args:
        static: ti.field(dtype=[float], shape=shape) of 3D scalar field.

    Returns:
        Largest absolute value in `scalar_field`.
    """
    value = ti.abs(scalar_field[0, 0, 0])
    for I in ti.grouped(scalar_field):
        ti.atomic_max(value, ti.abs(scalar_field[I]))
    return value

def check_convergence(dW_dt, source_point, tol=1e-3, target_point=None):
    """
    Check whether the IVP method has converged by comparing the Hamiltonian
    `dW_dt` to tolerance `tol`. If `target_point` is provided, only check
    convergence at `target_point`; otherwise check throughout the domain.
    """
    if target_point is None:
        dW_dt[source_point[0]+1, source_point[1]+1, source_point[2]] = 0. # Source is fixed.
        error = field_abs_max(dW_dt)
    else:
        error = ti.abs(dW_dt[target_point])
    print(error)
    is_converged = error < tol
    return is_converged


# Backtracking

# @ti.func
# def get_next_point(
#     point: ti.types.vector(n=3, dtype=ti.f32),
#     gradient_at_point: ti.types.vector(n=3, dtype=ti.f32),
#     dt: ti.f32
# ) -> ti.types.vector(n=3, dtype=ti.f32):
#     """
#     @taichi.func

#     Compute the next point in the gradient descent.

#     Args:
#         `point`: ti.types.vector(n=2, dtype=[float]) coordinates of current 
#           point.
#         `gradient_at_point`: ti.types.vector(n=2, dtype=[float]) value of 
#           gradient at current point.
#         `dt`: Gradient descent step size, taking values greater than 0.

#     Returns:
#         Next point in the gradient descent.
#     """
#     new_point = ti.Vector([0., 0., 0.], dt=ti.f32)
#     new_point[0] = point[0] - dt * gradient_at_point[0]
#     new_point[1] = point[1] - dt * gradient_at_point[1]
#     new_point[2] = point[2] - dt * gradient_at_point[2]
#     return new_point

@ti.func
def get_next_point(
    point: ti.types.vector(n=3, dtype=ti.f32),
    gradient_at_point: ti.types.vector(n=3, dtype=ti.f32),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    dt: ti.f32
) -> ti.types.vector(n=3, dtype=ti.f32):
    """
    @taichi.func

    Compute the next point in the gradient descent.

    Args:
        `point`: ti.types.vector(n=2, dtype=[float]) coordinates of current 
          point.
        `gradient_at_point`: ti.types.vector(n=2, dtype=[float]) value of 
          gradient at current point.
        `dt`: Gradient descent step size, taking values greater than 0.

    Returns:
        Next point in the gradient descent.
    """
    new_point = ti.Vector([0., 0., 0.], dt=ti.f32)
    gradient_norm_l2 = norm_l2(gradient_at_point, dα, dβ, dφ)
    new_point[0] = point[0] - dt * gradient_at_point[0] / gradient_norm_l2
    new_point[1] = point[1] - dt * gradient_at_point[1] / gradient_norm_l2
    new_point[2] = point[2] - dt * gradient_at_point[2] / gradient_norm_l2
    return new_point

@ti.func
def norm_l2(
    vec: ti.types.vector(3, ti.f32),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the Euclidean norm of `vec` represented in the left invariant frame.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.

    Returns:
        Norm of `vec`.
    """
    return ti.math.sqrt(
            (vec[0] / dα)**2 +
            (vec[1] / dβ)**2 +
            (vec[2] / dφ)**2
    )

@ti.func
def distance_in_pixels(
    distance: ti.types.vector(3, ti.f32),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the distance in pixels given the difference in coordinates and the
    pixel size.

    Args:
        `distance`: ti.types.vector(n=3, dtype=[float]) difference in
          coordinates.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
    """
    return ti.math.sqrt(
        (distance[0] / dα)**2 + 
        (distance[1] / dβ)**2 + 
        (mod_offset(distance[2], 2 * ti.math.pi, -ti.math.pi) / dφ)**2
    )

@ti.func
def mod_offset(
    x: ti.f32,
    period: ti.f32,
    offset: ti.f32,
) -> ti.f32:
    return x - (x - offset)//period * period

# Coordinate Transforms

@ti.func
def vectorfield_LI_to_static(
    vectorfield_LI: ti.template(),
    αs: ti.template(),
    φs: ti.template(),
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
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in
          static coordinates.
    """
    for I in ti.grouped(vectorfield_LI):
        vectorfield_static[I] = vector_LI_to_static(vectorfield_LI[I], αs[I], φs[I])

@ti.func
def vector_LI_to_static(
    vector_LI: ti.types.vector(3, ti.f32),
    α: ti.f32,
    φ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Change the coordinates of the vector represented by `vector_LI` from the 
    left invariant to the static frame, given that the angle coordinate of the 
    point on the manifold corresponding to this vector is φ.

    Args:
      Static:
        `vector_LI`: ti.Vector(n=3, dtype=[float]) represented in LI
        coordinates.
        `α`: α-coordinate of corresponding point on the manifold.
        `φ`: angle coordinate of corresponding point on the manifold.
    """
    # Find coordinates w.r.t. static frame by plugging vector into corresponding
    # dual basis vectors.

    # B1 = [cos(φ),sin(φ)/cos(α),sin(φ)tan(α)]
    # B2 = [-sin(φ),cos(φ)/cos(α),cos(φ)tan(α)]
    # B3 = [0,0,1]

    # ν1 = [cos(φ),sin(φ)cos(α),0]
    # ν2 = [-sin(φ),cos(φ)cos(α),0]
    # ν3 = [0,-sin(α),1]

    cosα = ti.math.cos(α)
    tanα = ti.math.tan(α)
    cosφ = ti.math.cos(φ)
    sinφ = ti.math.sin(φ)

    return ti.Vector([
        vector_LI[0] * cosφ - vector_LI[1] * sinφ,
        vector_LI[0] * sinφ / cosα + vector_LI[1] * cosφ / cosα,
        vector_LI[0] * sinφ * tanα + vector_LI[1] * cosφ * tanα + vector_LI[2]
    ], dt=ti.f32)

@ti.func
def vectorfield_static_to_LI(
    vectorfield_static: ti.template(),
    αs: ti.template(),
    φs: ti.template(),
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
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in
          LI coordinates.
    """
    for I in ti.grouped(vectorfield_static):
        vectorfield_static[I] = vector_static_to_LI(vectorfield_LI[I], αs[I], φs[I])

@ti.func
def vector_static_to_LI(
    vector_static: ti.types.vector(3, ti.f32),
    α: ti.f32,
    φ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Change the coordinates of the vector represented by `vector_static` from the 
    left invariant to the static frame, given that the angle coordinate of the 
    point on the manifold corresponding to this vector is θ.

    Args:
      Static:
        `vector_static`: ti.Vector(n=3, dtype=[float]) represented in static
        coordinates.
        `α`: α-coordinate of corresponding point on the manifold.
        `φ`: angle coordinate of corresponding point on the manifold.
    """
    # Find coordinates w.r.t. left invariant frame by plugging vector into
    # corresponding dual basis vectors.

    # B1 = [cos(φ),sin(φ)/cos(α),sin(φ)tan(α)]
    # B2 = [-sin(φ),cos(φ)/cos(α),cos(φ)tan(α)]
    # B3 = [0,0,1]

    # ν1 = [cos(φ),sin(φ)cos(α),0]
    # ν2 = [-sin(φ),cos(φ)cos(α),0]
    # ν3 = [0,-sin(α),1]

    cosα = ti.math.cos(α)
    sinα = ti.math.sin(α)
    cosφ = ti.math.cos(φ)
    sinφ = ti.math.sin(φ)

    return ti.Vector([
        vector_static[0] * cosφ + vector_static[1] * sinφ * cosα,
        -vector_static[0] * sinφ + vector_static[1] * cosφ * cosα,
        -vector_static[1] * sinα + vector_static[2]
    ], dt=ti.f32)


def coordinate_real_to_array(α, β, φ, α_min, β_min, φ_min, dα, dβ, dφ):
    """
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    (`α`, `β`, `φ`). Can broadcast over entire arrays of real coordinates.

    Args:
        `α`: α-coordinate of the point.
        `β`: β-coordinate of the point.
        `φ`: φ-coordinate of the point.
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
    """
    I = np.rint((α - α_min) / dα).astype(int)
    J = np.rint((β - β_min) / dβ).astype(int)
    K = np.rint((φ - φ_min) / dφ).astype(int)
    return I, J, K

@ti.func
def coordinate_real_to_array_ti(
    point: ti.types.vector(3, ti.f32),
    α_min: ti.f32,
    β_min: ti.f32,
    φ_min: ti.f32,
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func
    
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    `point`. Can broadcast over entire arrays of real coordinates.

    Args:
        `point`: vector of α-, β-, and φ-coordinates of the point.
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
    """
    I = (point[0] - α_min) / dα
    J = (point[1] - β_min) / dβ
    K = (point[2] - φ_min) / dφ
    return ti.Vector([I, J, K], dt=ti.f32)


def coordinate_array_to_real(I, J, K, α_min, β_min, φ_min, dα, dβ, dφ):
    """
    Compute the real coordinates (α, β, φ) of the point defined by array indices 
    (`I`, `J`, `K`). Can broadcast over entire arrays of array indices.

    Args:
        `I`: I index of the point.
        `J`: J index of the point.
        `K`: K index of the point.
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
    """
    α = α_min + I * dα
    β = β_min + J * dβ
    φ = φ_min + K * dφ
    return α, β, φ

def align_to_real_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to standard array convention,
    in terms of indices with respect to arrays aligned with real axes (see Notes
    for more explanation). Here, `shape` gives the shape of the array in which
    we index _after_ aligning with real axes, so [Nα, Nβ, Nφ].

    Args:
        `point`: Tuple[int, int, int] describing point with respect to standard
          array indexing convention.
        `shape`: shape of array, aligned to real axes, in which we want to
          index. Note that `0 <= point[0] <= shape[0] - 1`, 
          `0 <= point[1] <= shape[1] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up (i.e. in the positive α-direction), you do so by decreasing I,
        while if you want to move a single pixel to the left (i.e. in the
        positive β-direction), you do so by decreasing J. Hence, the shape of
        the array is [Nβ, Nα, Nφ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing I, and moving left a single pixel is achieved by increasing
        J. Hence, the shape of the array is [Nα, Nβ, Nφ].

        Alignment is achieved by flipping the array twice.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I α ------
            | | |    |        =>          | | |    |
            v α ------                    v v ------
                 <- β                          β ->
                 J ->                          J ->  
    """
    return shape[0] - 1 - point[0], shape[1] - 1 - point[1], point[2]

def align_to_real_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `field`: np.ndarray of scalar field given with respect to standard array
          convention.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up (i.e. in the positive α-direction), you do so by decreasing I,
        while if you want to move a single pixel to the left (i.e. in the
        positive β-direction), you do so by decreasing J. Hence, the shape of
        the array is [Nβ, Nα, Nφ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing I, and moving left a single pixel is achieved by increasing
        J. Hence, the shape of the array is [Nα, Nβ, Nφ].

        Alignment is achieved by flipping the array twice.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I α ------
            | | |    |        =>          | | |    |
            v α ------                    v v ------
                 <- β                          β ->
                 J ->                          J ->  
    """
    field_aligned = np.flip(np.flip(field, axis=0), axis=1)
    return field_aligned

def align_to_real_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in indices with respect to standard array 
    convention, with real axes (see Notes for more explanation).
    
    Args:
        `vector_field`: np.ndarray of vector field given with respect to 
          standard array convention.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up (i.e. in the positive α-direction), you do so by decreasing I,
        while if you want to move a single pixel to the left (i.e. in the
        positive β-direction), you do so by decreasing J. Hence, the shape of
        the array is [Nβ, Nα, Nφ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing I, and moving left a single pixel is achieved by increasing
        J. Hence, the shape of the array is [Nα, Nβ, Nφ].

        Alignment is achieved by flipping the array twice.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I α ------
            | | |    |        =>          | | |    |
            v α ------                    v v ------
                 <- β                          β ->
                 J ->                          J ->   
    """
    vector_field_aligned = np.flip(np.flip(vector_field, axis=0), axis=1)
    return vector_field_aligned

def align_to_standard_array_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to arrays aligned with real 
    axes, in terms of indices with respect to standard array convention, (see 
    Notes for more explanation). Here, `shape` gives the shape of the array in 
    which we index _after_ aligning with standard array convention, so
    [Nβ, Nα, Nφ].

    Args:
        `point`: Tuple[int, int] describing point with respect to arrays aligned
          with real axes.
        `shape`: shape of array, with respect to standard array convention, in 
          which we want to index. Note that `0 <= point[0] <= shape[0] - 1`, 
          `0 <= point[1] <= shape[1] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up (i.e. in the positive α-direction), you do so by decreasing I,
        while if you want to move a single pixel to the left (i.e. in the
        positive β-direction), you do so by decreasing J. Hence, the shape of
        the array is [Nβ, Nα, Nφ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing I, and moving left a single pixel is achieved by increasing
        J. Hence, the shape of the array is [Nα, Nβ, Nφ].

        Alignment is achieved by flipping the array twice.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I α ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v α ------
                 β ->                          <- β
                 J ->                          J -> 
    """
    return shape[0] - 1 - point[0], shape[1] - 1 - point[1], point[2]

def align_to_standard_array_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to arrays aligned with real
    axes, with respect to standard array convention (see Notes for more 
    explanation).

    Args:
        `field`: np.ndarray of scalar field given in indices with respect to
          arrays aligned with real axes.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up (i.e. in the positive α-direction), you do so by decreasing I,
        while if you want to move a single pixel to the left (i.e. in the
        positive β-direction), you do so by decreasing J. Hence, the shape of
        the array is [Nβ, Nα, Nφ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing I, and moving left a single pixel is achieved by increasing
        J. Hence, the shape of the array is [Nα, Nβ, Nφ].

        Alignment is achieved by flipping the array twice.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I α ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v α ------
                 β ->                          <- β
                 J ->                          J -> 
    """
    field_aligned = np.flip(np.flip(field, axis=1), axis=0)
    return field_aligned

def align_to_standard_array_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `vector_field`: np.ndarray of vector field given in indices with respect
          to arrays aligned with real axes.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up (i.e. in the positive α-direction), you do so by decreasing I,
        while if you want to move a single pixel to the left (i.e. in the
        positive β-direction), you do so by decreasing J. Hence, the shape of
        the array is [Nβ, Nα, Nφ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing I, and moving left a single pixel is achieved by increasing
        J. Hence, the shape of the array is [Nα, Nβ, Nφ].

        Alignment is achieved by flipping the array twice.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I α ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v α ------
                 β ->                          <- β
                 J ->                          J -> 
    """
    vector_field_aligned = np.flip(np.flip(vector_field, axis=1), axis=0)
    return vector_field_aligned


# Maps from SO(3) into SE(2).
# Π_forward: SO(3) -> SE(2), (α, β, φ) |-> (x, y, θ)
# To get the angle θ from φ, we consider what happens along horizontal curves:
#   θ(t) = arg(dx_dt(t) + i dy_dt(t)).
# We can now find that
#   θ(t) = arg((dx_dα dα_dt(t) + dx_dβ dβ_dt(t)) + i (dy_dα dα_dt(t) + dy_dβ dβ_dt(t))).
# On horizontal curves it holds that dα_dt = cos(φ) and dβ_dt = sin(φ) / cos(α);
# to find the other derivatives we simply need to differentiate
# π_forward: S2 -> R2, (α, β) |-> (x, y), which can be derived from relatively
# straightforward geometric arguments.

def Π_forward_np(α, β, φ, a, c):
    """
    Map coordinates in SO(3) into SE(2), by projecting down from the sphere onto
    a plane.
    
    Args:
        `α`: α-coordinate.
        `β`: β-coordinate.
        `φ`: φ-coordinate.
        `a`: Distance between nodal point of projection and centre of sphere.
        `c`: Distance between projection plane and centre of sphere reflected
          around nodal point.

    Returns:
        np.ndarray(shape=(3,)) of coordinates in SE(2).
    """
    # π_forward: S2 -> R2
    cosα = np.cos(α)
    sinα = np.sin(α)
    cosβ = np.cos(β)
    sinβ = np.sin(β)

    x = (a + c) * sinα / (a + cosα * cosβ)
    y = (a + c) * cosα * sinβ / (a + cosα * cosβ)

    # Partial derivatives, up to proportionality constant
    # (a + c) / (a + cosα * cosβ)**2, which does not influence the angle
    dπ_forward_x_dα = a * cosα + cosβ
    dπ_forward_x_dβ = cosα * sinα * sinβ
    dπ_forward_y_dα = -a * sinα * sinβ
    dπ_forward_y_dβ = a * cosα * cosβ + cosα**2
    
    # Combine into Π_forward: SO(3) -> SE(2)
    cosφ = np.cos(φ)
    sinφ = np.sin(φ)

    dα = cosφ
    dβ = sinφ / cosα

    θ = np.arctan2( # y, x
        dπ_forward_y_dα * dα + dπ_forward_y_dβ * dβ,
        dπ_forward_x_dα * dα + dπ_forward_x_dβ * dβ
    )

    return ti.Vector([x, y, θ], dt=ti.f32)

@ti.func
def Π_forward(
    α: ti.f32,
    β: ti.f32,
    φ: ti.f32,
    a: ti.f32,
    c: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func
    
    Map coordinates in SO(3) into SE(2), by projecting down from the sphere onto
    a plane.
    
    Args:
        `α`: α-coordinate.
        `β`: β-coordinate.
        `φ`: φ-coordinate.
        `a`: Distance between nodal point of projection and centre of sphere.
        `c`: Distance between projection plane and centre of sphere reflected
          around nodal point.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of coordinates in SE(2).
    """
    # π_forward: S2 -> R2
    cosα = ti.math.cos(α)
    sinα = ti.math.sin(α)
    cosβ = ti.math.cos(β)
    sinβ = ti.math.sin(β)

    x = (a + c) * sinα / (a + cosα * cosβ)
    y = (a + c) * cosα * sinβ / (a + cosα * cosβ)

    # Partial derivatives, up to proportionality constant
    # (a + c) / (a + cosα * cosβ)**2, which does not influence the angle
    dπ_forward_x_dα = a * cosα + cosβ
    dπ_forward_x_dβ = cosα * sinα * sinβ
    dπ_forward_y_dα = -a * sinα * sinβ
    dπ_forward_y_dβ = a * cosα * cosβ + cosα**2
    
    # Combine into Π_forward: SO(3) -> SE(2)
    cosφ = ti.math.cos(φ)
    sinφ = ti.math.sin(φ)

    dα = cosφ
    dβ = sinφ / cosα

    θ = ti.math.atan2( # y, x
        dπ_forward_y_dα * dα + dπ_forward_y_dβ * dβ,
        dπ_forward_x_dα * dα + dπ_forward_x_dβ * dβ
    )

    return ti.Vector([x, y, θ], dt=ti.f32)

# For the backward map which we don't need for interpolation.
@ti.func
def Π_backward(
    x: ti.f32,
    y: ti.f32,
    θ: ti.f32,
    a: ti.f32,
    c: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func
    
    Map coordinates in SE(2) into SO(3), by projecting down from the sphere onto
    a plane.
    
    Args:
        `x`: x-coordinate.
        `y`: y-coordinate.
        `θ`: θ-coordinate.
        `a`: Distance between nodal point of projection and centre of sphere.
        `c`: Distance between projection plane and centre of sphere reflected
          around nodal point.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of coordinates in SO(3).
    """
    # π_forward: R2 -> S2
    p1 = p_1(x, y, a, c)
    poverline = p_overline(x, y, a, c)

    α = ti.math.asin(x * poverline)
    β = ti.math.atan2(y * poverline, p1)

    # Partial derivatives, up to proportionality constant
    # (a + c) / (a + cosα * cosβ)**2, which does not influence the angle
    dπ_forward_x_dα = a * cosα + cosβ
    dπ_forward_x_dβ = cosα * sinα * sinβ
    dπ_forward_y_dα = -a * sinα * sinβ
    dπ_forward_y_dβ = a * cosα * cosβ + cosα**2
    
    # Combine into Π_forward: SO(3) -> SE(2)
    cosφ = ti.math.cos(φ)
    sinφ = ti.math.sin(φ)

    dα = cosφ
    dβ = sinφ / cosα

    θ = ti.math.atan2( # y, x
        dπ_forward_y_dα * dα + dπ_forward_y_dβ * dβ,
        dπ_forward_x_dα * dα + dπ_forward_x_dβ * dβ
    )

    return ti.Vector([x, y, θ], dt=ti.f32)

# @ti.func
# def π_backward(
#     x: ti.f32,
#     y: ti.f32,
#     a: ti.f32,
#     c: ti.f32
# ) -> ti.types.vector(2, ti.f32):
#     """
#     @taichi.func
    
#     Map coordinates in R^2 into S^2.
    
#     Args:
#         `x`: x-coordinate.
#         `y`: y-coordinate.
#         `a`: Distance between nodal point of projection and centre of sphere.
#         `c`: Distance between projection plane and centre of sphere reflected
#           around nodal point.

#     Returns:
#         ti.types.vector(n=2, dtype=[float]) of coordinates in S^2.
#     """
#     p1 = p_1(x, y, a, c)
#     poverline = p_overline(x, y, a, c)

#     α = ti.math.asin(x * poverline)
#     β = ti.math.atan2(y * poverline, p1)

#     return ti.Vector([α, β], dt=ti.f32)

@ti.func
def p_overline(
    x: ti.f32,
    y: ti.f32,
    a: ti.f32,
    c: ti.f32
) -> ti.f32:
    """"""
    return (
        a * (a + c) * ti.math.sqrt((x**2 + y**2) * (1 - a**2) + (a + c)**2) / 
        ((x**2 + y**2) + (a + c)**2)
    )

@ti.func
def p_1(
    x: ti.f32,
    y: ti.f32,
    a: ti.f32,
    c: ti.f32
) -> ti.f32:
    """"""
    return (
        ((a + c) * ti.math.sqrt((x**2 + y**2) * (1 - a**2) + (a + c)**2) - a * (x**2 + y**2)) /
        ((x**2 + y**2) + (a + c)**2)
    )

# 😢
@ti.func
def dπ_backward1_dx(
    x: ti.f32,
    y: ti.f32,
    a: ti.f32,
    c: ti.f32
) -> ti.f32:
    """"""
    t1 = 1 - a**2
    t2 = a + c
    t3 = x**2 + y**2
    t4 = t2**2 + t3
    t5 = ti.math.sqrt(t2**2 + t1 * t3)
    return (
        (
            (t1 * x**2 * t4) / t5 +
            (t4 - 2 * x**2) * (a * t2 + t5)
        ) /
        (
            t4 * ti.math.sqrt(
                t4**2 - x**2 * (a * t2 + t5)
            )
        )
    )

@ti.func
def dπ_backward2_dx(
    x: ti.f32,
    y: ti.f32,
    a: ti.f32,
    c: ti.f32
) -> ti.f32:
    """"""
    t1 = (1 - a**2)
    t2 = a + c
    t3 = x**2 + y**2
    t4 = t2**2 + t3
    t5 = ti.math.sqrt(t2 + t1 * t3)
    return (
        (
            (t1 * x**2 * t4) / t5 +
            (t4 - 2 * x**2) * (a * t2 + t5)
        ) /
        (
            t4 * ti.math.sqrt(
                t4**2 - x**2 * (a * t2 + t5)
            )
        )
    )