"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used with all
    controllers on SE(2).

    TODO: improve documentation
      1. `vector_static_to_LI`: compute the components of a vector, given with
      respect to the static frame, in the left invariant frame.
      2. `vectorfield_static_to_LI`: compute the components of a vectorfield,
      given with respect to the static frame, in the left invariant frame.
      3. `vector_LI_to_static`: compute the components of a vector, given with
      respect to the left invariant frame, in the left static frame.
      4. `vectorfield_LI_to_static`: compute the components of a vectorfield,
      given with respect to the left invariant, in the static frame.
"""

import numpy as np
import taichi as ti
from eikivp.utils import linear_interpolate

# Safe Indexing

# Do we also want to ensure that α and β remain in the correct domain?
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
        value = ti.atomic_max(value, ti.abs(scalar_field[I]))
    return value

def check_convergence(dW_dt, tol=1e-3, target_point=None):
    """
    Check whether the IVP method has converged by comparing the Hamiltonian
    `dW_dt` to tolerance `tol`. If `target_point` is provided, only check
    convergence at `target_point`; otherwise check throughout the domain.
    """
    is_converged = False
    if target_point is None:
        error = field_abs_max(dW_dt)
        print(error)
        is_converged = error < tol
    else:
        error = ti.abs(dW_dt[target_point])
        print(error)
        is_converged = error < tol
    return is_converged


# Backtracking

@ti.func
def get_next_point(
    point: ti.types.vector(n=3, dtype=ti.f32),
    gradient_at_point: ti.types.vector(n=3, dtype=ti.f32),
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
    new_point[0] = point[0] - dt * gradient_at_point[0]
    new_point[1] = point[1] - dt * gradient_at_point[1]
    new_point[2] = point[2] - dt * gradient_at_point[2]
    return new_point

def convert_continuous_indices_to_real_space(γ_ci_np, αs_np, βs_np, φs_np):
    """
    Convert the continuous indices in the geodesic `γ_ci_np` to the 
    corresponding real space coordinates described by `αs_np`, `βs_np`, and
    `φs_np`.
    """
    γ_ci = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_ci_np.shape[0])
    γ_ci.from_numpy(γ_ci_np)
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_ci.shape)

    αs = ti.field(dtype=ti.f32, shape=αs_np.shape)
    αs.from_numpy(αs_np)
    βs = ti.field(dtype=ti.f32, shape=βs_np.shape)
    βs.from_numpy(βs_np)
    φs = ti.field(dtype=ti.f32, shape=φs_np.shape)
    φs.from_numpy(φs_np)

    continuous_indices_to_real(γ_ci, αs, βs, φs, γ)

    return γ.to_numpy()

@ti.kernel
def continuous_indices_to_real(
    γ_ci: ti.template(),
    αs: ti.template(),
    βs: ti.template(),
    φs: ti.template(),
    γ: ti.template()
):
    """
    @taichi.kernel

    Interpolate the real space coordinates described by `αs`, `βs`, and `φs` at 
    the continuous indices in `γ_ci`.
    """
    for I in ti.grouped(γ_ci):
        γ[I][0] = scalar_trilinear_interpolate(αs, γ_ci[I])
        γ[I][1] = scalar_trilinear_interpolate(βs, γ_ci[I])
        γ[I][2] = scalar_trilinear_interpolate(φs, γ_ci[I])

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
    
    # B1 = [cos(φ),sin(φ)/cos(α),sin(φ)tan(α)]
    # B2 = [-sin(φ),cos(φ)/cos(α),cos(φ)tan(α)]
    # B3 = [0,0,1]

    cosα = ti.math.cos(α)
    tanα = ti.math.sin(α)
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

    # B1 = [cos(φ),sin(φ)/cos(α),sin(φ)tan(α)]
    # B2 = [-sin(φ),cos(φ)/cos(α),cos(φ)tan(α)]
    # B3 = [0,0,1]

    cosα = ti.math.cos(α)
    sinα = ti.math.sin(α)
    cosφ = ti.math.cos(φ)
    sinφ = ti.math.sin(φ)

    return ti.Vector([
        vector_static[0] * cosφ + vector_static[1] * sinφ * cosα,
        -vector_static[0] * sinφ + vector_static[1] * cosφ * cosα,
        -vector_static[1] * sinα + vector_static[2]
    ], dt=ti.f32)


def coordinate_real_to_array(α, β, φ, α_min, β_min, φ_min, dαβ, dφ):
    """
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    (`α`, `β`, `φ`). Can broadcast over entire arrays of real coordinates.

    Args:
        `α`: α-coordinate of the point.
        `β`: β-coordinate of the point.
        `θ`: θ-coordinate of the point.
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dαβ`: spatial resolution, which is equal in the α- and β-directions,
          taking values greater than 0.
        `dφ`: orientational resolution, taking values greater than 0.
    """
    I = np.rint((α - α_min) / dαβ).astype(int)
    J = np.rint((β - β_min) / dαβ).astype(int)
    K = np.rint((φ - φ_min) / dφ).astype(int)
    return I, J, K


def coordinate_array_to_real(I, J, K, α_min, β_min, φ_min, dαβ, dφ):
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
        `dαβ`: spatial resolution, which is equal in the α- and β-directions,
          taking values greater than 0.
        `dφ`: orientational resolution, taking values greater than 0.
    """
    α = α_min + I * dαβ
    β = β_min + J * dαβ
    φ = φ_min + K * dφ
    return α, β, φ

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
          index. Note that `0 <= point[0] <= shape[0] - 1`, 
          `0 <= point[1] <= shape[1] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
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
    which we index _after_ aligning with standard array convention.

    Args:
        `point`: Tuple[int, int] describing point with respect to arrays aligned
          with real axes.
        `shape`: shape of array, with respect to standard array convention, in 
          which we want to index. Note that `0 <= point[0] <= shape[0] - 1`, 
          `0 <= point[1] <= shape[1] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
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