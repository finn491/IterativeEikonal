"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used with all
    controllers on SE(2).
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
    repeated linear interpolation (x, y, θ). Adapted from Gijs Bellaard.

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
    linear interpolation (x, y, θ). Copied from Gijs Bellaard.

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

def get_boundary_conditions_multi_source(source_points):
    """
    Determine the boundary conditions from `source_points`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    N_points = len(source_points)
    boundarypoints_np = np.array([[i_0 + 1, j_0 + 1, k_0] for (i_0, j_0, k_0) in source_points], dtype=int) # Account for padding.
    boundaryvalues_np = np.array([0.] * N_points, dtype=float)
    boundarypoints = ti.Vector.field(n=3, dtype=ti.i32, shape=N_points)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=N_points, dtype=ti.f32)
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

def check_convergence_multi_source(dW_dt, source_points, tol=1e-3, target_point=None):
    """
    Check whether the IVP method has converged by comparing the Hamiltonian
    `dW_dt` to tolerance `tol`. If `target_point` is provided, only check
    convergence at `target_point`; otherwise check throughout the domain.
    """
    if target_point is None:
        for i_0, j_0, k_0 in source_points:
            dW_dt[i_0+1, j_0+1, k_0] = 0. # Source is fixed.
        error = field_abs_max(dW_dt)
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
    dxy: ti.f32,
    dθ: ti.f32,
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
    gradient_norm_l2 = norm_l2(gradient_at_point, dxy, dθ)
    new_point[0] = point[0] - dt * gradient_at_point[0] / gradient_norm_l2
    new_point[1] = point[1] - dt * gradient_at_point[1] / gradient_norm_l2
    new_point[2] = point[2] - dt * gradient_at_point[2] / gradient_norm_l2
    return new_point

@ti.func
def norm_l2(
    vec: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32
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
            (vec[0] / dxy)**2 +
            (vec[1] / dxy)**2 +
            (vec[2] / dθ)**2
    )

@ti.func
def distance_in_pixels(
    point: ti.types.vector(3, ti.f32),
    source_point: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the distance in pixels given the difference in coordinates and the
    pixel size.

    Args:
        `point`: ti.types.vector(n=3, dtype=[float]) current point.
        `source_point`: ti.types.vector(n=3, dtype=[float]) describing index of 
          source point in `W_np`.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    distance_vec = point - source_point
    return ti.math.sqrt(
        (distance_vec[0] / dxy)**2 +
        (distance_vec[1] / dxy)**2 +
        (mod_offset(distance_vec[2], 2 * ti.math.pi, -ti.math.pi) / dθ)**2
    )

@ti.kernel
def distance_in_pixels_multi_source(
    point: ti.types.vector(3, ti.f32),
    source_points: ti.template(),
    distances: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the distance in pixels given the difference in coordinates and the
    pixel size.

    Args:
        `point`: ti.types.vector(n=3, dtype=[float]) current point.
        `source_points`: ti.Vector.field(n=3, dtype=[float]) describing index of 
          source points in `W_np`.
        `distances`: ti.Vector.field(n=3, dtype=[float]) distances to source
          points.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.

    Returns:
        Minimum distance.
    """
    min_distance = ti.math.inf
    for I in ti.grouped(distances):
        distance_vec = point - source_points[I]
        distance = ti.math.sqrt(
            (distance_vec[0] / dxy)**2 +
            (distance_vec[1] / dxy)**2 +
            (mod_offset(distance_vec[2], 2 * ti.math.pi, -ti.math.pi) / dθ)**2
        )
        distances[I] = distance
        ti.atomic_min(min_distance, distance)
    return min_distance

@ti.func
def mod_offset(
    x: ti.f32,
    period: ti.f32,
    offset: ti.f32,
) -> ti.f32:
    return x - (x - offset)//period * period

# Coordinate and Frame Transforms

@ti.func
def vectorfield_LI_to_static(
    vectorfield_LI: ti.template(),
    θs: ti.template(),
    vectorfield_static: ti.template()
):
    """
    @taichi.func

    Compute the components in the static frame of the vectorfield represented in
    the left invariant frame by `vectorfield_LI`.

    Args:
      Static:
        `vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in the
          left invariant frame.
        `θs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in
          the static frame.
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

    Compute the components in the static frame of the vector represented in
    the left invariant frame by `vector_LI`, given that the angle coordinate of
    the point on the manifold corresponding to this vector is θ.

    Args:
      Static:
        `vector_LI`: ti.Vector(n=3, dtype=[float]) represented in the left
          invariant frame.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.Vector(n=3, dtype=[float]) represented in the static frame.
    """
    # A1 = [cos(θ),sin(θ),0],
    # A2 = [-sin(θ),cos(θ),0],
    # A3 = [0,0,1], whence
    # ω1 = [cos(θ),-sin(θ),0],
    # ω2 = [sin(θ),cos(θ),0],
    # ω3 = [0,0,1].

    cos = ti.math.cos(θ)
    sin = ti.math.sin(θ)

    return ti.Vector([
        cos * vector_LI[0] - sin * vector_LI[1],
        sin * vector_LI[0] + cos * vector_LI[1],
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

    Compute the components in the left invariant frame of the vectorfield
    represented in the static frame by `vectorfield_LI`.

    Args:
      Static:
        `vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in
          the static frame.
        `θs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in the
          left invariant frame.
    """
    for I in ti.grouped(vectorfield_static):
        vectorfield_static[I] = vector_static_to_LI(vectorfield_LI[I], θs[I])

@ti.func
def vector_static_to_LI(
    vector_static: ti.types.vector(3, ti.f32),
    θ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Compute the components in the left invariant frame of the vector represented
    in the static frame by `vector_static`, given that the angle coordinate of
    the point on the manifold corresponding to this vector is θ.

    Args:
      Static:
        `vector_static`: ti.Vector(n=3, dtype=[float]) represented in the static
        frame.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.Vector(n=3, dtype=[float]) vector represented in the left invariant
        frame.
    """
    # A1 = [cos(θ),sin(θ),0],
    # A2 = [-sin(θ),cos(θ),0],
    # A3 = [0,0,1].

    cos = ti.math.cos(θ)
    sin = ti.math.sin(θ)

    return ti.Vector([
        cos * vector_static[0] + sin * vector_static[1],
        -sin * vector_static[0] + cos * vector_static[1],
        vector_static[2]
    ], dt=ti.f32)


def coordinate_real_to_array(x, y, θ, x_min, y_min, θ_min, dxy, dθ):
    """
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    (`x`, `y`, `θ`). Can broadcast over entire arrays of real coordinates.

    Args:
        `x`: x-coordinate of the point.
        `y`: y-coordinate of the point.
        `θ`: θ-coordinate of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    I = np.rint((x - x_min) / dxy).astype(int)
    J = np.rint((y - y_min) / dxy).astype(int)
    K = np.rint((θ - θ_min) / dθ).astype(int)
    return I, J, K

@ti.func
def coordinate_real_to_array_ti(
    point: ti.types.vector(3, ti.f32),
    x_min: ti.f32,
    y_min: ti.f32,
    θ_min: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func
    
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    `point`. Can broadcast over entire arrays of real coordinates.

    Args:
        `point`: vector of x-, y-, and θ-coordinates of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    I = (point[0] - x_min) / dxy
    J = (point[1] - y_min) / dxy
    K = (point[2] - θ_min) / dθ
    return ti.Vector([I, J, K], dt=ti.f32)

def coordinate_array_to_real(I, J, K, x_min, y_min, θ_min, dxy, dθ):
    """
    Compute the real coordinates (x, y, θ) of the point defined by array indices 
    (`I`, `J`, `K`). Can broadcast over entire arrays of array indices.

    Args:
        `I`: I index of the point.
        `J`: J index of the point.
        `K`: K index of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    x = x_min + I * dxy
    y = y_min + J * dxy
    θ = θ_min + K * dθ
    return x, y, θ

def align_to_real_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to standard array convention,
    in terms of indices with respect to arrays aligned with real axes (see Notes
    for more explanation). Here, `shape` gives the shape of the array in which
    we index _after_ aligning with real axes, so [Nx, Ny, Nθ].

    Args:
        `point`: Tuple[int, int, int] describing point with respect to standard
          array indexing convention.
        `shape`: shape of array, aligned to real axes, in which we want to
          index. Note that `0 <= point[0] <= shape[1] - 1`, 
          `0 <= point[1] <= shape[0] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

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
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

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
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

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
    vector_field_aligned = vector_field_flipped.swapaxes(1, 0)
    return vector_field_aligned

def align_to_standard_array_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to arrays aligned with real 
    axes, in terms of indices with respect to standard array convention, (see 
    Notes for more explanation). Here, `shape` gives the shape of the array in 
    which we index _after_ aligning with standard array convention, so
    [Ny, Nx, Nθ].

    Args:
        `point`: Tuple[int, int] describing point with respect to arrays aligned
          with real axes.
        `shape`: shape of array, with respect to standard array convention, in 
          which we want to index. Note that `0 <= point[0] <= shape[1] - 1`, 
          `0 <= point[1] <= shape[0] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

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
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

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
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

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
    vector_field_transposed = vector_field.swapaxes(1, 0)
    vector_field_aligned = np.flip(vector_field_transposed, axis=0)
    return vector_field_aligned