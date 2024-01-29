# derivativesSE2.py

import taichi as ti
from iterativeeikonal.derivativesR2 import linear_interpolate, select_upwind_derivative

# Helper Functions


@ti.func
def sanitize_index(
    index: ti.types.vector(3, ti.i32),
    input: ti.template()
) -> ti.types.vector(3, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Copied from Gijs Bellaard.

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

# I don't think this works properly, we should actually also interpolate the frame...
@ti.func
def vectorfield_trilinear_interpolate_LI(
    vectorfield: ti.template(),
    index: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32)
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    ---------------------------PROBABLY DOESN'T WORK---------------------------

    Interpolate vector field, normalised to 1 and given in left invariant
    coordinates, `vectorfield` at continuous `index` trilinearly, via repeated 
    linear interpolation (x, y, θ).

    Args:
        `vectorfield`: ti.Vector.field(n=3, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=3, dtype=[float]) continuous index at which 
          we want to interpolate.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse of metric tensor with respect to left invariant basis.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, vectorfield)

    u000, v000, w000 = vectorfield[f[0], f[1], f[2]]
    u001, v001, w001 = vectorfield[f[0], f[1], c[2]]
    u010, v010, w010 = vectorfield[f[0], c[1], f[2]]
    u011, v011, w011 = vectorfield[f[0], c[1], c[2]]
    u100, v100, w100 = vectorfield[c[0], f[1], f[2]]
    u101, v101, w101 = vectorfield[c[0], f[1], c[2]]
    u110, v110, w110 = vectorfield[c[0], c[1], f[2]]
    u111, v111, w111 = vectorfield[c[0], c[1], c[2]]

    u = trilinear_interpolate(u000, u001, u010, u011, u100, u101, u110, u111, r)
    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)
    w = trilinear_interpolate(w000, w001, w010, w011, w100, w101, w110, w111, r)

    return normalise_LI(ti.Vector([u, v, w]), G_inv)

@ti.func
def vectorfield_trilinear_interpolate_static(
    vectorfield: ti.template(),
    index: ti.template(),
    θs: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32)
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Interpolate vector field, normalised to 1 and given in static
    coordinates, `vectorfield` at continuous `index` trilinearly, via repeated 
    linear interpolation (x, y, θ).

    Args:
        `vectorfield`: ti.Vector.field(n=3, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=3, dtype=[float]) continuous index at which 
          we want to interpolate.
        `θs`: angle coordinate at each grid point.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse of metric tensor with respect to left invariant basis.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, vectorfield)

    u000, v000, w000 = vectorfield[f[0], f[1], f[2]]
    u001, v001, w001 = vectorfield[f[0], f[1], c[2]]
    u010, v010, w010 = vectorfield[f[0], c[1], f[2]]
    u011, v011, w011 = vectorfield[f[0], c[1], c[2]]
    u100, v100, w100 = vectorfield[c[0], f[1], f[2]]
    u101, v101, w101 = vectorfield[c[0], f[1], c[2]]
    u110, v110, w110 = vectorfield[c[0], c[1], f[2]]
    u111, v111, w111 = vectorfield[c[0], c[1], c[2]]

    u = trilinear_interpolate(u000, u001, u010, u011, u100, u101, u110, u111, r)
    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)
    w = trilinear_interpolate(w000, w001, w010, w011, w100, w101, w110, w111, r)

    θ = scalar_trilinear_interpolate(θs, index)

    return normalise_static(ti.Vector([u, v, w]), G_inv, θ)

@ti.func
def normalise_LI(
    vec: ti.types.vector(3, ti.f32),
    G_inv: ti.types.matrix(3, 3, ti.f32)
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in left invariant coordinates, to 1 with 
    respect to the left invariant metric tensor defined by `G_inv`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse of metric tensor with respect to left invariant basis.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    norm = norm_LI(vec, G_inv)
    return vec / norm

@ti.func
def norm_LI(
    vec: ti.types.vector(3, ti.f32),
    G_inv: ti.types.matrix(3, 3, ti.f32)
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in static coordinates with respect to 
    the left invariant metric tensor defined by `G_inv`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse of metric tensor with respect to left invariant basis.

    Returns:
        Norm of `vec`.
    """
    c_1, c_2, c_3 = vec[0], vec[1], vec[2]
    return ti.math.sqrt(
            1 * G_inv[0, 0] * c_1 * c_1 +
            2 * G_inv[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            2 * G_inv[0, 2] * c_1 * c_3 +
            1 * G_inv[1, 1] * c_2 * c_2 +
            2 * G_inv[1, 2] * c_2 * c_3 +
            1 * G_inv[2, 2] * c_3 * c_3
    )

@ti.func
def normalise_static(
    vec: ti.types.vector(3, ti.f32),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    θ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in static coordinates, to 1 with respect to the 
    left invariant metric tensor defined by `G_inv`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse of metric tensor with respect to left invariant basis.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    # Can do this but it's not necessary
    # vec_LI = vector_LI_to_static(vec, θ)
    # vec_normalised_LI = normalise_LI(vec_LI, G_inv)
    # vec_normalised = vector_static_to_LI(vec_normalised_LI, θ)
    # return vec_normalised
    norm = norm_static(vec, G_inv, θ)
    return vec / norm

@ti.func
def norm_static(
    vec: ti.types.vector(3, ti.f32),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    θ: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in static coordinates with respect to 
    the left invariant metric tensor defined by `G_inv`.

    Args:
        `vec`: ti.types.vector(n=3, dtype=[float]) which we want to normalise.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse of metric tensor with respect to left invariant basis.
        `θ`: angle coordinate of corresponding point on the manifold.

    Returns:
        Norm of `vec`.
    """
    a_1, a_2, a_3 = vec[0], vec[1], vec[2]
    c_1 = a_1 * ti.math.cos(θ) + a_2 * ti.math.sin(θ)
    c_2 = -a_1 * ti.math.sin(θ) + a_2 * ti.math.cos(θ)
    c_3 = a_3
    return ti.math.sqrt(
            1 * G_inv[0, 0] * c_1 * c_1 +
            2 * G_inv[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            2 * G_inv[0, 2] * c_1 * c_3 +
            1 * G_inv[1, 1] * c_2 * c_2 +
            2 * G_inv[1, 2] * c_2 * c_3 +
            1 * G_inv[2, 2] * c_3 * c_3
    )

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

# Left Invariant Derivatives


@ti.func
def derivatives_LI(
    u: ti.template(),
    dxy: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivatives of `u` with spatial step size `dxy` and 
    orientational step size `2π / u.shape[2]`. Copied from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    dθ = 2.0 * ti.math.pi / ti.static(u.shape[2])
    I_A3 = ti.Vector([0.0,  0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(A1_forward):
        θ = I[2] * dθ
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)

        A1_forward[I] = (scalar_trilinear_interpolate(u, I + I_A1) - u[I]) / dxy
        A2_forward[I] = (scalar_trilinear_interpolate(u, I + I_A2) - u[I]) / dxy
        A3_forward[I] = (scalar_trilinear_interpolate(u, I + I_A3) - u[I]) / dθ
        A1_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_A1)) / dxy
        A2_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_A2)) / dxy
        A3_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_A3)) / dθ


@ti.func
def abs_derivatives_LI(
    u: ti.template(),
    dxy: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    abs_A1: ti.template(),
    abs_A2: ti.template(),
    abs_A3: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivatives of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_A*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives_LI(u, dxy, A1_forward, A1_backward, A2_forward, A2_backward, 
                   A3_forward, A3_backward)
    for I in ti.grouped(u):
        abs_A1[I] = ti.math.max(-A1_forward[I], A1_backward[I], 0)
        abs_A2[I] = ti.math.max(-A2_forward[I], A2_backward[I], 0)
        abs_A3[I] = ti.math.max(-A3_forward[I], A3_backward[I], 0)


@ti.func
def upwind_derivatives_LI(
    u: ti.template(),
    dxy: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    upwind_A1: ti.template(),
    upwind_A2: ti.template(),
    upwind_A3: ti.template()
):
    """
    @taichi.func

    Compute an upwind approximation of the derivative of `u` in the `x`, `y`, 
    and `θ` directions.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_A*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives_LI(u, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward)
    for I in ti.grouped(u):
        upwind_A1[I] = select_upwind_derivative(A1_forward[I], A1_backward[I])
        upwind_A2[I] = select_upwind_derivative(A2_forward[I], A2_backward[I])
        upwind_A3[I] = select_upwind_derivative(A3_forward[I], A3_backward[I])


# Gauge Frame ???