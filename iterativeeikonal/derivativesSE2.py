# derivativesSE2.py

import taichi as ti
from iterativeeikonal.derivativesR2 import linear_interpolate

# Helper Functions


@ti.func
def sanitize_index(
    index: ti.types.vector(3, ti.i32),
    input: ti.template()
) -> ti.types.vector(3, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Copied from Gijs.

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
    repeated linear interpolation (x, y, θ). Adapted from Gijs.

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
    linear interpolation (x, y, θ). Copied from Gijs.

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

@ti.func
def vectorfield_trilinear_interpolate(
    vectorfield: ti.template(),
    index: ti.template()
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Interpolate vector field, normalised to 1, `vectorfield` at continuous 
    `index` bilinearly, via repeated linear interpolation (x, y, θ).

    Args:
        `vectorfield`: ti.Vector.field(n=3, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=3, dtype=ti.f32) continuous index at which we 
          want to interpolate.

    Returns:
        ti.types.vector(n=3, dtype=ti.f32) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, vectorfield)

    u000, v000, w000 = input[f[0], f[1], f[2]]
    u001, v001, w001 = input[f[0], f[1], c[2]]
    u010, v010, w010 = input[f[0], c[1], f[2]]
    u011, v011, w011 = input[f[0], c[1], c[2]]
    u100, v100, w100 = input[c[0], f[1], f[2]]
    u101, v101, w101 = input[c[0], f[1], c[2]]
    u110, v110, w110 = input[c[0], c[1], f[2]]
    u111, v111, w111 = input[c[0], c[1], c[2]]

    u = trilinear_interpolate(u000, u001, u010, u011, u100, u101, u110, u111, r)
    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)
    w = trilinear_interpolate(w000, w001, w010, w011, w100, w101, w110, w111, r)

    return ti.math.normalize(ti.Vector([u, v, w]))


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
    orientational step size `2π / u.shape[2]`. Copied from Gijs.

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

        A1_forward[I] = (trilinear_interpolate(u, I + I_A1) - u[I]) / dxy
        A2_forward[I] = (trilinear_interpolate(u, I + I_A2) - u[I]) / dxy
        A3_forward[I] = (trilinear_interpolate(u, I + I_A3) - u[I]) / dθ
        A1_backward[I] = (u[I] - trilinear_interpolate(u, I - I_A1)) / dxy
        A2_backward[I] = (u[I] - trilinear_interpolate(u, I - I_A2)) / dxy
        A3_backward[I] = (u[I] - trilinear_interpolate(u, I - I_A3)) / dθ


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

    Compute an approximation of the absolute value of the left invariant 
    derivatives of `u`. Adapted from Gijs.

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

# Gauge Frame ???