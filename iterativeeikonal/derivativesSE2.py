# derivativesSE2.py

import taichi as ti

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
        Value interpolation of `input` at `index`.
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

    v00 = v000 * (1.0 - r[0]) + v100 * r[0]
    v01 = v001 * (1.0 - r[0]) + v101 * r[0]
    v10 = v010 * (1.0 - r[0]) + v110 * r[0]
    v11 = v011 * (1.0 - r[0]) + v111 * r[0]

    v0 = v00 * (1.0 - r[1]) + v10 * r[1]
    v1 = v01 * (1.0 - r[1]) + v11 * r[1]

    v = v0 * (1.0 - r[2]) + v1 * r[2]

    return v

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