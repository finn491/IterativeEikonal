# derivativesR2.py

import taichi as ti

# Helper Functions


@ti.func
def sanitize_index(
    index: ti.types.vector(2, ti.i32),
    input: ti.template()
) -> ti.types.vector(2, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`.

    Args:
        `index`: ti.types.vector(n=2, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=2, dtype=int) of index that is within `input`.
    """
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
    ], dt=ti.i32)

@ti.func
def bilinear_interpolate(
    input: ti.template(),
    index: ti.template()
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of `input` at continuous `index` bilinearly, via repeated
    linear interpolation (x, y). Adapted from Gijs.

    Args:
        `input`: ti.field(dtype=[float]) in which we want to interpolate.
        `index`: ti.types.vector(n=2, dtype=ti.f32) continuous index at which we 
          want to interpolate.

    Returns:
        Value interpolation of `input` at `index`.
    """
    r = ti.math.fract(index)

    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, input)

    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, input)
    
    v00 = input[f[0], f[1]]
    v01 = input[f[0], c[1]]
    v10 = input[c[0], f[1]]
    v11 = input[c[0], c[1]]

    v0 = v00 * (1.0 - r[0]) + v10 * r[0]
    v1 = v01 * (1.0 - r[0]) + v11 * r[0]

    v = v0 * (1.0 - r[1]) + v1 * r[1]

    return v

# Actual Derivatives


@ti.func
def derivatives(
    u: ti.template(),
    dxy: ti.f32,
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite differences of `u` with step size 
    `dxy`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    I_dx = ti.Vector([1, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1], dt=ti.i32)
    for I in ti.grouped(u):
        # We do not need to interpolate because we always end up on the grid.
        I_dx_forward = sanitize_index(I + I_dx, u)
        I_dx_backward = sanitize_index(I - I_dx, u)
        I_dy_forward = sanitize_index(I + I_dy, u)
        I_dy_backward = sanitize_index(I - I_dy, u)
        dx_forward[I] = (u[I_dx_forward] - u[I]) / dxy
        dx_backward[I] = (u[I] - u[I_dx_backward]) / dxy
        dy_forward[I] = (u[I_dy_forward] - u[I]) / dxy
        dy_backward[I] = (u[I] - u[I_dy_backward]) / dxy


@ti.func
def abs_derivatives(
    u: ti.template(),
    dxy: ti.f32,
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template(),
    abs_dx: ti.template(),
    abs_dy: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the derivative of `u` in 
    the `x` and `y` directions.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_d*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward)
    for I in ti.grouped(u):
        abs_dx[I] = ti.math.max(-dx_forward[I], dx_backward[I], 0)
        abs_dy[I] = ti.math.max(-dy_forward[I], dy_backward[I], 0)

# @ti.func
# def upwind_derivatives(
#     u: ti.template(),
#     dxy: ti.f32,
#     dx_forward: ti.template(),
#     dx_backward: ti.template(),
#     dy_forward: ti.template(),
#     dy_backward: ti.template(),
#     upwind_dx: ti.template(),
#     upwind_dy: ti.template()
# ):
#     """
#     Compute an upwind approximation of the derivative of `u` in the `x` and `y` 
#     directions.
#     """
#     derivatives(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward)
#     for I in ti.grouped(u):
#         upwind_dx[I] = select_upwind_derivative(dx_forward[I], dx_backward[I])
#         upwind_dy[I] = select_upwind_derivative(dy_forward[I], dy_backward[I])

# @ti.func
# def select_upwind_derivative(
#     d_forward: ti.f32,
#     d_backward: ti.f32
# ) -> ti.f32:
#     return ti.math.max(-d_forward, d_backward, 0) * (-1.)**(-d_forward >= d_backward)
