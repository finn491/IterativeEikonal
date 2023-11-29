# derivativesR2.py

import taichi as ti

# Helper Functions


@ti.func
def sanitize_index(
    index: ti.types.vector(2, ti.i32),
    input: ti.template()
) -> ti.types.vector(2, ti.i32):
    """Make sure the `index` is inside the shape of `input`."""
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
    ], dt=ti.i32)

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
    Compute the forward and backward finite differences of `u` with step size 
    `dxy`.
    """
    I_dx = ti.Vector([1, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1], dt=ti.i32)
    for I in ti.grouped(u):
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
    Compute an approximation of the absolute value of the derivative of `u` in 
    the `x` and `y` directions.
    """
    derivatives(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward)
    for I in ti.grouped(u):
        abs_dx[I] = ti.math.max(-dx_forward[I], dx_backward[I], 0)
        abs_dy[I] = ti.math.max(-dy_forward[I], dy_backward[I], 0)

@ti.func
def upwind_derivatives(
    u: ti.template(),
    dxy: ti.f32,
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template(),
    upwind_dx: ti.template(),
    upwind_dy: ti.template()
):
    """
    Compute an upwind approximation of the derivative of `u` in the `x` and `y` 
    directions.
    """
    derivatives(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward)
    for I in ti.grouped(u):
        upwind_dx[I] = select_upwind_derivative(dx_forward[I], dx_backward[I])
        upwind_dy[I] = select_upwind_derivative(dy_forward[I], dy_backward[I])

@ti.func
def select_upwind_derivative(
    d_forward: ti.f32,
    d_backward: ti.f32
) -> ti.f32:
    return ti.math.max(-d_forward, d_backward, 0) * (-1.)**(-d_forward >= d_backward)