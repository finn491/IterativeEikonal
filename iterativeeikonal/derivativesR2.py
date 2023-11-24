# derivativesR2.py

import taichi as ti

@ti.func
def derivatives(
    u: ti.template(),
    h: ti.f32,
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template()
):
    """
    Compute the forward and backward finite differences of `u` with step size `h`.
    """
    I_dx = ti.Vector([1, 0])
    I_dy = ti.Vector([0, 1])
    for I in ti.grouped(dx_forward):
        dx_forward[I] = (u[I + I_dx] - u[I]) / h
        dx_backward[I] = (u[I] - u[I - I_dx]) / h
        dy_forward[I] = (u[I + I_dy] - u[I]) / h
        dy_backward[I] = (u[I] - u[I - I_dy]) / h

@ti.func
def abs_derivatives(
    u: ti.template(),
    h: ti.f32,
    abs_dx: ti.template(),
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    abs_dy: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template()
):
    """
    Compute an approximation of the absolute value of the derivative of `u` in the `x` and `y` directions.
    """
    derivatives(u, h, dx_forward, dx_backward, dy_forward, dy_backward)
    for I in ti.grouped(abs_dx):
        abs_dx[I] = ti.math.max(-dx_forward[I], dx_backward[I], 0)
        abs_dy[I] = ti.math.max(-dy_forward[I], dy_backward[I], 0)