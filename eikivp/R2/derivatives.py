# derivatives.py

import taichi as ti
from eikivp.utils import (
    sanitize_index_R2,
    select_upwind_derivative
)


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
    I_dx = ti.Vector([0, 1], dt=ti.i32)
    I_dy = ti.Vector([1, 0], dt=ti.i32)
    for I in ti.grouped(u):
        # We do not need to interpolate because we always end up on the grid.
        I_dx_forward = sanitize_index_R2(I + I_dx, u)
        I_dx_backward = sanitize_index_R2(I - I_dx, u)
        I_dy_forward = sanitize_index_R2(I + I_dy, u)
        I_dy_backward = sanitize_index_R2(I - I_dy, u)
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
    the `x` and `y` directions. Adapted from Gijs.

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
    @taichi.func

    Compute an upwind approximation of the derivative of `u` in the `x` and `y` 
    directions.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_d*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward)
    for I in ti.grouped(u):
        upwind_dx[I] = select_upwind_derivative(dx_forward[I], dx_backward[I])
        upwind_dy[I] = select_upwind_derivative(dy_forward[I], dy_backward[I])


