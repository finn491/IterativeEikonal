# derivatives.py

import taichi as ti
from eikivp.SE2.interpolate import (
    scalar_trilinear_interpolate, 
    select_upwind_derivative
)


@ti.func
def derivatives(
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
def abs_derivatives(
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
    derivatives(u, dxy, A1_forward, A1_backward, A2_forward, A2_backward, 
                   A3_forward, A3_backward)
    for I in ti.grouped(u):
        abs_A1[I] = ti.math.max(-A1_forward[I], A1_backward[I], 0)
        abs_A2[I] = ti.math.max(-A2_forward[I], A2_backward[I], 0)
        abs_A3[I] = ti.math.max(-A3_forward[I], A3_backward[I], 0)


@ti.func
def upwind_derivatives(
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
    derivatives(u, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward)
    for I in ti.grouped(u):
        upwind_A1[I] = select_upwind_derivative(A1_forward[I], A1_backward[I])
        upwind_A2[I] = select_upwind_derivative(A2_forward[I], A2_backward[I])
        upwind_A3[I] = select_upwind_derivative(A3_forward[I], A3_backward[I])


# Gauge Frame ???