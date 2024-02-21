"""
    derivatives
    ===========

    Provides a variety of derivative operators on R^2, namely:
      1. `derivatives`: computes the forward and backward finite difference
      approximations of the A1-, A2, and A3-derivatives.
      2. `abs_derivatives`: computes the absolute value of the upwind
      approximations of the A1-, A2-, and A3-derivatives.
      2. `upwind_derivatives`: computes the the upwind approximations of the
      A1-, A2-, and A3-derivatives.
    Each of these methods has variants to compute only the derivatives in the
    A1-, A2-, or A3-direction.
"""

import taichi as ti
from eikivp.utils import select_upwind_derivative
from eikivp.SE2.utils import scalar_trilinear_interpolate


# All at once

@ti.func
def derivatives(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
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
    orientational step size `dθ`. Copied from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in spatial directions, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    I_A3 = ti.Vector([0.0,  0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(A1_forward):
        θ = θs[I]
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
    dθ: ti.f32,
    θs: ti.template(),
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
        `dxy`: step size in spatial directions, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_A*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives(u, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward)
    for I in ti.grouped(u):
        abs_A1[I] = ti.math.max(-A1_forward[I], A1_backward[I], 0)
        abs_A2[I] = ti.math.max(-A2_forward[I], A2_backward[I], 0)
        abs_A3[I] = ti.math.max(-A3_forward[I], A3_backward[I], 0)


@ti.func
def upwind_derivatives(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
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

    Compute an upwind approximation of the left invariant derivatives of `u`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in spatial directions, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_A*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives(u, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward)
    for I in ti.grouped(u):
        upwind_A1[I] = select_upwind_derivative(A1_forward[I], A1_backward[I])
        upwind_A2[I] = select_upwind_derivative(A2_forward[I], A2_backward[I])
        upwind_A3[I] = select_upwind_derivative(A3_forward[I], A3_backward[I])

# Individual derivatives

@ti.func
def derivative_A1(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivative A1 of `u` with spatial step size `dxy`. Adapted
    from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in spatial directions, taking values greater than 0.
      Mutated:
        `A1_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    for I in ti.grouped(A1_forward):
        θ = θs[I]
        I_A1 = ti.Vector([ti.math.cos(θ), ti.math.sin(θ), 0.0], dt=ti.f32)

        A1_forward[I] = (scalar_trilinear_interpolate(u, I + I_A1) - u[I]) / dxy
        A1_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_A1)) / dxy


@ti.func
def derivative_A2(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivative A2 of `u` with spatial step size `dxy`. Adapted
    from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in spatial directions, taking values greater than 0.
      Mutated:
        `A1_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    for I in ti.grouped(A2_forward):
        θ = θs[I]
        I_A2 = ti.Vector([-ti.math.sin(θ), ti.math.cos(θ), 0.0], dt=ti.f32)

        A2_forward[I] = (scalar_trilinear_interpolate(u, I + I_A2) - u[I]) / dxy
        A2_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_A2)) / dxy


@ti.func
def derivative_A3(
    u: ti.template(),
    dθ: ti.f32,
    A3_forward: ti.template(),
    A3_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivative A3 of `u` with orientational step size 
    `dθ`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dθ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `A3_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    I_A3 = ti.Vector([0.0,  0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(A3_forward):
        A3_forward[I] = (scalar_trilinear_interpolate(u, I + I_A3) - u[I]) / dθ
        A3_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_A3)) / dθ


@ti.func
def abs_A1(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    abs_A1_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivative A1 of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in spatial directions, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A1_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_A1_u`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivative_A1(u, dxy, θs, A1_forward, A1_backward)
    for I in ti.grouped(u):
        abs_A1_u[I] = ti.math.max(-A1_forward[I], A1_backward[I], 0)


@ti.func
def abs_A2(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    abs_A2_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivative A2 of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in spatial directions, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A2_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_A2_u`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivative_A2(u, dxy, θs, A2_forward, A2_backward)
    for I in ti.grouped(u):
        abs_A2_u[I] = ti.math.max(-A2_forward[I], A2_backward[I], 0)


@ti.func
def abs_A3(
    u: ti.template(),
    dθ: ti.f32,
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    abs_A3_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivative A3 of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dθ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `A3_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_A3_u`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivative_A3(u, dθ, A3_forward, A3_backward)
    for I in ti.grouped(u):
        abs_A3_u[I] = ti.math.max(-A3_forward[I], A3_backward[I], 0)


@ti.func
def upwind_A1(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    upwind_A1_u: ti.template()
):
    """
    @taichi.func

    Compute an upwind approximation of the left invariant derivative A1 of `u`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in spatial directions, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A1_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_A1_u`: ti.field(dtype=[float], shape=shape) of upwind 
          derivatives, which are updated in place.
    """
    derivative_A1(u, dxy, θs, A1_forward, A1_backward)
    for I in ti.grouped(u):
        upwind_A1_u[I] = select_upwind_derivative(A1_forward[I], A1_backward[I])


@ti.func
def upwind_A2(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    upwind_A2_u: ti.template()
):
    """
    @taichi.func

    Compute an upwind approximation of the left invariant derivative A2 of `u`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in spatial directions, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A2_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_A2_u`: ti.field(dtype=[float], shape=shape) of upwind 
          derivatives, which are updated in place.
    """
    derivative_A2(u, dxy, θs, A2_forward, A2_backward)
    for I in ti.grouped(u):
        upwind_A2_u[I] = select_upwind_derivative(A2_forward[I], A2_backward[I])


@ti.func
def upwind_A3(
    u: ti.template(),
    dθ: ti.f32,
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    upwind_A3_u: ti.template()
):
    """
    @taichi.func

    Compute an upwind approximation of the left invariant derivative A3 of `u`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dθ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `A3_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_A3_u`: ti.field(dtype=[float], shape=shape) of upwind
          derivatives, which are updated in place.
    """
    derivative_A3(u, dθ, A3_forward, A3_backward)
    for I in ti.grouped(u):
        upwind_A3_u[I] = select_upwind_derivative(A3_forward[I], A3_backward[I])

# Gauge Frame ???