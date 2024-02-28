"""
    derivatives
    ===========

    Provides a variety of derivative operators on SO(3), namely:
      1. `derivatives`: computes the forward and backward finite difference
      approximations of the B1-, B2, and B3-derivatives.
      2. `abs_derivatives`: computes the absolute value of the upwind
      approximations of the B1-, B2-, and B3-derivatives.
      2. `upwind_derivatives`: computes the the upwind approximations of the
    B1-, B2-, and B3-derivatives.
    Each of these methods has variants to compute only the derivatives in the
    B1-, B2-, or B3-direction.
"""

import taichi as ti
from eikivp.utils import select_upwind_derivative
from eikivp.SO3.utils import scalar_trilinear_interpolate


# All at once

@ti.func
def derivatives(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivatives of `u` with spatial step size `dαβ` and 
    orientational step size `dφ`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    h = ti.math.min(dα, dβ, dφ)
    for I in ti.grouped(B1_forward):
        α = αs[I]
        φ = φs[I]
        cosα = ti.math.cos(α)
        tanα = ti.math.tan(α)
        cosφ = ti.math.cos(φ)
        sinφ = ti.math.sin(φ)
        I_B1 = ti.Vector([cosφ / dα, sinφ/cosα / dβ, sinφ*tanα / dφ], dt=ti.f32) * h
        I_B2 = ti.Vector([-sinφ / dα, cosφ/cosα / dβ, cosφ*tanα / dφ], dt=ti.f32) * h
        I_B3 = ti.Vector([0., 0., 1. / dφ], dt=ti.f32) * h

        B1_forward[I] = (scalar_trilinear_interpolate(u, I + I_B1) - u[I]) / h
        B2_forward[I] = (scalar_trilinear_interpolate(u, I + I_B2) - u[I]) / h
        B3_forward[I] = (scalar_trilinear_interpolate(u, I + I_B3) - u[I]) / h
        B1_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_B1)) / h
        B2_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_B2)) / h
        B3_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_B3)) / h


@ti.func
def abs_derivatives(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    abs_B1: ti.template(),
    abs_B2: ti.template(),
    abs_B3: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivatives of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_B*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives(u, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward, B3_forward, B3_backward)
    for I in ti.grouped(u):
        abs_B1[I] = ti.math.max(-B1_forward[I], B1_backward[I], 0)
        abs_B2[I] = ti.math.max(-B2_forward[I], B2_backward[I], 0)
        abs_B3[I] = ti.math.max(-B3_forward[I], B3_backward[I], 0)


@ti.func
def upwind_derivatives(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
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
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_B*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivatives(u, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward, B3_forward, B3_backward)
    for I in ti.grouped(u):
        upwind_A1[I] = select_upwind_derivative(B1_forward[I], B1_backward[I])
        upwind_A2[I] = select_upwind_derivative(B2_forward[I], B2_backward[I])
        upwind_A3[I] = select_upwind_derivative(B3_forward[I], B3_backward[I])

# Individual derivatives

@ti.func
def derivative_B1(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivative B1 of `u` with spatial step size `dαβ`. Adapted
    from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B1_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    h = ti.math.min(dα, dβ, dφ)
    for I in ti.grouped(B1_forward):
        α = αs[I]
        φ = φs[I]
        cosα = ti.math.cos(α)
        tanα = ti.math.tan(α)
        cosφ = ti.math.cos(φ)
        sinφ = ti.math.sin(φ)
        I_B1 = ti.Vector([cosφ / dα, sinφ/cosα / dβ, sinφ*tanα / dφ], dt=ti.f32) * h

        B1_forward[I] = (scalar_trilinear_interpolate(u, I + I_B1) - u[I]) / h
        B1_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_B1)) / h


@ti.func
def derivative_B2(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template()
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivative B2 of `u` with spatial step size `dαβ`. Adapted
    from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B2_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    h = ti.math.min(dα, dβ, dφ)
    for I in ti.grouped(B2_forward):
        α = αs[I]
        φ = φs[I]
        cosα = ti.math.cos(α)
        tanα = ti.math.tan(α)
        cosφ = ti.math.cos(φ)
        sinφ = ti.math.sin(φ)
        I_B2 = ti.Vector([-sinφ / dα, cosφ/cosα / dβ, cosφ*tanα / dφ], dt=ti.f32) * h

        B2_forward[I] = (scalar_trilinear_interpolate(u, I + I_B2) - u[I]) / h
        B2_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_B2)) / h

@ti.func
def derivative_B3(
    u: ti.template(),
    dφ: ti.f32,
    B3_forward: ti.template(),
    B3_backward: ti.template(),
):
    """
    @taichi.func

    Compute the forward and backward finite difference approximations of the 
    left invariant derivative B3 of `u` with orientational step size 
    `dφ`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B3_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
    """
    for I in ti.grouped(B3_forward):
        I_B3 = ti.Vector([0., 0., 1.], dt=ti.f32)

        B3_forward[I] = (scalar_trilinear_interpolate(u, I + I_B3) - u[I]) / dφ
        B3_backward[I] = (u[I] - scalar_trilinear_interpolate(u, I - I_B3)) / dφ


@ti.func
def abs_B1(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    abs_B1_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivative B1 of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B1_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_B1_u`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivative_B1(u, dα, dβ, dφ, αs, φs, B1_forward, B1_backward)
    for I in ti.grouped(u):
        abs_B1_u[I] = ti.math.max(-B1_forward[I], B1_backward[I], 0)


@ti.func
def abs_B2(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    abs_B2_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivative B2 of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B2_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_B2_u`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivative_B2(u, dα, dβ, dφ, αs, φs, B2_forward, B2_backward)
    for I in ti.grouped(u):
        abs_B2_u[I] = ti.math.max(-B2_forward[I], B2_backward[I], 0)


@ti.func
def abs_A3(
    u: ti.template(),
    dφ: ti.f32,
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    abs_B3_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the absolute value of the upwind left invariant 
    derivative B2 of `u`. Adapted from Gijs Bellaard.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B3_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `abs_B3_u`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    derivative_B3(u, dφ, B3_forward, B3_backward)
    for I in ti.grouped(u):
        abs_B3_u[I] = ti.math.max(-B3_forward[I], B3_backward[I], 0)


@ti.func
def upwind_B1(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    upwind_B1_u: ti.template()
):
    """
    @taichi.func

    Compute an upwind approximation of the left invariant derivative B1 of `u`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B1_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_B1_u`: ti.field(dtype=[float], shape=shape) of upwind 
          derivatives, which are updated in place.
    """
    derivative_B1(u, dα, dβ, dφ, αs, φs, B1_forward, B1_backward)
    for I in ti.grouped(u):
        upwind_B1_u[I] = select_upwind_derivative(B1_forward[I], B1_backward[I])


@ti.func
def upwind_B2(
    u: ti.template(),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    upwind_B2_u: ti.template()
):
    """
    @taichi.func

    Compute an upwind approximation of the left invariant derivative B2 of `u`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B2_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_B2_u`: ti.field(dtype=[float], shape=shape) of upwind 
          derivatives, which are updated in place.
    """
    derivative_B2(u, dα, dβ, dφ, αs, φs, B2_forward, B2_backward)
    for I in ti.grouped(u):
        upwind_B2_u[I] = select_upwind_derivative(B2_forward[I], B2_backward[I])


@ti.func
def upwind_B3(
    u: ti.template(),
    dφ: ti.f32,
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    upwind_B3_u: ti.template()
):
    """
    @taichi.func

    Compute an upwind approximation of the left invariant derivative B3 of `u`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dφ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `B3_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `upwind_B3_u`: ti.field(dtype=[float], shape=shape) of upwind 
          derivatives, which are updated in place.
    """
    derivative_B3(u, dφ, B3_forward, B3_backward)
    for I in ti.grouped(u):
        upwind_B3_u[I] = select_upwind_derivative(B3_forward[I], B3_backward[I])