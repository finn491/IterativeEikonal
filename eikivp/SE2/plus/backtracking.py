"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in SE(2). The primary methods are:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the corresponding methods in the distancemap module.
"""

import taichi as ti
from eikivp.SE2.plus.interpolate import (
    vectorfield_trilinear_interpolate_LI,
    scalar_trilinear_interpolate
)
from eikivp.SE2.utils import (
    get_next_point,
    convert_continuous_indices_to_real_space,
    vector_LI_to_static
)

def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, xs_np, ys_np, θs_np, ξ, dt=None, β=0.,
                           n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map.
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1.
        `xs_np`: x-coordinate at every point in the grid on which `cost` is
          sampled.
        `ys_np`: y-coordinate at every point in the grid on which `cost` is
          sampled.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of `cost_np`.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    """
    # Set hyperparameters
    shape = grad_W_np.shape[0:-1]
    if dt is None:
        # It would make sense to also include ξ somehow, but I am not sure how.
        dt = cost_np.min()

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    # Perform backtracking
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=n_max)

    γ_len = geodesic_back_tracking_backend(grad_W, source_point, target_point, θs, ξ, cost, dt, n_max, β, γ)
    print(f"Geodesic consists of {γ_len} points.")
    γ_ci = γ.to_numpy()[:γ_len]

    # Cleanup
    γ_np = convert_continuous_indices_to_real_space(γ_ci, xs_np, ys_np, θs_np)
    return γ_np

@ti.kernel
def geodesic_back_tracking_backend(
    grad_W: ti.template(),
    source_point: ti.types.vector(3, ti.f32),
    target_point: ti.types.vector(3, ti.f32),
    θs: ti.template(),
    ξ: ti.f32,
    cost: ti.template(),
    dt: ti.f32,
    n_max: ti.i32,
    β: ti.f32,
    γ: ti.template()
) -> ti.i32:
    """
    @taichi.kernel

    Find the geodesic connecting `target_point` to `source_point`, using
    gradient descent backtracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
      Static:
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind gradient with
          respect to some cost of the approximate distance map.
        `dt`: Gradient descent step size, taking values greater than 0.
        `source_point`: ti.types.vector(n=3, dtype=[float]) describing index of 
          source point in `W_np`.
        `target_point`: ti.types.vector(n=3, dtype=[float]) describing index of 
          target point in `W_np`.
        `θs`: angle coordinate at each grid point.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `cost`: ti.field(dtype=[float]) of cost function, taking values between
          0 and 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.
        `β`: *Currently not used* Momentum parameter in gradient descent, taking 
          values between 0 and 1. Defaults to 0. 
        `*_target`: Indices of the target point.
      Mutated:
        `γ`: ti.Vector.field(n=2, dtype=[float]) of coordinates of points on the
          geodesic.

    Returns:
        Number of points in the geodesic.
    """
    point = target_point
    γ[0] = point
    tol = 2 
    n = 1
    gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point, ξ, cost)
    θ = scalar_trilinear_interpolate(θs, point)
    gradient_at_point = vector_LI_to_static(gradient_at_point_LI, θ)
    while (ti.math.length(point - source_point) >= tol) and (n < n_max - 1):
        gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point, ξ, cost)
        θ = scalar_trilinear_interpolate(θs, point)
        gradient_at_point_next = vector_LI_to_static(gradient_at_point_LI, θ)
        gradient_at_point = β * gradient_at_point + (1 - β) * gradient_at_point_next
        new_point = get_next_point(point, gradient_at_point, dt)
        γ[n] = new_point
        point = new_point
        n += 1
    γ[n] = source_point
    return n + 1