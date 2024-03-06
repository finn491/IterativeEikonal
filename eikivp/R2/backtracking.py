"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in R^2. The primary method is:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the methods in the distancemap module.
"""

import numpy as np
import taichi as ti
from eikivp.R2.interpolate import vectorfield_bilinear_interpolate
from eikivp.R2.utils import (
    coordinate_array_to_real,
    coordinate_real_to_array_ti
)


def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, x_min, y_min, dxy, G_np=None, dt=None, β=0.,
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
      Optional:
        `G_np`: np.ndarray(shape=(2,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to standard basis. Defaults to
          standard Euclidean metric.
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of `cost_np`.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    """
    # Set hyperparameters
    shape = grad_W_np.shape[0:-1]
    if G_np is None:
        G_np = np.ones(2)
    G = ti.Vector(G_np, ti.f32)
    if dt is None:
        # It would make sense to also include G somehow, but I am not sure how.
        # dt = cost_np.min() * dxy # Step roughly 1 pixel at a time.
        dt = cost_np[target_point] * dxy # Step roughly 1 pixel at a time.

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    # We perform backtracking in real coordinates instead of in array indices.
    source_point = coordinate_array_to_real(*source_point, x_min, y_min, dxy)
    target_point = coordinate_array_to_real(*target_point, x_min, y_min, dxy)
    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)

    # Perform backtracking
    γ = ti.Vector.field(n=2, dtype=ti.f32, shape=n_max)

    γ_len = geodesic_back_tracking_backend(grad_W, source_point, target_point, G, cost, x_min, y_min, dxy, dt, n_max, β,
                                           γ)
    print(f"Geodesic consists of {γ_len} points.")
    γ_np = γ.to_numpy()[:γ_len]
    return γ_np

@ti.kernel
def geodesic_back_tracking_backend(
    grad_W: ti.template(),
    source_point: ti.types.vector(2, ti.f32),
    target_point: ti.types.vector(2, ti.f32),
    G: ti.types.vector(2, ti.f32),
    cost: ti.template(),
    x_min: ti.f32,
    y_min: ti.f32,
    dxy: ti.f32,
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
        `source_point`: ti.types.vector(n=2, dtype=[float]) describing index of 
          source point in `W_np`.
        `target_point`: ti.types.vector(n=2, dtype=[float]) describing index of 
          target point in `W_np`.
        `G`: ti.types.vector(n=2, dtype=[float]) of constants of the diagonal
          metric tensor with respect to standard basis.
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
    # To get the gradient, we need the corresponding array indices.
    point_array = coordinate_real_to_array_ti(point, x_min, y_min, dxy)
    tol = 2. * dxy # Stop if we are within two pixels of the source.
    n = 1
    # Get gradient using componentwise bilinear interpolation.
    gradient_at_point = vectorfield_bilinear_interpolate(grad_W, point_array, G, cost)
    while (ti.math.length(point - source_point) >= tol) and (n < n_max - 1):
        # Get gradient using componentwise bilinear interpolation.
        gradient_at_point_next = vectorfield_bilinear_interpolate(grad_W, point_array, G, cost)
        # Take weighted average with previous gradients for momentum.
        gradient_at_point = β * gradient_at_point + (1 - β) * gradient_at_point_next
        new_point = get_next_point(point, gradient_at_point, dt)
        γ[n] = new_point
        point = new_point
        # To get the gradient, we need the corresponding array indices.
        point_array = coordinate_real_to_array_ti(point, x_min, y_min, dxy)
        n += 1
    γ[n] = source_point
    return n + 1

@ti.func
def get_next_point(
    point: ti.types.vector(n=2, dtype=ti.f32),
    gradient_at_point: ti.types.vector(n=2, dtype=ti.f32),
    dt: ti.f32
) -> ti.types.vector(n=2, dtype=ti.f32):
    """
    @taichi.func

    Compute the next point in the gradient descent.

    Args:
        `point`: ti.types.vector(n=2, dtype=[float]) coordinates of current 
          point.
        `gradient_at_point`: ti.types.vector(n=2, dtype=[float]) value of 
          gradient at current point.
        `dt`: Gradient descent step size, taking values greater than 0.

    Returns:
        Next point in the gradient descent.
    """
    new_point = ti.Vector([0., 0.], dt=ti.f32)
    new_point[0] = point[0] - dt * gradient_at_point[0]
    new_point[1] = point[1] - dt * gradient_at_point[1]
    return new_point