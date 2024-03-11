"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in SO(3). The primary method is:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the corresponding methods in the distancemap module.
"""

import taichi as ti
from eikivp.SO3.Riemannian.interpolate import (
    vectorfield_trilinear_interpolate_LI,
    scalar_trilinear_interpolate
)
from eikivp.SO3.utils import (
    get_next_point,
    coordinate_array_to_real,
    coordinate_real_to_array_ti,
    vector_LI_to_static,
    distance_in_pixels
)

def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, α_min, β_min, φ_min, dα, dβ, dφ, αs_np,
                           φs_np, G_np, dt=None, β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map, with shape [Nα, Nβ, Nφ, 3].
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
        `αs_np`: α-coordinate at every point in the grid on which `cost_np` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
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
    G = ti.Vector(G_np, ti.f32)
    if dt is None:
        # It would make sense to also include G somehow, but I am not sure how.
        dt = cost_np[target_point] * min(dα, dβ, dφ) # Step roughly 1 pixel at a time.

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    source_point = coordinate_array_to_real(*source_point, α_min, β_min, φ_min, dα, dβ, dφ)
    target_point = coordinate_array_to_real(*target_point, α_min, β_min, φ_min, dα, dβ, dφ)
    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)
    αs = ti.field(dtype=ti.f32, shape=αs_np.shape)
    αs.from_numpy(αs_np)
    φs = ti.field(dtype=ti.f32, shape=φs_np.shape)
    φs.from_numpy(φs_np)

    # Perform backtracking
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=n_max)

    γ_len = geodesic_back_tracking_backend(grad_W, source_point, target_point, αs, φs, G, cost, α_min, β_min, φ_min, dα,
                                           dβ, dφ, dt, n_max, β, γ)
    print(f"Geodesic consists of {γ_len} points.")
    γ_np = γ.to_numpy()[:γ_len]
    return γ_np

@ti.kernel
def geodesic_back_tracking_backend(
    grad_W: ti.template(),
    source_point: ti.types.vector(3, ti.f32),
    target_point: ti.types.vector(3, ti.f32),
    αs: ti.template(),
    φs: ti.template(),
    G: ti.types.vector(3, ti.f32),
    cost: ti.template(),
    α_min: ti.f32,
    β_min: ti.f32,
    φ_min: ti.f32,
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
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
        `grad_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ, 3]) of upwind
          gradient with respect to some cost of the approximate distance map.
        `source_point`: ti.types.vector(n=3, dtype=[float]) describing index of 
          source point in `W_np`.
        `target_point`: ti.types.vector(n=3, dtype=[float]) describing index of 
          target point in `W_np`.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `G`: ti.types.vector(n=3, dtype=[float]) of constants of diagonal metric
          tensor with respect to left invariant basis.
        `cost`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of cost function,
          taking values between 0 and 1.
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
        `dt`: Gradient descent step size, taking values greater than 0.
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
    point_array = coordinate_real_to_array_ti(point, α_min, β_min, φ_min, dα, dβ, dφ)
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    # Get gradient using componentwise trilinear interpolation.
    gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point_array, G, cost)
    α = scalar_trilinear_interpolate(αs, point_array)
    φ = scalar_trilinear_interpolate(φs, point_array)
    # Get gradient with respect to static frame.
    gradient_at_point = vector_LI_to_static(gradient_at_point_LI, α, φ)
    while (distance_in_pixels(point - source_point, dα, dβ, dφ) >= tol) and (n < n_max - 1):
        # Get gradient using componentwise trilinear interpolation.
        gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point, G, cost)
        α = scalar_trilinear_interpolate(αs, point_array)
        φ = scalar_trilinear_interpolate(φs, point_array)
        # Get gradient with respect to static frame.
        gradient_at_point_next = vector_LI_to_static(gradient_at_point_LI, α, φ)
        # Take weighted average with previous gradients for momentum.
        gradient_at_point = β * gradient_at_point + (1 - β) * gradient_at_point_next
        new_point = get_next_point(point, gradient_at_point, dt)
        γ[n] = new_point
        point = new_point
        # To get the gradient, we need the corresponding array indices.
        point_array = coordinate_real_to_array_ti(point, α_min, β_min, φ_min, dα, dβ, dφ)
        n += 1
    γ[n] = source_point
    return n + 1