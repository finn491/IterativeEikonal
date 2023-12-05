# solvers.py

import numpy as np
import taichi as ti
import iterativeeikonal as eik

# Helper Functions


@ti.kernel
def sparse_to_dense(
    sparse_thing: ti.template(),
    dense_thing: ti.template()
):
    """
    @taichi.func

    Convert a sparse TaiChi object on an SNode into a dense object.

    Args:
      Static:
        `sparse_thing`: Sparse TaiChi object.
      Mutated:
        `dense_thing`: Preinitialised dense TaiChi object of correct size, which
          is updated in place.
    """
    for I in ti.grouped(sparse_thing):
        dense_thing[I] = sparse_thing[I]
    sparse_thing.deactivate()


# Eikonal PDE

# R2

def eikonal_solver_R2(cost_np, source_point, target_point, n_max=1e5):
    """
    Solve the Eikonal PDE on R2, with source at `source_point` and metric 
    defined by `cost_np`, using the iterative method described in Bekkers et al.
    "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `target_point`: Tuple[int] describing index of target point in 
          `cost_np`.

    Returns:
        np.ndarray of (approximate) distance map with respect to the cost 
          function described by `cost_np`.
    """
    N = cost_np.shape[0]
    ε = cost_np.min()
    cost = get_padded_cost(cost_np)
    W = get_initial_W(N)
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    eik.cleanarrays.apply_boundary_conditions(W, boundarypoints, boundaryvalues)
    dx_forward, dx_backward, dy_forward, dy_backward, abs_dx, abs_dy, dW_dt = get_initial_derivatives(W)

    # Maybe extract this loop into its own TaiChi kernel?
    step_size_target = 100.
    tol = 1e-5  # What is a good stopping criterion?
    n = 0
    while (np.abs(step_size_target) > tol) and (n <= n_max):
        step_size_target = step_W(W, cost, dx_forward, dx_backward, dy_forward, dy_backward, abs_dx, abs_dy, ε, dW_dt,
                                  target_point[0] + 1, target_point[1] + 1)
        eik.cleanarrays.apply_boundary_conditions(
            W, boundarypoints, boundaryvalues)
        n += 1
    print(f"Converged after {n - 1} steps!")
    print(step_size_target)

    W_np = W.to_numpy()
    return eik.cleanarrays.unpad_array(W_np)


def get_padded_cost(cost_unpadded):
    """Pad the cost function `cost_unpadded` and convert to TaiChi object."""
    cost_np = eik.cleanarrays.pad_array(cost_unpadded, pad_value=1., pad_shape=1)
    cost = ti.field(dtype=ti.f32, shape=cost_np.shape)
    cost.from_numpy(cost_np)
    return cost


def get_initial_W(N, initial_condition=100.):
    """Initialise the (approximate) distance map as TaiChi object."""
    W_unpadded = np.full(shape=(N, N), fill_value=initial_condition)
    W_np = eik.cleanarrays.pad_array(W_unpadded, pad_value=initial_condition, pad_shape=1)
    W = ti.field(dtype=ti.f32, shape=W_np.shape)
    W.from_numpy(W_np)
    return W


def get_boundary_conditions(source_point):
    """
    Determine the boundary conditions from `source_point`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    i_0, j_0 = source_point
    boundarypoints_np = np.array([[i_0, j_0]], dtype=int)
    boundaryvalues_np = np.array([0.], dtype=float)
    boundarypoints = ti.Vector.field(n=2, dtype=ti.i32, shape=1)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=1, dtype=ti.f32)
    boundaryvalues.from_numpy(boundaryvalues_np)
    return boundarypoints, boundaryvalues


def get_initial_derivatives(W):
    """
    Initialise empty TaiChi objects for the various derivatives of the 
    (approximate) distance map.
    """
    dx_forward = ti.field(dtype=ti.f32, shape=W.shape)
    dx_backward = ti.field(dtype=ti.f32, shape=W.shape)
    dy_forward = ti.field(dtype=ti.f32, shape=W.shape)
    dy_backward = ti.field(dtype=ti.f32, shape=W.shape)
    abs_dx = ti.field(dtype=ti.f32, shape=W.shape)
    abs_dy = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    return dx_forward, dx_backward, dy_forward, dy_backward, abs_dx, abs_dy, dW_dt


@ti.kernel
def step_W(
    W: ti.template(),
    cost: ti.template(),
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template(),
    abs_dx: ti.template(),
    abs_dy: ti.template(),
    ε: ti.f32,
    dW_dt: ti.template(),
    i_target: ti.i32,
    j_target: ti.i32
) -> ti.f32:
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative 
    method described in Bekkers et al. in "A PDE approach to Data-Driven Sub-
    Riemannian Geodesics in SE(2)" (2015).

    Args:
      Static:
        `cost`: ti.field(dtype=[float], shape=shape) of cost function.
        `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map, 
          which is updated in place.
        `dW_dt`: ti.field(dtype=[float], shape=shape) of error of the distance 
          map with respect to the Eikonal PDE, which is updated in place.
        `abs_d*`: ti.field(dtype=[float], shape=shape) of absolute values of
          derivatives, which are updated in place.

    Returns:
        Change in (approximate) distance map at target point.
    """
    eik.derivativesR2.abs_derivatives(W, 1., dx_forward, dx_backward, dy_forward, dy_backward, abs_dx, abs_dy)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - (ti.math.sqrt((abs_dx[I] ** 2 + abs_dy[I] ** 2)) / cost[I])
        W[I] += dW_dt[I] * ε
    # adding this makes it a bit slower, but allows us to terminate when converged.
    return dW_dt[i_target, j_target] * ε

# SE(2)


# Gradient Back Tracking

# R2

def geodesic_back_tracking_R2(W_np, cost_np, source_point, target_point, β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `W_np`: np.ndarray of (approximate) distance map
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
      Optional:
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    """
    dt = 100 * cost_np.min() ** 2  # What is a good step size?

    W = ti.field(dtype=ti.f32, shape=W_np.shape)
    W.from_numpy(W_np)
    cost = ti.field(dtype=ti.f32, shape=cost_np.shape)
    cost.from_numpy(cost_np)

    γ_list = ti.root.dynamic(ti.i, n_max)
    γ = ti.Vector.field(n=2, dtype=ti.f32)
    γ_list.place(γ)

    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)

    γ_len = geodesic_back_tracking_R2_backend(W, cost, source_point, target_point, dt, n_max, β, γ)
    γ_dense = ti.Vector.field(n=2, dtype=ti.f32, shape=γ_len)
    print(f"Geodesic consists of {γ_len} points.")
    sparse_to_dense(γ, γ_dense)

    return γ_dense.to_numpy()


@ti.kernel
def geodesic_back_tracking_R2_backend(
    W: ti.template(),
    cost: ti.template(),
    source_point: ti.types.vector(2, ti.f32),
    target_point: ti.types.vector(2, ti.f32),
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
        `W`: ti.field(dtype=[float], shape=shape) of (approximate) distance map.
        `cost`: ti.field(dtype=[float], shape=shape) of cost function.
        `dt`: Gradient descent step size, taking values greater than 0.
        `source_point`: ti.types.vector(n=2, dtype=[float]) describing index of 
          source point in `W_np`.
        `target_point`: ti.types.vector(n=2, dtype=[float]) describing index of 
          target point in `W_np`.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `*_target`: Indices of the target point.
      Mutated:
        `γ`: ti.Vector.field(n=2, dtype=[float]) of coordinates of points on the
          geodesic. #SNode stuff#

    Returns:
        Number of points in the geodesic.
    """
    point = target_point
    γ.append(point)
    tol = 1e-5
    n = 0
    gradient_at_point = compute_gradient_R2(
        W, point, ti.Vector([0., 0.], dt=ti.f32), 0.)
    while (ti.math.length(gradient_at_point) >= tol) and (n < n_max - 2):
        cost_at_point = eik.derivativesR2.bilinear_interpolate(cost, point)
        gradient_at_point = compute_gradient_R2(W, point, gradient_at_point, β)
        new_point = get_next_point_R2(point, gradient_at_point, cost_at_point, dt)
        γ.append(new_point)
        point = new_point
        n += 1
    γ.append(source_point)
    return γ.length()


@ti.func
def compute_gradient_R2(
    W: ti.template(),
    point: ti.types.vector(n=2, dtype=ti.f32),
    old_gradient: ti.types.vector(n=2, dtype=ti.f32),
    β: ti.f32
) -> ti.types.vector(n=2, dtype=ti.f32):
    """
    @taichi.func

    Compute the gradient of the (approximate) distance map `W` with respect to
    x-y coordinates at `point`.

    Args:
        `W`: ti.field(dtype=[float], shape=shape) of (approximate) distance map.
        `point`: ti.types.vector(n=2, dtype=[float]) coordinates of current
          point.
        `old_gradient`: Gradient at previous point, for use with momentum.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1.

    Returns:
        Gradient of the (approximate) distance map `W` with respect to x-y 
        coordinates at `point`.
    """
    # Central Difference Derivatives
    x_shift = ti.Vector([0.5, 0.], dt=ti.f32)
    y_shift = ti.Vector([0., 0.5], dt=ti.f32)
    W_x_forward = eik.derivativesR2.bilinear_interpolate(W, point + x_shift)
    W_x_backward = eik.derivativesR2.bilinear_interpolate(W, point - x_shift)
    W_y_forward = eik.derivativesR2.bilinear_interpolate(W, point + y_shift)
    W_y_backward = eik.derivativesR2.bilinear_interpolate(W, point - y_shift)
    gradient_x = (W_x_forward - W_x_backward)
    gradient_y = (W_y_forward - W_y_backward)

    # Upwind Derivatives
    # x_shift = ti.Vector([1., 0.], dt=ti.f32)
    # y_shift = ti.Vector([0., 1.], dt=ti.f32)
    # W_point = eik.derivativesR2.bilinear_interpolate(W, point)
    # W_x_forward = eik.derivativesR2.bilinear_interpolate(W, point + x_shift)
    # W_x_backward = eik.derivativesR2.bilinear_interpolate(W, point - x_shift)
    # W_y_forward = eik.derivativesR2.bilinear_interpolate(W, point + y_shift)
    # W_y_backward = eik.derivativesR2.bilinear_interpolate(W, point - y_shift)
    # dx_forward = W_x_forward - W_point
    # dx_backward = W_point - W_x_backward
    # gradient_x = ti.math.max(-dx_forward, dx_backward, 0.) * (-1.)**(dx_forward >= dx_backward)
    # dy_forward = W_y_forward - W_point
    # dy_backward = W_point - W_y_backward
    # gradient_y = ti.math.max(-dy_forward, dy_backward, 0.) * (-1.)**(dy_forward >= dy_backward)

    new_gradient_x = β * old_gradient[0] + (1 - β) * gradient_x
    new_gradient_y = β * old_gradient[1] + (1 - β) * gradient_y
    return ti.Vector([new_gradient_x, new_gradient_y], ti.f32)


@ti.func
def get_next_point_R2(
    point: ti.types.vector(n=2, dtype=ti.f32),
    gradient_at_point: ti.types.vector(n=2, dtype=ti.f32),
    cost_at_point: ti.f32,
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
        `cost_at_point`: Value of the cost function at current point, taking
          values greater than 0.
        `dt`: Gradient descent step size, taking values greater than 0.

    Returns:
        Next point in the gradient descent.
    """
    new_point = ti.Vector([0., 0.], dt=ti.f32)
    new_point[0] = point[0] - dt * gradient_at_point[0] / cost_at_point**2
    new_point[1] = point[1] - dt * gradient_at_point[1] / cost_at_point**2
    return new_point


def convert_continuous_indices_to_real_space_R2(γ_ci_np, xs_np, ys_np):
    """
    Convert the continuous indices in the geodesic `γ_ci_np` to the 
    corresponding real space coordinates described by `xs_np` and `ys_np`.
    """
    γ_ci = ti.Vector.field(n=2, dtype=ti.f32, shape=γ_ci_np.shape[0])
    γ_ci.from_numpy(γ_ci_np)
    γ = ti.Vector.field(n=2, dtype=ti.f32, shape=γ_ci.shape)

    xs = ti.field(dtype=ti.f32, shape=xs_np.shape)
    xs.from_numpy(xs_np)
    ys = ti.field(dtype=ti.f32, shape=ys_np.shape)
    ys.from_numpy(ys_np)

    continuous_indices_to_real_R2(γ_ci, xs, ys, γ)

    return γ.to_numpy()


@ti.kernel
def continuous_indices_to_real_R2(
    γ_ci: ti.template(),
    xs: ti.template(),
    ys: ti.template(),
    γ: ti.template()
):
    """
    @taichi.kernel

    Interpolate the real space coordinates described by `xs` and `ys` at the 
    continuous indices in `γ_ci`.
    """
    for I in ti.grouped(γ_ci):
        γ[I][0] = eik.derivativesR2.bilinear_interpolate(xs, γ_ci[I])
        γ[I][1] = eik.derivativesR2.bilinear_interpolate(ys, γ_ci[I])


# SE(2)
