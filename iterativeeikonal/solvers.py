# solvers.py

import numpy as np
import taichi as ti
import iterativeeikonal as eik
from tqdm import tqdm

# Helper Functions

# Maybe does not belong here...


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


def get_padded_cost(cost_unpadded):
    """Pad the cost function `cost_unpadded` and convert to TaiChi object."""
    cost_np = eik.cleanarrays.pad_array(cost_unpadded, pad_value=1., pad_shape=1)
    cost = ti.field(dtype=ti.f32, shape=cost_np.shape)
    cost.from_numpy(cost_np)
    return cost


def get_initial_W(shape, initial_condition=100.):
    """Initialise the (approximate) distance map as TaiChi object."""
    W_unpadded = np.full(shape=shape, fill_value=initial_condition)
    W_np = eik.cleanarrays.pad_array(W_unpadded, pad_value=initial_condition, pad_shape=1)
    W = ti.field(dtype=ti.f32, shape=W_np.shape)
    W.from_numpy(W_np)
    return W


# Eikonal PDE

# R2

def eikonal_solver_R2(cost_np, source_point, n_max=1e5):
    """
    Solve the Eikonal PDE on R2, with source at `source_point` and metric 
    defined by `cost_np`, using the iterative method described in Bekkers et al.
    "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
      Optional:
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.

    Returns:
        np.ndarray of (approximate) distance map with respect to the cost 
          function described by `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map with
          respect to cost function described by `cost_np`.
    """
    shape = cost_np.shape
    ε = cost_np.min()
    cost = get_padded_cost(cost_np)
    W = get_initial_W(shape, 2)

    # Create empty Taichi objects
    dx_forward = ti.field(dtype=ti.f32, shape=W.shape)
    dx_backward = ti.field(dtype=ti.f32, shape=W.shape)
    dy_forward = ti.field(dtype=ti.f32, shape=W.shape)
    dy_backward = ti.field(dtype=ti.f32, shape=W.shape)
    abs_dx = ti.field(dtype=ti.f32, shape=W.shape)
    abs_dy = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    dx_W = ti.field(dtype=ti.f32, shape=W.shape)
    dy_W = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=2, dtype=ti.f32, shape=W.shape)
    
    boundarypoints, boundaryvalues = get_boundary_conditions_R2(source_point)
    eik.cleanarrays.apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    # Compute approximate distance map
    for _ in tqdm(range(int(n_max))):
        step_W_R2(W, cost, dx_forward, dx_backward, dy_forward, dy_backward, abs_dx, abs_dy, ε, dW_dt)
        eik.cleanarrays.apply_boundary_conditions(W, boundarypoints, boundaryvalues)
    # print(f"Converged after {n - 1} steps!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field_R2(W, cost, dx_forward, dx_backward,dy_forward, dy_backward, dx_W, dy_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    return eik.cleanarrays.unpad_array(W_np), eik.cleanarrays.unpad_array(grad_W_np, pad_shape=(1, 1, 0))


def get_boundary_conditions_R2(source_point):
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


@ti.kernel
def step_W_R2(
    W: ti.template(),
    cost: ti.template(),
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template(),
    abs_dx: ti.template(),
    abs_dy: ti.template(),
    ε: ti.f32,
    dW_dt: ti.template()
):
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
    """
    eik.derivativesR2.abs_derivatives(W, 1., dx_forward, dx_backward, dy_forward, dy_backward, abs_dx, abs_dy)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - (ti.math.sqrt((abs_dx[I]**2 + abs_dy[I]**2)) / cost[I])
        W[I] += dW_dt[I] * ε


@ti.kernel
def distance_gradient_field_R2(
    W: ti.template(),
    cost: ti.template(),
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template(),
    dx_W: ti.template(),
    dy_W: ti.template(),
    grad_W: ti.template()
):
    """
    @taichi.kernel

    Compute the gradient with respect to `cost` of the (approximate) distance
    map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map.
        `cost`: ti.field(dtype=[float], shape=shape) of cost function.
      Mutated:
        `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `dx_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the x direction, which is updated in 
          place.
        `dy_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the y direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    dxy = 1.
    eik.derivativesR2.derivatives(W, dxy, dx_forward, dx_backward, dy_forward, dy_backward)
    eik.derivativesR2.upwind_derivatives(W, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dx_W, dy_W)
    for I in ti.grouped(dx_W):
        grad_W[I] = ti.Vector([dx_W[I], dy_W[I]]) / cost[I]

# SE(2)

def eikonal_solver_SE2_LI(G_inv_np, cost_np, source_point, dxy, n_max=1e5):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    metric tensor field defined by `metric_tensor_diagonal` and `cost_np`, with 
    source at `source_point` and metric, using the iterative method described in 
    Bekkers et al. "A PDE approach to Data-Driven Sub-Riemannian Geodesics in 
    SE(2)" (2015).

    Args:
        `G_inv_np`: np.ndarray(shape=(3, 3), dtype=[float]) of constants of the 
          inverse metric tensor with respect to left invariant basis.
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `dxy`: Spatial step size, taking values greater than 0.
      Optional:
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.

    Returns:
        np.ndarray of (approximate) distance map with respect to the cost 
          function described by `metric_tensor` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map with
          respect to cost function described by `cost_np`.
    """
    shape = cost_np.shape
    ε = cost_np.min() * dxy / G_inv_np.max()
    cost = get_padded_cost(cost_np)
    W = get_initial_W(shape, initial_condition=25.)
    G_inv = ti.Matrix(G_inv_np, ti.f32)

    # Create empty Taichi objects
    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A2_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)
    
    boundarypoints, boundaryvalues = get_boundary_conditions_SE2(source_point)
    eik.cleanarrays.apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    # Compute approximate distance map
    for _ in tqdm(range(int(n_max))):
        step_W_SE2_LI(W, cost, G_inv, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, 
                      A2_W, A3_W, dxy, ε, dW_dt)
        eik.cleanarrays.apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field_SE2_LI(W, cost, G_inv, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, 
                                   A3_backward, A1_W, A2_W, A3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return eik.cleanarrays.unpad_array(W_np), eik.cleanarrays.unpad_array(grad_W_np, pad_shape=(1, 1, 1, 0))

@ti.kernel
def step_W_SE2_LI(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
    dxy: ti.f32,
    ε: ti.f32,
    dW_dt: ti.template()
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative 
    method described in Bekkers et al. in "A PDE approach to Data-Driven Sub-
    Riemannian Geodesics in SE(2)" (2015).

    Args:
      Static:
        `cost`: ti.field(dtype=[float], shape=shape) of cost function.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map, 
          which is updated in place.
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `dW_dt`: ti.field(dtype=[float], shape=shape) of error of the distance 
          map with respect to the Eikonal PDE, which is updated in place.
    """
    eik.derivativesSE2.derivatives_LI(W, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward)
    eik.derivativesSE2.upwind_derivatives_LI(W, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward,
                                             A3_backward, A1_W, A2_W, A3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - (ti.math.sqrt(
            1 * G_inv[0, 0] * A1_W[I] * A1_W[I] +
            2 * G_inv[0, 1] * A1_W[I] * A2_W[I] + # Metric tensor is symmetric.
            2 * G_inv[0, 2] * A1_W[I] * A3_W[I] +
            1 * G_inv[1, 1] * A2_W[I] * A2_W[I] +
            2 * G_inv[1, 2] * A2_W[I] * A3_W[I] +
            1 * G_inv[2, 2] * A3_W[I] * A3_W[I]
        ) / cost[I])
        # dW_dt[I] = 1 - (ti.math.sqrt((abs_A1[I]**2 / G[0] + abs_A2[I]**2 / G[1] + abs_A3[I]**2 / G[2])) / cost[I])
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field_SE2_LI(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    dxy: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
    grad_W: ti.template()
):
    """
    @taichi.kernel

    Compute the gradient with respect to `cost` of the (approximate) distance
    map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map.
        `cost`: ti.field(dtype=[float], shape=shape) of cost function.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    eik.derivativesSE2.derivatives_LI(W, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward)
    eik.derivativesSE2.upwind_derivatives_LI(W, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward,
                                             A3_backward, A1_W, A2_W, A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            G_inv[0, 0] * A1_W[I] + G_inv[1, 0] * A2_W[I] + G_inv[2, 0] * A3_W[I],
            G_inv[0, 1] * A1_W[I] + G_inv[1, 1] * A2_W[I] + G_inv[2, 1] * A3_W[I],
            G_inv[0, 2] * A1_W[I] + G_inv[1, 2] * A2_W[I] + G_inv[2, 2] * A3_W[I]
        ]) / cost[I]

def get_boundary_conditions_SE2(source_point):
    """
    Determine the boundary conditions from `source_point`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    i_0, j_0, θ_0 = source_point
    boundarypoints_np = np.array([[i_0 + 1, j_0 + 1, θ_0 + 1]], dtype=int) # Account for padding.
    boundaryvalues_np = np.array([0.], dtype=float)
    boundarypoints = ti.Vector.field(n=3, dtype=ti.i32, shape=1)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=1, dtype=ti.f32)
    boundaryvalues.from_numpy(boundaryvalues_np)
    return boundarypoints, boundaryvalues


# Gradient Back Tracking

# R2


def geodesic_back_tracking_R2(grad_W_np, source_point, target_point, dt=1., β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map.
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    """
    shape = grad_W_np.shape[0:2]
    grad_W = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)

    # Perform backtracking
    γ_list = ti.root.dynamic(ti.i, n_max)
    γ = ti.Vector.field(n=2, dtype=ti.f32)
    γ_list.place(γ)

    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)

    γ_len = geodesic_back_tracking_R2_backend(grad_W, source_point, target_point, dt, n_max, β, γ)
    γ_dense = ti.Vector.field(n=2, dtype=ti.f32, shape=γ_len)
    print(f"Geodesic consists of {γ_len} points.")
    sparse_to_dense(γ, γ_dense)

    return γ_dense.to_numpy()

@ti.kernel
def geodesic_back_tracking_R2_backend(
    grad_W: ti.template(),
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
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind gradient with
          respect to some cost of the approximate distance map.
        `dt`: Gradient descent step size, taking values greater than 0.
        `source_point`: ti.types.vector(n=2, dtype=[float]) describing index of 
          source point in `W_np`.
        `target_point`: ti.types.vector(n=2, dtype=[float]) describing index of 
          target point in `W_np`.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.
        `β`: *Currently not used* Momentum parameter in gradient descent, taking 
          values between 0 and 1. Defaults to 0. 
        `*_target`: Indices of the target point.
      Mutated:
        `γ`: ti.Vector.field(n=2, dtype=[float]) of coordinates of points on the
          geodesic. #SNode stuff#

    Returns:
        Number of points in the geodesic.
    """
    point = target_point
    γ.append(point)
    tol = 2.
    n = 0
    gradient_at_point = eik.derivativesR2.vectorfield_bilinear_interpolate(grad_W, target_point)
    while (ti.math.length(point - source_point) >= tol) and (n < n_max - 2):
        gradient_at_point = eik.derivativesR2.vectorfield_bilinear_interpolate(grad_W, point)
        new_point = get_next_point_R2(point, gradient_at_point, dt)
        γ.append(new_point)
        point = new_point
        n += 1
    γ.append(source_point)
    return γ.length()

@ti.func
def get_next_point_R2(
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
        γ[I][0] = eik.derivativesR2.scalar_bilinear_interpolate(xs, γ_ci[I])
        γ[I][1] = eik.derivativesR2.scalar_bilinear_interpolate(ys, γ_ci[I])


# SE(2)

def geodesic_back_tracking_SE2(grad_W_np, source_point, target_point, dt=1., β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map.
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    """
    shape = grad_W_np.shape[0:3]
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)

    # Perform backtracking
    γ_list = ti.root.dynamic(ti.i, n_max)
    γ = ti.Vector.field(n=3, dtype=ti.f32)
    γ_list.place(γ)

    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)

    γ_len = geodesic_back_tracking_SE2_backend(grad_W, source_point, target_point, dt, n_max, β, γ)
    γ_dense = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_len)
    print(f"Geodesic consists of {γ_len} points.")
    sparse_to_dense(γ, γ_dense)

    return γ_dense.to_numpy()

@ti.kernel
def geodesic_back_tracking_SE2_backend(
    grad_W: ti.template(),
    source_point: ti.types.vector(3, ti.f32),
    target_point: ti.types.vector(3, ti.f32),
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
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.
        `β`: *Currently not used* Momentum parameter in gradient descent, taking 
          values between 0 and 1. Defaults to 0. 
        `*_target`: Indices of the target point.
      Mutated:
        `γ`: ti.Vector.field(n=2, dtype=[float]) of coordinates of points on the
          geodesic. #SNode stuff#

    Returns:
        Number of points in the geodesic.
    """
    point = target_point
    γ.append(point)
    tol = 2.
    n = 0
    gradient_at_point = eik.derivativesSE2.vectorfield_trilinear_interpolate(grad_W, target_point)
    while (ti.math.length(point - source_point) >= tol) and (n < n_max - 2):
        gradient_at_point = eik.derivativesSE2.vectorfield_trilinear_interpolate(grad_W, point)
        new_point = get_next_point_SE2(point, gradient_at_point, dt)
        γ.append(new_point)
        point = new_point
        n += 1
    γ.append(source_point)
    return γ.length()

@ti.func
def get_next_point_SE2(
    point: ti.types.vector(n=3, dtype=ti.f32),
    gradient_at_point: ti.types.vector(n=3, dtype=ti.f32),
    dt: ti.f32
) -> ti.types.vector(n=3, dtype=ti.f32):
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
    new_point = ti.Vector([0., 0., 0.], dt=ti.f32)
    new_point[0] = point[0] - dt * gradient_at_point[0]
    new_point[1] = point[1] - dt * gradient_at_point[1]
    new_point[2] = point[2] - dt * gradient_at_point[2]
    return new_point

def convert_continuous_indices_to_real_space_SE2(γ_ci_np, xs_np, ys_np, θs_np):
    """
    Convert the continuous indices in the geodesic `γ_ci_np` to the 
    corresponding real space coordinates described by `xs_np`, `ys_np`, and
    `θs_np`.
    """
    γ_ci = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_ci_np.shape[0])
    γ_ci.from_numpy(γ_ci_np)
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_ci.shape)

    xs = ti.field(dtype=ti.f32, shape=xs_np.shape)
    xs.from_numpy(xs_np)
    ys = ti.field(dtype=ti.f32, shape=ys_np.shape)
    ys.from_numpy(ys_np)
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    continuous_indices_to_real_SE2(γ_ci, xs, ys, θs, γ)

    return γ.to_numpy()

@ti.kernel
def continuous_indices_to_real_SE2(
    γ_ci: ti.template(),
    xs: ti.template(),
    ys: ti.template(),
    θs: ti.template(),
    γ: ti.template()
):
    """
    @taichi.kernel

    Interpolate the real space coordinates described by `xs`, `ys`, and `θs` at 
    the continuous indices in `γ_ci`.
    """
    for I in ti.grouped(γ_ci):
        γ[I][0] = eik.derivativesSE2.scalar_trilinear_interpolate(xs, γ_ci[I])
        γ[I][1] = eik.derivativesSE2.scalar_trilinear_interpolate(ys, γ_ci[I])
        γ[I][2] = eik.derivativesSE2.scalar_trilinear_interpolate(θs, γ_ci[I])