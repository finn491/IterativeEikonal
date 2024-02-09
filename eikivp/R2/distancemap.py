# distancemap.py

import numpy as np
import taichi as ti
from tqdm import tqdm
from eikivp.R2.derivatives import upwind_derivatives
from eikivp.R2.metric import (
    invert_metric,
    align_to_real_axis_point,
    align_to_real_axis_scalar_field,
    align_to_standard_array_axis_scalar_field,
    align_to_standard_array_axis_vector_field
)
from eikivp.utils import (
    get_initial_W,
    apply_boundary_conditions,
    get_padded_cost,
    unpad_array
)


def eikonal_solver(cost_np, source_point, G_np=None, dxy=1., n_max=1e5):
    """
    Solve the Eikonal PDE on R2, with source at `source_point` and datadriven
    left invariant metric defined by `G_np` and `cost_np`, using the iterative 
    method described in Bekkers et al. "A PDE approach to Data-Driven 
    Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
      Optional:
        `G_np`: np.ndarray(shape=(2, 2), dtype=[float]) of matrix of left 
          invariant metric tensor field with respect to standard basis. Defaults
          to standard Euclidean metric.
        `dxy`: Spatial step size, taking values greater than 0. Defaults to 1.
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.
    """
    # Align with (x, y)-frame.
    cost_np = align_to_real_axis_scalar_field(cost_np)
    shape = cost_np.shape
    source_point = align_to_real_axis_point(source_point, shape)

    # Set hyperparameters
    if G_np is None:
        G_np = np.identity(2)
    G_inv = ti.Matrix(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(4) comes from the fact that the norm of the gradient consists of
    # 4 terms.
    ε = cost_np.min() * dxy / np.sqrt(4 * G_inv.max())
    cost = get_padded_cost(cost_np)

    # Initialise Taichi objects
    W = get_initial_W(shape, initial_condition=100.)
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)
    
    dx_forward = ti.field(dtype=ti.f32, shape=W.shape)
    dx_backward = ti.field(dtype=ti.f32, shape=W.shape)
    dy_forward = ti.field(dtype=ti.f32, shape=W.shape)
    dy_backward = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    dx_W = ti.field(dtype=ti.f32, shape=W.shape)
    dy_W = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=2, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    for _ in tqdm(range(int(n_max))):
        step_W(W, cost, G_inv, dx_forward, dx_backward, dy_forward, dy_backward, dx_W, dy_W, dxy, ε, dW_dt)
        apply_boundary_conditions(W, boundarypoints, boundaryvalues)
    # print(f"Converged after {n - 1} steps!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(W, cost, G_inv, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dx_W, dy_W, grad_W)

    # Align with (I, J)-frame.
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    W_np = align_to_standard_array_axis_scalar_field(W_np)
    grad_W_np = align_to_standard_array_axis_vector_field(grad_W_np)

    return unpad_array(W_np), unpad_array(grad_W_np, pad_shape=(1, 1, 0))


def get_boundary_conditions(source_point):
    """
    Determine the boundary conditions from `source_point`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    i_0, j_0 = source_point
    boundarypoints_np = np.array([[i_0 + 1, j_0 + 1]], dtype=int) # Account for padding.
    boundaryvalues_np = np.array([0.], dtype=float)
    boundarypoints = ti.Vector.field(n=2, dtype=ti.i32, shape=1)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=1, dtype=ti.f32)
    boundaryvalues.from_numpy(boundaryvalues_np)
    return boundarypoints, boundaryvalues


@ti.kernel
def step_W(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.matrix(2, 2, ti.f32),
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template(),
    dx_W: ti.template(),
    dy_W: ti.template(),
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
          inverse metric tensor with respect to standard basis.
        `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives.
        `dxy`: Spatial step size, taking values greater than 0.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map, 
          which is updated in place.
        `dW_dt`: ti.field(dtype=[float], shape=shape) of error of the distance 
          map with respect to the Eikonal PDE, which is updated in place.
        `d*_W*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
          which are updated in place.
    """
    upwind_derivatives(W, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dx_W, dy_W)
    for I in ti.grouped(W):
        dW_dt[I] = 1 - (ti.math.sqrt(
            1 * G_inv[0, 0] * dx_W[I] * dx_W[I] +
            2 * G_inv[0, 1] * dx_W[I] * dy_W[I] + # Metric tensor is symmetric.
            1 * G_inv[1, 1] * dy_W[I] * dy_W[I]
        ) / cost[I])
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.matrix(2, 2, ti.f32),
    dxy: ti.f32,
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
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse metric tensor with respect to standard basis.
        `dxy`: Spatial step size, taking values greater than 0.
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
    upwind_derivatives(W, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dx_W, dy_W)
    for I in ti.grouped(dx_W):
        grad_W[I] = ti.Vector([
            G_inv[0, 0] * dx_W[I] + G_inv[0, 1] * dy_W[I], 
            G_inv[1, 0] * dx_W[I] + G_inv[1, 1] * dy_W[I]
        ]) / cost[I]**2