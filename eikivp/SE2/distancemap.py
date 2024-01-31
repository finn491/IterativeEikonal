# distancemap.py

import numpy as np
import taichi as ti
from tqdm import tqdm
from eikivp.SE2.derivatives import upwind_derivatives
from eikivp.SE2.metric import invert_metric
from eikivp.utils import (
    get_initial_W,
    apply_boundary_conditions,
    get_padded_cost,
    unpad_array
)


def eikonal_solver(cost_np, source_point, G_np, dxy, n_max=1e5):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    metric tensor field defined by `G_np` and `cost_np`, with source at 
    `source_point` and metric, using the iterative method described in Bekkers 
    et al. "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" 
    (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `G_np`: np.ndarray(shape=(3, 3), dtype=[float]) of constants of the 
          metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
      Optional:
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.
    """
    shape = cost_np.shape
    G_inv = ti.Matrix(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(4) comes from the fact that the norm of the gradient consists of
    # 4 terms.
    ε = (cost_np.min() * dxy / G_inv.max()) / np.sqrt(9)
    cost = get_padded_cost(cost_np)
    W = get_initial_W(shape, initial_condition=100.)

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
    
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    # Compute approximate distance map
    for _ in tqdm(range(int(n_max))):
        step_W(W, cost, G_inv, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, A2_W, 
               A3_W, dxy, ε, dW_dt)
        apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(W, cost, G_inv, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, 
                            A3_backward, A1_W, A2_W, A3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np), unpad_array(grad_W_np, pad_shape=(1, 1, 1, 0))

@ti.kernel
def step_W(
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
    upwind_derivatives(W, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, A2_W,
                       A3_W)
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
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field(
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
    upwind_derivatives(W, dxy, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, A2_W,
                       A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            G_inv[0, 0] * A1_W[I] + G_inv[1, 0] * A2_W[I] + G_inv[2, 0] * A3_W[I],
            G_inv[0, 1] * A1_W[I] + G_inv[1, 1] * A2_W[I] + G_inv[2, 1] * A3_W[I],
            G_inv[0, 2] * A1_W[I] + G_inv[1, 2] * A2_W[I] + G_inv[2, 2] * A3_W[I]
        ]) / cost[I]

def get_boundary_conditions(source_point):
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