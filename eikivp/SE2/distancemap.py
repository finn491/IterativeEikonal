"""
    distancemap
    ============

    Provides methods to compute the distance map on SE(2) with respect to various
    metrics, by solving the Eikonal PDE using the iterative Initial Value 
    Problem (IVP) technique described in Bekkers et al. "A PDE approach to 
    Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015). The primary methods
    are:
      1. `eikonal_solver`: solve the Eikonal PDE with respect to some 
      data-driven left invariant metric, defined by a matrix giving the
      underlying left invariant metric and a cost function. Currently, the 
      method gives incorrect results when the underlying metric is not diagonal
      (with respect to the left invariant frame). This is likely caused by the
      upwind derivatives that are used.
      2. `eikonal_solver_sub_Riemannian`: solve the Eikonal PDE with respect to
      some data-driven left invariant sub-Riemannian metric, defined by a 
      stiffness parameter ξ a cost function. The stiffness parameter ξ fixes the
      relative cost of moving in the A1-direction compared to the A3-direction
      (it corresponds to β in the paper by Bekkers et al.); motion in the 
      A2-direction is inhibited.
      3. `eikonal_solver_plus`: solve the Eikonal PDE with respect to some
      data-driven left invariant plus controller, defined by a stiffness 
      parameter ξ, a plus softness ε, and a cost function. The stiffness 
      parameter ξ fixes the relative cost of moving in the A1-direction compared
      to the A3-direction (it corresponds to β in the paper by Bekkers et al.);
      the plus softness ε restricts the motion in the reverse A1-direction; 
      motion in the A2-direction is inhibited.
    Each of these methods has a uniform cost variant, found by appending to the
    method name.
"""

import numpy as np
import taichi as ti
from tqdm import tqdm
from eikivp.SE2.derivatives import (
    upwind_derivatives,
    upwind_A1,
    upwind_A3
)
from eikivp.SE2.metric import (
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

# Data-driven left invariant

## Riemannian Eikonal PDE solver

def eikonal_solver(cost_np, source_point, G_np, dxy, dθ, θs_np, target_point=None, n_max=1e5, n_max_initialisation=1e5,
                   n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    metric tensor field defined by `G_np` and `cost_np`, with source at 
    `source_point`, using the iterative method described in Bekkers et al. 
    "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `G_np`: np.ndarray(shape=(3, 3), dtype=[float]) of constants of the 
          metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.
    """
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_uniform(cost_np.shape, source_point, G_np, dxy, dθ, θs_np, target_point=target_point,
                                          n_max=n_max_initialisation, n_check=n_check, tol=tol, dε=dε,
                                          initial_condition=initial_condition)
    
    print("Solving Eikonal PDE data-driven left invariant metric.")
    # Align with (x, y, θ)-frame
    W_init_np = align_to_real_axis_scalar_field(W_init_np)
    cost_np = align_to_real_axis_scalar_field(cost_np)
    shape = cost_np.shape
    source_point = align_to_real_axis_point(source_point, shape)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters
    G_inv = ti.Matrix(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(4) comes from the fact that the norm of the gradient consists of
    # 4 terms.
    ε = dε * (dxy / G_inv.max()) / np.sqrt(9) # cost_np.min() * 
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    cost = get_padded_cost(cost_np, pad_shape=((1,), (1,), (0,)))
    W = get_padded_cost(W_init_np, pad_shape=((1,), (1,), (0,)), pad_value=initial_condition)
    # W = get_initial_W(shape, initial_condition=100., pad_shape=((1,), (1,), (0,)))
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

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

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W(W, cost, G_inv, dxy, dθ, θs, ε, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward,
                   A3_backward, A1_W, A2_W, A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(W, cost, G_inv, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, 
                            A3_backward, A1_W, A2_W, A3_W, grad_W)

    # Align with (I, J, K)-frame
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    W_np = align_to_standard_array_axis_scalar_field(W_np)
    grad_W_np = align_to_standard_array_axis_vector_field(grad_W_np)

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
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
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
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
    upwind_derivatives(W, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, 
                       A2_W, A3_W)
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
        W[I] += dW_dt[I] * ε * cost[I]

@ti.kernel
def distance_gradient_field(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
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
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_derivatives(W, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, 
                       A2_W, A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            G_inv[0, 0] * A1_W[I] + G_inv[1, 0] * A2_W[I] + G_inv[2, 0] * A3_W[I],
            G_inv[0, 1] * A1_W[I] + G_inv[1, 1] * A2_W[I] + G_inv[2, 1] * A3_W[I],
            G_inv[0, 2] * A1_W[I] + G_inv[1, 2] * A2_W[I] + G_inv[2, 2] * A3_W[I]
        ]) / cost[I]**2


## Sub-Riemannian Eikonal PDE solver

def eikonal_solver_sub_Riemannian(cost_np, source_point, ξ, dxy, dθ, θs_np, target_point=None, n_max=1e5,
                                  n_max_initialisation=1e5, n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    metric tensor field defined by `ξ` and `cost_np`, with source at 
    `source_point`, using the iterative method described in Bekkers et al. 
    "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base metric tensor field (i.e. with uniform cost), is given, for a
        pair of vectors v = v^i A_i and w = w^i A_i at point p, by 
          G_p(v, w) = ξ^2 v^1 w^2 + v^3 w^3.
    """
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_sub_Riemannian_uniform(cost_np.shape, source_point, ξ, dxy, dθ, θs_np,
                                                         target_point=target_point, n_max=n_max_initialisation,
                                                         n_check=n_check, tol=tol, dε=dε,
                                                         initial_condition=initial_condition)
    
    print("Solving Eikonal PDE data-driven left invariant metric.")
    # Align with (x, y, θ)-frame
    W_init_np = align_to_real_axis_scalar_field(W_init_np)
    cost_np = align_to_real_axis_scalar_field(cost_np)
    shape = cost_np.shape
    source_point = align_to_real_axis_point(source_point, shape)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    ε = dε * (dxy / (1 + ξ**-2)) / np.sqrt(9) # cost_np.min() * 
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    cost = get_padded_cost(cost_np, pad_shape=((1,), (1,), (0,)))
    W = get_padded_cost(W_init_np, pad_shape=((1,), (1,), (0,)), pad_value=initial_condition)
    # W = get_initial_W(shape, initial_condition=100., pad_shape=((1,), (1,), (0,)))
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_sub_Riemannian(W, cost, ξ, dxy, dθ, θs, ε, A1_forward, A1_backward, A3_forward, A3_backward, A1_W,
                                  A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field_sub_Riemannian(W, cost, ξ, dxy, dθ, θs, A1_forward, A1_backward, A3_forward, A3_backward, 
                                           A1_W, A3_W, grad_W)

    # Align with (I, J, K)-frame
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    W_np = align_to_standard_array_axis_scalar_field(W_np)
    grad_W_np = align_to_standard_array_axis_vector_field(grad_W_np)

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_sub_Riemannian(
    W: ti.template(),
    cost: ti.template(),
    ξ: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A3_W: ti.template(),
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
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
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
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - (ti.math.sqrt(
            A1_W[I]**2 / ξ**2 +
            A3_W[I]**2 
        ) / cost[I])
        W[I] += dW_dt[I] * ε * cost[I]

@ti.kernel
def distance_gradient_field_sub_Riemannian(
    W: ti.template(),
    cost: ti.template(),
    ξ: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
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
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            A1_W[I] / ξ**2,
            0.,
            A3_W[I]
        ]) / cost[I]**2


## Plus-controller Eikonal PDE solver

def eikonal_solver_plus(cost_np, source_point, ξ, plus_softness, dxy, dθ, θs_np, target_point=None, n_max=1e5,
                        n_max_initialisation=1e5, n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    Finsler function defined by `ξ` and `cost_np`, with source at 
    `source_point`, using the iterative method described in Bekkers et al. 
    "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base Finsler function (i.e. with uniform cost), is given, for vector
        v = v^i A_i at point p, by 
          F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
        where (x)_+ := max{x, 0} is the positive part of x.
    """
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_plus_uniform(cost_np.shape, source_point, ξ, plus_softness, dxy, dθ, θs_np,
                                               n_max=n_max_initialisation, n_check=n_check, tol=tol, dε=dε,
                                               initial_condition=initial_condition)
    
    print("Solving Eikonal PDE data-driven left invariant metric.")
    # Align with (x, y, θ)-frame
    W_init_np = align_to_real_axis_scalar_field(W_init_np)
    cost_np = align_to_real_axis_scalar_field(cost_np)
    shape = cost_np.shape
    source_point = align_to_real_axis_point(source_point, shape)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    ε = dε * (dxy / (1 + ξ**-2)) / np.sqrt(9) # cost_np.min() * 
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    cost = get_padded_cost(cost_np, pad_shape=((1,), (1,), (0,)))
    W = get_padded_cost(W_init_np, pad_shape=((1,), (1,), (0,)), pad_value=initial_condition)
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_plus(W, cost, ξ, plus_softness, dxy, dθ, θs, ε, A1_forward, A1_backward, A3_forward, A3_backward, A1_W,
                        A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field_plus(W, cost, ξ, dxy, dθ, θs, A1_forward, A1_backward, A3_forward, A3_backward, A1_W, A3_W,
                                 grad_W)

    # Align with (I, J, K)-frame
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    W_np = align_to_standard_array_axis_scalar_field(W_np)
    grad_W_np = align_to_standard_array_axis_vector_field(grad_W_np)

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_plus(
    W: ti.template(),
    cost: ti.template(),
    ξ: ti.f32,
    plus_softness: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A3_W: ti.template(),
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
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
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
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - (ti.math.sqrt(
            soft_plus(A1_W[I], plus_softness)**2 / ξ**2 +
            A3_W[I]**2 
        ) / cost[I])
        W[I] += dW_dt[I] * ε * cost[I]

@ti.kernel
def distance_gradient_field_plus(
    W: ti.template(),
    cost: ti.template(),
    ξ: ti.f32,
    plus_softness: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
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
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            soft_plus(A1_W[I], plus_softness) / ξ**2,
            0.,
            A3_W[I]
        ]) / cost[I]**2


# Left invariant

## Riemannian Eikonal PDE solver

def eikonal_solver_uniform(domain_shape, source_point, G_np, dxy, dθ, θs_np, target_point=None, n_max=1e5, n_check=None,
                           tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    metric tensor field defined by `G_np`, with source at `source_point`, using
    the iterative method described in Bekkers et al. "A PDE approach to 
    Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, with
          respect to standard array indexing.
        `source_point`: Tuple[int] describing index of source point in 
          `domain_shape`.
        `G_np`: np.ndarray(shape=(3, 3), dtype=[float]) of constants of the 
          metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left 
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.
    """
    # Align with (x, y, θ)-frame
    shape = (domain_shape[1], domain_shape[0], domain_shape[2])
    source_point = align_to_real_axis_point(source_point, shape)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters.
    G_inv = ti.Matrix(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(4) comes from the fact that the norm of the gradient consists of
    # 4 terms.
    ε = dε * (dxy / G_inv.max()) / np.sqrt(9)
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    W = get_initial_W(shape, initial_condition=initial_condition, pad_shape=((1,), (1,), (0,)))
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

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

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_uniform(W, G_inv, dxy, dθ, θs, ε, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, 
                           A3_backward, A1_W, A2_W, A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_uniform(W, G_inv, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward,
                                    A3_backward, A1_W, A2_W, A3_W, grad_W)

    # Align with (I, J, K)-frame
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    W_np = align_to_standard_array_axis_scalar_field(W_np)
    grad_W_np = align_to_standard_array_axis_vector_field(grad_W_np)

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_uniform(
    W: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
    dW_dt: ti.template()
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative 
    method described in Bekkers et al. in "A PDE approach to Data-Driven Sub-
    Riemannian Geodesics in SE(2)" (2015).

    Args:
      Static:
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          inverse metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
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
    upwind_derivatives(W, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, 
                       A2_W, A3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - ti.math.sqrt(
            1 * G_inv[0, 0] * A1_W[I] * A1_W[I] +
            2 * G_inv[0, 1] * A1_W[I] * A2_W[I] + # Metric tensor is symmetric.
            2 * G_inv[0, 2] * A1_W[I] * A3_W[I] +
            1 * G_inv[1, 1] * A2_W[I] * A2_W[I] +
            2 * G_inv[1, 2] * A2_W[I] * A3_W[I] +
            1 * G_inv[2, 2] * A3_W[I] * A3_W[I]
        )
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field_uniform(
    W: ti.template(),
    G_inv: ti.types.matrix(3, 3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
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

    Compute the gradient of the (approximate) distance map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map.
        `G_inv`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of 
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_derivatives(W, dxy, dθ, θs, A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward, A1_W, 
                       A2_W, A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            G_inv[0, 0] * A1_W[I] + G_inv[1, 0] * A2_W[I] + G_inv[2, 0] * A3_W[I],
            G_inv[0, 1] * A1_W[I] + G_inv[1, 1] * A2_W[I] + G_inv[2, 1] * A3_W[I],
            G_inv[0, 2] * A1_W[I] + G_inv[1, 2] * A2_W[I] + G_inv[2, 2] * A3_W[I]
        ])


## Sub-Riemannian Eikonal PDE solver

def eikonal_solver_sub_Riemannian_uniform(domain_shape, source_point, ξ, dxy, dθ, θs_np, target_point=None, n_max=1e5,
                                          n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    metric tensor field defined by `ξ`, with source at `source_point`,
    using the iterative method described in Bekkers et al. "A PDE approach to 
    Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, with
          respect to standard array indexing.
        `source_point`: Tuple[int] describing index of source point in 
          `domain_shape`.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base metric tensor field (i.e. with uniform cost), is given, for a
        pair of vectors v = v^i A_i and w = w^i A_i at point p, by 
          G_p(v, w) = ξ^2 v^1 w^2 + v^3 w^3.
    """
    # Align with (x, y, θ)-frame
    shape = (domain_shape[1], domain_shape[0], domain_shape[2])
    source_point = align_to_real_axis_point(source_point, shape)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    ε = dε * (dxy / (1 + ξ**-2)) / np.sqrt(9)
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    W = get_initial_W(shape, initial_condition=initial_condition, pad_shape=((1,), (1,), (0,)))
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_sub_Riemannian_uniform(W, ξ, dxy, dθ, θs, ε, A1_forward, A1_backward, A3_forward, A3_backward, A1_W,
                                          A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_sub_Riemannian_uniform(W, ξ, dxy, dθ, θs, A1_forward, A1_backward, A3_forward, A3_backward,
                                                   A1_W, A3_W, grad_W)

    # Align with (I, J, K)-frame
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    W_np = align_to_standard_array_axis_scalar_field(W_np)
    grad_W_np = align_to_standard_array_axis_vector_field(grad_W_np)

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_sub_Riemannian_uniform(
    W: ti.template(),
    ξ: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A3_W: ti.template(),
    dW_dt: ti.template()
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative 
    method described in Bekkers et al. in "A PDE approach to Data-Driven Sub-
    Riemannian Geodesics in SE(2)" (2015).

    Args:
      Static:
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
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
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - ti.math.sqrt(
            A1_W[I]**2 / ξ**2 +
            A3_W[I]**2 
        )
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field_sub_Riemannian_uniform(
    W: ti.template(),
    ξ: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A3_W: ti.template(),
    grad_W: ti.template()
):
    """
    @taichi.kernel

    Compute the gradient of the (approximate) distance map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            A1_W[I] / ξ**2,
            0.,
            A3_W[I]
        ])


## Plus-controller Eikonal PDE solver

def eikonal_solver_plus_uniform(domain_shape, source_point, ξ, plus_softness, dxy, dθ, θs_np, target_point=None,
                                n_max=1e5, n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    Finsler function defined by `ξ`, with source at `source_point`, using the 
    iterative method described in Bekkers et al. "A PDE approach to Data-Driven
    Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, with
          respect to standard array indexing.
        `source_point`: Tuple[int] describing index of source point in 
          `domain_shape`.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base Finsler function (i.e. with uniform cost), is given, for vector
        v = v^i A_i at point p, by 
          F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
        where (x)_+ := max{x, 0} is the positive part of x.
    """
    # Align with (x, y, θ)-frame
    shape = (domain_shape[1], domain_shape[0], domain_shape[2])
    source_point = align_to_real_axis_point(source_point, shape)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    ε = dε * (dxy / (1 + ξ**-2)) / np.sqrt(9)
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    W = get_initial_W(shape, initial_condition=initial_condition, pad_shape=((1,), (1,), (0,)))
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_plus_uniform(W, ξ, plus_softness, dxy, dθ, θs, ε, A1_forward, A1_backward, A3_forward, A3_backward, A1_W,
                                A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_plus_uniform(W, ξ, plus_softness, dxy, dθ, θs, A1_forward, A1_backward, A3_forward, 
                                         A3_backward, A1_W, A3_W, grad_W)

    # Align with (I, J, K)-frame
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()
    W_np = align_to_standard_array_axis_scalar_field(W_np)
    grad_W_np = align_to_standard_array_axis_vector_field(grad_W_np)

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_plus_uniform(
    W: ti.template(),
    ξ: ti.f32,
    plus_softness: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A3_W: ti.template(),
    dW_dt: ti.template()
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative 
    method described in Bekkers et al. in "A PDE approach to Data-Driven Sub-
    Riemannian Geodesics in SE(2)" (2015).

    Args:
      Static:
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
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
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - ti.math.sqrt(
            soft_plus(A1_W[I], plus_softness)**2 / ξ**2 +
            A3_W[I]**2 
        )
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field_plus_uniform(
    W: ti.template(),
    ξ: ti.f32,
    plus_softness: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
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
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector([
            soft_plus(A1_W[I], plus_softness) / ξ**2,
            0.,
            A3_W[I]
        ])


# Helper functions

def get_boundary_conditions(source_point):
    """
    Determine the boundary conditions from `source_point`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    i_0, j_0, θ_0 = source_point
    boundarypoints_np = np.array([[i_0 + 1, j_0 + 1, θ_0]], dtype=int) # Account for padding.
    boundaryvalues_np = np.array([0.], dtype=float)
    boundarypoints = ti.Vector.field(n=3, dtype=ti.i32, shape=1)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=1, dtype=ti.f32)
    boundaryvalues.from_numpy(boundaryvalues_np)
    return boundarypoints, boundaryvalues

@ti.func
def soft_plus(
    x: ti.f32, 
    ε: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Return the `ε`-softplus of `x`:
      `soft_plus(x, ε)` = (x)_+ + ε (x)_-,
    where (x)_+ := max{x, 0} and (x)_- := min{x, 0} are the positive and
    negative parts of x, respectively.
    """
    return ti.math.max(x, 0) + ε * ti.math.min(x, 0)

@ti.kernel
def field_abs_max(
    scalar_field: ti.template()
) -> ti.f32:
    """
    @taichi.kernel

    Find the largest absolute value in `scalar_field`.

    Args:
        static: ti.field(dtype=[float], shape=shape) of 2D scalar field.

    Returns:
        Largest absolute value in `scalar_field`.
    """
    value = ti.abs(scalar_field[0, 0, 0])
    for I in ti.grouped(scalar_field):
        value = ti.atomic_min(value, ti.abs(scalar_field[I]))
    return value

def check_convergence(dW_dt, tol=1e-3, target_point=None):
    """
    Check whether the IVP method has converged by comparing the Hamiltonian
    `dW_dt` to tolerance `tol`. If `target_point` is provided, only check
    convergence at `target_point`; otherwise check throughout the domain.
    """
    is_converged = False
    if target_point is None:
        is_converged = field_abs_max(dW_dt) < tol
    else:
        is_converged = ti.abs(dW_dt[target_point[1], target_point[0], target_point[2]]) < tol
    return is_converged