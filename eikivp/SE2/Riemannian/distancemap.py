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
)
from eikivp.SE2.utils import (
    get_boundary_conditions,
    check_convergence,
    align_to_real_axis_point,
    align_to_real_axis_scalar_field,
    align_to_standard_array_axis_scalar_field,
    align_to_standard_array_axis_vector_field
)
from eikivp.SE2.Riemannian.metric import (
    invert_metric
)
from eikivp.utils import (
    get_initial_W,
    apply_boundary_conditions,
    get_padded_cost,
    unpad_array
)

# Data-driven left invariant

def eikonal_solver(cost_np, source_point, G_np, dxy, dθ, θs_np, target_point=None, n_max=1e5, n_max_initialisation=1e4,
                   n_check=None, n_check_initialisation=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    metric tensor field defined by `G_np` and `cost_np`, with source at 
    `source_point`, using the iterative method described in Bekkers et al. 
    "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
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
          initialisation, taking positive values. Defaults to 1e4.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max`. Defaults to `None`; if no
          `n_check` is passed, convergence is only checked at `n_max`.
        `n_check_initialisation`: Number of iterations between each convergence
          check in the initialisation, taking positive values. Should be at most
          `n_max_initialisation`. Defaults to `None`; if no
          `n_check_initialisation` is passed, convergence is only checked at
          `n_max_initialisation`.
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
                                          n_max=n_max_initialisation, n_check=n_check_initialisation, tol=tol, dε=dε,
                                          initial_condition=initial_condition)
    
    print("Solving Eikonal PDE data-driven left invariant metric.")
    # Align with (x, y, θ)-frame
    W_init_np = align_to_real_axis_scalar_field(W_init_np)
    cost_np = align_to_real_axis_scalar_field(cost_np)
    shape = cost_np.shape
    source_point = align_to_real_axis_point(source_point, shape)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    ε = dε * (dxy / G_inv.max()) / np.sqrt(3) # * cost_np.min() 
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
    G_inv: ti.types.vector(3, ti.f32),
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
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
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
        dW_dt[I] = (1 - (ti.math.sqrt(
            G_inv[0] * A1_W[I]**2 +
            G_inv[1] * A2_W[I]**2 +
            G_inv[2] * A3_W[I]**2
        ) / cost[I])) * cost[I]
        W[I] += dW_dt[I] * ε # ti.math.max(dW_dt[I] * ε, -W[I]) # 🤢

@ti.kernel
def distance_gradient_field(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
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
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the
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
            G_inv[0] * A1_W[I],
            G_inv[1] * A2_W[I],
            G_inv[2] * A3_W[I]
        ]) / cost[I]**2

# Left invariant

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
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the 
          diagonal metric tensor with respect to left invariant basis.
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
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    print(G_inv)
    ε = dε * (dxy / G_inv.max()) / np.sqrt(3)
    print(f"ε = {ε}")
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
    G_inv: ti.types.vector(3, ti.f32),
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
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
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
            G_inv[0] * A1_W[I]**2 +
            G_inv[1] * A2_W[I]**2 +
            G_inv[2] * A3_W[I]**2
        )
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field_uniform(
    W: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
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
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
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
            G_inv[0] * A1_W[I],
            G_inv[1] * A2_W[I],
            G_inv[2] * A3_W[I]
        ])