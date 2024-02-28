"""
    distancemap
    ============

    Provides methods to compute the distance map on SO(2) with respect to a
    data-driven left invariant sub-Riemannian metric, by solving the Eikonal PDE
    using the iterative Initial Value Problem (IVP) technique described in
    Bekkers et al. "A PDE approach to Data-Driven Sub-Riemannian Geodesics in
    SE(2)" (2015). The primary methods are:
      1. `eikonal_solver`: solve the Eikonal PDE with respect to
      some data-driven left invariant sub-Riemannian metric, defined by a 
      stiffness parameter ξ a cost function. The stiffness parameter ξ fixes the
      relative cost of moving in the B1-direction compared to the B3-direction
      (it corresponds to β in the paper by Bekkers et al.); motion in the 
      B2-direction is inhibited.
      2. `eikonal_solver_uniform`: solve the Eikonal PDE with respect to
      some left invariant sub-Riemannian metric, defined by a stiffness
      parameter ξ a cost function. The stiffness parameter ξ fixes the relative
      cost of moving in the B1-direction compared to the B3-direction (it
      corresponds to β in the paper by Bekkers et al.); motion in the
      B2-direction is inhibited.
"""

import numpy as np
import taichi as ti
from tqdm import tqdm
from eikivp.SO3.derivatives import (
    upwind_B1,
    upwind_B3
)
from eikivp.SO3.utils import (
    get_boundary_conditions,
    check_convergence
)
from eikivp.utils import (
    get_initial_W,
    apply_boundary_conditions,
    get_padded_cost,
    unpad_array
)

# Data-driven left invariant

def eikonal_solver(cost_np, source_point, ξ, dα, dβ, dφ, αs_np, φs_np, target_point=None, n_max=1e5,
                   n_max_initialisation=1e4, n_check=None, n_check_initialisation=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    sub-Riemannian metric tensor field defined by `ξ` and `cost_np`, with source
    at `source_point`, using the iterative method described in Bekkers et al. 
    "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1.
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost_np` is
          sampled.
        `φs_np`: Orientation co
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

    Notes:
        The base metric tensor field (i.e. with uniform cost), is given, for a
        pair of vectors v = v^i A_i and w = w^i A_i at point p, by 
          G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
    """
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_uniform(cost_np.shape, source_point, ξ, dα, dβ, dφ, αs_np, φs_np,
                                          target_point=target_point, n_max=n_max_initialisation,
                                          n_check=n_check_initialisation, tol=tol, dε=dε,
                                          initial_condition=initial_condition)
    
    print("Solving Eikonal PDE data-driven left invariant metric.")
    # Align with (x, y, θ)-frame

    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    # The sqrt(2) comes from the fact that the norm of the gradient consists of
    # 2 terms.
    ε = dε * (min(dα, dβ, dφ) / (1 + ξ**-2)) / np.sqrt(2) # * cost_np.min()
    print(f"ε = {ε}")
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    cost = get_padded_cost(cost_np, pad_shape=((1,), (1,), (0,)))
    W = get_padded_cost(W_init_np, pad_shape=((1,), (1,), (0,)), pad_value=initial_condition)
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    αs = ti.field(dtype=ti.f32, shape=αs_np.shape)
    αs.from_numpy(αs_np)
    φs = ti.field(dtype=ti.f32, shape=φs_np.shape)
    φs.from_numpy(φs_np)

    B1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B1_W = ti.field(dtype=ti.f32, shape=W.shape)
    B3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W(W, cost, ξ, dα, dβ, dφ, αs_np, φs_np, ε, B1_forward, B1_backward, B3_forward, B3_backward, B1_W,
                   B3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(W, cost, ξ, dα, dβ, dφ, αs_np, φs_np, B1_forward, B1_backward, B3_forward, B3_backward,
                            B1_W, B3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W(
    W: ti.template(),
    cost: ti.template(),
    ξ: ti.f32,
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    ε: ti.f32,
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B3_W: ti.template(),
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
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map, 
          which is updated in place.
        `B*_*`: ti.field(dtype=[float], shape=shape) of derivatives.
        `B*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `dW_dt`: ti.field(dtype=[float], shape=shape) of error of the distance 
          map with respect to the Eikonal PDE, which is updated in place.
    """
    upwind_B1(W, dα, dβ, dφ, αs, B1_forward, B1_backward, B1_W)
    upwind_B3(W, dφ, B3_forward, B3_backward, B3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = (1 - (ti.math.sqrt(
            B1_W[I]**2 / ξ**2 +
            B3_W[I]**2 
        ) / cost[I])) * cost[I]
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field(
    W: ti.template(),
    cost: ti.template(),
    ξ: ti.f32,
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B3_W: ti.template(),
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
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
      Mutated:
        `B*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `B*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_B1(W, dα, dβ, dφ, αs, B1_forward, B1_backward, B1_W)
    upwind_B3(W, dφ, B3_forward, B3_backward, B3_W)
    for I in ti.grouped(B1_W):
        grad_W[I] = ti.Vector([
            B1_W[I] / ξ**2,
            0.,
            B3_W[I]
        ]) / cost[I]**2

# Left invariant

def eikonal_solver_uniform(domain_shape, source_point, ξ, dα, dβ, dφ, αs_np, φs_np, target_point=None, n_max=1e5,
                           n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SO(3) equipped with a datadriven left invariant 
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
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost_np` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `domain_shape`. Defaults to `None`. If `target_point` is provided, the
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
    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    # The sqrt(2) comes from the fact that the norm of the gradient consists of
    # 2 terms.
    ε = dε * (min(dα, dβ, dφ) / (1 + ξ**-2)) / np.sqrt(2)
    print(f"ε = {ε}")
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    W = get_initial_W(domain_shape, initial_condition=initial_condition, pad_shape=((1,), (1,), (0,)))
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    αs = ti.field(dtype=ti.f32, shape=αs_np.shape)
    αs.from_numpy(αs_np)
    φs = ti.field(dtype=ti.f32, shape=φs_np.shape)
    φs.from_numpy(φs_np)

    B1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B1_W = ti.field(dtype=ti.f32, shape=W.shape)
    B3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_uniform(W, ξ, dα, dβ, dφ, αs, φs, ε, B1_forward, B1_backward, B3_forward, B3_backward, B1_W, B3_W,
                           dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_uniform(W, ξ, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B3_forward, B3_backward, B1_W,
                                    B3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_uniform(
    W: ti.template(),
    ξ: ti.f32,
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    ε: ti.f32,
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B3_W: ti.template(),
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
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=shape) of approximate distance map, 
          which is updated in place.
        `B*_*`: ti.field(dtype=[float], shape=shape) of derivatives.
        `B*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `dW_dt`: ti.field(dtype=[float], shape=shape) of error of the distance 
          map with respect to the Eikonal PDE, which is updated in place.
    """
    upwind_B1(W, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B1_W)
    upwind_B3(W, dφ, B3_forward, B3_backward, B3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - ti.math.sqrt(
            B1_W[I]**2 / ξ**2 +
            B3_W[I]**2 
        )
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field_uniform(
    W: ti.template(),
    ξ: ti.f32,
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B3_W: ti.template(),
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
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
          updated in place.
        `A*_W`: ti.field(dtype=[float], shape=shape) of upwind derivative of the 
          approximate distance map in the A* direction, which is updated in 
          place.
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind derivatives of 
          approximate distance map, which is updated inplace.
    """
    upwind_B1(W, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B1_W)
    upwind_B3(W, dφ, B3_forward, B3_backward, B3_W)
    for I in ti.grouped(B1_W):
        grad_W[I] = ti.Vector([
            B1_W[I] / ξ**2,
            0.,
            B3_W[I]
        ])