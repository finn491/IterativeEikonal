"""
    distancemap
    ===========

    Provides methods to compute the distance map on SE(2) with respect to
    various metrics, by solving the Eikonal PDE using the iterative Initial
    Value Problem (IVP) technique described by Bekkers et al.[1] In particular,
    provides the class `DistanceSE2Plus`, which can compute the distance map and
    its gradient, and store them with their parameters.
    
    The primary methods
    are:
      1. `eikonal_solver`: solve the Eikonal PDE with respect to some
      data-driven left invariant plus controller, defined by a stiffness 
      parameter ξ, a plus softness ε, and a cost function. The stiffness 
      parameter ξ fixes the relative cost of moving in the A1-direction compared
      to the A3-direction (it corresponds to β used by Bekkers et al.[1]);
      the plus softness ε restricts the motion in the reverse A1-direction; 
      motion in the A2-direction is inhibited.
      2. `eikonal_solver_uniform`: solve the Eikonal PDE with respect to some
      left invariant plus controller, defined by a stiffness parameter ξ, a plus
      softness ε, and a cost function. The stiffness parameter ξ fixes the
      relative cost of moving in the A1-direction compared to the A3-direction
      (it corresponds to β used by Bekkers et al.[1]); the plus softness ε
      restricts the motion in the reverse A1-direction; motion in the
      A2-direction is inhibited.

    References:
      [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
      "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
      In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
      DOI:10.1137/15M1018460.
"""

import numpy as np
import h5py
import taichi as ti
from tqdm import tqdm
from eikivp.SE2.derivatives import (
    upwind_A1,
    upwind_A3
)
from eikivp.SE2.utils import (
    get_boundary_conditions,
    check_convergence
)
from eikivp.utils import (
    get_initial_W,
    apply_boundary_conditions,
    get_padded_cost,
    unpad_array
)
from eikivp.SE2.costfunction import CostSE2

class DistanceSE2Plus():
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    Finsler function defined by `ξ` and `cost_np`, with source at 
    `source_point`, using the iterative method described by Bekkers et al.[1]

    Attributes:
        `W`: np.ndarray of distance function data.
        `grad_W`: np.ndarray of gradient of distance function data.
        `σ_s_list`: standard deviations in pixels of the internal regularisation
          in the spatial directions before taking derivatives.
        `σ_o`: standard deviation in pixels of the internal regularisation
          in the orientational direction before taking derivatives.
        `σ_s_ext`: standard deviation in pixels of the external regularisation
          in the spatial direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.
        `σ_o_ext`: standard deviation in pixels of the internal regularisation
          in the orientational direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.
        `image_name`: identifier of image used to generate vesselness.
        `λ`: Vesselness prefactor, taking values greater than 0.
        `p`: Vesselness exponent, taking values greater than 0.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `source_point`: Tuple[int] describing index of source point.
        `target_point`: Tuple[int] describing index of target point. Defaults to
          `None`. If `target_point` is provided, the algorithm will terminate
          when the Hamiltonian has converged at `target_point`; otherwise it
          will terminate when the Hamiltonian has converged throughout the
          domain.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """

    def __init__(self, C: CostSE2, ξ, source_point, target_point):
        # Vesselness attributes
        self.σ_s_list = C.σ_s_list
        self.σ_o = C.σ_o
        self.σ_s_ext = C.σ_s_ext
        self.σ_o_ext = C.σ_o_ext
        self.image_name = C.image_name
        # Cost attributes
        self.λ = C.λ
        self.p = C.p
        # Distance attributes
        self.ξ = ξ
        self.source_point = source_point
        self.target_point = target_point

    def compute_W(self, C: CostSE2, dxy, dθ, θs_np, plus_softness=0., n_max=1e5, n_max_initialisation=1e4, n_check=None,
                  n_check_initialisation=None, tol=1e-3, dε=1., initial_condition=100.):
        """
        Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
        Finsler function defined by `ξ` and `cost_np`, with source at 
        `source_point`, using the iterative method described by Bekkers et al.[1]

        Args:
            `cost_np`: np.ndarray of cost function throughout domain, taking values
              between 0 and 1, with shape [Nx, Ny, Nθ].
            `source_point`: Tuple[int] describing index of source point in 
              `cost_np`.
            `ξ`: Stiffness of moving in the A1 direction compared to the A3
              direction, taking values greater than 0.
            `dxy`: Spatial step size, taking values greater than 0.
            `dθ`: Orientational step size, taking values greater than 0.
            `θs_np`: Orientation coordinate at every point in the grid on which
              `cost_np` is sampled.
          Optional:
            `plus_softness`: Strength of the plus controller, taking values between
              0 and 1. As `plus_softness` is decreased, motion in the reverse A1
              direction is increasingly inhibited. For `plus_softness` 0, motion is
              possibly exclusively in the forward A1 direction; for `plus_softness`
              1, we recover the sub-Riemannian metric that is symmetric in the A1
              direction. Defaults to 0.
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
            The base Finsler function (i.e. with uniform cost), is given, for vector
            v = v^i A_i at point p, by 
              F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
            where (x)_+ := max{x, 0} is the positive part of x.
        
        References:
            [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
              "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
              In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
              DOI:10.1137/15M1018460.
        """
        W, grad_W = eikonal_solver(C.C, self.source_point, self.ξ, dxy, dθ, θs_np, plus_softness=plus_softness,
                                   target_point=self.target_point, n_max=n_max,
                                   n_max_initialisation=n_max_initialisation, n_check=n_check,
                                   n_check_initialisation=n_check_initialisation, tol=tol, dε=dε,
                                   initial_condition=initial_condition)
        self.W = W
        self.grad_W = grad_W

    def import_W(self, folder):
        """
        Import the distance and its gradient matching the attributes `σ_s_list`,
        `σ_o`, `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`,
        `source_point`, and `target_point`.
        """
        distance_filename = f"{folder}\\SE2_p_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_ext={self.σ_s_ext}_s_o_ext={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_s={self.source_point}.hdf5"
        with h5py.File(distance_filename, "r") as distance_file:
            assert (
                np.all(self.σ_s_list == distance_file.attrs["σ_s_list"]) and
                self.σ_o == distance_file.attrs["σ_o"] and
                self.σ_s_ext == distance_file.attrs["σ_s_ext"] and
                self.σ_o_ext == distance_file.attrs["σ_o_ext"] and
                self.image_name == distance_file.attrs["image_name"] and
                self.λ == distance_file.attrs["λ"] and
                self.p == distance_file.attrs["p"] and
                self.ξ == distance_file.attrs["ξ"] and
                np.all(self.source_point == distance_file.attrs["source_point"]) and
                (
                    np.all(self.target_point == distance_file.attrs["target_point"]) or
                    distance_file.attrs["target_point"] == "default"
                )
            ), "There is a parameter mismatch!"
            self.W = distance_file["Distance"][()]
            self.grad_W = distance_file["Gradient"][()]
            
    def export_W(self, folder):
        """
        Export the distance and its gradient to hdf5 with attributes `σ_s_list`,
        `σ_o`, `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`,
        `source_point`, and `target_point` stored as metadata.
        """
        distance_filename = f"{folder}\\SE2_p_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_ext={self.σ_s_ext}_s_o_ext={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_s={self.source_point}.hdf5"
        with h5py.File(distance_filename, "w") as distance_file:
            distance_file.create_dataset("Distance", data=self.W)
            distance_file.create_dataset("Gradient", data=self.grad_W)
            distance_file.attrs["σ_s_list"] = self.σ_s_list
            distance_file.attrs["σ_o"] = self.σ_o
            distance_file.attrs["σ_s_ext"] = self.σ_s_ext
            distance_file.attrs["σ_o_ext"] = self.σ_o_ext
            distance_file.attrs["image_name"] = self.image_name
            distance_file.attrs["λ"] = self.λ
            distance_file.attrs["p"] = self.p
            distance_file.attrs["ξ"] = self.ξ
            distance_file.attrs["source_point"] = self.source_point
            if self.target_point is None:
                distance_file.attrs["target_point"] = "default"
            else:
                distance_file.attrs["target_point"] = self.target_point

    def print(self):
        """Print attributes."""
        print(f"σ_s_list => {self.σ_s_list}")
        print(f"σ_o => {self.σ_o}")
        print(f"σ_s_ext => {self.σ_s_ext}")
        print(f"σ_o_ext => {self.σ_o_ext}")
        print(f"image_name => {self.image_name}")
        print(f"λ => {self.λ}")
        print(f"p => {self.p}")
        print(f"ξ => {self.ξ}")
        print(f"source_point => {self.source_point}")
        print(f"target_point => {self.target_point}")

# Data-driven left invariant

def eikonal_solver(cost_np, source_point, ξ, dxy, dθ, θs_np, plus_softness=0., target_point=None, n_max=1e5, 
                   n_max_initialisation=1e4, n_check=None, n_check_initialisation=None, tol=1e-3, dε=1., 
                   initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    Finsler function defined by `ξ` and `cost_np`, with source at 
    `source_point`, using the iterative method described by Bekkers et al.[1]

    Args:
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny, Nθ].
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
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
        The base Finsler function (i.e. with uniform cost), is given, for vector
        v = v^i A_i at point p, by 
          F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
        where (x)_+ := max{x, 0} is the positive part of x.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_uniform(cost_np.shape, source_point, ξ, dxy, dθ, θs_np, plus_softness=plus_softness,
                                          n_max=n_max_initialisation, n_check=n_check_initialisation, tol=tol, dε=dε,
                                          initial_condition=initial_condition)
    
    print("Solving Eikonal PDE data-driven left invariant metric.")

    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    # The sqrt(2) comes from the fact that the norm of the gradient consists of
    # 2 terms.
    ε = dε * (dxy / (1 + ξ**-2)) / np.sqrt(2) # cost_np.min() * 
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
            step_W(W, cost, ξ, plus_softness, dxy, dθ, θs, ε, A1_forward, A1_backward, A3_forward, A3_backward, A1_W, 
                   A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, source_point, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(W, cost, ξ, plus_softness, dxy, dθ, θs, A1_forward, A1_backward, A3_forward, A3_backward,
                            A1_W, A3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W(
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
    method described by Bekkers et al.[1]

    Args:
      Static:
        `cost`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of cost function.
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
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map, which is updated in place.
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `dW_dt`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of error of the
          distance map with respect to the Eikonal PDE, which is updated in
          place.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    upwind_A1(W, dxy, θs, A1_forward, A1_backward, A1_W)
    upwind_A3(W, dθ, A3_forward, A3_backward, A3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = (1 - (ti.math.sqrt(
            soft_plus(A1_W[I], plus_softness)**2 / ξ**2 +
            A3_W[I]**2
        ) / cost[I])) * cost[I]
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field(
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
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map.
        `cost`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of cost function.
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
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives,
          which are updated in place.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `grad_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind
          derivatives of approximate distance map, which is updated inplace.
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

def eikonal_solver_uniform(domain_shape, source_point, ξ, dxy, dθ, θs_np, plus_softness=0., target_point=None, n_max=1e5,
                           n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant 
    Finsler function defined by `ξ`, with source at `source_point`, using the 
    iterative method described by Bekkers et al.[1]

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nx, Ny, Nθ].
        `source_point`: Tuple[int] describing index of source point in 
          `domain_shape`.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
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
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base Finsler function (i.e. with uniform cost), is given, for vector
        v = v^i A_i at point p, by 
          F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
        where (x)_+ := max{x, 0} is the positive part of x.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # Set hyperparameters.
    # Heuristic, so that W does not become negative.
    # The sqrt(2) comes from the fact that the norm of the gradient consists of
    # 2 terms.
    ε = dε * (dxy / (1 + ξ**-2)) / np.sqrt(2)
    print(f"ε = {ε}")
    if n_check is None: # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    W = get_initial_W(domain_shape, initial_condition=initial_condition, pad_shape=((1,), (1,), (0,)))
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
            step_W_uniform(W, ξ, plus_softness, dxy, dθ, θs, ε, A1_forward, A1_backward, A3_forward, A3_backward, A1_W, 
                           A3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, source_point, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_uniform(W, ξ, plus_softness, dxy, dθ, θs, A1_forward, A1_backward, A3_forward, A3_backward,
                                    A1_W, A3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_uniform(
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
    method described by Bekkers et al.[1]

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
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map, which is updated in place.
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `dW_dt`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of error of the
          distance map with respect to the Eikonal PDE, which is updated in
          place.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
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
def distance_gradient_field_uniform(
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
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives,
          which are updated in place.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `grad_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ, 3]) of upwind
          derivatives of approximate distance map, which is updated inplace.
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