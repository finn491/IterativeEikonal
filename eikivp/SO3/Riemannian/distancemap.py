"""
    distancemap
    ===========

    Provides methods to compute the distance map on SO(3) with respect to a
    data-driven left invariant Riemannian metric, by solving the Eikonal PDE
    using the iterative Initial Value Problem (IVP) technique described by
    Bekkers et al.[1] In particular, provides the class `DistanceSO3Riemannian`,
    which can compute the distance map and its gradient, and store them with
    their parameters.
    
    The primary methods are:
      1. `eikonal_solver`: solve the Eikonal PDE with respect to some 
      data-driven left invariant Riemannian metric, defined by the diagonal
      components of the underlying left invariant metric, with respect to the
      left invariant basis {B1, B2, B3}, and a cost function.
      2. `eikonal_solver_uniform`: solve the Eikonal PDE with respect to some 
      left invariant Riemannian metric, defined by its diagonal components, with
      respect to the left invariant basis {B1, B2, B3}.
    
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
from eikivp.SO3.derivatives import (
    upwind_derivatives,
)
from eikivp.SO3.utils import (
    get_boundary_conditions,
    check_convergence
)
from eikivp.SO3.Riemannian.metric import (
    invert_metric
)
from eikivp.utils import (
    get_initial_W,
    apply_boundary_conditions,
    get_padded_cost,
    unpad_array
)
from eikivp.SO3.costfunction import CostSO3

class DistanceSO3Riemannian():
    """
    Solve the Eikonal PDE on SO(3) equipped with a datadriven left invariant
    Riemannian metric tensor field defined by `G_np` and `cost_np`, with source
    at `source_point`, using the iterative method described by Bekkers et al.[1]

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
        `G`: np.ndarray(shape=(2,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to standard basis. Defaults to
          standard Euclidean metric.
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

    def __init__(self, C: CostSO3, G, source_point, target_point):
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
        self.G = G
        self.source_point = source_point
        self.target_point = target_point

    def compute_W(self, C: CostSO3, dα, dβ, dφ, αs_np, φs_np, n_max=1e5, n_max_initialisation=1e4, n_check=None,
                  n_check_initialisation=None, tol=1e-3, dε=1., initial_condition=100.):
        """
        Solve the Eikonal PDE on SO(3) equipped with a datadriven left invariant
        Riemannian metric tensor field defined by `G_np` and `cost_np`, with source
        at `source_point`, using the iterative method described by Bekkers et al.[1]

        Args:
            `cost_np`: np.ndarray of cost function throughout domain, taking values
              between 0 and 1, with shape [Nα, Nβ, Nφ].
            `source_point`: Tuple[int] describing index of source point in 
              `cost_np`.
            `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
              diagonal metric tensor with respect to left invariant basis.
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
        
        References:
            [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
              "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
              In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
              DOI:10.1137/15M1018460.
        """
        W, grad_W = eikonal_solver(C.C, self.source_point, self.G, dα, dβ, dφ, αs_np, φs_np,
                                   target_point=self.target_point, n_max=n_max,
                                   n_max_initialisation=n_max_initialisation, n_check=n_check,
                                   n_check_initialisation=n_check_initialisation, tol=tol, dε=dε,
                                   initial_condition=initial_condition)
        self.W = W
        self.grad_W = grad_W

    def import_W(self, folder):
        """
        Import the distance and its gradient matching the attributes `σ_s_list`,
        `σ_o`, `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `G`,
        `source_point`, and `target_point`.
        """
        distance_filename = f"{folder}\\SO3_R_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_ext={self.σ_s_ext}_s_o_ext={self.σ_o_ext}_l={self.λ}_p={self.p}_G={[g for g in self.G]}_s={self.source_point}.hdf5"
        with h5py.File(distance_filename, "r") as distance_file:
            assert (
                np.all(self.σ_s_list == distance_file.attrs["σ_s_list"]) and
                self.σ_o == distance_file.attrs["σ_o"] and
                self.σ_s_ext == distance_file.attrs["σ_s_ext"] and
                self.σ_o_ext == distance_file.attrs["σ_o_ext"] and
                self.image_name == distance_file.attrs["image_name"] and
                self.λ == distance_file.attrs["λ"] and
                self.p == distance_file.attrs["p"] and
                np.all(self.G == distance_file.attrs["G"]) and
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
        distance_filename = f"{folder}\\SO3_R_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_ext={self.σ_s_ext}_s_o_ext={self.σ_o_ext}_l={self.λ}_p={self.p}_G={[g for g in self.G]}_s={self.source_point}.hdf5"
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
            distance_file.attrs["G"] = self.G
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

def eikonal_solver(cost_np, source_point, G_np, dα, dβ, dφ, αs_np, φs_np, target_point=None, n_max=1e5,
                   n_max_initialisation=1e4, n_check=None, n_check_initialisation=None, tol=1e-3, dε=1.,
                   initial_condition=100.):
    """
    Solve the Eikonal PDE on SO(3) equipped with a datadriven left invariant
    Riemannian metric tensor field defined by `G_np` and `cost_np`, with source
    at `source_point`, using the iterative method described by Bekkers et al.[1]

    Args:
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `source_point`: Tuple[int] describing index of source point in 
          `cost_np`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
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
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_uniform(cost_np.shape, source_point, G_np, dα, dβ, dφ, αs_np, φs_np,
                                          target_point=target_point, n_max=n_max_initialisation,
                                          n_check=n_check_initialisation, tol=tol, dε=dε,
                                          initial_condition=initial_condition)
    
    print("Solving Eikonal PDE data-driven left invariant metric.")

    # Set hyperparameters
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    ε = dε * (min(dα, dβ, dφ) / G_inv.max()) / np.sqrt(3) # * cost_np.min() 
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
    B2_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B2_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B1_W = ti.field(dtype=ti.f32, shape=W.shape)
    B2_W = ti.field(dtype=ti.f32, shape=W.shape)
    B3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W(W, cost, G_inv, dα, dβ, dφ, αs, φs, ε, B1_forward, B1_backward, B2_forward, B2_backward,
                   B3_forward, B3_backward, B1_W, B2_W, B3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, source_point, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(W, cost, G_inv, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward,
                            B3_forward, B3_backward, B1_W, B2_W, B3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    ε: ti.f32,
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B2_W: ti.template(),
    B3_W: ti.template(),
    dW_dt: ti.template()
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative 
    method described by Bekkers et al.[1]

    Args:
      Static:
        `cost`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of cost function.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of approximate distance
          map, which is updated in place.
        `B*_*`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of derivatives.
        `B*_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of upwind derivative
          of the approximate distance map in the B* direction, which is updated
          in place.
        `dW_dt`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of error of the
          distance map with respect to the Eikonal PDE, which is updated in
          place.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    upwind_derivatives(W, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward, B3_forward, B3_backward,
                       B1_W, B2_W, B3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = (1 - (ti.math.sqrt(
            G_inv[0] * B1_W[I]**2 +
            G_inv[1] * B2_W[I]**2 +
            G_inv[2] * B3_W[I]**2
        ) / cost[I])) * cost[I]
        W[I] += dW_dt[I] * ε # ti.math.max(dW_dt[I] * ε, -W[I]) # 🤢

@ti.kernel
def distance_gradient_field(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B2_W: ti.template(),
    B3_W: ti.template(),
    grad_W: ti.template()
):
    """
    @taichi.kernel

    Compute the gradient with respect to `cost` of the (approximate) distance
    map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of approximate distance
          map.
        `cost`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of cost function.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
      Mutated:
        `B*_*`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of derivatives,
          which are updated in place.
        `B*_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of upwind derivative
          of the approximate distance map in the B* direction, which is updated
          in place.
        `grad_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ, 3]) of upwind
          derivatives of approximate distance map, which is updated inplace.
    """
    upwind_derivatives(W, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward, B3_forward, B3_backward,
                       B1_W, B2_W, B3_W)
    for I in ti.grouped(B1_W):
        grad_W[I] = ti.Vector([
            G_inv[0] * B1_W[I],
            G_inv[1] * B2_W[I],
            G_inv[2] * B3_W[I]
        ]) / cost[I]**2

# Left invariant

def eikonal_solver_uniform(domain_shape, source_point, G_np, dα, dβ, dφ, αs_np, φs_np, target_point=None, n_max=1e5,
                           n_check=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SO(3) equipped with a datadriven left invariant 
    metric tensor field defined by `G_np`, with source at `source_point`, using
    the iterative method described by Bekkers et al.[1]

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nα, Nβ, Nφ].
        `source_point`: Tuple[int] describing index of source point in 
          `domain_shape`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the 
          diagonal metric tensor with respect to left invariant basis.
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
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # Set hyperparameters.
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    ε = dε * (min(dα, dβ, dφ) / G_inv.max()) / np.sqrt(3) # * cost_np.min()
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
    B2_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B2_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    B3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    B1_W = ti.field(dtype=ti.f32, shape=W.shape)
    B2_W = ti.field(dtype=ti.f32, shape=W.shape)
    B3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_uniform(W, G_inv, dα, dβ, dφ, αs, φs, ε, B1_forward, B1_backward, B2_forward, B2_backward,
                           B3_forward, B3_backward, B1_W, B2_W, B3_W, dW_dt)
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(dW_dt, source_point, tol=tol, target_point=target_point)
        if is_converged: # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_uniform(W, G_inv, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward,
                                    B3_forward, B3_backward, B1_W, B2_W, B3_W, grad_W)

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(grad_W_np, pad_shape=(1, 1, 0, 0))

@ti.kernel
def step_W_uniform(
    W: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    ε: ti.f32,
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B2_W: ti.template(),
    B3_W: ti.template(),
    dW_dt: ti.template()
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative 
    method described by Bekkers et al.[1]

    Args:
      Static:
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of approximate distance
          map, which is updated in place.
        `B*_*`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of derivatives.
        `B*_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of upwind derivative
          of the approximate distance map in the B* direction, which is updated
          in place.
        `dW_dt`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of error of the
          distance map with respect to the Eikonal PDE, which is updated in
          place.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    upwind_derivatives(W, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward, B3_forward, B3_backward,
                       B1_W, B2_W, B3_W)
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - ti.math.sqrt(
            G_inv[0] * B1_W[I]**2 +
            G_inv[1] * B2_W[I]**2 +
            G_inv[2] * B3_W[I]**2
        )
        W[I] += dW_dt[I] * ε

@ti.kernel
def distance_gradient_field_uniform(
    W: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    αs: ti.template(),
    φs: ti.template(),
    B1_forward: ti.template(),
    B1_backward: ti.template(),
    B2_forward: ti.template(),
    B2_backward: ti.template(),
    B3_forward: ti.template(),
    B3_backward: ti.template(),
    B1_W: ti.template(),
    B2_W: ti.template(),
    B3_W: ti.template(),
    grad_W: ti.template()
):
    """
    @taichi.kernel

    Compute the gradient of the (approximate) distance map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of approximate distance
          map.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
        `dα`: step size in spatial α-direction, taking values greater than 0.
        `dβ`: step size in spatial β-direction, taking values greater than 0.
        `dφ`: Orientational step size, taking values greater than 0.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
      Mutated:
        `B*_*`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of derivatives,
          which are updated in place.
        `B*_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of upwind derivative
          of the approximate distance map in the B* direction, which is updated
          in place.
        `grad_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ, 3]) of upwind
          derivatives of approximate distance map, which is updated inplace.
    """
    upwind_derivatives(W, dα, dβ, dφ, αs, φs, B1_forward, B1_backward, B2_forward, B2_backward, B3_forward, B3_backward,
                       B1_W, B2_W, B3_W)
    for I in ti.grouped(B1_W):
        grad_W[I] = ti.Vector([
            G_inv[0] * B1_W[I],
            G_inv[1] * B2_W[I],
            G_inv[2] * B3_W[I]
        ])