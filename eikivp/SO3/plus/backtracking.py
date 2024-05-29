"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in SO(3). The primary methods are:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the corresponding methods in the distancemap module.
"""

import numpy as np
import h5py
import taichi as ti
from eikivp.SO3.plus.interpolate import (
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
from eikivp.SO3.costfunction import CostSO3
from eikivp.SO3.plus.distancemap import DistanceSO3Plus

class GeodesicSO3Plus():
    """
    Compute the geodesic of a plus controller distance map on SO(3).

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
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of the cost function.
    """

    def __init__(self, W: DistanceSO3Plus, target_point=None, dt=None):
        # Vesselness attributes
        self.σ_s_list = W.σ_s_list
        self.σ_o = W.σ_o
        self.σ_s_ext = W.σ_s_ext
        self.σ_o_ext = W.σ_o_ext
        self.image_name = W.image_name
        # Cost attributes
        self.λ = W.λ
        self.p = W.p
        # Distance attributes
        self.ξ = W.ξ
        self.source_point = W.source_point
        self.target_point = W.target_point
        if target_point is not None:
            self.target_point = target_point
        # Geodesic attributes
        self.dt = dt

    def compute_γ_path(self, W: DistanceSO3Plus, C: CostSO3, α_min, β_min, φ_min, dα, dβ, dφ, αs_np, φs_np, n_max=2000):
        self.γ_path = geodesic_back_tracking(W.grad_W, self.source_point, self.target_point, C.C, α_min, β_min, φ_min,
                                             dα, dβ, dφ, αs_np, φs_np, self.ξ, dt=self.dt, n_max=n_max)

    def import_γ_path(self, folder):
        """
        Import the geodesic matching the attributes `σ_s_list`, `σ_o`,
        `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`, `source_point`, and
        `target_point`.
        """
        geodesic_filename = f"{folder}\\SO3_p_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_e={self.σ_s_ext}_s_o_e={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_s={self.source_point}_t={self.target_point}.hdf5"
        with h5py.File(geodesic_filename, "r") as geodesic_file:
            assert (
                np.all(self.σ_s_list == geodesic_filename.attrs["σ_s_list"]) and
                self.σ_o == geodesic_filename.attrs["σ_o"] and
                self.σ_s_ext == geodesic_filename.attrs["σ_s_ext"] and
                self.σ_o_ext == geodesic_filename.attrs["σ_o_ext"] and
                self.image_name == geodesic_file.attrs["image_name"] and
                self.λ == geodesic_file.attrs["λ"] and
                self.p == geodesic_file.attrs["p"] and
                self.ξ == geodesic_file.attrs["ξ"] and
                np.all(self.source_point == geodesic_file.attrs["source_point"]) and
                np.all(self.target_point == geodesic_file.attrs["target_point"]) and
                (
                    self.dt == geodesic_file.attrs["dt"] or
                    geodesic_file.attrs["dt"] == "default"
                )              
            ), "There is a parameter mismatch!"
            self.γ_path = geodesic_file["Geodesic"][()]
            
    def export_γ_path(self, folder):
        """
        Export the geodesic to hdf5 with attributes `σ_s_list`, `σ_o`,
        `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`, `source_point`, and `target_point``.
        """
        geodesic_filename = f"{folder}\\SO3_p_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_e={self.σ_s_ext}_s_o_e={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_s={self.source_point}_t={self.target_point}.hdf5"
        with h5py.File(geodesic_filename, "w") as geodesic_file:
            geodesic_file.create_dataset("Geodesic", data=self.γ_path)
            geodesic_file.attrs["σ_s_list"] = self.σ_s_list
            geodesic_file.attrs["σ_o"] = self.σ_o
            geodesic_file.attrs["σ_s_ext"] = self.σ_s_ext
            geodesic_file.attrs["σ_o_ext"] = self.σ_o_ext
            geodesic_file.attrs["image_name"] = self.image_name
            geodesic_file.attrs["λ"] = self.λ
            geodesic_file.attrs["p"] = self.p
            geodesic_file.attrs["ξ"] = self.ξ
            geodesic_file.attrs["source_point"] = self.source_point
            if self.target_point is None:
                geodesic_file.attrs["target_point"] = "default"
            else:
                geodesic_file.attrs["target_point"] = self.target_point
            if self.dt is None:
                geodesic_file.attrs["dt"] = "default"
            else:
                geodesic_file.attrs["dt"] = self.dt

    # def plot(self, x_min, x_max, y_min, y_max):
    #     """Quick visualisation of distance map."""
    #     fig, ax, cbar = plot_image_array(-self.V, x_min, x_max, y_min, y_max)
    #     fig.colorbar(cbar, ax=ax);

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
        print(f"dt => {self.dt}")

def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, α_min, β_min, φ_min, dα, dβ, dφ, αs_np,
                           φs_np, ξ, dt=None, β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described by Bekkers et al.[1]

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
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # Set hyperparameters
    shape = grad_W_np.shape[0:-1]
    if dt is None:
        # It would make sense to also include ξ somehow, but I am not sure how.
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

    γ_len = geodesic_back_tracking_backend(grad_W, source_point, target_point, αs, φs, ξ, cost, α_min, β_min, φ_min, dα,
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
    ξ: ti.f32,
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
    gradient descent backtracking, as described by Bekkers et al.[1]

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
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
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
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    point = target_point
    γ[0] = point
    # To get the gradient, we need the corresponding array indices.
    point_array = coordinate_real_to_array_ti(point, α_min, β_min, φ_min, dα, dβ, dφ)
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    # Get gradient using componentwise trilinear interpolation.
    gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point_array, ξ, cost)
    α = scalar_trilinear_interpolate(αs, point_array)
    φ = scalar_trilinear_interpolate(φs, point_array)
    # Get gradient with respect to static frame.
    gradient_at_point = vector_LI_to_static(gradient_at_point_LI, α, φ)
    while (distance_in_pixels(point - source_point, dα, dβ, dφ) >= tol) and (n < n_max - 1):
        # Get gradient using componentwise trilinear interpolation.
        gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point_array, ξ, cost)
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