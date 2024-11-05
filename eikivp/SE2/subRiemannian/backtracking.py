"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in SE(2). In particular, provides the class
    `GeodesicSE2SubRiemannian`, which can compute the geodesic and store it with
    its parameters.
    
    The primary methods are:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the corresponding methods in the distancemap module.
"""

import numpy as np
import h5py
import taichi as ti
from eikivp.SE2.subRiemannian.interpolate import (
    vectorfield_trilinear_interpolate_LI,
    scalar_trilinear_interpolate
)
from eikivp.SE2.utils import (
    get_next_point,
    coordinate_array_to_real,
    coordinate_real_to_array_ti,
    vector_LI_to_static,
    distance_in_pixels,
    distance_in_pixels_multi_source
)
from eikivp.SE2.costfunction import CostSE2
from eikivp.SE2.subRiemannian.distancemap import (
    DistanceSE2SubRiemannian,
    DistanceMultiSourceSE2SubRiemannian
)

class GeodesicSE2SubRiemannian():
    """
    Compute the geodesic of a sub-Riemannian distance map on SE(2).

    Attributes:
        `γ_path`: np.ndarray of path of geodesic.
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

    def __init__(self, W: DistanceSE2SubRiemannian, target_point=None, dt=1.):
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

    def compute_γ_path(self, W: DistanceSE2SubRiemannian, C: CostSE2, x_min, y_min, θ_min, dxy, dθ, θs_np, n_max=2000):
        self.γ_path = geodesic_back_tracking(W.grad_W, self.source_point, self.target_point, C.C, x_min, y_min, θ_min,
                                             dxy, dθ, θs_np, self.ξ, dt=self.dt, n_max=n_max)

    def import_γ_path(self, folder):
        """
        Import the geodesic matching the attributes `σ_s_list`, `σ_o`,
        `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`, `source_point`, and
        `target_point`.
        """
        geodesic_filename = f"{folder}\\SE2_sR_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_e={self.σ_s_ext}_s_o_e={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_s={self.source_point}_t={self.target_point}.hdf5"
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
        `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`, `source_point`, and
        `target_point``.
        """
        geodesic_filename = f"{folder}\\SE2_sR_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_e={self.σ_s_ext}_s_o_e={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_s={self.source_point}_t={self.target_point}.hdf5"
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

class GeodesicMultiSourceSE2SubRiemannian():
    """
    Compute the geodesic of a sub-Riemannian distance map on SE(2).

    Attributes:
        `γ_path`: np.ndarray of path of geodesic.
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
        `source_points`: Tuple[Tuple[int]] describing index of source points.
        `target_point`: Tuple[int] describing index of target point. Defaults to
          `None`. If `target_point` is provided, the algorithm will terminate
          when the Hamiltonian has converged at `target_point`; otherwise it
          will terminate when the Hamiltonian has converged throughout the
          domain.
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of the cost function.
    """

    def __init__(self, W: DistanceMultiSourceSE2SubRiemannian, target_point=None, dt=1.):
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
        self.source_points = W.source_points
        self.target_point = W.target_point
        if target_point is not None:
            self.target_point = target_point
        # Geodesic attributes
        self.dt = dt

    def compute_γ_path(self, W: DistanceMultiSourceSE2SubRiemannian, C: CostSE2, x_min, y_min, θ_min, dxy, dθ, θs_np, n_max=2000):
        self.γ_path = geodesic_back_tracking_multi_source(W.grad_W, self.source_points, self.target_point, C.C, x_min,
                                                          y_min, θ_min, dxy, dθ, θs_np, self.ξ, dt=self.dt, n_max=n_max)

    def import_γ_path(self, folder):
        """
        Import the geodesic matching the attributes `σ_s_list`, `σ_o`,
        `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`, `source_points`, and
        `target_point`.
        """
        geodesic_filename = f"{folder}\\SE2_sR_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_e={self.σ_s_ext}_s_o_e={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_t={self.target_point}.hdf5"
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
                np.all(self.source_points == geodesic_file.attrs["source_points"]) and
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
        `σ_s_ext`, `σ_o_ext`, `image_name`, `λ`, `p`, `ξ`, `source_points`, and
        `target_point``.
        """
        geodesic_filename = f"{folder}\\SE2_sR_ss_s={[s for s in self.σ_s_list]}_s_o={self.σ_o}_s_s_e={self.σ_s_ext}_s_o_e={self.σ_o_ext}_l={self.λ}_p={self.p}_x={self.ξ}_t={self.target_point}.hdf5"
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
            geodesic_file.attrs["source_points"] = self.source_points
            if self.target_point is None:
                geodesic_file.attrs["target_point"] = "default"
            else:
                geodesic_file.attrs["target_point"] = self.target_point
            if self.dt is None:
                geodesic_file.attrs["dt"] = "default"
            else:
                geodesic_file.attrs["dt"] = self.dt

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
        print(f"source points => {self.source_points}")
        print(f"target point => {self.target_point}")
        print(f"dt => {self.dt}")

# Sub-Riemannian backtracking

def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, x_min, y_min, θ_min, dxy, dθ, θs_np, ξ,
                           dt=1., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described by Bekkers et al.[1]

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map, with shape [Nx, Ny, Nθ, 3].
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny, Nθ].
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
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

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    # We perform backtracking in real coordinates instead of in array indices.
    source_point = coordinate_array_to_real(*source_point, x_min, y_min, θ_min, dxy, dθ)
    target_point = coordinate_array_to_real(*target_point, x_min, y_min, θ_min, dxy, dθ)
    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    # Perform backtracking
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=n_max)

    point = target_point
    γ[0] = point
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    distance = ti.math.inf
    while (distance >= tol) and (n < n_max - 1):
        point = geodesic_back_tracking_step(grad_W, θs, point, ξ, cost, x_min, y_min, θ_min, dxy, dθ, dt)
        distance = distance_in_pixels(point, source_point, dxy, dθ)
        γ[n] = point
        n += 1
    γ_len = n
    print(f"Geodesic consists of {γ_len} points.")
    γ_np = γ.to_numpy()[:γ_len]
    γ_np[-1] = source_point
    return γ_np

def geodesic_back_tracking_multi_source(grad_W_np, source_points, target_point, cost_np, x_min, y_min, θ_min, dxy, dθ,
                                        θs_np, ξ, dt=1., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_points`, using 
    gradient descent back tracking, as described by Bekkers et al.[1]

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map, with shape [Nx, Ny, Nθ, 3].
        `source_points`: Tuple[Tuple[int]] describing index of source point in
          `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny, Nθ].
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
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

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    # We perform backtracking in real coordinates instead of in array indices.
    source_points_np = np.array(tuple(coordinate_array_to_real(*p, x_min, y_min, θ_min, dxy, dθ) for p in source_points))
    N_source_points = len(source_points)
    source_points = ti.Vector.field(n=3, shape=(N_source_points,), dtype=ti.f32)
    source_points.from_numpy(source_points_np)
    target_point = coordinate_array_to_real(*target_point, x_min, y_min, θ_min, dxy, dθ)
    target_point = ti.Vector(target_point, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    # Perform backtracking
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=n_max)
    distances = ti.field(dtype=ti.f32, shape=(N_source_points,))

    point = target_point
    γ[0] = point
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    min_distance = ti.math.inf
    while (min_distance >= tol) and (n < n_max - 1):
        point = geodesic_back_tracking_step(grad_W, θs, point, ξ, cost, x_min, y_min, θ_min, dxy, dθ, dt)
        min_distance = distance_in_pixels_multi_source(point, source_points, dxy, dθ)
        γ[n] = point
        n += 1
    γ_len = n
    print(f"Geodesic consists of {γ_len} points.")
    γ_np = γ.to_numpy()[:γ_len]
    distances = distances.to_numpy()
    γ_np[-1] = source_points_np[np.argmin(distances)]
    return γ_np

@ti.kernel
def geodesic_back_tracking_step(
    grad_W: ti.template(),
    θs: ti.template(),
    point: ti.types.vector(3, ti.f32),
    ξ: ti.f32,
    cost: ti.template(),
    x_min: ti.f32,
    y_min: ti.f32,
    θ_min: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    dt: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.kernel

    Find the geodesic connecting `target_point` to `source_point`, using
    gradient descent backtracking, as described by Bekkers et al.[1]

    Args:
      Static:
        `grad_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ, 3]) of upwind
          gradient with respect to some cost of the approximate distance map.
        `θs`: angle coordinate at each grid point.
        `point`: ti.types.vector(n=3, dtype=[float]) current point.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `cost`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of cost function,
          taking values between 0 and 1.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
        `dt`: Gradient descent step size, taking values greater than 0.

    Returns:
        Next point.
    
    References:
        [1]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # To get the gradient, we need the corresponding array indices.
    point_array = coordinate_real_to_array_ti(point, x_min, y_min, θ_min, dxy, dθ)
    # Get gradient using componentwise trilinear interpolation.
    gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point_array, ξ, cost)
    θ = scalar_trilinear_interpolate(θs, point_array)
    # Get gradient with respect to static frame.
    gradient_at_point = vector_LI_to_static(gradient_at_point_LI, θ)
    new_point = get_next_point(point, gradient_at_point, dxy, dθ, dt)
    return new_point