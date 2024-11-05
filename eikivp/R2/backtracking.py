"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in R^2. In particular, provides the class
    `GeodesicR2`, which can compute the geodesic and store it with its
    parameters.
    
    The primary method is:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the methods in the distancemap module.
"""

import numpy as np
import h5py
import taichi as ti
from eikivp.R2.interpolate import vectorfield_bilinear_interpolate
from eikivp.R2.utils import (
    coordinate_array_to_real,
    coordinate_real_to_array_ti
)
from eikivp.R2.costfunction import CostR2
from eikivp.R2.distancemap import (
    DistanceR2,
    DistanceMultiSourceR2
)

class GeodesicR2():
    """
    Compute the geodesic of a distance map on R2.

    Attributes:
        `γ_path`: np.ndarray of path of geodesic.
        `scales`: iterable of standard deviations of Gaussian derivatives,
          taking values greater than 0. 
        `α`: anisotropy penalty, taking values between 0 and 1.
        `γ`: variance sensitivity, taking values between 0 and 1.
        `ε`: structure penalty, taking values between 0 and 1.
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
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of the cost function.
    """

    def __init__(self, W: DistanceR2, target_point=None, dt=1.):
        # Vesselness attributes
        self.scales = W.scales
        self.α = W.α
        self.γ = W.γ
        self.ε = W.ε
        self.image_name = W.image_name
        # Cost attributes
        self.λ = W.λ
        self.p = W.p
        # Distance attributes
        self.G = W.G
        self.source_point = W.source_point
        self.target_point = W.target_point
        if target_point is not None:
            self.target_point = target_point
        # Geodesic attributes
        self.dt = dt

    def compute_γ_path(self, W: DistanceR2, C: CostR2, x_min, y_min, dxy=1., n_max=2000):
        self.γ_path = geodesic_back_tracking(W.grad_W, self.source_point, self.target_point, C.C, x_min, y_min, dxy,
                                             G_np=self.G, dt=self.dt, n_max=n_max)

    def import_γ_path(self, folder):
        """
        Import the geodesic matching the attributes `scales`, `α`, `γ`, `ε`,
        `image_name`, `λ`, `p`, `G`, `source_point`, and `target_point`.
        """
        geodesic_filename = f"{folder}\\R2_ss={[s for s in self.scales]}_a={self.α}_g={self.γ}_e={self.ε}_l={self.λ}_p={self.p}_G={[g for g in self.G]}_s={self.source_point}_t={self.target_point}.hdf5"
        with h5py.File(geodesic_filename, "r") as geodesic_file:
            assert (
                np.all(self.scales == geodesic_file.attrs["scales"]) and
                self.α == geodesic_file.attrs["α"] and
                self.γ == geodesic_file.attrs["γ"] and
                self.ε == geodesic_file.attrs["ε"] and
                self.image_name == geodesic_file.attrs["image_name"] and
                self.λ == geodesic_file.attrs["λ"] and
                self.p == geodesic_file.attrs["p"] and
                np.all(self.G == geodesic_file.attrs["G"]) and
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
        Export the geodesic to hdf5 with attributes `scales`, `α`, `γ`, `ε`,
        `image_name`, `λ`, `p`, `G`, `source_point`, and `target_point`.
        """
        geodesic_filename = f"{folder}\\R2_ss={[s for s in self.scales]}_a={self.α}_g={self.γ}_e={self.ε}_l={self.λ}_p={self.p}_G={[g for g in self.G]}_s={self.source_point}_t={self.target_point}.hdf5"
        with h5py.File(geodesic_filename, "w") as geodesic_file:
            geodesic_file.create_dataset("Geodesic", data=self.γ_path)
            geodesic_file.attrs["scales"] = self.scales
            geodesic_file.attrs["α"] = self.α
            geodesic_file.attrs["γ"] = self.γ
            geodesic_file.attrs["ε"] = self.ε
            geodesic_file.attrs["image_name"] = self.image_name
            geodesic_file.attrs["λ"] = self.λ
            geodesic_file.attrs["p"] = self.p
            geodesic_file.attrs["G"] = self.G
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
        print(f"scales => {self.scales}")
        print(f"α => {self.α}")
        print(f"γ => {self.γ}")
        print(f"ε => {self.ε}")
        print(f"image_name => {self.image_name}")
        print(f"λ => {self.λ}")
        print(f"p => {self.p}")
        print(f"G => {self.G}")
        print(f"source_point => {self.source_point}")
        print(f"target_point => {self.target_point}")
        print(f"dt => {self.dt}")

class GeodesicMultiSourceR2():
    """
    Compute the geodesic of a distance map on R2.

    Attributes:
        `γ_path`: np.ndarray of path of geodesic.
        `scales`: iterable of standard deviations of Gaussian derivatives,
          taking values greater than 0. 
        `α`: anisotropy penalty, taking values between 0 and 1.
        `γ`: variance sensitivity, taking values between 0 and 1.
        `ε`: structure penalty, taking values between 0 and 1.
        `image_name`: identifier of image used to generate vesselness.
        `λ`: Vesselness prefactor, taking values greater than 0.
        `p`: Vesselness exponent, taking values greater than 0.
        `G`: np.ndarray(shape=(2,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to standard basis. Defaults to
          standard Euclidean metric.
        `source_points`: Tuple[Tuple[int]] describing index of source points.
        `target_point`: Tuple[int] describing index of target point. Defaults to
          `None`. If `target_point` is provided, the algorithm will terminate
          when the Hamiltonian has converged at `target_point`; otherwise it
          will terminate when the Hamiltonian has converged throughout the
          domain.
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of the cost function.
    """

    def __init__(self, W: DistanceMultiSourceR2, target_point=None, dt=1.):
        # Vesselness attributes
        self.scales = W.scales
        self.α = W.α
        self.γ = W.γ
        self.ε = W.ε
        self.image_name = W.image_name
        # Cost attributes
        self.λ = W.λ
        self.p = W.p
        # Distance attributes
        self.G = W.G
        self.source_points = W.source_points
        self.target_point = W.target_point
        if target_point is not None:
            self.target_point = target_point
        # Geodesic attributes
        self.dt = dt

    def compute_γ_path(self, W: DistanceMultiSourceR2, C: CostR2, x_min, y_min, dxy=1., n_max=2000):
        self.γ_path = geodesic_back_tracking_multi_source(W.grad_W, self.source_points, self.target_point, C.C, x_min,
                                                          y_min, dxy, G_np=self.G, dt=self.dt, n_max=n_max)

    def import_γ_path(self, folder):
        """
        Import the geodesic matching the attributes `scales`, `α`, `γ`, `ε`,
        `image_name`, `λ`, `p`, `G`, `source_points`, and `target_point`.
        """
        geodesic_filename = f"{folder}\\R2_ss={[s for s in self.scales]}_a={self.α}_g={self.γ}_e={self.ε}_l={self.λ}_p={self.p}_G={[g for g in self.G]}_t={self.target_point}.hdf5"
        with h5py.File(geodesic_filename, "r") as geodesic_file:
            assert (
                np.all(self.scales == geodesic_file.attrs["scales"]) and
                self.α == geodesic_file.attrs["α"] and
                self.γ == geodesic_file.attrs["γ"] and
                self.ε == geodesic_file.attrs["ε"] and
                self.image_name == geodesic_file.attrs["image_name"] and
                self.λ == geodesic_file.attrs["λ"] and
                self.p == geodesic_file.attrs["p"] and
                np.all(self.G == geodesic_file.attrs["G"]) and
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
        Export the geodesic to hdf5 with attributes `scales`, `α`, `γ`, `ε`,
        `image_name`, `λ`, `p`, `G`, `source_points`, and `target_point`.
        """
        geodesic_filename = f"{folder}\\R2_ss={[s for s in self.scales]}_a={self.α}_g={self.γ}_e={self.ε}_l={self.λ}_p={self.p}_G={[g for g in self.G]}_t={self.target_point}.hdf5"
        with h5py.File(geodesic_filename, "w") as geodesic_file:
            geodesic_file.create_dataset("Geodesic", data=self.γ_path)
            geodesic_file.attrs["scales"] = self.scales
            geodesic_file.attrs["α"] = self.α
            geodesic_file.attrs["γ"] = self.γ
            geodesic_file.attrs["ε"] = self.ε
            geodesic_file.attrs["image_name"] = self.image_name
            geodesic_file.attrs["λ"] = self.λ
            geodesic_file.attrs["p"] = self.p
            geodesic_file.attrs["G"] = self.G
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
        print(f"scales => {self.scales}")
        print(f"α => {self.α}")
        print(f"γ => {self.γ}")
        print(f"ε => {self.ε}")
        print(f"image_name => {self.image_name}")
        print(f"λ => {self.λ}")
        print(f"p => {self.p}")
        print(f"G => {self.G}")
        print(f"source points => {self.source_points}")
        print(f"target point => {self.target_point}")
        print(f"dt => {self.dt}")

def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, x_min, y_min, dxy, G_np=None, dt=1.,
                           n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al.[1]

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map, with shape [Nx, Ny, 2]
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny]
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
      Optional:
        `G_np`: np.ndarray(shape=(2,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to standard basis. Defaults to
          standard Euclidean metric.
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of `cost_np`.
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
    if G_np is None:
        G_np = np.ones(2)
    G = ti.Vector(G_np, ti.f32)

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    # We perform backtracking in real coordinates instead of in array indices.
    source_point = coordinate_array_to_real(*source_point, x_min, y_min, dxy)
    target_point = coordinate_array_to_real(*target_point, x_min, y_min, dxy)
    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)

    # Perform backtracking
    γ = ti.Vector.field(n=2, dtype=ti.f32, shape=n_max)

    point = target_point
    γ[0] = point
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    distance = ti.math.inf
    while (distance >= tol) and (n < n_max - 1):
        point = geodesic_back_tracking_step(grad_W, point, G, cost, x_min, y_min, dxy, dt)
        distance = distance_in_pixels(point, source_point, dxy)
        γ[n] = point
        n += 1
    γ_len = n
    print(f"Geodesic consists of {γ_len} points.")
    γ_np = γ.to_numpy()[:γ_len]
    γ_np[-1] = source_point
    return γ_np

def geodesic_back_tracking_multi_source(grad_W_np, source_points, target_point, cost_np, x_min, y_min, dxy, G_np=None,
                                        dt=1., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_points`, using 
    gradient descent back tracking, as described in Bekkers et al.[1]

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map, with shape [Nx, Ny, 2]
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny]
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
      Optional:
        `G_np`: np.ndarray(shape=(2,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to standard basis. Defaults to
          standard Euclidean metric.
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of `cost_np`.
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
    if G_np is None:
        G_np = np.ones(2)
    G = ti.Vector(G_np, ti.f32)

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    # We perform backtracking in real coordinates instead of in array indices.
    source_points_np = np.array(tuple(coordinate_array_to_real(*p, x_min, y_min, dxy) for p in source_points))
    N_source_points = len(source_points)
    source_points = ti.Vector.field(n=2, shape=(N_source_points,), dtype=ti.f32)
    source_points.from_numpy(source_points_np)
    target_point = coordinate_array_to_real(*target_point, x_min, y_min, dxy)
    target_point = ti.Vector(target_point, dt=ti.f32)

    # Perform backtracking
    γ = ti.Vector.field(n=2, dtype=ti.f32, shape=n_max)
    distances = ti.field(dtype=ti.f32, shape=(N_source_points,))

    point = target_point
    γ[0] = point
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    min_distance = ti.math.inf
    while (min_distance >= tol) and (n < n_max - 1):
        point = geodesic_back_tracking_step(grad_W, point, G, cost, x_min, y_min, dxy, dt)
        min_distance = distance_in_pixels_multi_source(point, source_points, distances, dxy)
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
    point: ti.types.vector(2, ti.f32),
    G: ti.types.vector(2, ti.f32),
    cost: ti.template(),
    x_min: ti.f32,
    y_min: ti.f32,
    dxy: ti.f32,
    dt: ti.f32,
) -> ti.types.vector(2, ti.f32):
    """
    @taichi.kernel

    Find the geodesic connecting `target_point` to `source_points`, using
    gradient descent backtracking, as described in Bekkers et al.[1]

    Args:
      Static:
        `grad_W`: ti.field(dtype=[float], shape=[Nx, Ny, 2]) of upwind gradient
          with respect to some cost of the approximate distance map.
        `point`: ti.types.vector(n=2, dtype=[float]) current point.
        `G`: ti.types.vector(n=2, dtype=[float]) of constants of the diagonal
          metric tensor with respect to standard basis.
        `cost`: ti.field(dtype=[float], shape=[Nx, Ny]) of cost function, taking
          values between 0 and 1.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
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
    point_array = coordinate_real_to_array_ti(point, x_min, y_min, dxy)
    # Get gradient using componentwise bilinear interpolation.
    gradient_at_point = vectorfield_bilinear_interpolate(grad_W, point_array, G, cost)
    new_point = get_next_point(point, gradient_at_point, dxy, dt)
    return new_point

@ti.func
def get_next_point(
    point: ti.types.vector(n=2, dtype=ti.f32),
    gradient_at_point: ti.types.vector(n=2, dtype=ti.f32),
    dxy: ti.f32,
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
    gradient_norm_l2 = norm_l2(gradient_at_point, dxy)
    new_point[0] = point[0] - dt * gradient_at_point[0] / gradient_norm_l2
    new_point[1] = point[1] - dt * gradient_at_point[1] / gradient_norm_l2
    return new_point

@ti.func
def norm_l2(
    vec: ti.types.vector(2, ti.f32),
    dxy: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the Euclidean norm of `vec` represented in the standard basis.

    Args:
        `vec`: ti.types.vector(n=2, dtype=[float]) which we want to normalise.

    Returns:
        Norm of `vec`.
    """
    return ti.math.sqrt(vec[0]**2 + vec[1]**2) / dxy

@ti.kernel
def distance_in_pixels(
    point: ti.types.vector(2, ti.f32),
    source_point: ti.types.vector(2, ti.f32),
    dxy: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the distance in pixels given the difference in coordinates and the
    pixel size.

    Args:
        `point`: ti.types.vector(n=2, dtype=[float]) current point.
        `source_points`: ti.types.vector(n=2, dtype=[float]) describing index of 
          source point in `W_np`.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    distance_vec = point - source_point
    return ti.math.sqrt((distance_vec[0] / dxy)**2 + (distance_vec[1] / dxy)**2)

@ti.kernel
def distance_in_pixels_multi_source(
    point: ti.types.vector(2, ti.f32),
    source_points: ti.template(),
    distances: ti.template(),
    dxy: ti.f32
) -> ti.f32:
    """
    @taichi.kernel

    Compute the distance in pixels given the difference in coordinates and the
    pixel size.

    Args:
        `point`: ti.types.vector(n=2, dtype=[float]) current point.
        `source_points`: ti.Vector.field(n=2, dtype=[float]) describing index of 
          source points in `W_np`.
        `distances`: ti.Vector.field(n=2, dtype=[float]) distances to source
          points.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.

    Returns:
        Minimum distance.
    """
    min_distance = ti.math.inf
    for I in ti.grouped(distances):
        distance_vec = point - source_points[I]
        distance = ti.math.sqrt((distance_vec[0] / dxy)**2 + (distance_vec[1] / dxy)**2)
        distances[I] = distance
        ti.atomic_min(min_distance, distance)
    return min_distance