# """
#     costfunction
#     ============

#     Compute the cost function on SO(3) by interpolating the cost function on
#     SE(2).
#     In particular, provides the following classes:
#       1. `CostR2`, which can compute the cost function from a vesselness on R^2
#       and store it with its parameters.
# """

# import numpy as np
# import taichi as ti
# from eikivp.SO3.utils import Π_forward
# from eikivp.SE2.utils import(
#     scalar_trilinear_interpolate,
#     coordinate_real_to_array_ti
# )
# from eikivp.SE2.vesselness import VesselnessSE2
# from eikivp.SE2.costfunction import CostSE2

# class CostSO3():
#     """
#     Compute the cost function from the R2 vesselness.

#     Attributes:
#         `C`: np.ndarray of cost function data.
#         `scales`: iterable of standard deviations of Gaussian derivatives,
#           taking values greater than 0. 
#         `α`: anisotropy penalty, taking values between 0 and 1.
#         `γ`: variance sensitivity, taking values between 0 and 1.
#         `ε`: structure penalty, taking values between 0 and 1.
#         `image_name`: identifier of image used to generate vesselness.
#         `λ`: vesselness prefactor, taking values greater than 0.
#         `p`: vesselness exponent, taking values greater than 0.
#     """

#     def __init__(self, V: VesselnessR2, λ, p):
#         # Vesselness attributes
#         self.scales = V.scales
#         self.α = V.α
#         self.γ = V.γ
#         self.ε = V.ε
#         self.image_name = V.image_name
#         # Cost attributes
#         self.λ = λ
#         self.p = p

#         self.C = cost_function(V.V, λ, p)

#     # def plot(self, x_min, x_max, y_min, y_max):
#     #     """Quick visualisation of cost."""
#     #     fig, ax, cbar = plot_image_array(self.C, x_min, x_max, y_min, y_max)
#     #     fig.colorbar(cbar, ax=ax);

#     def print(self):
#         """Print attributes."""
#         print(f"scales => {self.scales}")
#         print(f"α => {self.α}")
#         print(f"γ => {self.γ}")
#         print(f"ε => {self.ε}")
#         print(f"image_name => {self.image_name}")
#         print(f"λ => {self.λ}")
#         print(f"p => {self.p}")

# class CostSE2():
#     """
#     Compute the cost function from the SE(2) vesselness.

#     Attributes:
#         `C`: np.ndarray of cost function data.
#         `σ_s_list`: standard deviations in pixels of the internal regularisation
#           in the spatial directions before taking derivatives.
#         `σ_o`: standard deviation in pixels of the internal regularisation
#           in the orientational direction before taking derivatives.
#         `σ_s_ext`: standard deviation in pixels of the external regularisation
#           in the spatial direction after taking derivatives.
#           Notably, this regularisation is NOT truly external, because it
#           commutes with the derivatives.
#         `σ_o_ext`: standard deviation in pixels of the internal regularisation
#           in the orientational direction after taking derivatives.
#           Notably, this regularisation is NOT truly external, because it
#           commutes with the derivatives.
#         `image_name`: identifier of image used to generate vesselness.
#         `λ`: vesselness prefactor, taking values greater than 0.
#         `p`: vesselness exponent, taking values greater than 0.
#     """

#     def __init__(self, V: VesselnessSE2, λ, p):
#         # Vesselness attributes
#         self.σ_s_list = V.σ_s_list
#         self.σ_o = V.σ_o
#         self.σ_s_ext = V.σ_s_ext
#         self.σ_o_ext = V.σ_o_ext
#         self.image_name = V.image_name
#         # Cost attributes
#         self.λ = λ
#         self.p = p

#         self.C = cost_function(V.V, λ, p)

#     # def plot(self, x_min, x_max, y_min, y_max):
#     #     """Quick visualisation of cost."""
#     #     fig, ax, cbar = plot_image_array(self.C, x_min, x_max, y_min, y_max)
#     #     fig.colorbar(cbar, ax=ax);

#     def print(self):
#         """Print attributes."""
#         print(f"σ_s_list => {self.σ_s_list}")
#         print(f"σ_o => {self.σ_o}")
#         print(f"σ_s_ext => {self.σ_s_ext}")
#         print(f"σ_o_ext => {self.σ_o_ext}")
#         print(f"image_name => {self.image_name}")
#         print(f"λ => {self.λ}")
#         print(f"p => {self.p}")

# def cost_function(vesselness, λ, p):
#     """
#     Compute the cost function corresponding to `vesselness`.

#     Args:
#         `vesselness`: np.ndarray of vesselness scores, taking values between 0 
#           and 1.
#         `λ`: Vesselness prefactor, taking values greater than 0.
#         `p`: Vesselness exponent, taking values greater than 0.

#     Returns:
#         np.ndarray of the cost function corresponding to `vesselness` with 
#         parameters `λ` and `p`, taking values between 0 and 1.
#     """
#     return 1 / (1 + λ * np.abs(vesselness)**p)

# @ti.kernel
# def interpolate_cost_function(
#     cost_SE2: ti.template(),
#     αs: ti.template(),
#     βs: ti.template(),
#     φs: ti.template(),
#     a: ti.f32,
#     c: ti.f32,
#     x_min: ti.f32,
#     y_min: ti.f32,
#     θ_min: ti.f32,
#     dxy: ti.f32,
#     dθ: ti.f32,
#     cost_SO3: ti.template()
# ):
#     """
#     @ti.kernel

#     Sample cost function `cost_SE2`, given as a volume sampled uniformly on
#     SE(2), as a volume in SO(3)

#     Args:
#         `αs`: α-coordinates at which we want to sample.
#         `βs`: β-coordinates at which we want to sample.
#         `φs`: φ-coordinates at which we want to sample.
#         `a`: distance between nodal point of projection and centre of sphere.
#         `c`: distance between projection plane and centre of sphere reflected
#           around nodal point.
#         `x_min`: minimum value of x-coordinates in rectangular domain.
#         `y_min`: minimum value of y-coordinates in rectangular domain.
#         `θ_min`: minimum value of θ-coordinates in rectangular domain.
#         `dxy`: spatial resolution, which is equal in the x- and y-directions,
#           taking values greater than 0.
#         `dθ`: orientational resolution, taking values greater than 0.
#     """
#     for I in ti.grouped(cost_SE2):
#         point = Π_forward(αs[I], βs[I], φs[I], a, c)
#         index = coordinate_real_to_array_ti(point, x_min, y_min, θ_min, dxy, dθ)
#         cost_SO3[I] = scalar_trilinear_interpolate(cost_SE2, index)