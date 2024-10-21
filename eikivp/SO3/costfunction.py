"""
    costfunction
    ============

    Compute the cost function on SO(3) by interpolating the cost function on
    SE(2). In particular, provides the `CostSO3` class, which can compute the
    cost function from a cost function on SE(2) and store it with its parameters.
"""

import numpy as np
import taichi as ti
from eikivp.SO3.utils import Π_forward
from eikivp.SE2.utils import(
    scalar_trilinear_interpolate,
    coordinate_real_to_array_ti
)
from eikivp.SE2.costfunction import CostSE2

class CostSO3(CostSE2):
    """
    Compute the cost function on SO(3) by interpolating the cost function on
    SE(2).

    Attributes:
        `C`: np.ndarray of cost function data.
        `scales`: iterable of standard deviations of Gaussian derivatives,
          taking values greater than 0. 
        `α`: anisotropy penalty, taking values between 0 and 1.
        `γ`: variance sensitivity, taking values between 0 and 1.
        `ε`: structure penalty, taking values between 0 and 1.
        `image_name`: identifier of image used to generate vesselness.
        `λ`: vesselness prefactor, taking values greater than 0.
        `p`: vesselness exponent, taking values greater than 0.
    """

    def __init__(self, V, λ, p, αs, βs, φs, a, c, x_min, y_min, θ_min, dxy, dθ):
        super().__init__(V, λ, p)
        # Cost function on SE(2).
        CSE2 = self.C
        # Interpolate SE(2) cost function.
        shape = CSE2.shape
        CSE2_ti = ti.field(dtype=ti.f32, shape=shape)
        CSE2_ti.from_numpy(CSE2)
        αs_ti = ti.field(dtype=ti.f32, shape=shape)
        αs_ti.from_numpy(αs)
        βs_ti = ti.field(dtype=ti.f32, shape=shape)
        βs_ti.from_numpy(βs)
        φs_ti = ti.field(dtype=ti.f32, shape=shape)
        φs_ti.from_numpy(φs)
        CSO3_ti = ti.field(dtype=ti.f32, shape=shape)
        interpolate_cost_function(CSE2_ti, αs_ti, βs_ti, φs_ti, a, c, x_min, y_min, θ_min, dxy, dθ, CSO3_ti)
        self.C = CSO3_ti.to_numpy()


@ti.kernel
def interpolate_cost_function(
    cost_SE2: ti.template(),
    αs: ti.template(),
    βs: ti.template(),
    φs: ti.template(),
    a: ti.f32,
    c: ti.f32,
    x_min: ti.f32,
    y_min: ti.f32,
    θ_min: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    cost_SO3: ti.template()
):
    """
    @ti.kernel

    Sample cost function `cost_SE2`, given as a volume sampled uniformly on
    SE(2), as a volume in SO(3)

    Args:
        `αs`: α-coordinates at which we want to sample.
        `βs`: β-coordinates at which we want to sample.
        `φs`: φ-coordinates at which we want to sample.
        `a`: distance between nodal point of projection and centre of sphere.
        `c`: distance between projection plane and centre of sphere reflected
          around nodal point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    for I in ti.grouped(cost_SE2):
        point = Π_forward(αs[I], βs[I], φs[I], a, c)
        index = coordinate_real_to_array_ti(point, x_min, y_min, θ_min, dxy, dθ)
        cost_SO3[I] = scalar_trilinear_interpolate(cost_SE2, index)