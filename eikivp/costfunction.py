"""
    costfunction
    ============

    Compute the cost function from the R^2- or SE(2)-vesselness, and interpolate
    it on SO(3).
"""

import numpy as np
import taichi as ti
from eikivp.SO3.utils import Π_forward
from eikivp.SE2.utils import(
    scalar_trilinear_interpolate,
    coordinate_real_to_array_ti
)

def cost_function(vesselness, λ, p):
    """
    Compute the cost function corresponding to `vesselness`.

    Args:
        `vesselness`: np.ndarray of vesselness scores, taking values between 0 
          and 1.
        `λ`: Vesselness prefactor, taking values greater than 0.
        `p`: Vesselness exponent, taking values greater than 0.

    Returns:
        np.ndarray of the cost function corresponding to `vesselness` with 
        parameters `λ` and `p`, taking values between 0 and 1.
    """
    return 1 / (1 + λ * np.abs(vesselness)**p)

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