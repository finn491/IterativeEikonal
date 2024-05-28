"""
    costfunction
    ============

    Compute the cost function from the SE(2)-vesselness.
    In particular, provides the `CostSE2` class, which can compute the cost
    function from a vesselness on SE(2) and store it with its parameters.
"""

import numpy as np
import taichi as ti
from eikivp.utils import cost_function
from eikivp.SE2.vesselness import VesselnessSE2
from eikivp.R2.vesselness import VesselnessR2

class CostSE2():
    """
    Compute the cost function from the SE(2) vesselness.

    Attributes:
        `C`: np.ndarray of cost function data.
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
        `λ`: vesselness prefactor, taking values greater than 0.
        `p`: vesselness exponent, taking values greater than 0.
    """

    def __init__(self, V: VesselnessSE2 | VesselnessR2, λ, p, dim_K=32):
        # Vesselness attributes
        self.σ_s_list = V.σ_s_list
        self.σ_o = V.σ_o
        self.σ_s_ext = V.σ_s_ext
        self.σ_o_ext = V.σ_o_ext
        self.image_name = V.image_name
        # Cost attributes
        self.λ = λ
        self.p = p
        C = cost_function(V.V, λ, p)
        if isinstance(V, VesselnessR2):
            C = np.transpose(np.array([C] * dim_K), axes=(1, 2, 0))
        self.C = C

    # def plot(self, x_min, x_max, y_min, y_max):
    #     """Quick visualisation of cost."""
    #     fig, ax, cbar = plot_image_array(self.C, x_min, x_max, y_min, y_max)
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