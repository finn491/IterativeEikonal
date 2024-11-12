"""
    costfunction
    ============

    Compute the cost function from the SE(2)-vesselness. In particular, provides
    the `CostSE2` class, which can compute the cost function from a vesselness
    on SE(2) and store it with its parameters.
"""

from eikivp.utils import cost_function
from eikivp.SE2.vesselness import VesselnessSE2
from eikivp.R2.vesselness import VesselnessR2
import numpy as np

class CostSE2():
    """
    Compute the cost function from the R^2 or SE(2) vesselness.

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
        # Cost attributes
        self.λ = λ
        self.p = p
        self.image_name = V.image_name
        # Vesselness attributes
        if isinstance(V, VesselnessR2):
            self.domain_V = "R2"
            self.σ_s_list = V.scales
            self.σ_o = 0.
            self.σ_s_ext = 0.
            self.σ_o_ext = 0.
            self.C = cost_function(V.V, λ, p)
        elif isinstance(V, VesselnessSE2):
            self.domain_V = "SE2"
            self.σ_s_list = V.σ_s_list
            self.σ_o = V.σ_o
            self.σ_s_ext = V.σ_s_ext
            self.σ_o_ext = V.σ_o_ext
            self.C = cost_function(np.array([V.V]*dim_K).transpose(axes=(1, 2, 0)), λ, p)

    def print(self):
        """Print attributes."""
        print(f"Vesselness Type => {self.domain_V}")
        print(f"σ_s_list => {self.σ_s_list}")
        print(f"σ_o => {self.σ_o}")
        print(f"σ_s_ext => {self.σ_s_ext}")
        print(f"σ_o_ext => {self.σ_o_ext}")
        print(f"image_name => {self.image_name}")
        print(f"λ => {self.λ}")
        print(f"p => {self.p}")