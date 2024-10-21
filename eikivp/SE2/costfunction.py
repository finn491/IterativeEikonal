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
        # Vesselness attributes
        # if isinstance(V, VesselnessR2):
        #     self.scales = V.scales
        #     self.α = V.α
        #     self.γ = V.γ
        #     self.ε = V.ε
        # elif isinstance(V, VesselnessSE2):
        self.σ_s_list = V.σ_s_list
        self.σ_o = V.σ_o
        self.σ_s_ext = V.σ_s_ext
        self.σ_o_ext = V.σ_o_ext
        self.image_name = V.image_name
        # Cost attributes
        self.λ = λ
        self.p = p
        C = cost_function(V.V, λ, p)
        # if isinstance(V, VesselnessR2):
        #     C = np.transpose(np.array([C] * dim_K), axes=(1, 2, 0))
        self.C = C

    def print(self):
        """Print attributes."""
        # if hasattr(self, "scales"): # Cost comes from R^2 vesselness
        #     print(f"scales => {self.scales}")
        #     print(f"α => {self.α}")
        #     print(f"γ => {self.γ}")
        #     print(f"ε => {self.ε}")
        # elif hasattr(self, "σ_s_list"): # Cost comes from SE(2) vesselness
        print(f"σ_s_list => {self.σ_s_list}")
        print(f"σ_o => {self.σ_o}")
        print(f"σ_s_ext => {self.σ_s_ext}")
        print(f"σ_o_ext => {self.σ_o_ext}")
        print(f"image_name => {self.image_name}")
        print(f"λ => {self.λ}")
        print(f"p => {self.p}")