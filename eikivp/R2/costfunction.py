"""
    costfunction
    ============

    Compute the cost function from the R^2-vesselness. In particular, provides
    the `CostR2` class, which can compute the cost function from a vesselness on
    R^2 and store it with its parameters.
"""

from eikivp.utils import cost_function
from eikivp.R2.vesselness import VesselnessR2

class CostR2():
    """
    Compute the cost function from the R2 vesselness.

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

    def __init__(self, V: VesselnessR2, λ, p):
        # Vesselness attributes
        self.scales = V.scales
        self.α = V.α
        self.γ = V.γ
        self.ε = V.ε
        self.image_name = V.image_name
        # Cost attributes
        self.λ = λ
        self.p = p

        self.C = cost_function(V.V, λ, p)

    def print(self):
        """Print attributes."""
        print(f"scales => {self.scales}")
        print(f"α => {self.α}")
        print(f"γ => {self.γ}")
        print(f"ε => {self.ε}")
        print(f"image_name => {self.image_name}")
        print(f"λ => {self.λ}")
        print(f"p => {self.p}")