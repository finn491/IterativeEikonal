# costfunction.py

import numpy as np

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