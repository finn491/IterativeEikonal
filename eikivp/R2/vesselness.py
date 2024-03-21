"""
    vesselness
    ==========

    Provides tools compute vesselness scores on R^2. The available methods are:
      1. `rc_vessel_enhancement`: compute the singlescale vesselness using a
      Frangi filter.
      2. `multiscale_frangi_filter`: compute the multiscale vesselness by
      applying the Frangi filter at numerous scales and combining the results
      via maximum projection.
"""

import numpy as np
import diplib as dip


def rc_vessel_enhancement(image, σ, α=0.2, γ=0.75, ε=0.2):
    """
    Compute Frangi filter[1] of vessels in `image` at a single scale `σ`.
    Implementation adapted from "Code A - Vesselness in SE(2)".

    Args:
        `image`: np.ndarray of a grayscale image, taking values between 0 and 1,
          with shape [Nx, Ny].
        `σ`: Standard deviation of Gaussian derivatives, taking values greater 
          than 0.
        `α`: Anisotropy penalty, taking values between 0 and 1.
        `γ`: Variance sensitivity, taking values between 0 and 1.
        `ε`: Structure penalty, taking values between 0 and 1.

    Returns:
        np.ndarray of the vesselness of `image`, taking values between 0 and 1.
    
    References:
        [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever.
          "Multiscale vessel enhancement filtering". In: Medical Image Computing
          and Computer-Assisted Intervention (1998), pp. 130--137.
          DOI:10.1007/BFb0056195.
    """
    # Calculate Hessian derivatives.
    Lxx = np.array(dip.Gauss(image, (σ, σ), (2, 0)))
    Lxy = np.array(dip.Gauss(image, (σ, σ), (1, 1)))
    Lyy = np.array(dip.Gauss(image, (σ, σ), (0, 2)))

    # Calculate eigenvalues.
    λ = Lxx + Lyy
    λδ = np.sign(λ) * np.sqrt((2 * Lxy)**2 + (Lxx - Lyy)**2)
    λ1, λ2 = (σ**γ / 2) * np.array((λ + λδ, λ - λδ))

    # Calculate vesselness. Not quite sure what these variables represent.
    R2 = (λ2 / (λ1 + np.finfo(np.float64).eps)) ** 2
    nR2 = -1 / (2 * α**2)
    S2 = λ1**2 + λ2**2
    nS2 = -1 / (2 * ε**2 * np.max(S2))
    vesselness = (np.exp(nR2 * R2**2)
                  * (1 - np.exp(nS2 * S2))
                  * np.heaviside(-λ1, 1.))
    return vesselness


def multiscale_frangi_filter(image, σs, α=0.3, γ=0.75, ε=0.3):
    """
    Compute Frangi filter[1] of vessels in `image` at scales in `σs`.
    Implementation adapted from "Code A - Vesselness in SE(2)".

    Args:
        `image`: np.ndarray of a grayscale image, taking values between 0 and 1.
        `σs`: Iterable of standard deviations of Gaussian derivatives, taking
          values greater than 0.
        `α`: Anisotropy penalty, taking values between 0 and 1.
        `γ`: Variance sensitivity, taking values between 0 and 1.
        `ε`: Structure penalty, taking values between 0 and 1.

    Returns:
        np.ndarray of the vesselness of `image`, taking values between 0 and 1.
    
    References:
        [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever.
          "Multiscale vessel enhancement filtering". In: Medical Image Computing
          and Computer-Assisted Intervention (1998), pp. 130--137.
          DOI:10.1007/BFb0056195.
    """
    # Compute vesselness at each scale σ in σs, and select the maximum at
    # each point.
    vesselnesses = []
    for σ in σs:
        vesselnesses.append(rc_vessel_enhancement(image, σ, α=α, γ=γ, ε=ε))
    vesselness = np.maximum.reduce(vesselnesses)
    return vesselness