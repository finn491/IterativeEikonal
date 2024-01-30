# vesselness.py

import numpy as np
import diplib as dip


def rc_vessel_enhancement_R2(image, σ, α=0.2, γ=0.75, ε=0.2):
    """
    Compute Frangi filter of vessels in `image` at a single scale `σ`. Copied 
    from "Code A - Vesselness in SE(2)".

    Args:
        `image`: np.ndarray of a grayscale image, taking values between 0 and 1.
        `σ`: Standard deviation of Gaussian derivatives, taking values greater 
          than 0.
        `α`: Anisotropy penalty, taking values between 0 and 1.
        `γ`: Variance sensitivity, taking values between 0 and 1.
        `ε`: Structure penalty, taking values between 0 and 1.

    Returns:
        np.ndarray of the vesselness of `image`, taking values between 0 and 1.
    """
    # Calculate Hessian derivatives.
    # Apparently, higher order Gaussian derivatives in Scipy are not very
    # accurate.
    # if σ > 2:
    #     Lxx = sp.ndimage.gaussian_filter(image, sigma=(σ, σ), order=(0, 2))
    #     Lxy = sp.ndimage.gaussian_filter(image, sigma=(σ, σ), order=(1, 1))
    #     Lyy = sp.ndimage.gaussian_filter(image, sigma=(σ, σ), order=(2, 0))
    # else: # Is this equivalent to what goes on in RcVesselEnhancement?
    #     Lxx = sp.ndimage.gaussian_filter(image, sigma=σ, order=(0, 2))
    #     Lxy = sp.ndimage.gaussian_filter(image, sigma=σ, order=(1, 1))
    #     Lyy = sp.ndimage.gaussian_filter(image, sigma=σ, order=(2, 0))
    # DIPlib provides more accurate Gaussian derivatives, but maybe we want our
    # own implementation. Note that DIPlib flips the order of dimensions.
    if σ > 2:
        Lxx = np.array(dip.Gauss(image, (σ, σ), (2, 0)))
        Lxy = np.array(dip.Gauss(image, (σ, σ), (1, 1)))
        Lyy = np.array(dip.Gauss(image, (σ, σ), (0, 2)))
    else:  # Is this equivalent to what goes on in RcVesselEnhancement?
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


def multiscale_frangi_filter_R2(image, σs, α=0.3, γ=0.75, ε=0.3):
    """
    Compute Frangi filter of vessels in `image` at scales in `σs`. Copied from 
    "Code A - Vesselness in SE(2)".

    Args:
        `image`: np.ndarray of a grayscale image, taking values between 0 and 1.
        `σs`: Iterable of standard deviations of Gaussian derivatives, taking
          values greater than 0.
        `α`: Anisotropy penalty, taking values between 0 and 1.
        `γ`: Variance sensitivity, taking values between 0 and 1.
        `ε`: Structure penalty, taking values between 0 and 1.

    Returns:
        np.ndarray of the vesselness of `image`, taking values between 0 and 1.
    """
    # Compute vesselness at each scale σ in σs, and select the maximum at
    # each point.
    vesselnesses = []
    for σ in σs:
        vesselnesses.append(rc_vessel_enhancement_R2(image, σ, α=α, γ=γ, ε=ε))
    vesselness = np.maximum.reduce(vesselnesses)
    return vesselness

# Does not work as well as the one based on the Mathematica implementation.
# def multiscale_frangi_filter_R2(image, α, ε, σs):
#     """
#     Compute Frangi filter of vessels in `image` at scales in `σs`. Wrapper for
#     scikit-image function `skimage.filters.frangi()`.
#     """
#     # Compute vesselness at each scale σ in σs, and select the maximum at
#     # each point.
#     return skimage.filters.frangi(image, sigmas=σs, alpha=α, beta=ε, gamma=3/4)