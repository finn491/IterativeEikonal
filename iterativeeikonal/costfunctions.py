# costfunctions.py

import taichi as ti
import numpy as np
import scipy as sp
import skimage
import diplib as dip


# Helper Functions

# Broken implementation
def rc_vessel_enhancement(image, α, ε, σ):
    """
    Compute Frangi filter of vessels in `image` at a single scale `σ`. Copied 
    from "Code A - Vesselness in SE(2)".
    """
    γ = 3 / 4
    # Calculate Hessian derivatives. Apparently, higher order Gaussian 
    # derivatives are not very accurate.
    # if σ > 2:
    #     Lxx = sp.ndimage.gaussian_filter(image, sigma=(3 * σ, σ), order=(0, 2))
    #     Lxy = sp.ndimage.gaussian_filter(image, sigma=(3 * σ, σ), order=(1, 1))
    #     Lyy = sp.ndimage.gaussian_filter(image, sigma=(3 * σ, σ), order=(2, 0))
    # else: # Is this equivalent to what goes on in RcVesselEnhancement?
    #     Lxx = sp.ndimage.gaussian_filter(image, sigma=σ, order=(0, 2))
    #     Lxy = sp.ndimage.gaussian_filter(image, sigma=σ, order=(1, 1))
    #     Lyy = sp.ndimage.gaussian_filter(image, sigma=σ, order=(2, 0))
    if σ > 2:
        Lxx = np.array(dip.Gauss(image, (3 * σ, σ), (0, 2)))
        Lxy = np.array(dip.Gauss(image, (3 * σ, σ), (1, 1)))
        Lyy = np.array(dip.Gauss(image, (3 * σ, σ), (2, 0)))
    else: # Is this equivalent to what goes on in RcVesselEnhancement?
        Lxx = np.array(dip.Gauss(image, (σ, σ), (0, 2)))
        Lxy = np.array(dip.Gauss(image, (σ, σ), (1, 1)))
        Lyy = np.array(dip.Gauss(image, (σ, σ), (2, 0)))

    # Calculate eigenvalues.
    λ = Lxx + Lyy
    λδ = np.sign(λ) * np.sqrt((2 * Lxy)**2 + (Lxx - Lyy)**2)
    λ1, λ2 = (σ**γ / 2) * np.array((λ + λδ, λ - λδ))

    # Calculate vesselness.
    R2 = (λ2 / (λ1 + np.finfo(np.float64).eps)) ** 2
    nR2 = -1 / (2 * α**2)
    S2 = λ1**2 + λ2**2
    nS2 = -1 / (2 * ε**2 * np.max(S2))
    vesselness = (np.exp(nR2 * R2**2) 
                  * (1 - np.exp(nS2 * S2)) 
                  * np.heaviside(-λ1, 1.))
    return vesselness

def multiscale_frangi_filter(image, α, ε, σs):
    """
    Compute Frangi filter of vessels in `image` at scales in `σs`. Copied from 
    "Code A - Vesselness in SE(2)".

    Bright structures (where `image` has a high positive value) are enhanced.
    """
    # Compute vesselness at each scale σ in σs, and select the maximum at 
    # each point.
    vesselnesses = []
    for σ in σs:
        vesselnesses.append(rc_vessel_enhancement(image, α, ε, σ))
    vesselness = np.maximum.reduce(vesselnesses)
    return vesselness

# def multiscale_frangi_filter(image, α, ε, σs):
#     """
#     Compute Frangi filter of vessels in `image` at scales in `σs`. Copied from 
#     "Code A - Vesselness in SE(2)".

#     Wrapper for scikit-image function `skimage.filters.frangi()`.
#     """
#     # Compute vesselness at each scale σ in σs, and select the maximum at 
#     # each point.
#     return skimage.filters.frangi(image, sigmas=σs, alpha=α, beta=ε, gamma=3/4)