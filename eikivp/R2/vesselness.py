"""
    vesselness
    ==========

    Provides tools compute vesselness scores on R^2. In particular, provides the
    class `VesselnessR2`, which can compute the vesselness and store it with its
    parameters.
    
    The available methods are:
      1. `rc_vessel_enhancement`: compute the singlescale vesselness using a
      Frangi filter[1].
      2. `multiscale_frangi_filter`: compute the multiscale vesselness by
      applying the Frangi filter at numerous scales and combining the results
      via maximum projection.
      
    References:
        [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever.
        "Multiscale vessel enhancement filtering". In: Medical Image Computing
        and Computer-Assisted Intervention (1998), pp. 130--137.
        DOI:10.1007/BFb0056195.
"""

import numpy as np
import scipy as sp
import diplib as dip
import h5py
from eikivp.utils import image_rescale
# from eikivp.visualisations import plot_image_array

class VesselnessR2():
    """
    The vesselness of a retinal image in R2 computed using multiscale Frangi
    filters[1].

    Attributes:
        `V`: np.ndarray of vesselness data.
        `scales`: iterable of standard deviations of Gaussian derivatives,
          taking values greater than 0. 
        `α`: anisotropy penalty, taking values between 0 and 1.
        `γ`: variance sensitivity, taking values between 0 and 1.
        `ε`: structure penalty, taking values between 0 and 1.
        `image_name`: identifier of image used to generate vesselness.

    References:
        [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever.
        "Multiscale vessel enhancement filtering". In: Medical Image Computing
        and Computer-Assisted Intervention (1998), pp. 130--137.
        DOI:10.1007/BFb0056195.
    """

    def __init__(self, scales, α, γ, ε, image_name):
        # Vesselness attributes
        self.scales = scales
        self.α = α
        self.γ = γ
        self.ε = ε
        self.image_name = image_name

    def compute_V(self, retinal_array):
        """
        Compute Frangi filter[1] of vessels in `retinal_array` at scales in `σs`.
        Implementation adapted from "Code A - Vesselness in SE(2)".

        Args:
            `retinal_array`: np.ndarray of a grayscale image, taking values
              between 0 and 1.

        Returns:
            np.ndarray of the vesselness of `image`, taking values between 0 and
            1.
        
        References:
            [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A.
              Viergever.
              "Multiscale vessel enhancement filtering". In: Medical Image
              Computing and Computer-Assisted Intervention (1998), pp. 130--137.
              DOI:10.1007/BFb0056195.
          """
        V_unmasked = multiscale_frangi_filter(-retinal_array, self.scales, α=self.α, γ=self.γ, ε=self.ε)
        mask = (retinal_array > 0) # Remove boundary
        V_unnormalised = V_unmasked * sp.ndimage.binary_erosion(mask, iterations=int(np.ceil(self.scales.max() * 2)))
        print(f"Before rescaling, vesselness is in [{V_unnormalised.min()}, {V_unnormalised.max()}].")
        self.V = image_rescale(V_unnormalised)

    def import_V(self, folder):
        """
        Import the vesselness matching the attributes `scales`, `α`, `γ`, `ε`,
        and `image_name`.
        """
        vesselness_filename = f".\\{folder}\\R2_sigmas={[s for s in self.scales]}_alpha={self.α}_gamma={self.γ}_epsilon={self.ε}.hdf5"
        with h5py.File(vesselness_filename, "r") as vesselness_file:
            assert (
                np.all(self.scales == vesselness_file.attrs["scales"]) and
                self.α == vesselness_file.attrs["α"] and
                self.γ == vesselness_file.attrs["γ"] and
                self.ε == vesselness_file.attrs["ε"] and
                self.image_name == vesselness_file.attrs["image_name"]
            ), "There is a parameter mismatch!"
            self.V = vesselness_file["Vesselness"][()]
            
    def export_V(self, folder):
        """
        Export the vesselness to hdf5 with attributes `scales`, `α`, `γ`, `ε`,
        and `image_name` stored as metadata.
        """
        vesselness_filename = f".\\{folder}\\R2_sigmas={[s for s in self.scales]}_alpha={self.α}_gamma={self.γ}_epsilon={self.ε}.hdf5"
        with h5py.File(vesselness_filename, "w") as vesselness_file:
            vesselness_file.create_dataset("Vesselness", data=self.V)
            vesselness_file.attrs["scales"] = self.scales
            vesselness_file.attrs["α"] = self.α
            vesselness_file.attrs["γ"] = self.γ
            vesselness_file.attrs["ε"] = self.ε

    # def plot(self, x_min, x_max, y_min, y_max):
    #     """Quick visualisation of vesselness."""
    #     fig, ax, cbar = plot_image_array(-self.V, x_min, x_max, y_min, y_max)
    #     fig.colorbar(cbar, ax=ax);

    def print(self):
        """Print attributes."""
        print(f"scales => {self.scales}")
        print(f"α => {self.α}")
        print(f"γ => {self.γ}")
        print(f"ε => {self.ε}")


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