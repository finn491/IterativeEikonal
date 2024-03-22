"""
    orientationscore
    ================

    Provides methods to compute the orientation score of a 2D image. The primary
    methods are:
      1. `CakeWaveletStack`: compute the stack of cakewavelets, as described in
      Duits "Perceptual Organization in Image Analysis" (2005)
      (https://www.win.tue.nl/~rduits/THESISRDUITS.pdf).
      2. `WaveletTransform2D`: compute the wavelet transform of a 2D image,
      with respect to some wavelet (which need not be a cakewavelet).
"""

import numpy as np
import scipy as sp

def mod_offset(x, period, offset):
    return x - (x - offset)//period * period

def rotate_left(array, k):
    """idk."""
    return rotate_right(array, -k)

def rotate_right(array, k):
    """idk."""
    if type(k) == int or type(k) == float:
        arr1 = array[:-k]
        arr2 = array[-k:]
        rotated_array = np.concatenate((arr2, arr1), axis=0)
    elif len(k) == 2 and len(array.shape) == 2:
        k_0, k_1 = k
        arr1 = array[:, :-k_1]
        arr2 = array[:, -k_1:]
        array = np.concatenate((arr2, arr1), axis=1)
        arr1 = array[:-k_0, :]
        arr2 = array[-k_0:, :]
        rotated_array = np.concatenate((arr2, arr1), axis=0)
    else:
        raise ValueError(f"k = {k} is not a supported point for rotating. k should be a number or a pair of numbers.")
    return rotated_array

def Gauss_window(N_spatial, σ_s):
    """WindowGauss retuns the spatial Gauss envelope"""
    xs, ys = np.meshgrid(np.arange(-np.floor(N_spatial / 2), np.ceil(N_spatial / 2)),
                         np.arange(-np.floor(N_spatial / 2), np.ceil(N_spatial / 2)),
                         indexing="ij")
    out = np.exp(-(xs**2 + ys**2) / (2 * σ_s**2))
    return out

def angular_grid(N_spatial):
    """
    PolarCoordinateGridRadial returns a matrix in which each element gives the
    corresponding radial coordinate (with the origin in the center of the matrix
    """
    centerx = np.ceil((N_spatial - 1) / 2)
    centery = centerx
    xs, ys = np.meshgrid(np.arange(N_spatial), np.arange(N_spatial), indexing="ij")
    dxs = xs - centerx
    dys = ys - centery
    m = np.arctan2(dys, dxs)
    return m

def radial_grid(N_spatial):
    """
    PolarCoordinateGridRadial returns a matrix in which each element gives the 
    corresponding radial coordinate (with the origin in the center of the matrix
    """
    centerx = np.ceil((N_spatial-1)/2)
    centery = centerx
    xs, ys = np.meshgrid(np.arange(N_spatial), np.arange(N_spatial), indexing="ij")
    dxs = centerx - xs
    dys = centery - ys
    m = (np.sqrt(dxs**2 + dys**2) + np.finfo(np.float64).eps) / ((N_spatial - 1) / 2)
    return m


def radial_window(N_spatial, n, inflection_point):
    """
    MnWindow gives the radial windowing matrix for sampling the fourier domain
    """
    ε = np.finfo(np.float64).eps
    po_matrix = ε + radial_grid(N_spatial) / np.sqrt(2 * inflection_point**2 / (1 + 2 * n))
    s = np.zeros_like(po_matrix)
    for k in range(n + 1):
        s = s + np.exp(-po_matrix**2) * po_matrix**(2*k) / sp.special.factorial(k)
    return s

def B_spline_matrix(n, x):
    ε = np.finfo(np.float64).eps
    r = 0
    for i in np.arange(-n/2, n/2 + 1):
        s = 0
        for k in range(n + 2):
            binom_cof = sp.special.binom(n + 1, k)
            sign = np.sign(i + (n + 1) / 2 - k)
            s += binom_cof * (x + (n + 1) / 2 - k) ** (n + 1 - 1) * (-1)**k * sign

        f = s / (2 * sp.special.factorial(n+1-1))
        ic = np.heaviside((x - (i - 1/2 + ε)), 1) * np.heaviside(-(x - (i + 1/2 - ε*(i>=n/2))), 1)
        
        r += f * np.round(ic)
    return r
## This does not work because it breaks broadcasting over x.
#     ε = np.finfo(np.float64).eps
#     js = np.arange(-n/2, n/2 + 1)
#     ss = np.zeros(n + 1)
#     for k in range(n + 2):
#         binom_cof = sp.special.binom(n + 1, k)
#         signs = np.sign(js + (n + 1) / 2 - k)
#         ss += binom_cof * (x + (n + 1) / 2 - k) ** (n + 1 - 1) * (-1)**k * signs
#     fs = ss / (2 * sp.special.factorial(n))
#     ics = np.heaviside((x - (js - 1/2 + ε)), 1) * np.heaviside(-(x - (js + 1/2 - ε*(js>=n/2))), 1)    
#     r = np.sum(fs * np.round(ics))
#     return r

def cakewavelet_stack_fourier(N_spatial, dθ, spline_order, overlap_factor, inflection_point, mn_order, DC_σ):
    """
    CakeWaveletStackFourier constructs the cake wavelets in the Fourier domain 
    (note that windowing in the spatial domain is still required after this
    """
    DC_window = np.ones((N_spatial, N_spatial)) - Gauss_window(N_spatial, DC_σ) 
    mn_window = radial_window(N_spatial, mn_order, inflection_point)
    window = DC_window * mn_window
    angle_grid = angular_grid(N_spatial)
    dθ_overlapped = dθ / overlap_factor
    s = 2 * np.pi
    θs = np.arange(0, s, dθ_overlapped)
    filters = np.zeros((θs.shape[0] + 1, N_spatial, N_spatial))
    for i, θ in enumerate(θs):
        x = mod_offset(angle_grid - θ - np.pi / 2, 2 * np.pi, -np.pi) / dθ
        filters[i, ...] = window * B_spline_matrix(spline_order, x) / overlap_factor   
    filters[-1, ...] = 1 - DC_window
    return filters

def cakewavelet_stack(N_spatial, Nθ, inflection_point=0.9, mn_order=10, spline_order=3, overlap_factor=1, DC_σ_pixels=5):
    """
    directional     - Determines whenever the filter goes in both directions;
    design          - Indicates which design is used N = Subscript[N, \[Psi]] or M = Subscript[M, \[Psi]]
    inflectionPoint - Is the location of the inflection point as a factor in (positive) radial direction
    splineOrder     - Order of the B - Spline that is used to construct the wavelet
    mnOrder         - The order of the (Taylor expansion) gaussian decay used to construct the wavelet
    dcStdDev        - The standard deviation of the gaussian window (in the Spatial domain) \
                      that removes the center of the pie, to avoid long tails in the spatial domain
    overlapFactor   - How much the cakepieces overlaps in \[Phi] - direction, this can be \
                      seen as subsampling the angular direction
                      size = 45
    nOrientations = 16
    design = "N"
    inflectionPoint = 0.5
    mnOrder = 8
    splineOrder = 3
    overlapFactor = 1
    dcStdDev = 8
    directional = False
    """
    dθ = 2 * np.pi / Nθ
    DC_σ = 1 / (dθ * DC_σ_pixels)
    filters = cakewavelet_stack_fourier(N_spatial, dθ, spline_order, overlap_factor, inflection_point, mn_order, DC_σ)
    
    cake_fourier = filters[:-1, ...]
    dc_filter = filters[-1, ...]

    cake = np.zeros_like(cake_fourier, dtype=np.complex_)
    rotation_amounts = np.array((N_spatial // 2, N_spatial // 2))
    for i, slice_fourier in enumerate(cake_fourier):
        slice_fourier = rotate_left(slice_fourier, rotation_amounts)
        slice = np.conj(np.fft.ifftn(slice_fourier))        
        slice = rotate_right(slice, rotation_amounts)
        cake[i, ...] = slice

    cake = np.vstack((cake, np.conj(cake)))
    
    dc_filter = rotate_left(dc_filter, rotation_amounts)
    dc_filter = np.fft.ifftn(dc_filter).real # dcFilter is real
    dc_filter = rotate_right(dc_filter, rotation_amounts)

    return cake, dc_filter
    
def wavelet_transform(f, kernels):
    """
    Return the real part of the wavelet transform of image `f` under the
    `kernels`.
    """
    ost = np.zeros((kernels.shape[0], f.shape[0], f.shape[1]))
    f_hat = np.fft.fftn(f)
    rotation_amount = np.ceil(0.1 + np.array(f.shape) / 2).astype(int) # Why?
    for i, ψ_θ in enumerate(kernels):
        ψ_θ_hat = np.fft.fftn(ψ_θ)
        U_θ_hat = np.fft.ifftn(ψ_θ_hat * f_hat).real
        U_θ_hat = rotate_right(U_θ_hat, rotation_amount)
        ost[i, ...] = U_θ_hat
    return ost

def wavelet_transform_complex(f, kernels):
    """
    Return the wavelet transform of image `f` under the `kernels`.
    """
    ost = np.zeros((kernels.shape[0], f.shape[0], f.shape[1]))
    f_hat = np.fft.fftn(f)
    rotation_amount = np.ceil(0.1 + np.array(f.shape) / 2).astype(int) # Why?
    for i, ψ_θ in enumerate(kernels):
        ψ_θ_hat = np.fft.fftn(ψ_θ)
        U_θ_hat = np.fft.ifftn(ψ_θ_hat * f_hat)
        U_θ_hat = rotate_right(U_θ_hat, rotation_amount)
        ost[i, ...] = U_θ_hat
    return ost