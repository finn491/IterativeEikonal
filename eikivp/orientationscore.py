"""
    orientationscore
    ================

    Provides methods to compute the orientation score of a 2D image. The primary
    methods are:
      1. CakeWaveletStack: compute the stack of cakewavelets, as described in
      Duits "Perceptual Organization in Image Analysis" (2005)
      (https://www.win.tue.nl/~rduits/THESISRDUITS.pdf).
      2. WaveletTransform2D: compute the wavelet transform of a 2D image,
      with respect to some wavelet (which need not be a cakewavelet).
"""

import numpy as np
import scipy as sp

def ErfSet(size, No, periodicity):
    """
    ErfSet returns a set of 2 D error functions.This function is used to cut the
    wavelets in two (in the spatial domain)
    """
    out = np.zeros((No, size, size))
    for i in range(No):
        xx = 0
        for x in np.arange(-np.floor(size / 2), np.ceil(size / 2)):
            yy = 0
            for y in np.arange(-np.floor(size / 2), np.ceil(size / 2)):
                out[i, xx, yy] = (1 + sp.special.erf(
                    x * np.cos((periodicity * i) / No) + 
                    y * np.sin((periodicity * i) / No)
                    )) / 2
                yy += 1
            xx += 1
    return out

def WindowGauss(size, σ_s):
    """WindowGauss retuns the spatial Gauss envelope"""
    out = np.zeros((size,size))
    i = 0
    for x in np.arange(-np.floor(size / 2), np.ceil(size / 2)):
        j = 0
        for y in np.arange(-np.floor(size / 2), np.ceil(size / 2)):
            out[i, j] = np.exp(-(x**2 + y**2) / (2 * σ_s**2))
            j += 1
        i += 1
    return out


def PolarCoordinateGridAngular(size):
    """
    PolarCoordinateGridRadial returns a matrix in which each element gives the 
    corresponding radial coordinate (with the origin in the center of the matrix
    """
    m = np.zeros((size, size))
    centerx = np.ceil((size - 1) / 2)
    centery = centerx
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    dxs = centerx - xs
    dys = centery - ys
    m = np.arctan2(dys, dxs)
    # for i in range(size):
    #     for j in range(size):
    #         dx = i-centerx
    #         dy = j-centery
    #         m[i, j] = cmath.phase(complex(dx,dy))
    return m


def PolarCoordinateGridRadial(size):
    """
    PolarCoordinateGridRadial returns a matrix in which each element gives the 
    corresponding radial coordinate (with the origin in the center of the matrix
    """
    m = np.zeros((size,size))
    centerx = np.ceil((size-1)/2)
    centery = centerx
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    dxs = centerx - xs
    dys = centery - ys
    m = (np.sqrt(dxs**2 + dys**2) + np.finfo(np.float64).eps) / ((size - 1) / 2)
    # for i in range(size):
    #     for j in range(size):
    #         dx = centerx-i
    #         dy = centery-j
    #         m[i,j] = (np.sqrt(dx**2 + dy**2) + np.finfo(np.float64).eps) / ((size - 1) / 2)
    return m

def MnWindow(size, n, inflectionPoint):
    """
    MnWindow gives the radial windowing matrix for sampling the fourier domain
    """
    ε = np.finfo(np.float64).eps
    po_matrix = ε + PolarCoordinateGridRadial(size) / np.sqrt(2 * inflectionPoint**2 / (1 + 2 * n))
    s = np.zeros_like(po_matrix)
    for k in range(n + 1):
        s = s + np.exp(-po_matrix**2) * po_matrix**(2*k) / np.math.factorial(k)
    return s


def BSplineMatrixFunc(n, x):
    ε = np.finfo(np.float64).eps
    r = 0
    for i in np.arange(-n/2, n/2 + 1):
        s = 0
        for k in range(n + 2):
            binom_cof = sp.special.binom(n + 1, k)
            sign = np.sign(i + (n + 1) / 2 - k)
            s += binom_cof * (x + (n + 1) / 2 - k) ** (n + 1 - 1) * (-1)**k * sign

        f = s / (2 * np.math.factorial(n+1-1))
        ic = np.heaviside((x - (i - 1/2 + ε)), 1) * np.heaviside(-(x - (i + 1/2 - ε*(i>=n/2))), 1)
        # if i < n/2:
        #     ic = np.heaviside((x - (i - 1/2 + ε)), 1) * np.heaviside(-(x - (i + 1/2)), 1)
        # else:
        #     ic = np.heaviside((x - (i - 1/2 + ε)), 1) * np.heaviside(-(x - (i + 1/2 - ε)), 1)
        
        r += f * np.round(ic)
    return r

def mod_offset(arr, divv, offset):
    return arr - (arr - offset) // divv**2

def CakeWaveletStackFourier(size, sΦ, splineOrder, overlapFactor, inflectionPoint, mnOrder, dcStdDev,
                            noSymmetry):
    """
    CakeWaveletStackFourier constructs the cake wavelets in the Fourier domain 
    (note that windowing in the spatial domain is still required after this
    """
    dcWindow = np.ones((size, size)) - WindowGauss(size, dcStdDev) 
    mnWindow = MnWindow(size, mnOrder, inflectionPoint)
    angleGrid = PolarCoordinateGridAngular(size)
    sΦOverlapped = sΦ/overlapFactor
    if noSymmetry:
        s = 2*np.pi
    else:
        s = np.pi
    θs = np.arange(0, s, sΦOverlapped)
    filters = np.zeros((θs.shape[0]+1, size, size))
    for i, θ in enumerate(θs):
        x = mod_offset(angleGrid - θ - np.pi / 2, 2 * np.pi, -np.pi) / sΦ 
        f = dcWindow * mnWindow * BSplineMatrixFunc(splineOrder, x) / overlapFactor
        f = np.expand_dims(f, axis=0)
        filters[i, :, :] = f
    
    filters[-1, :, :] = np.expand_dims((1 - dcWindow), axis=0)
    return filters


def CakeWaveletStack(size=15, nOrientations=8, design="N", inflectionPoint=0.9, mnOrder=10, splineOrder=3, 
                     overlapFactor=1, dcStdDev=5, directional=False):
    """
    directional     - Determines whenever the filter goes in both directions;
    design          - Indicates which design is used N = Subscript[N, \[Psi]] or
                      M = Subscript[M, \[Psi]]
    inflectionPoint - Is the location of the inflection point as a factor in 
                      (positive) radial direction
    splineOrder     - Order of the B - Spline that is used to construct the 
                      wavelet
    mnOrder         - The order of the (Taylor expansion) gaussian decay used to
                      construct the wavelet
    dcStdDev        - The standard deviation of the gaussian window (in the 
                      Spatial domain) that removes the center of the pie, to 
                      avoid long tails in the spatial domain
    overlapFactor   - How much the cakepieces overlaps in \[Phi] - direction, 
                      this can be seen as subsampling the angular direction
    size = 15
    nOrientations = 8
    design = "N"
    inflectionPoint = 0.9
    mnOrder = 10
    splineOrder = 3
    overlapFactor = 1
    dcStdDev = 5
    directional = False
    """
    noSymmetry = nOrientations % 2 == 1
    dcSigma = size / (2 * np.pi * dcStdDev)
    filters = CakeWaveletStackFourier(size, 2 * np.pi / nOrientations, splineOrder, overlapFactor, inflectionPoint, 
                                      mnOrder, dcSigma, noSymmetry)
    #print(filters.shape)
    cakeF = filters[:-1, :, :]
    #print(cakeF.shape)
    dcFilter = filters[-1, :, :]
    if design == "M":
        cakeF = np.sqrt(cakeF)
        dcFilter = np.sqrt(dcFilter)

    cake = np.zeros_like(cakeF, dtype=np.complex_)
    for i in range(cakeF.shape[0]):
        cakeIF = RotateLeft(cakeF[i, :, :], np.floor(np.array([size, size])/2).astype(int))
       
        ##### ifftn gives result not similar to wolfram (gives conjucate)########
        cakeIF = np.conj(np.fft.ifftn(cakeIF))
        
        cakeIF = RotateRight(cakeIF, np.floor(np.array([size,size])/2).astype(int))
        cake[i, :, :] = cakeIF

    if directional and noSymmetry:
        cake = cake * ErfSet(size, (overlapFactor * nOrientations), 2*np.pi)
    elif not noSymmetry: # necessarily not directional
        cake = np.vstack([cake, np.conj(cake)])
    # elif necessarily not directional and noSymmetry: then do nothing, apparently.
    
    # Why do we bother doing stuff with complex numbers if we only take the real part?

    # cake = np.expand_dims(cake.real, axis=1)
    cake = cake.real

    dcFilter = RotateLeft(dcFilter, np.floor(np.array([size, size])/2).astype(int))
    dcFilter = np.conj(np.fft.ifftn(dcFilter))
    dcFilter = RotateRight(dcFilter, np.floor(np.array([size, size])/2).astype(int))
    # dcFilter = np.expand_dims(dcFilter.real, axis=0)
    dcFilter = dcFilter.real

    return cake, dcFilter

def RotateLeft(array, k):
    """idk."""
    if type(k) == int or type(k) == float:
        arr1 = array[:k]
        arr2 = array[k:]
        rotated_array = np.concatenate((arr2, arr1), axis=0)
    elif len(k) == 2 and len(array.shape) == 2:
        k_0, k_1 = k
        arr1 = array[:, :k_1]
        arr2 = array[:, k_1:]
        array = np.concatenate((arr2, arr1), axis=1)
        arr1 = array[:k_0, :]
        arr2 = array[k_0:, :]
        rotated_array = np.concatenate((arr2, arr1), axis=0)
    else:
        raise ValueError(f"k = {k} is not a supported point for rotating. k should be a number or a pair of numbers.")
    return rotated_array

def RotateRight(array, k):
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

def WaveletTransform2D(f, kernels):
    """Return the wavelet transform of image `f` under the `kernels`"""
    os = np.zeros((f.shape[0], f.shape[1], kernels.shape[0]))
    f_hat = np.fft.fftn(f)
    for i in range(kernels.shape[0]):
        ψ_θ = kernels[i, :, :]
        ψ_θ_hat = np.fft.fftn(ψ_θ)
        U_θ_hat = np.fft.ifftn(ψ_θ_hat * f_hat).real
        U_θ_hat = RotateRight(U_θ_hat, np.ceil(0.1 + np.array(f.shape) / 2).astype(int))
        os[:, :, i] = U_θ_hat
    return os