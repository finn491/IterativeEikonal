"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used on R^2,
    SE(2), and SO(3).
"""

import numpy as np
import taichi as ti
import diplib as dip


# Interpolation

@ti.func
def linear_interpolate(
    v0: ti.f32,
    v1: ti.f32,
    r: ti.i32
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v*` depending on the distance `r`, via 
    linear interpolation. Adapted from Gijs.

    Args:
        `v*`: values at points between which we want to interpolate, taking real 
          values.
        `r`: distance to the points between which we to interpolate, taking real
          values.

    Returns:
        Interpolated value.
    """
    return v0 * (1.0 - r) + v1 * r

# Derivatives

@ti.func
def select_upwind_derivative(
    d_forward: ti.f32,
    d_backward: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Select the correct derivative for the upwind derivative for morphological
    erosion.

    Args:
        `d_forward`: derivative in the forward direction.
        `d_backward`: derivative in the backward direction.
          
    Returns:
        derivative in the correct direction.
    """
    return ti.math.max(-d_forward, d_backward, 0) * (-1.)**(-d_forward >= d_backward)

# Cost

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

# Padding

def pad_array(u, pad_value=0., pad_shape=1):
    """
    Pad a numpy array `u` in each direction with `pad_shape` entries of values 
    `pad_value`.

    Args:
        `u`: np.ndarray that is to be padded.
        `pad_value`: Value with which `u` is padded.
        `pad_shape`: Number of cells that is padded in each direction, taking
          positive integral values.

    Returns:
        np.ndarray of the padded array.
    """
    return np.pad(u, pad_width=pad_shape, constant_values=pad_value)


def extract_centre_slice(u, pad_shape=1):
    """
    Select the centre part of `u`, removing padding of size `pad_shape` in each 
    direction.

    Args:
        `u`: np.ndarray from which the centre is to be extracted.
        `pad_shape`: Number of cells that is padded in each direction.

    Returns:
        np.ndarray of the unpadded array.
    """
    if type(pad_shape) == int:
        centre_slice = tuple(slice(pad_shape, dim_len - pad_shape, 1) for dim_len in u.shape)
    else: 
        centre_slice = tuple(slice(pad_shape[i], dim_len - pad_shape[i], 1) for i, dim_len in enumerate(u.shape))
    return centre_slice


def unpad_array(u, pad_shape=1):
    """
    Remove the outer `pad_shape` entries numpy array `u`.

    Args:
        `u`: np.ndarray from which the centre is to be extracted.
        `pad_shape`: Number of cells that is padded in each direction, taking
          positive integral values.

    Returns:
        np.ndarray of the unpadded array. Note that 
        `unpad_array(*, pad_shape=pad_shape)` is the left-inverse of
        `pad_array(*, pad_value=pad_value, pad_shape=pad_shape)` for all
        `pad_value`, `pad_shape`.
    """
    centre_slice = extract_centre_slice(u, pad_shape=pad_shape)
    return u[centre_slice]

# Initialisation

def get_padded_cost(cost_unpadded, pad_shape=1, pad_value=1.):
    """Pad the cost function `cost_unpadded` and convert to TaiChi object."""
    cost_np = pad_array(cost_unpadded, pad_value=pad_value, pad_shape=pad_shape)
    cost = ti.field(dtype=ti.f32, shape=cost_np.shape)
    cost.from_numpy(cost_np)
    return cost

def get_initial_W(shape, initial_condition=100., pad_shape=1):
    """Initialise the (approximate) distance map as TaiChi object."""
    W_unpadded = np.full(shape=shape, fill_value=initial_condition)
    W_np = pad_array(W_unpadded, pad_value=initial_condition, pad_shape=pad_shape)
    W = ti.field(dtype=ti.f32, shape=W_np.shape)
    W.from_numpy(W_np)
    return W

@ti.kernel
def apply_boundary_conditions(
    u: ti.template(), 
    boundarypoints: ti.template(), 
    boundaryvalues: ti.template()
):
    """
    @taichi.kernel

    Apply `boundaryvalues` at `boundarypoints` to `u`.

    Args:
        `u`: ti.field(dtype=[float]) to which the boundary conditions should be 
          applied.
        `boundarypoints`: ti.Vector.field(n=dim, dtype=[int], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.
        `boundaryvalues`: ti.Vector.field(n=dim, dtype=[float], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.

    Returns:
        ti.field equal to `u`, except at `boundarypoints`, where it equals
          `boundaryvalues`.
    """
    for I in ti.grouped(boundarypoints):
        u[boundarypoints[I]] = boundaryvalues[I]

# Image Preprocessing
        
def image_rescale(image_array, new_max=1.):
    """
    Affinely rescale values in numpy array `image_array` to be between 0. and
    `new_max`.
    """
    image_max = image_array.max()
    image_min = image_array.min()
    return new_max * (image_array - image_min) / (image_max - image_min)

def high_pass_filter(image_array, σs):
    """Apply a high pass filter with Gaussian scales `σs` to `image_array`."""
    low_frequencies = dip.Gauss(image_array, σs)
    image_array_unnormalised = image_array - low_frequencies
    image_array_filtered = image_rescale(image_array_unnormalised)
    return image_array_filtered