# cleanarrays.py

import taichi as ti
import numpy as np
from PIL import Image

# NumPy Arrays


def pad_array(u, pad_value=0., pad_shape=1):
    """
    Pad a numpy array `u` in each direction with `pad_shape` entries of values 
    `pad_value`.
    """
    padded_shape = tuple(dim_len + 2 * pad_shape for dim_len in u.shape)
    u_padded = np.full(padded_shape, fill_value=pad_value, dtype=u.dtype)
    centre_slice = extract_centre_slice(u_padded, pad_shape=pad_shape)
    u_padded[centre_slice] = u
    return u_padded


def extract_centre_slice(u, pad_shape=1):
    """
    Select the centre part of `u`, removing padding of size `pad_shape` in each 
    direction.
    """
    return tuple(slice(pad_shape, dim_len - pad_shape, 1) for dim_len in u.shape)


def unpad_array(u, pad_shape=1):
    """
    Remove the outer `pad_shape` entries numpy array `u`.
    """
    centre_slice = extract_centre_slice(u, pad_shape=pad_shape)
    return u[centre_slice]

def view_image_array(image_array):
    """View numpy array `image_array` as a grayscale image."""
    return Image.fromarray((image_array * 255).astype("uint8"), mode="L")

# TaiChi Fields


@ti.kernel
def apply_boundary_conditions(
    u: ti.template(), 
    boundarypoints: ti.template(), 
    boundaryvalues: ti.template()
):
    """Apply `boundaryvalues` at `boundarypoints` to `u`."""
    for I in ti.grouped(boundarypoints):
        u[boundarypoints[I]] = boundaryvalues[I]
