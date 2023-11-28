# cleanarrays.py

import taichi as ti
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def convert_array_to_image(image_array):
    """Convert numpy array `image_array` to a grayscale PIL Image object."""
    if image_array.dtype == "uint8":
        image = Image.fromarray(image_array, mode="L")
    else:
        image = Image.fromarray((image_array * 255).astype("uint8"), mode="L")
    return image

def view_image_array(image_array):
    """View numpy array `image_array` as a grayscale image."""
    image = convert_array_to_image(image_array)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()
    return image, fig

def view_image_arrays_side_by_side(image_array_list):
    """
    View list of numpy array `image_array_list` side by side as grayscale 
    images.
    """
    image_list = []
    ncols = len(image_array_list)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(5 * ncols, 5))
    for i, image_array in enumerate(image_array_list):
        image = convert_array_to_image(image_array)
        image_list.append(image)
        ax[i].imshow(image_array, cmap="gray")
        ax[i].set_axis_off()
    return image_list, fig

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
