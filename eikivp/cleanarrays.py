# cleanarrays.py

import numpy as np
import diplib as dip
from PIL import Image
import matplotlib.pyplot as plt


def convert_array_to_image(image_array):
    """Convert numpy array `image_array` to a grayscale PIL Image object."""
    if image_array.dtype == "uint8":
        image = Image.fromarray(image_array, mode="L")
    else:
        image = Image.fromarray((image_array * 255).astype("uint8"), mode="L")
    return image

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

def view_image_array(image_array):
    """View numpy array `image_array` as a grayscale image."""
    image = convert_array_to_image(image_array)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_axis_off()
    return image, fig, ax

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
        ax[i].imshow(image_array, cmap="gray", origin="upper")
        ax[i].set_axis_off()
    return image_list, fig, ax