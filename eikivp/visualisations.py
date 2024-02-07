# cleanarrays.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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

def plot_image_array(image_array, x_min, x_max, y_min, y_max, cmap="gray", figsize=(10, 10), fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.imshow(image_array, cmap=cmap, extent=(x_min, x_max, y_min, y_max))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return fig, ax