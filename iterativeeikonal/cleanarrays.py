# cleanarrays.py

import taichi as ti
import numpy as np

def pad_array(u, pad_value=0, pad_shape=1):
    """"""
    padded_shape = tuple(dim_len + 2 * pad_shape for dim_len in u.shape)
    u_padded = np.full(padded_shape, fill_value=pad_value, dtype=u.dtype)
    centre_slice = extract_centre_slice(u_padded, pad_shape=pad_shape)
    u_padded[centre_slice] = u
    return u_padded

def extract_centre_slice(u, pad_shape=1):
    return tuple(slice(pad_shape, dim_len - pad_shape, 1) for dim_len in u.shape)

def unpad_array(u, pad_shape=1):
    centre_slice = extract_centre_slice(u, pad_shape=pad_shape)
    return u[centre_slice]