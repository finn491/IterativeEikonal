# metric.py

import taichi as ti
import numpy as np

# Metric Inversion

def invert_metric(G):
    """
    Invert the metric tensor defined by the matrix `G`. If `G` is semidefinite,
    e.g. when we are dealing with a sub-Riemannian metric, the metric is first
    made definite by adding `1e-8` to the diagonal.

    Args:
        `G`: np.ndarray(shape=(2, 2), dtype=[float]) of matrix of left invariant
          metric tensor field with respect to standard basis.
    """
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.inv(G + np.identity(2) * 1e-8)
    return G_inv


# Normalisation

@ti.func
def normalise(
    vec: ti.types.vector(2, ti.f32),
    G: ti.types.matrix(2, 2, ti.f32),
    cost: ti.f32
) -> ti.types.vector(2, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in standard coordinates, to 1 with respect to 
    the datadriven left invariant metric tensor defined by `G` and `cost`.

    Args:
        `vec`: ti.types.vector(n=2, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=2, m=2, dtype=[float]) of constants of metric 
          tensor with respect to standard basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm(vec, G, cost)

@ti.func
def norm(
    vec: ti.types.vector(2, ti.f32),
    G: ti.types.matrix(2, 2, ti.f32),
    cost: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in standard coordinates with respect
    to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=2, dtype=[float]) which we want to normalise.
        `G`: ti.types.matrix(n=2, m=2, dtype=[float]) of constants of metric 
          tensor with respect to standard basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        Norm of `vec`.
    """
    c_1, c_2 = vec[0], vec[1]
    return ti.math.sqrt(
            1 * G[0, 0] * c_1 * c_1 +
            2 * G[0, 1] * c_1 * c_2 + # Metric tensor is symmetric.
            1 * G[1, 1] * c_2 * c_2
    ) * cost


# Coordinate Transforms

def coordinate_real_to_array(x, y, x_min, y_max, dxy):
    """
    Compute the array indices (I, J) of the point defined by real coordinates 
    (`x`, `y`). Can broadcast over entire arrays of real coordinates.

    Args:
        `x`: x-coordinate of the point.
        `y`: y-coordinate of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    J = np.rint((x - x_min) / dxy).astype(int)
    I = np.rint((y_max - y) / dxy).astype(int)
    return I, J


def coordinate_array_to_real(I, J, x_min, y_max, dxy):
    """
    Compute the real coordinates (x, y) of the point defined by array indices 
    (`I`, `J`). Can broadcast over entire arrays of array indices.

    Args:
        `I`: I index of the point.
        `J`: J index of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    x = x_min + J * dxy
    y = y_max - I * dxy
    return x, y

# Apparently shouldn't do this...
# @ti.func
# def vector_standard_to_array(
#     vector_standard: ti.types.vector(2, ti.f32),
#     dxy: ti.f32
# ) -> ti.types.vector(2, ti.f32):
#     """
#     @taichi.func

#     Change the coordinates of the vector represented by `vector_standard` from 
#     the standard frame to the frame of array indices, given that the spatial
#     resolution is `dxy`.

#     Args:
#       Static:
#         `vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in LI
#           coordinates.
#         `dxy`: Spatial resolution, taking values greater than 0.
#     """
#     return vector_standard / dxy