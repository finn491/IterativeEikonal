"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used on R^2.
"""

import numpy as np
import taichi as ti

# Safe Indexing

@ti.func
def sanitize_index(
    index: ti.types.vector(2, ti.i32),
    input: ti.template()
) -> ti.types.vector(2, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Adapted from Gijs.

    Args:
        `index`: ti.types.vector(n=2, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=2, dtype=ti.i32) of index that is within `input`.
    """
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
    ], dt=ti.i32)

# Coordinate Transforms

def coordinate_real_to_array(x, y, x_min, y_max, dxy):
    """
    Compute the array indices (I, J) of the point defined by real coordinates 
    (`x`, `y`). Can broadcast over entire arrays of real coordinates.

    Args:
        `x`: x-coordinate of the point.
        `y`: y-coordinate of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_max`: maximum value of y-coordinates in rectangular domain.
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
        `y_max`: maximum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    x = x_min + J * dxy
    y = y_max - I * dxy
    return x, y

def align_to_real_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to standard array convention,
    in terms of indices with respect to arrays aligned with real axes (see Notes
    for more explanation). Here, `shape` gives the shape of the array in which
    we index _after_ aligning with real axes.

    Args:
        `point`: Tuple[int, int] describing point with respect to standard array
          indexing convention.
        `shape`: shape of array, aligned to real axes, in which we want to
          index. Note that `0 <= point[0] <= shape[1] - 1` and 
          `0 <= point[1] <= shape[0]`.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0]

def align_to_real_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `field`: np.ndarray of scalar field given with respect to standard array
          convention.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    field_flipped = np.flip(field, axis=0)
    field_aligned = np.transpose(field_flipped, axes=(1, 0))
    return field_aligned

def align_to_real_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in indices with respect to standard array 
    convention, with real axes (see Notes for more explanation).
    
    Args:
        `vector_field`: np.ndarray of vector field given with respect to 
          standard array convention.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    vector_field_flipped = np.flip(vector_field, axis=0)
    vector_field_aligned = np.transpose(vector_field_flipped, axes=(1, 0, 2))
    return vector_field_aligned

def align_to_standard_array_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to arrays aligned with real 
    axes, in terms of indices with respect to standard array convention, (see 
    Notes for more explanation). Here, `shape` gives the shape of the array in 
    which we index _after_ aligning with standard array convention.

    Args:
        `point`: Tuple[int, int] describing point with respect to arrays aligned
          with real axes.
        `shape`: shape of array, with respect to standard array convention, in 
          which we want to index. Note that `0 <= point[0] <= shape[1] - 1` and 
          `0 <= point[1] <= shape[0]`.

    Notes:
        Alignment is achieved by first transposing and subsequently flipping the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0]

def align_to_standard_array_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to arrays aligned with real
    axes, with respect to standard array convention (see Notes for more 
    explanation).

    Args:
        `field`: np.ndarray of scalar field given in indices with respect to
          arrays aligned with real axes.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    field_transposed = np.transpose(field, axes=(1, 0))
    field_aligned = np.flip(field_transposed, axis=0)
    return field_aligned

def align_to_standard_array_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `vector_field`: np.ndarray of vector field given in indices with respect
          to arrays aligned with real axes.

    Notes:
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    vector_field_transposed = np.transpose(vector_field, axes=(1, 0, 2))
    vector_field_aligned = np.flip(vector_field_transposed, axis=0)
    return vector_field_aligned

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