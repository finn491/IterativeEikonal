# utils.py

import taichi as ti

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

@ti.func
def sanitize_index_R2(
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

@ti.func
def sanitize_index_SE2(
    index: ti.types.vector(3, ti.i32),
    input: ti.template()
) -> ti.types.vector(3, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Copied from Gijs Bellaard.

    Args:
        `index`: ti.types.vector(n=3, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=3, dtype=ti.i32) of index that is within `input`.
    """
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
        ti.math.mod(index[2], shape[2])
    ], dt=ti.i32)

@ti.func
def select_upwind_derivative(
    d_forward: ti.f32,
    d_backward: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Select the correct derivative for the upwind derivative.

    Args:
        `d_forward`: derivative in the forward direction.
        `d_backward`: derivative in the backward direction.
          
    Returns:
        derivative in the correct direction.
    """
    return ti.math.max(-d_forward, d_backward, 0) * (-1.)**(-d_forward >= d_backward)


@ti.kernel
def sparse_to_dense(
    sparse_thing: ti.template(),
    dense_thing: ti.template()
):
    """
    @taichi.func

    Convert a sparse TaiChi object on an SNode into a dense object.

    Args:
      Static:
        `sparse_thing`: Sparse TaiChi object.
      Mutated:
        `dense_thing`: Preinitialised dense TaiChi object of correct size, which
          is updated in place.
    """
    for I in ti.grouped(sparse_thing):
        dense_thing[I] = sparse_thing[I]
    sparse_thing.deactivate()