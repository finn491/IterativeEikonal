# interpolate.py

import taichi as ti
from eikivp.utils import (
    linear_interpolate,
    sanitize_index_R2
)
from eikivp.R2.metric import normalise


@ti.func
def bilinear_interpolate(
    v00: ti.f32,
    v01: ti.f32,
    v10: ti.f32,
    v11: ti.f32,
    r: ti.types.vector(2, ti.i32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v**` depending on the distance `r`, via 
    repeated linear interpolation (x, y). Adapted from Gijs.

    Args:
        `v**`: values at points between which we want to interpolate, taking 
          real values.
        `r`: ti.types.vector(n=2, dtype=ti.f32) defining the distance to the
          points between which we to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
        Interpolated value.
    """
    v0 = linear_interpolate(v00, v10, r[0])
    v1 = linear_interpolate(v01, v11, r[0])

    v = linear_interpolate(v0, v1, r[1])

    return v

@ti.func
def scalar_bilinear_interpolate(
    input: ti.template(),
    index: ti.template()
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of `input` at continuous `index` bilinearly, via repeated
    linear interpolation (x, y). Adapted from Gijs.

    Args:
        `input`: ti.field(dtype=[float]) in which we want to interpolate.
        `index`: ti.types.vector(n=2, dtype=ti.f32) continuous index at which we 
          want to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
    """
    r = ti.math.fract(index)

    f = ti.math.floor(index, ti.i32)
    f = sanitize_index_R2(f, input)

    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index_R2(c, input)

    v00 = input[f[0], f[1]]
    v01 = input[f[0], c[1]]
    v10 = input[c[0], f[1]]
    v11 = input[c[0], c[1]]

    v = bilinear_interpolate(v00, v01, v10, v11, r)

    return v

@ti.func
def vectorfield_bilinear_interpolate(
    vectorfield: ti.template(),
    index: ti.template(),
    G: ti.types.matrix(2, 2, ti.f32),
    cost_field: ti.template()
) -> ti.types.vector(2, ti.f32):
    """
    @taichi.func

    Interpolate vector field, normalised to 1, `vectorfield` at continuous 
    `index` bilinearly, via repeated linear interpolation (x, y).

    Args:
        `vectorfield`: ti.Vector.field(n=2, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=2, dtype=ti.f32) continuous index at which we 
          want to interpolate.
        `G`: ti.types.matrix(n=2, m=2, dtype=[float]) of constants of metric 
          tensor with respect to standard basis.
        `cost_field`: ti.field(dtype=[float]) of cost function, taking values 
          between 0 and 1.

    Returns:
        ti.types.vector(n=2, dtype=ti.f32) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index_R2(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index_R2(c, vectorfield)

    v00, w00 = vectorfield[f[0], f[1]]
    v01, w01 = vectorfield[f[0], c[1]]
    v10, w10 = vectorfield[c[0], f[1]]
    v11, w11 = vectorfield[c[0], c[1]]

    v = bilinear_interpolate(v00, v01, v10, v11, r)
    w = bilinear_interpolate(w00, w01, w10, w11, r)

    cost = scalar_bilinear_interpolate(cost_field, index)

    return normalise(ti.Vector([v, w]), G, cost)

# Apparently, interpolating angles does not work well (or I implemented something incorrectly)...
# @ti.func
# def vectorfield_bilinear_interpolate(
#     vectorfield: ti.template(),
#     index: ti.template()
# ) -> ti.types.vector(2, ti.f32):
#     """
#     @taichi.func

#     Interpolate vector field, normalised to 1, `vectorfield` at continuous 
#     `index` bilinearly, via repeated linear interpolation (x, y).

#     Args:
#         `vectorfield`: ti.Vector.field(n=2, dtype=[float]) in which we want to 
#           interpolate.
#         `index`: ti.types.vector(n=2, dtype=ti.f32) continuous index at which we 
#           want to interpolate.

#     Returns:
#         ti.types.vector(n=2, dtype=ti.f32) of value interpolation of 
#         `vectorfield` at `index`.
#     """
#     r = ti.math.fract(index)
#     f = ti.math.floor(index, ti.i32)
#     f = sanitize_index_R2(f, vectorfield)
#     c = ti.math.ceil(index, ti.i32)
#     c = sanitize_index_R2(c, vectorfield)

#     v00 = vectorfield[f[0], f[1]]
#     v01 = vectorfield[f[0], c[1]]
#     v10 = vectorfield[c[0], f[1]]
#     v11 = vectorfield[c[0], c[1]]

#     # Interpolate angles bilinearly
#     arg00 = ti.math.atan2(v00[1], v00[0])
#     arg01 = ti.math.atan2(v01[1], v01[0])
#     arg10 = ti.math.atan2(v10[1], v10[0])
#     arg11 = ti.math.atan2(v11[1], v11[0])

#     arg0 = arg00 * (1.0 - r[0]) + arg10 * r[0]
#     arg1 = arg01 * (1.0 - r[0]) + arg11 * r[0]

#     arg = arg0 * (1.0 - r[1]) + arg1 * r[1]

#     return ti.Vector([ti.math.cos(arg), ti.math.sin(arg)])