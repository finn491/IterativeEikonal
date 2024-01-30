# interpolate.py

import taichi as ti
from eikivp.utils import linear_interpolate, sanitize_index_SE2
from eikivp.R2.metric import normalise_LI, normalise_static

# Helper Functions


@ti.func
def trilinear_interpolate(
    v000: ti.f32, 
    v001: ti.f32, 
    v010: ti.f32, 
    v011: ti.f32, 
    v100: ti.f32, 
    v101: ti.f32, 
    v110: ti.f32, 
    v111: ti.f32,
    r: ti.types.vector(3, ti.i32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v***` depending on the distance `r`, via 
    repeated linear interpolation (x, y, θ). Adapted from Gijs Bellaard.

    Args:
        `v***`: values at points between which we want to interpolate, taking 
          real values.
        `r`: ti.types.vector(n=3, dtype=ti.f32) defining the distance to the
          points between which we to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
        Interpolated value.
    """
    v00 = linear_interpolate(v000, v100, r[0])
    v01 = linear_interpolate(v001, v101, r[0])
    v10 = linear_interpolate(v010, v110, r[0])
    v11 = linear_interpolate(v011, v111, r[0])

    v0 = linear_interpolate(v00, v10, r[1])
    v1 = linear_interpolate(v01, v11, r[1])

    v = linear_interpolate(v0, v1, r[2])

    return v

@ti.func
def scalar_trilinear_interpolate(
    input: ti.template(), 
    index: ti.types.vector(3, ti.f32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of `input` at continuous `index` trilinearly, via repeated
    linear interpolation (x, y, θ). Copied from Gijs Bellaard.

    Args:
        `input`: ti.field(dtype=[float]) in which we want to interpolate.
        `index`: ti.types.vector(n=3, dtype=ti.f32) continuous index at which we 
          want to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
    """
    r = ti.math.fract(index)

    f = ti.math.floor(index, ti.i32)
    f = sanitize_index_SE2(f, input)

    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index_SE2(c, input)
    
    v000 = input[f[0], f[1], f[2]]
    v001 = input[f[0], f[1], c[2]]
    v010 = input[f[0], c[1], f[2]]
    v011 = input[f[0], c[1], c[2]]
    v100 = input[c[0], f[1], f[2]]
    v101 = input[c[0], f[1], c[2]]
    v110 = input[c[0], c[1], f[2]]
    v111 = input[c[0], c[1], c[2]]

    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)

    return v

# I don't think this works properly, we should actually also interpolate the frame...
@ti.func
def vectorfield_trilinear_interpolate_LI(
    vectorfield: ti.template(),
    index: ti.template(),
    G: ti.types.matrix(3, 3, ti.f32)
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    ---------------------------PROBABLY DOESN'T WORK---------------------------

    Interpolate vector field, normalised to 1 and given in left invariant
    coordinates, `vectorfield` at continuous `index` trilinearly, via repeated 
    linear interpolation (x, y, θ).

    Args:
        `vectorfield`: ti.Vector.field(n=3, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=3, dtype=[float]) continuous index at which 
          we want to interpolate.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index_SE2(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index_SE2(c, vectorfield)

    u000, v000, w000 = vectorfield[f[0], f[1], f[2]]
    u001, v001, w001 = vectorfield[f[0], f[1], c[2]]
    u010, v010, w010 = vectorfield[f[0], c[1], f[2]]
    u011, v011, w011 = vectorfield[f[0], c[1], c[2]]
    u100, v100, w100 = vectorfield[c[0], f[1], f[2]]
    u101, v101, w101 = vectorfield[c[0], f[1], c[2]]
    u110, v110, w110 = vectorfield[c[0], c[1], f[2]]
    u111, v111, w111 = vectorfield[c[0], c[1], c[2]]

    u = trilinear_interpolate(u000, u001, u010, u011, u100, u101, u110, u111, r)
    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)
    w = trilinear_interpolate(w000, w001, w010, w011, w100, w101, w110, w111, r)

    return normalise_LI(ti.Vector([u, v, w]), G)

@ti.func
def vectorfield_trilinear_interpolate_static(
    vectorfield: ti.template(),
    index: ti.template(),
    θs: ti.template(),
    G: ti.types.matrix(3, 3, ti.f32)
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Interpolate vector field, normalised to 1 and given in static
    coordinates, `vectorfield` at continuous `index` trilinearly, via repeated 
    linear interpolation (x, y, θ).

    Args:
        `vectorfield`: ti.Vector.field(n=3, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=3, dtype=[float]) continuous index at which 
          we want to interpolate.
        `θs`: angle coordinate at each grid point.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index_SE2(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index_SE2(c, vectorfield)

    u000, v000, w000 = vectorfield[f[0], f[1], f[2]]
    u001, v001, w001 = vectorfield[f[0], f[1], c[2]]
    u010, v010, w010 = vectorfield[f[0], c[1], f[2]]
    u011, v011, w011 = vectorfield[f[0], c[1], c[2]]
    u100, v100, w100 = vectorfield[c[0], f[1], f[2]]
    u101, v101, w101 = vectorfield[c[0], f[1], c[2]]
    u110, v110, w110 = vectorfield[c[0], c[1], f[2]]
    u111, v111, w111 = vectorfield[c[0], c[1], c[2]]

    u = trilinear_interpolate(u000, u001, u010, u011, u100, u101, u110, u111, r)
    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)
    w = trilinear_interpolate(w000, w001, w010, w011, w100, w101, w110, w111, r)

    θ = scalar_trilinear_interpolate(θs, index)

    return normalise_static(ti.Vector([u, v, w]), G, θ)