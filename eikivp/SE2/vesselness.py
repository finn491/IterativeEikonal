"""
    vesselness
    ==========
    Provides tools to compute vesselness scores on SE(2), namely:
      1. `single_scale_vesselness`: compute the singlescale vesselness using a
      crossing-preserving vesselness[1].
      2. `multi_scale_vesselness`: compute the multiscale vesselness by
      combining crossing-preserving vesselnesses at various scales via maximum
      projection.
    Additionally, we have code for regularising functions on SE(2), namely:
      1. `convolve_with_kernel_x_dir`: convolve a field with a 1D kernel along
      the x-direction.
      2. `convolve_with_kernel_y_dir`: convolve a field with a 1D kernel along
      the y-direction.
      3. `convolve_with_kernel_θ_dir`: convolve a field with a 1D kernel along
      the θ-direction.
      4. `gaussian_derivative_kernel`: computes 1D Gaussian derivative kernels
      of order 0 and 1, using an algorithm that improves the accuracy of higher
      order derivative kernels with small widths, based on the DIPlib[2]
      algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.
    We use that the spatially isotropic diffusion equation on SE(2) can be
    solved by convolving in the x-, y-, and θ-direction with some 1D kernel. For
    the x- and y-directions, this kernel is a Gaussian; for the θ-direction the
    kernel looks like a Gaussian if the amount of diffusion is sufficiently
    small.

    TODO: maybe add in correct kernel for θ-direction?

    References:
      [1]: J. Hannink, R. Duits, and E. Bekkers.
      "Crossing-Preserving Multi-scale Vesselness". In: Medical Image Computing
      and Computer-Assisted Intervention 8674 (2014), pp. 603-610.
      DOI:10.1007/978-3-319-10470-6_75
      [2]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
      E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
      M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
      J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
      and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
"""

import numpy as np
import taichi as ti
from eikivp.SE2.utils import sanitize_index
from eikivp.SE2.derivatives import (
    A11_central,
    A22_central
)

# Vesselness

def single_scale_vesselness(U_np, mask_np, θs_np, σ_s, σ_o, σ_s_ext, σ_o_ext, dxy, dθ):
    """
    Compute the crossing-preserving vesselness[1] of the orientation score of an
    image `U_np`.

    Args:
        `U_np`: np.ndarray of orientation score, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray of mask in which vesselness is computed, to deal
          with boundary effects of the image, with shape [Nx, Ny, Nθ].
        `θs_np`: orientation coordinate at every point in the grid on which
          `cost` is sampled, with shape [Nx, Ny, Nθ].
        `σ_s`: standard deviation in spatial units of the internal
          regularisation in the spatial directions before taking derivatives.
        `σ_o`: standard deviation in radians of the internal regularisation
          in the orientational direction before taking derivatives.
        `σ_s_ext`: standard deviation in spatial units of the external
          regularisation in the spatial direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.
        `σ_o_ext`: standard deviation in radians of the internal regularisation
          in the orientational direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.

    Returns:
        np.ndarray of vesselness of orientation score of retinal image `U_np`.

    References:
      [1]: J. Hannink, R. Duits, and E. Bekkers.
        "Crossing-Preserving Multi-scale Vesselness". In: Medical Image
        Computing and Computer-Assisted Intervention 8674 (2014), pp. 603-610.
        DOI:10.1007/978-3-319-10470-6_75
    """
    # Initialise TaiChi objects.
    shape = U_np.shape
    U = ti.field(dtype=ti.f32, shape=shape)
    U.from_numpy(U_np)
    mask = ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)
    θs = ti.field(dtype=ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    convolution_storage_1 = ti.field(dtype=ti.f32, shape=shape)
    convolution_storage_2 = ti.field(dtype=ti.f32, shape=shape)
    U_int = ti.field(dtype=ti.f32, shape=shape)
    A11_U = ti.field(dtype=ti.f32, shape=shape)
    A22_U = ti.field(dtype=ti.f32, shape=shape)
    A11_U_ext = ti.field(dtype=ti.f32, shape=shape)
    A22_U_ext = ti.field(dtype=ti.f32, shape=shape)
    Q = ti.field(dtype=ti.f32, shape=shape)
    S = ti.field(dtype=ti.f32, shape=shape)
    R = ti.field(dtype=ti.f32, shape=shape)
    V = ti.field(dtype=ti.f32, shape=shape)
    ## Compute Gaussian kernels.
    σ_s_pixels = σ_s / dxy
    k_s, radius_s = gaussian_derivative_kernel(σ_s_pixels, 0, dxy=dxy)
    σ_o_pixels = σ_o / dθ
    k_o, radius_o = gaussian_derivative_kernel(σ_o_pixels, 0, dxy=dθ)
    σ_s_ext_pixels = σ_s_ext / dxy
    k_s_ext, radius_s_ext = gaussian_derivative_kernel(σ_s_ext_pixels, 0, dxy=dxy)
    σ_o_ext_pixels = σ_o_ext / dθ
    k_o_ext, radius_o_ext = gaussian_derivative_kernel(σ_o_ext_pixels, 0, dxy=dθ)
    
    single_scale_vesselness_backend(U, mask, dxy, θs, k_s, radius_s, k_o, radius_o, U_int, A11_U, A22_U, k_s_ext,
                                    radius_s_ext, k_o_ext, radius_o_ext, A11_U_ext, A22_U_ext, Q, S, R, V,
                                    convolution_storage_1, convolution_storage_2)
    return V.to_numpy()

@ti.kernel
def single_scale_vesselness_backend(
    U: ti.template(),
    mask: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    k_s: ti.template(),
    radius_s: ti.i32,
    k_o: ti.template(),
    radius_o: ti.i32,
    U_int: ti.template(),
    A11_U: ti.template(),
    A22_U: ti.template(),
    k_s_ext: ti.template(),
    radius_s_ext: ti.i32,
    k_o_ext: ti.template(),
    radius_o_ext: ti.i32,
    A11_U_ext: ti.template(),
    A22_U_ext: ti.template(),
    Q: ti.template(),
    S: ti.template(),
    R: ti.template(),
    V: ti.template(),
    convolution_storage_1: ti.template(),
    convolution_storage_2: ti.template()
):
    """
    @taichi.kernel

    Compute the crossing-preserving vesselness[1] of the orientation score of an
    image `U`.

    Args:
      Static:
        `U`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of orientation score.
        `mask`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of mask in which
          vesselness is computed, to deal with boundary effects of the image.
        `dxy`: Spatial step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
        `k_s`: ti.field(dtype=ti.f32, shape=2*`radius_s`+1) Gaussian kernel used
          for spatial regularisation.
        `radius_s`: radius at which kernel `k_s` is truncated, taking integer
          values greater than 0.
        `k_o`: ti.field(dtype=ti.f32, shape=2*`radius_o`+1) Gaussian kernel used
          for orientational regularisation.
        `radius_o`: radius at which kernel `k_o` is truncated, taking integer
          values greater than 0.
        `k_ext_s`: ti.field(dtype=[float], shape=2*`radius_s`+1) Gaussian kernel
          used for spatial regularisation.
        `radius_ext_s`: radius at which kernel `k_s` is truncated, taking
          integer values greater than 0.
        `k_ext_o`: ti.field(dtype=[float], shape=2*`radius_o`+1) Gaussian kernel
          used for orientational regularisation.
        `radius_ext_o`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `U_int`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of internally
          regularised orientation score, which is updated in place.
        `A**_U`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of central
          derivatives, which are updated in place.
        `A**_U_ext`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of externally
          regularised central derivatives, which are updated in place.
        `Q`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of convexity criterion,
          which is updated in place.
        `S`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of structure measure,
          which is updated in place.
        `R`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of anisotropy measure,
          which is updated in place.
        `V`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of vesselness, which is
          updated in place.
        `convolution_storage_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ])
          arrays to hold intermediate results when performing convolutions.

    References:
      [1]: J. Hannink, R. Duits, and E. Bekkers.
        "Crossing-Preserving Multi-scale Vesselness". In: Medical Image
        Computing and Computer-Assisted Intervention 8674 (2014), pp. 603-610.
        DOI:10.1007/978-3-319-10470-6_75
    """
    # Compute relevant "Hessian" components:
    ## Apply internal regularisation.
    convolve_with_kernel_x_dir(U, k_s, radius_s, convolution_storage_1)
    convolve_with_kernel_y_dir(convolution_storage_1, k_s, radius_s, convolution_storage_2)
    convolve_with_kernel_θ_dir(convolution_storage_2, k_o, radius_o, U_int)
    ## Compute A_11 and A_22 derivatives.
    A11_central(U_int, dxy, θs, A11_U)
    A22_central(U_int, dxy, θs, A22_U)
    ## Apply external regularisation to derivatives.
    convolve_with_kernel_x_dir(A11_U, k_s_ext, radius_s_ext, convolution_storage_1)
    convolve_with_kernel_y_dir(convolution_storage_1, k_s_ext, radius_s_ext, convolution_storage_2)
    convolve_with_kernel_θ_dir(convolution_storage_2, k_o_ext, radius_o_ext, A11_U_ext)
    convolve_with_kernel_x_dir(A22_U, k_s_ext, radius_s_ext, convolution_storage_1)
    convolve_with_kernel_y_dir(convolution_storage_1, k_s_ext, radius_s_ext, convolution_storage_2)
    convolve_with_kernel_θ_dir(convolution_storage_2, k_o_ext, radius_o_ext, A22_U_ext)

    # Combine components.
    for I in ti.grouped(V):
        # Adapted from "SE2-Vesselness-LI-Simple.nb", found in
        # S:\Lieanalysis\VICI\researchers\FinnSherry\Mathematica\Vascular Tracking OS\CodeA-SE2-Vesselness\Relevant-Sub-Routines-in-A\SE2-Vesselness
        λ1 = A11_U_ext[I]
        c = A22_U_ext[I]
        Q[I] = c # Convexity criterion.
        S[I] = ti.math.sqrt(λ1**2 + c**2) # Structure measure.
        R[I] = λ1 / c # (c + ε * (-1)**(c < 0.)) # Anisotropy measure. 

    σ1 = 0.5
    σ2 = S[0, 0, 0]
    for I in ti.grouped(S):
        ti.atomic_max(σ2, S[I])

    for I in ti.grouped(V):
        lineness = ti.math.exp(-R[I]**2 / (2 * σ1**2)) * (1 - ti.math.exp(-S[I]**2 / (0.1 * σ2**2)))
        # Vessels are dark lines, so they are locally convex. We can assess
        # local convexity by looking at the left invariant perpendicular
        # Laplacian, given by A_22.
        is_convex = Q[I] > 0.
        V[I] = lineness * mask[I] * is_convex

def multi_scale_vesselness(U, mask, θs, σ_s_list, σ_o, σ_s_ext, σ_o_ext, dxy, dθ):
    """
    Compute the multiscale vesselness of the orientation score of an image
    `U_np` by combining crossing-preserving vesselnesses[1] at various scales
    via maximum projection.

    Args:
        `U_np`: np.ndarray of orientation score, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray of mask in which vesselness is computed, to deal
          with boundary effects of the image, with shape [Nx, Ny, Nθ].
        `θs_np`: orientation coordinate at every point in the grid on which
          `cost` is sampled, with shape [Nx, Ny, Nθ].
        `σ_s_list`: standard deviations in pixels of the internal regularisation
          in the spatial directions before taking derivatives.
        `σ_o`: standard deviation in pixels of the internal regularisation
          in the orientational direction before taking derivatives.
        `σ_s_ext`: standard deviation in pixels of the external regularisation
          in the spatial direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.
        `σ_o_ext`: standard deviation in pixels of the internal regularisation
          in the orientational direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.

    Returns:
        np.ndarray of multi scale vesselness of orientation score of retinal
          image `U_np`.

    References:
      [1]: J. Hannink, R. Duits, and E. Bekkers.
        "Crossing-Preserving Multi-scale Vesselness". In: Medical Image
        Computing and Computer-Assisted Intervention 8674 (2014), pp. 603-610.
        DOI:10.1007/978-3-319-10470-6_75
    """
    Nx, Ny, Nθ = U.shape
    Vs = np.zeros((len(σ_s_list), Nx, Ny, Nθ))
    for i, σ_s in enumerate(σ_s_list):
        Vs[i] = single_scale_vesselness(U, mask, θs, σ_s, σ_o, σ_s_ext, σ_o_ext, dxy, dθ)
    V_unnormalised = Vs.max(0) # Vs.sum(0) ?
    V = (V_unnormalised - V_unnormalised.min()) / (V_unnormalised.max() - V_unnormalised.min())
    return V

# Regularisers

@ti.func
def convolve_with_kernel_x_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` the 1D kernel `k` in the x-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
        of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = sanitize_index(ti.Vector([x - radius + i, y, θ], dt=ti.i32), u)
            s += u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

@ti.func
def convolve_with_kernel_y_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` the 1D kernel `k` in the y-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = sanitize_index(ti.Vector([x, y - radius + i, θ], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

@ti.func
def convolve_with_kernel_θ_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` the 1D kernel `k` in the y-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            # This may in fact give the correct convolution...
            index = sanitize_index(ti.Vector([x, y, θ - radius + i], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

def gaussian_derivative_kernel(σ, order, truncate=5., dxy=1.):
    """Compute kernel for 1D Gaussian derivative of order `order` at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
        `σ`: scale of Gaussian, taking values greater than 0.
        `order`: order of the derivative, taking values 0 or 1.
        `truncate`: number of scales `σ` at which kernel is truncated, taking 
          values greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.

    Returns:
        Tuple ti.field(dtype=[float], shape=2*radius+1) of the Gaussian kernel
          and the radius of the kernel.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    radius = int(σ * truncate + 0.5)
    k = ti.field(dtype=ti.f32, shape=2*radius+1)
    match order:
        case 0:
            gaussian_derivative_kernel_order_0(σ, radius, dxy, k)
        case 1:
            gaussian_derivative_kernel_order_1(σ, radius, dxy, k)
        case _:
            raise(NotImplementedError(f"Order {order} has not been implemented yet; choose order 0 or 1."))
    return k, radius

@ti.kernel
def gaussian_derivative_kernel_order_0(
    σ: ti.f32,
    radius: ti.i32,
    dxy: ti.f32,
    k: ti.template()
):
    """
    @taichi.kernel
    
    Compute 1D Gaussian kernel at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
      Static:
        `σ`: scale of Gaussian, taking values greater than 0.
        `radius`: radius at which kernel is truncated, taking integer values
          greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel, which is
          updated in place.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    for i in range(2*radius+1):
        x = -radius + i
        val = ti.math.exp(-x**2 / (2 * σ**2))
        k[i] = val
    normalise_field(k, 1/dxy)

@ti.kernel
def gaussian_derivative_kernel_order_1(
    σ: ti.f32,
    radius: ti.i32,
    dxy: ti.f32,
    k: ti.template()
):
    """
    @taichi.kernel
    
    Compute kernel for 1D Gaussian derivative of order 1 at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
      Static:
        `σ`: scale of Gaussian, taking values greater than 0.
        `radius`: radius at which kernel is truncated, taking integer values
          greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel, which is
          updated in place.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    moment = 0.
    for i in range(2*radius+1):
        x = -radius + i
        val = x * ti.math.exp(-x**2 / (2 * σ**2))
        moment += x * val
        k[i] = val
    divide_field(k, -moment * dxy)



# Helper Functions

@ti.func
def normalise_field(
    field: ti.template(),
    norm: ti.f32
):
    """
    @ti.func

    Normalise `field` to sum to `norm`.

    Args:
      Static:
        `norm`: desired norm for `field`, taking values greater than 0.
      Mutated:
        `field`: ti.field that is to be normalised, which is updated in place.    
    """
    current_norm = 0.
    for I in ti.grouped(field):
        ti.atomic_add(current_norm, field[I])
    norm_factor = norm / current_norm
    for I in ti.grouped(field):
        field[I] *= norm_factor

@ti.func
def divide_field(
    field: ti.template(),
    denom: ti.f32
):
    for I in ti.grouped(field):
        field[I] /= denom