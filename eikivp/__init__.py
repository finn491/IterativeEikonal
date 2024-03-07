"""
    EikIVP
    ======

    The Python package *eikivp* contains methods to solve the Eikonal PDE on
    R^2, SE(2), and SO(3) using the iterative Initial Value Problem (IVP)
    technique described in Bekkers et al. "A PDE approach to Data-Driven
    Sub-Riemannian Geodesics in SE(2)" (2015), and to find geodesics connecting
    points with respect to the distance map that solves the Eikonal PDE.

    One application in which we want to solve the Eikonal PDE and subsequently
    find geodesics connecting pairs of points is vascular tracking. This package
    contains methods to construct data-driven metrics on R^2 and SE(2), based
    on multiscale vesselness filters, that will lead to geodesics that 
    (hopefully) track vessels.

    Summary: compute distance map and geodesics with respect to data-driven 
    metric on R^2, SE(2), and SO(3).
"""

# Access entire backend
import eikivp.utils
import eikivp.visualisations
import eikivp.costfunction
import eikivp.orientationscore
import eikivp.R2
import eikivp.SE2

# Most important functions are available at top level
## R2
from eikivp.R2.distancemap import eikonal_solver as eikonal_solver_R2
from eikivp.R2.distancemap import eikonal_solver_uniform as eikonal_solver_R2_uniform
from eikivp.R2.backtracking import geodesic_back_tracking as geodesic_back_tracking_R2
## SE2
### Riemannian
from eikivp.SE2.Riemannian.distancemap import eikonal_solver as eikonal_solver_SE2_Riemannian
from eikivp.SE2.Riemannian.distancemap import eikonal_solver_uniform as eikonal_solver_SE2_Riemannian_uniform
from eikivp.SE2.Riemannian.backtracking import geodesic_back_tracking as geodesic_back_tracking_SE2_Riemannian
### Sub-Riemannian
from eikivp.SE2.subRiemannian.distancemap import eikonal_solver as eikonal_solver_SE2_sub_Riemannian
from eikivp.SE2.subRiemannian.distancemap import eikonal_solver_uniform as eikonal_solver_SE2_sub_Riemannian_uniform
from eikivp.SE2.subRiemannian.backtracking import geodesic_back_tracking as geodesic_back_tracking_SE2_sub_Riemannian
### Plus controller
from eikivp.SE2.plus.distancemap import eikonal_solver as eikonal_solver_SE2_plus
from eikivp.SE2.plus.distancemap import eikonal_solver_uniform as eikonal_solver_SE2_plus_uniform
from eikivp.SE2.plus.backtracking import geodesic_back_tracking as geodesic_back_tracking_SE2_plus
### Single top level function to select any controller
def eikonal_solver_SE2(cost, source_point, dxy, dθ, θs, controller="sub-Riemannian", G=None, ξ=None, plus_softness=0.,
                       target_point=None, n_max=1e5, n_max_initialisation=1e4, n_check=None,
                       n_check_initialisation=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a datadriven left invariant
    norm, with source at `source_point`, using the iterative method described in
    Bekkers et al. "A PDE approach to Data-Driven Sub-Riemannian Geodesics in
    SE(2)" (2015).

    Args:
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1.
        `source_point`: Tuple[int] describing index of source point in 
          `cost`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `cost`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e4.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max`. Defaults to `None`; if no
          `n_check` is passed, convergence is only checked at `n_max`.
        `n_check_initialisation`: Number of iterations between each convergence
          check in the initialisation, taking positive values. Should be at most
          `n_max_initialisation`. Defaults to `None`; if no
          `n_check_initialisation` is passed, convergence is only checked at
          `n_max_initialisation`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by 
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by 
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!")
        W, grad_W = eikonal_solver_SE2_Riemannian(cost, source_point, G, dxy, dθ, θs, target_point=target_point,
                                                  n_max=n_max, n_max_initialisation=n_max_initialisation,
                                                  n_check=n_check, n_check_initialisation=n_check_initialisation,
                                                  tol=tol, dε=dε, initial_condition=initial_condition)
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SE2_sub_Riemannian(cost, source_point, ξ, dxy, dθ, θs, target_point=target_point,
                                                      n_max=n_max, n_max_initialisation=n_max_initialisation,
                                                      n_check=n_check, n_check_initialisation=n_check_initialisation,
                                                      tol=tol, dε=dε, initial_condition=initial_condition)
    elif controller == "plus":
        if ξ is None:
            raise ValueError(f"When using the plus controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SE2_plus(cost, source_point, ξ, dxy, dθ, θs, plus_softness=plus_softness,
                                            target_point=target_point, n_max=n_max,
                                            n_max_initialisation=n_max_initialisation, n_check=n_check,
                                            n_check_initialisation=n_check_initialisation, tol=tol, dε=dε,
                                            initial_condition=initial_condition)
    else:
        raise ValueError(f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus".""")
    return W, grad_W

def eikonal_solver_SE2_uniform(domain_shape, source_point, dxy, dθ, θs, controller="sub-Riemannian", G=None, ξ=None,
                               plus_softness=0., target_point=None, n_max=1e5,n_check=None, tol=1e-3, dε=1.,
                               initial_condition=100.):
    """
    Solve the Eikonal PDE on SE(2) equipped with a left invariant norm, with
    source at `source_point`, using the iterative method described in Bekkers et
    al. "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)"
    (2015).

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, with
          respect to standard array indexing.
        `source_point`: Tuple[int] describing index of source point in 
          `domain_shape`.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `domain_shape`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left 
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by 
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by 
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!")
        W, grad_W = eikonal_solver_SE2_Riemannian_uniform(domain_shape, source_point, G, dxy, dθ, θs,
                                                          target_point=target_point, n_max=n_max, n_check=n_check,
                                                          tol=tol, dε=dε, initial_condition=initial_condition)
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SE2_sub_Riemannian_uniform(domain_shape, source_point, ξ, dxy, dθ, θs,
                                                              target_point=target_point, n_max=n_max, n_check=n_check,
                                                              tol=tol, dε=dε, initial_condition=initial_condition)
    elif controller == "plus":
        if ξ is None:
            raise ValueError(f"When using the plus controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SE2_plus_uniform(domain_shape, source_point, ξ, dxy, dθ, θs,
                                                    plus_softness=plus_softness, target_point=target_point, n_max=n_max,
                                                    n_check=n_check, tol=tol, dε=dε,
                                                    initial_condition=initial_condition)
    else:
        raise ValueError(f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus".""")
    return W, grad_W

def geodesic_back_tracking_SE2(grad_W, source_point, target_point, cost, x_min, y_min, θ_min, dxy, dθ, θs, controller="sub-Riemannian",
                               G=None, ξ=None, dt=None, β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `grad_W`: np.ndarray of upwind gradient with respect to some cost of the
          approximate distance map.
        `source_point`: Tuple[int] describing index of source point in `cost`.
        `target_point`: Tuple[int] describing index of target point in `cost`.
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1.
        `xs`: x-coordinate at every point in the grid on which `cost` is
          sampled.
        `ys`: y-coordinate at every point in the grid on which `cost` is
          sampled.
        `θs`: Orientation coordinate at every point in the grid on which `cost`
          is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by 
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by 
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!")
        γ = geodesic_back_tracking_SE2_Riemannian(grad_W, source_point, target_point, cost, x_min, y_min, θ_min, dxy, dθ, θs, G, dt=dt, β=β,
                                                  n_max=n_max)
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!")
        γ = geodesic_back_tracking_SE2_sub_Riemannian(grad_W, source_point, target_point, cost, x_min, y_min, θ_min, dxy, dθ, θs, ξ, dt=dt,
                                                      β=β, n_max=n_max)
    elif controller == "plus":
        if ξ is None:
            raise ValueError(f"When using the plus controller you must pass the the stiffness parameter ξ!")
        γ = geodesic_back_tracking_SE2_plus(grad_W, source_point, target_point, cost, x_min, y_min, θ_min, dxy, dθ, θs, ξ, dt=dt, β=β, 
                                            n_max=n_max)
    else:
        raise ValueError(f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus".""")
    return γ
## SO3
### Riemannian
from eikivp.SO3.Riemannian.distancemap import eikonal_solver as eikonal_solver_SO3_Riemannian
from eikivp.SO3.Riemannian.distancemap import eikonal_solver_uniform as eikonal_solver_SO3_Riemannian_uniform
from eikivp.SO3.Riemannian.backtracking import geodesic_back_tracking as geodesic_back_tracking_SO3_Riemannian
### Sub-Riemannian
from eikivp.SO3.subRiemannian.distancemap import eikonal_solver as eikonal_solver_SO3_sub_Riemannian
from eikivp.SO3.subRiemannian.distancemap import eikonal_solver_uniform as eikonal_solver_SO3_sub_Riemannian_uniform
from eikivp.SO3.subRiemannian.backtracking import geodesic_back_tracking as geodesic_back_tracking_SO3_sub_Riemannian
### Plus controller
from eikivp.SO3.plus.distancemap import eikonal_solver as eikonal_solver_SO3_plus
from eikivp.SO3.plus.distancemap import eikonal_solver_uniform as eikonal_solver_SO3_plus_uniform
from eikivp.SO3.plus.backtracking import geodesic_back_tracking as geodesic_back_tracking_SO3_plus
### Single top level function to select any controller
def eikonal_solver_SO3(cost, source_point, dα, dβ, dφ, αs_np, φs_np, controller="sub-Riemannian", G=None, ξ=None,
                       plus_softness=0., target_point=None, n_max=1e5, n_max_initialisation=1e4, n_check=None,
                       n_check_initialisation=None, tol=1e-3, dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SO(3) equipped with a datadriven left invariant
    norm, with source at `source_point`, using the iterative method described in
    Bekkers et al. "A PDE approach to Data-Driven Sub-Riemannian Geodesics in
    SE(2)" (2015).

    Args:
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1.
        `source_point`: Tuple[int] describing index of source point in 
          `cost`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse B1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward B1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the B1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `cost`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e4.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max`. Defaults to `None`; if no
          `n_check` is passed, convergence is only checked at `n_max`.
        `n_check_initialisation`: Number of iterations between each convergence
          check in the initialisation, taking positive values. Should be at most
          `n_max_initialisation`. Defaults to `None`; if no
          `n_check_initialisation` is passed, convergence is only checked at
          `n_max_initialisation`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by 
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by 
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!")
        W, grad_W = eikonal_solver_SO3_Riemannian(cost, source_point, G, dα, dβ, dφ, αs_np, φs_np,
                                                  target_point=target_point, n_max=n_max,
                                                  n_max_initialisation=n_max_initialisation, n_check=n_check,
                                                  n_check_initialisation=n_check_initialisation, tol=tol, dε=dε,
                                                  initial_condition=initial_condition)
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SO3_sub_Riemannian(cost, source_point, ξ, dα, dβ, dφ, αs_np, φs_np,
                                                      target_point=target_point, n_max=n_max,
                                                      n_max_initialisation=n_max_initialisation, n_check=n_check,
                                                      n_check_initialisation=n_check_initialisation, tol=tol, dε=dε,
                                                      initial_condition=initial_condition)
    elif controller == "plus":
        if ξ is None:
            raise ValueError(f"When using the plus controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SO3_plus(cost, source_point, ξ, dα, dβ, dφ, αs_np, φs_np,
                                            plus_softness=plus_softness, target_point=target_point, n_max=n_max,
                                            n_max_initialisation=n_max_initialisation, n_check=n_check,
                                            n_check_initialisation=n_check_initialisation, tol=tol, dε=dε,
                                            initial_condition=initial_condition)
    else:
        raise ValueError(f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus".""")
    return W, grad_W

def eikonal_solver_SO3_uniform(domain_shape, source_point, dα, dβ, dφ, αs_np, φs_np, controller="sub-Riemannian",
                               G=None, ξ=None, plus_softness=0., target_point=None, n_max=1e5,n_check=None, tol=1e-3,
                               dε=1., initial_condition=100.):
    """
    Solve the Eikonal PDE on SO(3) equipped with a left invariant norm, with
    source at `source_point`, using the iterative method described in Bekkers et
    al. "A PDE approach to Data-Driven Sub-Riemannian Geodesics in SE(2)"
    (2015).

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, with
          respect to standard array indexing.
        `source_point`: Tuple[int] describing index of source point in 
          `domain_shape`.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse B1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the B1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `domain_shape`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain. 
        `n_max`: Maximum number of iterations, taking positive values. Defaults 
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left 
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by 
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by 
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!")
        W, grad_W = eikonal_solver_SO3_Riemannian_uniform(domain_shape, source_point, G, dβ, dφ, αs_np, φs_np,
                                                          target_point=target_point, n_max=n_max, n_check=n_check,
                                                          tol=tol, dε=dε, initial_condition=initial_condition)
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SO3_sub_Riemannian_uniform(domain_shape, source_point, ξ, dβ, dφ, αs_np, φs_np,
                                                              target_point=target_point, n_max=n_max, n_check=n_check,
                                                              tol=tol, dε=dε, initial_condition=initial_condition)
    elif controller == "plus":
        if ξ is None:
            raise ValueError(f"When using the plus controller you must pass the the stiffness parameter ξ!")
        W, grad_W = eikonal_solver_SO3_plus_uniform(domain_shape, source_point, ξ, dβ, dφ, αs_np, φs_np,
                                                    plus_softness=plus_softness, target_point=target_point, n_max=n_max,
                                                    n_check=n_check, tol=tol, dε=dε,
                                                    initial_condition=initial_condition)
    else:
        raise ValueError(f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus".""")
    return W, grad_W

def geodesic_back_tracking_SO3(grad_W, source_point, target_point, cost, α_min, β_min, φ_min, dα, dβ, dφ, αs, φs, controller="sub-Riemannian",
                               G=None, ξ=None, dt=None, β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `grad_W`: np.ndarray of upwind gradient with respect to some cost of the
          approximate distance map.
        `source_point`: Tuple[int] describing index of source point in `cost`.
        `target_point`: Tuple[int] describing index of target point in `cost`.
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1.
        `αs`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `βs`: β-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by 
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by 
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!")
        γ = geodesic_back_tracking_SO3_Riemannian(grad_W, source_point, target_point, cost, α_min, β_min, φ_min, dα, dβ, dφ, αs, φs, G, dt=dt, β=β,
                                                  n_max=n_max)
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!")
        γ = geodesic_back_tracking_SO3_sub_Riemannian(grad_W, source_point, target_point, cost, α_min, β_min, φ_min, dα, dβ, dφ, αs, φs, ξ, dt=dt,
                                                      β=β, n_max=n_max)
    elif controller == "plus":
        if ξ is None:
            raise ValueError(f"When using the plus controller you must pass the the stiffness parameter ξ!")
        γ = geodesic_back_tracking_SO3_plus(grad_W, source_point, target_point, cost, α_min, β_min, φ_min, dα, dβ, dφ, αs, φs, ξ, dt=dt, β=β, 
                                            n_max=n_max)
    else:
        raise ValueError(f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus".""")
    return γ