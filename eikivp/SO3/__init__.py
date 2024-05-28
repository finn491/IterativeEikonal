"""
    SO3
    ===

    Solve the Eikonal PDE on SO(3).

    Contains three submodules for different controller types on SO(3), which
    each contain methods for solving the corresponding Eikonal PDE and computing
    geodesics:
      1. Riemannian.
      2. subRiemannian.
      3. plus.

    Additionally, we have the following "internal" submodules:
      1. derivatives: compute various derivatives of functions on SO(3).
      2. utils
"""

# Access entire backend
import eikivp.SO3.derivatives
import eikivp.SO3.costfunction
import eikivp.SO3.utils
import eikivp.SO3.Riemannian
import eikivp.SO3.subRiemannian
import eikivp.SO3.plus