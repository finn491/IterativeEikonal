"""
    SE2
    ===

    Solve the Eikonal PDE on SE(2).

    Contains three submodules for different controller types on SE(2), which
    each contain methods for solving the corresponding Eikonal PDE and computing
    geodesics:
      1. Riemannian.
      2. subRiemannian.
      3. plus.

    Moreover provides the following "top level" submodule:
      1. TODO vesselness: compute the SE(2) vesselness of an image, which can be
      put into a cost function and subsequently into a data-driven metric. 

    Additionally, we have the following "internal" submodules
      1. derivatives: compute various derivatives of functions on SE(2).
      2. utils
"""

# Access entire backend
import eikivp.SE2.derivatives
import eikivp.SE2.vesselness
import eikivp.SE2.utils
import eikivp.SE2.Riemannian
import eikivp.SE2.subRiemannian
import eikivp.SE2.plus