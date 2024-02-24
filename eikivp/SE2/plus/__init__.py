"""
    plus
    ====

    Solve the Eikonal PDE on SE(2) using the plus controller.

    Provides the following "top level" submodules:
      1. distancemap: compute the distance map (as solution of the Eikonal PDE)
      with respect to some data-driven Finsler function.
      2. backtracking: compute the geodesic, with respect to the distance map,
      connecting two points.

    Additionally, we have the following "internal" submodules:
      1. interpolate: interpolate scalar and vector fields between grid points
      with trilinear interpolation.
      2. metric: compute the norm of vectors given some data-driven Finsler
      function.
"""

# Access entire backend
import eikivp.SE2.plus.distancemap
import eikivp.SE2.plus.backtracking
import eikivp.SE2.plus.interpolate
import eikivp.SE2.plus.metric