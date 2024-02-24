"""
    Riemannian
    ==============

    Solve the Eikonal PDE on SE(2) using a Riemannian controller.

    Provides the following "top level" submodules:
      1. distancemap: compute the distance map (as solution of the Eikonal PDE)
      with respect to a data-driven Riemannian metric. The metric must be
      diagonal with respect to the left invariant frame.
      2. backtracking: compute the geodesic, with respect to the distance map,
      connecting two points. 

    Additionally, we have the following "internal" submodules:
      1. interpolate: interpolate scalar and vector fields between grid points
      with bilinear interpolation.
      2. metric: compute the norm of vectors given some data-driven Riemannian
      metric.
"""

# Access entire backend
import eikivp.SE2.Riemannian.distancemap
import eikivp.SE2.Riemannian.backtracking
import eikivp.SE2.Riemannian.interpolate
import eikivp.SE2.Riemannian.metric