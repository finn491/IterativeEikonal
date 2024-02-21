"""
    Sub-Riemannian
    ==============

    Solve the Eikonal PDE on SE(2) using a sub-Riemannian controller.

    Provides the following "top level" submodules:
      1. distancemap: compute the distance map (as solution of the Eikonal PDE)
      with respect to some data-driven metric. This may be Riemannian, 
      sub-Riemannian, or plus controller, so long as the metric is diagonal with
      respect to the left invariant frame.
      2. backtracking: compute the geodesic, with respect to the distance map,
      connecting two points.
      3. TODO vesselness: compute the SE(2) vesselness of an image, which can be
      put into a cost function and subsequently into a data-driven metric. 

    Additionally, we have the following "internal" submodules
      1. derivatives: compute various derivatives of functions on SE(2).
      2. interpolate: interpolate scalar and vector fields between grid points
      with bilinear interpolation.
      3. metric: compute the norm of vectors given some data-driven metric.
"""

# Access entire backend
import eikivp.SE2.subRiemannian.distancemap
import eikivp.SE2.subRiemannian.backtracking
import eikivp.SE2.subRiemannian.interpolate
import eikivp.SE2.subRiemannian.metric