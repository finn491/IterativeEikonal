"""
    R2
    ==

    Solve the Eikonal PDE on R^2.

    Provides the following "top level" submodules:
      1. distancemap: compute the distance map (as solution of the Eikonal PDE)
      with respect to some data-driven metric. The metric must be diagonal with
      respect to the standard Euclidean basis.
      2. backtracking: compute the geodesic, with respect to the distance map,
      connecting two points.
      3. vesselness: compute the 2D Frangi vesselness of an image, which can be
      put into a cost function and subsequently into a data-driven metric.

    Additionally, we have the following "internal" submodules:
      1. derivatives: compute various derivatives of functions on R^2.
      2. interpolate: interpolate scalar and vector fields between grid points
      with bilinear interpolation.
      3. metric: compute the norm of vectors given some data-driven metric.
      4. utils: index safely into arrays and align the data in arrays with real
      axes or standard array indexing conventions.
"""

# Access entire backend
import eikivp.R2.distancemap
import eikivp.R2.backtracking
import eikivp.R2.derivatives
import eikivp.R2.interpolate
import eikivp.R2.vesselness
import eikivp.R2.metric