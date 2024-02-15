"""
    EikIVP
    ======

    The Python package *eikivp* contains methods to solve the Eikonal PDE on R^2
    and SE(2) using the iterative Initial Value Problem (IVP) technique
    described in Bekkers et al. "A PDE approach to Data-Driven Sub-Riemannian 
    Geodesics in SE(2)" (2015), and to find geodesics connecting points with
    respect to the distance map that solves the Eikonal PDE.

    One application in which we want to solve the Eikonal PDE and subsequently
    find geodesics connecting pairs of points is vascular tracking. This package
    contains methods to construct data-driven metrics on R^2 and SE(2), based
    on multiscale vesselness filters, that will lead to geodesics that 
    (hopefully) track vessels.

    Summary: compute distance map and geodesics with respect to data-driven 
    metric on R^2 and SE(2).
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
from eikivp.SE2.distancemap import eikonal_solver as eikonal_solver_SE2
from eikivp.SE2.distancemap import eikonal_solver_uniform as eikonal_solver_SE2_uniform
from eikivp.SE2.distancemap import (
    eikonal_solver_sub_Riemannian,
    eikonal_solver_sub_Riemannian_uniform,
    eikonal_solver_plus,
    eikonal_solver_plus_uniform
)
from eikivp.SE2.backtracking import geodesic_back_tracking as geodesic_back_tracking_SE2