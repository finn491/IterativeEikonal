# Access entire backend
import eikivp.utils
import eikivp.visualisations
import eikivp.costfunction
import eikivp.R2
import eikivp.SE2

# Most important functions are available at top level
## R2
from eikivp.R2.distancemap import eikonal_solver as eikonal_solver_R2
from eikivp.R2.backtracking import geodesic_back_tracking as geodesic_back_tracking_R2
## SE2
from eikivp.SE2.distancemap import eikonal_solver as eikonal_solver_SE2
from eikivp.SE2.distancemap import eikonal_solver_plus, eikonal_solver_sub_Riemannian
from eikivp.SE2.backtracking import geodesic_back_tracking as geodesic_back_tracking_SE2