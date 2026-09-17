"""Hull, friction and windage models.

Importing this package is what puts them in the registry, as for `sailbench.foils`.
The three hull models are alternatives for the same slot: each is built from the
`hull` section and returns a surge force, a sway force and a yaw moment, so a
config picks one by name and the rest of the boat is unaffected.
"""

from sailbench.dynamics import friction
from sailbench.dynamics.basic_hull_model import BasicHullModel
from sailbench.dynamics.linear_hydro import LinearHydroModel
from sailbench.dynamics.quadratic_drag_hydro import QuadraticHydroModel
from sailbench.dynamics.windage import Windage

__all__ = [
    "BasicHullModel",
    "LinearHydroModel",
    "QuadraticHydroModel",
    "Windage",
    "friction",
]
