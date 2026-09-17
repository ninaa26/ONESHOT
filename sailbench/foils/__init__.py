"""Sail, keel and rudder models.

Importing this package is what puts them in the registry: each class registers
itself when its module is loaded, so a model in a file nobody imports is a model
the shipyard cannot offer and a config cannot name.
"""

from sailbench.foils.basic_keel import BasicKeel, FiniteSpanKeel
from sailbench.foils.basic_rudder import BasicRudder, FiniteSpanRudder
from sailbench.foils.basic_sail import BasicSail
from sailbench.foils.orc_sail import ORCMainSail, ORCWithJibSail

__all__ = [
    "BasicKeel",
    "BasicRudder",
    "BasicSail",
    "FiniteSpanKeel",
    "FiniteSpanRudder",
    "ORCMainSail",
    "ORCWithJibSail",
]
