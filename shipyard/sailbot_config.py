from dataclasses import dataclass
from typing import List


@dataclass
class KeelConfig:
    airfoil_name: str
    model_type: str
    alpha_min: float
    alpha_max: float
    res: List[float]
    area: float

@dataclass
class SailConfig:
    airfoil_name: str
    model_type: str
    alpha_min: int
    alpha_max: int
    res: List[float]
    area: float
    wind_speed: float
    wind_dir_deg: float

@dataclass
class RudderConfig:
    airfoil_name: NACA0012
    model_type: keel
    alpha_min: int
    alpha_max: 25
    res: [1e5]
    area: float