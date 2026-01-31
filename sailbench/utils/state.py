from dataclasses import dataclass

@dataclass
class State:
    """State class."""

    x:float
    y:float
    # angle is c + i*s
    c:float #cosine for angle
    s: float #i sine for angle
    u: float
    v: float
    r: float