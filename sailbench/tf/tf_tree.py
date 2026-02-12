"""Simple 2D TF tree implementation."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transform2D:
    """Rigid transform in 2D."""

    x: float
    y: float
    theta: float  # radians

    def rotation_matrix(self) -> np.ndarray:
        """Return 2x2 rotation matrix."""
        c = float(np.cos(self.theta))
        s = float(np.sin(self.theta))
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def inverse(self) -> "Transform2D":
        """Return inverse transform."""
        r = self.rotation_matrix().T
        t = -r @ np.array([self.x, self.y], dtype=np.float64)
        theta = -self.theta
        return Transform2D(float(t[0]), float(t[1]), theta)

    def __matmul__(self, other: "Transform2D") -> "Transform2D":
        """Compose transforms."""
        r = self.rotation_matrix()
        t = r @ np.array([other.x, other.y], dtype=np.float64)
        return Transform2D(
            x=float(self.x + t[0]),
            y=float(self.y + t[1]),
            theta=float(self.theta + other.theta),
        )


class TFTree2D:
    """Simple parent-child TF tree."""

    def __init__(self) -> None:
        self._edges: Dict[str, Tuple[str, Transform2D]] = {}

    def set_transform(
        self,
        parent: str,
        child: str,
        transform: Transform2D,
    ) -> None:
        """Register transform parent→child."""
        self._edges[child] = (parent, transform)

    def _to_root(self, frame: str) -> Transform2D:
        t = Transform2D(0.0, 0.0, 0.0)
        current = frame

        while current in self._edges:
            parent, edge = self._edges[current]
            t = edge @ t
            current = parent

        return t

    def lookup(self, from_frame: str, to_frame: str) -> Transform2D:
        """Get transform from_frame → to_frame."""
        t_from = self._to_root(from_frame)
        t_to = self._to_root(to_frame)
        return t_to.inverse() @ t_from

    def rotate_vector(
        self,
        vec: np.ndarray,
        from_frame: str,
        to_frame: str,
    ) -> np.ndarray:
        """Rotate vector between frames (no translation)."""
        t = self.lookup(from_frame, to_frame)
        return t.rotation_matrix() @ vec
