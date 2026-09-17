from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transform2D:
    """Rigid transform in 2D."""

    x: float
    y: float
    c: float  # cos(theta)
    s: float  # sin(theta)

    def rotation_matrix(self) -> np.ndarray:
        """Return 2x2 rotation matrix."""
        return np.array([[self.c, -self.s], [self.s, self.c]], dtype=float)

    @staticmethod
    def identity() -> Transform2D:
        """Return identity transform."""
        return Transform2D(0.0, 0.0, 1.0, 0.0)


class TFTree2D:
    """Simple 2D transform tree."""

    def __init__(self) -> None:
        self.transforms: dict[str, Transform2D] = {}
        self.parents: dict[str, str | None] = {}
        self.root: str = "world"

    def add_root(self, name: str) -> None:
        """Add a root frame."""
        self.root = name

    def add_frame(
        self,
        name: str,
        parent: str | None,
        transform: Transform2D,
    ) -> None:
        """Add or update a frame. Transform is from parent -> child."""
        self.transforms[name] = transform
        self.parents[name] = parent

    def vector_to_frame(
        self,
        vec: np.ndarray,
        from_frame: str,
        to_frame: str,
    ) -> np.ndarray:
        """Rotate vector between frames (no translation)."""
        tf_from = self.get_to_root(from_frame)
        tf_to = self.get_to_root(to_frame)

        r_from = tf_from.rotation_matrix()
        r_to = tf_to.rotation_matrix()

        world_vec = r_from @ vec
        return np.asarray(r_to.T @ world_vec, dtype=float)

    def get_to_root(self, name: str) -> Transform2D:
        """Get transform from frame to root."""
        if name == self.root:
            return Transform2D.identity()

        tf = self.transforms[name]
        parent = self.parents[name]

        while parent is not None:
            if parent == self.root:
                break

            parent_tf = self.transforms[parent]
            tf = self._compose(parent_tf, tf)
            parent = self.parents[parent]

        return tf

    def _compose(self, a: Transform2D, b: Transform2D) -> Transform2D:
        """Return transform a ∘ b."""
        r = a.rotation_matrix()
        t = r @ np.array([b.x, b.y]) + np.array([a.x, a.y])

        c = a.c * b.c - a.s * b.s
        s = a.s * b.c + a.c * b.s

        return Transform2D(t[0], t[1], c, s)
