import numpy as np
import pytest

from sailbench.tf.tf_tree import TFTree2D, Transform2D


def test_compose_translation_only() -> None:
    tree = TFTree2D()

    a = Transform2D(1.0, 2.0, 1.0, 0.0)
    b = Transform2D(3.0, 4.0, 1.0, 0.0)

    result = tree._compose(a, b)

    assert result.x == 4.0
    assert result.y == 6.0
    assert result.c == 1.0
    assert result.s == 0.0


def test_compose_rotation_90_twice() -> None:
    tree = TFTree2D()

    # 90° rotation
    a = Transform2D(0.0, 0.0, 0.0, 1.0)
    b = Transform2D(0.0, 0.0, 0.0, 1.0)

    result = tree._compose(a, b)

    # 90° + 90° = 180°
    assert pytest.approx(result.c) == -1.0
    assert pytest.approx(result.s) == 0.0


# -------------------------
# get_to_root Tests
# -------------------------


def test_get_to_root_single_level() -> None:
    tree = TFTree2D()
    tree.add_root("world")

    tf = Transform2D(1.0, 0.0, 1.0, 0.0)
    tree.add_frame("child", "world", tf)

    result = tree.get_to_root("child")

    assert result.x == 1.0
    assert result.y == 0.0
    assert result.c == 1.0
    assert result.s == 0.0


@pytest.mark.parametrize(
    "theta_a, theta_b",
    [
        (0.0, 0.0),
        (np.pi / 2, 0.0),
        (0.0, np.pi / 2),
        (np.pi / 2, np.pi / 2),
        (np.pi, np.pi / 2),
        (-np.pi / 2, np.pi / 2),
        (np.pi / 4, np.pi / 6),
    ],
)
def test_get_to_root_multi_level_rotation(theta_a: float, theta_b: float) -> None:
    tree = TFTree2D()
    tree.add_root("world")

    # world -> A
    tree.add_frame(
        "A",
        "world",
        Transform2D(0.0, 0.0, np.cos(theta_a), np.sin(theta_a)),
    )

    # A -> B
    tree.add_frame(
        "B",
        "A",
        Transform2D(0.0, 0.0, np.cos(theta_b), np.sin(theta_b)),
    )

    result = tree.get_to_root("B")

    # Expected rotation = theta_a + theta_b
    expected_theta = theta_a + theta_b

    assert pytest.approx(result.c, abs=1e-10) == np.cos(expected_theta)
    assert pytest.approx(result.s, abs=1e-10) == np.sin(expected_theta)


def test_get_to_root_is_identity_for_root() -> None:
    tree = TFTree2D()
    tree.add_root("map")

    result = tree.get_to_root("map")

    np.testing.assert_allclose(result.rotation_matrix(), np.eye(2))
    assert result.x == 0.0
    assert result.y == 0.0


# -------------------------
# vector_to_frame Tests
# -------------------------


def test_vector_to_frame_no_rotation() -> None:
    tree = TFTree2D()
    tree.add_root("world")

    tree.add_frame("A", "world", Transform2D(0.0, 0.0, 1.0, 0.0))

    vec = np.array([1.0, 0.0])
    result = tree.vector_to_frame(vec, "A", "world")

    np.testing.assert_allclose(result, vec)


def test_vector_to_frame_90_deg_rotation() -> None:
    tree = TFTree2D()
    tree.add_root("world")

    # A rotated 90° relative to world
    tree.add_frame("A", "world", Transform2D(0.0, 0.0, 0.0, 1.0))

    vec_A = np.array([1.0, 0.0])  # x-axis of A

    result = tree.vector_to_frame(vec_A, "A", "world")

    # Should align with +y in world
    expected = np.array([0.0, 1.0])
    np.testing.assert_allclose(result, expected)


def test_vector_between_two_rotated_frames() -> None:
    tree = TFTree2D()
    tree.add_root("world")

    # A: +90°
    tree.add_frame("A", "world", Transform2D(0.0, 0.0, 0.0, 1.0))

    # B: -90°
    tree.add_frame("B", "world", Transform2D(0.0, 0.0, 0.0, -1.0))

    vec_A = np.array([1.0, 0.0])

    result = tree.vector_to_frame(vec_A, "A", "B")

    # 90° to world gives (0,1)
    # world to B (transpose of -90°) gives (1,0)
    expected = np.array([-1.0, 0.0])

    np.testing.assert_allclose(result, expected)
