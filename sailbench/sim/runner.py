"""Main simulation runner for SailBench."""

import math

import numpy as np
import pygame

from sailbench.models.model import State
from sailbench.sim.sailboat_hub import SailboatHub
from sailbench.solvers.rk4 import rk4_step

# --- pygame setup ---
W, H = 1280, 800
SCALE = 80  # pixels per meter
WATER = (20, 90, 160)  # blue water
BOAT_COLOR = (220, 240, 255)
TRACE_COLOR = (180, 255, 240)  # light aqua
HUD_COLOR = (235, 245, 255)

# Boat polygon in body frame (meters): bow at +x, stern at -x
BOAT_POLY = np.array([[1.3, 0.0], [-1.3, 0.5], [-1.0, 0.0], [-1.3, -0.5]])


def world_to_screen(x: float, y: float, cam_x: float, cam_y: float) -> tuple[int, int]:
    """Convert world (meter) coords to screen pixels. Y flipped for screen."""
    sx = (x - cam_x) * SCALE + W / 2
    sy = H / 2 - (y - cam_y) * SCALE
    return int(sx), int(sy)


def rotate_poly(poly: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate polygon by angle (radians) around origin."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return poly @ R.T


def run() -> None:
    """Run loop."""
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SailBench: Boat")
    clock = pygame.time.Clock()

    sailboat = SailboatHub(config_file="basic_sailbot.yaml")
    dt_sec = sailboat.simulation_cfg.get("dt", 0.02)
    fps = int(1.0 / dt_sec) if dt_sec > 0 else 60

    state = State(x=0.0, y=0.0, psi=(1.0, 0.0), u=0.0, v=0.0, r=0.0)
    cam_x, cam_y = 0.0, 0.0

    running = True
    while running:
        dt = clock.tick(fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

        state = sailboat.step(state, dt, solver=rk4_step)

        # --- draw ---
        screen.fill(WATER)

        psi = state.get_heading_rad
        hull_world = rotate_poly(BOAT_POLY, psi) + np.array([[state.x, state.y]])
        hull_pts = [world_to_screen(px, py, cam_x, cam_y) for (px, py) in hull_world]
        pygame.draw.polygon(screen, BOAT_COLOR, hull_pts)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
