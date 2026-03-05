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

# Display lengths for foils (meters)
KEEL_DISPLAY_LEN = 0.5
RUDDER_DISPLAY_LEN = 0.35
SAIL_DISPLAY_LEN = 1.0
RUDDER_MAX_DEG = 35.0
RUDDER_RATE_DEG_PER_S = 80.0
SAIL_MAX_RAD = math.radians(90.0)   # ±90° sail angle
SAIL_RATE_RAD_PER_S = math.radians(90.0)  # rad/s

# Real-time plot scales (world meters per unit)
FORCE_SCALE = 0.015  # m per N (e.g. 100 N -> 1.5 m arrow)
WIND_ARROW_SCALE = 0.3   # m arrow length per m/s wind speed
ARROW_HEAD_LEN = 0.15    # meters
FORCE_COLOR = (255, 100, 100)
WIND_COLOR = (255, 255, 200)


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


def draw_arrow(
    screen: pygame.Surface,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    head_len: float,
    cam_x: float,
    cam_y: float,
) -> None:
    """Draw a line with arrowhead; start/end in world meters."""
    p0 = world_to_screen(start[0], start[1], cam_x, cam_y)
    p1 = world_to_screen(end[0], end[1], cam_x, cam_y)
    pygame.draw.line(screen, color, p0, p1, 2)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length > 1e-6:
        ux, uy = dx / length, dy / length
        # Arrowhead perpendicular
        ax = end[0] - ux * head_len + 0.4 * head_len * (-uy)
        ay = end[1] - uy * head_len + 0.4 * head_len * ux
        bx = end[0] - ux * head_len - 0.4 * head_len * (-uy)
        by = end[1] - uy * head_len - 0.4 * head_len * ux
        pygame.draw.polygon(
            screen,
            color,
            [
                world_to_screen(end[0], end[1], cam_x, cam_y),
                world_to_screen(ax, ay, cam_x, cam_y),
                world_to_screen(bx, by, cam_x, cam_y),
            ],
        )


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
    time_accum = 0.0
    rudder_angle_deg = 0.0
    sail_angle_rad = 0.0

    # Colors for foils
    KEEL_COLOR = (80, 80, 120)
    RUDDER_COLOR = (60, 60, 100)
    SAIL_COLOR = (240, 240, 255)

    running = True
    while running:
        elapsed = clock.tick(fps) / 1000.0
        time_accum += elapsed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

        # Arrow keys: rudder (left/right), sail (up/down)
        rudder_delta = RUDDER_RATE_DEG_PER_S * elapsed
        if pygame.key.get_pressed()[pygame.K_LEFT]:
            rudder_angle_deg = max(-RUDDER_MAX_DEG, rudder_angle_deg - rudder_delta)
        if pygame.key.get_pressed()[pygame.K_RIGHT]:
            rudder_angle_deg = min(RUDDER_MAX_DEG, rudder_angle_deg + rudder_delta)
        sail_delta = SAIL_RATE_RAD_PER_S * elapsed
        if pygame.key.get_pressed()[pygame.K_UP]:
            sail_angle_rad = min(SAIL_MAX_RAD, sail_angle_rad + sail_delta)
        if pygame.key.get_pressed()[pygame.K_DOWN]:
            sail_angle_rad = max(-SAIL_MAX_RAD, sail_angle_rad - sail_delta)

        wind_speed = sailboat.sail_cfg.get("wind_speed", 0.0)
        wind_dir_deg = sailboat.sail_cfg.get("wind_dir_deg", 0.0)

        # Fixed physics step: sub-step with config dt to avoid blow-up from variable dt
        while time_accum >= dt_sec:
            state = sailboat.step(
                state,
                dt_sec,
                solver=rk4_step,
                sail_angle=sail_angle_rad,
                rudder_angle=rudder_angle_deg,
            )
            time_accum -= dt_sec

        # --- draw ---
        screen.fill(WATER)

        psi = state.get_heading_rad
        cx, cy = state.x, state.y
        c, s = math.cos(psi), math.sin(psi)
        R = np.array([[c, -s], [s, c]])

        # Hull
        hull_world = rotate_poly(BOAT_POLY, psi) + np.array([[cx, cy]])
        hull_pts = [world_to_screen(px, py, cam_x, cam_y) for (px, py) in hull_world]
        pygame.draw.polygon(screen, BOAT_COLOR, hull_pts)

        # Keel: draw on boat centerline amidships (visual only)
        half = KEEL_DISPLAY_LEN / 2
        k0_body = np.array([-half, 0.0])
        k1_body = np.array([half, 0.0])
        k0_world = R @ k0_body + np.array([cx, cy])
        k1_world = R @ k1_body + np.array([cx, cy])
        pygame.draw.line(
            screen,
            KEEL_COLOR,
            world_to_screen(k0_world[0], k0_world[1], cam_x, cam_y),
            world_to_screen(k1_world[0], k1_world[1], cam_x, cam_y),
            3,
        )

        # Rudder: draw at stern 
        rudder_rad = -math.radians(rudder_angle_deg)
        r0_body = np.array([-1.0, 0.0])  # base at stern
        r1_body = r0_body - np.array([RUDDER_DISPLAY_LEN * math.cos(rudder_rad), RUDDER_DISPLAY_LEN * math.sin(rudder_rad)])
        r0_world = R @ r0_body + np.array([cx, cy])
        r1_world = R @ r1_body + np.array([cx, cy])
        pygame.draw.line(
            screen,
            RUDDER_COLOR,
            world_to_screen(r0_world[0], r0_world[1], cam_x, cam_y),
            world_to_screen(r1_world[0], r1_world[1], cam_x, cam_y),
            3,
        )

        # Sail
        sail_rad = -sail_angle_rad
        s0_body = np.array([0.0, 0.0])  # mast base
        s1_body = s0_body - np.array([SAIL_DISPLAY_LEN * math.cos(sail_rad), SAIL_DISPLAY_LEN * math.sin(sail_rad)])
        s0_world = R @ s0_body + np.array([cx, cy])
        s1_world = R @ s1_body + np.array([cx, cy])
        pygame.draw.line(
            screen,
            SAIL_COLOR,
            world_to_screen(s0_world[0], s0_world[1], cam_x, cam_y),
            world_to_screen(s1_world[0], s1_world[1], cam_x, cam_y),
            3,
        )

        # --- Forces and wind (real-time) ---
        fx, fy, mz = sailboat._forces(state)
        # Force arrow: from boat, in world frame (body force rotated by psi)
        fx_world = c * fx - s * fy
        fy_world = s * fx + c * fy
        force_len = math.hypot(fx, fy) * FORCE_SCALE
        if force_len > 0.05:
            end_fx = cx + fx_world * FORCE_SCALE
            end_fy = cy + fy_world * FORCE_SCALE
            draw_arrow(
                screen,
                (cx, cy),
                (end_fx, end_fy),
                FORCE_COLOR,
                ARROW_HEAD_LEN,
                cam_x,
                cam_y,
            )
        # Wind arrow: from boat, direction wind is blowing TO (config convention)
        if wind_speed > 0.1:
            wind_rad = math.radians(wind_dir_deg)
            wdx = wind_speed * WIND_ARROW_SCALE * math.cos(wind_rad)
            wdy = wind_speed * WIND_ARROW_SCALE * math.sin(wind_rad)
            draw_arrow(
                screen,
                (cx, cy),
                (cx + wdx, cy + wdy),
                WIND_COLOR,
                ARROW_HEAD_LEN,
                cam_x,
                cam_y,
            )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()

