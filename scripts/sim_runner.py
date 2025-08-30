#!/usr/bin/env python3

import argparse, math
from pathlib import Path
import numpy as np
import pygame

from sailbench.utils.loader import load_config
from sailbench.sim.boat import Boat3DOF
from sailbench.dynamics.linear_drag_hydro_3dof import LinearDragModel
from sailbench.foils.keel_3dof import KeelModel3DOF
from sailbench.foils.rudder_3dof import RudderModel3DOF
from sailbench.foils.sail_3dof import SailModel3DOF
from sailbench.solvers.rk4 import rk4_step


def rotate_poly(poly, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return poly @ R.T

def world_to_screen(x, y, cam_x, cam_y, scale, W, H):
    sx = (x - cam_x) * scale + W / 2
    sy = H / 2 - (y - cam_y) * scale
    return int(sx), int(sy)

def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--scale", type=float, default=5.0, help="pixels per meter")
    args = ap.parse_args()

    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    # load config & build boat
    cfg = load_config(args.config)
    boat = Boat3DOF.from_config(
        cfg,
        hull_cls=LinearDragModel,
        keel_cls=KeelModel3DOF,
        rudder_cls=RudderModel3DOF,
        sail_cls=SailModel3DOF,
        use_coriolis=True,
    )

    # angle limits
    delta_r_max = math.radians(cfg["rudder"]["delta_max_deg"])
    sail_cfg = cfg.get("sail", {})
    delta_s_max = math.radians(sail_cfg.get("delta_max_deg", min(85.0, 1.2 * sail_cfg.get("stall_deg", 30.0))))

    # pygame setup
    W, H = 1100, 750
    SCALE = args.scale
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("sailbench")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # simple boat polygon (meters, body frame)
    BOAT_POLY = np.array([
        [ 1.3,  0.0],   # bow
        [-1.3,  0.5],
        [-1.0,  0.0],
        [-1.3, -0.5]
    ])

    # controls
    delta_r = 0.0     # rudder (rad)
    delta_s = 0.0     # sail   (rad)
    rate_r = math.radians(45.0)  # deg/s -> rad/s (rudder rate)
    rate_s = math.radians(45.0)  # deg/s -> rad/s (sail rate)

    running = True
    while running:
        dt = clock.tick(args.fps) / 1000.0

        # input
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            running = False
        if keys[pygame.K_r]:
            boat.reset()
            delta_r = 0.0
            delta_s = 0.0

        # rudder (←/→)
        if keys[pygame.K_LEFT]:
            delta_r += rate_r * dt
        if keys[pygame.K_RIGHT]:
            delta_r -= rate_r * dt
        delta_r = clip(delta_r, -delta_r_max, +delta_r_max)

        # sail (↑/↓) – 0 = in; positive = eased out
        if keys[pygame.K_UP]:
            delta_s -= rate_s * dt
        if keys[pygame.K_DOWN]:
            delta_s += rate_s * dt
        delta_s = clip(delta_s, 0.0, delta_s_max)

        # step sim
        boat.set_controls(delta_rudder=delta_r, delta_sail=delta_s)
        state = boat.step(dt, rk4_step)

        # draw
        screen.fill((28, 28, 32))
        u, v, r, x, y, psi = state
        cam_x, cam_y = x, y

        hull_world = rotate_poly(BOAT_POLY, psi) + np.array([[x, y]])
        pts = [world_to_screen(px, py, cam_x, cam_y, SCALE, W, H) for px, py in hull_world]
        pygame.draw.polygon(screen, (180, 220, 255), pts)

        # rudder line at stern for visualization
        rud_len = 0.8
        rud_local = np.array([
            [-1.3, 0.0],
            [-1.3 - rud_len * math.cos(delta_r), rud_len * math.sin(delta_r)]
        ])
        rud_world = rotate_poly(rud_local, psi) + np.array([[x, y]])
        p0 = world_to_screen(*rud_world[0], cam_x, cam_y, SCALE, W, H)
        p1 = world_to_screen(*rud_world[1], cam_x, cam_y, SCALE, W, H)
        pygame.draw.line(screen, (255, 120, 120), p0, p1, 3)

        # simple sail arc indicator (trim amount)
        arc_r = 40
        cx, cy = world_to_screen(x, y, cam_x, cam_y, SCALE, W, H)
        pygame.draw.arc(
            screen, (120, 200, 120),
            (cx - arc_r, cy - arc_r, 2 * arc_r, 2 * arc_r),
            -math.pi/2, -math.pi/2 + delta_s, 2
        )

        # HUD
        hud = [
            f"FPS: {clock.get_fps():5.1f}",
            f"u,v [m/s]: {u:6.2f}, {v:6.2f}",
            f"r [deg/s]: {math.degrees(r):6.2f}",
            f"ψ [deg]  : {math.degrees(psi)%360:6.2f}",
            f"δ_r [deg]: {math.degrees(delta_r):6.2f}",
            f"δ_s [deg]: {math.degrees(delta_s):6.2f}",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (220, 220, 220)), (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
