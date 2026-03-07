import yaml

import numpy as np
import matplotlib.pyplot as plt

from sailbench.dynamics.quadratic_drag_hydro import QuadraticHydroModel
from sailbench.foils.basic_keel import BasicKeel
from sailbench.foils.basic_rudder import BasicRudder
from sailbench.foils.hybrid_sail import HybridSail
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D, Transform2D

# ----------------------------
# Load config and build components
# ----------------------------
with open("configs/basic_sailbot.yaml", "r") as f:
    config = yaml.safe_load(f)

params_sail = config["sail"].copy()
params_keel = config["keel"].copy()
params_rudder = config["rudder"].copy()
params_hull = config["hull"].copy()

sail = HybridSail(params_sail)
keel = BasicKeel(params_keel)
rudder = BasicRudder(params_rudder)
hull = QuadraticHydroModel(params_hull)

# ----------------------------
# Build transform tree (all component frames in boat frame)
# ----------------------------
tf_tree = TFTree2D()
tf_tree.add_root("world")
tf_tree.add_frame("boat", "world", Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
tf_tree.add_frame("fluid", "boat", Transform2D(x=0.0, y=0.0, c=1.0, s=0.0))
tf_tree.add_frame(
    "keel",
    "boat",
    Transform2D(
        x=params_keel.get("x_pos", 0.0),
        y=params_keel.get("y_pos", 0.0),
        c=1.0,
        s=0.0,
    ),
)
tf_tree.add_frame(
    "rudder",
    "boat",
    Transform2D(
        x=params_rudder.get("x_pos", 0.0),
        y=params_rudder.get("y_pos", 0.0),
        c=1.0,
        s=0.0,
    ),
)
tf_tree.add_frame(
    "sail",
    "boat",
    Transform2D(
        x=params_sail.get("x_pos", 0.0),
        y=params_sail.get("y_pos", 0.0),
        c=1.0,
        s=0.0,
    ),
)

# ----------------------------
# Plot: force on each component in boat frame (one scenario)
# ----------------------------
# Scenario: boat moving, wind from 45°, rudder 0°, sail aligned with boat
u, v, r = 2.0, 0.3, 0.0
wind_deg = 45.0
rudder_deg = 0.0
sail_angle_rad = 0.0

state = State(u=u, v=v, x=0.0, y=0.0, psi=(1.0, 0.0), r=r)
sail.p["wind_dir_deg"] = wind_deg

# Update fluid frame from (u, v)
norm = np.hypot(u, v)
if norm > 1e-6:
    c_fluid, s_fluid = -u / norm, -v / norm
else:
    c_fluid, s_fluid = 1.0, 0.0
tf_tree.add_frame("fluid", "boat", Transform2D(x=0.0, y=0.0, c=c_fluid, s=s_fluid))
tf_tree.add_frame(
    "rudder",
    "boat",
    Transform2D(
        x=params_rudder.get("x_pos", 0.0),
        y=params_rudder.get("y_pos", 0.0),
        c=np.cos(np.radians(rudder_deg)),
        s=np.sin(np.radians(rudder_deg)),
    ),
)
tf_tree.add_frame(
    "sail",
    "boat",
    Transform2D(
        x=params_sail.get("x_pos", 0.0),
        y=params_sail.get("y_pos", 0.0),
        c=np.cos(sail_angle_rad),
        s=np.sin(sail_angle_rad),
    ),
)

components = [
    ("hull", hull),
    ("keel", keel),
    ("rudder", rudder),
    ("sail", sail),
]
names = [c[0] for c in components]
fx_per = []
fy_per = []
for _name, comp in components:
    out = np.atleast_1d(comp.compute(state, tf_tree))
    fx_per.append(float(out[0]))
    fy_per.append(float(out[1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(names))
w = 0.35
ax1.bar(x - w / 2, fx_per, width=w, color="C0")
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax1.set_ylabel("Fx (N)")
ax1.set_title("Force per component (boat frame)")
ax1.grid(True, axis="y")
ax2.bar(x - w / 2, fy_per, width=w, color="C1")
ax2.set_xticks(x)
ax2.set_xticklabels(names)
ax2.set_ylabel("Fy (N)")
ax2.set_title("Force per component (boat frame)")
ax2.grid(True, axis="y")
plt.tight_layout()
plt.show()

# ----------------------------
# Sail: wind angle vs sail force (boat frame)
# ----------------------------

# ----------------------------
# Sail: wind angle vs sail force (boat frame)
# ----------------------------
wind_angles_deg = np.linspace(0, 360, 200, endpoint=False)  # degrees
sail_fx_list = []
sail_fy_list = []
# Boat stationary for wind sweep
state_stationary = State(u=0.0, v=0.0, x=0.0, y=0.0, psi=(1.0, 0.0), r=0.0)

for wind_ang_deg in wind_angles_deg:
    sail.p["wind_dir_deg"] = wind_ang_deg
    f_boat = sail.compute(state_stationary, tf_tree)
    sail_fx_list.append(f_boat[0])
    sail_fy_list.append(f_boat[1])

sail_fx_list = np.array(sail_fx_list)
sail_fy_list = np.array(sail_fy_list)

# Plot sail force (boat frame) vs wind angle
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(wind_angles_deg, sail_fx_list)
ax1.set_ylabel("Sail force Fx (N)")
ax1.set_title("Sail force (boat frame) vs wind angle")
ax1.grid(True)
ax2.plot(wind_angles_deg, sail_fy_list)
ax2.set_xlabel("Wind angle (deg)")
ax2.set_ylabel("Sail force Fy (N)")
ax2.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Keel: flow angle vs keel force (boat frame)
# ----------------------------
angles_uv = np.linspace(0, 360, 200, endpoint=False)  # degrees
radians_uv = np.radians(angles_uv)
speed = 1.0  # set a unit speed for visualization

fx_list = []
fy_list = []

for ang in radians_uv:
    # (u, v) are body-frame; flow direction is opposite to velocity
    # Update the "fluid" frame to align with the negative velocity direction (opposite of (u,v))
    u = speed * np.cos(ang)
    v = speed * np.sin(ang)
    norm = np.hypot(u, v)
    if norm > 1e-6:
        c, s = -u / norm, -v / norm
    else:
        c, s = 1.0, 0.0
    # Overwrite/add the "fluid" frame for each (u, v)
    tf_tree.add_frame(
        name="fluid",
        parent="boat",
        transform=Transform2D(x=0.0, y=0.0, c=c, s=s),
    )

    u = speed * np.cos(ang)
    v = speed * np.sin(ang)
    # Update the state with current u, v
    test_state = State(
        x=0.0,
        y=0.0,
        psi=(1.0, 0.0),
        u=u,
        v=v,
        r=0.0
    )
    # Transform tree can be reset/built for each, but in this simplified example, it's fine as is
    f_boat = keel.compute(test_state, tf_tree)
    fx_list.append(f_boat[0])
    fy_list.append(f_boat[1])

fx_list = np.array(fx_list)
fy_list = np.array(fy_list)

# ----------------------------
# Plot Fx
# ----------------------------
plt.figure()
plt.plot(angles_uv, fx_list)
plt.xlabel("Flow / track angle (deg)")
plt.ylabel("Force X (N)")
plt.title("Keel force (boat frame) vs flow angle")
plt.grid(True)
plt.show()

# ----------------------------
# Plot Fy
# ----------------------------
plt.figure()
plt.plot(angles_uv, fy_list)
plt.xlabel("Flow / track angle (deg)")
plt.ylabel("Force Y (N)")
plt.title("Keel force (boat frame) vs flow angle")
plt.grid(True)
plt.show()

# ----------------------------
# Hull: quadratic drag vs surge / sway speed
# ----------------------------
speeds = np.linspace(-3.0, 3.0, 200)
hull_fx = []
hull_fy = []

for u_val in speeds:
    state_surge = State(
        u=u_val,
        v=0.0,
        r=0.0,
        x=0.0,
        y=0.0,
        psi=(1.0, 0.0),
    )
    f = hull.compute(state_surge, tf_tree)
    hull_fx.append(f[0])

for v_val in speeds:
    state_sway = State(
        u=0.0,
        v=v_val,
        r=0.0,
        x=0.0,
        y=0.0,
        psi=(1.0, 0.0),
    )
    f = hull.compute(state_sway, tf_tree)
    hull_fy.append(f[1])

hull_fx = np.array(hull_fx)
hull_fy = np.array(hull_fy)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(speeds, hull_fx)
ax1.axhline(0.0, color="k", linewidth=0.5)
ax1.set_ylabel("Fx (N)")
ax1.set_title("Hull quadratic drag vs surge / sway speed")
ax1.grid(True)

ax2.plot(speeds, hull_fy, color="C1")
ax2.axhline(0.0, color="k", linewidth=0.5)
ax2.set_xlabel("Speed (m/s)")
ax2.set_ylabel("Fy (N)")
ax2.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------
# Hull: drag / side force vs track angle
# ----------------------------
track_angles_deg = np.linspace(0.0, 360.0, 200, endpoint=False)
track_angles_rad = np.radians(track_angles_deg)
speed_track = 1.0  # [m/s] magnitude of boat speed

drag_along_track = []
side_force = []
hull_fx_track = []
hull_fy_track = []

for ang in track_angles_rad:
    u = speed_track * np.cos(ang)
    v = speed_track * np.sin(ang)
    state_track = State(
        u=u,
        v=v,
        r=0.0,
        x=0.0,
        y=0.0,
        psi=(1.0, 0.0),  # boat x-axis = 0 deg track
    )
    fx, fy, _ = hull.compute(state_track, tf_tree)
    hull_fx_track.append(fx)
    hull_fy_track.append(fy)

    # Unit vectors along and normal to track direction
    tx, ty = np.cos(ang), np.sin(ang)
    nx, ny = -np.sin(ang), np.cos(ang)

    # Force components in track frame
    f_along = fx * tx + fy * ty
    f_side = fx * nx + fy * ny

    drag_along_track.append(f_along)
    side_force.append(f_side)

drag_along_track = np.array(drag_along_track)
side_force = np.array(side_force)
hull_fx_track = np.array(hull_fx_track)
hull_fy_track = np.array(hull_fy_track)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(track_angles_deg, drag_along_track)
ax1.axhline(0.0, color="k", linewidth=0.5)
ax1.set_ylabel("F_along (N)")
ax1.set_title("Hull force components vs track angle (track-frame)")
ax1.grid(True)

ax2.plot(track_angles_deg, side_force, color="C2")
ax2.axhline(0.0, color="k", linewidth=0.5)
ax2.set_xlabel("Track angle (deg, 0 = along +x)")
ax2.set_ylabel("F_side (N)")
ax2.grid(True)

plt.tight_layout()
plt.show()

# Also plot the same hull force in the boat frame (Fx, Fy).
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(track_angles_deg, hull_fx_track)
ax1.axhline(0.0, color="k", linewidth=0.5)
ax1.set_ylabel("Fx (boat) [N]")
ax1.set_title("Hull force in boat frame vs track angle")
ax1.grid(True)

ax2.plot(track_angles_deg, hull_fy_track, color="C3")
ax2.axhline(0.0, color="k", linewidth=0.5)
ax2.set_xlabel("Track angle (deg, 0 = along +x)")
ax2.set_ylabel("Fy (boat) [N]")
ax2.grid(True)

plt.tight_layout()
plt.show()
