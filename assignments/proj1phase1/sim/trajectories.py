from __future__ import annotations

import numpy as np


def hover_trajectory(t: float, true_s: np.ndarray) -> np.ndarray:
    """Hover at the origin."""
    return np.zeros(11)


def circle_trajectory(t: float, true_s: np.ndarray) -> np.ndarray:
    """Helix in the xy-plane with growing radius."""
    s_des = np.zeros(11)
    omega = 25.0

    angle = t * omega / 180.0 * np.pi
    x_des = 4.0 * np.cos(angle) * t / omega
    y_des = 4.0 * np.sin(angle) * t / omega
    z_des = 3.0 / 25.0 * t

    x_vdes = (4.0 * np.cos(angle) - omega / 180.0 * np.pi * 4.0 * np.sin(angle) * t) / omega
    y_vdes = (4.0 * np.sin(angle) + omega / 180.0 * np.pi * 4.0 * np.cos(angle) * t) / omega
    z_vdes = 3.0 / 25.0

    x_ades = (
        -2.0 * omega / 180.0 * np.pi * 4.0 * np.sin(angle)
        - omega / 180.0 * np.pi * omega / 180.0 * np.pi * 4.0 * np.cos(angle) * t
    ) / omega
    y_ades = (
        2.0 * omega / 180.0 * np.pi * 4.0 * np.cos(angle)
        - omega / 180.0 * np.pi * omega / 180.0 * np.pi * 4.0 * np.sin(angle) * t
    ) / omega
    z_ades = 0.0

    yaw_des = np.mod(0.1 * np.pi * t, 2.0 * np.pi)
    dyaw_des = 0.1 * np.pi

    s_des[0:3] = [x_des, y_des, z_des]
    s_des[3:6] = [x_vdes, y_vdes, z_vdes]
    s_des[6:9] = [x_ades, y_ades, z_ades]
    s_des[9] = yaw_des
    s_des[10] = dyaw_des
    return s_des


def square_trajectory(t: float, true_s: np.ndarray) -> np.ndarray:
    """Piecewise linear trajectory through the five waypoints."""
    s_des = np.zeros(11)
    omega = 25.0
    corner_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * omega

    corner_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    corner_y = np.array([0.0, 2.0, 2.0, 0.0, 0.0])
    corner_z = np.array([0.0, 0.0, 2.0, 2.0, 0.0])

    t = min(t, corner_t[-1])

    x_des = corner_x[-1]
    y_des = corner_y[-1]
    z_des = corner_z[-1]
    x_vdes = 0.0
    y_vdes = 0.0
    z_vdes = 0.0

    for i in range(1, len(corner_t)):
        if t <= corner_t[i] + 1e-8:
            dt = corner_t[i] - corner_t[i - 1]
            x_vdes = (corner_x[i] - corner_x[i - 1]) / dt
            y_vdes = (corner_y[i] - corner_y[i - 1]) / dt
            z_vdes = (corner_z[i] - corner_z[i - 1]) / dt
            ratio = (t - corner_t[i - 1]) / dt
            x_des = corner_x[i - 1] + ratio * (corner_x[i] - corner_x[i - 1])
            y_des = corner_y[i - 1] + ratio * (corner_y[i] - corner_y[i - 1])
            z_des = corner_z[i - 1] + ratio * (corner_z[i] - corner_z[i - 1])
            break

    x_ades = 0.0
    y_ades = 0.0
    z_ades = 0.0

    yaw_des = np.mod(0.2 * np.pi * t, 2.0 * np.pi)
    dyaw_des = 0.2 * np.pi

    s_des[0:3] = [x_des, y_des, z_des]
    s_des[3:6] = [x_vdes, y_vdes, z_vdes]
    s_des[6:9] = [x_ades, y_ades, z_ades]
    s_des[9] = yaw_des
    s_des[10] = dyaw_des
    return s_des


def diamond_trajectory(t: float, true_s: np.ndarray) -> np.ndarray:
    """Diamond (rhombus) trajectory in the yz-plane with slow x drift."""
    s_des = np.zeros(11)

    Period = 12.0
    Seg = np.sqrt(2.0)
    Vel = Seg * 4.0 / Period

    # slow x drift
    x_des = 4.0 / 25.0 * t
    x_vdes = 4.0 / 25.0
    x_ades = 0.0

    # wrap t into one period
    t1 = t % Period
    T1 = Period / 4.0
    T2 = Period / 2.0
    T3 = 3.0 * Period / 4.0

    if t1 <= T1:
        y_vdes = Vel;  z_vdes = Vel
        y_des  = y_vdes * t1
        z_des  = z_vdes * t1
    elif t1 <= T2:
        y_vdes = -Vel; z_vdes = Vel
        y_des  = Seg  + y_vdes * (t1 - T1)
        z_des  = Seg  + z_vdes * (t1 - T1)
    elif t1 <= T3:
        y_vdes = -Vel; z_vdes = -Vel
        y_des  =        y_vdes * (t1 - T2)
        z_des  = 2.0 * Seg + z_vdes * (t1 - T2)
    else:
        y_vdes = Vel;  z_vdes = -Vel
        y_des  = -Seg + y_vdes * (t1 - T3)
        z_des  = Seg  + z_vdes * (t1 - T3)

    yaw_des  = np.mod(0.2 * np.pi * t, 2.0 * np.pi)
    dyaw_des = 0.2 * np.pi

    s_des[0:3] = [x_des,  y_des,  z_des]
    s_des[3:6] = [x_vdes, y_vdes, z_vdes]
    s_des[6:9] = [x_ades, 0.0,    0.0]
    s_des[9]   = yaw_des
    s_des[10]  = dyaw_des
    return s_des


def heart_trajectory(t: float, true_s: np.ndarray) -> np.ndarray:
    """Heart-shaped trajectory in the xy-plane at fixed height."""
    s_des = np.zeros(11)
    T = 20.0       # seconds per full heart loop
    scale = 0.50   # spatial scale (~±4 m wide, ~3.25 m tall)
    z0 = 1.5       # constant flight height (m)

    u = 2.0 * np.pi * t / T
    du = 2.0 * np.pi / T
    du2 = du ** 2

    # Parametric heart: x = 16sin³(u),  y = 13cos(u)-5cos(2u)-2cos(3u)-cos(4u)
    x  = scale * 16.0 * np.sin(u) ** 3
    y  = scale * (13.0 * np.cos(u) - 5.0 * np.cos(2.0*u) - 2.0 * np.cos(3.0*u) - np.cos(4.0*u))

    vx = scale * 48.0 * np.sin(u)**2 * np.cos(u) * du
    vy = scale * (-13.0*np.sin(u) + 10.0*np.sin(2.0*u) + 6.0*np.sin(3.0*u) + 4.0*np.sin(4.0*u)) * du

    ax = scale * 48.0 * (2.0*np.sin(u)*np.cos(u)**2 - np.sin(u)**3) * du2
    ay = scale * (-13.0*np.cos(u) + 20.0*np.cos(2.0*u) + 18.0*np.cos(3.0*u) + 16.0*np.cos(4.0*u)) * du2

    s_des[0:3] = [x,  y,  z0]
    s_des[3:6] = [vx, vy, 0.0]
    s_des[6:9] = [ax, ay, 0.0]
    s_des[9]   = 0.0
    s_des[10]  = 0.0
    return s_des
