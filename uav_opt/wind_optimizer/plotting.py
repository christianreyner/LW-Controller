"""
Plotting utilities for wind fields and optimized missions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from uav_opt.wind_optimizer.field import get_wind_at


def plot_wind_field(
    wind,
    xlim=(-100.0, 200.0),
    ylim=(-100.0, 200.0),
    nx: int = 15,
    ny: int = 15,
    ax=None,
):
    """
    Plot wind field arrows.

    Args:
        wind:
            Wind object supported by get_wind_at(...).
        xlim:
            x-axis limits.
        ylim:
            y-axis limits.
        nx:
            Number of x samples.
        ny:
            Number of y samples.
        ax:
            Optional matplotlib axes.

    Returns:
        ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)

    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for iy in range(ny):
        for ix in range(nx):
            sample = get_wind_at(wind, X[iy, ix], Y[iy, ix])
            U[iy, ix] = sample.wx
            V[iy, ix] = sample.wy

    ax.quiver(
        X,
        Y,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0025,
        alpha=0.55,
        color="#d62728",
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X East (m)")
    ax.set_ylabel("Y North (m)")
    ax.set_title("Wind Field")
    ax.grid(True, alpha=0.35)

    return ax


def plot_optimized_mission(
    DYN,
    mission_points=None,
    wind=None,
    title: str = "Optimized Mission",
    show: bool = True,
):
    """
    Plot optimized mission trajectory.
    """
    traj = np.asarray(DYN, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 7))

    if traj.size > 0:
        xs = traj[:, 1]
        ys = traj[:, 2]

        xmin = float(np.min(xs))
        xmax = float(np.max(xs))
        ymin = float(np.min(ys))
        ymax = float(np.max(ys))
    else:
        xs = np.array([])
        ys = np.array([])

        xmin, xmax = -100.0, 100.0
        ymin, ymax = -100.0, 100.0

    if mission_points:
        mx = []
        my = []

        for wp in mission_points:
            if isinstance(wp, dict):
                mx.append(float(wp["x"]))
                my.append(float(wp["y"]))
            elif hasattr(wp, "x") and hasattr(wp, "y"):
                mx.append(float(wp.x))
                my.append(float(wp.y))
            else:
                mx.append(float(wp[0]))
                my.append(float(wp[1]))

        xmin = min(xmin, min(mx))
        xmax = max(xmax, max(mx))
        ymin = min(ymin, min(my))
        ymax = max(ymax, max(my))

    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    margin = 0.2 * max(dx, dy)

    xlim = (xmin - margin, xmax + margin)
    ylim = (ymin - margin, ymax + margin)

    if wind is not None:
        plot_wind_field(
            wind,
            xlim=xlim,
            ylim=ylim,
            nx=13,
            ny=13,
            ax=ax,
        )

    if traj.size > 0:
        ax.plot(xs, ys, color="#1f77b4", lw=2.5, label="Optimized path")

    if mission_points:
        ax.plot(mx, my, "o--", color="#333333", lw=1.2, ms=5, label="Mission waypoints")

        for i, (x, y) in enumerate(zip(mx, my)):
            ax.text(x, y, f" WP{i}", fontsize=10, va="center")

    total_cost = float(np.sum(traj[:, 10])) if traj.size > 0 and traj.shape[1] > 10 else 0.0
    total_time = float(traj[-1, 0]) if traj.size > 0 else 0.0
    total_dist = (
        float(np.sum(np.hypot(np.diff(traj[:, 1]), np.diff(traj[:, 2]))))
        if traj.size > 0
        else 0.0
    )

    ax.text(
        0.02,
        0.98,
        f"Distance: {total_dist:.1f} m\n"
        f"Time:     {total_time:.2f} s\n"
        f"Cost:     {total_cost:.3f} Wh",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb"),
        fontfamily="monospace",
        fontsize=10,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X East (m)")
    ax.set_ylabel("Y North (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax
