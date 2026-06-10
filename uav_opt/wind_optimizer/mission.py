"""
Mission and waypoint helpers for wind-aware trajectory optimization.
"""

from __future__ import annotations

import numpy as np


def get_xy_from_waypoint(wp) -> tuple[float, float]:
    """
    Extract x/y from a waypoint-like object.

    Supports:
        - [x, y]
        - [x, y, ...]
        - dict with x/y
        - object with .x/.y
        - object with .lat/.lon is intentionally not converted here
    """

    if isinstance(wp, dict):
        if "x" in wp and "y" in wp:
            return float(wp["x"]), float(wp["y"])

        if "X" in wp and "Y" in wp:
            return float(wp["X"]), float(wp["Y"])

    if hasattr(wp, "x") and hasattr(wp, "y"):
        return float(wp.x), float(wp.y)

    if isinstance(wp, (tuple, list, np.ndarray)) and len(wp) >= 2:
        return float(wp[0]), float(wp[1])

    raise TypeError(
        "Could not extract x/y from waypoint. "
        "Expected [x, y], dict with x/y, or object with .x/.y."
    )


def course_between_points(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Course from point 1 to point 2.

    Convention:
        0 rad = North
        positive clockwise
    """
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return 0.0

    return float((np.pi / 2.0 - np.arctan2(dy, dx)) % (2.0 * np.pi))


def get_heading_from_waypoints(prev_wp, current_wp, next_wp=None) -> float:
    """
    Estimate course/heading at a waypoint.

    If next_wp is provided:
        heading from current_wp to next_wp.

    Otherwise:
        heading from prev_wp to current_wp.
    """
    if next_wp is not None:
        x1, y1 = get_xy_from_waypoint(current_wp)
        x2, y2 = get_xy_from_waypoint(next_wp)
    else:
        x1, y1 = get_xy_from_waypoint(prev_wp)
        x2, y2 = get_xy_from_waypoint(current_wp)

    return course_between_points(x1, y1, x2, y2)


def build_waypoint_pair(wp1, wp2, chi_1: float, chi_2: float) -> list[float]:
    """
    Build maneuver waypoint format expected by uav_opt.maneuvers:

        [x1, y1, x2, y2, chi_1, chi_2]
    """
    x1, y1 = get_xy_from_waypoint(wp1)
    x2, y2 = get_xy_from_waypoint(wp2)

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2),
        float(chi_1),
        float(chi_2),
    ]


def build_maneuver_waypoints(mission_points) -> list[list[float]]:
    """
    Convert a list of mission waypoints into maneuver waypoint pairs.

    For each leg i -> i+1:
        chi_1 is the course entering/leaving waypoint i
        chi_2 is the course leaving waypoint i+1

    For the final waypoint, chi_2 uses the final leg course.
    """

    if len(mission_points) < 2:
        return []

    maneuver_waypoints = []

    n = len(mission_points)

    for i in range(n - 1):
        wp1 = mission_points[i]
        wp2 = mission_points[i + 1]

        if i == 0:
            chi_1 = get_heading_from_waypoints(wp1, wp1, wp2)
        else:
            chi_1 = get_heading_from_waypoints(mission_points[i - 1], wp1, wp2)

        if i + 2 < n:
            chi_2 = get_heading_from_waypoints(wp1, wp2, mission_points[i + 2])
        else:
            chi_2 = get_heading_from_waypoints(wp1, wp2, None)

        maneuver_waypoints.append(build_waypoint_pair(wp1, wp2, chi_1, chi_2))

    return maneuver_waypoints
