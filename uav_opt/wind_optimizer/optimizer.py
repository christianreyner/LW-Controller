# -------------------------------------------------------------------------
# Original wind_module.py compatible implementation
# -------------------------------------------------------------------------

import math
import numpy as np

from uav_opt.maneuvers.selector import optimal_path
from uav_opt.maneuvers.geometry import aero_cost_wh


pitch = 0


def get_angle_and_distance(point1, point2):
    """
    Original wind_module.py helper.
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = math.atan2(dx, dy)
    distance = math.sqrt(dx**2 + dy**2)
    return angle, distance


def get_simplewindcorrection(AS, course, WS, wind_direction):
    """
    Original wind_module.py helper.

    Returns ground speed for straight segment.
    """
    wind_angle = course - (np.pi + wind_direction)
    wca = np.arcsin((WS / AS) * np.sin(wind_angle))
    GS = np.sqrt(
        AS**2
        + WS**2
        - (2 * AS * WS * np.cos(wca - wind_direction + course))
    )
    return GS


def get_windcorrection(AS, course, WS, wind_direction):
    """
    Original wind_module.py helper.
    """
    wind_angle = course - (np.pi + wind_direction)
    wca = np.arcsin((WS / AS) * np.sin(wind_angle))
    des_heading = course + wca
    GS = np.sqrt(
        AS**2
        + WS**2
        - (2 * AS * WS * np.cos(wca - wind_direction + course))
    )
    return wca, des_heading, GS


def get_listcourses(coordinates):
    """
    Original wind_module.py helper.

    Course convention:
        angle = atan2(dx, dy)
    """
    list_courses = []

    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]

        dx = x2 - x1
        dy = y2 - y1

        angle = math.atan2(dx, dy)
        list_courses.append(angle)

    return list_courses


def get_optpoints(
    intersect_points,
    AS,
    WS,
    wind_direction,
    min_phi,
    max_phi,
    timestep,
    func_x,
    func_y,
    Aero,
    roll_rate,
):
    """
    Original wind_module.py get_optpoints implementation.

    Signature preserved exactly:

        get_optpoints(
            intersect_points,
            AS,
            WS,
            wind_direction,
            min_phi,
            max_phi,
            timestep,
            func_x,
            func_y,
            Aero,
            roll_rate,
        )

    Returns:

        intersect_points,
        turning_cost,
        straight_cost,
        turning_time,
        straight_time

    Notes:
        - min_phi is currently unused, matching the active original code.
        - func_x and func_y are currently unused, matching the active original code.
        - This preserves the original flip logic and coordinate insertion behavior.
    """

    intersect_points = np.asarray(intersect_points, dtype=float)

    if intersect_points.ndim != 2 or intersect_points.shape[1] != 2:
        raise ValueError(
            "intersect_points must be an array-like of shape (N, 2). "
            f"Got shape {intersect_points.shape}."
        )

    constants = [
        9.81,
        AS,
        max_phi,
        WS,
        wind_direction,
        timestep,
    ]

    wind_direction_flipped = (2 * np.pi - (wind_direction + np.pi)) % (2 * np.pi)

    constants_flipped = [
        9.81,
        AS,
        max_phi,
        WS,
        wind_direction_flipped,
        timestep,
    ]

    list_GS = []
    list_courses = []
    list_distances = []
    list_times = []

    # ------------------------------------------------------------------
    # Straight cost evaluation
    # ------------------------------------------------------------------
    list_courses = get_listcourses(intersect_points)

    for course in list_courses:
        GS = get_simplewindcorrection(
            AS,
            course,
            WS,
            wind_direction,
        )
        list_GS.append(GS)

    # Calculate distance between adjacent coordinates.
    list_distances = np.linalg.norm(
        np.diff(intersect_points, axis=0),
        axis=1,
    )

    # Calculate time between adjacent coordinates.
    list_times = list_distances / list_GS

    # Original logic:
    # odd straight rows only via [::2]
    list_times_straight = list_times[::2]

    straight_cost = aero_cost_wh(
        0,
        AS,
        np.sum(list_times_straight),
        Aero,
    )

    straight_time = np.sum(list_times_straight)

    # ------------------------------------------------------------------
    # Add optimal turn strategy at each turning point
    # ------------------------------------------------------------------
    add_ = 0
    flip = False
    turning_cost = 0
    turning_time = 0

    original_num_turns = len(intersect_points) // 2

    for n in range(original_num_turns):
        DYN = np.zeros((1, 11))

        start_idx = 2 * n + 1 + add_
        end_idx = 2 * n + 2 + add_

        if end_idx < len(intersect_points):
            Target = intersect_points[end_idx] - intersect_points[start_idx]

            if flip:
                Target[1] *= -1

                waypoints = [
                    0,
                    0,
                    Target[0],
                    Target[1],
                    0,
                    np.pi,
                ]

                DYN = optimal_path(
                    waypoints,
                    constants_flipped,
                    Aero,
                    roll_rate,
                )

                DYN = np.asarray(DYN, dtype=float)

                windpath = np.column_stack(
                    (
                        DYN[:, 1] + intersect_points[start_idx, 0],
                        -DYN[:, 2] + intersect_points[start_idx, 1],
                    )
                )

                flip = False

            else:
                waypoints = [
                    0,
                    0,
                    Target[0],
                    Target[1],
                    0,
                    np.pi,
                ]

                DYN = optimal_path(
                    waypoints,
                    constants,
                    Aero,
                    roll_rate,
                )

                DYN = np.asarray(DYN, dtype=float)

                windpath = np.column_stack(
                    (
                        DYN[:, 1] + intersect_points[start_idx, 0],
                        DYN[:, 2] + intersect_points[start_idx, 1],
                    )
                )

                flip = True

            intersect_points = np.insert(
                intersect_points,
                start_idx + 1,
                windpath,
                axis=0,
            )

            add_ += len(DYN)

        turning_cost += np.sum(DYN[:, 10])
        turning_time += len(DYN) * timestep

    return (
        intersect_points,
        turning_cost,
        straight_cost,
        turning_time,
        straight_time,
    )
