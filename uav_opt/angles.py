import numpy as np


def wrap_2pi(angle_rad: float) -> float:
    return float(angle_rad % (2.0 * np.pi))


def wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def heading_from_xy_delta(dx: float, dy: float) -> float:
    """
    Heading convention:
    - x is East
    - y is North
    - heading is clockwise from North
    """
    return wrap_2pi(np.arctan2(dx, dy))


def course_between_points(p0, p1) -> float:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    return heading_from_xy_delta(delta[0], delta[1])


def signed_angle_to_point(position, heading_rad: float, target) -> float:
    """
    Signed heading error from current heading to target point.
    """
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)

    vec = target - position
    angle_to_target = heading_from_xy_delta(vec[0], vec[1])
    return wrap_pi(angle_to_target - heading_rad)
