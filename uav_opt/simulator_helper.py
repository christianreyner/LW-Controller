import numpy as np


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """
    2D cross product scalar for xy convention.

    Existing simulator convention:
        x velocity = V * sin(heading)
        y velocity = V * cos(heading)

    heading = 0 means +Y/north.
    positive heading means turn toward +X/east/right.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def heading_to_xy_velocity_unit(heading_rad: float) -> np.ndarray:
    return np.array(
        [
            np.sin(heading_rad),
            np.cos(heading_rad),
        ],
        dtype=float,
    )


def bearing_xy(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    """
    Bearing angle using this simulator's convention:
        0 rad = +Y
        +pi/2 = +X
    """
    delta = np.asarray(to_xy, dtype=float) - np.asarray(from_xy, dtype=float)
    return float(np.arctan2(delta[0], delta[1]))


def unit_vector(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(v))

    if n > 1e-9:
        return np.asarray(v, dtype=float) / n

    if fallback is not None:
        fb = np.asarray(fallback, dtype=float)
        fb_n = float(np.linalg.norm(fb))
        if fb_n > 1e-9:
            return fb / fb_n

    return np.array([0.0, 1.0], dtype=float)


def ardupilot_l1_distance(
    ground_speed_mps: float,
    damping: float,
    period_s: float,
    dist_min: float = 0.0,
) -> float:
    """
    ArduPilot-style L1 distance for waypoint following.

    AP_L1_Control uses:
        L1_dist = max(0.3183099 * damping * period * groundSpeed, dist_min)

    0.3183099 ~= 1/pi.
    """
    return float(
        max(
            (1.0 / np.pi) * damping * period_s * ground_speed_mps,
            dist_min,
        )
    )


def turn_angle_rad(
    prev_wp: np.ndarray,
    this_wp: np.ndarray,
    next_wp: np.ndarray,
) -> float:
    v1 = np.asarray(this_wp[:2], dtype=float) - np.asarray(prev_wp[:2], dtype=float)
    v2 = np.asarray(next_wp[:2], dtype=float) - np.asarray(this_wp[:2], dtype=float)

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    c = float(np.dot(v1, v2) / (n1 * n2))
    c = float(np.clip(c, -1.0, 1.0))

    return float(np.arccos(c))


def ardupilot_turn_distance(
    wp_radius_m: float,
    l1_distance_m: float,
    turn_angle_rad: float,
) -> float:
    """
    Approximate ArduPilot turn_distance(wp_radius, turn_angle).

    For a 90-degree or sharper turn:
        distance = min(wp_radius, L1_dist)

    For a shallow turn:
        distance is reduced linearly with turn angle.

    This prevents nearly straight legs from being considered complete too early.
    """
    distance_90 = min(float(wp_radius_m), float(l1_distance_m))

    turn_angle_deg = abs(float(np.rad2deg(turn_angle_rad)))

    if turn_angle_deg >= 90.0:
        return distance_90

    return distance_90 * turn_angle_deg / 90.0


def active_waypoint_acceptance_radius(
    wp: np.ndarray,
    target_index: int,
    wp_radius_m: float,
    l1_distance_m: float,
) -> float:
    """
    Radius used for deciding when to switch from the current leg to the next leg.

    For internal waypoints, use ArduPilot-style turn distance.
    For final waypoint, use WP_RADIUS.
    """
    if 0 < target_index < len(wp) - 1:
        angle = turn_angle_rad(
            wp[target_index - 1],
            wp[target_index],
            wp[target_index + 1],
        )

        return ardupilot_turn_distance(
            wp_radius_m=wp_radius_m,
            l1_distance_m=l1_distance_m,
            turn_angle_rad=angle,
        )

    return float(wp_radius_m)


def waypoint_reached_ardupilot_style(
    position_xy: np.ndarray,
    prev_wp_xy: np.ndarray,
    next_wp_xy: np.ndarray,
    acceptance_radius_m: float,
) -> bool:
    """
    ArduPilot-style waypoint completion approximation.

    A waypoint is considered reached if either:
    - aircraft is within the active waypoint acceptance radius, or
    - aircraft crosses the finish line through the waypoint perpendicular to
      the inbound path.

    ArduPilot's WP_RADIUS docs mention this extra finish-line logic to prevent
    looping around a missed waypoint.
    """
    position_xy = np.asarray(position_xy, dtype=float)
    prev_wp_xy = np.asarray(prev_wp_xy, dtype=float)
    next_wp_xy = np.asarray(next_wp_xy, dtype=float)

    distance_to_wp = float(np.linalg.norm(next_wp_xy - position_xy))

    if distance_to_wp <= acceptance_radius_m:
        return True

    ab = next_wp_xy - prev_wp_xy
    ab_len = float(np.linalg.norm(ab))

    if ab_len < 1e-9:
        return distance_to_wp <= acceptance_radius_m

    ab_unit = ab / ab_len

    along_track = float(np.dot(position_xy - prev_wp_xy, ab_unit))

    # Crossed finish line through next_wp perpendicular to AB.
    return along_track >= ab_len


def prevent_indecision_ardupilot_style(
    nu: float,
    last_nu: float,
    position_xy: np.ndarray,
    target_xy: np.ndarray,
    heading_rad: float,
) -> float:
    """
    Approximation of AP_L1_Control::_prevent_indecision().

    It prevents the aircraft from rapidly changing turn direction when it is
    pointed mostly away from the target and Nu sign flips.
    """
    nu_limit = 0.9 * np.pi

    target_bearing = bearing_xy(position_xy, target_xy)
    bearing_error = wrap_pi(target_bearing - heading_rad)

    if (
        abs(nu) > nu_limit
        and abs(last_nu) > nu_limit
        and abs(bearing_error) > np.deg2rad(120.0)
        and nu * last_nu < 0.0
    ):
        return float(last_nu)

    return float(nu)


def ardupilot_l1_waypoint_bank_command(
    position_xy: np.ndarray,
    heading_rad: float,
    prev_wp_xy: np.ndarray,
    next_wp_xy: np.ndarray,
    ground_velocity_xy: np.ndarray,
    damping: float,
    period_s: float,
    last_nu: float = 0.0,
    xtrack_i: float = 0.0,
    xtrack_i_gain: float = 0.0,
    dt_s: float = 0.02,
    dist_min: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """
    ArduPilot-style L1 waypoint-following command.

    Returns:
        desired_bank_rad
        l1_distance_m
        nu_limited_rad
        crosstrack_error_m
        along_track_m
        updated_last_nu
    """
    position_xy = np.asarray(position_xy, dtype=float)
    prev_wp_xy = np.asarray(prev_wp_xy, dtype=float)
    next_wp_xy = np.asarray(next_wp_xy, dtype=float)

    ground_velocity = np.asarray(ground_velocity_xy, dtype=float)

    ground_speed_mps = float(np.linalg.norm(ground_velocity))

    if ground_speed_mps < 0.1:
        # Match ArduPilot-style fallback: if GPS ground speed is too small,
        # use a small forward vector based on aircraft heading.
        ground_speed_mps = 0.1
        ground_velocity = (
            _heading_to_xy_velocity_unit(heading_rad) * ground_speed_mps
        )

    k_l1 = 4.0 * float(damping) ** 2

    l1_distance = ardupilot_l1_distance(
        ground_speed_mps=ground_speed_mps,
        damping=damping,
        period_s=period_s,
        dist_min=dist_min,
    )

    ab = next_wp_xy - prev_wp_xy
    ab_length = float(np.linalg.norm(ab))

    if ab_length < 1e-9:
        ab = next_wp_xy - position_xy
        ab_length = float(np.linalg.norm(ab))

        if ab_length < 1e-9:
            ab = heading_to_xy_velocity_unit(heading_rad)
            ab_length = 1.0

    ab_unit = ab / ab_length

    a_air = position_xy - prev_wp_xy

    # Sign adapted to this simulator's xy convention.
    # For a northbound path, aircraft east/right of track gives negative error,
    # which commands a left turn.
    crosstrack_error = cross2(ab_unit, a_air)

    wp_a_dist = float(np.linalg.norm(a_air))
    along_track = float(np.dot(a_air, ab_unit))

    # ------------------------------------------------------------------
    # Case 1:
    # Aircraft is behind WP A by more than L1 distance and outside the
    # +-135 degree arc. Fly to WP A.
    # ------------------------------------------------------------------
    if (
        wp_a_dist > l1_distance
        and along_track / max(wp_a_dist, 1.0) < -0.7071
    ):
        target_vec = prev_wp_xy - position_xy
        target_unit = unit_vector(
            target_vec,
            fallback=heading_to_xy_velocity_unit(heading_rad),
        )

        xtrack_vel = cross2(target_unit, ground_velocity)
        ltrack_vel = float(np.dot(ground_velocity, target_unit))

        nu = float(np.arctan2(xtrack_vel, ltrack_vel))

    # ------------------------------------------------------------------
    # Case 2:
    # Aircraft has passed WP B by roughly 3 seconds. Fly back to WP B.
    # ------------------------------------------------------------------
    elif along_track > ab_length + ground_speed_mps * 3.0:
        target_vec = next_wp_xy - position_xy
        target_unit = unit_vector(
            target_vec,
            fallback=heading_to_xy_velocity_unit(heading_rad),
        )

        xtrack_vel = cross2(target_unit, ground_velocity)
        ltrack_vel = float(np.dot(ground_velocity, target_unit))

        nu = float(np.arctan2(xtrack_vel, ltrack_vel))

    # ------------------------------------------------------------------
    # Case 3:
    # Normal active-leg L1 path following.
    # ------------------------------------------------------------------
    else:
        xtrack_vel = cross2(ab_unit, ground_velocity)
        ltrack_vel = float(np.dot(ground_velocity, ab_unit))

        nu2 = float(np.arctan2(xtrack_vel, ltrack_vel))

        sine_nu1 = crosstrack_error / max(l1_distance, 0.1)

        # ArduPilot limits the capture angle to +-45 deg.
        sine_nu1 = float(np.clip(sine_nu1, -0.7071, 0.7071))

        nu1 = float(np.arcsin(sine_nu1))

        # Optional cross-track integrator. Default gain should be zero unless
        # you explicitly add NAVL1_XTRACK_I behavior to L1Config.
        if xtrack_i_gain <= 0.0:
            xtrack_i = 0.0
        elif abs(nu1) < np.deg2rad(5.0):
            xtrack_i += nu1 * xtrack_i_gain * dt_s
            xtrack_i = float(np.clip(xtrack_i, -0.1, 0.1))

        nu1 += xtrack_i

        nu = nu1 + nu2

    nu = prevent_indecision_ardupilot_style(
        nu=nu,
        last_nu=last_nu,
        position_xy=position_xy,
        target_xy=next_wp_xy,
        heading_rad=heading_rad,
    )

    updated_last_nu = float(nu)

    # ArduPilot limits Nu to +-90 deg before computing lateral acceleration.
    nu_limited = float(np.clip(nu, -0.5 * np.pi, 0.5 * np.pi))

    lateral_accel = (
        k_l1
        * ground_speed_mps
        * ground_speed_mps
        / max(l1_distance, 1e-6)
        * np.sin(nu_limited)
    )

    desired_bank = float(np.arctan(lateral_accel / 9.81))

    return (
        desired_bank,
        l1_distance,
        nu_limited,
        crosstrack_error,
        along_track,
        updated_last_nu,
    )

