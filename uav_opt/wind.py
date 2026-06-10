import numpy as np
from uav_opt.angles import wrap_2pi


def wind_from_to_components(wind_speed_mps: float, wind_from_direction_rad: float) -> tuple[float, float]:
    """
    Convert wind speed and wind-from direction into x/y wind velocity.

    Direction convention:
    - 0 rad means wind from North, blowing South.
    - pi/2 means wind from East, blowing West.
    """
    wx = wind_speed_mps * np.sin(wind_from_direction_rad + np.pi)
    wy = wind_speed_mps * np.cos(wind_from_direction_rad + np.pi)
    return float(wx), float(wy)


def wind_correction(
    airspeed_mps: float,
    course_rad: float,
    wind_speed_mps: float,
    wind_from_direction_rad: float,
) -> tuple[float, float, float]:
    """
    Compute wind correction angle, desired heading, and ground speed.

    This keeps the same convention as your original get_windcorrection().
    """
    if airspeed_mps <= 1e-6:
        raise ValueError("airspeed_mps must be positive.")

    ratio = wind_speed_mps / airspeed_mps
    wind_angle = course_rad - (np.pi + wind_from_direction_rad)

    arg = ratio * np.sin(wind_angle)
    arg = np.clip(arg, -1.0, 1.0)

    wca = np.arcsin(arg)
    desired_heading = wrap_2pi(course_rad + wca)

    ground_speed = np.sqrt(
        airspeed_mps**2
        + wind_speed_mps**2
        - 2.0
        * airspeed_mps
        * wind_speed_mps
        * np.cos(wca - wind_from_direction_rad + course_rad)
    )

    return float(wca), float(desired_heading), float(ground_speed)


def ground_velocity_from_heading(
    airspeed_mps: float,
    heading_rad: float,
    wind_speed_mps: float,
    wind_from_direction_rad: float,
) -> np.ndarray:
    wx, wy = wind_from_to_components(wind_speed_mps, wind_from_direction_rad)

    vx = airspeed_mps * np.sin(heading_rad) + wx
    vy = airspeed_mps * np.cos(heading_rad) + wy

    return np.array([vx, vy], dtype=float)

def wind_to_xy_velocity(
    wind_speed_mps: float,
    wind_from_direction_rad: float,
) -> np.ndarray:
    """
    Convert meteorological 'wind from' direction into xy velocity.

    Simulator convention:
        x = east/right
        y = north/up
        heading 0 = +Y/north
        heading +pi/2 = +X/east

    wind_from_direction_rad means the direction the wind comes FROM.
    Therefore wind velocity points toward wind_from_direction + pi.
    """
    wind_to_direction = wind_from_direction_rad + np.pi

    return np.array(
        [
            wind_speed_mps * np.sin(wind_to_direction),
            wind_speed_mps * np.cos(wind_to_direction),
        ],
        dtype=float,
    )


def air_and_ground_velocity_xy(
    airspeed_mps: float,
    heading_rad: float,
    wind_speed_mps: float,
    wind_from_direction_rad: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns:
        air_velocity_xy
        ground_velocity_xy
        ground_speed_mps
        ground_track_rad
    """
    air_velocity = np.array(
        [
            airspeed_mps * np.sin(heading_rad),
            airspeed_mps * np.cos(heading_rad),
        ],
        dtype=float,
    )

    wind_velocity = wind_to_xy_velocity(
        wind_speed_mps=wind_speed_mps,
        wind_from_direction_rad=wind_from_direction_rad,
    )

    ground_velocity = air_velocity + wind_velocity

    ground_speed = float(np.linalg.norm(ground_velocity))

    if ground_speed > 1e-9:
        ground_track = float(np.arctan2(ground_velocity[0], ground_velocity[1]))
    else:
        ground_track = float(heading_rad)

    return air_velocity, ground_velocity, ground_speed, ground_track
