from dataclasses import dataclass
import numpy as np
from uav_opt.angles import heading_from_xy_delta


@dataclass(frozen=True)
class LocalFrame:
    """
    Coordinate frame used for planning.

    - UTM frame:
      x east, y north.

    - North-aligned planning frame:
      first waypoint is origin.
      first leg points along positive y.
    """

    origin_utm: np.ndarray
    flight_path_angle_rad: float
    rotation_utm_to_north: np.ndarray

    @staticmethod
    def from_waypoints_utm(waypoints_utm: np.ndarray) -> "LocalFrame":
        waypoints_utm = np.asarray(waypoints_utm, dtype=float)
        if len(waypoints_utm) < 2:
            raise ValueError("Need at least two waypoints to define LocalFrame.")

        origin = waypoints_utm[0].copy()
        first_leg = waypoints_utm[1] - waypoints_utm[0]

        angle = heading_from_xy_delta(first_leg[0], first_leg[1])

        # This maps first-leg vector [D sin(angle), D cos(angle)] to [0, D].
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=float,
        )

        return LocalFrame(
            origin_utm=origin,
            flight_path_angle_rad=angle,
            rotation_utm_to_north=R,
        )

    def to_north_frame(self, points_utm: np.ndarray) -> np.ndarray:
        points_utm = np.asarray(points_utm, dtype=float)
        translated = points_utm - self.origin_utm
        return translated @ self.rotation_utm_to_north.T

    def from_north_frame(self, points_north: np.ndarray) -> np.ndarray:
        points_north = np.asarray(points_north, dtype=float)
        return points_north @ self.rotation_utm_to_north + self.origin_utm
