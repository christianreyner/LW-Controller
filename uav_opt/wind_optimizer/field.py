"""
Wind field utilities.

Convention:
    x: East
    y: North

Heading/course convention:
    0 rad = North
    positive clockwise

Wind direction convention:
    theta_wa is wind-from direction.
    The wind velocity vector points toward theta_wa + pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


def wrap_angle_2pi(angle_rad: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    return float(angle_rad % (2.0 * np.pi))


def wind_components(V_w: float, theta_wa: float) -> tuple[float, float]:
    """
    Convert wind speed and wind-from direction to x/y velocity components.

    Args:
        V_w:
            Wind speed in m/s.
        theta_wa:
            Wind-from direction in radians.

    Returns:
        wx, wy in m/s.
    """
    wx = V_w * np.sin(theta_wa + np.pi)
    wy = V_w * np.cos(theta_wa + np.pi)
    return float(wx), float(wy)


def wind_speed_direction_from_components(wx: float, wy: float) -> tuple[float, float]:
    """
    Convert x/y wind velocity components to speed and wind-from direction.

    Args:
        wx:
            East component, m/s.
        wy:
            North component, m/s.

    Returns:
        V_w, theta_wa
    """
    V_w = float(np.hypot(wx, wy))

    if V_w <= 1e-12:
        return 0.0, 0.0

    # Velocity vector points toward direction:
    #     wx = V sin(theta_to)
    #     wy = V cos(theta_to)
    theta_to = np.arctan2(wx, wy)

    # Wind-from is opposite.
    theta_from = wrap_angle_2pi(theta_to - np.pi)

    return V_w, theta_from


@dataclass(frozen=True)
class WindSample:
    """
    Wind sample at a point.

    Attributes:
        V_w:
            Wind speed in m/s.
        theta_wa:
            Wind-from direction in radians.
        wx:
            East velocity component in m/s.
        wy:
            North velocity component in m/s.
    """

    V_w: float
    theta_wa: float
    wx: float
    wy: float

    @classmethod
    def from_speed_direction(cls, V_w: float, theta_wa: float) -> "WindSample":
        wx, wy = wind_components(V_w, theta_wa)
        return cls(
            V_w=float(V_w),
            theta_wa=wrap_angle_2pi(theta_wa),
            wx=wx,
            wy=wy,
        )

    @classmethod
    def from_components(cls, wx: float, wy: float) -> "WindSample":
        V_w, theta_wa = wind_speed_direction_from_components(wx, wy)
        return cls(
            V_w=V_w,
            theta_wa=theta_wa,
            wx=float(wx),
            wy=float(wy),
        )


class ConstantWindField:
    """
    Constant wind field.

    Example:
        wind = ConstantWindField(V_w=5.0, theta_wa=np.radians(270.0))
        sample = wind.at(10.0, 50.0)
    """

    def __init__(self, V_w: float = 0.0, theta_wa: float = 0.0):
        self.sample = WindSample.from_speed_direction(V_w, theta_wa)

    def at(self, x: float, y: float) -> WindSample:
        return self.sample


class GridWindField:
    """
    Regular-grid wind field with bilinear interpolation.

    Args:
        x_grid:
            1D array of x coordinates.
        y_grid:
            1D array of y coordinates.
        wx_grid:
            2D array of east wind components.
            Shape should be (len(y_grid), len(x_grid)).
        wy_grid:
            2D array of north wind components.
            Shape should be (len(y_grid), len(x_grid)).
    """

    def __init__(
        self,
        x_grid: Sequence[float],
        y_grid: Sequence[float],
        wx_grid,
        wy_grid,
    ):
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.y_grid = np.asarray(y_grid, dtype=float)
        self.wx_grid = np.asarray(wx_grid, dtype=float)
        self.wy_grid = np.asarray(wy_grid, dtype=float)

        expected_shape = (len(self.y_grid), len(self.x_grid))

        if self.wx_grid.shape != expected_shape:
            raise ValueError(
                f"wx_grid shape must be {expected_shape}, got {self.wx_grid.shape}."
            )

        if self.wy_grid.shape != expected_shape:
            raise ValueError(
                f"wy_grid shape must be {expected_shape}, got {self.wy_grid.shape}."
            )

        if len(self.x_grid) < 2 or len(self.y_grid) < 2:
            raise ValueError("x_grid and y_grid must each contain at least 2 points.")

    def _interp_2d(self, grid, x: float, y: float) -> float:
        x = float(np.clip(x, self.x_grid[0], self.x_grid[-1]))
        y = float(np.clip(y, self.y_grid[0], self.y_grid[-1]))

        ix = int(np.searchsorted(self.x_grid, x) - 1)
        iy = int(np.searchsorted(self.y_grid, y) - 1)

        ix = int(np.clip(ix, 0, len(self.x_grid) - 2))
        iy = int(np.clip(iy, 0, len(self.y_grid) - 2))

        x0 = self.x_grid[ix]
        x1 = self.x_grid[ix + 1]
        y0 = self.y_grid[iy]
        y1 = self.y_grid[iy + 1]

        tx = 0.0 if abs(x1 - x0) <= 1e-12 else (x - x0) / (x1 - x0)
        ty = 0.0 if abs(y1 - y0) <= 1e-12 else (y - y0) / (y1 - y0)

        q00 = grid[iy, ix]
        q10 = grid[iy, ix + 1]
        q01 = grid[iy + 1, ix]
        q11 = grid[iy + 1, ix + 1]

        q0 = (1.0 - tx) * q00 + tx * q10
        q1 = (1.0 - tx) * q01 + tx * q11

        return float((1.0 - ty) * q0 + ty * q1)

    def at(self, x: float, y: float) -> WindSample:
        wx = self._interp_2d(self.wx_grid, x, y)
        wy = self._interp_2d(self.wy_grid, x, y)
        return WindSample.from_components(wx, wy)


def get_wind_at(wind, x: float, y: float) -> WindSample:
    """
    Generic wind accessor.

    Supports:
        - None
        - WindSample
        - ConstantWindField
        - GridWindField
        - object with .at(x, y)
        - callable wind(x, y)
        - tuple/list [V_w, theta_wa]
        - tuple/list [wx, wy, "components"]
        - dict with V_w/theta_wa
        - dict with wx/wy
    """

    if wind is None:
        return WindSample.from_speed_direction(0.0, 0.0)

    if isinstance(wind, WindSample):
        return wind

    if hasattr(wind, "at") and callable(wind.at):
        return wind.at(x, y)

    if callable(wind):
        value = wind(x, y)
        return get_wind_at(value, x, y)

    if isinstance(wind, dict):
        if "V_w" in wind and "theta_wa" in wind:
            return WindSample.from_speed_direction(wind["V_w"], wind["theta_wa"])

        if "speed" in wind and "direction" in wind:
            return WindSample.from_speed_direction(wind["speed"], wind["direction"])

        if "wx" in wind and "wy" in wind:
            return WindSample.from_components(wind["wx"], wind["wy"])

        raise ValueError(
            "Wind dict must contain either V_w/theta_wa, speed/direction, or wx/wy."
        )

    if isinstance(wind, (tuple, list, np.ndarray)):
        arr = list(wind)

        if len(arr) >= 3 and str(arr[2]).lower() in {"components", "component", "xy"}:
            return WindSample.from_components(float(arr[0]), float(arr[1]))

        if len(arr) >= 2:
            return WindSample.from_speed_direction(float(arr[0]), float(arr[1]))

    raise TypeError(f"Unsupported wind representation: {type(wind)!r}")
