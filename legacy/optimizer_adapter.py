"""
Adapter around the existing legacy optimal path generator.

Keep your current wind_module.py and Trochoidal.py available on PYTHONPATH.
"""

from __future__ import annotations
import numpy as np
from uav_opt.wind_optimizer.field import *
from uav_opt.wind_optimizer.mission import *
from uav_opt.wind_optimizer.optimizer import *
from uav_opt.wind_optimizer.plotting import *

def compute_optimal_points(
    waypoints_north_frame: np.ndarray,
    airspeed_mps: float,
    wind_speed_mps: float,
    relative_wind_from_direction_rad: float,
    min_bank_rad: float,
    max_bank_rad: float,
    dt_s: float,
    aero_array: np.ndarray,
    roll_rate_rad_s: float,
):

    return get_optpoints(
        waypoints_north_frame,
        airspeed_mps,
        wind_speed_mps,
        relative_wind_from_direction_rad,
        min_bank_rad,
        max_bank_rad,
        dt_s,
        None,
        None,
        aero_array,
        roll_rate_rad_s,
    )
