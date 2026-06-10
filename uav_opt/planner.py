from dataclasses import dataclass
import numpy as np

from uav_opt.config import AircraftConfig, L1Config
from uav_opt.transforms import LocalFrame
from uav_opt.path_utils import dynamic_split_2d
from legacy.optimizer_adapter import compute_optimal_points


@dataclass(frozen=True)
class PlannedPath:
    local_frame: LocalFrame
    original_waypoints_utm: np.ndarray
    original_waypoints_north: np.ndarray
    optimal_path_north: np.ndarray
    optimal_path_utm: np.ndarray
    subarrays: list[tuple[np.ndarray, np.ndarray]]
    initial_position_utm: np.ndarray
    initial_heading_rad: float
    turning_cost: float
    straight_cost: float
    turning_time_s: float
    straight_time_s: float


class OptimalPathPlanner:
    def __init__(self, aircraft: AircraftConfig, l1: L1Config):
        self.aircraft = aircraft
        self.l1 = l1

    def plan(self, waypoints_utm: np.ndarray) -> PlannedPath:
        waypoints_utm = np.asarray(waypoints_utm, dtype=float)

        frame = LocalFrame.from_waypoints_utm(waypoints_utm)
        waypoints_north = frame.to_north_frame(waypoints_utm)

        relative_wind = self.l1.wind_from_direction_rad - frame.flight_path_angle_rad

        coords_opt_north, turning_cost, straight_cost, turning_time, straight_time = compute_optimal_points(
            waypoints_north_frame=waypoints_north,
            airspeed_mps=self.aircraft.airspeed_mps,
            wind_speed_mps=self.l1.wind_speed_mps,
            relative_wind_from_direction_rad=relative_wind,
            min_bank_rad=self.aircraft.min_bank_rad,
            max_bank_rad=self.aircraft.max_bank_optimizer_rad,
            dt_s=self.l1.dt_s,
            aero_array=self.aircraft.aero.as_array(),
            roll_rate_rad_s=self.aircraft.roll_rate_rad_s,
        )

        coords_opt_north = np.asarray(coords_opt_north, dtype=float)
        coords_opt_utm = frame.from_north_frame(coords_opt_north)

        subarrays = dynamic_split_2d(coords_opt_utm, threshold_m=10.0)

        return PlannedPath(
            local_frame=frame,
            original_waypoints_utm=waypoints_utm,
            original_waypoints_north=waypoints_north,
            optimal_path_north=coords_opt_north,
            optimal_path_utm=coords_opt_utm,
            subarrays=subarrays,
            initial_position_utm=waypoints_utm[0],
            initial_heading_rad=frame.flight_path_angle_rad,
            turning_cost=float(turning_cost),
            straight_cost=float(straight_cost),
            turning_time_s=float(turning_time),
            straight_time_s=float(straight_time),
        )
