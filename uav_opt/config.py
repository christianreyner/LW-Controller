from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AeroConfig:
    """
    Aerodynamic model parameters.

    W:
        Aircraft mass in kg.
    S:
        Wing reference area in m^2.
    rho:
        Air density in kg/m^3.
    Cl_0:
        Zero-alpha lift coefficient.
    Cd_0:
        Zero-lift drag coefficient.
    Cl_alpha:
        Lift slope per rad.
    K:
        Induced drag factor.
    """

    W: float = 5.4
    S: float = 0.49
    rho: float = 1.105
    Cl_0: float = 0.0
    Cd_0: float = 0.021
    Cl_alpha: float = 2.0 * np.pi
    K: float = 0.055

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.W, self.S, self.rho, self.Cl_0, self.Cd_0, self.Cl_alpha, self.K],
            dtype=float,
        )


@dataclass(frozen=True)
class AircraftConfig:
    """
    Aircraft and maneuver limits.
    """

    airspeed_mps: float = 18.5
    roll_rate_rad_s: float = np.deg2rad(30.0)
    min_bank_rad: float = np.deg2rad(5.0)
    max_bank_rad: float = np.deg2rad(45.0)
    max_bank_optimizer_rad: float = np.deg2rad(35.0)
    aero: AeroConfig = AeroConfig()


@dataclass(frozen=True)
class L1Config:
    """
    L1 path-following and wind configuration.
    """

    wind_speed_mps: float = 10.0
    wind_from_direction_rad: float = np.deg2rad(0.0)
    

    damping: float = 1.0 / np.sqrt(2.0)
    period_s: float = 10.0
    dt_s: float = 0.05

    waypoint_radius_m: float = 30.0
    
    # Only for LW Controller
    use_wind_aware_bank_solver: bool = True 
    tol_pos=0.1
    dt_solver=0.002
    max_iter = 40


@dataclass(frozen=True)
class ExecutionConfig:
    """
    ArduPilot execution settings.
    """

    connection_string: str = "udp:0.0.0.0:14550"
    temp_mission_file: str = "temp.waypoints"

    evaluate_before_flight: bool = True
    internal_sim_time_s: float = 500.0
    guidance_max_time_s: float = 500.0

    arm_timeout_s: float = 20.0

    takeoff_alt_m: float = 15.0
    cruise_alt_m: float = 60.0

    transition_throttle_pwm: int = 1500
    transition_min_speed_mps: float = 15.0
    transition_hold_s: float = 3.0


@dataclass(frozen=True)
class AppConfig:
    aircraft: AircraftConfig
    l1: L1Config
    execution: ExecutionConfig

    @staticmethod
    def default() -> "AppConfig":
        aircraft = AircraftConfig()
        l1 = L1Config(
            waypoint_radius_m=30,
        )
        execution = ExecutionConfig()
        return AppConfig(aircraft=aircraft, l1=l1, execution=execution)
