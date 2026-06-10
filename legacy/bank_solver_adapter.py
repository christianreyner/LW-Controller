"""
Adapter around your existing wind-aware bank solver in angle_solver.py.
"""


def desired_bank_with_wind_and_roll(*args, **kwargs):
    try:
        from angle_solver import desired_bank_to_point_with_wind_roll
    except Exception as exc:
        raise ImportError(
            "Could not import angle_solver.desired_bank_to_point_with_wind_roll. "
            "Keep angle_solver.py in the project root for now."
        ) from exc

    return desired_bank_to_point_with_wind_roll(*args, **kwargs)
