"""
SBB/BBS maneuver planners.

SBB:
    Straight - Bend - Bend

BBS:
    Bend - Bend - Straight

This file also contains robust bisection helpers used by these planners.

Assumptions:
    * DYN[:, 0] contains segment-local time.
    * DYN[:, 1] and DYN[:, 2] contain x/y position.
    * DYN[:, 7] contains yaw/heading.
    * Positive bank produces increasing yaw.
    * local_delta_xy() returns (lateral, longitudinal).

The bend-angle search range is configurable through max_del_phi. By default,
the planners search from zero through 2*pi.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from uav_opt.maneuvers.geometry import (
    local_delta_xy,
    course_from_yaw,
    yaw_from_course,
)
from uav_opt.maneuvers.turns import compute_turn
from uav_opt.maneuvers.straight import compute_straight


POSITION_TOLERANCE = 1.0
BISECTION_ACCEPTANCE_TOLERANCE = 0.5 * POSITION_TOLERANCE
HEADING_TOLERANCE = np.deg2rad(1.0)
DISTANCE_TOLERANCE = 0.01

# Default maximum additional bend-angle magnitude.
#
# Change this constant to alter the default globally, or pass max_del_phi
# directly to SBB_BBS_maneuver(), SBB_BBS_inv_maneuver(), or the bisection
# helpers.
DEFAULT_MAX_DEL_PHI = 2.0 * np.pi

# Small numerical allowance used only during final angle validation.
ANGLE_LIMIT_TOLERANCE = 1e-9

# The old 65-point scan over pi had an angular spacing of approximately
# pi / 64. Using 129 points over 2*pi preserves approximately the same
# angular resolution.
BISECTION_SCAN_POINTS = 129


class ManeuverPlanningError(RuntimeError):
    """Raised when no valid maneuver solution can be constructed."""


def _angle_error(angle: float, reference: float) -> float:
    """Return wrapped signed angle error in [-pi, pi]."""
    return float(
        np.arctan2(
            np.sin(angle - reference),
            np.cos(angle - reference),
        )
    )


def _validate_max_del_phi(max_del_phi: float) -> float:
    """Validate and return the maximum bend-angle search magnitude."""
    max_del_phi = float(max_del_phi)

    if not np.isfinite(max_del_phi):
        raise ValueError("max_del_phi must be finite")

    if max_del_phi <= 0.0:
        raise ValueError("max_del_phi must be positive")

    return max_del_phi


def _validate_trajectory(DYN: np.ndarray, name: str) -> np.ndarray:
    """Validate the minimum trajectory shape required by this module."""
    DYN = np.asarray(DYN, dtype=float)

    if DYN.ndim != 2:
        raise ManeuverPlanningError(
            f"{name} must be a two-dimensional array"
        )

    if DYN.shape[0] == 0:
        raise ManeuverPlanningError(f"{name} is empty")

    if DYN.shape[1] <= 7:
        raise ManeuverPlanningError(
            f"{name} must contain at least 8 columns"
        )

    if not np.all(np.isfinite(DYN)):
        raise ManeuverPlanningError(
            f"{name} contains non-finite values"
        )

    return DYN


def _stack_segments(*segments: np.ndarray) -> np.ndarray:
    """
    Concatenate trajectory segments.

    The first sample of every segment after the first is omitted because it
    represents the same boundary state as the preceding segment's final
    sample.

    Column zero is treated as segment-local time and is offset so that time
    remains continuous across segments.
    """
    valid_segments: list[np.ndarray] = []

    for index, segment in enumerate(segments):
        segment = _validate_trajectory(
            segment,
            f"trajectory segment {index}",
        )
        valid_segments.append(segment)

    if not valid_segments:
        raise ManeuverPlanningError(
            "No trajectory segments were supplied"
        )

    column_count = valid_segments[0].shape[1]

    if any(
        segment.shape[1] != column_count
        for segment in valid_segments
    ):
        raise ManeuverPlanningError(
            "All trajectory segments must have the same number of columns"
        )

    result = valid_segments[0].copy()

    for segment in valid_segments[1:]:
        segment = segment.copy()

        # Convert segment-local time to continuous maneuver time.
        segment[:, 0] += result[-1, 0] - segment[0, 0]

        # Skip the repeated segment boundary sample.
        if segment.shape[0] > 1:
            result = np.vstack((result, segment[1:]))

    return result


def _compute_two_bends(
    x_start,
    y_start,
    initial_heading,
    final_heading,
    intermediate_heading,
    g,
    first_bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    roll_rate,
    roll_in=0.0,
):
    """
    Simulate two connected bends.

    The first bend ends while maintaining first_bank_angle. The second bend
    starts at that bank and ends level.
    """
    DYN_first = compute_turn(
        x_start,
        y_start,
        initial_heading,
        intermediate_heading,
        g,
        first_bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
        roll_in=roll_in,
        roll_out=first_bank_angle,
    )
    DYN_first = _validate_trajectory(
        DYN_first,
        "first bend",
    )

    DYN_second = compute_turn(
        DYN_first[-1, 1],
        DYN_first[-1, 2],
        DYN_first[-1, 7],
        final_heading,
        g,
        -first_bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate=roll_rate,
        roll_in=first_bank_angle,
        roll_out=0.0,
    )
    DYN_second = _validate_trajectory(
        DYN_second,
        "second bend",
    )

    return DYN_first, DYN_second


def compute_final_position(
    x_turn,
    y_turn,
    initial_heading,
    psi_2,
    del_phi,
    g,
    bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    roll_rate,
    roll_in=0.0,
    first_heading_base=None,
):
    """
    Simulate two connected bends and return the final x/y position.

    Args:
        first_heading_base:
            Heading around which del_phi is applied. If omitted, the initial
            heading is used for backward compatibility.

            The intermediate heading is:

                first_heading_base + sign(bank_angle) * del_phi
    """
    if not np.isfinite(del_phi) or del_phi < 0.0:
        raise ManeuverPlanningError(
            "del_phi must be a finite nonnegative magnitude"
        )

    if not np.isfinite(bank_angle) or np.isclose(bank_angle, 0.0):
        raise ManeuverPlanningError(
            "bank_angle must be finite and nonzero"
        )

    if first_heading_base is None:
        first_heading_base = initial_heading

    turn_sign = 1.0 if bank_angle > 0.0 else -1.0
    intermediate_heading = (
        first_heading_base + turn_sign * del_phi
    )

    _, DYN_second = _compute_two_bends(
        x_turn,
        y_turn,
        initial_heading,
        psi_2,
        intermediate_heading,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        roll_rate,
        roll_in=roll_in,
    )

    return (
        float(DYN_second[-1, 1]),
        float(DYN_second[-1, 2]),
    )


def _bisection_method(
    component: int,
    x_turn,
    y_turn,
    initial_heading,
    psi_2,
    reference_course,
    g,
    bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    x2,
    y2,
    roll_rate,
    tol=0.01,
    max_iter=40,
    change_threshold=1e-5,
    roll_in=0.0,
    first_heading_base=None,
    max_del_phi=DEFAULT_MAX_DEL_PHI,
):
    """
    Find a bend-angle magnitude that approximately zeros a local position
    component.

    The objective is generated by a time-discretized trajectory simulation,
    so it is not guaranteed to be perfectly continuous. Exact bisection
    tolerance is preferred, but a result within
    BISECTION_ACCEPTANCE_TOLERANCE is accepted because the complete maneuver
    is validated separately.

    The search is performed over:

        [0, max_del_phi]

    By default, max_del_phi is 2*pi.

    Returns:
        del_phi_magnitude, final_x, final_y

    Raises:
        ManeuverPlanningError:
            If no usable root or near-root can be found in the configured
            search interval.
    """
    if component not in (0, 1):
        raise ValueError(
            "component must be either 0 or 1"
        )

    if tol <= 0.0:
        raise ValueError("tol must be positive")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    if change_threshold <= 0.0:
        raise ValueError(
            "change_threshold must be positive"
        )

    max_del_phi = _validate_max_del_phi(max_del_phi)

    # compute_turn() is relatively expensive. The scan and bisection can
    # evaluate identical angles, so cache all successful evaluations.
    cache: dict[float, tuple[float, float, float]] = {}

    def evaluate(
        del_phi: float,
    ) -> tuple[float, float, float]:
        del_phi = float(del_phi)

        if del_phi in cache:
            return cache[del_phi]

        x_final, y_final = compute_final_position(
            x_turn,
            y_turn,
            initial_heading,
            psi_2,
            del_phi,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate=roll_rate,
            roll_in=roll_in,
            first_heading_base=first_heading_base,
        )

        local_error = local_delta_xy(
            x_final,
            y_final,
            x2,
            y2,
            reference_course,
        )

        error = float(local_error[component])
        x_final = float(x_final)
        y_final = float(y_final)

        if not np.all(
            np.isfinite(
                [
                    error,
                    x_final,
                    y_final,
                ]
            )
        ):
            raise ManeuverPlanningError(
                "Bisection objective produced a non-finite result"
            )

        result = error, x_final, y_final
        cache[del_phi] = result

        return result

    # Search the complete configured interval, including max_del_phi.
    scan_angles = np.linspace(
        0.0,
        max_del_phi,
        BISECTION_SCAN_POINTS,
    )

    best_result: tuple[float, float, float, float] | None = None
    previous_result: tuple[float, float, float, float] | None = None
    bracket: tuple[float, float, float, float] | None = None

    for angle_value in scan_angles:
        angle = float(angle_value)

        try:
            error, x_final, y_final = evaluate(angle)
        except ManeuverPlanningError:
            # Never bracket across an invalid simulation region.
            previous_result = None
            continue

        result = (
            angle,
            error,
            x_final,
            y_final,
        )

        if (
            best_result is None
            or abs(error) < abs(best_result[1])
        ):
            best_result = result

        # Return the first root or near-root satisfying the requested
        # tolerance.
        if abs(error) <= tol:
            return angle, x_final, y_final

        if previous_result is not None:
            (
                previous_angle,
                previous_error,
                _,
                _,
            ) = previous_result

            if previous_error * error < 0.0:
                bracket = (
                    previous_angle,
                    previous_error,
                    angle,
                    error,
                )
                break

        previous_result = result

    if bracket is None:
        # A discretized objective can touch zero without crossing it, or
        # jump across the ideal root. Accept a sufficiently close sampled
        # result and let _validate_final_result() make the final decision.
        if (
            best_result is not None
            and abs(best_result[1])
            <= BISECTION_ACCEPTANCE_TOLERANCE
        ):
            return (
                best_result[0],
                best_result[2],
                best_result[3],
            )

        best_error = (
            abs(best_result[1])
            if best_result is not None
            else float("inf")
        )

        raise ManeuverPlanningError(
            "Unable to find a usable bisection root in "
            f"[0, {max_del_phi:.9g}]; "
            f"best absolute error was {best_error:.6g}"
        )

    low, error_low, high, error_high = bracket

    error_at_low, x_at_low, y_at_low = evaluate(low)
    error_at_high, x_at_high, y_at_high = evaluate(high)

    if abs(error_at_low) <= abs(error_at_high):
        best_angle = low
        best_error = error_at_low
        best_x = x_at_low
        best_y = y_at_low
    else:
        best_angle = high
        best_error = error_at_high
        best_x = x_at_high
        best_y = y_at_high

    previous_mid = None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)

        error_mid, x_mid, y_mid = evaluate(mid)

        if abs(error_mid) < abs(best_error):
            best_angle = mid
            best_error = error_mid
            best_x = x_mid
            best_y = y_mid

        if abs(error_mid) <= tol:
            return mid, x_mid, y_mid

        if abs(high - low) <= change_threshold:
            break

        if (
            previous_mid is not None
            and abs(mid - previous_mid) <= change_threshold
        ):
            break

        if error_low * error_mid <= 0.0:
            high = mid
            error_high = error_mid
        else:
            low = mid
            error_low = error_mid

        previous_mid = mid

    # The objective may be discontinuous because the simulated endpoint can
    # change by one integration step. Accept only a near-root consistent
    # with the complete maneuver's position tolerance.
    if abs(best_error) > BISECTION_ACCEPTANCE_TOLERANCE:
        raise ManeuverPlanningError(
            "Bisection did not produce a usable near-root; "
            f"best absolute error was {abs(best_error):.6g}"
        )

    return best_angle, best_x, best_y


def bisection_method_x(
    x_turn,
    y_turn,
    initial_heading,
    psi_2,
    reference_course,
    g,
    bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    x2,
    y2,
    roll_rate,
    tol=0.01,
    max_iter=100,
    change_threshold=1e-7,
    roll_in=0.0,
    first_heading_base=None,
    max_del_phi=DEFAULT_MAX_DEL_PHI,
):
    """
    Find del_phi so the final lateral error is approximately zero.

    The search interval is [0, max_del_phi].
    """
    return _bisection_method(
        0,
        x_turn,
        y_turn,
        initial_heading,
        psi_2,
        reference_course,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        x2,
        y2,
        roll_rate,
        tol=tol,
        max_iter=max_iter,
        change_threshold=change_threshold,
        roll_in=roll_in,
        first_heading_base=first_heading_base,
        max_del_phi=max_del_phi,
    )


def bisection_method_y(
    x_turn,
    y_turn,
    initial_heading,
    psi_2,
    reference_course,
    g,
    bank_angle,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
    x2,
    y2,
    roll_rate,
    tol=0.01,
    max_iter=100,
    change_threshold=1e-7,
    roll_in=0.0,
    first_heading_base=None,
    max_del_phi=DEFAULT_MAX_DEL_PHI,
):
    """
    Find del_phi so the final longitudinal error is approximately zero.

    The search interval is [0, max_del_phi].
    """
    return _bisection_method(
        1,
        x_turn,
        y_turn,
        initial_heading,
        psi_2,
        reference_course,
        g,
        bank_angle,
        V_TAS,
        V_w,
        theta_wa,
        dt,
        aero,
        x2,
        y2,
        roll_rate,
        tol=tol,
        max_iter=max_iter,
        change_threshold=change_threshold,
        roll_in=roll_in,
        first_heading_base=first_heading_base,
        max_del_phi=max_del_phi,
    )


def _compute_straight_segment(
    x_start,
    y_start,
    course,
    distance,
    V_TAS,
    V_w,
    theta_wa,
    dt,
    aero,
):
    """Construct and validate a nonnegative straight segment."""
    if not np.isfinite(distance):
        raise ManeuverPlanningError(
            "Straight distance is non-finite"
        )

    if distance < -DISTANCE_TOLERANCE:
        raise ManeuverPlanningError(
            f"Straight distance is negative: {distance:.6g}"
        )

    distance = max(0.0, float(distance))

    DYN = compute_straight(
        x_start,
        y_start,
        course,
        V_TAS,
        V_w,
        theta_wa,
        distance,
        dt,
        aero,
    )

    return _validate_trajectory(
        DYN,
        "straight segment",
    )


def _validate_final_result(
    DYN,
    del_phi,
    x_target,
    y_target,
    target_yaw,
    max_del_phi=DEFAULT_MAX_DEL_PHI,
):
    """
    Check final position, heading, trajectory values, and bend angle.

    The accepted bend-angle magnitude is:

        0 <= abs(del_phi) <= max_del_phi
    """
    try:
        DYN = _validate_trajectory(
            DYN,
            "complete maneuver",
        )
    except ManeuverPlanningError:
        return False

    try:
        max_del_phi = _validate_max_del_phi(max_del_phi)
    except ValueError:
        return False

    if not np.isfinite(del_phi):
        return False

    final_x = float(DYN[-1, 1])
    final_y = float(DYN[-1, 2])
    final_yaw = float(DYN[-1, 7])

    final_distance = float(
        np.hypot(
            final_x - x_target,
            final_y - y_target,
        )
    )

    final_heading_error = abs(
        _angle_error(
            final_yaw,
            target_yaw,
        )
    )

    bend_angle_magnitude = abs(float(del_phi))

    bad_angle = not (
        0.0
        <= bend_angle_magnitude
        <= max_del_phi + ANGLE_LIMIT_TOLERANCE
    )

    return bool(
        np.isfinite(final_distance)
        and np.isfinite(final_heading_error)
        and not bad_angle
        and final_distance <= POSITION_TOLERANCE
        and final_heading_error <= HEADING_TOLERANCE
    )


def _empty_result():
    """Return a standard failed-planning result."""
    return (
        np.empty((0, 0), dtype=float),
        float("nan"),
        False,
    )


def _try_attempt(
    attempt: Callable[[bool], tuple[np.ndarray, float, bool]],
    turn_right: bool,
):
    """Run one handedness attempt and convert planning failures to failure."""
    try:
        return attempt(turn_right)
    except ManeuverPlanningError:
        return _empty_result()


def SBB_BBS_maneuver(
    waypoints,
    constants,
    aero,
    roll_rate,
    max_del_phi=DEFAULT_MAX_DEL_PHI,
):
    """
    Plan either a forward SBB or BBS maneuver.

    The first handedness is selected from the waypoint geometry. If that
    solution fails validation, the opposite handedness is attempted.

    Args:
        waypoints:
            Sequence containing:

                x1, y1, x2, y2, chi_1, chi_2

        constants:
            Sequence containing:

                g, V_TAS, bank_angle, V_w, theta_wa, dt

        aero:
            Aerodynamic model passed to the trajectory simulators.

        roll_rate:
            Positive aircraft roll-rate magnitude.

        max_del_phi:
            Maximum additional bend-angle magnitude searched by the
            bisection solver. Defaults to 2*pi.

    Returns:
        DYN:
            Complete trajectory.

        del_phi:
            Signed additional bend angle. Positive denotes the positive-bank
            handedness and negative denotes the negative-bank handedness.

        success:
            True when position, heading, and geometry checks pass.
    """
    max_del_phi = _validate_max_del_phi(max_del_phi)

    x1, y1, x2, y2, chi_1, chi_2 = map(
        float,
        waypoints,
    )

    g, V_TAS, bank_angle0, V_w, theta_wa, dt = map(
        float,
        constants,
    )

    bank_angle_magnitude = abs(bank_angle0)

    if not np.all(
        np.isfinite(
            [
                x1,
                y1,
                x2,
                y2,
                chi_1,
                chi_2,
                g,
                V_TAS,
                bank_angle_magnitude,
                V_w,
                theta_wa,
                dt,
                roll_rate,
            ]
        )
    ):
        raise ValueError(
            "Maneuver inputs must all be finite"
        )

    if bank_angle_magnitude <= 0.0:
        raise ValueError(
            "bank_angle must be nonzero"
        )

    if V_TAS <= 0.0:
        raise ValueError(
            "V_TAS must be positive"
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive"
        )

    if roll_rate <= 0.0:
        raise ValueError(
            "roll_rate must be positive"
        )

    psi_1 = float(
        yaw_from_course(
            chi_1,
            V_TAS,
            V_w,
            theta_wa,
        )
    )

    psi_2 = float(
        yaw_from_course(
            chi_2,
            V_TAS,
            V_w,
            theta_wa,
        )
    )

    goal_lateral, _ = local_delta_xy(
        x1,
        y1,
        x2,
        y2,
        chi_1,
    )

    first_turn_right = goal_lateral >= 0.0

    def solve_delta(
        bank_angle,
        reference_course,
    ):
        return bisection_method_x(
            x1,
            y1,
            psi_1,
            psi_2,
            reference_course,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            x2,
            y2,
            roll_rate=roll_rate,
            roll_in=0.0,
            # Forward construction ends its first bend around psi_2.
            first_heading_base=psi_2,
            max_del_phi=max_del_phi,
        )

    def build_bends(
        x_start,
        y_start,
        start_heading,
        bank_angle,
        signed_del_phi,
    ):
        intermediate_heading = psi_2 + signed_del_phi

        return _compute_two_bends(
            x_start,
            y_start,
            start_heading,
            psi_2,
            intermediate_heading,
            g,
            bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate,
            roll_in=0.0,
        )

    def attempt(turn_right: bool):
        bank_angle = (
            bank_angle_magnitude
            if turn_right
            else -bank_angle_magnitude
        )

        # First test whether translating the two bends along the initial
        # course produces an SBB construction.
        del_phi_magnitude, x_mid, y_mid = solve_delta(
            bank_angle,
            chi_1,
        )

        signed_del_phi = (
            del_phi_magnitude
            if turn_right
            else -del_phi_magnitude
        )

        _, initial_course_distance = local_delta_xy(
            x_mid,
            y_mid,
            x2,
            y2,
            chi_1,
        )

        if initial_course_distance >= -DISTANCE_TOLERANCE:
            # -------------------------------------------------------------
            # SBB: Straight - Bend - Bend
            # -------------------------------------------------------------
            DYN_straight = _compute_straight_segment(
                x1,
                y1,
                chi_1,
                initial_course_distance,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
            )

            x_straight = float(DYN_straight[-1, 1])
            y_straight = float(DYN_straight[-1, 2])
            heading_straight = float(DYN_straight[-1, 7])

            DYN_turn_1, DYN_turn_2 = build_bends(
                x_straight,
                y_straight,
                heading_straight,
                bank_angle,
                signed_del_phi,
            )

            DYN = _stack_segments(
                DYN_straight,
                DYN_turn_1,
                DYN_turn_2,
            )

        else:
            # -------------------------------------------------------------
            # BBS: Bend - Bend - Straight
            # -------------------------------------------------------------
            del_phi_magnitude, _, _ = solve_delta(
                bank_angle,
                chi_2,
            )

            signed_del_phi = (
                del_phi_magnitude
                if turn_right
                else -del_phi_magnitude
            )

            DYN_turn_1, DYN_turn_2 = build_bends(
                x1,
                y1,
                psi_1,
                bank_angle,
                signed_del_phi,
            )

            x_turn = float(DYN_turn_2[-1, 1])
            y_turn = float(DYN_turn_2[-1, 2])
            heading_turn = float(DYN_turn_2[-1, 7])

            course_turn = float(
                course_from_yaw(
                    heading_turn,
                    V_TAS,
                    V_w,
                    theta_wa,
                )
            )

            lateral_error, straight_distance = local_delta_xy(
                x_turn,
                y_turn,
                x2,
                y2,
                course_turn,
            )

            if abs(lateral_error) > POSITION_TOLERANCE:
                raise ManeuverPlanningError(
                    "BBS bend endpoint has excessive lateral error"
                )

            DYN_straight = _compute_straight_segment(
                x_turn,
                y_turn,
                course_turn,
                straight_distance,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
            )

            DYN = _stack_segments(
                DYN_turn_1,
                DYN_turn_2,
                DYN_straight,
            )

        success = _validate_final_result(
            DYN,
            signed_del_phi,
            x2,
            y2,
            psi_2,
            max_del_phi=max_del_phi,
        )

        return DYN, signed_del_phi, success

    first_result = _try_attempt(
        attempt,
        first_turn_right,
    )

    if first_result[2]:
        return first_result

    second_result = _try_attempt(
        attempt,
        not first_turn_right,
    )

    if second_result[2]:
        return second_result

    # Return the better failed trajectory when both handednesses produced a
    # trajectory. Otherwise return the standard empty failure.
    candidates = [
        result
        for result in (
            first_result,
            second_result,
        )
        if result[0].size > 0
    ]

    if not candidates:
        return _empty_result()

    return min(
        candidates,
        key=lambda result: float(
            np.hypot(
                result[0][-1, 1] - x2,
                result[0][-1, 2] - y2,
            )
        ),
    )


def SBB_BBS_inv_maneuver(
    waypoints,
    constants,
    aero,
    roll_rate,
    max_del_phi=DEFAULT_MAX_DEL_PHI,
):
    """
    Plan an inverse SBB/BBS maneuver.

    In the inverse construction, the first bend uses the opposite bank from
    the selected handedness:

        0 -> -bank_angle -> 0

    If the first handedness fails, the opposite handedness is attempted.

    Args:
        waypoints:
            Sequence containing:

                x1, y1, x2, y2, chi_1, chi_2

        constants:
            Sequence containing:

                g, V_TAS, bank_angle, V_w, theta_wa, dt

        aero:
            Aerodynamic model passed to the trajectory simulators.

        roll_rate:
            Positive aircraft roll-rate magnitude.

        max_del_phi:
            Maximum additional bend-angle magnitude searched by the
            bisection solver. Defaults to 2*pi.

    Returns:
        DYN, signed_del_phi, success
    """
    max_del_phi = _validate_max_del_phi(max_del_phi)

    x1, y1, x2, y2, chi_1, chi_2 = map(
        float,
        waypoints,
    )

    g, V_TAS, bank_angle0, V_w, theta_wa, dt = map(
        float,
        constants,
    )

    bank_angle_magnitude = abs(bank_angle0)

    if not np.all(
        np.isfinite(
            [
                x1,
                y1,
                x2,
                y2,
                chi_1,
                chi_2,
                g,
                V_TAS,
                bank_angle_magnitude,
                V_w,
                theta_wa,
                dt,
                roll_rate,
            ]
        )
    ):
        raise ValueError(
            "Maneuver inputs must all be finite"
        )

    if bank_angle_magnitude <= 0.0:
        raise ValueError(
            "bank_angle must be nonzero"
        )

    if V_TAS <= 0.0:
        raise ValueError(
            "V_TAS must be positive"
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive"
        )

    if roll_rate <= 0.0:
        raise ValueError(
            "roll_rate must be positive"
        )

    psi_1 = float(
        yaw_from_course(
            chi_1,
            V_TAS,
            V_w,
            theta_wa,
        )
    )

    psi_2 = float(
        yaw_from_course(
            chi_2,
            V_TAS,
            V_w,
            theta_wa,
        )
    )

    goal_lateral, _ = local_delta_xy(
        x1,
        y1,
        x2,
        y2,
        chi_1,
    )

    first_turn_right = goal_lateral >= 0.0

    def solve_delta(
        selected_bank_angle,
        reference_course,
    ):
        first_bank_angle = -selected_bank_angle

        return bisection_method_x(
            x1,
            y1,
            psi_1,
            psi_2,
            reference_course,
            g,
            first_bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            x2,
            y2,
            roll_rate=roll_rate,
            roll_in=0.0,
            # Inverse construction offsets its first bend around psi_1.
            first_heading_base=psi_1,
            max_del_phi=max_del_phi,
        )

    def build_bends(
        x_start,
        y_start,
        start_heading,
        selected_bank_angle,
        signed_del_phi,
    ):
        first_bank_angle = -selected_bank_angle
        intermediate_heading = psi_1 - signed_del_phi

        return _compute_two_bends(
            x_start,
            y_start,
            start_heading,
            psi_2,
            intermediate_heading,
            g,
            first_bank_angle,
            V_TAS,
            V_w,
            theta_wa,
            dt,
            aero,
            roll_rate,
            roll_in=0.0,
        )

    def attempt(turn_right: bool):
        selected_bank_angle = (
            bank_angle_magnitude
            if turn_right
            else -bank_angle_magnitude
        )

        # First determine whether an end straight along chi_2 produces BBS.
        del_phi_magnitude, x_mid, y_mid = solve_delta(
            selected_bank_angle,
            chi_2,
        )

        signed_del_phi = (
            del_phi_magnitude
            if turn_right
            else -del_phi_magnitude
        )

        _, final_course_distance = local_delta_xy(
            x_mid,
            y_mid,
            x2,
            y2,
            chi_2,
        )

        if final_course_distance >= -DISTANCE_TOLERANCE:
            # -------------------------------------------------------------
            # BBS: Bend - Bend - Straight
            # -------------------------------------------------------------
            DYN_turn_1, DYN_turn_2 = build_bends(
                x1,
                y1,
                psi_1,
                selected_bank_angle,
                signed_del_phi,
            )

            x_turn = float(DYN_turn_2[-1, 1])
            y_turn = float(DYN_turn_2[-1, 2])
            heading_turn = float(DYN_turn_2[-1, 7])

            course_turn = float(
                course_from_yaw(
                    heading_turn,
                    V_TAS,
                    V_w,
                    theta_wa,
                )
            )

            lateral_error, straight_distance = local_delta_xy(
                x_turn,
                y_turn,
                x2,
                y2,
                course_turn,
            )

            if abs(lateral_error) > POSITION_TOLERANCE:
                raise ManeuverPlanningError(
                    "Inverse BBS endpoint has excessive lateral error"
                )

            DYN_straight = _compute_straight_segment(
                x_turn,
                y_turn,
                course_turn,
                straight_distance,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
            )

            DYN = _stack_segments(
                DYN_turn_1,
                DYN_turn_2,
                DYN_straight,
            )

        else:
            # -------------------------------------------------------------
            # SBB: Straight - Bend - Bend
            # -------------------------------------------------------------
            del_phi_magnitude, x_mid, y_mid = solve_delta(
                selected_bank_angle,
                chi_1,
            )

            signed_del_phi = (
                del_phi_magnitude
                if turn_right
                else -del_phi_magnitude
            )

            lateral_error, straight_distance = local_delta_xy(
                x_mid,
                y_mid,
                x2,
                y2,
                chi_1,
            )

            if abs(lateral_error) > POSITION_TOLERANCE:
                raise ManeuverPlanningError(
                    "Inverse SBB endpoint has excessive lateral error"
                )

            DYN_straight = _compute_straight_segment(
                x1,
                y1,
                chi_1,
                straight_distance,
                V_TAS,
                V_w,
                theta_wa,
                dt,
                aero,
            )

            x_straight = float(DYN_straight[-1, 1])
            y_straight = float(DYN_straight[-1, 2])
            heading_straight = float(DYN_straight[-1, 7])

            DYN_turn_1, DYN_turn_2 = build_bends(
                x_straight,
                y_straight,
                heading_straight,
                selected_bank_angle,
                signed_del_phi,
            )

            DYN = _stack_segments(
                DYN_straight,
                DYN_turn_1,
                DYN_turn_2,
            )

        success = _validate_final_result(
            DYN,
            signed_del_phi,
            x2,
            y2,
            psi_2,
            max_del_phi=max_del_phi,
        )

        return DYN, signed_del_phi, success

    first_result = _try_attempt(
        attempt,
        first_turn_right,
    )

    if first_result[2]:
        return first_result

    second_result = _try_attempt(
        attempt,
        not first_turn_right,
    )

    if second_result[2]:
        return second_result

    candidates = [
        result
        for result in (
            first_result,
            second_result,
        )
        if result[0].size > 0
    ]

    if not candidates:
        return _empty_result()

    return min(
        candidates,
        key=lambda result: float(
            np.hypot(
                result[0][-1, 1] - x2,
                result[0][-1, 2] - y2,
            )
        ),
    )
