import csv
import numpy as np
from pathlib import Path


def dynamic_split_2d(path_xy: np.ndarray, threshold_m: float = 10.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split a 2D path into dense turn/maneuver segments and sparse straight segments.

    Returns list of (x_array, y_array).
    """
    path_xy = np.asarray(path_xy, dtype=float)
    if len(path_xy) < 2:
        raise ValueError("Path must contain at least two points.")

    x = path_xy[:, 0]
    y = path_xy[:, 1]

    subarrays = []

    current_x = [x[0]]
    current_y = [y[0]]

    for k in range(1, len(x)):
        dist = np.hypot(x[k] - x[k - 1], y[k] - y[k - 1])

        if dist > threshold_m:
            if len(current_x) > 1:
                subarrays.append((np.asarray(current_x), np.asarray(current_y)))

            subarrays.append(
                (
                    np.asarray([x[k - 1], x[k]], dtype=float),
                    np.asarray([y[k - 1], y[k]], dtype=float),
                )
            )

            current_x = [x[k]]
            current_y = [y[k]]
        else:
            current_x.append(x[k])
            current_y.append(y[k])

    if len(current_x) > 1:
        subarrays.append((np.asarray(current_x), np.asarray(current_y)))

    return subarrays


def fill_sparse_points(segment: tuple[np.ndarray, np.ndarray], threshold_m: float = 10.0, step_m: float = 2.0):
    x, y = segment
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    new_x = [x[0]]
    new_y = [y[0]]

    for k in range(1, len(x)):
        dist = np.hypot(x[k] - x[k - 1], y[k] - y[k - 1])

        if dist > threshold_m:
            n_insert = int(dist // step_m)
            xs = np.linspace(x[k - 1], x[k], n_insert + 2)[1:-1]
            ys = np.linspace(y[k - 1], y[k], n_insert + 2)[1:-1]
            new_x.extend(xs)
            new_y.extend(ys)

        new_x.append(x[k])
        new_y.append(y[k])

    return np.asarray(new_x), np.asarray(new_y)


def extrapolate_end(segment: tuple[np.ndarray, np.ndarray], extension_factor: float = 0.5):
    x, y = segment
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return x, y

    total_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    extension_len = total_len * extension_factor

    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    seg_len = np.hypot(dx, dy)

    if seg_len < 1e-9:
        return x, y

    ux = dx / seg_len
    uy = dy / seg_len

    n_points = max(int(extension_len // seg_len), 1)

    extra_x = [x[-1] + ux * seg_len * i for i in range(1, n_points + 1)]
    extra_y = [y[-1] + uy * seg_len * i for i in range(1, n_points + 1)]

    return np.hstack([x, extra_x]), np.hstack([y, extra_y])


def find_closest_point(
    aircraft_position_xy: np.ndarray,
    x_path: np.ndarray,
    y_path: np.ndarray,
    previous_index: int,
    search_radius: int = 20,
) -> tuple[np.ndarray, int]:
    """
    Find closest path point, searching near previous_index for stability.
    """
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)

    start = max(0, previous_index - search_radius)
    end = min(len(x_path), previous_index + search_radius + 1)

    dx = x_path[start:end] - aircraft_position_xy[0]
    dy = y_path[start:end] - aircraft_position_xy[1]
    d2 = dx**2 + dy**2

    local_idx = int(np.argmin(d2))
    idx = start + local_idx

    return np.array([x_path[idx], y_path[idx]], dtype=float), idx


def find_l1_point_by_straight_distance(
    closest_index: int,
    x_path: np.ndarray,
    y_path: np.ndarray,
    l1_distance_m: float,
) -> tuple[np.ndarray, int]:
    """
    Find the first future path sample whose straight-line distance from closest point
    is at least l1_distance_m.
    """
    x0 = x_path[closest_index]
    y0 = y_path[closest_index]

    for idx in range(closest_index, len(x_path)):
        d = np.hypot(x_path[idx] - x0, y_path[idx] - y0)
        if d >= l1_distance_m:
            return np.array([x_path[idx], y_path[idx]], dtype=float), idx

    return np.array([x_path[-1], y_path[-1]], dtype=float), len(x_path) - 1
    
def find_l1_point_by_path_distance(
    closest_index: int,
    x_path: np.ndarray,
    y_path: np.ndarray,
    l1_distance_m: float,
) -> tuple[np.ndarray, int]:
    """
    Find the point located l1_distance_m ahead of closest_index along the path.

    Returns:
        l1_point:
            Interpolated XY point on the path.
        l1_index:
            Index of the segment endpoint at or after the L1 point.
    """
    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)

    n = len(x_path)

    if n == 0:
        raise ValueError("Path is empty.")

    if len(y_path) != n:
        raise ValueError("x_path and y_path must have the same length.")

    closest_index = int(np.clip(closest_index, 0, n - 1))

    if l1_distance_m <= 0.0:
        return np.array([x_path[closest_index], y_path[closest_index]], dtype=float), closest_index

    if closest_index >= n - 1:
        return np.array([x_path[-1], y_path[-1]], dtype=float), n - 1

    remaining = float(l1_distance_m)

    for idx in range(closest_index + 1, n):
        x_prev = x_path[idx - 1]
        y_prev = y_path[idx - 1]
        x_next = x_path[idx]
        y_next = y_path[idx]

        seg_dx = x_next - x_prev
        seg_dy = y_next - y_prev
        seg_len = float(np.hypot(seg_dx, seg_dy))

        if seg_len <= 1e-9:
            continue

        if remaining <= seg_len:
            ratio = remaining / seg_len
            x_l1 = x_prev + ratio * seg_dx
            y_l1 = y_prev + ratio * seg_dy
            return np.array([x_l1, y_l1], dtype=float), idx

        remaining -= seg_len

    return np.array([x_path[-1], y_path[-1]], dtype=float), n - 1


def stack_with_next_segment(
    subarrays: list[tuple[np.ndarray, np.ndarray]],
    index: int,
    current_x: np.ndarray,
    current_y: np.ndarray,
):
    """
    Current segment plus next straight segment if available.
    """
    stacked_x = current_x
    stacked_y = current_y

    if index + 1 < len(subarrays):
        next_x, next_y = subarrays[index + 1]

        if len(next_x) == 2:
            next_x, next_y = fill_sparse_points((next_x, next_y))
            stacked_x = np.hstack([current_x, next_x])
            stacked_y = np.hstack([current_y, next_y])

    return stacked_x, stacked_y

def export_path_to_csv(path_xy, output_file: Path, path_name: str = "optimal") -> None:
    """
    Export an Nx2 path array to CSV.

    Parameters
    ----------
    path_xy:
        Array-like object with columns [x, y], typically UTM coordinates.
    output_file:
        Destination CSV file.
    path_name:
        Name stored in the CSV file.
    """
    path_xy = np.asarray(path_xy)

    if path_xy.ndim != 2 or path_xy.shape[1] < 2:
        raise ValueError(
            f"Expected path with shape Nx2, got {path_xy.shape}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "index",
            "path",
            "x_utm_m",
            "y_utm_m",
        ])

        for index, point in enumerate(path_xy):
            writer.writerow([
                index,
                path_name,
                float(point[0]),
                float(point[1]),
            ])

    print(f"{path_name.capitalize()} path exported to: {output_file}")
