"""
fault_injection.py

Module tạo và tiêm các dạng dị thường nhân tạo (Synthetic Fault Injection)
vào chuỗi tín hiệu cảm biến phục vụ mô phỏng độ suy thoái thực tế.
"""

from typing import Sequence, Any
import numpy as np


DEFAULT_FAULT_METADATA = {
    "has_rare_fault": False,
    "fault_type": None,
    "fault_sensor": None,
    "fault_sensor_index": None,
    "fault_start": None,
    "fault_end": None,
    "fault_duration": 0,
    "fault_magnitude": 0.0,
}


def _sensor_scale(
    trajectory: np.ndarray,
    sensor_index: int,
) -> float:
    """
    Ước lượng độ rung lắc tự nhiên (nhiễu) giữa 2 chu kỳ liền kề.
    Dùng để tạo nhiễu trắng cho Plateau.
    """
    values = trajectory[:, sensor_index]
    if len(values) < 3:
        return 1.0

    differences = np.diff(values)
    scale = np.median(np.abs(differences))
    if scale <= 1e-8:
        scale = np.std(differences)
    if scale <= 1e-8:
        scale = 1.0
    return float(scale)


def inject_plateau(
    trajectory: np.ndarray,
    start: int,
    end: int,
    sensor_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Tạo lỗi kẹt tín hiệu (Plateau) tại giá trị start + thêm nhiễu nhẹ.
    """
    result = trajectory.copy()
    values = result[:, sensor_index]
    scale = _sensor_scale(trajectory, sensor_index)

    plateau_value = values[start]
    noise = rng.normal(loc=0.0, scale=scale, size=end - start)
    result[start:end, sensor_index] = plateau_value + noise
    return result


def inject_drop(
    trajectory: np.ndarray,
    start: int,
    end: int,
    sensor_index: int,
    magnitude: float,
) -> np.ndarray:
    """
    Tạo lỗi sụt giá trị đột ngột (Drop) theo dải đo (Span) của cảm biến.
    """
    result = trajectory.copy()
    values = trajectory[:, sensor_index]

    sensor_span = np.ptp(values)
    if sensor_span <= 1e-4:
        sensor_span = np.abs(np.mean(values)) if np.abs(np.mean(values)) > 1e-4 else 1.0

    offset = magnitude * sensor_span
    duration = end - start
    if duration <= 0:
        return result

    result[start:end, sensor_index] -= offset
    return result


def inject_drift(
    trajectory: np.ndarray,
    start: int,
    end: int,
    sensor_index: int,
    magnitude: float,
) -> np.ndarray:
    """
    Tạo lỗi trôi tín hiệu tuyến tính (Drift) theo dải đo (Span) của cảm biến.
    """
    result = trajectory.copy()
    values = trajectory[:, sensor_index]

    sensor_span = np.ptp(values)
    if sensor_span <= 1e-4:
        sensor_span = np.abs(np.mean(values)) if np.abs(np.mean(values)) > 1e-4 else 1.0

    offset = magnitude * sensor_span
    duration = end - start
    if duration <= 0:
        return result

    drift = np.linspace(0.0, -offset, duration)
    result[start:end, sensor_index] += drift
    return result


def inject_rare_fault(
    trajectory: np.ndarray,
    cutoff: int,
    config: Any,
    rng: np.random.Generator,
    sensor_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Điều phối tiêm ngẫu nhiên 1 lỗi vào 1 cảm biến trước điểm cutoff.
    """
    result = trajectory.copy()
    n_sensors = result.shape[1]

    max_duration = min(config.rare_fault_max_duration, cutoff)

    # Trả về metadata mặc định an toàn nếu cutoff quá ngắn
    if max_duration < 2:
        return result, DEFAULT_FAULT_METADATA.copy()

    fault_type = rng.choice(config.rare_fault_types)
    duration = int(rng.integers(2, max_duration + 1))

    # Lấy mẫu điểm bắt đầu từ Cycle 60 cho tới trước cutoff
    earliest_start = min(config.rare_fault_start_cycle_min, max(0, cutoff - duration - 1))
    latest_start = max(earliest_start, cutoff - duration)
    start = int(rng.integers(earliest_start, latest_start + 1))
    end = start + duration

    sensor_index = int(rng.integers(0, n_sensors))
    magnitude = float(rng.uniform(config.rare_fault_magnitude_min, config.rare_fault_magnitude_max))

    if fault_type == "plateau":
        result = inject_plateau(result, start, end, sensor_index, rng)
        logged_magnitude = 0.0
    elif fault_type == "drop":
        result = inject_drop(result, start, end, sensor_index, magnitude)
        logged_magnitude = magnitude
    elif fault_type == "drift":
        result = inject_drift(result, start, end, sensor_index, magnitude)
        logged_magnitude = magnitude
    else:
        raise ValueError(f"Unknown rare fault type: {fault_type}")

    fault_sensor_name = (
        sensor_names[sensor_index]
        if sensor_names is not None and sensor_index < len(sensor_names)
        else f"sensor_{sensor_index}"
    )

    metadata = {
        "has_rare_fault": True,
        "fault_type": fault_type,
        "fault_sensor": fault_sensor_name,
        "fault_sensor_index": sensor_index,
        "fault_start": start,
        "fault_end": end,
        "fault_duration": duration,
        "fault_magnitude": logged_magnitude,
    }

    return result, metadata
