"""
schema.py -- CẤU TRÚC DỮ LIỆU ĐỊNH VỊ CẢM BIẾN LỖI ("Ở ĐÂU" - SENSOR ATTRIBUTION)

Định nghĩa các dataclass thuần túy chứa kết quả định lượng độ lệch cảm biến,
tách biệt hoàn toàn khỏi thuật toán tính toán và tầng hiển thị.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SensorDeviation:
    """Thông tin định lượng độ lệch của 1 sensor cụ thể so với trạng thái chuẩn kỳ vọng."""
    sensor_name: str
    sensor_index: int
    actual_raw_mean: float       # Giá trị trung bình đo đạc thực tế từ cảm biến thô (X_window)
    actual_model_mean: float     # Giá trị ước lượng sau khi khử nhiễu qua mô hình (x_hat_current)
    recon_mean: float            # Giá trị chuẩn so sánh (x_hat_baseline nếu suy thoái, x_hat_current nếu đột biến)
    raw_diff: float              # Độ lệch số học trung bình (mean deviation)
    rms_diff: float              # Độ lệch hiệu dụng RMS (không bị triệt tiêu khi cảm biến dao động rung lắc)
    percent_diff: float          # Độ lệch phần trăm so với chuẩn: (raw_diff / |recon_mean|) * 100
    direction: str               # "HIGHER" (vọt cao), "LOWER" (sụt giảm), "OSCILLATING" (dao động rung giật)
    error_score: float           # Sai số chuẩn hóa (đã chia cho phương sai nền của chính sensor đó)
    contribution_pct: float      # Tỷ lệ % đóng góp vào tổng sai số bất thường của cả hệ thống
    fault_nature: str            # Bản chất lỗi: "DOT_BIEN" hoặc "SUY_THOAI"


@dataclass
class SensorAttributionResult:
    """Kết quả định vị tổng thể của cả hệ thống tại một cửa sổ thời gian (window/cycle)."""
    cycle: int | None
    is_alert: bool                                  # Kế thừa trung thực từ Tầng 1 (không tự ý phủ quyết)
    anomaly_type: str                               # "NORMAL", "SUDDEN_FAULT", "DEGRADATION"
    primary_sensor: str | None                      # Cảm biến có tỷ lệ bất thường cao nhất (Top-1 Culprit)
    top_sensors: list[tuple[str, float, float]]     # Danh sách [(tên_sensor, error_score, contribution_pct)]
    sensor_scores: dict[str, float]                 # {sensor_name: error_score}
    sensor_contributions: dict[str, float]          # {sensor_name: contribution_pct}
    deviations: dict[str, SensorDeviation]          # Chi tiết định lượng từng sensor
    total_anomaly_score: float                      # Tổng sai số chuẩn hóa của toàn bộ sensor
    stage1_triggered: bool = True                   # Trạng thái cờ từ tầng "Có hay không"
