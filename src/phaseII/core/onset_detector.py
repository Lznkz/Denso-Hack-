"""
onset_detector.py -- KHỐI 2: PHÁT HIỆN ĐIỂM BẮT ĐẦU SUY THOÁI (t_onset)

Nguyên lý (không đổi, nhưng input H_t giờ đến từ HealthIndicatorSmoother):
  1. Chặn dự báo RUL khi máy đang chạy bình thường.
  2. Ngưỡng thống kê từ t_0 chu kỳ khỏe mạnh ban đầu:
       th_d = mean(H_{1:t_0}) + k_sigma * std(H_{1:t_0})
     Vì H_raw = Z_base - Z_t, baseline H ≈ 0 khi khỏe, nên th_d ≈ k_sigma * std_H.
  3. Quy tắc 5-consecutive persistent trigger:
       count_t = count_{t-1} + 1  nếu H_t > th_d
                 0                 nếu H_t <= th_d
  4. Khi count_t == consecutive_required:
       t_onset = t - (consecutive_required - 1)
       H_onset = H_{t_onset}
"""

from __future__ import annotations
import numpy as np

from src.phaseII.core.schema import OnsetConfig


class DegradationOnsetDetector:
    """Bộ phát hiện điểm bắt đầu suy thoái cơ học (Two-phase: Shock Detection + Settling)."""

    def __init__(self, config: OnsetConfig | None = None):
        self.config = config or OnsetConfig()
        self.th_d: float | None = None
        self.count: int = 0
        self.t_onset: int | None = None
        self.h_onset: float | None = None
        self.is_degraded: bool = False
        self.history_h: list[float] = []

        # Biến trạng thái cho cú rơi (Shock/Transition Phase)
        self.shock_detected: bool = False
        self.settling_count: int = 0

    def fit_threshold(self, h_healthy_initial: list[float] | np.ndarray) -> float:
        """Tính ngưỡng th_d từ t_0 measurement khỏe mạnh đầu tiên."""
        arr = np.asarray(h_healthy_initial, dtype=np.float32)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        raw_th = mean_val + self.config.k_sigma * std_val
        min_th = getattr(self.config, "min_th_d", 0.8)
        self.th_d = float(max(raw_th, min_th))
        return self.th_d

    def step(self, t: int, h_val: float) -> tuple[bool, int | None, float | None]:
        """Cập nhật trạng thái tại chu kỳ t (1-based index).

        Parameters
        ----------
        t : int
            Measurement cycle hiện tại (1-based).
        h_val : float
            H_t sau Rolling Median + EWMA từ HealthIndicatorSmoother.

        Returns
        -------
        is_degraded : bool
        t_onset : int | None
        h_onset : float | None
        """
        self.history_h.append(h_val)

        # Tích lũy trong t0 chu kỳ đầu
        if self.th_d is None:
            if len(self.history_h) >= self.config.t0_initial:
                self.fit_threshold(self.history_h[:self.config.t0_initial])
            else:
                return False, None, None

        # Nếu đã xác nhận suy thoái từ trước
        if self.is_degraded:
            return True, self.t_onset, self.h_onset

        # Giai đoạn 1: Phát hiện bất thường ban đầu (vượt ngưỡng th_d liên tiếp)
        if not self.shock_detected:
            if h_val > self.th_d:
                self.count += 1
                if self.count >= self.config.consecutive_required:
                    self.shock_detected = True
            else:
                self.count = 0
            return False, None, None

        # Giai đoạn 2: Đang trong cú rơi dốc -> Chờ ổn định (Settling Phase)
        # Tính đạo hàm cục bộ qua khoảng cách 3 chu kỳ
        if len(self.history_h) >= 4:
            dh = abs(self.history_h[-1] - self.history_h[-4]) / 3.0
            if dh < self.config.max_settling_slope:
                self.settling_count += 1
                if self.settling_count >= self.config.settling_window:
                    # Đã ổn định đáy cú rơi -> Kích hoạt Steady Onset tại đây!
                    self.is_degraded = True
                    self.t_onset = t
                    self.h_onset = h_val
                    return True, self.t_onset, self.h_onset
            else:
                self.settling_count = 0

        return False, None, None

    def reset(self) -> None:
        """Khởi tạo lại detector cho bearing mới."""
        self.th_d = None
        self.count = 0
        self.t_onset = None
        self.h_onset = None
        self.is_degraded = False
        self.history_h = []
        self.shock_detected = False
        self.settling_count = 0
