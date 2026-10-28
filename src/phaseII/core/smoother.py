"""
smoother.py -- KHỐI 1: CHUYỂN ĐỔI SCALAR LATENT Z -> DAMAGE HEALTH INDEX H

Thay thế OperatingCodebook cũ. Không dùng W, không KMeans.

Nguyên lý:
  Phase I mới trích xuất Z ∈ ℝ (z_dim=1) với ràng buộc dZ/dt ≤ 0:
    - Vòng bi khỏe: Z cao (gần Z_base)
    - Vòng bi suy thoái: Z giảm đơn điệu

  Ánh xạ sang Damage Index H ≥ 0 tăng dần:
    H_raw = Z_base - Z_t      (≥ 0 nếu Z_t ≤ Z_base)

  Làm mượt EWMA:
    H_t = alpha * H_raw + (1 - alpha) * H_{t-1}

  Z_base = mean(Z_{1:t0}) được tính từ t0 chu kỳ khỏe mạnh đầu tiên
  của chính vòng bi target (online initialization).
"""

from __future__ import annotations
from collections import deque
import numpy as np

from src.phaseII.core.schema import SmoothingConfig


class HealthIndicatorSmoother:
    """Bộ chuyển đổi Z -> Damage Health Index H với Rolling Median + EWMA.

    Không cần W, không KMeans, không Codebook.
    Chỉ nhận scalar Z từ Phase I encoder.
    """

    def __init__(self, config: SmoothingConfig | None = None):
        self.config = config or SmoothingConfig()
        self.z_base: float | None = None
        self.prev_h: float | None = None
        self._raw_buffer: deque[float] = deque(maxlen=self.config.median_window)

    def init_baseline(self, z_init: np.ndarray) -> float:
        """Khởi tạo Z_base từ t0 chu kỳ đầu tiên của vòng bi target.

        Parameters
        ----------
        z_init : np.ndarray
            Shape (t0,) hoặc (t0, 1) — các giá trị Z trong t0 chu kỳ đầu.

        Returns
        -------
        z_base : float
            Mean của z_init, dùng làm điểm tham chiếu khỏe mạnh.
        """
        arr = np.asarray(z_init, dtype=np.float64).flatten()
        if len(arr) == 0:
            raise ValueError("z_init không được rỗng.")
        self.z_base = float(np.mean(arr))
        # Reset filter states
        self.prev_h = None
        self._raw_buffer.clear()
        return self.z_base

    def step(self, z_t: float) -> tuple[float, float]:
        """Tính H_raw và H_obs (sau Rolling Median + EWMA) cho một measurement.

        Parameters
        ----------
        z_t : float
            Scalar latent Z tại chu kỳ t.

        Returns
        -------
        h_raw : float
            Damage raw = Z_base - Z_t  (tăng khi Z giảm)
        h_val : float
            Damage sau Rolling Median + EWMA (làm mượt hai tầng)
        """
        if self.z_base is None:
            raise RuntimeError("Phải gọi init_baseline() trước khi step().")

        h_raw = float(self.z_base) - float(np.squeeze(z_t))
        self._raw_buffer.append(h_raw)

        # Tầng 1: Lọc trung vị (Rolling Median) dập tắt gai nhiễu
        h_median = float(np.median(self._raw_buffer))

        # Tầng 2: EWMA làm mịn
        alpha = self.config.ewma_alpha
        if self.prev_h is None:
            h_val = h_median
        else:
            h_val = alpha * h_median + (1.0 - alpha) * self.prev_h

        self.prev_h = h_val
        return h_raw, h_val

    def reset(self) -> None:
        """Đặt lại trạng thái cho vòng bi mới (giữ z_base nếu cần tái dùng)."""
        self.z_base = None
        self.prev_h = None
        self._raw_buffer.clear()

    def reset_ewma_only(self) -> None:
        """Chỉ reset filter state, giữ nguyên z_base đã fit."""
        self.prev_h = None
        self._raw_buffer.clear()
