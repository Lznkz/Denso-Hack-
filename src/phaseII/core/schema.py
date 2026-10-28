"""
schema.py -- CẤU TRÚC DỮ LIỆU CHO HỆ THỐNG DỰ BÁO SUY THOÁI LAI
(Nonlinear Wiener Process with Adaptive Drift & Diffusion)

Cập nhật kiến trúc để tích hợp với Phase I mới (z_dim=1, scalar latent,
negative gradient constraint, không cần W trong Phase II).

Chứa:
  1. Cấu hình các khối (Smoothing, Onset, Wiener).
  2. Enum trạng thái máy (HealthState) và trạng thái dự báo (RULStatus).
  3. Value object FailureThreshold với metadata nguồn gốc và coordinate scale.
  4. FPTResult + SolverDiagnostics cho kết quả giải tích Wald CDF.
  5. Kết quả dự báo RULPrediction với canonical fields.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Cấu hình các khối
# ---------------------------------------------------------------------------

@dataclass
class SmoothingConfig:
    """Cấu hình Khối 1: Lọc mượt hai tầng (Rolling Median + EWMA)."""
    median_window: int = 11       # Cửa sổ Rolling Median dập tắt gai nhiễu biên độ ±2
    ewma_alpha: float = 0.08      # EWMA tốc độ chậm để loại bỏ hoàn toàn dốc giả


@dataclass
class OnsetConfig:
    """Cấu hình Khối 2: Nhận diện Onset hai pha (Vượt ngưỡng + Ổn định sau cú rơi)."""
    t0_initial: int = 20          # Số measurement khỏe mạnh đầu để thiết lập baseline
    k_sigma: float = 3.0          # th_d = mean(H_{1:t0}) + k_sigma * std(H_{1:t0})
    consecutive_required: int = 5 # Quy tắc 5 chu kỳ liên tiếp H_t > th_d
    settling_window: int = 4      # Số chu kỳ liên tiếp tốc độ rơi phải chậm lại
    max_settling_slope: float = 0.12  # |dH/dt| < 0.12 đơn vị/chu kỳ thì coi là bắt đầu pha mòn ổn định
    min_th_d: float = 0.8         # Sàn ngưỡng tối thiểu để tránh kích hoạt bởi rung lắc ngẫu nhiên lúc khỏe


@dataclass
class WienerConfig:
    """Cấu hình Khối 3, 4, 5: Mô hình Wiener tuyến tính cho pha mòn ổn định."""
    gamma: float = 1.0            # Đổi từ 1.75 về 1.0 (Linear Wiener) vì pha mòn ổn định là đường thẳng
    mu_prior: float = 0.022       # Ước tính ban đầu phù hợp với độ dốc mòn thực tế
    sigma_prior: float = 0.02     # Độ bất định ban đầu nhỏ, không cho phép drift nhảy vọt
    sigma_B_initial: float = 0.20 # Hệ số khuếch tán duy trì độ mở khoảng tin cậy 90%
    min_mu: float = 1e-4          # Chặn dưới drift để tránh chia cho 0
    k_innovation: float = 3.0     # Hệ số Quality Gate 3-sigma Innovation Filter


# ---------------------------------------------------------------------------
# Enums trạng thái máy và dự báo
# ---------------------------------------------------------------------------

class HealthState(str, Enum):
    """Trạng thái sức khỏe máy theo quan sát vật lý.
    Axis độc lập với RULStatus — máy có thể DEGRADED + SOLVER_FAILED cùng lúc.
    """
    CALIBRATING               = "CALIBRATING"            # t < t0_initial
    HEALTHY                   = "HEALTHY"                # Sau calibration, chưa onset
    DEGRADED                  = "DEGRADED"               # Sau onset, tích lũy suy thoái
    FAILURE_THRESHOLD_REACHED = "FAILURE_THRESHOLD_REACHED"  # Y_acc >= D_failure_rel


class RULStatus(str, Enum):
    """Trạng thái availability của dự báo RUL — độc lập với HealthState.

    Mapping điển hình:
      CALIBRATING  + NOT_CALIBRATED      (chưa đủ t0 measurements)
      HEALTHY      + WAITING_FOR_ONSET   (đã calibrate, chưa onset)
      DEGRADED     + OK                  (quantiles hợp lệ)
      DEGRADED     + SOLVER_FAILED       (solver không converge)
      FAILURE...   + THRESHOLD_REACHED   (RUL = 0)
    """
    NOT_CALIBRATED    = "NOT_CALIBRATED"
    WAITING_FOR_ONSET = "WAITING_FOR_ONSET"
    OK                = "OK"
    THRESHOLD_REACHED = "THRESHOLD_REACHED"
    SOLVER_FAILED     = "SOLVER_FAILED"


# ---------------------------------------------------------------------------
# FailureThreshold value object
# ---------------------------------------------------------------------------

class ThresholdScale(str, Enum):
    """Coordinate system của giá trị ngưỡng."""
    ABSOLUTE_Z        = "ABSOLUTE_Z"          # Ngưỡng trên thang Z gốc của Phase I
    ABSOLUTE_HI       = "ABSOLUTE_HI"         # Ngưỡng trong H-space tuyệt đối
    RELATIVE_TO_ONSET = "RELATIVE_TO_ONSET"   # Y-space tính từ onset (Y=0)


class ThresholdSource(str, Enum):
    """Nguồn gốc của ngưỡng hỏng hóc."""
    HISTORICAL_FLEET_MEDIAN   = "HISTORICAL_FLEET_MEDIAN"   # median(Z_end của train bearings)
    ENGINEERING_SPEC          = "ENGINEERING_SPEC"
    EXPERIMENTAL_CALIBRATION  = "EXPERIMENTAL_CALIBRATION"
    MANUAL                    = "MANUAL"


@dataclass(frozen=True)
class FailureThreshold:
    """Value object bất biến mang ngưỡng hỏng hóc và metadata nguồn gốc.

    Notes
    -----
    Khi scale=RELATIVE_TO_ONSET: value = H_onset_corrected - H_failure_rel
      dùng để so sánh với Y_acc.

    Invariant:
      delta_d = value_absolute - hi_state
              = value_relative  - y_acc
    """
    value: float                  # Giá trị ngưỡng (hữu hạn; dương đối với H-space/rel, cho phép âm với ABSOLUTE_Z)
    scale: ThresholdScale
    source: ThresholdSource
    method: str = ""
    calibration_n: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError(
                f"FailureThreshold.value phải là số hữu hạn, nhận {self.value!r}"
            )
        if self.scale in (ThresholdScale.ABSOLUTE_HI, ThresholdScale.RELATIVE_TO_ONSET):
            if self.value <= 0:
                raise ValueError(
                    f"FailureThreshold.value trong thang {self.scale.value} phải là số dương, nhận {self.value!r}"
                )


# ---------------------------------------------------------------------------
# FPT Solver result types
# ---------------------------------------------------------------------------

class FPTSolverStatus(str, Enum):
    """Trạng thái kết quả giải tích Wald CDF."""
    OK             = "OK"
    BRACKET_FAILED = "BRACKET_FAILED"
    SOLVER_FAILED  = "SOLVER_FAILED"


@dataclass
class SolverDiagnostics:
    """Thông tin debug khi FPT solver thất bại."""
    nu: float
    lambd: float
    p: float
    bracket_low: float = 0.0
    bracket_high: float = 0.0
    error_message: str = ""


@dataclass
class FPTResult:
    """Kết quả giải tích FPT quantiles từ Wald CDF inversion."""
    q05: Optional[float]
    q50: Optional[float]
    q95: Optional[float]
    solver_status: FPTSolverStatus
    diagnostics: Optional[SolverDiagnostics] = None


# ---------------------------------------------------------------------------
# RULPrediction — output chính của pipeline
# ---------------------------------------------------------------------------

@dataclass
class RULPrediction:
    """Kết quả dự báo RUL và chỉ số trạng thái cho một chu kỳ measurement.

    HI Flow (Phase I mới, z_dim=1):
      z_raw       : Z_t thô từ Phase I encoder (giảm dần khi thoái hóa)
      hi_raw      : H_raw = Z_base - Z_t  (tăng dần, ≥ 0)
      hi_observed : H_t sau EWMA
      hi_state    : H_onset + Y_acc  (sau Quality Gate)

    Coordinate invariant (sau onset):
      D_failure_rel - y_acc  ==  D_failure_abs_H - hi_state

    Quantile semantics:
      rul_q05/q50/q95 là phân vị predictive distribution của model.
      Không phải statistical confidence bound.
    """
    cycle: int
    tau: Optional[int]
    health_state: HealthState
    rul_status: RULStatus

    # --- Raw Z từ Phase I ---
    z_raw: float                     # Z_t thô từ encoder (giảm khi thoái hóa)

    # --- HI semantics ---
    hi_raw: float                    # H_raw = Z_base - Z_t (chưa EWMA)
    hi_observed: float               # H_t sau EWMA
    hi_state: float                  # H_onset + Y_acc (sau Quality Gate)

    # --- Wiener process state ---
    y_acc: float                     # Mức suy thoái tích lũy Y_tau (Y_0=0 tại onset)
    mu_tau: float                    # Kỳ vọng drift đã cập nhật Bayes
    sigma_b_tau: float               # Hệ số khuếch tán thích nghi

    # --- RUL quantiles ---
    rul_q05: Optional[float]
    rul_q50: Optional[float]
    rul_q95: Optional[float]

    # --- Metadata ---
    is_outlier: bool = False
    d_failure_abs_h: float = 0.0     # D_failure tuyệt đối trong H-space
    th_onset: float = 0.0
    solver_diagnostics: Optional[SolverDiagnostics] = None

    # --- Backward compat ---
    @property
    def is_degraded(self) -> bool:
        return self.health_state in (
            HealthState.DEGRADED,
            HealthState.FAILURE_THRESHOLD_REACHED,
        )

    @property
    def rul_median(self) -> Optional[float]:
        return self.rul_q50

    @property
    def rul_lower(self) -> Optional[float]:
        return self.rul_q05

    @property
    def rul_upper(self) -> Optional[float]:
        return self.rul_q95
