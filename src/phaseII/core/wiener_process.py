"""
wiener_process.py -- KHỐI 3, 4, 5: MÔ HÌNH HÓA WIENER PHI TUYẾN & BỘ LỌC BAYESIAN O(1)

Không thay đổi toán học cốt lõi. Input H_t giờ từ HealthIndicatorSmoother
thay vì Codebook. Bất biến hi_state = hi_onset + y_acc được bảo toàn.

Nguyên lý:
  Khối 3: Trục thời gian tau = t - t_onset, hàm gia tốc mỏi Paris Law:
           DeltaLambda_tau = tau^gamma - (tau - 1)^gamma
  Khối 4: Quality Gate (3-sigma Innovation Outlier Filter):
           Kỳ vọng: DeltaY_hat = mu_{tau-1} * DeltaLambda_tau
           Var[DeltaY] = sigma_{tau-1}^2 * DeltaLambda^2 + sigma_B^2 * DeltaLambda
           Nếu hợp lệ: DeltaY = max(Delta_y_raw, 0)  (đơn điệu)
           Nếu outlier: DeltaY = max(DeltaY_hat, 0)
  Khối 5: Tích lũy Y_tau = Y_{tau-1} + DeltaY_tau  (Y_0=0)
           Cập nhật đệ quy Bayesian O(1) cho mu_tau, sigma_tau^2, sigma_B^2

State semantics:
  hi_observed : H_t sau EWMA (từ HealthIndicatorSmoother). Input của tracker.
  hi_state    : H_onset + Y_acc. Có thể ≠ hi_observed vì Quality Gate.

  Invariant (sau init_onset):
    hi_state_t = self.hi_onset + self.y_acc
"""

from __future__ import annotations
import numpy as np

from src.phaseII.core.schema import WienerConfig


class NonlinearWienerBayesianTracker:
    """Bộ theo dõi suy thoái Wiener phi tuyến với cập nhật Bayesian thích nghi O(1)."""

    def __init__(self, config: WienerConfig | None = None):
        self.config = config or WienerConfig()
        self.reset()

    def reset(self) -> None:
        """Khởi tạo lại trạng thái cho bearing mới."""
        self.tau: int = 0
        self.y_acc: float = 0.0
        self.prev_h: float | None = None
        self.hi_onset: float = 0.0

        self.mu_tau: float = self.config.mu_prior
        self.sigma_tau_sq: float = self.config.sigma_prior ** 2
        self.sigma_b_sq: float = self.config.sigma_B_initial ** 2

    def init_onset(self, hi_onset: float) -> None:
        """Thiết lập điểm mốc onset (tau = 0).

        Parameters
        ----------
        hi_onset : float
            H_observed tại t_onset (sau EWMA). Mốc tham chiếu cho invariant.
        """
        self.hi_onset = hi_onset
        self.tau = 0
        self.y_acc = 0.0
        self.prev_h = hi_onset

    def step(self, hi_observed: float) -> tuple[int, float, float, float, float, bool]:
        """Cập nhật một bước đo đạc mới sau onset (tau >= 1).

        Parameters
        ----------
        hi_observed : float
            H_t tại chu kỳ hiện tại (sau EWMA, trước Quality Gate).

        Returns
        -------
        tau : int
        y_acc : float
        mu_tau : float
        sigma_b_tau : float
        hi_state : float
        is_outlier : bool
        """
        self.tau += 1
        tau = self.tau
        gamma = self.config.gamma

        # Khối 3: Bước nhảy phi tuyến Paris Law
        delta_lambda = float(tau ** gamma - (tau - 1.0) ** gamma)

        if self.prev_h is None:
            self.prev_h = hi_observed
        delta_y_raw = float(hi_observed - self.prev_h)
        self.prev_h = hi_observed

        # Khối 4: Quality Gate 3-sigma Innovation Filter
        mu_prev = self.mu_tau
        sigma_sq_prev = self.sigma_tau_sq
        sigma_b_sq_prev = self.sigma_b_sq

        delta_y_hat = mu_prev * delta_lambda
        var_delta_y = sigma_sq_prev * (delta_lambda ** 2) + sigma_b_sq_prev * delta_lambda
        std_delta_y = float(np.sqrt(max(var_delta_y, 1e-12)))

        # Khối 4: Quality Gate Innovation Filter chống "Hiệu ứng bánh cóc" (Ratchet Effect)
        # Chỉ lọc sốc tải nhảy vọt bất thường (positive shock spike).
        # Khi tốc độ quan sát chậm lại hoặc bằng phẳng, chấp nhận delta_y_raw để Bayesian drift thích nghi giảm.
        innovation = delta_y_raw - delta_y_hat
        is_outlier = innovation > (self.config.k_innovation * std_delta_y)

        if not is_outlier:
            # Hợp lệ: đảm bảo đơn điệu (không cho phép "tự phục hồi")
            delta_y = max(delta_y_raw, 0.0)
        else:
            # Sốc tải đột ngột: gán kỳ vọng
            delta_y = max(delta_y_hat, 0.0)

        # Khối 5a: Tích lũy đơn điệu
        self.y_acc += delta_y

        # Khối 5b: Cập nhật đệ quy Bayesian O(1)
        if not is_outlier:
            inv_sigma_tau_sq = (1.0 / sigma_sq_prev) + (delta_lambda / sigma_b_sq_prev)
            self.sigma_tau_sq = 1.0 / max(inv_sigma_tau_sq, 1e-12)

            mu_new = self.sigma_tau_sq * (
                (mu_prev / sigma_sq_prev) + (delta_y / sigma_b_sq_prev)
            )
            self.mu_tau = float(max(mu_new, self.config.min_mu))

            # Thích nghi sigma_B: Cập nhật đệ quy mẫu Bayesian chuẩn theo tau,
            # duy trì sàn khuếch tán sigma_B_initial để bảo toàn độ phủ khoảng tin cậy 90%
            residual = delta_y - self.mu_tau * delta_lambda
            sq_residual = (residual ** 2) / max(delta_lambda, 1e-12)
            sigma_b_sq_new = ((tau - 1.0) / tau) * sigma_b_sq_prev + (1.0 / tau) * sq_residual
            self.sigma_b_sq = max(float(sigma_b_sq_new), self.config.sigma_B_initial ** 2)

        hi_state = self.hi_onset + self.y_acc
        return tau, self.y_acc, self.mu_tau, float(np.sqrt(self.sigma_b_sq)), hi_state, is_outlier
