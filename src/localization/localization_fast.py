"""
localization_fast.py -- TẦNG 2: ĐỊNH VỊ CẢM BIẾN LỖI ("Ở ĐÂU" - SENSOR ATTRIBUTION)

Nguyên lý hoạt động:
  1. Lỗi đột biến (SUDDEN_FAULT):
     Sai số tái tạo dư: delta = X_window - X_hat_current
     -> Bắt chính xác cảm biến đứt cáp, kẹt cứng (plateau), sụt áp (drop).
  2. Lỗi suy thoái (DEGRADATION):
     Tái tạo đối chứng: delta = X_hat_current - X_hat_baseline
     Trong đó:
         X_hat_current  = Decoder(W_current, Z_current)
         X_hat_baseline = Decoder(W_current, Z_baseline)
     -> Triệt tiêu 100% biến thiên điều kiện tải W và nhiễu ngẫu nhiên.
  3. Cơ chế Cổng chặn (Event-Triggered):
     Nếu is_anomaly_triggered = False: Thoát tức thì với độ trễ ~0 ms.
"""

from __future__ import annotations
from typing import Sequence, Any
import numpy as np
import torch

from src.localization.schema import SensorDeviation, SensorAttributionResult


class FastReconstructionAttributor:
    """
    Bộ định vị Tầng 2: Real-time Sensor Attribution.
    Tập trung 100% vào tính toán thuật toán, độc lập với tập dữ liệu (dataset-agnostic).
    """

    def __init__(
        self,
        sensor_names: Sequence[str],
        baseline_stds: np.ndarray | None = None,
        eps: float = 1e-6,
    ):
        self.sensor_names = list(sensor_names)
        self.n_sensors = len(self.sensor_names)
        self.baseline_means = np.asarray(baseline_stds, dtype=np.float32) if baseline_stds is not None else None
        self.baseline_recon_stds: np.ndarray | None = None
        self.eps = eps
        self.z_baseline: np.ndarray | None = None

    def set_z_baseline(self, z_baseline: np.ndarray | Any) -> None:
        """Thiết lập trực tiếp vector latent Z chuẩn khỏe mạnh."""
        if hasattr(z_baseline, "detach"):
            z_baseline = z_baseline.detach().cpu().numpy()
        self.z_baseline = np.asarray(z_baseline, dtype=np.float32).reshape(1, -1)

    def fit_baseline_from_healthy_windows(
        self,
        healthy_x: np.ndarray,
        healthy_w: np.ndarray,
        model: Any,
        device: str = "cpu",
    ) -> None:
        """
        Ước lượng trạng thái chuẩn từ các cửa sổ khỏe mạnh ban đầu:
          1. Vector Z_baseline (chuẩn khỏe mạnh).
          2. Phương sai sai số Counterfactual Drift (để chuẩn hóa lỗi suy thoái).
          3. Phương sai sai số Reconstruction Residual (để chuẩn hóa lỗi đột biến).
        """
        model.eval()
        w_targets = healthy_w[:, -1, :] if healthy_w.ndim == 3 else healthy_w
        with torch.no_grad():
            x_t = torch.from_numpy(healthy_x).float().to(device)
            w_t = torch.from_numpy(w_targets).float().to(device)

            # 1. Trích xuất Z và chốt Z_baseline
            z_seq = model.encode(x_t).cpu().numpy()
            self.z_baseline = np.mean(z_seq, axis=0, keepdims=True).astype(np.float32)
            z_base_expanded = torch.from_numpy(
                np.repeat(self.z_baseline, len(healthy_x), axis=0)
            ).float().to(device)

            # 2. Phương sai sai số Counterfactual Drift nền (khi máy khỏe):
            x_hat_healthy = model.decode(w_t, z_base_expanded).cpu().numpy()
            diff_drift = (healthy_x - x_hat_healthy) ** 2
            self.baseline_means = np.mean(diff_drift, axis=(0, 1)) + self.eps

            # 3. Phương sai sai số Reconstruction Residual nền (để bắt đột biến):
            x_hat_recon = model.decode(w_t, torch.from_numpy(z_seq).float().to(device)).cpu().numpy()
            diff_recon = (healthy_x - x_hat_recon) ** 2
            self.baseline_recon_stds = np.mean(diff_recon, axis=(0, 1)) + self.eps

    def localize(
        self,
        x_window: np.ndarray | Any,
        w_window: np.ndarray | Any,
        model: Any,
        anomaly_type: str | None = None,
        is_anomaly_triggered: bool = True,
        cycle: int | None = None,
        top_k: int = 3,
        device: str = "cpu",
    ) -> SensorAttributionResult:
        """
        PHƯƠNG THỨC ĐỊNH VỊ CHÍNH (KẾT NỐI TẦNG 1 VÀ TẦNG 2).
        """
        # Cổng chặn: Nếu Tầng 1 chưa kích hoạt -> Bỏ qua toàn bộ (~0 ms)
        if not is_anomaly_triggered:
            return self._build_normal_result(cycle=cycle)

        # 1. Kiểm tra nghiêm ngặt anomaly_type (Fail-Fast)
        if anomaly_type not in ("SUDDEN_FAULT", "DEGRADATION"):
            raise ValueError(
                f"LỖI ĐỊNH VỊ: Khi Tầng 1 phát hiện bất thường (is_anomaly_triggered=True), "
                f"bắt buộc phải truyền anomaly_type là 'SUDDEN_FAULT' hoặc 'DEGRADATION'. "
                f"Nhận được: '{anomaly_type}'."
            )

        # 2. Kiểm tra nghiêm ngặt z_baseline
        if self.z_baseline is None:
            raise RuntimeError(
                "LỖI ĐỊNH VỊ: z_baseline chưa được thiết lập! "
                "Cần gọi fit_baseline_from_healthy_windows() từ các chu kỳ máy khỏe ban đầu "
                "hoặc set_z_baseline() trước khi thực hiện định vị."
            )

        x_arr = x_window.detach().cpu().numpy() if hasattr(x_window, "detach") else np.asarray(x_window)
        w_arr = w_window.detach().cpu().numpy() if hasattr(w_window, "detach") else np.asarray(w_window)

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        w_vec = w_arr[-1:] if w_arr.ndim == 2 else (w_arr.reshape(1, -1) if w_arr.ndim == 1 else w_arr)

        model.eval()
        with torch.no_grad():
            x_input = x_arr.copy()
            x_t = torch.from_numpy(x_input).unsqueeze(0).float().to(device) if x_input.ndim == 2 else torch.from_numpy(x_input).float().to(device)
            w_t = torch.from_numpy(w_vec).float().to(device)

            z_current = model.encode(x_t)
            x_hat_current = model.decode(w_t, z_current).squeeze(0).cpu().numpy()

            z_base_t = torch.from_numpy(self.z_baseline).float().to(device)
            x_hat_baseline = model.decode(w_t, z_base_t).squeeze(0).cpu().numpy()

        # Phân nhánh theo loại bất thường từ Tầng 1:
        if anomaly_type == "SUDDEN_FAULT":
            delta = x_arr - x_hat_current
            recon_target = x_hat_current
            norm_denoms = self.baseline_recon_stds
            fault_nature = "DOT_BIEN"
            target_name = "baseline_recon_stds"
        else:
            delta = x_hat_current - x_hat_baseline
            recon_target = x_hat_baseline
            norm_denoms = self.baseline_means
            fault_nature = "SUY_THOAI"
            target_name = "baseline_means"

        # 3. Kiểm tra nghiêm ngặt phương sai chuẩn hóa
        if norm_denoms is None or len(norm_denoms) != self.n_sensors:
            raise RuntimeError(
                f"LỖI ĐỊNH VỊ: Phương sai nền '{target_name}' chưa được tính toán! "
                "Cần gọi fit_baseline_from_healthy_windows() để học phương sai chuẩn hóa của từng sensor."
            )

        # Tính toán sai số chuẩn hóa và tỷ lệ đóng góp
        diff_sq = delta ** 2
        raw_sensor_errors = np.mean(diff_sq, axis=0)
        normalized_errors = raw_sensor_errors / norm_denoms

        total_err = float(np.sum(normalized_errors))
        if total_err > 1e-12:
            contributions = (normalized_errors / total_err) * 100.0
        else:
            contributions = np.zeros(self.n_sensors, dtype=np.float32)

        ranked_indices = np.argsort(contributions)[::-1]
        sensor_scores = {}
        sensor_contrib_dict = {}
        deviations = {}
        top_sensors = []

        for idx in ranked_indices:
            s_name = self.sensor_names[idx]
            score = float(normalized_errors[idx])
            contrib = float(contributions[idx])

            sensor_scores[s_name] = score
            sensor_contrib_dict[s_name] = contrib

            actual_raw = float(np.mean(x_arr[:, idx]))
            actual_model = float(np.mean(x_hat_current[:, idx]))
            recon_val = float(np.mean(recon_target[:, idx]))
            raw_diff_val = float(np.mean(delta[:, idx]))
            rms_diff_val = float(np.sqrt(np.mean(diff_sq[:, idx])))

            denom = abs(recon_val)
            pct_diff = (raw_diff_val / denom) * 100.0 if denom > self.eps else 0.0

            # Phân loại hướng lệch: HIGHER, LOWER hoặc OSCILLATING (dao động/rung lắc)
            if abs(raw_diff_val) >= 0.5 * rms_diff_val:
                direction = "HIGHER" if raw_diff_val >= 0 else "LOWER"
            elif rms_diff_val > self.eps:
                direction = "OSCILLATING"
            else:
                direction = "HIGHER" if raw_diff_val >= 0 else "LOWER"

            deviations[s_name] = SensorDeviation(
                sensor_name=s_name,
                sensor_index=idx,
                actual_raw_mean=actual_raw,
                actual_model_mean=actual_model,
                recon_mean=recon_val,
                raw_diff=raw_diff_val,
                rms_diff=rms_diff_val,
                percent_diff=pct_diff,
                direction=direction,
                error_score=score,
                contribution_pct=contrib,
                fault_nature=fault_nature,
            )

            if len(top_sensors) < top_k:
                top_sensors.append((s_name, score, contrib))

        primary_sensor = top_sensors[0][0] if len(top_sensors) > 0 else None

        return SensorAttributionResult(
            cycle=cycle,
            is_alert=is_anomaly_triggered,
            anomaly_type=anomaly_type,
            primary_sensor=primary_sensor,
            top_sensors=top_sensors,
            sensor_scores=sensor_scores,
            sensor_contributions=sensor_contrib_dict,
            deviations=deviations,
            total_anomaly_score=total_err,
            stage1_triggered=True,
        )

    def _build_normal_result(self, cycle: int | None = None) -> SensorAttributionResult:
        """Tạo kết quả an toàn mặc định khi tầng 1 chưa kích hoạt (~0 ms)."""
        return SensorAttributionResult(
            cycle=cycle,
            is_alert=False,
            anomaly_type="NORMAL",
            primary_sensor=None,
            top_sensors=[],
            sensor_scores={s: 0.0 for s in self.sensor_names},
            sensor_contributions={s: 0.0 for s in self.sensor_names},
            deviations={},
            total_anomaly_score=0.0,
            stage1_triggered=False,
        )
