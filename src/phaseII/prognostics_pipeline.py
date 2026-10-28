"""
prognostics_pipeline.py -- BỘ ĐIỀU PHỐI PIPELINE DỰ BÁO SUY THOÁI (PHASE II)

KIẾN TRÚC MỚI (tích hợp Phase I scalar latent Z):
  Phase I: SmartConv1DAutoencoder -> Z ∈ ℝ (z_dim=1)
           Ràng buộc: dZ/dt ≤ 0, vòng bi khỏe Z cao, suy thoái Z giảm.
  Phase II: Không dùng W (operating conditions) để bù tải.
            Z được ánh xạ thẳng thành Damage Index H.

PIPELINE CÁC KHỐI:
  Khối 1 (HealthIndicatorSmoother):
    H_raw = Z_base - Z_t  (Z_base = mean(Z_{1:t0}) của bearing target)
    H_t   = EWMA(H_raw)

  Khối 2 (DegradationOnsetDetector):
    th_d  = mean(H_{1:t0}) + 3*std(H_{1:t0})
    t_onset: 5 lần liên tiếp H_t > th_d

  Khối 3-5 (NonlinearWienerBayesianTracker):
    Accumulator Wiener phi tuyến Paris Law + Bayesian O(1) update

  Khối 6 (solve_fpt_quantiles):
    Giải phân vị RUL qua Wald CDF inversion (fail-closed)

NGƯỠNG HỎNG MỚI:
  Z_fail_abs = median(Z_end của các train bearings)   [offline, 1 lần]
  D_failure_H = Z_base - Z_fail_abs                  [trong H-space]
  D_failure_rel = D_failure_H - H_onset               [relative to onset, so sánh với Y_acc]

API PUBLIC:
  from_checkpoint(ckpt_path, train_bearing_data) -> PrognosticsPipeline
  pipeline.run_trajectory(z_seq) -> list[RULPrediction]
  pipeline.step(cycle, z_t) -> RULPrediction
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import logging
import numpy as np
import torch

from src.phaseI.core.model import SmartConv1DAutoencoder
from src.phaseII.core.schema import (
    SmoothingConfig,
    OnsetConfig,
    WienerConfig,
    FailureThreshold,
    ThresholdScale,
    ThresholdSource,
    HealthState,
    RULStatus,
    RULPrediction,
    FPTSolverStatus,
)
from src.phaseII.core.smoother import HealthIndicatorSmoother
from src.phaseII.core.onset_detector import DegradationOnsetDetector
from src.phaseII.core.wiener_process import NonlinearWienerBayesianTracker
from src.phaseII.core.fpt_solver import compute_fleet_failure_threshold, solve_fpt_quantiles

logger = logging.getLogger(__name__)


def _encode_bearing(
    model: SmartConv1DAutoencoder,
    X_spev: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    """Encode X_hi_spev -> Z array shape (N,1) qua Phase I encoder.

    Parameters
    ----------
    model : SmartConv1DAutoencoder
    X_spev : np.ndarray  shape (N, n_sensors, n_buckets)
    x_mean : np.ndarray  shape (1, n_sensors, n_buckets)
    x_std  : np.ndarray  shape (1, n_sensors, n_buckets)
    """
    X_log = np.log1p(X_spev.astype(np.float32))
    X_norm = (X_log - x_mean) / np.where(x_std > 0, x_std, 1.0)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X_norm.astype(np.float32)))
    return z.numpy()  # (N, 1)


def _compute_z_from_checkpoint(
    ckpt: dict,
    X_spev: np.ndarray,
) -> np.ndarray:
    """Build model từ checkpoint và encode một bearing."""
    mc = ckpt["model_config"]
    norm = ckpt["normalization"]
    model = SmartConv1DAutoencoder(
        n_sensors=mc["n_sensors"],
        window_len=mc["n_buckets"],
        w_dim=mc["w_dim"],
        z_dim=mc["z_dim"],
        hidden_channels=tuple(mc["hidden_channels"]),
        kernel_size=mc.get("kernel_size", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_mean = np.array(norm["x_mean"], dtype=np.float32)
    x_std  = np.array(norm["x_std"], dtype=np.float32)
    return _encode_bearing(model, X_spev, x_mean, x_std)


class PrognosticsPipeline:
    """Hệ thống dự báo RUL cho vòng bi dựa trên scalar latent Z từ Phase I.

    Khởi tạo theo một trong hai cách:
      1. PrognosticsPipeline(z_fail_abs=...) — dùng ngưỡng tính sẵn
      2. PrognosticsPipeline.from_checkpoint(ckpt_path, train_data) — tự tính

    Sau đó:
      preds = pipeline.run_trajectory(z_seq)  # z_seq: array (N,) hoặc (N,1)
    """

    def __init__(
        self,
        z_fail_abs: float,
        smoothing_cfg: SmoothingConfig | None = None,
        onset_cfg: OnsetConfig | None = None,
        wiener_cfg: WienerConfig | None = None,
    ):
        """
        Parameters
        ----------
        z_fail_abs : float
            Ngưỡng hỏng tuyệt đối trên thang Z của Phase I.
            Tính bởi compute_fleet_failure_threshold(train_z_end_values).
        """
        self.z_fail_abs = float(z_fail_abs)
        self.smoothing_cfg = smoothing_cfg or SmoothingConfig()
        self.onset_cfg = onset_cfg or OnsetConfig()
        self.wiener_cfg = wiener_cfg or WienerConfig()

        # Khởi tạo các sub-components
        self.smoother = HealthIndicatorSmoother(self.smoothing_cfg)
        self.onset_detector = DegradationOnsetDetector(self.onset_cfg)
        self.wiener_tracker = NonlinearWienerBayesianTracker(self.wiener_cfg)

        # State cho lifecycle của một bearing
        self._history_h: list[float] = []     # lịch sử hi_observed
        self._history_z: list[float] = []     # lịch sử z_t
        self._onset_initialized: bool = False
        self._onset_exceeded_threshold: bool = False
        self._threshold_abs: FailureThreshold | None = None   # D_failure trong H-space
        self._threshold_rel: FailureThreshold | None = None   # D_failure - H_onset

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        train_bearing_data: dict[str, np.ndarray],
        smoothing_cfg: SmoothingConfig | None = None,
        onset_cfg: OnsetConfig | None = None,
        wiener_cfg: WienerConfig | None = None,
    ) -> "PrognosticsPipeline":
        """Factory: xây dựng pipeline từ checkpoint .pt và dữ liệu train.

        Parameters
        ----------
        ckpt_path : str | Path
            Đường dẫn tới file .pt từ Phase I (ví dụ: F:/topk7/topk7/thien_gay_topk7.pt).
        train_bearing_data : dict[str, np.ndarray]
            Key = tên bearing (e.g. "B10"), Value = X_hi_spev array shape (N, 2, 64).
            Dùng để tính Z_end của mỗi train bearing -> fleet threshold.

        Returns
        -------
        PrognosticsPipeline
        """
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        mc = ckpt["model_config"]
        norm = ckpt["normalization"]
        x_mean = np.array(norm["x_mean"], dtype=np.float32)
        x_std  = np.array(norm["x_std"], dtype=np.float32)

        model = SmartConv1DAutoencoder(
            n_sensors=mc["n_sensors"],
            window_len=mc["n_buckets"],
            w_dim=mc["w_dim"],
            z_dim=mc["z_dim"],
            hidden_channels=tuple(mc["hidden_channels"]),
            kernel_size=mc.get("kernel_size", 3),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Tính Z_end cho mỗi train bearing
        z_end_values: list[float] = []
        for bearing_id, X_spev in train_bearing_data.items():
            z_arr = _encode_bearing(model, X_spev, x_mean, x_std)
            z_end = float(z_arr[-1, 0])
            z_end_values.append(z_end)
            logger.info("Train bearing %s: Z_end=%.4f", bearing_id, z_end)

        z_fail_abs = compute_fleet_failure_threshold(z_end_values)
        logger.info("z_failure_abs (fleet median) = %.4f", z_fail_abs)

        return cls(
            z_fail_abs=z_fail_abs,
            smoothing_cfg=smoothing_cfg,
            onset_cfg=onset_cfg,
            wiener_cfg=wiener_cfg,
        )

    @staticmethod
    def encode_bearing_from_checkpoint(
        ckpt_path: str | Path,
        X_spev: np.ndarray,
    ) -> np.ndarray:
        """Helper tiện ích: nạp checkpoint Phase I và encode một bearing X_spev -> Z.

        Parameters
        ----------
        ckpt_path : str | Path
            Đường dẫn checkpoint .pt (ví dụ: F:/topk7/topk7/thien_gay_topk7.pt)
        X_spev : np.ndarray
            Mảng log-SPEV hoặc raw SPEV shape (N, n_sensors, n_buckets)

        Returns
        -------
        z : np.ndarray  shape (N, 1)
        """
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        return _compute_z_from_checkpoint(ckpt, X_spev)

    def reset_bearing(self) -> None:
        """Khởi tạo lại pipeline cho một bearing mới (giữ nguyên z_fail_abs)."""
        self.smoother.reset()
        self.onset_detector.reset()
        self.wiener_tracker.reset()
        self._history_h = []
        self._history_z = []
        self._onset_initialized = False
        self._onset_exceeded_threshold = False
        self._threshold_abs = None
        self._threshold_rel = None

    # ------------------------------------------------------------------
    # STEP: xử lý một measurement
    # ------------------------------------------------------------------

    def _finalize_calibration(self, z_base: float) -> None:
        """Sau t0 measurements: thiết lập D_failure trong H-space.

        D_failure_H = Z_base - Z_fail_abs  (> 0 nếu Z_base > Z_fail_abs)
        """
        d_fail_h = z_base - self.z_fail_abs
        if not (np.isfinite(d_fail_h) and d_fail_h > 0):
            logger.warning(
                "D_failure_H = %.4f <= 0. z_base=%.4f z_fail_abs=%.4f. "
                "Threshold sẽ là NOT_CALIBRATED.",
                d_fail_h, z_base, self.z_fail_abs
            )
            return
        self._threshold_abs = FailureThreshold(
            value=d_fail_h,
            scale=ThresholdScale.ABSOLUTE_HI,
            source=ThresholdSource.HISTORICAL_FLEET_MEDIAN,
            method="median_z_end_fleet",
            calibration_n=len(self._history_z),
        )

    def step(self, cycle: int, z_t: float) -> RULPrediction:
        """Xử lý một measurement và trả RULPrediction.

        Parameters
        ----------
        cycle : int
            Measurement index (1-based).
        z_t : float
            Scalar latent Z từ Phase I encoder tại chu kỳ cycle.
        """
        z_t = float(z_t)
        self._history_z.append(z_t)
        t0 = self.onset_cfg.t0_initial

        # ----------------------------------------------------------------
        # Giai đoạn CALIBRATING: chưa đủ t0 để khởi tạo baseline
        # ----------------------------------------------------------------
        if len(self._history_z) <= t0:
            # Khởi tạo smoother baseline sau khi đủ t0
            if len(self._history_z) == t0:
                z_base = self.smoother.init_baseline(np.array(self._history_z))
                self._finalize_calibration(z_base)
                # Replay t0 measurements qua smoother để khởi động EWMA
                for z_prev in self._history_z:
                    h_raw, h_val = self.smoother.step(z_prev)
                    self._history_h.append(h_val)
                # Chạy onset_detector qua lịch sử H đã tích lũy
                for t_idx, h_v in enumerate(self._history_h, start=1):
                    self.onset_detector.step(t_idx, h_v)
                # Lấy hi_observed của bước hiện tại
                hi_raw_cur = float(self.smoother.z_base) - z_t
                hi_observed = self._history_h[-1]
            else:
                hi_raw_cur = 0.0
                hi_observed = 0.0

            return RULPrediction(
                cycle=cycle,
                tau=None,
                health_state=HealthState.CALIBRATING,
                rul_status=RULStatus.NOT_CALIBRATED,
                z_raw=z_t,
                hi_raw=hi_raw_cur,
                hi_observed=hi_observed,
                hi_state=hi_observed,
                y_acc=0.0,
                mu_tau=self.wiener_cfg.mu_prior,
                sigma_b_tau=self.wiener_cfg.sigma_B_initial,
                rul_q05=None,
                rul_q50=None,
                rul_q95=None,
                d_failure_abs_h=self._threshold_abs.value if self._threshold_abs else 0.0,
                th_onset=self.onset_detector.th_d or 0.0,
            )

        # ----------------------------------------------------------------
        # Sau t0: smoother đã được khởi động; xử lý bình thường
        # ----------------------------------------------------------------
        h_raw_cur, hi_observed = self.smoother.step(z_t)
        self._history_h.append(hi_observed)

        # Chạy onset detector
        is_degraded, t_onset, h_onset = self.onset_detector.step(cycle, hi_observed)

        # ----------------------------------------------------------------
        # TRƯỜNG HỢP 1: Chưa onset — HEALTHY
        # ----------------------------------------------------------------
        if not is_degraded:
            health_st = (
                HealthState.HEALTHY
                if self._threshold_abs is not None
                else HealthState.CALIBRATING
            )
            rul_st = (
                RULStatus.WAITING_FOR_ONSET
                if self._threshold_abs is not None
                else RULStatus.NOT_CALIBRATED
            )
            return RULPrediction(
                cycle=cycle,
                tau=None,
                health_state=health_st,
                rul_status=rul_st,
                z_raw=z_t,
                hi_raw=h_raw_cur,
                hi_observed=hi_observed,
                hi_state=hi_observed,
                y_acc=0.0,
                mu_tau=self.wiener_cfg.mu_prior,
                sigma_b_tau=self.wiener_cfg.sigma_B_initial,
                rul_q05=None,
                rul_q50=None,
                rul_q95=None,
                d_failure_abs_h=self._threshold_abs.value if self._threshold_abs else 0.0,
                th_onset=self.onset_detector.th_d or 0.0,
            )

        # ----------------------------------------------------------------
        # TRƯỜNG HỢP 2: Onset đã xảy ra — khởi tạo Wiener tracker
        # ----------------------------------------------------------------
        if not self._onset_initialized:
            self._onset_initialized = True
            onset_idx = (t_onset - 1) if t_onset is not None else (len(self._history_h) - 1)
            base_h = self._history_h[onset_idx]
            self.wiener_tracker.init_onset(base_h)

            # Thiết lập threshold relative
            if self._threshold_abs is not None:
                rel_val = self._threshold_abs.value - base_h
                if rel_val <= 0:
                    self._onset_exceeded_threshold = True
                    logger.warning("Onset vượt threshold ngay tại t=%d (H_onset=%.4f > D_fail=%.4f).",
                                   t_onset, base_h, self._threshold_abs.value)
                else:
                    self._threshold_rel = FailureThreshold(
                        value=rel_val,
                        scale=ThresholdScale.RELATIVE_TO_ONSET,
                        source=self._threshold_abs.source,
                        method=self._threshold_abs.method,
                        calibration_n=self._threshold_abs.calibration_n,
                    )

            # Replay lịch sử H từ (onset + 1) đến trước chu kỳ hiện tại
            for step_h in self._history_h[onset_idx + 1 : -1]:
                self.wiener_tracker.step(step_h)

        # --- Khối 3, 4, 5: Wiener step ---
        tau, y_acc, mu_tau, sigma_b_tau, hi_state, is_outlier = self.wiener_tracker.step(hi_observed)
        d_failure_ref = self._threshold_abs.value if self._threshold_abs else 0.0

        # ----------------------------------------------------------------
        # Trường hợp 2a: NOT_CALIBRATED (onset trước t0)
        # ----------------------------------------------------------------
        if self._threshold_rel is None and not self._onset_exceeded_threshold:
            return RULPrediction(
                cycle=cycle,
                tau=tau,
                health_state=HealthState.DEGRADED,
                rul_status=RULStatus.NOT_CALIBRATED,
                z_raw=z_t,
                hi_raw=h_raw_cur,
                hi_observed=hi_observed,
                hi_state=hi_state,
                y_acc=y_acc,
                mu_tau=mu_tau,
                sigma_b_tau=sigma_b_tau,
                rul_q05=None,
                rul_q50=None,
                rul_q95=None,
                is_outlier=is_outlier,
                d_failure_abs_h=d_failure_ref,
                th_onset=self.onset_detector.th_d or 0.0,
            )

        # ----------------------------------------------------------------
        # Trường hợp 2b: THRESHOLD_REACHED
        # ----------------------------------------------------------------
        threshold_reached = self._onset_exceeded_threshold or (
            self._threshold_rel is not None and y_acc >= self._threshold_rel.value
        )

        if threshold_reached:
            return RULPrediction(
                cycle=cycle,
                tau=tau,
                health_state=HealthState.FAILURE_THRESHOLD_REACHED,
                rul_status=RULStatus.THRESHOLD_REACHED,
                z_raw=z_t,
                hi_raw=h_raw_cur,
                hi_observed=hi_observed,
                hi_state=hi_state,
                y_acc=y_acc,
                mu_tau=mu_tau,
                sigma_b_tau=sigma_b_tau,
                rul_q05=0.0,
                rul_q50=0.0,
                rul_q95=0.0,
                is_outlier=is_outlier,
                d_failure_abs_h=d_failure_ref,
                th_onset=self.onset_detector.th_d or 0.0,
            )

        # ----------------------------------------------------------------
        # Trường hợp 2c: DEGRADED — gọi FPT solver
        # ----------------------------------------------------------------
        assert self._threshold_rel is not None
        delta_d = self._threshold_rel.value - y_acc

        if not (np.isfinite(delta_d) and delta_d > 0):
            return RULPrediction(
                cycle=cycle,
                tau=tau,
                health_state=HealthState.FAILURE_THRESHOLD_REACHED,
                rul_status=RULStatus.THRESHOLD_REACHED,
                z_raw=z_t,
                hi_raw=h_raw_cur,
                hi_observed=hi_observed,
                hi_state=hi_state,
                y_acc=y_acc,
                mu_tau=mu_tau,
                sigma_b_tau=sigma_b_tau,
                rul_q05=0.0,
                rul_q50=0.0,
                rul_q95=0.0,
                is_outlier=is_outlier,
                d_failure_abs_h=d_failure_ref,
                th_onset=self.onset_detector.th_d or 0.0,
            )

        fpt = solve_fpt_quantiles(
            delta_d=delta_d,
            mu_tau=mu_tau,
            sigma_b_tau=sigma_b_tau,
            tau=tau,
            gamma=self.wiener_cfg.gamma,
        )

        if fpt.solver_status != FPTSolverStatus.OK:
            return RULPrediction(
                cycle=cycle,
                tau=tau,
                health_state=HealthState.DEGRADED,
                rul_status=RULStatus.SOLVER_FAILED,
                z_raw=z_t,
                hi_raw=h_raw_cur,
                hi_observed=hi_observed,
                hi_state=hi_state,
                y_acc=y_acc,
                mu_tau=mu_tau,
                sigma_b_tau=sigma_b_tau,
                rul_q05=None,
                rul_q50=None,
                rul_q95=None,
                is_outlier=is_outlier,
                d_failure_abs_h=d_failure_ref,
                th_onset=self.onset_detector.th_d or 0.0,
                solver_diagnostics=fpt.diagnostics,
            )

        return RULPrediction(
            cycle=cycle,
            tau=tau,
            health_state=HealthState.DEGRADED,
            rul_status=RULStatus.OK,
            z_raw=z_t,
            hi_raw=h_raw_cur,
            hi_observed=hi_observed,
            hi_state=hi_state,
            y_acc=y_acc,
            mu_tau=mu_tau,
            sigma_b_tau=sigma_b_tau,
            rul_q05=fpt.q05,
            rul_q50=fpt.q50,
            rul_q95=fpt.q95,
            is_outlier=is_outlier,
            d_failure_abs_h=d_failure_ref,
            th_onset=self.onset_detector.th_d or 0.0,
        )

    def run_trajectory(
        self,
        z_seq: np.ndarray,
    ) -> list[RULPrediction]:
        """Chạy toàn bộ trajectory Z của một vòng bi.

        Parameters
        ----------
        z_seq : np.ndarray
            Shape (N,) hoặc (N, 1) — chuỗi Z theo thứ tự measurement.

        Returns
        -------
        list[RULPrediction]
        """
        self.reset_bearing()
        arr = np.asarray(z_seq, dtype=np.float32).flatten()
        predictions = []
        for cycle, z_t in enumerate(arr, start=1):
            pred = self.step(cycle, float(z_t))
            predictions.append(pred)
        return predictions
