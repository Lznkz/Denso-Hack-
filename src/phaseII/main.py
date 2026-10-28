"""
main.py -- ENTRYPOINT PHASE II: HỆ THỐNG DỰ BÁO SUY THOÁI LAI
(Nonlinear Wiener Process with Adaptive Drift & Diffusion)

Kiến trúc mới (Phase I scalar Z, không W):
  1. Phase I checkpoint (.pt) cung cấp model + normalization.
  2. Pipeline tính Z_fail_abs từ median(Z_end) của các train bearings.
  3. Chạy trajectory trực tuyến trên bearing mục tiêu chỉ cần Z sequence.
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo root workspace luôn có trong sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch

from src.phaseII.core.schema import (
    SmoothingConfig,
    OnsetConfig,
    WienerConfig,
    RULStatus,
    RULPrediction,
)
from src.phaseII.prognostics_pipeline import PrognosticsPipeline
from src.phaseII.core.dashboard import format_dashboard_table, format_dashboard_header_note
from src.phaseII.core.metrics import compute_rul_errors, compute_interval_coverage

logger = logging.getLogger(__name__)


def run_phase2(
    ckpt_path: str | Path,
    data_dir: str | Path,
    train_bearings: list[str] | None = None,
    val_bearing: str = "B12",
    z_fail_abs: float | None = -20.0,
    gamma: float = 1.0,           # Sửa mặc định từ 1.75 -> 1.0 (Linear Wiener)
    t0_initial: int = 20,          # Tăng từ 10 lên 20 để baseline khỏe mạnh vững chắc hơn
    ewma_alpha: float = 0.08,      # Sửa mặc định từ 0.3 -> 0.08
    median_window: int = 11,       # Thêm tham số rolling median
    true_failure_cycle: int | None = 767,
    dashboard_step: int | None = None,
) -> tuple[PrognosticsPipeline, list[RULPrediction]]:
    """Chạy Phase II Prognostic Pipeline với checkpoint Phase I và dataset processed_dual.

    Parameters
    ----------
    ckpt_path : str | Path
        Đường dẫn tới file .pt Phase I
        (ví dụ: F:/topk7/topk7/thien_gay_topk7.pt)
    data_dir : str | Path
        Thư mục processed_dual chứa các subfolder bearing
        (ví dụ: F:/topk7/topk7/processed_dual)
    train_bearings : list[str] | None
        Danh sách bearing train để tính Z_fail_abs (nếu z_fail_abs=None)
    val_bearing : str
        Bearing mục tiêu cần dự báo (mặc định: "B12")
    z_fail_abs : float | None
        Ngưỡng hỏng cố định trên thang Z gốc (mặc định: -20.0; nếu None thì tính median từ train)
    gamma : float
        Hệ số Paris Law (mặc định 1.0 - Linear Wiener)
    t0_initial : int
        Số measurement đầu tiên dùng làm baseline khỏe mạnh (mặc định 20)
    ewma_alpha : float
        Hệ số làm mượt EWMA cho H_t (mặc định 0.08)
    median_window : int
        Kích thước cửa sổ lọc Rolling Median (mặc định 11)
    true_failure_cycle : int | None
        Chu kỳ hỏng thực tế (nếu có) để tính RMSE / Coverage (mặc định 767)
    dashboard_step : int | None
        Bước in dashboard (mặc định auto = N//20)

    Returns
    -------
    pipeline : PrognosticsPipeline
    predictions : list[RULPrediction]
    """
    data_dir = Path(data_dir)
    smoothing_cfg = SmoothingConfig(median_window=median_window, ewma_alpha=ewma_alpha)
    onset_cfg = OnsetConfig(t0_initial=t0_initial)
    wiener_cfg = WienerConfig(gamma=gamma)

    if z_fail_abs is not None:
        print(f"--> [Phase II] Dùng ngưỡng hỏng cố định: z_failure_abs = {z_fail_abs:.4f} (bỏ qua ước lượng median)")
        pipeline = PrognosticsPipeline(
            z_fail_abs=z_fail_abs,
            smoothing_cfg=smoothing_cfg,
            onset_cfg=onset_cfg,
            wiener_cfg=wiener_cfg,
        )
    else:
        # Load train data để tính fleet median threshold
        if train_bearings is None:
            candidates = ["B10", "B11", "B12", "B13", "B14"]
            train_bearings = [b for b in candidates if b != val_bearing]

        print(f"--> [Phase II] Nạp dữ liệu train ({len(train_bearings)} bearings) để tính Z_fail_abs...")
        train_data: dict[str, np.ndarray] = {}
        for b in train_bearings:
            fpath = data_dir / b / "X_hi_spev.npy"
            if not fpath.exists():
                logger.warning("Không tìm thấy %s, bỏ qua.", fpath)
                continue
            train_data[b] = np.load(str(fpath))
            print(f"   {b}: {train_data[b].shape}")

        if len(train_data) == 0:
            raise FileNotFoundError(f"Không tìm thấy dữ liệu train trong {data_dir}")

        print(f"--> [Phase II] Xây dựng pipeline từ checkpoint: {ckpt_path}")
        pipeline = PrognosticsPipeline.from_checkpoint(
            ckpt_path=ckpt_path,
            train_bearing_data=train_data,
            smoothing_cfg=smoothing_cfg,
            onset_cfg=onset_cfg,
            wiener_cfg=wiener_cfg,
        )
        print(f"   z_failure_abs = {pipeline.z_fail_abs:.4f}")

    # 3. Encode bearing mục tiêu và chạy trajectory
    val_path = data_dir / val_bearing / "X_hi_spev.npy"
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu bearing mục tiêu: {val_path}")

    X_val = np.load(str(val_path))
    z_val = PrognosticsPipeline.encode_bearing_from_checkpoint(ckpt_path, X_val)

    print(f"--> [Phase II] Chạy trajectory {val_bearing}: {len(z_val)} measurements...")
    predictions = pipeline.run_trajectory(z_val)

    # 4. Dashboard
    n = len(predictions)
    step = dashboard_step or max(1, n // 20)
    print("\n" + "=" * 100)
    print(f"KẾT QUẢ DỰ BÁO SUY THOÁI & RUL (WIENER HYBRID PROGNOSTIC SCHEME) - {val_bearing}")
    print("=" * 100)
    print(format_dashboard_table(predictions, step_interval=step))
    print(format_dashboard_header_note())

    # 5. Metrics nếu có ground truth
    degraded_ok = [p for p in predictions if p.is_degraded and p.rul_status == RULStatus.OK]
    if true_failure_cycle is not None and len(degraded_ok) > 0:
        pred_q50 = [p.rul_q50 for p in degraded_ok]
        pred_q05 = [p.rul_q05 for p in degraded_ok]
        pred_q95 = [p.rul_q95 for p in degraded_ok]
        true_ruls = [max(float(true_failure_cycle - p.cycle), 0.0) for p in degraded_ok]

        errs = compute_rul_errors(pred_q50, true_ruls)
        coverage = compute_interval_coverage(pred_q05, pred_q95, true_ruls)

        print("-" * 50)
        print("ĐÁNH GIÁ ĐỘ CHÍNH XÁC DỰ BÁO RUL:")
        print(f"  RMSE RUL Q50 : {errs['rmse']:.2f} chu kỳ")
        print(f"  MAE  RUL Q50 : {errs['mae']:.2f} chu kỳ")
        print(f"  Độ phủ 90%   : {coverage:.1f}%")
        print("-" * 50)

    # 6. Tóm tắt nhanh
    counts = {}
    for p in predictions:
        counts[p.rul_status.value] = counts.get(p.rul_status.value, 0) + 1
    print("Status breakdown:", counts)
    onset = pipeline.onset_detector.t_onset
    print(f"t_onset = {onset}, Z_base = {pipeline.smoother.z_base:.4f}" if pipeline.smoother.z_base else "t_onset = None")

    return pipeline, predictions


BEARING_EOL = {
    "B10": 3224,
    "B11": 1953,
    "B12": 767,
    "B13": 19417,
    "B14": 12317,
    "B17": 2733,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chạy Phase II Prognostics Pipeline")
    parser.add_argument("--bearing", type=str, default="B11", help="Bearing mục tiêu (mặc định: B11)")
    parser.add_argument("--z-fail", type=float, default=-20.0, help="Ngưỡng hỏng z_fail_abs (mặc định: -20.0)")
    parser.add_argument("--ckpt", type=str, default="F:/topk7/topk7/thien_gay_topk7.pt", help="Đường dẫn checkpoint")
    parser.add_argument("--data", type=str, default="F:/topk7/topk7/processed_dual", help="Thư mục processed_dual")
    parser.add_argument("--gamma", type=float, default=1.0, help="Hệ số Paris Law (mặc định: 1.0 - Linear)")
    parser.add_argument("--t0", type=int, default=20, help="Số chu kỳ baseline t0 (mặc định: 20)")
    parser.add_argument("--alpha", type=float, default=0.08, help="EWMA alpha (mặc định: 0.08)")
    parser.add_argument("--median-win", type=int, default=11, help="Rolling median window (mặc định: 11)")
    parser.add_argument("--eol", type=int, default=None, help="Chu kỳ hỏng thực tế (mặc định: tự tra cứu theo bearing)")
    parser.add_argument("--step", type=int, default=None, help="Bước hiển thị dashboard")

    args = parser.parse_args()

    eol = args.eol if args.eol is not None else BEARING_EOL.get(args.bearing, 1953)

    pipeline, preds = run_phase2(
        ckpt_path=args.ckpt,
        data_dir=args.data,
        val_bearing=args.bearing,
        z_fail_abs=args.z_fail,
        gamma=args.gamma,
        t0_initial=args.t0,
        ewma_alpha=args.alpha,
        median_window=args.median_win,
        true_failure_cycle=eol,
        dashboard_step=args.step,
    )

