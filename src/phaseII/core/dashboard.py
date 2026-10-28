"""
dashboard.py -- KHỐI 7: XUẤT CHỈ SỐ VẬN HÀNH DASHBOARD (TPM ACTIONABLE METRICS)

Mục tiêu:
  Cung cấp thông tin từ mô hình dự báo suy thoái cho kế hoạch bảo trì phòng ngừa (TPM):
    1. RUL Q50 (Median): Trung vị của predictive RUL distribution. Phản ánh xu hướng trung tâm.
    2. RUL Q05: Phân vị thứ 5 của predictive distribution. Không phải guaranteed safety bound.
    3. RUL Q95: Phân vị thứ 95. Không phải statistical upper confidence bound.

  Lưu ý ngôn ngữ:
    Q05, Q50, Q95 là model quantiles — phân vị của predictive distribution.
    Chúng không phải certified safety margin, không phải statistical confidence bound.
    Không dùng "an toàn tuyệt đối" hay "đảm bảo X%" cho bất kỳ quantile nào.
"""

from __future__ import annotations
from typing import Sequence

from src.phaseII.core.schema import RULPrediction, RULStatus


def format_cycle_report(pred: RULPrediction) -> str:
    """Định dạng chuỗi báo cáo nhanh 1 chu kỳ."""
    status = f"[{pred.health_state.value}]"
    rul_st = f"[RUL:{pred.rul_status.value}]"
    cycle_str = f"Meas {pred.cycle:3d}"

    if not pred.is_degraded:
        return (
            f"{status} {rul_st} {cycle_str} "
            f"| HI_obs={pred.hi_observed:.4f} (th_d={pred.th_onset:.4f}) "
            f"| Chưa có dự báo RUL."
        )

    outlier_str = " [SỐC CƠ HỌC - ĐÃ LỌC]" if pred.is_outlier else ""

    if pred.rul_status == RULStatus.THRESHOLD_REACHED:
        return (
            f"{status} {rul_st} {cycle_str} | tau={pred.tau:2d} "
            f"| HI_state={pred.hi_state:.4f} | Y_acc={pred.y_acc:.4f} "
            f"| RUL Q05=0.0 | Q50=0.0 | Q95=0.0{outlier_str}"
        )

    q05 = f"{pred.rul_q05:.1f}" if pred.rul_q05 is not None else "N/A"
    q50 = f"{pred.rul_q50:.1f}" if pred.rul_q50 is not None else "N/A"
    q95 = f"{pred.rul_q95:.1f}" if pred.rul_q95 is not None else "N/A"

    return (
        f"{status} {rul_st} {cycle_str} | tau={pred.tau:2d} "
        f"| HI_obs={pred.hi_observed:.4f} | HI_state={pred.hi_state:.4f} | Y_acc={pred.y_acc:.4f} "
        f"| RUL Q05={q05} | Q50={q50} | Q95={q95}{outlier_str}"
    )


def format_dashboard_table(predictions: Sequence[RULPrediction], step_interval: int = 5) -> str:
    """Xuất bảng theo dõi tiến trình suy thoái theo thời gian."""
    header = (
        f"{'Meas':<5} | {'Tau':<4} | {'Health':<25} | {'RUL Status':<20} | "
        f"{'HI_obs':<8} | {'HI_state':<8} | {'Y_acc':<8} | "
        f"{'RUL Q05':<9} | {'RUL Q50':<9} | {'RUL Q95':<9} | {'Q95-Q05':<9} | {'Ghi chú'}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for p in predictions:
        if p.cycle % step_interval != 0 and p.cycle != len(predictions) and not p.is_outlier:
            continue

        tau_str = str(p.tau) if p.tau is not None else "-"
        q05 = f"{p.rul_q05:.1f}" if p.rul_q05 is not None else "-"
        q50 = f"{p.rul_q50:.1f}" if p.rul_q50 is not None else "-"
        q95 = f"{p.rul_q95:.1f}" if p.rul_q95 is not None else "-"

        # Q95 - Q05 width (không gọi là "bất định ±" vì distribution lệch phải)
        if p.rul_q05 is not None and p.rul_q95 is not None:
            width = f"{p.rul_q95 - p.rul_q05:.1f}"
        else:
            width = "-"

        note = "OUTLIER_FILTERED" if p.is_outlier else ""

        rows.append(
            f"{p.cycle:<5d} | {tau_str:<4} | {p.health_state.value:<25} | {p.rul_status.value:<20} | "
            f"{p.hi_observed:<8.4f} | {p.hi_state:<8.4f} | {p.y_acc:<8.4f} | "
            f"{q05:<9} | {q50:<9} | {q95:<9} | {width:<9} | {note}"
        )

    return "\n".join(rows)


def format_dashboard_header_note() -> str:
    """Trả về chú thích cột dashboard để đính kèm vào báo cáo."""
    return (
        "Chú thích:\n"
        "  RUL Q05     : 5th percentile của model's predictive RUL distribution.\n"
        "  RUL Q50     : 50th percentile (median). Đại diện xu hướng trung tâm.\n"
        "  RUL Q95     : 95th percentile.\n"
        "  Q95-Q05     : Độ rộng 90% predictive interval.\n"
        "  QUAN TRỌNG  : Các giá trị trên là model quantiles, không phải certified safety margin.\n"
        "                Inverse Gaussian lệch phải: Q50-Q05 ≠ Q95-Q50.\n"
    )
