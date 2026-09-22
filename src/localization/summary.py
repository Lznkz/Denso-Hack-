"""
summary.py -- KẾT XUẤT BÁO CÁO ĐỊNH VỊ CẢM BIẾN LỖI ("Ở ĐÂU")

Chức năng:
  Nhận trực tiếp đối tượng kết quả `SensorAttributionResult` từ module `localization_fast`
  để kết xuất ra chuỗi báo cáo súc tích hoặc bảng chi tiết (tiếng Việt).
"""

from __future__ import annotations
from typing import Any


def summarize(result: Any) -> str:
    """
    Nhận kết quả từ FastReconstructionAttributor.localize() và tạo chuỗi báo cáo súc tích.

    Parameters
    ----------
    result: Đối tượng SensorAttributionResult trả về từ localization_fast.
    """
    if not getattr(result, "stage1_triggered", True):
        cycle_str = f"Cycle {result.cycle}" if result.cycle is not None else "Current Window"
        return f"[AN TOÀN] {cycle_str} | Tầng 'Có hay không' chưa kích hoạt (Hệ thống hoạt động bình thường)."

    status = "[CẢNH BÁO]" if result.is_alert else "[BÌNH THƯỜNG]"
    cycle_str = f"Cycle {result.cycle}" if result.cycle is not None else "Current Window"
    type_str = f"Loại lỗi: {result.anomaly_type}" if result.is_alert else "An toàn"
    lines = [f"{status} {cycle_str} | {type_str} | Tổng điểm bất thường: {result.total_anomaly_score:.4f}"]

    if result.is_alert and result.primary_sensor:
        top = result.top_sensors[0]
        dev = result.deviations.get(result.primary_sensor)
        nature_desc = "Đột biến tức thời" if dev and dev.fault_nature == "DOT_BIEN" else "Suy thoái tích lũy"

        if dev and dev.direction == "OSCILLATING":
            dev_str = f"({nature_desc}, DAO ĐỘNG RUNG LẮC, biên độ RMS {dev.rms_diff:.2f} đơn vị, trôi trung bình {dev.raw_diff:+.2f})"
        elif dev:
            dev_str = f"({nature_desc}, {dev.direction}, lệch {dev.raw_diff:+.2f} đơn vị, {dev.percent_diff:+.1f}%)"
        else:
            dev_str = ""

        lines.append(f"  -> Cảm biến nghi vấn số 1: {top[0]} (Đóng góp {top[2]:.1f}% tổng lỗi) {dev_str}")

        lines.append("  -> Top cảm biến lệch nhiều nhất so với trạng thái chuẩn:")
        for rank, (s_name, score, pct) in enumerate(result.top_sensors[:3], 1):
            d = result.deviations.get(s_name)
            if d and d.direction == "OSCILLATING":
                d_str = f"DAO_DONG (RMS {d.rms_diff:.2f})"
            elif d:
                d_str = f"{d.direction:<6} ({d.raw_diff:+.2f})"
            else:
                d_str = "N/A"
            d_nature = f"[{d.fault_nature}]" if d else ""
            lines.append(f"     {rank}. {s_name:<8}: Đóng góp {pct:5.1f}% | Error={score:.4f} | Lệch={d_str} {d_nature}")
    else:
        lines.append("  -> Tất cả các cảm biến hoạt động trong trạng thái bình thường.")

    return "\n".join(lines)


# Cung cấp bí danh format_summary để tiện sử dụng
format_summary = summarize


def format_detail_table(result: Any) -> str:
    """
    Xuất bảng chi tiết toàn bộ các cảm biến theo thứ tự đóng góp lỗi giảm dần.
    """
    header = f"{'Rank':<5} | {'Sensor':<8} | {'Đóng góp (%)':<14} | {'Error Score':<12} | {'Trôi (Mean)':<12} | {'RMS (Biên độ)':<14} | {'% Lệch':<10} | {'Hướng':<11} | {'Bản chất'}"
    sep = "-" * len(header)
    rows = [header, sep]

    ranked_sensors = sorted(
        result.sensor_contributions.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (s_name, contrib) in enumerate(ranked_sensors, 1):
        dev = result.deviations.get(s_name)
        if dev:
            diff_str = f"{dev.raw_diff:+.2f}"
            rms_str = f"{dev.rms_diff:.2f}"
            pct_str = f"{dev.percent_diff:+.1f}%"
            direction = dev.direction
            nature = dev.fault_nature
            score = f"{dev.error_score:.4f}"
        else:
            diff_str, rms_str, pct_str, direction, nature, score = "0.00", "0.00", "0.0%", "NORMAL", "NORMAL", "0.0000"

        rows.append(
            f"{rank:<5} | {s_name:<8} | {contrib:12.2f}% | {score:<12} | {diff_str:<12} | {rms_str:<14} | {pct_str:<10} | {direction:<11} | {nature}"
        )

    return "\n".join(rows)
