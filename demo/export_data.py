"""
export_data.py

Script chạy prepare_cmapss_split() và xuất toàn bộ dữ liệu trả về thành các file vật lý
(CSV, JSON, NPZ) lưu vào thư mục `demo/data/exported_data/`.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.core.data import SplitConfig, prepare_cmapss_split


def dict_to_dataframe(dict_data: dict[int, np.ndarray], col_names: list[str]) -> pd.DataFrame:
    """Chuyển đổi {engine_id: ndarray} thành 1 DataFrame hoàn chỉnh có cột unit_number và cycle."""
    records = []
    for engine_id, arr in dict_data.items():
        n_cycles = len(arr)
        cycles = np.arange(1, n_cycles + 1, dtype=np.int32).reshape(-1, 1)
        units = np.full((n_cycles, 1), engine_id, dtype=np.int32)
        combined = np.hstack([units, cycles, arr])
        records.append(combined)
    
    all_data = np.vstack(records)
    columns = ["unit_number", "cycle"] + col_names
    return pd.DataFrame(all_data, columns=columns)


def export_all():
    output_dir = Path("demo/data/exported_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--> Đang khởi tạo dữ liệu với SplitConfig...")
    config = SplitConfig(
        data_path="demo/data/train_FD002.txt",
        train_ratio=0.70,
        safety_margin_min=0.10,
        safety_margin_max=0.20,
        rare_fault_probability=0.20,
        rare_fault_max_duration=10,
        rare_fault_start_cycle_min=60,
        rare_fault_magnitude_min=0.03,
        rare_fault_magnitude_max=0.10,
        rare_fault_types=("plateau", "drop", "drift"),
        test_fraction_min=0.10,
        test_fraction_max=0.30,
        random_seed=42,
    )
    
    data = prepare_cmapss_split(config)
    sensors = data["sensors"]
    op_settings = data["op_settings"]
    
    print("--> Bắt đầu xuất các file vật lý vào thư mục:", output_dir.resolve())
    
    # 1. Xuất tập TRAIN
    df_train_sensors = dict_to_dataframe(data["train_data"], sensors)
    df_train_sensors.to_csv(output_dir / "train_data.csv", index=False)
    print(" [1/8] Đã xuất train_data.csv (Dữ liệu cảm biến Train đã cắt & tiêm lỗi)")
    
    df_train_ops = dict_to_dataframe(data["train_op_data"], op_settings)
    df_train_ops.to_csv(output_dir / "train_op_data.csv", index=False)
    print(" [2/8] Đã xuất train_op_data.csv (Dữ liệu op settings Train)")
    
    # Train metadata (xuất cả JSON và CSV)
    with open(output_dir / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(data["train_metadata"], f, indent=2, default=str)
    
    df_train_meta = pd.DataFrame.from_dict(data["train_metadata"], orient="index")
    df_train_meta.index.name = "unit_number"
    df_train_meta.reset_index().to_csv(output_dir / "train_metadata.csv", index=False)
    print(" [3/8] Đã xuất train_metadata.json & train_metadata.csv (Thông tin RUL, lỗi tiêm)")
    
    # 2. Xuất tập TEST
    df_test_sensors = dict_to_dataframe(data["test_data"], sensors)
    df_test_sensors.to_csv(output_dir / "test_data.csv", index=False)
    print(" [4/8] Đã xuất test_data.csv (Dữ liệu cảm biến Test quan sát ban đầu)")
    
    df_test_ops = dict_to_dataframe(data["test_op_data"], op_settings)
    df_test_ops.to_csv(output_dir / "test_op_data.csv", index=False)
    print(" [5/8] Đã xuất test_op_data.csv (Dữ liệu op settings Test ban đầu)")
    
    with open(output_dir / "test_observation_points.json", "w", encoding="utf-8") as f:
        json.dump(data["test_observation_points"], f, indent=2)
    print(" [6/8] Đã xuất test_observation_points.json (Điểm cắt chu kỳ ban đầu của Test)")
    
    # 3. Xuất toàn bộ dữ liệu GỐC ĐẦY ĐỦ (Ground Truth)
    df_full_sensors = dict_to_dataframe(data["engine_data"], sensors)
    df_full_sensors.to_csv(output_dir / "full_engine_data.csv", index=False)
    print(" [7/8] Đã xuất full_engine_data.csv (Toàn bộ chu kỳ thực tế đầy đủ của 260 động cơ)")
    
    # 4. Xuất file tổng quan Summary Info
    summary = {
        "sensors": sensors,
        "op_settings": op_settings,
        "total_engines": len(data["engine_data"]),
        "train_count": len(data["train_ids"]),
        "test_count": len(data["test_ids"]),
        "train_ids": data["train_ids"],
        "test_ids": data["test_ids"],
    }
    with open(output_dir / "summary_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(" [8/8] Đã xuất summary_info.json (Tổng quan danh sách ID, cảm biến)")
    
    print("\n=== HOÀN TẤT XUẤT TẤT CẢ FILE VẬT LÝ THÀNH CÔNG ===")


if __name__ == "__main__":
    export_all()
