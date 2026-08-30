"""
windowing.py -- cắt sliding window từ dữ liệu ĐÃ chuẩn bị sẵn bên ngoài
(đã chuẩn hóa, đã tách W). File này CHỈ làm 1 việc: cắt window + giữ
metadata (unit_number, cycle) đi kèm -- không xác định vùng khỏe mạnh,
không fit normalization (khác hẳn data.py bản cũ, vốn đã outdated).

CONTRACT với pipeline chuẩn bị dữ liệu bên ngoài (ĐÂY LÀ GIẢ ĐỊNH CỦA
MÌNH -- cần bạn đối chiếu khớp với code chuẩn bị data thật):
  Mỗi engine cung cấp 1 EngineData:
    - X: (n_cycles, n_sensors) float, đã chuẩn hóa
    - W: (n_cycles, w_dim) float, operating condition đã tách riêng
    - cycle: (n_cycles,) int, số cycle thật (giữ thứ tự + tra ngược sau
      này, KHÔNG đưa vào mạng)
    - unit_number: int, id của engine

  Nếu pipeline ngoài của bạn xuất format khác (ví dụ 1 DataFrame gộp
  nhiều engine chưa tách), cần viết 1 hàm chuyển đổi nhỏ trước khi gọi
  cut_windows_for_engine.
"""

from dataclasses import dataclass
import warnings
import numpy as np


@dataclass
class EngineData:
    unit_number: int
    X: np.ndarray       # (n_cycles, n_sensors)
    W: np.ndarray        # (n_cycles, w_dim)
    cycle: np.ndarray    # (n_cycles,)


@dataclass
class WindowBatch:
    X: np.ndarray             # (n_windows, window_len, n_sensors) -- feed vào network
    W: np.ndarray             # (n_windows, w_dim) -- W tại CYCLE CUỐI của mỗi window, dùng làm target cho w_prediction_loss
    unit_number: np.ndarray   # (n_windows,) -- KHÔNG feed vào network, chỉ để group/sort/debug
    cycle: np.ndarray         # (n_windows,) -- cycle cuối của mỗi window, KHÔNG feed vào network


def cut_windows_for_engine(engine: EngineData, window_len: int, stride: int) -> WindowBatch:
    """
    QUY ƯỚC: nhãn (W, cycle) của 1 window = giá trị tại TIMESTEP CUỐI
    CÙNG trong window -- window "nhìn lại quá khứ" để mô tả trạng thái
    tại thời điểm hiện tại, đúng tinh thần real-time inference.

    PHẢI gọi hàm này riêng cho TỪNG engine -- KHÔNG được nối nhiều engine
    thành 1 mảng lớn rồi mới cắt, sliding window sẽ lẫn dữ liệu ở ranh
    giới nối giữa 2 engine khác nhau (lỗi này không crash, chỉ âm thầm
    cho ra kết quả sai).
    """
    n = engine.X.shape[0]
    if n < window_len:
        warnings.warn(f"Engine {engine.unit_number}: chỉ có {n} cycles < "
                       f"window_len={window_len}, bỏ qua engine này.")
        return WindowBatch(
            X=np.empty((0, window_len, engine.X.shape[1]), dtype=engine.X.dtype),
            W=np.empty((0, engine.W.shape[1]), dtype=engine.W.dtype),
            unit_number=np.empty((0,), dtype=int),
            cycle=np.empty((0,), dtype=int),
        )

    starts = list(range(0, n - window_len + 1, stride))
    X_windows = np.stack([engine.X[s:s + window_len] for s in starts], axis=0)
    end_idx = [s + window_len - 1 for s in starts]
    W_labels = engine.W[end_idx]
    cycle_labels = engine.cycle[end_idx]
    unit_numbers = np.full(len(starts), engine.unit_number, dtype=int)

    return WindowBatch(X=X_windows, W=W_labels, unit_number=unit_numbers, cycle=cycle_labels)


def concat_window_batches(batches: list[WindowBatch]) -> WindowBatch:
    """Gộp nhiều WindowBatch (nhiều engine) lại thành 1, GIỮ NGUYÊN THỨ
    TỰ đưa vào -- người gọi (train.py) tự sort lại theo (unit_number,
    cycle) sau đó, cần thiết cho loss smoothness/monotonicity."""
    return WindowBatch(
        X=np.concatenate([b.X for b in batches], axis=0),
        W=np.concatenate([b.W for b in batches], axis=0),
        unit_number=np.concatenate([b.unit_number for b in batches], axis=0),
        cycle=np.concatenate([b.cycle for b in batches], axis=0),
    )
