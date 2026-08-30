"""
metrics.py -- đo lường và kiểm chứng cho Smart AE (Phase I). Thay thế
bản metrics.py cũ.

THAY ĐỔI QUAN TRỌNG SO VỚI BẢN CŨ: vì Smart AE train trên TOÀN BỘ
trajectory (không chỉ healthy), reconstruction error KHÔNG CÒN tăng lên
ở đoạn suy thoái như bản AE cũ -- model đã học tái tạo tốt cả 2 loại
pattern (đó chính xác là điều loss function được tối ưu để đạt được).
Tín hiệu phát hiện event_bin giờ CHUYỂN SANG độ trôi của Z so với
baseline lúc mới bắt đầu theo dõi mỗi engine (xem z_drift_per_engine).

reconstruction_error vẫn giữ lại như tín hiệu PHỤ: nếu error tăng đột
biến dù đã train full trajectory, đó có thể là dấu hiệu 1 kiểu lỗi hoàn
toàn mới (novelty), chưa từng có trong tập train -- không còn là công
cụ chính để định vị event_bin nữa.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def reconstruction_error_per_window(model, windows: torch.Tensor,
                                     per_sensor_normalize: np.ndarray | None = None) -> np.ndarray:
    """
    windows: (n_windows, window_len, n_sensors)
    per_sensor_normalize: std của error từng sensor (tính từ vùng khỏe
        mạnh nếu có), dùng để chuẩn hóa error trước khi cộng gộp.

    Trả về (n_windows,) -- tín hiệu PHỤ (novelty detection), xem
    docstring đầu file. KHÔNG còn là tín hiệu chính để tìm event_bin.
    """
    model.eval()
    x_hat, _, _ = model(windows)                        # LƯU Ý: 3 giá trị trả về, khác bản cũ (2 giá trị)
    err = (x_hat - windows) ** 2                          # (n_windows, window_len, n_sensors)
    err = err.mean(dim=1)                                   # gộp theo thời gian -> (n_windows, n_sensors)
    err = err.cpu().numpy()

    if per_sensor_normalize is not None:
        err = err / np.where(per_sensor_normalize == 0, 1.0, per_sensor_normalize)

    return err.mean(axis=1)                                  # gộp theo sensor -> (n_windows,)


def z_drift_per_engine(z_seq: np.ndarray, n_baseline_bins: int = 5) -> np.ndarray:
    """
    z_seq: (n_bins, z_dim) -- CỦA 1 ENGINE, đã sắp theo cycle tăng dần
    (chính là field .X trong kết quả trả về của
    train.extract_latents_for_engine -- lưu ý field đó thực chất là Z).

    n_baseline_bins: số bin ĐẦU TIÊN dùng để ước lượng baseline "gần
    như chắc chắn khỏe". Đây là GIẢ ĐỊNH NỚI LỎNG -- sai lệch nhẹ ở đây
    (ví dụ vài bin đầu thực ra đã hơi suy thoái) chỉ ảnh hưởng ngưỡng,
    KHÔNG ảnh hưởng việc model học, khác hẳn rủi ro của việc xác định
    healthy-region để làm ranh giới TRAIN ở bản pipeline cũ.

    Trả về (n_bins,) -- khoảng cách Euclid từ mỗi bin tới baseline. Đây
    là TÍN HIỆU CHÍNH thay thế reconstruction_error để tìm event_bin.
    """
    n_baseline_bins = min(n_baseline_bins, z_seq.shape[0])
    baseline = z_seq[:n_baseline_bins].mean(axis=0)         # (z_dim,)
    return np.linalg.norm(z_seq - baseline, axis=-1)          # (n_bins,)


def fit_threshold(healthy_signal: np.ndarray, percentile: float = 95.0) -> float:
    """Không đổi logic so với bản cũ -- percentile 95 nghĩa là chấp
    nhận 5% false positive ngay trên chính dữ liệu tham chiếu. Giờ
    thường nhận z_drift thay vì reconstruction error làm input."""
    return float(np.percentile(healthy_signal, percentile))


def find_degradation_point(signal: np.ndarray, threshold: float,
                            n_consecutive: int = 3) -> int | None:
    """
    signal: (T,) đã làm mượt (nên EWMA/moving-average trước khi gọi --
    xem smooth_signal bên dưới). Không đổi logic so với bản cũ, chỉ đổi
    tên tham số cho tổng quát (không còn gắn chặt với "error").

    Trả về index đầu tiên mà signal vượt ngưỡng LIÊN TỤC ít nhất
    n_consecutive lần. Trả về None nếu chưa bao giờ đạt điều kiện này.

    LƯU Ý: index trả về là index trong mảng signal/z_seq (theo bin_stride
    lúc extract_latents_for_engine), KHÔNG phải cycle thật -- dùng mảng
    .cycle đi kèm (từ WindowBatch) để convert: DP_cycle_that = cycle[DP_index].
    """
    above = signal > threshold
    run_length = 0
    for i, flag in enumerate(above):
        run_length = run_length + 1 if flag else 0
        if run_length >= n_consecutive:
            return i - n_consecutive + 1
    return None


def smooth_signal(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """EWMA đơn giản, không phụ thuộc thư viện ngoài. Đổi tên từ
    smooth_errors -> smooth_signal (tổng quát, không gắn chặt error)."""
    smoothed = np.empty_like(signal)
    smoothed[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed[i] = alpha * signal[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def spearman_validation(signal: np.ndarray, true_rul: np.ndarray) -> float:
    """
    CHỈ dùng trên C-MAPSS (hoặc bất kỳ dataset nào có RUL thật). Trả về
    hệ số tương quan Spearman -- kỳ vọng ÂM MẠNH (signal cao khi RUL
    thấp). Không đổi logic so với bản cũ.
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(signal, true_rul)
    return float(corr)
