"""
loss.py -- các thành phần loss cho Smart AE, thay thế bản loss.py cũ
(chỉ có MSE reconstruction thuần).

4 hàm tách biệt + 1 hàm tổng hợp, để train.py có thể log riêng từng
thành phần -- quan trọng để debug: nếu recon tốt nhưng w_pred tệ, biết
ngay vấn đề nằm ở đâu, không phải đoán mò từ 1 con số loss gộp.

Về z_smoothness_loss/z_monotonicity_loss: đây là bản THAY THẾ cho ràng
buộc corr(Z, t_trong_1_cycle) gốc của bài báo Bajarunas et al. Bài báo
gốc dùng N-CMAPSS (có độ phân giải trong-1-cycle). CMAPSS FD002 (đang
dùng cho hackathon) mỗi cycle chỉ có 1 dòng dữ liệu -- không có
"trong-1-cycle" để mà decorrelate. Ràng buộc ở đây chuyển sang áp dụng
GIỮA các cycle liên tiếp thay vì trong nội bộ 1 cycle, giữ đúng tinh
thần vật lý ("degradation không nhảy loạn, chỉ tích lũy dần") nhưng ở
đúng granularity mà FD002 có.
"""

import torch


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE trung bình trên mọi phần tử (batch, window_len, n_sensors).
    Không đổi so với bản cũ."""
    return torch.mean((x_hat - x) ** 2)


def w_prediction_loss(w_hat: torch.Tensor, w_true: torch.Tensor) -> torch.Tensor:
    """w_true: (batch, w_dim) -- giá trị W THẬT tại cycle cuối cùng của
    mỗi window (đúng quy ước nhãn window trong windowing.py). Ép nhánh
    W_hat của encoder thực sự đoán đúng W từ X."""
    return torch.mean((w_hat - w_true) ** 2)


def z_smoothness_loss(z_seq: torch.Tensor, same_engine_mask: torch.Tensor) -> torch.Tensor:
    """
    z_seq: (n_windows, z_dim) -- ĐÃ SẮP XẾP theo (unit_number, cycle)
    tăng dần (train.py đảm bảo việc này trước khi gọi).
    same_engine_mask: (n_windows-1,) bool -- True tại vị trí i nếu
        z_seq[i] và z_seq[i+1] thuộc CÙNG 1 engine (để không tính diff
        xuyên qua ranh giới 2 engine khác nhau khi đã nối nhiều engine
        lại thành 1 chuỗi dài trong cùng 1 batch).

    Phạt sự thay đổi đột ngột giữa 2 cycle liên tiếp cùng 1 engine.
    """
    diffs = z_seq[1:] - z_seq[:-1]                       # (n_windows-1, z_dim)
    sq_diffs = (diffs ** 2).sum(dim=-1)                    # (n_windows-1,)
    if same_engine_mask.sum() == 0:
        return torch.zeros((), device=z_seq.device)
    return sq_diffs[same_engine_mask].mean()


def z_monotonicity_loss(z_seq: torch.Tensor, same_engine_mask: torch.Tensor) -> torch.Tensor:
    """
    Cùng quy ước sắp xếp/mask như z_smoothness_loss.
    Phạt khi Z GIẢM giữa 2 cycle liên tiếp -- giả định suy thoái chỉ
    tích lũy, không tự hồi phục. Áp dụng riêng từng chiều của Z rồi
    cộng lại (tổng quát hóa tự nhiên từ trường hợp z_dim=1 của bài báo
    gốc, nơi Monotonicity là 1 trong các chỉ số đánh giá -- ở đây biến
    nó thành 1 phần của loss thay vì chỉ dùng để evaluate).
    """
    diffs = z_seq[1:] - z_seq[:-1]                        # (n_windows-1, z_dim)
    penalty = torch.relu(-diffs).sum(dim=-1)               # (n_windows-1,)
    if same_engine_mask.sum() == 0:
        return torch.zeros((), device=z_seq.device)
    return penalty[same_engine_mask].mean()


def smart_ae_loss(x_hat: torch.Tensor, x: torch.Tensor,
                   w_hat: torch.Tensor, w_true: torch.Tensor,
                   z_seq: torch.Tensor, same_engine_mask: torch.Tensor,
                   lambda_w: float = 1.0, lambda_smooth: float = 1.0,
                   lambda_mono: float = 1.0) -> dict[str, torch.Tensor]:
    """Gộp cả 4 thành phần loss + trả về dict đủ từng key để train.py
    log riêng, dễ debug hơn 1 con số loss gộp duy nhất."""
    l_recon = reconstruction_loss(x_hat, x)
    l_w = w_prediction_loss(w_hat, w_true)
    l_smooth = z_smoothness_loss(z_seq, same_engine_mask)
    l_mono = z_monotonicity_loss(z_seq, same_engine_mask)
    total = l_recon + lambda_w * l_w + lambda_smooth * l_smooth + lambda_mono * l_mono
    return {
        "total": total,
        "recon": l_recon,
        "w_pred": l_w,
        "smooth": l_smooth,
        "mono": l_mono,
    }
