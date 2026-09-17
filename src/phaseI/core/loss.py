"""
loss.py -- sửa lần 4: nhận `baseline` làm tham số từ ngoài (train.py:
_compute_baseline, đã .detach()), KHÔNG tự tính baseline nội bộ nữa --
khớp đúng interface train.py đang gọi.

3 thành phần: recon + mono (dùng baseline ngoài) + anti-collapse
(không cần baseline, chỉ cần std theo từng engine). Không còn
z_smoothness_loss (đã bỏ, xem giải thích các lượt trao đổi trước).

sửa lần 5: reconstruction_loss thêm per_window_normalize -- normalize
absolute MSE theo variance của CHÍNH window đó (detach) trước khi lấy
trung bình batch. Lý do: X hiện chuẩn hóa bằng 1 cặp (mean, std) GLOBAL
duy nhất (main.py normalize_engines), trong khi biên độ rung tăng dần
hàng chục-hàng nghìn lần từ early-life đến late-life trong CÙNG 1
bearing -- late-life window nghiễm nhiên có absolute MSE lớn hơn nhiều
bậc so với early-life, dù model tái tạo TƯƠNG ĐỐI tốt như nhau (đã xác
nhận bằng số: absolute MSE early=0.002 vs late=3.3, nhưng relative MSE
cả hai ~0.6-0.75). Không normalize sẽ khiến gradient bị vài batch
late-life áp đảo, early-life gần như không đóng góp gì vào update.
"""

import torch


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor,
                         per_window_normalize: bool = True,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    per_window_normalize=True: chia sai số bình phương của MỖI WINDOW
    cho variance của chính window đó (theo x, KHÔNG theo x_hat -- tránh
    model tự "gian lận" bằng cách hạ variance của x_hat để giảm mẫu
    số) trước khi gộp batch. .detach() để hệ số chia không tự nó là
    một phần cần tối ưu.

    per_window_normalize=False: giữ nguyên hành vi cũ (MSE tuyệt đối,
    dùng để so sánh/debug hoặc khi X đã ở scale đồng nhất sẵn).
    """
    sq_err = (x_hat - x) ** 2
    if per_window_normalize:
        # x: (batch, window_len, n_sensors) -- var theo 2 chiều cuối, giữ batch
        window_var = x.var(dim=(1, 2), keepdim=True, unbiased=False).detach() + eps
        sq_err = sq_err / window_var
    return sq_err.mean()


def z_monotonicity_loss(z_seq: torch.Tensor, baseline: torch.Tensor,
                         same_engine_mask: torch.Tensor) -> torch.Tensor:
    """
    baseline: (n_windows, z_dim) -- ĐÃ detach() từ train.py._compute_baseline,
    baseline riêng cho từng engine, cùng shape với z_seq (mỗi hàng của
    z_seq có 1 baseline tương ứng, lặp lại cho mọi window cùng engine).

    drift = ||z_seq - baseline||_2 -- gộp z_dim chiều về 1 số vô hướng
    mỗi window, đo mức lệch khỏi trạng thái khỏe mạnh riêng của engine
    đó. Phạt khi drift GIẢM giữa 2 window liên tiếp cùng engine.
    """
    drift = torch.linalg.norm(z_seq - baseline, dim=-1)     # (n_windows,)
    drift_diff = drift[1:] - drift[:-1]                       # (n_windows-1,)
    penalty = torch.relu(-drift_diff)
    if same_engine_mask.sum() == 0:
        return torch.zeros((), device=z_seq.device)
    return penalty[same_engine_mask].mean()


def _segment_ids_from_mask(same_engine_mask: torch.Tensor, n: int) -> torch.Tensor:
    boundary = (~same_engine_mask).float()
    seg = torch.cat([torch.zeros(1, device=boundary.device), boundary]).cumsum(0)
    return seg.long()


def z_anti_collapse_loss(z_seq: torch.Tensor, same_engine_mask: torch.Tensor,
                          min_std: float = 0.1, min_len: int = 5) -> torch.Tensor:
    """
    Phạt khi std(Z) THEO TỪNG ENGINE (trục thời gian) quá thấp -- KHÔNG
    dùng std toàn batch (xem giải thích chi tiết ở lượt trao đổi trước:
    std batch có thể bị "lách" nếu mỗi engine collapse về 1 hằng số
    RIÊNG, khác nhau giữa các engine).

    min_len: engine có ít hơn min_len window trong batch này sẽ bị bỏ
    qua (std trên quá ít điểm không đáng tin).
    """
    n = z_seq.shape[0]
    seg_ids = _segment_ids_from_mask(same_engine_mask, n)
    _, counts = torch.unique_consecutive(seg_ids, return_counts=True)

    penalties = []
    start = 0
    for c_tensor in counts:
        c = int(c_tensor.item())
        if c >= min_len:
            seg = z_seq[start:start + c]
            std = seg.std(dim=0, unbiased=False)

            penalties.append(torch.relu(min_std - std).mean())
        start += c

    if not penalties:
        return torch.zeros((), device=z_seq.device)
    return torch.stack(penalties).mean()


def z_smoothness_loss(z_seq: torch.Tensor, same_engine_mask: torch.Tensor) -> torch.Tensor:
    """
    Phạt ĐỘ CONG (đạo hàm bậc 2) của Z theo thời gian, KHÔNG phải bản
    thân biến thiên (đạo hàm bậc 1) như bản cũ đã gây collapse. Z giảm
    ĐỀU (tuyến tính) -> đạo hàm bậc 2 = 0 -> loss này = 0, không phạt
    gì cả -- KHÔNG xung khắc với mono. Chỉ phạt khi tốc độ thay đổi của
    Z giật cục (tăng/giảm không đều giữa các bước liên tiếp).
    """
    n = z_seq.shape[0]
    seg_ids = _segment_ids_from_mask(same_engine_mask, n)
    _, counts = torch.unique_consecutive(seg_ids, return_counts=True)

    penalties = []
    start = 0
    for c_tensor in counts:
        c = int(c_tensor.item())
        if c >= 3:  # cần >=3 điểm mới có đạo hàm bậc 2
            seg = z_seq[start:start + c]
            d1 = seg[1:] - seg[:-1]
            d2 = d1[1:] - d1[:-1]
            penalties.append((d2 ** 2).sum(dim=-1).mean())
        start += c

    if not penalties:
        return torch.zeros((), device=z_seq.device)
    return torch.stack(penalties).mean()


def smart_ae_loss(x_hat: torch.Tensor, x: torch.Tensor,
                   z_seq: torch.Tensor, baseline: torch.Tensor,
                   same_engine_mask: torch.Tensor,
                   lambda_mono: float = 1.0,
                   lambda_anticollapse: float = 1.0,
                   lambda_smooth: float = 0.0,
                   min_std: float = 0.1,
                   per_window_normalize: bool = True) -> dict[str, torch.Tensor]:
    l_recon = reconstruction_loss(x_hat, x, per_window_normalize=per_window_normalize)
    l_mono = z_monotonicity_loss(z_seq, baseline, same_engine_mask)
    l_anticollapse = z_anti_collapse_loss(z_seq, same_engine_mask, min_std=min_std)
    l_smooth = z_smoothness_loss(z_seq, same_engine_mask)
    total = (l_recon + lambda_mono * l_mono + lambda_anticollapse * l_anticollapse
             + lambda_smooth * l_smooth)
    return {
        "total": total,
        "recon": l_recon,
        "mono": l_mono,
        "anticollapse": l_anticollapse,
        "smooth": l_smooth,
    }
