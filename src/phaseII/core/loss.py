"""
loss.py -- discrete-time hazard loss + monotonicity regularizer.

Công thức BCE-theo-bin ở đây CHÍNH XÁC là negative log-likelihood của mô
hình discrete-time hazard đã suy ra từ S(k) = prod(1 - h(t)) -- không phải
một xấp xỉ. Xem lại phần suy diễn trong hội thoại nếu cần đối chiếu lại.

Mọi phép tính đều nhân qua `mask` -- vị trí padding (mask=0) không bao giờ
đóng góp vào loss, dù giá trị h tại đó vẫn được model tính ra (rẻ, vô hại
nếu không dùng).
"""

import torch


def discrete_time_hazard_loss(h: torch.Tensor, y: torch.Tensor,
                               mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    h, y, mask: (batch, T_max)

    y[i, t] = 1 tại đúng bin xảy ra sự kiện của engine i, 0 mọi nơi khác
    (bao gồm toàn bộ chuỗi nếu engine đó bị censored -- xem data.py).

    Trả về loss trung bình trên tổng số bin THẬT (không tính padding),
    không phải trung bình trên batch*T_max (sẽ làm loss bị pha loãng sai
    nếu các sequence có độ dài rất khác nhau).
    """
    h = torch.clamp(h, eps, 1 - eps)
    bce = -(y * torch.log(h) + (1 - y) * torch.log(1 - h))
    bce = bce * mask
    return bce.sum() / mask.sum().clamp(min=1.0)


# def discrete_time_hazard_loss(h: torch.Tensor, y: torch.Tensor,
#                                mask: torch.Tensor, eps: float = 1e-7,
#                                pos_weight: float = 3.0) -> torch.Tensor:
#     """
#     pos_weight > 1: phạt nặng hơn khi model đoán h THẤP tại đúng bin có
#     event thật (y=1) -- tức phạt nặng hành vi "dự đoán muộn/overestimate".
#     pos_weight = 1: quay lại đúng công thức đối xứng cũ.

#     Đây là mở rộng trực tiếp của likelihood đã suy diễn trước đó -- về
#     bản chất là weighted likelihood, một kỹ thuật chuẩn khi chi phí hai
#     loại lỗi (bỏ sót vs báo giả) không cân bằng nhau, đúng thực tế bảo
#     trì: bỏ sót một event thật tốn kém hơn nhiều một lần báo giả.
#     """
#     h = torch.clamp(h, eps, 1 - eps)
#     bce = -(pos_weight * y * torch.log(h) + (1 - y) * torch.log(1 - h))
#     bce = bce * mask
#     return bce.sum() / mask.sum().clamp(min=1.0)

def monotonicity_penalty(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Phạt khi h(t+1) < h(t) mà không có lý do (không có sự kiện can thiệp
    giữa hai bin) -- ép mô hình học đúng xu hướng suy thoái không tự hồi
    phục. Chỉ tính trên cặp bin (t, t+1) đều là bin THẬT của cùng engine.
    """
    h_t = h[:, :-1]
    h_t1 = h[:, 1:]
    pair_mask = mask[:, :-1] * mask[:, 1:]

    drop = torch.clamp(h_t - h_t1, min=0.0)  # >0 chỉ khi hazard giảm ngược
    penalty = (drop * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
    return penalty


def combined_loss(h: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                   lam_monotonic: float = 0.1) -> dict[str, torch.Tensor]:
    """
    Trả về dict để log riêng từng thành phần (rất nên làm -- nếu monotonic
    penalty áp đảo hazard loss trong log, lam_monotonic đang để quá cao).
    """
    hazard = discrete_time_hazard_loss(h, y, mask)
    mono = monotonicity_penalty(h, mask)
    total = hazard + lam_monotonic * mono
    return {"total": total, "hazard": hazard, "monotonic": mono}
