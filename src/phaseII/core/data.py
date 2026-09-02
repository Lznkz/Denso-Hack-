from dataclasses import dataclass
import numpy as np


@dataclass
class EngineRecord:
    """Một engine trong development set."""
    engine_id: str
    latent_seq: np.ndarray   # shape (T, latent_dim) -- output encoder Phase I, ĐÃ cắt tại [:event_bin+1] nếu có event
    n_bins: int              # số bin thực sự dùng = len(latent_seq); bằng event_bin+1 nếu có event
    event_bin: int | None    # from z_drift


def build_hazard_target(n_bins: int, event_bin: int | None) -> np.ndarray:
    """
    Trả về vector nhãn y có độ dài n_bins.

    - Nếu event_bin is None (censored): toàn bộ y = 0. Engine chưa đạt đến
      degradation onset trong cửa sổ quan sát -- tất cả các bin đều là
      "chưa có gì xảy ra".
    - Nếu có event_bin: y = 0 ở mọi bin trước đó, y = 1 đúng tại event_bin
      (điểm chuyển từ healthy sang degradation).
      Engine bị REMOVE KHỎI RISK SET ngay sau event_bin -- caller có trách
      nhiệm đảm bảo latent_seq đã được cắt tại [:event_bin+1] trước khi
      tạo EngineRecord, do đó n_bins = event_bin + 1. Không có bin nào
      sau event_bin trong sequence.
    """
    y = np.zeros(n_bins, dtype=np.float32)
    if event_bin is not None:
        assert 0 <= event_bin < n_bins, "event_bin phải nằm trong [0, n_bins)"
        y[event_bin] = 1.0
    return y


def pad_batch(records: list[EngineRecord], latent_dim: int):
    """
    Gói một list EngineRecord (độ dài khác nhau) thành ba tensor cùng shape
    (batch, T_max, ...), kèm mask (batch, T_max) đánh dấu bin nào là thật.

    Trả về numpy arrays thuần (không phụ thuộc torch) -- converter sang
    torch.Tensor để ở ngoài file này, giữ data.py độc lập framework.
    """
    batch_size = len(records)
    t_max = max(r.n_bins for r in records)

    latents = np.zeros((batch_size, t_max, latent_dim), dtype=np.float32)
    targets = np.zeros((batch_size, t_max), dtype=np.float32)
    mask = np.zeros((batch_size, t_max), dtype=np.float32)

    for i, r in enumerate(records):
        t = r.n_bins
        latents[i, :t, :] = r.latent_seq[:t]
        targets[i, :t] = build_hazard_target(t, r.event_bin)
        mask[i, :t] = 1.0

    return latents, targets, mask
