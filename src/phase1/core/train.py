"""
train.py -- huấn luyện Smart AE + trích xuất latent cho Phase II.
Thay thế bản train.py cũ (train pure-healthy, shuffle window hoàn toàn).

THAY ĐỔI SO VỚI BẢN CŨ:
  - Train trên TOÀN BỘ trajectory mỗi engine (healthy + suy thoái trộn
    lẫn), KHÔNG chỉ pure healthy -- quyết định đã thống nhất: dữ liệu
    DENSO giả định luôn có 1 phần suy thoái (bị cắt trước điểm risk),
    và ràng buộc smoothness/monotonicity hoạt động cục bộ nên không cần
    biết ranh giới khỏe/hỏng để train đúng.
  - Batch theo ENGINE (giữ nguyên thứ tự cycle bên trong mỗi engine),
    KHÔNG shuffle window ngẫu nhiên hoàn toàn như bản cũ -- bắt buộc vì
    loss smoothness/monotonicity cần biết đúng cặp (Z_t, Z_{t-1}) liên
    tiếp cùng 1 engine. Chỉ thứ tự CÁC ENGINE trong mỗi epoch được xáo
    trộn, không xáo trộn window bên trong 1 engine.
  - model.forward giờ trả (x_hat, w_hat, z) -- 3 giá trị, không phải 2.

File này vẫn KHÔNG xác định healthy region, KHÔNG fit normalization --
các bước đó xử lý ở pipeline chuẩn bị dữ liệu bên ngoài, trước khi tạo
ra list[EngineData] truyền vào run_training().
"""

from __future__ import annotations

import numpy as np
import torch

from .windowing import EngineData, WindowBatch, cut_windows_for_engine, concat_window_batches
from .model import SmartConv1DAutoencoder
from .loss import smart_ae_loss


def _build_same_engine_mask(unit_number: np.ndarray) -> torch.Tensor:
    """unit_number: (n_windows,) ĐÃ sắp xếp theo (unit_number, cycle).
    Trả về (n_windows-1,) bool tensor -- True nếu window i và i+1 cùng
    1 engine. Dùng để loại các cặp bị "lẫn" ở ranh giới 2 engine khác
    nhau khi đã nối nhiều engine lại thành 1 chuỗi dài trong cùng batch."""
    same = unit_number[1:] == unit_number[:-1]
    return torch.from_numpy(same)


def _prepare_epoch_batches(engines: list[EngineData], window_len: int, stride: int,
                            engines_per_batch: int, rng: np.random.Generator) -> list[WindowBatch]:
    """Xáo trộn THỨ TỰ CÁC ENGINE (không xáo trộn window bên trong 1
    engine), rồi gộp nhóm engines_per_batch engine lại thành 1 batch.
    Mỗi batch được sort lại theo (unit_number, cycle) để mask ranh giới
    hoạt động đúng, kể cả khi engine trong nhóm có độ dài khác nhau."""
    order = rng.permutation(len(engines))
    batches = []
    for i in range(0, len(order), engines_per_batch):
        group_idx = order[i:i + engines_per_batch]
        group_windows = [cut_windows_for_engine(engines[j], window_len, stride) for j in group_idx]
        group_windows = [w for w in group_windows if w.X.shape[0] > 0]
        if not group_windows:
            continue
        merged = concat_window_batches(group_windows)
        sort_idx = np.lexsort((merged.cycle, merged.unit_number))
        batches.append(WindowBatch(
            X=merged.X[sort_idx], W=merged.W[sort_idx],
            unit_number=merged.unit_number[sort_idx], cycle=merged.cycle[sort_idx],
        ))
    return batches


@torch.no_grad()
def _evaluate(model: SmartConv1DAutoencoder, engines: list[EngineData], window_len: int,
              stride: int, engines_per_batch: int, device: str,
              lambda_w: float, lambda_smooth: float, lambda_mono: float,
              rng: np.random.Generator) -> float:
    """Tính total loss trung bình trên 1 tập engine, KHÔNG cập nhật
    trọng số (dùng cho validation). Thứ tự xáo trộn không ảnh hưởng tới
    giá trị loss trung bình cuối cùng nên tái sử dụng _prepare_epoch_batches
    bình thường, không cần logic riêng."""
    model.eval()
    batches = _prepare_epoch_batches(engines, window_len, stride, engines_per_batch, rng)
    total_loss = 0.0
    n_batches = 0
    for batch in batches:
        x = torch.from_numpy(batch.X).float().to(device)
        w_true = torch.from_numpy(batch.W).float().to(device)
        mask = _build_same_engine_mask(batch.unit_number).to(device)
        x_hat, w_hat, z = model(x)
        losses = smart_ae_loss(x_hat, x, w_hat, w_true, z, mask,
                                lambda_w=lambda_w, lambda_smooth=lambda_smooth,
                                lambda_mono=lambda_mono)
        total_loss += losses["total"].item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(engines: list[EngineData], n_sensors: int, w_dim: int, window_len: int,
                  val_engines: list[EngineData] | None = None,
                  stride: int = 1, z_dim: int = 8, n_epochs: int = 50, lr: float = 1e-3,
                  engines_per_batch: int = 8, seed: int = 0, device: str = "cpu",
                  lambda_w: float = 1.0, lambda_smooth: float = 1.0,
                  lambda_mono: float = 1.0, patience: int = 5,
                  min_delta: float = 1e-5) -> SmartConv1DAutoencoder:
    """
    engines: list các EngineData -- MỌI engine trong tập train (healthy +
    suy thoái trộn lẫn), đã chuẩn hóa và tách W từ trước bởi pipeline bên
    ngoài. KHÔNG lọc riêng pure-healthy ở đây (khác bản cũ).

    val_engines: MỘT SỐ engine TÁCH RIÊNG khỏi `engines`, không dùng để
    cập nhật trọng số, chỉ dùng để theo dõi early stopping. Nếu để
    None, train đủ n_epochs, không early-stop (giống hành vi bản trước
    khi thêm tính năng này).

    patience/min_delta: dừng training khi validation loss không cải
    thiện thêm ít nhất `min_delta` trong `patience` epoch liên tiếp --
    CHỐNG OVERFITTING theo cách không phụ thuộc scale loss cụ thể,
    khác với cách "dừng khi loss < ngưỡng tuyệt đối" (không dùng ở đây
    vì ngưỡng tuyệt đối của 1 bài báo khác không có ý nghĩa gì với tổ
    hợp loss 4 thành phần của bạn -- xem thảo luận).

    engines_per_batch: số ENGINE gộp vào 1 batch (không phải số window)
    -- vì mỗi engine có số cycle khác nhau, "batch size" tính theo
    engine để logic sort/mask dễ suy luận và đúng đắn.

    Trả về model TẠI THỜI ĐIỂM VALIDATION LOSS TỐT NHẤT (nếu có
    val_engines) -- không phải model ở epoch cuối cùng, để tránh trả về
    1 model đã overfit thêm vài epoch sau điểm tốt nhất.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)  # rng riêng cho eval, tách khỏi rng train

    model = SmartConv1DAutoencoder(n_sensors=n_sensors, window_len=window_len,
                                    w_dim=w_dim, z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        batches = _prepare_epoch_batches(engines, window_len, stride, engines_per_batch, rng)
        epoch_losses = {"total": 0.0, "recon": 0.0, "w_pred": 0.0, "smooth": 0.0, "mono": 0.0}
        n_batches = 0

        for batch in batches:
            x = torch.from_numpy(batch.X).float().to(device)
            w_true = torch.from_numpy(batch.W).float().to(device)
            mask = _build_same_engine_mask(batch.unit_number).to(device)

            optimizer.zero_grad()
            x_hat, w_hat, z = model(x)
            losses = smart_ae_loss(x_hat, x, w_hat, w_true, z, mask,
                                    lambda_w=lambda_w, lambda_smooth=lambda_smooth,
                                    lambda_mono=lambda_mono)
            losses["total"].backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        log = " | ".join(f"{k}={v / max(n_batches, 1):.6f}" for k, v in epoch_losses.items())

        if val_engines is not None:
            val_loss = _evaluate(model, val_engines, window_len, stride, engines_per_batch,
                                  device, lambda_w, lambda_smooth, lambda_mono, val_rng)
            print(f"epoch {epoch:3d} | {log} | val_total={val_loss:.6f}")

            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping tại epoch {epoch} "
                      f"(val loss không cải thiện {patience} epoch liên tiếp).")
                break
        else:
            print(f"epoch {epoch:3d} | {log}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def extract_latents_for_engine(model: SmartConv1DAutoencoder, engine: EngineData,
                                window_len: int, bin_stride: int,
                                device: str = "cpu") -> WindowBatch:
    """
    Trả về WindowBatch với field X THAY BẰNG Z (n_bins, z_dim) -- giữ
    nguyên unit_number/cycle đi kèm để Phase II ráp đúng thứ tự chuỗi
    khi các engine chạy bất đồng bộ. Field W của kết quả trả về là
    W_hat (encoder tự đoán), không phải W thật -- hữu ích để debug xem
    encoder có đang đoán W hợp lý không.

    LƯU Ý: field .X của WindowBatch trả về thực chất LÀ latent Z, không
    phải sensor thô -- cần nhớ khi dùng ở nơi gọi (dễ nhầm lẫn vì tên
    field giữ nguyên để tái sử dụng struct).
    """
    model.eval()
    windows = cut_windows_for_engine(engine, window_len=window_len, stride=bin_stride)

    if windows.X.shape[0] == 0:
        return WindowBatch(
            X=np.empty((0, model.z_dim), dtype=np.float32),
            W=windows.W, unit_number=windows.unit_number, cycle=windows.cycle,
        )

    x = torch.from_numpy(windows.X).float().to(device)
    _, w_hat, z = model(x)

    return WindowBatch(
        X=z.cpu().numpy(), W=w_hat.cpu().numpy(),
        unit_number=windows.unit_number, cycle=windows.cycle,
    )
