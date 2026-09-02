"""
train.py -- huấn luyện Smart AE + trích xuất latent cho Phase II.
Cập nhật theo kiến trúc đơn giản hóa (model.py/loss.py mới): model nhận
thêm W làm input cho decoder, không còn nhánh/loss dự đoán Ŵ.

THAY ĐỔI SO VỚI BẢN TRƯỚC:
  - model(x, w) -- giờ BẮT BUỘC truyền w khi gọi model, ở cả train lẫn
    extract_latents_for_engine (dù ở bước sau w không thực sự cần thiết
    cho mục đích cuối, model vẫn yêu cầu đủ tham số để forward chạy được
    vì decode() luôn cần w).
  - forward() trả về (x_hat, z) -- 2 giá trị, không phải 3.
  - smart_ae_loss không còn tham số w_hat/w_true/lambda_w.
  - epoch_losses không còn key "w_pred".
"""

import numpy as np
import torch

from src.phaseI.core.windowing import EngineData, WindowBatch, cut_windows_for_engine, concat_window_batches
from src.phaseI.core.model import SmartConv1DAutoencoder
from src.phaseI.core.loss import smart_ae_loss


def _build_same_engine_mask(unit_number: np.ndarray) -> torch.Tensor:
    """Không đổi so với bản trước."""
    same = unit_number[1:] == unit_number[:-1]
    return torch.from_numpy(same)


def _prepare_epoch_batches(engines: list[EngineData], window_len: int, stride: int,
                            engines_per_batch: int, rng: np.random.Generator) -> list[WindowBatch]:
    """Không đổi so với bản trước."""
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
              lambda_smooth: float, lambda_mono: float,
              rng: np.random.Generator) -> float:
    """model(x, w) -- giờ cần truyền cả w. smart_ae_loss không còn nhận
    w_hat/w_true/lambda_w."""
    model.eval()
    batches = _prepare_epoch_batches(engines, window_len, stride, engines_per_batch, rng)
    total_loss = 0.0
    n_batches = 0
    for batch in batches:
        x = torch.from_numpy(batch.X).float().to(device)
        w = torch.from_numpy(batch.W).float().to(device)
        mask = _build_same_engine_mask(batch.unit_number).to(device)
        x_hat, z = model(x, w)
        losses = smart_ae_loss(x_hat=x_hat, x=x, z_seq=z, same_engine_mask=mask,
                                lambda_smooth=lambda_smooth, lambda_mono=lambda_mono)
        total_loss += losses["total"].item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(engines: list[EngineData], n_sensors: int, w_dim: int, window_len: int,
                  val_engines: list[EngineData] | None = None,
                  stride: int = 1, z_dim: int = 8, n_epochs: int = 50, lr: float = 1e-3,
                  engines_per_batch: int = 8, seed: int = 0, device: str = "cpu",
                  lambda_smooth: float = 1.0, lambda_mono: float = 1.0,
                  patience: int = 5, min_delta: float = 1e-5) -> SmartConv1DAutoencoder:
    """
    Không còn lambda_w -- không còn loss nào liên quan tới việc dự đoán W.
    Mọi tham số khác giữ nguyên ý nghĩa như bản trước.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)

    model = SmartConv1DAutoencoder(n_sensors=n_sensors, window_len=window_len,
                                    w_dim=w_dim, z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        batches = _prepare_epoch_batches(engines, window_len, stride, engines_per_batch, rng)
        epoch_losses = {"total": 0.0, "recon": 0.0, "smooth": 0.0, "mono": 0.0}
        n_batches = 0

        for batch in batches:
            x = torch.from_numpy(batch.X).float().to(device)
            w = torch.from_numpy(batch.W).float().to(device)
            mask = _build_same_engine_mask(batch.unit_number).to(device)

            optimizer.zero_grad()
            x_hat, z = model(x, w)
            losses = smart_ae_loss(x_hat=x_hat, x=x, z_seq=z, same_engine_mask=mask,
                                    lambda_smooth=lambda_smooth, lambda_mono=lambda_mono)
            losses["total"].backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        log = " | ".join(f"{k}={v / max(n_batches, 1):.6f}" for k, v in epoch_losses.items())

        if val_engines is not None:
            val_loss = _evaluate(model, val_engines, window_len, stride, engines_per_batch,
                                  device, lambda_smooth, lambda_mono, val_rng)
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
    Trả về WindowBatch với field X THAY BẰNG Z. Field W giờ là W THẬT
    (không còn là Ŵ dự đoán như bản trước, vì model không còn sinh ra
    Ŵ nào cả) -- giữ nguyên nguyên trạng windows.W, không qua model.
    """
    model.eval()
    windows = cut_windows_for_engine(engine, window_len=window_len, stride=bin_stride)

    if windows.X.shape[0] == 0:
        return WindowBatch(
            X=np.empty((0, model.z_dim), dtype=np.float32),
            W=windows.W, unit_number=windows.unit_number, cycle=windows.cycle,
        )

    x = torch.from_numpy(windows.X).float().to(device)
    w = torch.from_numpy(windows.W).float().to(device)
    _, z = model(x, w)

    return WindowBatch(
        X=z.cpu().numpy(), W=windows.W,
        unit_number=windows.unit_number, cycle=windows.cycle,
    )
