"""
phase1_reconstruction.py -- ADAPTER CẦU NỐI GIỮA PHASE I PYTORCH VÀ PROTOCOL RECONSTRUCTION PROVIDER

Đảm nhận:
  1. Giao tiếp PyTorch (device, tensors, no_grad).
  2. Hỗ trợ 2 chế độ tái tạo đối xứng:
       - POOLED_Z: Lặp lại vector Z pooled (z_dim,) thành (n_subwindows, z_dim).
       - SUBWINDOW_Z: Sử dụng Z_subwindows kích thước (n_subwindows, z_dim).
  3. Chuẩn hóa layout đầu ra về đúng canonical: (n_subwindows, n_bins, n_sensors).
  4. Trả về mảng NumPy thuần túy cho Core Localization.
"""

from __future__ import annotations
import numpy as np
import torch

from src.localization.schema import ReconstructionMode


class PhaseIReconstructionAdapter:
    """Hiện thực hóa Protocol ReconstructionProvider cho SmartConv1DAutoencoder của Phase I."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def reconstruct(
        self,
        W: np.ndarray,
        Z: np.ndarray,
        *,
        n_subwindows: int,
        Z_subwindows: np.ndarray | None = None,
        mode: ReconstructionMode = "POOLED_Z",
    ) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            w_arr = np.asarray(W, dtype=np.float32)
            if w_arr.ndim == 1:
                w_arr = np.repeat(w_arr[None, :], n_subwindows, axis=0)
            elif w_arr.shape[0] != n_subwindows:
                raise ValueError(
                    f"W có {w_arr.shape[0]} subwindows, không khớp n_subwindows={n_subwindows}"
                )

            if mode == "SUBWINDOW_Z":
                if Z_subwindows is None:
                    raise ValueError("mode='SUBWINDOW_Z' đòi hỏi Z_subwindows không được là None.")
                z_arr = np.asarray(Z_subwindows, dtype=np.float32)
                if z_arr.shape[0] != n_subwindows:
                    raise ValueError(
                        f"Z_subwindows có {z_arr.shape[0]} subwindows, không khớp n_subwindows={n_subwindows}"
                    )
            elif mode == "POOLED_Z":
                z_pooled = np.asarray(Z, dtype=np.float32)
                if z_pooled.ndim != 1:
                    raise ValueError(f"mode='POOLED_Z' đòi hỏi Z là vector 1D, nhận shape {z_pooled.shape}")
                z_arr = np.repeat(z_pooled[None, :], n_subwindows, axis=0)
            else:
                raise ValueError(
                    f"Mode tái tạo không được hỗ trợ: '{mode}'. Phải là 'POOLED_Z' hoặc 'SUBWINDOW_Z'."
                )

            w_t = torch.from_numpy(w_arr).to(self.device)
            z_t = torch.from_numpy(z_arr).to(self.device)

            # Model Phase I decode(w, z) trả về: (batch, window_len, n_sensors)
            # tương ứng đúng (n_subwindows, n_bins, n_sensors)
            x_hat_t = self.model.decode(w_t, z_t)
            return x_hat_t.detach().cpu().numpy().astype(np.float32)
