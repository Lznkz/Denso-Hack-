"""
model.py -- "Smart AE": Conv1D Autoencoder với kiến trúc tách W_hat/Z,
lấy cảm hứng từ Bajarunas et al. (PHM 2023), thay thế bản Conv1DAutoencoder
cũ (đã outdated).

THAY ĐỔI SO VỚI BẢN CŨ (breaking change, không tương thích ngược):
  - Bỏ hẳn tham số `condition` kiểu Conditional-AE (ghép W thật vào input
    encoder/decoder). Thay bằng kiến trúc encoder tách 2 nhánh Linear:
      - W_hat: encoder TỰ DỰ ĐOÁN lại operating condition từ X, không
        nhận W thật làm input ở bất kỳ đâu trong mạng.
      - Z: phần latent còn lại (z_dim chiều, không còn là 1 số duy nhất
        như bản gốc bài báo -- xem thảo luận đã tổng quát hóa lên k-dim),
        được ràng buộc riêng bằng loss smoothness/monotonicity (xem
        loss.py) để mã hóa mức độ suy thoái.
  - Decoder nhận [W_hat, Z] (KHÔNG phải W thật) để tái tạo X.
  - Lý do đổi: lúc inference thật ngoài đời, không cần đo W để chạy
    được model (Ŵ tự suy ra từ X) -- an toàn hơn nếu sau này DENSO
    không đo được operating condition đồng bộ 100% với sensor.
  - forward() giờ trả về (x_hat, w_hat, z) -- 3 giá trị, KHÁC bản cũ
    (2 giá trị: x_hat, z). Mọi nơi gọi model (train.py, metrics.py)
    cần cập nhật cách unpack.
"""

import torch
import torch.nn as nn


class SmartConv1DAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, window_len: int, w_dim: int,
                 z_dim: int = 8, hidden_channels: tuple[int, int] = (16, 8)):
        super().__init__()
        self.n_sensors = n_sensors
        self.window_len = window_len
        self.w_dim = w_dim
        self.z_dim = z_dim
        c1, c2 = hidden_channels

        # --- Encoder: (batch, n_sensors, window_len) -> (w_hat, z) ---
        self.enc_conv = nn.Sequential(
            nn.Conv1d(n_sensors, c1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self._flat_len = self._infer_flat_len(window_len)
        flat_dim = c2 * self._flat_len

        # Hai nhánh tách biệt -- đây là điểm khác biệt cốt lõi so với AE cũ.
        # Buộc encoder phải "giải thích" được W từ X trước, phần Z còn lại
        # (đã bị tước quyền "giải thích W") mới có động lực mã hóa phần dư
        # -- ứng viên tốt cho degradation.
        self.enc_w_head = nn.Linear(flat_dim, w_dim)
        self.enc_z_head = nn.Linear(flat_dim, z_dim)

        # --- Decoder: (w_hat, z) -> (batch, n_sensors, window_len) ---
        self.dec_fc = nn.Linear(w_dim + z_dim, flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(c2, c1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(c1, n_sensors, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
        self._c2 = c2

    def _infer_flat_len(self, window_len: int) -> int:
        # Suy ra bằng 1 lần forward dummy thay vì tính tay công thức
        # output_len của Conv1d -- tránh lỗi khi window_len không chia
        # hết đẹp cho 4 (giữ nguyên kỹ thuật tốt từ bản cũ).
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_sensors, window_len)
            out = self.enc_conv(dummy)
        return out.shape[-1]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, window_len, n_sensors) -> (w_hat, z).
        CHỈ nhận X -- không có tham số condition/W thật nào ở đây, khác
        hẳn bản Conditional-AE cũ."""
        h = self.enc_conv(x.transpose(1, 2))          # (batch, c2, flat_len)
        h = h.flatten(start_dim=1)                      # (batch, flat_dim)
        w_hat = self.enc_w_head(h)                       # (batch, w_dim)
        z = self.enc_z_head(h)                            # (batch, z_dim)
        return w_hat, z

    def decode(self, w_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(w_hat, z) -> reconstruction: (batch, window_len, n_sensors)."""
        h = torch.cat([w_hat, z], dim=-1)
        h = self.dec_fc(h)
        h = h.view(-1, self._c2, self._flat_len)
        out = self.dec_conv(h)                           # (batch, n_sensors, ~window_len)
        out = out[:, :, :self.window_len]                 # cắt đúng độ dài nếu lệch do làm tròn
        return out.transpose(1, 2)                          # (batch, window_len, n_sensors)

    def forward(self, x: torch.Tensor):
        """Trả về (x_hat, w_hat, z) -- 3 giá trị, KHÁC bản cũ (2 giá trị).
        train.py cần w_hat + z để tính loss; metrics.py cần cả 3 để tính
        reconstruction error (phụ) và z_drift (chính)."""
        w_hat, z = self.encode(x)
        x_hat = self.decode(w_hat, z)
        return x_hat, w_hat, z
