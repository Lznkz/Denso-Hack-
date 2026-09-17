"""
model.py -- "Smart AE": Conv1D Autoencoder.
CẬP NHẬT so với bản trước: decode()/forward() nhận thêm w_dropout_p --
dropout trên toàn bộ vector W của từng sample (không phải dropout từng
phần tử riêng lẻ), theo lịch giảm dần điều khiển từ train.py, để buộc
decoder không thể luôn dựa hoàn toàn vào W mà bỏ qua Z.

sửa lần 2: hidden_channels giờ nhận TUPLE ĐỘ DÀI TÙY Ý (trước đây cố
định 2 lớp) -- encoder/decoder build bằng loop thay vì hardcode. Default
đổi từ (16, 8) thành (128, 64, 16), gần khớp kiến trúc paper (Bajarunas
et al.) làm điểm khởi đầu. LƯU Ý: paper tái tạo sensor reading đã mượt
(14 sensor, entropy thấp), còn ở đây tái tạo raw vibration broadband
(entropy cao hơn nhiều) -- nên default này là điểm XUẤT PHÁT để tăng
thêm, không phải đích cần đạt, và z_dim KHÔNG nên theo z_dim=1 của
paper (xem giải thích trong các lượt trao đổi trước -- z_dim của AE và
"Z 1D dùng làm Health Index" là hai việc tách biệt).
"""

import torch
import torch.nn as nn


class SmartConv1DAutoencoder(nn.Module):
    def __init__(self, n_sensors: int, window_len: int, w_dim: int,
                 z_dim: int = 8, hidden_channels: tuple[int, ...] = (128, 64, 16)):
        super().__init__()
        self.n_sensors = n_sensors
        self.window_len = window_len
        self.w_dim = w_dim
        self.z_dim = z_dim
        self.hidden_channels = tuple(hidden_channels)

        # Encoder: n_sensors -> hidden_channels[0] -> ... -> hidden_channels[-1]
        enc_layers = []
        in_ch = n_sensors
        for out_ch in self.hidden_channels:
            enc_layers += [nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2), nn.ReLU()]
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc_layers)

        self._flat_len = self._infer_flat_len(window_len)
        self._c_last = self.hidden_channels[-1]
        flat_dim = self._c_last * self._flat_len

        self.enc_z_head = nn.Linear(flat_dim, z_dim)

        self.dec_fc = nn.Linear(w_dim + z_dim, flat_dim)

        # Decoder: hidden_channels[-1] -> ... -> hidden_channels[0] -> n_sensors
        dec_layers = []
        rev_channels = list(reversed(self.hidden_channels))
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            dec_layers += [nn.ConvTranspose1d(in_ch, out_ch, kernel_size=5, stride=2,
                                               padding=2, output_padding=1), nn.ReLU()]
            in_ch = out_ch
        dec_layers += [nn.ConvTranspose1d(in_ch, n_sensors, kernel_size=5, stride=2,
                                           padding=2, output_padding=1)]
        self.dec_conv = nn.Sequential(*dec_layers)

    def _infer_flat_len(self, window_len: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_sensors, window_len)
            out = self.enc_conv(dummy)
        return out.shape[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_len, n_sensors) -> z: (batch, z_dim).
        Hàm DUY NHẤT cần gọi lúc inference thực tế (deployment)."""
        h = self.enc_conv(x.transpose(1, 2))
        h = h.flatten(start_dim=1)
        return self.enc_z_head(h)

    def decode(self, w: torch.Tensor, z: torch.Tensor,
               w_dropout_p: float = 0.0) -> torch.Tensor:
        """(w, z) -> reconstruction: (batch, window_len, n_sensors).

        w_dropout_p: xác suất "tắt" TOÀN BỘ vector W của 1 sample (không
        tắt từng chiều riêng lẻ -- tắt riêng lẻ dễ bị model "lách" bằng
        cách suy ra chiều bị tắt từ các chiều còn lại của chính W).
        CHỈ áp dụng khi self.training=True; lúc eval() dropout tự tắt
        bất kể giá trị truyền vào.
        """
        if self.training and w_dropout_p > 0:
            mask = (torch.rand(w.shape[0], 1, device=w.device) > w_dropout_p).float()
            w = w * mask
        h = torch.cat([w, z], dim=-1)
        h = self.dec_fc(h)
        h = h.view(-1, self._c_last, self._flat_len)
        out = self.dec_conv(h)
        out = out[:, :, :self.window_len]
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor, w_dropout_p: float = 0.0):
        """
        Trả về (x_hat, z).
        w_dropout_p: xem decode(). Mặc định 0.0 -- an toàn khi gọi từ
        nơi không cần dropout (ví dụ extract_latents_for_engine), vì
        model.eval() cũng tự tắt dropout dù có truyền giá trị khác 0.
        """
        z = self.encode(x)
        x_hat = self.decode(w, z, w_dropout_p=w_dropout_p)
        return x_hat, z
