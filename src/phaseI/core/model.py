import torch
import torch.nn as nn


class SmartConv1DAutoencoder(nn.Module):
    """
    Conditional Conv1D Autoencoder cho HI branch sau SPEV preprocessing.

    Input
    -----
    x:
        Standardized log-SPEV features, shape:
            (batch, n_sensors, n_buckets)

        Với preprocess hiện tại thường là:
            (batch, 2, 20)

    w:
        Standardized operating conditions, shape:
            (batch, w_dim)

        W là đầu vào BẮT BUỘC của decoder:

            X_tilde -> Encoder -> Z
                                  \
                                   concat(Z, W_tilde)
                                          |
                                          v
                                       Decoder
                                          |
                                          v
                                      X_hat_tilde

        Encoder không nhận W. Mục tiêu là để encoder rút representation Z
        từ X, còn decoder sử dụng W để giải thích phần biến thiên của X do
        operating condition gây ra.

    Output
    ------
    x_hat:
        Reconstruction của standardized log-SPEV X, cùng shape với x.

    z:
        Latent health representation, shape (batch, z_dim).

    Architecture notes
    ------------------
    SPEV hiện chỉ còn khoảng 20 bucket/channel, nên model được giữ nhỏ:
        Encoder channels mặc định: 2 -> 32 -> 16 -> 8
        Decoder channels mặc định: 8 -> 16 -> 32 -> 2

    Tất cả Conv1D dùng stride=1 và giữ nguyên chiều dài bucket axis.
    Encoder dùng LeakyReLU thay vì ReLU để tránh dying-ReLU trên input
    log-SPEV đã StandardScaler (có cả giá trị âm và dương). Decoder vẫn
    giữ ReLU; output reconstruction cuối cùng là linear.

    Không pooling/downsampling vì representation đầu vào đã rất compact.
    Bottleneck thật sự của model nằm ở Z, không nằm ở việc ép 20 bucket
    xuống một feature map rất ngắn.
    """

    def __init__(
        self,
        n_sensors: int,
        window_len: int,
        w_dim: int,
        z_dim: int = 1,
        hidden_channels: tuple[int, ...] = (
            32,
            16,
            8,
        ),
        kernel_size: int = 3,
    ):
        super().__init__()

        # ====================================================
        # VALIDATION
        # ====================================================

        if n_sensors <= 0:
            raise ValueError("n_sensors phải > 0")

        if window_len <= 0:
            raise ValueError("window_len phải > 0")

        if w_dim <= 0:
            raise ValueError("w_dim phải > 0")

        if z_dim <= 0:
            raise ValueError("z_dim phải > 0")

        if len(hidden_channels) == 0:
            raise ValueError("hidden_channels không được rỗng")

        if any(int(c) <= 0 for c in hidden_channels):
            raise ValueError("mọi hidden channel phải > 0")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size phải là số lẻ > 0")

        # ====================================================
        # CONFIG
        # ====================================================

        self.n_sensors = int(n_sensors)

        # Giữ tên ``window_len`` để tương thích train.py/main.py hiện tại.
        # Semantic mới: số SPEV buckets, KHÔNG còn là FFT window length.
        self.window_len = int(window_len)
        self.feature_len = self.window_len

        self.w_dim = int(w_dim)
        self.z_dim = int(z_dim)
        self.kernel_size = int(kernel_size)
        self.hidden_channels = tuple(int(c) for c in hidden_channels)

        padding = self.kernel_size // 2

        # ====================================================
        # ENCODER
        #
        # (B, sensors, buckets)
        #   -> Conv1D, giữ nguyên bucket axis
        #   -> ...
        #   -> flatten
        #   -> Z
        #
        # W cố ý KHÔNG đi vào encoder.
        # ====================================================

        encoder_layers = []
        in_channels = self.n_sensors

        for out_channels in self.hidden_channels:
            encoder_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=padding,
                ),
                # X da duoc StandardScaler, nen hidden pre-activation co the am.
                # LeakyReLU giu mot gradient/duong tin hieu nho o nhanh am,
                # tranh dying-ReLU lam ca feature map encoder ve 0.
                nn.LeakyReLU(
                    negative_slope=0.01,
                    inplace=True,
                ),
            ])
            in_channels = out_channels

        self.enc_conv = nn.Sequential(*encoder_layers)

        self._c_last = self.hidden_channels[-1]
        self._flat_len = self.feature_len
        self._flat_dim = self._c_last * self._flat_len

        self.enc_z_head = nn.Linear(
            self._flat_dim,
            self.z_dim,
        )

        # ====================================================
        # CONDITIONAL DECODER INPUT
        #
        # [Z, W_tilde] -> FC -> (B, C_last, buckets)
        #
        # Không còn nhánh decoder chỉ dùng Z.
        # ====================================================

        decoder_input_dim = self.z_dim + self.w_dim

        self.dec_fc = nn.Linear(
            decoder_input_dim,
            self._flat_dim,
        )

        # ====================================================
        # DECODER
        #
        # Với default hidden_channels=(32,16,8):
        #   FC -> (B, 8, buckets)
        #      -> Conv 8 -> 16
        #      -> Conv 16 -> 32
        #      -> Conv 32 -> n_sensors
        #
        # Giữ nguyên bucket axis. Không ConvTranspose/up-sampling.
        # ====================================================

        reversed_channels = list(reversed(self.hidden_channels))
        decoder_layers = []
        in_channels = reversed_channels[0]

        for out_channels in reversed_channels[1:]:
            decoder_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=padding,
                ),
                nn.ReLU(),
            ])
            in_channels = out_channels

        # X đã log1p + StandardScaler nên target reconstruction có thể âm
        # hoặc dương. Final layer vì vậy để linear, không sigmoid/ReLU.
        decoder_layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.n_sensors,
                kernel_size=self.kernel_size,
                stride=1,
                padding=padding,
            )
        )

        self.dec_conv = nn.Sequential(*decoder_layers)

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_x(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x phải có shape "
                "(batch, n_sensors, n_buckets), "
                f"nhận {tuple(x.shape)}"
            )

        if x.shape[1] != self.n_sensors:
            raise ValueError(
                f"Expected {self.n_sensors} sensors, "
                f"nhận {x.shape[1]}"
            )

        if x.shape[2] != self.feature_len:
            raise ValueError(
                f"Expected n_buckets={self.feature_len}, "
                f"nhận {x.shape[2]}"
            )

    def _validate_w(
        self,
        w: torch.Tensor,
        batch_size: int,
    ) -> None:
        if w is None:
            raise ValueError(
                "Conditional decoder bắt buộc nhận W, nhưng w=None"
            )

        if w.ndim != 2:
            raise ValueError(
                "w phải có shape (batch, w_dim), "
                f"nhận {tuple(w.shape)}"
            )

        if w.shape[0] != batch_size:
            raise ValueError("x/z và w khác batch size")

        if w.shape[1] != self.w_dim:
            raise ValueError(
                f"Expected w_dim={self.w_dim}, "
                f"nhận {w.shape[1]}"
            )

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_sensors, n_buckets)
        z: (batch, z_dim)
        """

        self._validate_x(x)

        h = self.enc_conv(x)
        h = h.flatten(start_dim=1)
        z = self.enc_z_head(h)

        return z

    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        z: torch.Tensor,
        w: torch.Tensor,
        w_dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Conditional decoder bắt buộc:
            (Z, W_tilde) -> X_hat_tilde

        ``w_dropout_p`` chỉ là regularization tùy chọn trên W; architecture
        vẫn luôn là conditional decoder và không có nhánh no-W.
        Main experiment nên để 0 nếu không chủ động kiểm tra regularization.
        """

        if z.ndim != 2:
            raise ValueError(
                "z phải có shape (batch, z_dim), "
                f"nhận {tuple(z.shape)}"
            )

        if z.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected z_dim={self.z_dim}, "
                f"nhận {z.shape[1]}"
            )

        self._validate_w(
            w,
            batch_size=z.shape[0],
        )

        if not (0.0 <= float(w_dropout_p) <= 1.0):
            raise ValueError(
                "w_dropout_p phải nằm trong [0, 1]"
            )

        w_used = w

        if self.training and w_dropout_p > 0.0:
            mask = (
                torch.rand(
                    w.shape[0],
                    1,
                    device=w.device,
                )
                > w_dropout_p
            ).to(w.dtype)
            w_used = w * mask

        h = torch.cat(
            [z, w_used],
            dim=1,
        )

        h = self.dec_fc(h)
        h = h.view(
            -1,
            self._c_last,
            self._flat_len,
        )

        x_hat = self.dec_conv(h)

        expected_shape = (
            z.shape[0],
            self.n_sensors,
            self.feature_len,
        )

        if x_hat.shape != expected_shape:
            raise RuntimeError(
                "Decoder output sai shape: "
                f"got {tuple(x_hat.shape)}, expected {expected_shape}"
            )

        return x_hat

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_dropout_p: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_hat:
            (batch, n_sensors, n_buckets)

        z:
            (batch, z_dim)
        """

        z = self.encode(x)
        x_hat = self.decode(
            z=z,
            w=w,
            w_dropout_p=w_dropout_p,
        )

        return x_hat, z
