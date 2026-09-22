from pathlib import Path

import numpy as np
import torch

from src.phaseI.core.windowing import EngineData
from src.phaseI.core.train import (
    SUBWINDOWS_PER_MEASUREMENT,
    run_training,
)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CONFIG = {
    "prepared_root": "processed",
    "bearings": ["B01", "B05", "B12", "B17"],
    "val_bearing": "B17",

    # FFT từ sub-window raw dài 4096 -> rfft có 2049 bins.
    "window_len": 2049,

    # Tính theo SUB-WINDOW.
    # 200 = 8 measurement x 25 sub-window.
    "batch_size": 200,

    "z_dim": 16,
    "hidden_channels": (16, 8),

    "n_epochs": 100,
    "lr": 1e-4,
    "patience": 5,
    "min_delta": 1e-5,
    "seed": 0,
    "device": _default_device(),

    # Baseline = mean latent của N measurement đầu của từng bearing.
    "n_baseline_bins": 5,

    # Temporal losses chạy trên z_mean measurement-level.
    "lambda_mono": 0.1,
    "lambda_anticollapse": 0.0,
    "lambda_smooth": 0.02,
    "min_std": 0.1,

    # Dùng W thật đầy đủ.
    "w_dropout_start": 0.0,
    "w_dropout_end": 0.0,
    "w_dropout_epochs": 0,

    "checkpoint_path": "smart_ae_paderborn.pt",
}


def _validate_measurement_layout(
    code: str,
    X: np.ndarray,
    W: np.ndarray,
    measurement_id: np.ndarray,
) -> None:
    if len(X) != len(W) or len(X) != len(measurement_id):
        raise ValueError(
            f"{code}: số dòng không khớp: "
            f"X={len(X)}, W={len(W)}, measurement_id={len(measurement_id)}"
        )

    if X.ndim != 3:
        raise ValueError(
            f"{code}: X phải có shape (N, channels, n_freqs), got {X.shape}"
        )

    if W.ndim != 2:
        raise ValueError(
            f"{code}: W phải có shape (N, w_dim), got {W.shape}"
        )

    if X.shape[2] != CONFIG["window_len"]:
        raise ValueError(
            f"{code}: expected n_freqs={CONFIG['window_len']}, "
            f"got {X.shape[2]}"
        )

    if len(X) % SUBWINDOWS_PER_MEASUREMENT != 0:
        raise ValueError(
            f"{code}: N={len(X)} không chia hết cho "
            f"{SUBWINDOWS_PER_MEASUREMENT} sub-window/measurement."
        )

    cycle_grouped = np.asarray(measurement_id).reshape(
        -1,
        SUBWINDOWS_PER_MEASUREMENT,
    )

    if not np.all(cycle_grouped == cycle_grouped[:, :1]):
        raise ValueError(
            f"{code}: measurement_id không được nhóm thành block "
            f"{SUBWINDOWS_PER_MEASUREMENT} sub-window liên tiếp."
        )

    W_grouped = np.asarray(W).reshape(
        -1,
        SUBWINDOWS_PER_MEASUREMENT,
        W.shape[1],
    )

    if not np.allclose(
        W_grouped,
        W_grouped[:, :1, :],
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError(
            f"{code}: W không giống nhau giữa các sub-window "
            f"của cùng measurement."
        )


def load_engines(
    prepared_root: str,
    bearings: list[str],
) -> list[EngineData]:
    """
    Mỗi bearing cần có:
        X_fft.npy           -> (N_subwindows, 2, 2049)
        W.npy               -> (N_subwindows, w_dim)
        measurement_id.npy  -> (N_subwindows,)
    """
    root = Path(prepared_root)
    engines = []

    for code in bearings:
        bearing_dir = root / code

        x_path = bearing_dir / "X_fft.npy"
        w_path = bearing_dir / "W.npy"
        id_path = bearing_dir / "measurement_id.npy"

        for path in (x_path, w_path, id_path):
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy: {path}")

        X = np.load(x_path, mmap_mode="r")
        W = np.load(w_path, mmap_mode="r")
        measurement_id = np.load(id_path, mmap_mode="r")

        _validate_measurement_layout(
            code=code,
            X=X,
            W=W,
            measurement_id=measurement_id,
        )

        engine = EngineData(
            unit_number=code,
            X=X,
            W=W,
            cycle=measurement_id,
        )
        engines.append(engine)

        n_measurements = len(X) // SUBWINDOWS_PER_MEASUREMENT

        print(
            f"{code}: X={X.shape}, W={W.shape}, "
            f"measurements={n_measurements}, "
            f"subwindows/measurement={SUBWINDOWS_PER_MEASUREMENT}"
        )

    return engines


def split_train_val(
    engines: list[EngineData],
    val_bearing: str,
) -> tuple[list[EngineData], list[EngineData]]:
    train_engines = [
        engine for engine in engines
        if engine.unit_number != val_bearing
    ]

    val_engines = [
        engine for engine in engines
        if engine.unit_number == val_bearing
    ]

    if not train_engines:
        raise ValueError("Không có bearing train.")

    if not val_engines:
        raise ValueError(
            f"Không tìm thấy validation bearing: {val_bearing}"
        )

    return train_engines, val_engines


def normalize_engines(
    train_engines: list[EngineData],
    val_engines: list[EngineData],
) -> dict[str, np.ndarray]:
    """
    X:
        z-score theo từng channel, fit trên TRAIN.
        Statistics được tính qua toàn bộ sub-window và toàn bộ frequency bin.

    W:
        min-max theo TRAIN.

    Validation chỉ dùng statistics đã fit từ TRAIN.
    """
    channel_sum = None
    channel_sq_sum = None
    total_count = 0

    for engine in train_engines:
        X = np.asarray(engine.X)

        s = X.sum(axis=(0, 2), dtype=np.float64)
        ss = np.square(
            X,
            dtype=np.float64,
        ).sum(axis=(0, 2))
        count = X.shape[0] * X.shape[2]

        if channel_sum is None:
            channel_sum = s
            channel_sq_sum = ss
        else:
            channel_sum += s
            channel_sq_sum += ss

        total_count += count

    x_mean = channel_sum / total_count
    x_var = channel_sq_sum / total_count - x_mean ** 2
    x_std = np.sqrt(np.maximum(x_var, 1e-12))

    x_mean = x_mean[None, :, None]
    x_std = x_std[None, :, None]

    for engine in train_engines + val_engines:
        engine.X = (
            (np.asarray(engine.X) - x_mean) / x_std
        ).astype(np.float32)

    W_train = np.concatenate(
        [np.asarray(engine.W) for engine in train_engines],
        axis=0,
    )

    if not np.isfinite(W_train).all():
        raise ValueError(
            "W_train chứa NaN/Inf. Kiểm tra operatingConditions trước khi train."
        )

    w_min = W_train.min(axis=0)
    w_range = np.ptp(W_train, axis=0)
    w_range[w_range < 1e-8] = 1.0

    for engine in train_engines + val_engines:
        W = np.asarray(engine.W)

        if not np.isfinite(W).all():
            raise ValueError(
                f"{engine.unit_number}: W chứa NaN/Inf."
            )

        engine.W = np.clip(
            (W - w_min) / w_range,
            0.0,
            1.0,
        ).astype(np.float32)

    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "w_min": w_min.astype(np.float32),
        "w_range": w_range.astype(np.float32),
    }


def main():
    cfg = CONFIG

    if cfg["batch_size"] % SUBWINDOWS_PER_MEASUREMENT != 0:
        raise ValueError(
            f"batch_size={cfg['batch_size']} phải là bội số của "
            f"{SUBWINDOWS_PER_MEASUREMENT}."
        )

    print("Device:", cfg["device"])
    print(
        "Measurements/batch:",
        cfg["batch_size"] // SUBWINDOWS_PER_MEASUREMENT,
    )

    # 1. Load
    all_engines = load_engines(
        prepared_root=cfg["prepared_root"],
        bearings=cfg["bearings"],
    )

    # 2. Split theo bearing
    train_engines, val_engines = split_train_val(
        all_engines,
        val_bearing=cfg["val_bearing"],
    )

    print("Train:", [e.unit_number for e in train_engines])
    print("Validation:", [e.unit_number for e in val_engines])

    # 3. Normalize bằng TRAIN statistics
    norm_stats = normalize_engines(
        train_engines=train_engines,
        val_engines=val_engines,
    )

    # 4. Model dimensions
    n_sensors = train_engines[0].X.shape[1]
    window_len = train_engines[0].X.shape[2]
    w_dim = train_engines[0].W.shape[1]

    print(
        f"n_sensors={n_sensors}, "
        f"window_len={window_len}, "
        f"w_dim={w_dim}, "
        f"z_dim={cfg['z_dim']}"
    )

    # 5. Train
    model = run_training(
        engines=train_engines,
        val_engines=val_engines,
        n_sensors=n_sensors,
        w_dim=w_dim,
        window_len=window_len,
        batch_size=cfg["batch_size"],
        z_dim=cfg["z_dim"],
        hidden_channels=cfg["hidden_channels"],
        n_epochs=cfg["n_epochs"],
        lr=cfg["lr"],
        seed=cfg["seed"],
        device=cfg["device"],
        lambda_mono=cfg["lambda_mono"],
        lambda_anticollapse=cfg["lambda_anticollapse"],
        lambda_smooth=cfg["lambda_smooth"],
        n_baseline_bins=cfg["n_baseline_bins"],
        min_std=cfg["min_std"],
        w_dropout_start=cfg["w_dropout_start"],
        w_dropout_end=cfg["w_dropout_end"],
        w_dropout_epochs=cfg["w_dropout_epochs"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
    )

    # 6. Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "normalization": {
            "x_mean": norm_stats["x_mean"],
            "x_std": norm_stats["x_std"],
            "w_min": norm_stats["w_min"],
            "w_range": norm_stats["w_range"],
        },
        "data_layout": {
            "x_file": "X_fft.npy",
            "subwindows_per_measurement": SUBWINDOWS_PER_MEASUREMENT,
            "n_sensors": n_sensors,
            "window_len": window_len,
            "w_dim": w_dim,
        },
    }

    torch.save(
        checkpoint,
        cfg["checkpoint_path"],
    )

    print(f"Đã lưu: {cfg['checkpoint_path']}")
    print(
        "Temporal loss đã được tính trên z_mean measurement-level; "
        "không cần group/broadcast Z trong main.py."
    )


if __name__ == "__main__":
    main()
