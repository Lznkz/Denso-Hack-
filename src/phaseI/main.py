from pathlib import Path

import torch

from src.phaseI.core.preprocess import (
    PreprocessConfig,
    preprocess_many,
)
from src.phaseI.core.windowing import (
    fit_normalization,
    load_engines,
)
from src.phaseI.core.train import (
    Z_DIM,
    run_training,
)


# ============================================================
# DEVICE
# ============================================================

def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------
    # Raw Paderborn root. Moi bearing duoc ky vong o:
    #   data/Bxx/vibrationData/*.mat
    #   data/Bxx/Bxx_operatingConditions.csv
    "raw_data_dir": "data",

    # Output root cua preprocess SPEV moi:
    #   processed_dual/Bxx/X_hi_spev.npy
    #   processed_dual/Bxx/W.npy
    #   processed_dual/Bxx/measurement_id.npy
    "preprocessed_dir": "processed_dual",

    # Main tu dong preprocess nhung bearing chua co du output HI.
    # Neu output da day du thi bearing do duoc bo qua.
    "auto_preprocess": False,
    "preprocess_overwrite": False,
    "refit_normalization": True,

    # De test nhanh co the gioi han measurement cho tung bearing, vi du:
    # {"B01": 20}. De {} de dung tat ca raw measurement dang co.
    "preprocess_measurement_max": {},

    "preprocess_config": {
        "target_fs": 64_000.0,
        "n_order_bins": 2049,
        "max_order": 600.0,
        "hi_n_buckets": 32,
    },

    # Scaler duoc fit LAI chi tu train bearings moi lan train.
    "normalization_path": "normalization_spev_except_17.npz",

    "train_engines": [
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
    ],

    # De trong neu chua chot validation split.
    # Neu them bearing vao day, chung chi duoc transform bang scaler
    # da fit tren train_engines; tuyet doi khong tham gia fit scaler.
    "val_engines": ["B17"],

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    # Z co dinh = 1 trong train.py/model pipeline.
    # Khong expose z_dim nhu hyperparameter o main.
    "hidden_channels": (
        32,
        16,
        8,
    ),

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    "batch_size": 128,
    "n_epochs": 20,
    "lr": 1e-3,
    "seed": 0,
    "device": _default_device(),

    # L = MAE + lambda_ng * L_NG
    "lambda_ng": 1,

    "grad_clip": 5.0,
    "patience": 5,
    "min_delta": 1e-4,

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    "checkpoint_path": "thien_gay_cut.pt",

    # True -> 1 epoch de kiem tra pipeline.
    "smoke_test": False,
}


# ============================================================
# HELPERS
# ============================================================

def _validate_split(
    train_engines: list[str],
    val_engines: list[str],
) -> None:
    if not train_engines:
        raise ValueError("train_engines khong duoc rong.")

    if len(set(train_engines)) != len(train_engines):
        raise ValueError("train_engines co bearing bi lap.")

    if len(set(val_engines)) != len(val_engines):
        raise ValueError("val_engines co bearing bi lap.")

    overlap = sorted(
        set(train_engines)
        & set(val_engines)
    )

    if overlap:
        raise ValueError(
            "Train/validation bearing bi overlap: "
            + ", ".join(overlap)
        )


def _check_loaded_contract(
    engines,
    label: str,
    n_sensors: int,
    n_buckets: int,
    w_dim: int,
) -> None:
    for engine in engines:
        if engine.X.shape[1:] != (
            n_sensors,
            n_buckets,
        ):
            raise ValueError(
                f"{label} {engine.unit_number}: "
                f"X shape={engine.X.shape[1:]}, "
                f"expected={(n_sensors, n_buckets)}"
            )

        if engine.W.shape[1] != w_dim:
            raise ValueError(
                f"{label} {engine.unit_number}: "
                f"W dim={engine.W.shape[1]}, "
                f"expected={w_dim}"
            )


_REQUIRED_HI_OUTPUTS = (
    "X_hi_spev.npy",
    "W.npy",
    "measurement_id.npy",
)


def _bearing_is_preprocessed(
    bearing_id: str,
    output_root: str | Path,
) -> bool:
    bearing_dir = Path(output_root) / bearing_id
    return all(
        (bearing_dir / name).is_file()
        for name in _REQUIRED_HI_OUTPUTS
    )


def _ensure_preprocessed(
    bearing_ids: list[str],
    cfg: dict,
) -> None:
    if not bearing_ids:
        return

    overwrite = bool(cfg["preprocess_overwrite"])

    missing = [
        bearing_id
        for bearing_id in bearing_ids
        if not _bearing_is_preprocessed(
            bearing_id=bearing_id,
            output_root=cfg["preprocessed_dir"],
        )
    ]

    # overwrite=True means explicitly rebuild every requested bearing.
    # This is useful after a smoke/partial preprocessing run.
    targets = list(bearing_ids) if overwrite else missing

    if not targets:
        print("Preprocess: all requested bearings already exist; skip.")
        return

    if not cfg["auto_preprocess"]:
        raise FileNotFoundError(
            "Missing preprocessed bearings: "
            + ", ".join(missing)
            + ". Set auto_preprocess=True or run preprocess.py first."
        )

    # HI preprocessing intentionally does NOT depend on localization
    # window_len / expected_windows_per_measurement. Those remain
    # internal defaults of preprocess.py for branch='localize' only.
    prep_args = cfg["preprocess_config"]
    prep_cfg = PreprocessConfig(
        target_fs=float(prep_args["target_fs"]),
        n_order_bins=int(prep_args["n_order_bins"]),
        max_order=float(prep_args["max_order"]),
        hi_n_buckets=int(prep_args["hi_n_buckets"]),
    )

    print("\nPreprocessing bearings from RAW data...")
    print("Raw root:", cfg["raw_data_dir"])
    print("Output root:", cfg["preprocessed_dir"])
    print("Bearings:", targets)
    if overwrite:
        print("Overwrite: True (rebuild requested bearings)")

    preprocess_many(
        bearings=targets,
        base_dir=cfg["raw_data_dir"],
        output_root=cfg["preprocessed_dir"],
        cfg=prep_cfg,
        branch="hi",
        measurement_max_by_bearing=dict(
            cfg["preprocess_measurement_max"]
        ),
        overwrite=overwrite,
    )

    # Fail early if preprocessing did not create the downstream contract.
    still_missing = [
        bearing_id
        for bearing_id in bearing_ids
        if not _bearing_is_preprocessed(
            bearing_id=bearing_id,
            output_root=cfg["preprocessed_dir"],
        )
    ]
    if still_missing:
        raise RuntimeError(
            "Preprocessing finished but required HI outputs are still missing for: "
            + ", ".join(still_missing)
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cfg = CONFIG

    train_ids = list(cfg["train_engines"])
    val_ids = list(cfg["val_engines"])

    _validate_split(
        train_engines=train_ids,
        val_engines=val_ids,
    )

    print("=" * 70)
    print("SMART AE - PHASE I / SPEV")
    print("=" * 70)
    print("Device:", cfg["device"])
    print("Z dim:", Z_DIM)

    # ========================================================
    # 0. RAW -> SPEV PREPROCESS (ONLY IF OUTPUT IS MISSING)
    # ========================================================

    requested_ids = list(dict.fromkeys(train_ids + val_ids))
    _ensure_preprocessed(
        bearing_ids=requested_ids,
        cfg=cfg,
    )

    # ========================================================
    # 1. FIT NORMALIZATION ON TRAIN BEARINGS ONLY
    # ========================================================

    print("\nFitting normalization on TRAIN bearings only...")

    norm = fit_normalization(
        train_engine_ids=train_ids,
        data_dir=cfg["preprocessed_dir"],
        output_path=cfg["normalization_path"],
    )

    # ========================================================
    # 2. LOAD TRAIN / VALIDATION WITH FROZEN SCALER
    # ========================================================

    print("\nLoading training bearings...")

    train_engines = load_engines(
        engine_ids=train_ids,
        data_dir=cfg["preprocessed_dir"],
        normalization_path=cfg["normalization_path"],
    )

    if val_ids:
        print("\nLoading validation bearings...")

        val_engines = load_engines(
            engine_ids=val_ids,
            data_dir=cfg["preprocessed_dir"],
            normalization_path=cfg["normalization_path"],
        )
    else:
        val_engines = None

    # ========================================================
    # 3. INPUT CONTRACT
    # ========================================================

    first = train_engines[0]

    n_sensors = int(first.X.shape[1])
    n_buckets = int(first.X.shape[2])
    w_dim = int(first.W.shape[1])

    _check_loaded_contract(
        engines=train_engines,
        label="train",
        n_sensors=n_sensors,
        n_buckets=n_buckets,
        w_dim=w_dim,
    )

    if val_engines:
        _check_loaded_contract(
            engines=val_engines,
            label="val",
            n_sensors=n_sensors,
            n_buckets=n_buckets,
            w_dim=w_dim,
        )

    print("\nInput contract:")
    print(
        "  X_tilde = "
        f"(N_measurements, {n_sensors}, {n_buckets})"
    )
    print(
        "  W_tilde = "
        f"(N_measurements, {w_dim})"
    )
    print("  Z = (N_measurements, 1)")
    print(
        "  X normalization: log1p -> StandardScaler(train only)"
    )
    print(
        "  W normalization: StandardScaler(train only)"
    )

    # ========================================================
    # 4. SMOKE / FULL TRAINING
    # ========================================================

    if cfg["smoke_test"]:
        n_epochs = 1
        print("\n*** SMOKE TEST: 1 epoch ***")
    else:
        n_epochs = int(cfg["n_epochs"])

    model = run_training(
        engines=train_engines,
        val_engines=val_engines,
        batch_size=cfg["batch_size"],
        hidden_channels=cfg["hidden_channels"],
        n_epochs=n_epochs,
        lr=cfg["lr"],
        seed=cfg["seed"],
        device=cfg["device"],
        lambda_ng=cfg["lambda_ng"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        grad_clip=cfg["grad_clip"],
    )

    if model.z_dim != Z_DIM:
        raise RuntimeError(
            f"Expected model.z_dim={Z_DIM}, got {model.z_dim}."
        )

    # ========================================================
    # 5. CHECKPOINT
    # ========================================================

    checkpoint = {
        "version": "smart_ae_spev_ng_v1",

        "model_state_dict": model.state_dict(),

        "model_config": {
            "n_sensors": n_sensors,
            "n_buckets": n_buckets,
            "w_dim": w_dim,
            "z_dim": Z_DIM,
            "hidden_channels": tuple(
                cfg["hidden_channels"]
            ),
            "kernel_size": int(model.kernel_size),
        },

        "training_config": {
            "train_engines": train_ids,
            "val_engines": val_ids,
            "batch_size": int(cfg["batch_size"]),
            "n_epochs": n_epochs,
            "lr": float(cfg["lr"]),
            "lambda_ng": float(cfg["lambda_ng"]),
            "seed": int(cfg["seed"]),
            "grad_clip": cfg["grad_clip"],
        },

        # Luu kem train-fitted scaler de inference co the dung dung
        # transform cua training, khong can fit lai tren test bearing.
        "normalization": {
            "x_mean": norm["x_mean"],
            "x_std": norm["x_std"],
            "w_mean": norm["w_mean"],
            "w_std": norm["w_std"],
            "x_transform": "log1p_then_standard_scaler",
            "w_transform": "standard_scaler",
            "fit_scope": "train_bearings_only",
        },

        "data_layout": {
            "layout": "channel_first",
            "sample_level": "measurement",
            "n_sensors": n_sensors,
            "n_buckets": n_buckets,
            "w_dim": w_dim,
            "x_shape": [
                n_sensors,
                n_buckets,
            ],
            "representation": "order_spev",
        },

        "latent_semantics": {
            "z_dim": Z_DIM,
            "temporal_constraint": "negative_gradient",
            "direction": "non_increasing_with_measurement_id",
            "plateau_penalty": False,
        },
    }

    path = Path(cfg["checkpoint_path"])
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if cfg["smoke_test"]:
        path = path.with_name(
            path.stem
            + "_smoke"
            + path.suffix
        )

    torch.save(
        checkpoint,
        path,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("Saved:", path)
    print(
        "X_tilde: "
        f"(batch, {n_sensors}, {n_buckets})"
    )
    print(
        "W_tilde: "
        f"(batch, {w_dim})"
    )
    print(
        "Z: "
        f"(batch, {Z_DIM})"
    )


if __name__ == "__main__":
    main()
