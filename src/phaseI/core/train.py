"""
train.py

Training loop cho Phase I / HI branch sau SPEV preprocessing moi.

Data contract
-------------
Moi row cua EngineData la dung MOT measurement:

    X:     (N_measurements, n_channels, n_spev_buckets)
    W:     (N_measurements, w_dim)
    cycle: (N_measurements,)   # ten legacy, thuc chat measurement_id

Normalization da duoc windowing.py xu ly:

    X_raw -> log1p -> StandardScaler(train bearings)
    W_raw -> StandardScaler(train bearings)

Model contract
--------------
    X_tilde -> Encoder -> Z
    (Z, W_tilde) -> Decoder -> X_hat_tilde

Phase I co dinh:
    z_dim = 1

Loss contract
-------------
    L = MAE(X_hat_tilde, X_tilde) + lambda_ng * L_NG

Theo quy uoc Bajarunas dang dung trong loss.py:
    Z khong duoc tang theo thoi gian (dZ/dt <= 0).

Temporal loss chi noi cac measurement co measurement_id lien tiep.
Khong con healthy baseline, latent-distance loss, trend, variance,
covariance hay smoothness trong training loop.
"""

import numpy as np
import torch

from src.phaseI.core.windowing import EngineData, WindowBatch
from src.phaseI.core.model import SmartConv1DAutoencoder
from src.phaseI.core.loss import smart_ae_loss


Z_DIM = 1


# ============================================================
# VALIDATION
# ============================================================

def _validate_engine_structure(
    engine: EngineData,
) -> None:
    """Validate contract moi ma khong materialize toan bo X."""

    if engine.X.ndim != 3:
        raise ValueError(
            f"{engine.unit_number}: X phai co shape "
            "(N_measurements, channels, SPEV_buckets), "
            f"got {engine.X.shape}"
        )

    n = len(engine.X)

    if n == 0:
        raise ValueError(
            f"{engine.unit_number}: khong co measurement."
        )

    W = np.asarray(engine.W)
    measurement_ids = np.asarray(engine.cycle)

    if W.ndim != 2:
        raise ValueError(
            f"{engine.unit_number}: W phai co shape "
            f"(N_measurements, w_dim), got {W.shape}"
        )

    if measurement_ids.ndim != 1:
        raise ValueError(
            f"{engine.unit_number}: measurement_id phai 1D, "
            f"got {measurement_ids.shape}"
        )

    if not (
        len(W) == n
        and len(measurement_ids) == n
    ):
        raise ValueError(
            f"{engine.unit_number}: X/W/measurement_id khac so dong: "
            f"X={n}, W={len(W)}, measurement_id={len(measurement_ids)}"
        )

    if not np.isfinite(W).all():
        raise ValueError(
            f"{engine.unit_number}: W chua NaN/Inf"
        )

    # Moi measurement chi co mot row. Cho phep gap, nhung khong cho
    # duplicate hay dao thu tu thoi gian.
    if n > 1 and not np.all(
        measurement_ids[1:] > measurement_ids[:-1]
    ):
        raise ValueError(
            f"{engine.unit_number}: measurement_id phai tang nghiem ngat."
        )

    # Sanity-check mot sample normalized X, tranh materialize ca bearing.
    x_sample = np.asarray(
        engine.X[0],
        dtype=np.float32,
    )

    expected_shape = (
        int(engine.X.shape[1]),
        int(engine.X.shape[2]),
    )

    if x_sample.shape != expected_shape:
        raise ValueError(
            f"{engine.unit_number}: X sample shape={x_sample.shape}, "
            f"expected={expected_shape}"
        )

    if not np.isfinite(x_sample).all():
        raise ValueError(
            f"{engine.unit_number}: X sample dau tien chua NaN/Inf"
        )


# ============================================================
# TEMPORAL ADJACENCY
# ============================================================

def _build_measurement_mask(
    measurement_ids: np.ndarray,
    device: str,
) -> torch.Tensor:
    """
    Chi cho L_NG noi hai measurement thuc su lien tiep.

    Vi du:
        measurement_id = [51, 52, 53, 55]
        mask           = [ T,  T,  F]
    """

    measurement_ids = np.asarray(measurement_ids)

    if measurement_ids.ndim != 1:
        raise ValueError(
            "measurement_ids phai la mang 1D."
        )

    if len(measurement_ids) <= 1:
        return torch.zeros(
            0,
            dtype=torch.bool,
            device=device,
        )

    mask = (
        measurement_ids[1:]
        == measurement_ids[:-1] + 1
    )

    return torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=device,
    )


# ============================================================
# CONTIGUOUS MEASUREMENT BATCHES
# ============================================================

def _build_batch_specs(
    engines: list[EngineData],
    batch_size: int,
) -> list[tuple[int, int, int]]:
    """
    Tao contiguous measurement blocks cua tung bearing.

    Chi shuffle thu tu cac block; thu tu measurement ben trong block
    luon duoc giu de L_NG co nghia theo thoi gian.
    """

    if batch_size <= 0:
        raise ValueError(
            f"batch_size phai > 0, got {batch_size}"
        )

    specs: list[tuple[int, int, int]] = []

    for engine_idx, engine in enumerate(engines):
        n = len(engine.X)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            specs.append(
                (
                    engine_idx,
                    int(start),
                    int(end),
                )
            )

    return specs


def _iter_epoch_batches(
    engines: list[EngineData],
    batch_size: int,
    rng: np.random.Generator,
):
    specs = _build_batch_specs(
        engines=engines,
        batch_size=batch_size,
    )

    order = rng.permutation(len(specs))

    for idx in order:
        engine_idx, start, end = specs[int(idx)]
        engine = engines[engine_idx]

        yield WindowBatch(
            X=np.asarray(
                engine.X[start:end],
                dtype=np.float32,
            ),
            W=np.asarray(
                engine.W[start:end],
                dtype=np.float32,
            ),
            unit_number=np.full(
                end - start,
                engine.unit_number,
                dtype=object,
            ),
            cycle=np.asarray(
                engine.cycle[start:end]
            ),
        )


# ============================================================
# BATCH LOSS
# ============================================================

def _compute_batch_losses(
    model: SmartConv1DAutoencoder,
    batch: WindowBatch,
    device: str,
    lambda_ng: float,
) -> dict[str, torch.Tensor]:
    """Forward mot contiguous measurement block va tinh MAE + NG."""

    x = torch.from_numpy(
        np.asarray(
            batch.X,
            dtype=np.float32,
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )

    w = torch.from_numpy(
        np.asarray(
            batch.W,
            dtype=np.float32,
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )

    if x.ndim != 3:
        raise ValueError(
            f"batch.X phai 3D, got {tuple(x.shape)}"
        )

    if w.ndim != 2 or w.shape[0] != x.shape[0]:
        raise ValueError(
            "batch.W phai co shape (batch, w_dim) va cung batch size voi X."
        )

    # Moi batch duoc tao tu dung mot bearing.
    unique_units = np.unique(batch.unit_number)
    if len(unique_units) != 1:
        raise ValueError(
            "Mot batch phai thuoc dung mot bearing."
        )

    x_hat, z = model(x, w)

    if z.ndim != 2 or z.shape[1] != Z_DIM:
        raise RuntimeError(
            f"Phase I yeu cau Z shape (N,{Z_DIM}), got {tuple(z.shape)}"
        )

    same_engine_mask = _build_measurement_mask(
        measurement_ids=batch.cycle,
        device=device,
    )

    return smart_ae_loss(
        x_hat=x_hat,
        x=x,
        z_seq=z,
        same_engine_mask=same_engine_mask,
        lambda_ng=lambda_ng,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def _evaluate(
    model: SmartConv1DAutoencoder,
    engines: list[EngineData],
    batch_size: int,
    device: str,
    lambda_ng: float,
) -> dict[str, float]:
    model.eval()

    totals = {
        "total": 0.0,
        "recon": 0.0,
        "ng": 0.0,
    }
    n_batches = 0

    # Validation khong can random order. De deterministic va de doc log.
    rng = np.random.default_rng(0)

    for batch in _iter_epoch_batches(
        engines=engines,
        batch_size=batch_size,
        rng=rng,
    ):
        losses = _compute_batch_losses(
            model=model,
            batch=batch,
            device=device,
            lambda_ng=lambda_ng,
        )

        for key in totals:
            totals[key] += float(
                losses[key].item()
            )

        n_batches += 1

    if n_batches == 0:
        raise RuntimeError(
            "Khong tao duoc validation batch."
        )

    return {
        key: value / n_batches
        for key, value in totals.items()
    }


# ============================================================
# TRAIN
# ============================================================

def run_training(
    engines: list[EngineData],
    val_engines: list[EngineData] | None = None,
    batch_size: int = 128,
    hidden_channels: tuple[int, ...] = (
        32,
        16,
        8,
    ),
    n_epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    lambda_ng: float = 1.0,
    patience: int = 5,
    min_delta: float = 1e-5,
    grad_clip: float | None = 5.0,
) -> SmartConv1DAutoencoder:
    """
    Train conditional AE voi scalar latent Z.

    ``z_dim`` khong con la hyperparameter: Phase I co dinh Z_DIM = 1.
    W luon duoc dua vao decoder.
    """

    if not engines:
        raise ValueError(
            "Khong co training engine."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size phai > 0."
        )

    if n_epochs <= 0:
        raise ValueError(
            "n_epochs phai > 0."
        )

    if lr <= 0:
        raise ValueError(
            "lr phai > 0."
        )

    if lambda_ng < 0:
        raise ValueError(
            "lambda_ng phai >= 0."
        )

    for engine in engines:
        _validate_engine_structure(engine)

    if val_engines:
        for engine in val_engines:
            _validate_engine_structure(engine)

    first = engines[0]

    n_sensors = int(first.X.shape[1])
    n_buckets = int(first.X.shape[2])
    w_dim = int(first.W.shape[1])

    def _check_contract(
        engine: EngineData,
    ) -> None:
        if engine.X.shape[1] != n_sensors:
            raise ValueError(
                f"{engine.unit_number}: n_sensors={engine.X.shape[1]}, "
                f"expected={n_sensors}"
            )

        if engine.X.shape[2] != n_buckets:
            raise ValueError(
                f"{engine.unit_number}: n_buckets={engine.X.shape[2]}, "
                f"expected={n_buckets}"
            )

        if engine.W.shape[1] != w_dim:
            raise ValueError(
                f"{engine.unit_number}: w_dim={engine.W.shape[1]}, "
                f"expected={w_dim}"
            )

    for engine in engines:
        _check_contract(engine)

    if val_engines:
        for engine in val_engines:
            _check_contract(engine)

    print(
        "Model input: "
        f"X=(batch,{n_sensors},{n_buckets}), "
        f"W=(batch,{w_dim}), "
        f"Z=(batch,{Z_DIM})"
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)

    model = SmartConv1DAutoencoder(
        n_sensors=n_sensors,
        window_len=n_buckets,  # model.py giu ten legacy window_len
        w_dim=w_dim,
        z_dim=Z_DIM,
        hidden_channels=hidden_channels,
    ).to(device)

    if model.z_dim != Z_DIM:
        raise RuntimeError(
            f"Model phai co z_dim={Z_DIM}."
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()

        epoch_losses = {
            "total": 0.0,
            "recon": 0.0,
            "ng": 0.0,
        }
        n_batches = 0

        for batch in _iter_epoch_batches(
            engines=engines,
            batch_size=batch_size,
            rng=rng,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            losses = _compute_batch_losses(
                model=model,
                batch=batch,
                device=device,
                lambda_ng=lambda_ng,
            )

            losses["total"].backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += float(
                    losses[key].item()
                )

            n_batches += 1

        if n_batches == 0:
            raise RuntimeError(
                "Khong tao duoc training batch."
            )

        train_avg = {
            key: value / n_batches
            for key, value in epoch_losses.items()
        }

        train_log = " | ".join(
            f"{key}={value:.6f}"
            for key, value in train_avg.items()
        )

        if val_engines:
            val_avg = _evaluate(
                model=model,
                engines=val_engines,
                batch_size=batch_size,
                device=device,
                lambda_ng=lambda_ng,
            )

            val_log = " | ".join(
                f"val_{key}={value:.6f}"
                for key, value in val_avg.items()
            )

            print(
                f"epoch {epoch + 1:03d} | "
                f"{train_log} | {val_log}"
            )

            val_loss = val_avg["total"]

            if (
                best_val_loss - val_loss
                > min_delta
            ):
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= patience
            ):
                print(
                    "Early stopping tai "
                    f"epoch {epoch + 1}"
                )
                break

        else:
            print(
                f"epoch {epoch + 1:03d} | {train_log}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ============================================================
# MEASUREMENT-LEVEL LATENT EXTRACTION
# ============================================================

@torch.no_grad()
def extract_measurement_latents_for_engine(
    model: SmartConv1DAutoencoder,
    engine: EngineData,
    batch_size: int = 256,
    device: str = "cpu",
) -> WindowBatch:
    """
    Encode tung measurement SPEV truc tiep thanh scalar Z.

    Output X cua WindowBatch:
        (N_measurements, 1)
    """

    _validate_engine_structure(engine)

    if model.z_dim != Z_DIM:
        raise ValueError(
            f"Phase I yeu cau z_dim={Z_DIM}, model co {model.z_dim}."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size phai > 0."
        )

    model.eval()
    z_parts: list[np.ndarray] = []

    for start in range(
        0,
        len(engine.X),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(engine.X),
        )

        x = torch.from_numpy(
            np.asarray(
                engine.X[start:end],
                dtype=np.float32,
            )
        ).to(
            device=device,
            dtype=torch.float32,
        )

        z = model.encode(x)
        z_parts.append(
            z.cpu().numpy()
        )

    Z = np.concatenate(
        z_parts,
        axis=0,
    )

    expected_shape = (
        len(engine.X),
        Z_DIM,
    )

    if Z.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected latent shape {Z.shape}, expected {expected_shape}."
        )

    return WindowBatch(
        X=Z,
        W=np.asarray(
            engine.W,
            dtype=np.float32,
        ),
        unit_number=np.full(
            len(engine.X),
            engine.unit_number,
            dtype=object,
        ),
        cycle=np.asarray(
            engine.cycle
        ),
    )


@torch.no_grad()
def extract_latents_for_engine(
    model: SmartConv1DAutoencoder,
    engine: EngineData,
    batch_size: int = 256,
    device: str = "cpu",
) -> WindowBatch:
    """Backward-compatible alias; latent hien tai da la measurement-level."""

    return extract_measurement_latents_for_engine(
        model=model,
        engine=engine,
        batch_size=batch_size,
        device=device,
    )
