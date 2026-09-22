import numpy as np
import torch

from src.phaseI.core.windowing import EngineData, WindowBatch
from src.phaseI.core.model import SmartConv1DAutoencoder
from src.phaseI.core.loss import smart_ae_loss


SUBWINDOWS_PER_MEASUREMENT = 25


def _to_model_layout(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2)


def _validate_engine_structure(engine: EngineData) -> None:
    n_x = len(engine.X)
    n_w = len(engine.W)
    n_cycle = len(engine.cycle)

    if n_x != n_w or n_x != n_cycle:
        raise ValueError(
            f"{engine.unit_number}: X/W/cycle không cùng số dòng: "
            f"X={n_x}, W={n_w}, cycle={n_cycle}"
        )

    if n_x % SUBWINDOWS_PER_MEASUREMENT != 0:
        raise ValueError(
            f"{engine.unit_number}: số sub-window={n_x} không chia hết cho "
            f"{SUBWINDOWS_PER_MEASUREMENT}."
        )

    cycle = np.asarray(engine.cycle)
    cycle_grouped = cycle.reshape(-1, SUBWINDOWS_PER_MEASUREMENT)

    if not np.all(cycle_grouped == cycle_grouped[:, :1]):
        raise ValueError(
            f"{engine.unit_number}: measurement_id không được nhóm liên tiếp "
            f"theo block {SUBWINDOWS_PER_MEASUREMENT} sub-window."
        )


def _mean_pool_z(
    z: torch.Tensor,
    cycle: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    n = z.shape[0]

    if n % SUBWINDOWS_PER_MEASUREMENT != 0:
        raise ValueError(
            f"Batch có {n} sub-window, không chia hết cho "
            f"{SUBWINDOWS_PER_MEASUREMENT}."
        )

    cycle = np.asarray(cycle)
    cycle_grouped = cycle.reshape(-1, SUBWINDOWS_PER_MEASUREMENT)

    if not np.all(cycle_grouped == cycle_grouped[:, :1]):
        raise ValueError(
            "Batch đang cắt ngang measurement hoặc measurement_id "
            "không được xếp liên tiếp."
        )

    n_measurements = cycle_grouped.shape[0]
    z_grouped = z.reshape(
        n_measurements,
        SUBWINDOWS_PER_MEASUREMENT,
        z.shape[-1],
    )
    z_mean = z_grouped.mean(dim=1)
    measurement_id = cycle_grouped[:, 0]

    return z_mean, measurement_id


def _build_measurement_mask(
    n_measurements: int,
    device: str,
) -> torch.Tensor:
    return torch.ones(
        max(n_measurements - 1, 0),
        dtype=torch.bool,
        device=device,
    )


def _prepare_epoch_batches(
    engines: list[EngineData],
    batch_size: int,
    rng: np.random.Generator,
) -> list[WindowBatch]:
    if batch_size % SUBWINDOWS_PER_MEASUREMENT != 0:
        raise ValueError(
            f"batch_size={batch_size} phải là bội số của "
            f"{SUBWINDOWS_PER_MEASUREMENT}."
        )

    batches = []

    for engine in engines:
        _validate_engine_structure(engine)
        n = len(engine.X)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            if (end - start) % SUBWINDOWS_PER_MEASUREMENT != 0:
                raise ValueError(
                    f"{engine.unit_number}: batch [{start}:{end}] "
                    f"cắt ngang measurement."
                )

            batches.append(
                WindowBatch(
                    X=engine.X[start:end],
                    W=engine.W[start:end],
                    unit_number=np.full(
                        end - start,
                        engine.unit_number,
                        dtype=object,
                    ),
                    cycle=engine.cycle[start:end],
                )
            )

    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


@torch.no_grad()
def _compute_engine_baseline(
    model: SmartConv1DAutoencoder,
    engine: EngineData,
    device: str,
    n_baseline_bins: int = 5,
) -> torch.Tensor:
    _validate_engine_structure(engine)

    n_total_measurements = len(engine.X) // SUBWINDOWS_PER_MEASUREMENT
    n_baseline = min(n_baseline_bins, n_total_measurements)

    if n_baseline <= 0:
        raise ValueError(f"{engine.unit_number}: không có measurement để tạo baseline.")

    n_subwindows = n_baseline * SUBWINDOWS_PER_MEASUREMENT

    x = torch.from_numpy(
        np.asarray(engine.X[:n_subwindows])
    ).float().to(device)

    x_model = _to_model_layout(x)
    z = model.encode(x_model)

    z_measurement = z.reshape(
        n_baseline,
        SUBWINDOWS_PER_MEASUREMENT,
        z.shape[-1],
    ).mean(dim=1)

    return z_measurement.mean(dim=0).detach()


@torch.no_grad()
def _compute_baselines(
    model: SmartConv1DAutoencoder,
    engines: list[EngineData],
    device: str,
    n_baseline_bins: int,
) -> dict[object, torch.Tensor]:
    baselines = {}

    for engine in engines:
        baselines[engine.unit_number] = _compute_engine_baseline(
            model=model,
            engine=engine,
            device=device,
            n_baseline_bins=n_baseline_bins,
        )

    return baselines


def _compute_batch_losses(
    model: SmartConv1DAutoencoder,
    batch: WindowBatch,
    baseline_by_engine: dict[object, torch.Tensor],
    device: str,
    lambda_mono: float,
    lambda_anticollapse: float,
    lambda_smooth: float,
    min_std: float,
    w_dropout_p: float,
) -> dict[str, torch.Tensor]:
    x = torch.from_numpy(np.asarray(batch.X)).float().to(device)
    w = torch.from_numpy(np.asarray(batch.W)).float().to(device)

    x_model = _to_model_layout(x)

    x_hat, z = model(
        x_model,
        w,
        w_dropout_p=w_dropout_p,
    )

    z_mean, _ = _mean_pool_z(z, batch.cycle)

    unique_units = np.unique(batch.unit_number)
    if len(unique_units) != 1:
        raise ValueError(
            "Một batch phải thuộc đúng 1 bearing để temporal loss có nghĩa."
        )

    unit = unique_units[0]
    baseline_value = baseline_by_engine[unit]
    baseline = baseline_value.unsqueeze(0).expand_as(z_mean)

    same_engine_mask = _build_measurement_mask(
        n_measurements=z_mean.shape[0],
        device=device,
    )

    return smart_ae_loss(
        x_hat=x_hat,
        x=x_model,
        z_seq=z_mean,
        baseline=baseline,
        same_engine_mask=same_engine_mask,
        lambda_mono=lambda_mono,
        lambda_anticollapse=lambda_anticollapse,
        lambda_smooth=lambda_smooth,
        min_std=min_std,
    )


@torch.no_grad()
def _evaluate(
    model: SmartConv1DAutoencoder,
    engines: list[EngineData],
    batch_size: int,
    device: str,
    lambda_mono: float,
    lambda_anticollapse: float,
    lambda_smooth: float,
    n_baseline_bins: int,
    min_std: float,
    rng: np.random.Generator,
) -> float:
    model.eval()

    baseline_by_engine = _compute_baselines(
        model=model,
        engines=engines,
        device=device,
        n_baseline_bins=n_baseline_bins,
    )

    batches = _prepare_epoch_batches(
        engines=engines,
        batch_size=batch_size,
        rng=rng,
    )

    total_loss = 0.0
    n_batches = 0

    for batch in batches:
        losses = _compute_batch_losses(
            model=model,
            batch=batch,
            baseline_by_engine=baseline_by_engine,
            device=device,
            lambda_mono=lambda_mono,
            lambda_anticollapse=lambda_anticollapse,
            lambda_smooth=lambda_smooth,
            min_std=min_std,
            w_dropout_p=0.0,
        )

        total_loss += losses["total"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_training(
    engines: list[EngineData],
    n_sensors: int,
    w_dim: int,
    window_len: int,
    val_engines: list[EngineData] | None = None,
    batch_size: int = 200,
    z_dim: int = 4,
    hidden_channels: tuple[int, int] = (16, 8),
    n_epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    lambda_mono: float = 1.0,
    lambda_anticollapse: float = 1.0,
    lambda_smooth: float = 0.0,
    n_baseline_bins: int = 5,
    min_std: float = 0.1,
    w_dropout_start: float = 0.0,
    w_dropout_end: float = 0.0,
    w_dropout_epochs: int = 0,
    patience: int = 5,
    min_delta: float = 1e-5,
) -> SmartConv1DAutoencoder:
    if batch_size % SUBWINDOWS_PER_MEASUREMENT != 0:
        raise ValueError(
            f"batch_size={batch_size} phải là bội số của "
            f"{SUBWINDOWS_PER_MEASUREMENT}."
        )

    for engine in engines:
        _validate_engine_structure(engine)

    if val_engines is not None:
        for engine in val_engines:
            _validate_engine_structure(engine)

    torch.manual_seed(seed)

    rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)

    model = SmartConv1DAutoencoder(
        n_sensors=n_sensors,
        window_len=window_len,
        w_dim=w_dim,
        z_dim=z_dim,
        hidden_channels=hidden_channels,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(n_epochs):
        model.eval()
        baseline_by_engine = _compute_baselines(
            model=model,
            engines=engines,
            device=device,
            n_baseline_bins=n_baseline_bins,
        )

        model.train()

        if w_dropout_epochs > 0:
            progress = min(epoch / w_dropout_epochs, 1.0)
        else:
            progress = 1.0

        w_dropout_p = (
            w_dropout_start
            + progress * (w_dropout_end - w_dropout_start)
        )

        batches = _prepare_epoch_batches(
            engines=engines,
            batch_size=batch_size,
            rng=rng,
        )

        epoch_losses = {
            "total": 0.0,
            "recon": 0.0,
            "mono": 0.0,
            "anticollapse": 0.0,
            "smooth": 0.0,
        }

        n_batches = 0

        for batch in batches:
            optimizer.zero_grad()

            losses = _compute_batch_losses(
                model=model,
                batch=batch,
                baseline_by_engine=baseline_by_engine,
                device=device,
                lambda_mono=lambda_mono,
                lambda_anticollapse=lambda_anticollapse,
                lambda_smooth=lambda_smooth,
                min_std=min_std,
                w_dropout_p=w_dropout_p,
            )

            losses["total"].backward()
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            n_batches += 1

        log = " | ".join(
            f"{key}={value / max(n_batches, 1):.6f}"
            for key, value in epoch_losses.items()
        )
        log += f" | w_dropout_p={w_dropout_p:.3f}"

        if val_engines is not None:
            val_loss = _evaluate(
                model=model,
                engines=val_engines,
                batch_size=batch_size,
                device=device,
                lambda_mono=lambda_mono,
                lambda_anticollapse=lambda_anticollapse,
                lambda_smooth=lambda_smooth,
                n_baseline_bins=n_baseline_bins,
                min_std=min_std,
                rng=val_rng,
            )

            print(
                f"epoch {epoch:3d} | "
                f"{log} | "
                f"val_total={val_loss:.6f}"
            )

            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping tại epoch {epoch} "
                    f"(val loss không cải thiện "
                    f"{patience} epoch liên tiếp)."
                )
                break
        else:
            print(f"epoch {epoch:3d} | {log}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def extract_latents_for_engine(
    model: SmartConv1DAutoencoder,
    engine: EngineData,
    batch_size: int = 200,
    device: str = "cpu",
) -> WindowBatch:
    model.eval()

    z_parts = []

    for start in range(0, len(engine.X), batch_size):
        end = min(start + batch_size, len(engine.X))

        x = torch.from_numpy(
            np.asarray(engine.X[start:end])
        ).float().to(device)

        x_model = _to_model_layout(x)
        z = model.encode(x_model)

        z_parts.append(z.cpu().numpy())

    Z = np.concatenate(z_parts, axis=0)

    return WindowBatch(
        X=Z,
        W=np.asarray(engine.W),
        unit_number=np.full(
            len(engine.X),
            engine.unit_number,
            dtype=object,
        ),
        cycle=np.asarray(engine.cycle),
    )


@torch.no_grad()
def extract_measurement_latents_for_engine(
    model: SmartConv1DAutoencoder,
    engine: EngineData,
    batch_size: int = 200,
    device: str = "cpu",
) -> WindowBatch:
    _validate_engine_structure(engine)

    window_latents = extract_latents_for_engine(
        model=model,
        engine=engine,
        batch_size=batch_size,
        device=device,
    )

    z = torch.from_numpy(np.asarray(window_latents.X)).float().to(device)
    z_mean, measurement_id = _mean_pool_z(
        z=z,
        cycle=window_latents.cycle,
    )

    W = np.asarray(engine.W).reshape(
        -1,
        SUBWINDOWS_PER_MEASUREMENT,
        engine.W.shape[-1],
    )[:, 0, :]

    return WindowBatch(
        X=z_mean.cpu().numpy(),
        W=W,
        unit_number=np.full(
            len(measurement_id),
            engine.unit_number,
            dtype=object,
        ),
        cycle=measurement_id,
    )
