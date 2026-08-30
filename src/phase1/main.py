from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .core.data import SENSORS, load_cmapss_files, normalize_engines
from .core.metrics import fit_threshold, find_degradation_point, smooth_signal, z_drift_per_engine
from .core.train import extract_latents_for_engine, run_training
from .core.windowing import EngineData


DATA_PATHS = [
    Path("data/train_FD002.txt"),
    Path("data/train_FD003.txt"),
    Path("data/train_FD004.txt"),
]

CONFIG = {
    "window_len": 20,
    "train_stride": 1,
    "bin_stride": 10,
    "z_dim": 4,
    "n_epochs": 50,
    "lr": 1e-3,
    "engines_per_batch": 8,
    "val_fraction": 0.2,
    "patience": 5,
    "seed": 0,
}


def split_train_val(
    engines: list[EngineData], val_fraction: float, seed: int,
) -> tuple[list[EngineData], list[EngineData]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(engines))
    n_val = max(1, int(len(engines) * val_fraction))
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    return [engines[index] for index in train_indices], [engines[index] for index in val_indices]


def main(smoke: bool = False) -> None:
    config = CONFIG.copy()
    all_engines = load_cmapss_files(DATA_PATHS)
    train_engines, val_engines = split_train_val(
        all_engines, config["val_fraction"], config["seed"],
    )

    if smoke:
        train_engines, val_engines = train_engines[:4], val_engines[:2]
        config.update(n_epochs=1, engines_per_batch=2, patience=1)

    train_engines, val_engines = normalize_engines(train_engines, val_engines)
    print(
        f"{len(train_engines)} engine train, {len(val_engines)} engine validation, "
        f"{len(SENSORS)} sensors",
    )

    model = run_training(
        engines=train_engines,
        val_engines=val_engines,
        n_sensors=len(SENSORS),
        w_dim=train_engines[0].W.shape[1],
        window_len=config["window_len"],
        stride=config["train_stride"],
        z_dim=config["z_dim"],
        n_epochs=config["n_epochs"],
        lr=config["lr"],
        engines_per_batch=config["engines_per_batch"],
        seed=config["seed"],
        patience=config["patience"],
    )

    output = Path("smart_ae_smoke.pt" if smoke else "smart_ae.pt")
    torch.save(model.state_dict(), output)
    print(f"Saved model: {output}")

    example_engine = val_engines[0]
    latents = extract_latents_for_engine(
        model, example_engine,
        window_len=config["window_len"],
        bin_stride=config["bin_stride"],
    )
    drift = smooth_signal(z_drift_per_engine(latents.X))
    threshold = fit_threshold(drift[:5])
    degradation_index = find_degradation_point(drift, threshold)
    degradation_cycle = None if degradation_index is None else int(latents.cycle[degradation_index])
    print(f"Engine {example_engine.unit_number}: degradation cycle={degradation_cycle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Train 1 epoch on 4 engines")
    main(parser.parse_args().smoke)
