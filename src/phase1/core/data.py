from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .windowing import EngineData


OP_SETTINGS = ["op1", "op2", "op3"]
ALL_SENSORS = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
    "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
    "Nfdmd", "PCNfRdmd", "W31", "W32",
]
SENSORS = ["T24", "T30", "T50", "P30", "Ps30", "phi", "BPR", "W31", "W32"]
COLUMNS = ["unit_number", "cycles", *OP_SETTINGS, *ALL_SENSORS]


def load_cmapss_files(paths: list[str | Path]) -> list[EngineData]:
    engines = []

    for value in paths:
        path = Path(value)
        match = re.search(r"FD(\d{3})", path.name)
        if not path.is_file() or match is None:
            raise ValueError(f"Invalid C-MAPSS path: {path}")

        frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
        if frame.empty or frame[COLUMNS].isna().any().any():
            raise ValueError(f"Invalid or incomplete C-MAPSS data: {path}")

        dataset_id = int(match.group(1))
        for unit_number, group in frame.groupby("unit_number", sort=True):
            group = group.sort_values("cycles")
            engines.append(EngineData(
                unit_number=dataset_id * 10_000 + int(unit_number),
                X=group[SENSORS].to_numpy(dtype=np.float32),
                W=group[OP_SETTINGS].to_numpy(dtype=np.float32),
                cycle=group["cycles"].to_numpy(dtype=np.int32),
            ))

    if not engines:
        raise ValueError("No engines loaded")
    return engines


def normalize_engines(
    train_engines: list[EngineData],
    val_engines: list[EngineData],
) -> tuple[list[EngineData], list[EngineData]]:
    if not train_engines:
        raise ValueError("Training split is empty")

    train_x = np.concatenate([engine.X for engine in train_engines])
    train_w = np.concatenate([engine.W for engine in train_engines])
    x_mean, x_std = train_x.mean(axis=0), train_x.std(axis=0)
    w_mean, w_std = train_w.mean(axis=0), train_w.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    w_std = np.where(w_std < 1e-8, 1.0, w_std)

    def normalize(items: list[EngineData]) -> list[EngineData]:
        return [replace(
            engine,
            X=((engine.X - x_mean) / x_std).astype(np.float32),
            W=((engine.W - w_mean) / w_std).astype(np.float32),
        ) for engine in items]

    return normalize(train_engines), normalize(val_engines)
