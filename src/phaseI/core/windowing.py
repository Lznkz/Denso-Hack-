"""
windowing.py -- data structures dùng chung cho Phase I Paderborn.

Paderborn đã được window ở bước prepare_dataset:
    raw .mat
    -> resample 32 kHz
    -> cắt 4096 sample / stride 4096
    -> lưu X.npy, W.npy, measurement_id.npy

Vì vậy file này KHÔNG cắt window nữa.

Contract hiện tại:
    EngineData.X:
        (n_windows, n_channels, window_len)
        ví dụ B01: (4524, 2, 4096)

    EngineData.W:
        (n_windows, w_dim)
        ví dụ B01: (4524, 3)

    EngineData.cycle:
        (n_windows,)
        hiện dùng để giữ measurement_id:
        [1,1,...,1,2,2,...,2,...]

    unit_number:
        bearing id, ví dụ "B01"

WindowBatch chỉ là container cho mini-batch / latent extraction.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class EngineData:
    unit_number: int | str

    # Paderborn đã window sẵn:
    # (n_windows, n_channels, window_len)
    X: np.ndarray

    # Operating condition tương ứng từng window:
    # (n_windows, w_dim)
    W: np.ndarray

    # Measurement id tương ứng từng window.
    # Ví dụ 12 window đầu của M0001 đều có cycle = 1.
    cycle: np.ndarray


@dataclass
class WindowBatch:
    # Trong training:
    #   X = (batch, n_channels, window_len)
    #
    # Trong latent extraction:
    #   X có thể chứa Z = (batch, z_dim)
    X: np.ndarray

    # (batch, w_dim)
    W: np.ndarray

    # (batch,)
    unit_number: np.ndarray

    # (batch,)
    # hiện là measurement_id tương ứng từng sample/window
    cycle: np.ndarray
