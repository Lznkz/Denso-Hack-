"""
Compact preprocessing for the Paderborn bearing project.

Branches
--------
HI branch
    raw vibration (full 1.6 s measurement)
    -> DC removal -> Hann -> one-sided FFT magnitude
    -> mean-RPM Hz-to-order remap
    -> SPEV-style max pooling over equal-width order buckets

Localization branch
    raw vibration -> TKEO -> fixed subwindows
    -> per-window DC removal -> Hann -> one-sided FFT magnitude
    -> mean-RPM Hz-to-order remap

Output policy
-------------
Only model-relevant arrays are saved.

For --branch hi:
    X_hi_spev.npy          shape (measurements, channels, hi_n_buckets)
    W.npy                  shape (measurements, 3)
    measurement_id.npy     shape (measurements,)

For --branch localize:
    X_localize_teager_order.npy
                           shape (measurements * windows_per_measurement,
                                  channels, order_bins)
    W.npy                  shape (measurements, 3)
    measurement_id.npy     shape (measurements,)

For --branch both, both X files are saved plus the shared W/measurement arrays.
For localization, rows are ordered in contiguous blocks of
``expected_windows_per_measurement`` windows per measurement, so a separate
window_id file is unnecessary.

Normalization is intentionally NOT performed here. Fit X/W transforms on
TRAIN bearings only, then freeze them for validation/test.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio
from numpy.lib.format import open_memmap


DEFAULT_CHANNELS = (
    "accHorizFrontal_C",
    "accHorizRear_A",
)

RPM_COLUMN = "meanAbs_speed / rpm"
W_COLUMNS = (
    "peak_dynLoad / N",
    "meanAbs_statLoad / N",
    RPM_COLUMN,
)

BRANCH_CHOICES = ("hi", "localize", "both")


@dataclass(frozen=True)
class PreprocessConfig:
    """Fixed physical preprocessing configuration for one experiment."""

    target_fs: float = 64_000.0

    # Localization only.
    window_len: int = 4096
    expected_windows_per_measurement: int = 25

    # Shared order-domain representation.
    n_order_bins: int = 2049
    max_order: float = 600.0

    # HI only.
    hi_n_buckets: int = 20

    channels: tuple[str, ...] = DEFAULT_CHANNELS

    # Sampling timestamps can contain small floating-point error.
    fs_rtol: float = 5e-3
    fs_atol: float = 5.0

    def validate(self) -> None:
        if self.target_fs <= 0:
            raise ValueError("target_fs must be > 0")
        if self.window_len < 4:
            raise ValueError("window_len must be >= 4")
        if self.expected_windows_per_measurement < 1:
            raise ValueError("expected_windows_per_measurement must be >= 1")
        if self.n_order_bins < 2:
            raise ValueError("n_order_bins must be >= 2")
        if self.max_order <= 0:
            raise ValueError("max_order must be > 0")
        if self.hi_n_buckets < 1:
            raise ValueError("hi_n_buckets must be >= 1")
        if self.hi_n_buckets > self.n_order_bins:
            raise ValueError("hi_n_buckets cannot exceed n_order_bins")
        if len(self.channels) < 1:
            raise ValueError("At least one vibration channel is required")


def _branch_enabled(branch: str, target: str) -> bool:
    return branch == target or branch == "both"


# =============================================================================
# 1. FILE DISCOVERY + OPERATING CONDITIONS
# =============================================================================


def list_measurement_files(
    bearing_id: str,
    base_dir: str | os.PathLike[str] = "data",
    measurement_max: int | None = None,
) -> list[tuple[int, Path]]:
    """Return sorted ``(measurement_id, .mat path)`` pairs for one bearing."""
    folder = Path(base_dir) / bearing_id / "vibrationData"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing vibration folder: {folder}")

    pattern = re.compile(rf"^data_{re.escape(bearing_id)}_M(\d+)\.mat$")
    files: list[tuple[int, Path]] = []

    for path in folder.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        measurement_id = int(match.group(1))
        if measurement_max is not None and measurement_id > measurement_max:
            continue
        files.append((measurement_id, path))

    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(
            f"No measurement .mat files found for {bearing_id} in {folder}"
        )
    return files


def load_operating_conditions(
    bearing_id: str,
    base_dir: str | os.PathLike[str] = "data",
) -> pd.DataFrame:
    """Load operating conditions and create a 1-based measurement index."""
    csv_path = Path(base_dir) / bearing_id / f"{bearing_id}_operatingConditions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = next(
        (candidate for candidate in ("Time", "dateTime") if candidate in df.columns),
        None,
    )
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    missing = [col for col in W_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{bearing_id}: missing operating-condition columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["idx"] = np.arange(1, len(df) + 1, dtype=np.int64)
    return df


def get_measurement_conditions(
    measurement_id: int,
    op_lookup: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Return raw W=[dynamic load, static load, RPM] and mean RPM."""
    if measurement_id not in op_lookup.index:
        raise KeyError(f"No operating-condition row for measurement M{measurement_id}")

    row = op_lookup.loc[measurement_id]
    w = np.asarray([row[col] for col in W_COLUMNS], dtype=np.float32)
    if not np.isfinite(w).all():
        raise ValueError(f"M{measurement_id}: W contains NaN/Inf: {w.tolist()}")

    rpm = float(row[RPM_COLUMN])
    if not np.isfinite(rpm) or rpm <= 0:
        raise ValueError(f"M{measurement_id}: invalid mean RPM={rpm}")

    return w, rpm


# =============================================================================
# 2. TIME-DOMAIN + SPECTRAL OPERATIONS
# =============================================================================


def infer_sampling_rate(meas_time: np.ndarray) -> float:
    t = np.asarray(meas_time, dtype=np.float64).reshape(-1)
    if t.size < 2:
        raise ValueError("measTime must contain at least two samples")
    if not np.isfinite(t).all():
        raise ValueError("measTime contains NaN/Inf")

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("measTime must be strictly increasing")

    return 1.0 / float(np.median(dt))


def assert_expected_sampling_rate(fs: float, cfg: PreprocessConfig) -> None:
    if not np.isclose(fs, cfg.target_fs, rtol=cfg.fs_rtol, atol=cfg.fs_atol):
        raise ValueError(
            f"Sampling rate mismatch: measured {fs:.6f} Hz, "
            f"expected {cfg.target_fs:.6f} Hz. This pipeline does not resample."
        )


def split_into_windows(
    x: np.ndarray,
    window_len: int,
    expected_windows: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN/Inf")

    n_windows = x.size // int(window_len)
    if n_windows != int(expected_windows):
        raise ValueError(
            f"Expected {expected_windows} complete windows but found {n_windows} "
            f"for signal length={x.size}, window_len={window_len}"
        )

    usable = n_windows * int(window_len)
    return x[:usable].reshape(n_windows, int(window_len))


def teager_energy_operator(x: np.ndarray) -> np.ndarray:
    """Discrete Teager-Kaiser Energy Operator applied before window splitting."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 3:
        raise ValueError("TKEO requires at least three samples")
    if not np.isfinite(x).all():
        raise ValueError("TKEO input contains NaN/Inf")

    y = np.empty_like(x)
    y[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    y[0] = y[1]
    y[-1] = y[-2]
    return y


def magnitude_spectrum(
    windows: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """DC removal -> Hann -> one-sided amplitude-corrected rFFT."""
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 2:
        raise ValueError(f"windows must be 2-D, got shape={windows.shape}")

    centered = windows - windows.mean(axis=1, keepdims=True)
    n = centered.shape[1]

    hann = np.hanning(n).astype(np.float64)
    coherent_sum = float(hann.sum())
    if coherent_sum <= 0:
        raise RuntimeError("Invalid Hann window coherent sum")

    spectrum = np.fft.rfft(centered * hann[None, :], axis=1)
    amplitude = np.abs(spectrum) / coherent_sum

    if n % 2 == 0:
        if amplitude.shape[1] > 2:
            amplitude[:, 1:-1] *= 2.0
    elif amplitude.shape[1] > 1:
        amplitude[:, 1:] *= 2.0

    freq_hz = np.fft.rfftfreq(n, d=1.0 / float(fs))
    return freq_hz, amplitude.astype(np.float32, copy=False)


# =============================================================================
# 3. Hz -> ORDER REMAP + HI SPEV
# =============================================================================


def make_order_axis(cfg: PreprocessConfig) -> np.ndarray:
    return np.linspace(
        0.0,
        cfg.max_order,
        cfg.n_order_bins,
        dtype=np.float32,
    )


def validate_order_support(
    freq_hz: np.ndarray,
    rpm: float,
    max_order: float,
) -> None:
    safe_max_order = float(freq_hz[-1]) * 60.0 / float(rpm)
    if max_order > safe_max_order * (1.0 + 1e-9):
        raise ValueError(
            f"Declared max_order={max_order:.3f} is not supported at "
            f"RPM={rpm:.3f}; safe maximum is {safe_max_order:.3f}."
        )


def spectra_to_order(
    spectra_hz: np.ndarray,
    freq_hz: np.ndarray,
    rpm: float,
    order_axis: np.ndarray,
) -> np.ndarray:
    """Approximate mean-RPM Hz-to-order remap via interpolation."""
    spectra_hz = np.asarray(spectra_hz, dtype=np.float32)
    freq_hz = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    order_axis = np.asarray(order_axis, dtype=np.float64).reshape(-1)

    if spectra_hz.ndim != 3:
        raise ValueError(
            "spectra_hz must have shape (windows, channels, freq_bins), "
            f"got {spectra_hz.shape}"
        )
    if spectra_hz.shape[-1] != freq_hz.size:
        raise ValueError("Frequency axis length does not match spectra")

    validate_order_support(freq_hz, rpm, float(order_axis[-1]))
    source_order = freq_hz * 60.0 / float(rpm)

    out = np.empty(
        (spectra_hz.shape[0], spectra_hz.shape[1], order_axis.size),
        dtype=np.float32,
    )

    for win in range(spectra_hz.shape[0]):
        for ch in range(spectra_hz.shape[1]):
            out[win, ch] = np.interp(
                order_axis,
                source_order,
                spectra_hz[win, ch],
            ).astype(np.float32, copy=False)

    return out


def spectrum_principal_energy_vector(
    spectra_order: np.ndarray,
    order_axis: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """SPEV-style max pooling over equal-width order buckets."""
    spectra_order = np.asarray(spectra_order, dtype=np.float32)
    order_axis = np.asarray(order_axis, dtype=np.float64).reshape(-1)

    if spectra_order.shape[-1] != order_axis.size:
        raise ValueError("Order axis length does not match spectra")

    edges = np.linspace(
        float(order_axis[0]),
        float(order_axis[-1]),
        int(n_buckets) + 1,
        dtype=np.float64,
    )
    out = np.empty(spectra_order.shape[:-1] + (n_buckets,), dtype=np.float32)

    for bucket in range(n_buckets):
        left = int(np.searchsorted(order_axis, edges[bucket], side="left"))
        right = (
            order_axis.size
            if bucket == n_buckets - 1
            else int(np.searchsorted(order_axis, edges[bucket + 1], side="left"))
        )
        if right <= left:
            raise RuntimeError(
                f"Empty SPEV bucket {bucket}: left={left}, right={right}. "
                "Increase n_order_bins or reduce hi_n_buckets."
            )
        out[..., bucket] = np.max(spectra_order[..., left:right], axis=-1)

    return out


# =============================================================================
# 4. ONE MEASUREMENT
# =============================================================================


def process_measurement(
    mat_path: str | os.PathLike[str],
    rpm: float,
    cfg: PreprocessConfig,
    order_axis: np.ndarray,
    branch: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Process only the requested branch(es) for one measurement.

    HI branch:
        uses the complete raw measurement directly. It does not depend on
        ``window_len`` or ``expected_windows_per_measurement``.

    Localization branch:
        uses the fixed subwindow configuration from ``PreprocessConfig``.
    """
    if branch not in BRANCH_CHOICES:
        raise ValueError(f"branch must be one of {BRANCH_CHOICES}, got {branch!r}")

    mat_path = Path(mat_path)
    mat = sio.loadmat(mat_path, squeeze_me=True)

    if "measTime" not in mat:
        raise KeyError(f"{mat_path}: missing measTime")

    fs = infer_sampling_rate(mat["measTime"])
    assert_expected_sampling_rate(fs, cfg)

    need_hi = _branch_enabled(branch, "hi")
    need_loc = _branch_enabled(branch, "localize")

    n_channels = len(cfg.channels)
    n_windows = cfg.expected_windows_per_measurement

    hi_channels: list[np.ndarray] = []
    loc_hz: np.ndarray | None = None
    reference_freq_hi: np.ndarray | None = None
    reference_freq_loc: np.ndarray | None = None

    if need_loc:
        loc_hz = np.empty(
            (n_windows, n_channels, cfg.window_len // 2 + 1),
            dtype=np.float32,
        )

    for ch_idx, channel in enumerate(cfg.channels):
        if channel not in mat:
            raise KeyError(f"{mat_path}: missing channel {channel!r}")

        raw = np.asarray(mat[channel], dtype=np.float64).reshape(-1)
        if raw.size < 4:
            raise ValueError(f"{mat_path} {channel}: signal is too short")
        if not np.isfinite(raw).all():
            raise ValueError(f"{mat_path} {channel}: signal contains NaN/Inf")

        # HI: one full measurement -> one spectrum -> one SPEV row.
        if need_hi:
            freq_hi, amp_hi = magnitude_spectrum(raw[None, :], fs)
            if reference_freq_hi is None:
                reference_freq_hi = freq_hi
            elif not np.allclose(
                freq_hi,
                reference_freq_hi,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"{mat_path}: HI channels have different signal lengths/frequency axes"
                )
            hi_channels.append(amp_hi[0])

        # Localization only: TKEO -> fixed subwindows.
        if need_loc:
            teager = teager_energy_operator(raw)
            teager_windows = split_into_windows(
                teager,
                cfg.window_len,
                cfg.expected_windows_per_measurement,
            )
            freq_loc, amp_loc = magnitude_spectrum(teager_windows, fs)
            if reference_freq_loc is None:
                reference_freq_loc = freq_loc
            elif not np.allclose(
                freq_loc,
                reference_freq_loc,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(f"{mat_path}: localization frequency axis mismatch")
            assert loc_hz is not None
            loc_hz[:, ch_idx] = amp_loc

    x_hi_spev: np.ndarray | None = None
    x_localize_order: np.ndarray | None = None

    if need_hi:
        if reference_freq_hi is None or len(hi_channels) != n_channels:
            raise RuntimeError(f"{mat_path}: HI branch produced incomplete data")
        hi_hz = np.stack(hi_channels, axis=0)[None, ...].astype(
            np.float32,
            copy=False,
        )
        x_hi_order = spectra_to_order(
            hi_hz,
            reference_freq_hi,
            rpm,
            order_axis,
        )
        x_hi_spev = spectrum_principal_energy_vector(
            x_hi_order,
            order_axis,
            cfg.hi_n_buckets,
        )[0]

    if need_loc:
        if loc_hz is None or reference_freq_loc is None:
            raise RuntimeError(f"{mat_path}: localization branch produced no data")
        x_localize_order = spectra_to_order(
            loc_hz,
            reference_freq_loc,
            rpm,
            order_axis,
        )

    return x_hi_spev, x_localize_order


# =============================================================================
# 5. ONE BEARING -> MINIMAL OUTPUTS
# =============================================================================


def _selected_output_names(branch: str) -> tuple[str, ...]:
    names = ["W.npy", "measurement_id.npy"]
    if _branch_enabled(branch, "hi"):
        names.append("X_hi_spev.npy")
    if _branch_enabled(branch, "localize"):
        names.append("X_localize_teager_order.npy")
    return tuple(names)


def _prepare_output_dir(path: Path, branch: str, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    selected = _selected_output_names(branch)
    existing = [name for name in selected if (path / name).exists()]

    if existing and not overwrite:
        raise FileExistsError(
            f"{path} already contains outputs for this run: {existing}. "
            "Use --overwrite to replace them."
        )

    if overwrite:
        for name in existing:
            (path / name).unlink()


def preprocess_bearing(
    bearing_id: str,
    base_dir: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    cfg: PreprocessConfig,
    branch: str,
    measurement_max: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Preprocess one bearing and save only the arrays required downstream."""
    cfg.validate()
    if branch not in BRANCH_CHOICES:
        raise ValueError(f"branch must be one of {BRANCH_CHOICES}, got {branch!r}")

    files = list_measurement_files(
        bearing_id,
        base_dir=base_dir,
        measurement_max=measurement_max,
    )
    op_df = load_operating_conditions(bearing_id, base_dir=base_dir)
    op_lookup = op_df.set_index("idx")

    n_measurements = len(files)
    n_channels = len(cfg.channels)
    need_loc = _branch_enabled(branch, "localize")
    n_windows = cfg.expected_windows_per_measurement if need_loc else 0
    total_windows = n_measurements * n_windows

    order_axis = make_order_axis(cfg)

    out_dir = Path(output_root) / bearing_id
    _prepare_output_dir(out_dir, branch=branch, overwrite=overwrite)

    x_hi_mm = None
    x_loc_mm = None

    if _branch_enabled(branch, "hi"):
        x_hi_mm = open_memmap(
            out_dir / "X_hi_spev.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_measurements, n_channels, cfg.hi_n_buckets),
        )

    if _branch_enabled(branch, "localize"):
        x_loc_mm = open_memmap(
            out_dir / "X_localize_teager_order.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_windows, n_channels, cfg.n_order_bins),
        )

    # Shared, non-redundant metadata: one row per measurement only.
    W = np.empty((n_measurements, len(W_COLUMNS)), dtype=np.float32)
    measurement_ids = np.empty(n_measurements, dtype=np.int64)

    for file_pos, (measurement_id, mat_path) in enumerate(files):
        w_vector, rpm = get_measurement_conditions(measurement_id, op_lookup)
        x_hi, x_loc = process_measurement(
            mat_path=mat_path,
            rpm=rpm,
            cfg=cfg,
            order_axis=order_axis,
            branch=branch,
        )

        W[file_pos] = w_vector
        measurement_ids[file_pos] = measurement_id

        if x_hi_mm is not None:
            if x_hi is None or x_hi.shape != (n_channels, cfg.hi_n_buckets):
                raise RuntimeError(
                    f"{bearing_id} M{measurement_id}: unexpected HI shape "
                    f"{None if x_hi is None else x_hi.shape}"
                )
            x_hi_mm[file_pos] = x_hi

        if x_loc_mm is not None:
            expected_shape = (n_windows, n_channels, cfg.n_order_bins)
            if x_loc is None or x_loc.shape != expected_shape:
                raise RuntimeError(
                    f"{bearing_id} M{measurement_id}: unexpected localization shape "
                    f"{None if x_loc is None else x_loc.shape}; "
                    f"expected {expected_shape}"
                )
            start = file_pos * n_windows
            stop = start + n_windows
            x_loc_mm[start:stop] = x_loc

        done = file_pos + 1
        if done == 1 or done % 100 == 0 or done == n_measurements:
            print(
                f"[{bearing_id}] {done}/{n_measurements} measurements | "
                f"M{measurement_id} | rpm={rpm:.2f}"
            )

    if x_hi_mm is not None:
        x_hi_mm.flush()
        del x_hi_mm
    if x_loc_mm is not None:
        x_loc_mm.flush()
        del x_loc_mm

    np.save(out_dir / "W.npy", W)
    np.save(out_dir / "measurement_id.npy", measurement_ids)

    outputs: list[str] = []
    if _branch_enabled(branch, "hi"):
        outputs.append("X_hi_spev.npy")
    if _branch_enabled(branch, "localize"):
        outputs.append("X_localize_teager_order.npy")
    outputs.extend(["W.npy", "measurement_id.npy"])

    print(f"[{bearing_id}] DONE | branch={branch} | output={out_dir}")
    print(f"[{bearing_id}] files: {', '.join(outputs)}")

    return {
        "bearing_id": bearing_id,
        "branch": branch,
        "n_measurements": n_measurements,
        "output_dir": str(out_dir),
        "outputs": outputs,
    }


# =============================================================================
# 6. MULTI-BEARING DRIVER
# =============================================================================


def preprocess_many(
    bearings: Sequence[str],
    base_dir: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    cfg: PreprocessConfig,
    branch: str,
    measurement_max_by_bearing: dict[str, int] | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    if not bearings:
        raise ValueError("No bearings supplied")
    if branch not in BRANCH_CHOICES:
        raise ValueError(f"branch must be one of {BRANCH_CHOICES}, got {branch!r}")

    measurement_max_by_bearing = measurement_max_by_bearing or {}
    summaries: list[dict[str, object]] = []

    print("=" * 72)
    print("PADERBORN PREPROCESSING")
    print("=" * 72)
    print(f"Branch            : {branch}")
    print(f"Bearings          : {list(bearings)}")
    print(f"Target fs         : {cfg.target_fs:.1f} Hz (strict; no resampling)")
    if _branch_enabled(branch, "hi"):
        print(f"HI SPEV buckets   : {cfg.hi_n_buckets}")
    if _branch_enabled(branch, "localize"):
        print(f"LOC window length : {cfg.window_len}")
        print(f"LOC windows/meas  : {cfg.expected_windows_per_measurement}")
        print(f"LOC order bins    : {cfg.n_order_bins}")
    print(f"Fixed max order   : {cfg.max_order:.3f}")
    print("Normalization     : NONE here; fit on TRAIN bearings later")
    print()

    for bearing_id in bearings:
        summaries.append(
            preprocess_bearing(
                bearing_id=bearing_id,
                base_dir=base_dir,
                output_root=output_root,
                cfg=cfg,
                branch=branch,
                measurement_max=measurement_max_by_bearing.get(bearing_id),
                overwrite=overwrite,
            )
        )

    return summaries


# =============================================================================
# 7. CLI
# =============================================================================


def parse_measurement_max(items: Iterable[str] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if items is None:
        return result

    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid --measurement-max {item!r}; expected BEARING=INTEGER"
            )
        bearing, value = item.split("=", 1)
        bearing = bearing.strip()
        if not bearing:
            raise argparse.ArgumentTypeError("Bearing id cannot be empty")
        try:
            value_int = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid measurement maximum in {item!r}"
            ) from exc
        if value_int < 1:
            raise argparse.ArgumentTypeError(
                f"Measurement maximum must be >=1 in {item!r}"
            )
        result[bearing] = value_int

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HI/SPEV and TKEO-localization preprocessing for Paderborn bearings"
    )
    parser.add_argument("--base-dir", default="data")
    parser.add_argument("--output-root", default="processed_dual")
    parser.add_argument(
        "--branch",
        choices=BRANCH_CHOICES,
        default="hi",
        help="Select what to compute: hi, localize, or both. Default: hi.",
    )
    parser.add_argument(
        "--bearings",
        nargs="+",
        default=["B09", "B10", "B11", "B12", "B13", "B14", "B17"],
    )
    parser.add_argument("--target-fs", type=float, default=64_000.0)
    parser.add_argument("--window-len", type=int, default=4096)
    parser.add_argument("--expected-windows", type=int, default=25)
    parser.add_argument("--n-order-bins", type=int, default=2049)
    parser.add_argument(
        "--hi-n-buckets",
        type=int,
        default=20,
        help="Number of equal-width order buckets for HI SPEV max pooling.",
    )
    parser.add_argument(
        "--max-order",
        type=float,
        default=600.0,
        help="Fixed experiment-wide maximum order.",
    )
    parser.add_argument(
        "--measurement-max",
        action="append",
        default=None,
        metavar="BEARING=MAX_ID",
        help=(
            "Optional repeated cutoff, e.g. --measurement-max B11=100. "
            "Bearings without a cutoff use all measurements."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    measurement_max = parse_measurement_max(args.measurement_max)

    cfg = PreprocessConfig(
        target_fs=args.target_fs,
        window_len=args.window_len,
        expected_windows_per_measurement=args.expected_windows,
        n_order_bins=args.n_order_bins,
        max_order=args.max_order,
        hi_n_buckets=args.hi_n_buckets,
    )
    cfg.validate()

    preprocess_many(
        bearings=args.bearings,
        base_dir=args.base_dir,
        output_root=args.output_root,
        cfg=cfg,
        branch=args.branch,
        measurement_max_by_bearing=measurement_max,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
