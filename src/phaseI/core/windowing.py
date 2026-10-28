"""
windowing.py

Data layer cho Phase I / HI branch sau preprocessing SPEV mới.

Contract đầu vào mới (mỗi bearing một thư mục):

    <data_dir>/<Bxx>/X_hi_spev.npy
        shape (N_measurements, n_channels, hi_n_buckets)

    <data_dir>/<Bxx>/W.npy
        shape (N_measurements, w_dim)

    <data_dir>/<Bxx>/measurement_id.npy
        shape (N_measurements,)

Điểm quan trọng:
    - preprocessing đã biến MỖI measurement thành đúng MỘT SPEV sample;
      file này KHÔNG tạo thêm sliding window/sub-window theo thời gian.
    - X_hi_spev.npy đã là .npy nên có thể mmap trực tiếp; không còn bước
      .npz -> extract X_db -> tạo mmap cache thứ hai.
    - W và measurement_id nhỏ nên giữ trong RAM.
    - Normalization được fit CHỈ từ training bearings rồi freeze cho
      validation/test. Không fit theo từng measurement và không dùng OC
      để chọn scaler riêng cho X.
    - X dùng đúng pipeline:
          raw SPEV -> log1p -> feature-wise StandardScaler
      mỗi (channel, bucket) là một feature với mean/std riêng.
    - W dùng feature-wise StandardScaler:
          raw W -> (W - w_mean) / w_std
      cho dynamic load, static load và RPM.

Tên ``cycle`` trong EngineData/WindowBatch được giữ để tương thích với
train.py hiện tại, nhưng giá trị thực tế là ``measurement_id``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


HI_FILENAME = "X_hi_spev.npy"
W_FILENAME = "W.npy"
MEASUREMENT_ID_FILENAME = "measurement_id.npy"


# ============================================================
# LAZY NORMALIZED X
# ============================================================

class LazyNormalizedArray:
    """
    Wrapper quanh X_hi_spev.npy được mở bằng mmap.

    Hỗ trợ:
        len(X)
        X.shape
        X[start:end]

    Khi slice:
        đọc đúng slice từ disk
        -> log1p
        -> feature-wise StandardScaler bằng train-fitted mean/std
        -> trả float32 ndarray

    Không cho phép np.asarray(X) toàn bộ để tránh vô tình materialize
    cả bearing. Dù SPEV mới nhỏ hơn FFT cũ rất nhiều, giữ hành vi này
    giúp train.py hiện tại tiếp tục hoạt động mà không đổi contract.
    """

    def __init__(
        self,
        raw: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self._raw = raw

        self._mean = np.asarray(
            mean,
            dtype=np.float32,
        )

        self._std = np.asarray(
            std,
            dtype=np.float32,
        )

        if raw.ndim != 3:
            raise ValueError(
                "X_hi_spev phải 3D "
                "(measurement, channel, bucket), "
                f"got {raw.shape}"
            )

        expected = (
            1,
            raw.shape[1],
            raw.shape[2],
        )

        if self._mean.shape != expected:
            raise ValueError(
                "x_mean shape không khớp X_hi_spev: "
                f"{self._mean.shape} vs {expected}"
            )

        if self._std.shape != self._mean.shape:
            raise ValueError(
                "x_std shape không khớp x_mean"
            )

        if not np.isfinite(self._mean).all():
            raise ValueError(
                "x_mean có NaN/Inf"
            )

        if not np.isfinite(self._std).all():
            raise ValueError(
                "x_std có NaN/Inf"
            )

        if np.any(self._std <= 0):
            raise ValueError(
                "x_std phải > 0"
            )

    @property
    def shape(self):
        return self._raw.shape

    @property
    def ndim(self):
        return self._raw.ndim

    @property
    def dtype(self):
        # Output sau normalize luôn float32.
        return np.dtype(np.float32)

    def __len__(self):
        return len(self._raw)

    def __getitem__(
        self,
        index,
    ):
        """Chỉ page-in phần dữ liệu được request."""

        x = np.asarray(
            self._raw[index],
            dtype=np.float32,
        )

        if np.any(x < 0):
            raise ValueError(
                "X_hi_spev phải không âm trước log1p; "
                f"min={float(x.min())}"
            )

        # log1p được áp dụng TRƯỚC StandardScaler, đúng cùng transform
        # đã dùng khi fit x_mean/x_std trên training bearings.
        x = np.log1p(x)

        # Batch/slice:
        #   (B, channels, buckets)
        if x.ndim == 3:
            x = (
                x - self._mean
            ) / self._std

        # Một measurement:
        #   (channels, buckets)
        elif x.ndim == 2:
            x = (
                x - self._mean[0]
            ) / self._std[0]

        else:
            raise ValueError(
                "Unexpected X slice shape: "
                f"{x.shape}"
            )

        return x.astype(
            np.float32,
            copy=False,
        )

    def __array__(
        self,
        dtype=None,
    ):
        raise RuntimeError(
            "Không được convert toàn bộ engine.X "
            "sang ndarray. "
            "Hãy dùng engine.X[start:end]."
        )


# ============================================================
# DATA CONTAINERS
# ============================================================

@dataclass
class EngineData:
    """
    Dữ liệu của một bearing theo thứ tự measurement tăng dần.

    X:
        LazyNormalizedArray.
        Raw layout: (N_measurements, channels, SPEV_buckets).

    W:
        Operating conditions đã StandardScaler bằng statistics học từ
        training bearings. Shape (N_measurements, w_dim).

    cycle:
        Tên field legacy để tương thích train.py.
        Nội dung thực tế là measurement_id, shape (N_measurements,).
    """

    unit_number: str
    X: LazyNormalizedArray
    W: np.ndarray
    cycle: np.ndarray


@dataclass
class WindowBatch:
    # Giữ tên class để tương thích train.py hiện tại.
    # Với preprocess mới, mỗi row của X training là MỘT measurement SPEV,
    # không còn là một sub-window của measurement.
    #
    # Training:
    #   X: (batch, channels, SPEV_buckets)
    #
    # Latent extraction:
    #   X có thể là (batch, z_dim)
    X: np.ndarray
    W: np.ndarray
    unit_number: np.ndarray
    cycle: np.ndarray


# ============================================================
# NORMALIZATION
# ============================================================

def load_normalization(
    path: str | Path = "normalization_v1.npz",
) -> dict[str, np.ndarray]:
    """
    Load StandardScaler statistics đã fit CHỈ từ training bearings.

    Contract:
        x_mean: (1, channels, SPEV_buckets)
        x_std:  (1, channels, SPEV_buckets)
            Statistics của ``log1p(X_raw)`` theo từng feature
            (channel, bucket).

        w_mean: (w_dim,)
        w_std:  (w_dim,)
            Statistics của raw operating conditions theo từng biến.

    Validation/test chỉ dùng các statistics này để transform; không fit lại.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        required = (
            "x_mean",
            "x_std",
            "w_mean",
            "w_std",
        )

        for key in required:
            if key not in data.files:
                raise KeyError(
                    f"{path} thiếu '{key}'. "
                    "Normalization cũ dùng w_min/w_range không còn tương thích; "
                    "hãy fit lại scaler từ training bearings."
                )

        norm = {
            "x_mean": np.asarray(
                data["x_mean"],
                dtype=np.float32,
            ),
            "x_std": np.asarray(
                data["x_std"],
                dtype=np.float32,
            ),
            "w_mean": np.asarray(
                data["w_mean"],
                dtype=np.float32,
            ),
            "w_std": np.asarray(
                data["w_std"],
                dtype=np.float32,
            ),
        }

    if norm["x_mean"].ndim != 3:
        raise ValueError(
            "x_mean phải có shape "
            "(1, channels, SPEV_buckets), "
            f"got {norm['x_mean'].shape}"
        )

    if norm["x_mean"].shape[0] != 1:
        raise ValueError(
            "x_mean dimension đầu phải bằng 1, "
            f"got {norm['x_mean'].shape}"
        )

    if norm["x_std"].shape != norm["x_mean"].shape:
        raise ValueError(
            "x_std shape không khớp x_mean: "
            f"{norm['x_std'].shape} vs "
            f"{norm['x_mean'].shape}"
        )

    if norm["w_mean"].ndim != 1:
        raise ValueError(
            "w_mean phải 1D, "
            f"got {norm['w_mean'].shape}"
        )

    if norm["w_std"].shape != norm["w_mean"].shape:
        raise ValueError(
            "w_std shape không khớp w_mean: "
            f"{norm['w_std'].shape} vs "
            f"{norm['w_mean'].shape}"
        )

    for name in ("x_mean", "x_std", "w_mean", "w_std"):
        if not np.isfinite(norm[name]).all():
            raise ValueError(
                f"{name} có NaN/Inf"
            )

    if np.any(norm["x_std"] <= 0):
        raise ValueError(
            "x_std phải > 0"
        )

    if np.any(norm["w_std"] <= 0):
        raise ValueError(
            "w_std phải > 0"
        )

    return norm


# ============================================================
# NEW PREPROCESS CONTRACT
# ============================================================

def _bearing_paths(
    engine_id: str,
    data_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Resolve ba file tối thiểu mà HI branch cần."""

    bearing_dir = (
        Path(data_dir)
        / engine_id
    )

    if not bearing_dir.is_dir():
        raise FileNotFoundError(
            f"Không có thư mục bearing: {bearing_dir}"
        )

    x_path = bearing_dir / HI_FILENAME
    w_path = bearing_dir / W_FILENAME
    measurement_path = (
        bearing_dir
        / MEASUREMENT_ID_FILENAME
    )

    missing = [
        str(path)
        for path in (
            x_path,
            w_path,
            measurement_path,
        )
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{engine_id}: thiếu output preprocess: "
            + ", ".join(missing)
        )

    return (
        x_path,
        w_path,
        measurement_path,
    )


def fit_normalization(
    train_engine_ids: list[str],
    data_dir: str | Path = "processed_dual",
    output_path: str | Path = "normalization_v1.npz",
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    Fit scaler CHỈ trên training bearings và lưu statistics ra ``.npz``.

    X:
        X_raw -> log1p(X_raw) -> StandardScaler theo từng
        (channel, SPEV bucket).

    W:
        W_raw -> StandardScaler theo từng operating-condition variable.

    Không có normalization theo measurement và không chia scaler theo OC.
    Mỗi measurement của mọi training bearing đóng góp một observation vào
    cùng một global training scaler.

    Với feature có variance gần 0, std được đặt bằng 1.0, tương đương
    hành vi của StandardScaler: feature đó chỉ được center, không gây chia 0.
    """

    if not train_engine_ids:
        raise ValueError(
            "train_engine_ids không được rỗng"
        )

    if eps <= 0:
        raise ValueError(
            "eps phải > 0"
        )

    x_sum = None
    x_sq_sum = None
    w_sum = None
    w_sq_sum = None
    n_x = 0
    n_w = 0
    x_shape = None
    w_dim = None

    for engine_id in train_engine_ids:
        x_path, w_path, measurement_path = _bearing_paths(
            engine_id=engine_id,
            data_dir=data_dir,
        )

        X_raw = np.load(
            x_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        W_raw = np.asarray(
            np.load(
                w_path,
                allow_pickle=False,
            ),
            dtype=np.float64,
        )

        measurement_id = _validate_measurement_ids(
            engine_id=engine_id,
            measurement_id=np.load(
                measurement_path,
                allow_pickle=False,
            ),
        )

        if X_raw.ndim != 3:
            raise ValueError(
                f"{engine_id}: X_hi_spev phải 3D, got {X_raw.shape}"
            )

        if W_raw.ndim != 2:
            raise ValueError(
                f"{engine_id}: W phải 2D, got {W_raw.shape}"
            )

        if not (
            len(X_raw)
            == len(W_raw)
            == len(measurement_id)
        ):
            raise ValueError(
                f"{engine_id}: X/W/measurement_id khác số dòng"
            )

        if len(X_raw) == 0:
            raise ValueError(
                f"{engine_id}: không có measurement"
            )

        if not np.isfinite(X_raw).all():
            raise ValueError(
                f"{engine_id}: X_hi_spev có NaN/Inf"
            )

        if np.any(X_raw < 0):
            raise ValueError(
                f"{engine_id}: X_hi_spev phải không âm trước log1p"
            )

        if not np.isfinite(W_raw).all():
            raise ValueError(
                f"{engine_id}: W có NaN/Inf"
            )

        current_x_shape = tuple(X_raw.shape[1:])
        current_w_dim = int(W_raw.shape[1])

        if x_shape is None:
            x_shape = current_x_shape
            x_sum = np.zeros(x_shape, dtype=np.float64)
            x_sq_sum = np.zeros(x_shape, dtype=np.float64)
        elif current_x_shape != x_shape:
            raise ValueError(
                f"{engine_id}: X shape={current_x_shape}, expected={x_shape}"
            )

        if w_dim is None:
            w_dim = current_w_dim
            w_sum = np.zeros(w_dim, dtype=np.float64)
            w_sq_sum = np.zeros(w_dim, dtype=np.float64)
        elif current_w_dim != w_dim:
            raise ValueError(
                f"{engine_id}: W dim={current_w_dim}, expected={w_dim}"
            )

        # X_hi_spev nhỏ; convert từng bearing sang float64 sau log1p để
        # tính train statistics ổn định hơn. Không hề fit riêng từng bearing.
        X_log = np.log1p(
            np.asarray(
                X_raw,
                dtype=np.float64,
            )
        )

        x_sum += X_log.sum(axis=0)
        x_sq_sum += (X_log ** 2).sum(axis=0)
        n_x += int(X_log.shape[0])

        w_sum += W_raw.sum(axis=0)
        w_sq_sum += (W_raw ** 2).sum(axis=0)
        n_w += int(W_raw.shape[0])

    assert x_sum is not None
    assert x_sq_sum is not None
    assert w_sum is not None
    assert w_sq_sum is not None
    assert x_shape is not None
    assert w_dim is not None

    x_mean_2d = x_sum / n_x
    x_var_2d = np.maximum(
        x_sq_sum / n_x - x_mean_2d ** 2,
        0.0,
    )
    x_std_2d = np.sqrt(x_var_2d)
    x_std_2d = np.where(
        x_std_2d < eps,
        1.0,
        x_std_2d,
    )

    w_mean = w_sum / n_w
    w_var = np.maximum(
        w_sq_sum / n_w - w_mean ** 2,
        0.0,
    )
    w_std = np.sqrt(w_var)
    w_std = np.where(
        w_std < eps,
        1.0,
        w_std,
    )

    norm = {
        "x_mean": x_mean_2d[None, ...].astype(np.float32),
        "x_std": x_std_2d[None, ...].astype(np.float32),
        "w_mean": w_mean.astype(np.float32),
        "w_std": w_std.astype(np.float32),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        **norm,
        train_engines=np.asarray(
            train_engine_ids,
            dtype=str,
        ),
        x_transform=np.asarray(
            "log1p_then_standard_scaler"
        ),
        w_transform=np.asarray(
            "standard_scaler"
        ),
    )

    print(
        "Saved normalization:",
        output_path,
    )
    print(
        f"  fit measurements: X={n_x}, W={n_w}"
    )
    print(
        f"  x_mean/x_std: {norm['x_mean'].shape}"
    )
    print(
        f"  w_mean/w_std: {norm['w_mean'].shape}"
    )

    return norm


def _validate_measurement_ids(
    engine_id: str,
    measurement_id: np.ndarray,
) -> np.ndarray:
    """
    Preprocess mới có đúng một row cho mỗi measurement.

    Ta yêu cầu ID tăng nghiêm ngặt để giữ đúng thứ tự trajectory.
    Không yêu cầu phải liên tiếp tuyệt đối; nếu thiếu measurement thật
    thì train.py có thể dùng chính ID để ngắt temporal constraint.
    """

    raw = np.asarray(measurement_id)

    if raw.ndim != 1:
        raise ValueError(
            f"{engine_id}: measurement_id phải 1D, "
            f"got {raw.shape}"
        )

    if raw.size == 0:
        raise ValueError(
            f"{engine_id}: measurement_id rỗng"
        )

    if not np.isfinite(raw).all():
        raise ValueError(
            f"{engine_id}: measurement_id có NaN/Inf"
        )

    as_int = raw.astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        raw,
        as_int,
    ):
        raise ValueError(
            f"{engine_id}: measurement_id phải là số nguyên"
        )

    if np.any(np.diff(as_int) <= 0):
        raise ValueError(
            f"{engine_id}: measurement_id phải tăng nghiêm ngặt; "
            "mỗi measurement chỉ được xuất hiện một lần"
        )

    return as_int


# ============================================================
# LOAD ONE ENGINE
# ============================================================

def load_engine(
    engine_id: str,
    data_dir: str | Path = "processed_dual",
    normalization_path: str | Path = "normalization_v1.npz",
    mmap_dir: str | Path | None = None,
    norm: dict[str, np.ndarray] | None = None,
) -> EngineData:
    """
    Load một bearing từ output trực tiếp của preprocess SPEV mới.

    ``mmap_dir`` chỉ được giữ như legacy argument để caller cũ không vỡ.
    Không còn tạo cache riêng vì ``X_hi_spev.npy`` vốn đã mmap trực tiếp
    được.
    """

    # Legacy compatibility: intentionally unused.
    _ = mmap_dir

    if norm is None:
        norm = load_normalization(
            normalization_path
        )

    (
        x_path,
        w_path,
        measurement_path,
    ) = _bearing_paths(
        engine_id=engine_id,
        data_dir=data_dir,
    )

    # X SPEV đã là .npy -> mmap trực tiếp, không extract/cache trung gian.
    X_raw = np.load(
        x_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    # W và measurement_id nhỏ -> RAM.
    W_raw = np.asarray(
        np.load(
            w_path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    measurement_id = _validate_measurement_ids(
        engine_id=engine_id,
        measurement_id=np.load(
            measurement_path,
            allow_pickle=False,
        ),
    )

    # ========================================================
    # CONTRACT VALIDATION
    # ========================================================

    if X_raw.ndim != 3:
        raise ValueError(
            f"{engine_id}: X_hi_spev phải có shape "
            "(N, channels, buckets), "
            f"got {X_raw.shape}"
        )

    if W_raw.ndim != 2:
        raise ValueError(
            f"{engine_id}: W phải 2D, "
            f"got {W_raw.shape}"
        )

    if not (
        len(X_raw)
        == len(W_raw)
        == len(measurement_id)
    ):
        raise ValueError(
            f"{engine_id}: "
            "X_hi_spev/W/measurement_id khác số dòng: "
            f"X={len(X_raw)}, "
            f"W={len(W_raw)}, "
            f"measurement_id={len(measurement_id)}"
        )

    if len(X_raw) == 0:
        raise ValueError(
            f"{engine_id}: không có measurement"
        )

    if not np.isfinite(X_raw).all():
        raise ValueError(
            f"{engine_id}: X_hi_spev có NaN/Inf"
        )

    if np.any(X_raw < 0):
        raise ValueError(
            f"{engine_id}: X_hi_spev phải không âm trước log1p"
        )

    if not np.isfinite(W_raw).all():
        raise ValueError(
            f"{engine_id}: W có NaN/Inf"
        )

    expected_x = norm["x_mean"].shape[1:]

    if X_raw.shape[1:] != expected_x:
        raise ValueError(
            f"{engine_id}: X_hi_spev shape={X_raw.shape[1:]}, "
            "normalization expects="
            f"{expected_x}. "
            "Hãy fit lại normalization bằng output SPEV mới."
        )

    if W_raw.shape[1] != norm["w_mean"].shape[0]:
        raise ValueError(
            f"{engine_id}: W dim={W_raw.shape[1]}, "
            "normalization expects="
            f"{norm['w_mean'].shape[0]}"
        )

    # ========================================================
    # STANDARDIZE W
    # ========================================================

    # Một global scaler đã fit trên training bearings. Không fit lại theo
    # bearing/measurement và không clip validation/test.
    W = (
        (
            W_raw
            - norm["w_mean"]
        )
        / norm["w_std"]
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(W).all():
        raise ValueError(
            f"{engine_id}: W normalized có NaN/Inf"
        )

    # ========================================================
    # LAZY X NORMALIZATION
    # ========================================================

    X = LazyNormalizedArray(
        raw=X_raw,
        mean=norm["x_mean"],
        std=norm["x_std"],
    )

    return EngineData(
        unit_number=engine_id,
        X=X,
        W=W,
        cycle=measurement_id,
    )


# ============================================================
# LOAD MULTIPLE ENGINES
# ============================================================

def load_engines(
    engine_ids: list[str],
    data_dir: str | Path = "processed_dual",
    normalization_path: str | Path = "normalization_v1.npz",
    mmap_dir: str | Path | None = None,
) -> list[EngineData]:
    """Load nhiều bearing, dùng chung một bộ train-fitted normalization."""

    if not engine_ids:
        raise ValueError(
            "engine_ids không được rỗng"
        )

    # Load normalization đúng một lần.
    norm = load_normalization(
        normalization_path
    )

    engines = []

    for engine_id in engine_ids:
        engine = load_engine(
            engine_id=engine_id,
            data_dir=data_dir,
            normalization_path=(
                normalization_path
            ),
            mmap_dir=mmap_dir,
            norm=norm,
        )

        engines.append(engine)

        print(
            f"{engine_id}: "
            f"X={engine.X.shape}, "
            f"W={engine.W.shape}, "
            f"measurements={len(engine.cycle)}"
        )

    return engines
