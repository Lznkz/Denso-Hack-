"""
metrics.py

Danh gia scalar latent Z cua Phase I xem no co cac dac trung mong muon
cua mot Health Index (HI) hay khong.

Pipeline hien tai
-----------------
    X_tilde -> Encoder -> Z, voi Z shape (N_measurements, 1)

Training dung negative-gradient constraint theo convention:
    dZ/dt <= 0

Do do metrics.py KHONG tao HI bang distance-to-healthy-center nua.
Z duoc danh gia truc tiep.

Core HI criteria theo Bajarunas et al. (Appendix C):
    - Monotonicity (Mon)
    - Trendability (Tren)
    - Prognosability (Prog)
    - Mutual Information (MutInf), neu co target/RUL that

Ngoai ra co cac diagnostic rieng cho pipeline nay:
    - NG violation rate/magnitude: Z co vi pham huong giam hay khong
    - Plateau rate: Z dung yen bao nhieu
    - OC dependence: Z con phu thuoc manh vao W hay khong
    - Reconstruction MAE per measurement: diagnostic phu cua AE

Luu y:
    Monotonicity C.1 cua paper KHONG cung mot thu voi NG violation.
    Mon C.1 chi do muc do thay doi giua cac cap quan sat theo cong thuc paper;
    NG diagnostic moi kiem tra dung huong dZ <= 0 cua model hien tai.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


DEFAULT_W_NAMES = (
    "dynamic_load",
    "static_load",
    "rpm",
)


# ============================================================
# VALIDATION / SHAPE HELPERS
# ============================================================

def _as_scalar_z(z: np.ndarray) -> np.ndarray:
    """Return Z as finite 1D float64 array.

    Accepted shapes:
        (N,)
        (N, 1)

    Phase I co dinh z_dim = 1; latent nhieu chieu bi tu choi de tranh
    vo tinh dung metrics scalar cho mot representation khac contract.
    """

    z = np.asarray(z, dtype=np.float64)

    if z.ndim == 2:
        if z.shape[1] != 1:
            raise ValueError(
                "Phase I yeu cau Z co shape (N, 1); "
                f"nhan {z.shape}."
            )
        z = z[:, 0]

    elif z.ndim != 1:
        raise ValueError(
            "Z phai co shape (N,) hoac (N, 1); "
            f"nhan {z.shape}."
        )

    if z.size == 0:
        raise ValueError("Z rong.")

    if not np.isfinite(z).all():
        raise ValueError("Z chua NaN/Inf.")

    return z


def _as_measurement_id(
    measurement_id: np.ndarray | None,
    n: int,
) -> np.ndarray:
    """Validate measurement IDs, or use 0..N-1 when not supplied."""

    if measurement_id is None:
        return np.arange(n, dtype=np.int64)

    measurement_id = np.asarray(measurement_id)

    if measurement_id.ndim != 1:
        raise ValueError(
            "measurement_id phai 1D; "
            f"nhan {measurement_id.shape}."
        )

    if len(measurement_id) != n:
        raise ValueError(
            "Z va measurement_id khac do dai: "
            f"{n} vs {len(measurement_id)}."
        )

    if not np.issubdtype(measurement_id.dtype, np.number):
        raise ValueError("measurement_id phai la numeric array.")

    measurement_id = measurement_id.astype(np.float64, copy=False)

    if not np.isfinite(measurement_id).all():
        raise ValueError("measurement_id chua NaN/Inf.")

    if np.any(np.diff(measurement_id) <= 0):
        raise ValueError(
            "measurement_id phai tang nghiem ngat theo thoi gian."
        )

    return measurement_id


def _transition_mask(
    measurement_id: np.ndarray,
    consecutive_only: bool,
) -> np.ndarray:
    """Mask for transitions used by NG diagnostics.

    consecutive_only=True mirrors training: only M_i -> M_{i+1} where
    measurement IDs differ exactly by one are constrained.
    """

    if len(measurement_id) <= 1:
        return np.zeros(0, dtype=bool)

    if not consecutive_only:
        return np.ones(len(measurement_id) - 1, dtype=bool)

    return np.isclose(
        np.diff(measurement_id),
        1.0,
        rtol=0.0,
        atol=0.0,
    )


# ============================================================
# RECONSTRUCTION DIAGNOSTIC
# ============================================================

@torch.no_grad()
def reconstruction_mae_per_measurement(
    model: torch.nn.Module,
    X: Any,
    W: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """MAE reconstruction cho tung measurement tren normalized space.

    Parameters
    ----------
    X:
        Sliceable array-like co shape (N, channels, spev_buckets).
        Co the la LazyNormalizedArray tu windowing.py.
    W:
        Standardized operating conditions, shape (N, w_dim).

    Returns
    -------
    np.ndarray, shape (N,)
        Mean absolute reconstruction error tren channel va SPEV bucket.

    Day la diagnostic phu cua AE, KHONG phai HI chinh.
    """

    if batch_size <= 0:
        raise ValueError("batch_size phai > 0.")

    n = len(X)
    W = np.asarray(W, dtype=np.float32)

    if W.ndim != 2 or len(W) != n:
        raise ValueError(
            "W phai co shape (N, w_dim) va cung N voi X; "
            f"X N={n}, W={W.shape}."
        )

    model.eval()
    errors: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        x_np = np.asarray(
            X[start:end],
            dtype=np.float32,
        )
        w_np = W[start:end]

        if x_np.ndim != 3:
            raise ValueError(
                "X batch phai co shape (B, channels, buckets); "
                f"nhan {x_np.shape}."
            )

        x = torch.from_numpy(x_np).to(
            device=device,
            dtype=torch.float32,
        )
        w = torch.from_numpy(w_np).to(
            device=device,
            dtype=torch.float32,
        )

        x_hat, _ = model(x, w)

        err = torch.abs(x_hat - x).mean(dim=(1, 2))
        errors.append(err.cpu().numpy())

    if not errors:
        return np.empty(0, dtype=np.float64)

    return np.concatenate(errors).astype(
        np.float64,
        copy=False,
    )


# ============================================================
# HI CRITERIA - BAJARUNAS APPENDIX C
# ============================================================

def monotonicity_score(z: np.ndarray) -> float:
    """Bajarunas Appendix C, Eq. C.1.

    M = 1/(m-1) * sum_j |
        Ind(h[j+1]-h[j]) - Ind(h[j]-h[j+1])
    |

    voi Ind(x)=1 neu x>0, nguoc lai 0.

    Theo DUNG cong thuc paper, plateau (delta=0) dong gop 0; mot buoc
    thay doi theo BAT KY huong nao dong gop 1. Vi vay metric nay khong
    tu no kiem tra huong giam ma NG loss dang ep. De kiem tra dung huong,
    dung ng_violation_stats().
    """

    z = _as_scalar_z(z)

    if len(z) < 2:
        return float("nan")

    dz = np.diff(z)
    inc = (dz > 0.0).astype(np.float64)
    dec = (dz < 0.0).astype(np.float64)

    return float(
        np.mean(np.abs(inc - dec))
    )


def trendability_score(
    z: np.ndarray,
    measurement_id: np.ndarray | None = None,
) -> float:
    """Bajarunas Appendix C, Eq. C.2: |Spearman(t, HI)|.

    Score gan 1: Z co quan he don dieu manh voi thoi gian.
    Score gan 0: khong co trend don dieu ro rang.

    Absolute value duoc dung dung theo paper. Neu can kiem tra HUONG
    (pipeline nay mong Z giam), xem signed_trend_spearman().
    """

    z = _as_scalar_z(z)
    t = _as_measurement_id(measurement_id, len(z))

    if len(z) < 2 or np.allclose(z, z[0]):
        return 0.0

    rho = spearmanr(t, z).statistic

    if not np.isfinite(rho):
        return 0.0

    return float(abs(rho))


def signed_trend_spearman(
    z: np.ndarray,
    measurement_id: np.ndarray | None = None,
) -> float:
    """Signed Spearman diagnostic.

    Khac Eq. C.2 cua paper o cho KHONG lay absolute value.
    Voi convention hien tai dZ/dt <= 0, gia tri mong doi gan -1.
    """

    z = _as_scalar_z(z)
    t = _as_measurement_id(measurement_id, len(z))

    if len(z) < 2 or np.allclose(z, z[0]):
        return 0.0

    rho = spearmanr(t, z).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def prognosability_score(
    z_by_bearing: Mapping[str, np.ndarray],
    eps: float = 1e-12,
) -> float:
    """Bajarunas Appendix C, Eq. C.3.

    P = exp(
        - std(h_end_u)
          / mean(|h_end_u - h_0_u|)
    )

    across all bearings u.

    Score gan 1 nghia la gia tri cuoi doi cua cac bearing tap trung tot
    so voi muc thay doi HI trung binh tu dau den cuoi.
    """

    if len(z_by_bearing) < 2:
        return float("nan")

    starts = []
    ends = []

    for bearing_id, z in z_by_bearing.items():
        z1 = _as_scalar_z(z)

        if len(z1) < 2:
            raise ValueError(
                f"{bearing_id}: can >=2 measurement de tinh prognosability."
            )

        starts.append(z1[0])
        ends.append(z1[-1])

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)

    denominator = np.mean(np.abs(ends - starts))

    if denominator <= eps:
        return float("nan")

    return float(
        np.exp(-np.std(ends, ddof=0) / denominator)
    )


def mutual_information_with_target(
    z: np.ndarray,
    target: np.ndarray,
    random_state: int = 0,
) -> float:
    """Optional MutInf diagnostic when a true target/RUL is available.

    Khong tu suy dien RUL tu measurement_id. Caller phai cung cap target
    co y nghia vat ly/ground-truth rieng.
    """

    z = _as_scalar_z(z)
    target = np.asarray(target, dtype=np.float64)

    if target.ndim != 1 or len(target) != len(z):
        raise ValueError(
            "target phai 1D va cung do dai voi Z; "
            f"Z={len(z)}, target={target.shape}."
        )

    if not np.isfinite(target).all():
        raise ValueError("target chua NaN/Inf.")

    from sklearn.feature_selection import mutual_info_regression

    score = mutual_info_regression(
        z.reshape(-1, 1),
        target,
        random_state=random_state,
    )[0]

    return float(score)


# ============================================================
# NEGATIVE-GRADIENT DIAGNOSTICS
# ============================================================

def ng_violation_stats(
    z: np.ndarray,
    measurement_id: np.ndarray | None = None,
    consecutive_only: bool = True,
    tolerance: float = 0.0,
) -> dict[str, float | int]:
    """Danh gia truc tiep constraint dZ/dt <= 0.

    Violation khi:
        Z[i+1] - Z[i] > tolerance

    Plateau khong bi xem la violation khi tolerance >= 0.

    consecutive_only=True giong training: bo qua transition qua gap ID.
    """

    if tolerance < 0:
        raise ValueError("tolerance phai >= 0.")

    z = _as_scalar_z(z)
    t = _as_measurement_id(measurement_id, len(z))

    if len(z) < 2:
        return {
            "n_transitions": 0,
            "n_violations": 0,
            "violation_rate": float("nan"),
            "mean_violation": 0.0,
            "max_violation": 0.0,
        }

    dz = np.diff(z)
    mask = _transition_mask(
        t,
        consecutive_only=consecutive_only,
    )

    dz = dz[mask]

    if dz.size == 0:
        return {
            "n_transitions": 0,
            "n_violations": 0,
            "violation_rate": float("nan"),
            "mean_violation": 0.0,
            "max_violation": 0.0,
        }

    violation_amount = np.maximum(
        dz - tolerance,
        0.0,
    )
    is_violation = violation_amount > 0.0

    if np.any(is_violation):
        mean_violation = float(
            violation_amount[is_violation].mean()
        )
        max_violation = float(
            violation_amount[is_violation].max()
        )
    else:
        mean_violation = 0.0
        max_violation = 0.0

    return {
        "n_transitions": int(len(dz)),
        "n_violations": int(is_violation.sum()),
        "violation_rate": float(is_violation.mean()),
        "mean_violation": mean_violation,
        "max_violation": max_violation,
    }


def plateau_rate(
    z: np.ndarray,
    measurement_id: np.ndarray | None = None,
    consecutive_only: bool = True,
    atol: float = 1e-6,
) -> float:
    """Ty le transition co |delta Z| <= atol.

    Plateau khong bi NG loss phat. Metric nay chi de diagnostic xem Z co
    bi phang qua nhieu hay khong.
    """

    if atol < 0:
        raise ValueError("atol phai >= 0.")

    z = _as_scalar_z(z)
    t = _as_measurement_id(measurement_id, len(z))

    if len(z) < 2:
        return float("nan")

    dz = np.diff(z)
    mask = _transition_mask(
        t,
        consecutive_only=consecutive_only,
    )
    dz = dz[mask]

    if dz.size == 0:
        return float("nan")

    return float(
        np.mean(np.abs(dz) <= atol)
    )


# ============================================================
# OPERATING-CONDITION DEPENDENCE DIAGNOSTIC
# ============================================================

def oc_spearman_dependence(
    z: np.ndarray,
    W: np.ndarray,
    w_names: Sequence[str] = DEFAULT_W_NAMES,
) -> dict[str, dict[str, float]]:
    """Spearman association giua scalar Z va tung operating condition.

    Day la diagnostic cho muc tieu disentanglement, KHONG phai metric
    trong Appendix C cua Bajarunas va KHONG chung minh quan he nhan qua.
    Correlation cao co the den tu OC leakage, nhung cung co the do W va
    degradation cung thay doi theo thoi gian.
    """

    z = _as_scalar_z(z)
    W = np.asarray(W, dtype=np.float64)

    if W.ndim != 2 or len(W) != len(z):
        raise ValueError(
            "W phai co shape (N, w_dim) va cung N voi Z; "
            f"Z={len(z)}, W={W.shape}."
        )

    if not np.isfinite(W).all():
        raise ValueError("W chua NaN/Inf.")

    if len(w_names) != W.shape[1]:
        raise ValueError(
            "So w_names phai bang w_dim; "
            f"names={len(w_names)}, w_dim={W.shape[1]}."
        )

    result: dict[str, dict[str, float]] = {}

    for j, name in enumerate(w_names):
        w = W[:, j]

        if np.allclose(w, w[0]) or np.allclose(z, z[0]):
            rho = 0.0
        else:
            value = spearmanr(z, w).statistic
            rho = float(value) if np.isfinite(value) else 0.0

        result[str(name)] = {
            "rho": rho,
            "abs_rho": abs(rho),
        }

    return result


# ============================================================
# CONVENIENCE EVALUATION
# ============================================================

def evaluate_bearing_z(
    z: np.ndarray,
    measurement_id: np.ndarray | None = None,
    W: np.ndarray | None = None,
    w_names: Sequence[str] = DEFAULT_W_NAMES,
    ng_consecutive_only: bool = True,
    plateau_atol: float = 1e-6,
) -> dict[str, Any]:
    """Tong hop diagnostics cho mot bearing."""

    z1 = _as_scalar_z(z)
    t = _as_measurement_id(
        measurement_id,
        len(z1),
    )

    result: dict[str, Any] = {
        "n_measurements": int(len(z1)),
        "z_start": float(z1[0]),
        "z_end": float(z1[-1]),
        "z_mean": float(z1.mean()),
        "z_std": float(z1.std(ddof=0)),
        "z_range": float(np.ptp(z1)),
        "monotonicity": monotonicity_score(z1),
        "trendability": trendability_score(z1, t),
        "signed_trend_spearman": signed_trend_spearman(z1, t),
        "plateau_rate": plateau_rate(
            z1,
            t,
            consecutive_only=ng_consecutive_only,
            atol=plateau_atol,
        ),
        "ng": ng_violation_stats(
            z1,
            t,
            consecutive_only=ng_consecutive_only,
        ),
    }

    if W is not None:
        result["oc_dependence"] = (
            oc_spearman_dependence(
                z1,
                W,
                w_names=w_names,
            )
        )

    return result


def evaluate_fleet_z(
    z_by_bearing: Mapping[str, np.ndarray],
    measurement_id_by_bearing: Mapping[str, np.ndarray] | None = None,
    W_by_bearing: Mapping[str, np.ndarray] | None = None,
    w_names: Sequence[str] = DEFAULT_W_NAMES,
) -> dict[str, Any]:
    """Danh gia HI criteria cho nhieu bearing.

    Tra ve:
        - per_bearing: Mon / Tren / NG diagnostics / optional OC dependence
        - fleet: Prog va trung binh cac score per-bearing
    """

    if not z_by_bearing:
        raise ValueError("z_by_bearing rong.")

    per_bearing: dict[str, dict[str, Any]] = {}

    for bearing_id, z in z_by_bearing.items():
        ids = None
        if measurement_id_by_bearing is not None:
            if bearing_id not in measurement_id_by_bearing:
                raise KeyError(
                    f"Thieu measurement_id cho {bearing_id}."
                )
            ids = measurement_id_by_bearing[bearing_id]

        W = None
        if W_by_bearing is not None:
            if bearing_id not in W_by_bearing:
                raise KeyError(
                    f"Thieu W cho {bearing_id}."
                )
            W = W_by_bearing[bearing_id]

        per_bearing[bearing_id] = evaluate_bearing_z(
            z=z,
            measurement_id=ids,
            W=W,
            w_names=w_names,
        )

    monotonicities = np.asarray(
        [m["monotonicity"] for m in per_bearing.values()],
        dtype=np.float64,
    )
    trendabilities = np.asarray(
        [m["trendability"] for m in per_bearing.values()],
        dtype=np.float64,
    )
    violation_rates = np.asarray(
        [m["ng"]["violation_rate"] for m in per_bearing.values()],
        dtype=np.float64,
    )

    def _nanmean_or_nan(x: np.ndarray) -> float:
        if np.all(np.isnan(x)):
            return float("nan")
        return float(np.nanmean(x))

    fleet = {
        "prognosability": prognosability_score(
            z_by_bearing
        ),
        "mean_monotonicity": _nanmean_or_nan(
            monotonicities
        ),
        "mean_trendability": _nanmean_or_nan(
            trendabilities
        ),
        "mean_ng_violation_rate": _nanmean_or_nan(
            violation_rates
        ),
    }

    return {
        "per_bearing": per_bearing,
        "fleet": fleet,
    }
