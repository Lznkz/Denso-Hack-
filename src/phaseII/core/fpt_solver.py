"""
fpt_solver.py -- KHỐI 6: NGƯỠNG HỎNG HÓC & TÍNH TOÁN PHÂN VỊ QUA WALD CDF

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ:
  - Xóa bỏ hoàn toàn Gumbel EVT và ISO 281 IQR margin.
  - Thay bằng compute_fleet_failure_threshold(): median(Z_end của train bearings).
  - Giữ nguyên solve_fpt_quantiles() với Wald CDF brentq (giải tích, fail-closed).

Nguyên lý tính ngưỡng mới:
  Offline (chạy 1 lần trên train set):
    Z_fail_abs = median({Z_end^(u) : u ∈ train_bearings})
  Đây là ngưỡng hỏng trên thang Z gốc của Phase I.

  Tại thời điểm onset (t_onset):
    H_onset = Z_base - Z_t_onset  (trong H-space)
    Z_onset = Z_t_onset
    D_failure_H = Z_base - Z_fail_abs  (ngưỡng trong H-space)
    rel_val = D_failure_H - H_onset = Z_onset - Z_fail_abs
    Điều kiện: rel_val > 0 (vòng bi chưa hỏng hoàn toàn tại onset)

  Tại mỗi chu kỳ suy thoái:
    delta_d = rel_val - Y_acc  > 0 => gọi solve_fpt_quantiles()

Wald CDF giải tích:
  nu = delta_d / mu_tau, lambda = delta_d^2 / sigma_B^2
  F_Wald(s) = Phi(sqrt(lambda/s)*(s/nu - 1)) + exp(2*lambda/nu)*Phi(-sqrt(lambda/s)*(s/nu+1))
  RUL_p = (tau^gamma + DeltaLambda_p)^(1/gamma) - tau

Fail-closed: không fallback, trả FPTResult với status != OK khi solver thất bại.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from scipy.optimize import brentq

from src.phaseII.core.schema import (
    FPTResult,
    FPTSolverStatus,
    SolverDiagnostics,
)


def compute_fleet_failure_threshold(
    train_z_end_values: Sequence[float],
) -> float:
    """Tính Z_failure_abs = median(Z_end của các bearing train).

    Đây là ngưỡng hỏng tuyệt đối trên thang đo Z gốc của Phase I
    (thang Z âm-dương, không phải H-space).

    Parameters
    ----------
    train_z_end_values : Sequence[float]
        Giá trị Z tại measurement cuối cùng (Z[-1, 0]) của mỗi train bearing.

    Returns
    -------
    z_failure_abs : float
        Median của các Z_end. Vòng bi được coi là hỏng khi Z_t ≤ z_failure_abs.
    """
    vals = np.asarray(train_z_end_values, dtype=np.float64)
    if len(vals) == 0:
        raise ValueError("train_z_end_values không được rỗng.")
    return float(np.median(vals))


def wald_cdf(s: float, nu: float, lambd: float) -> float:
    """Hàm phân phối tích lũy Wald (Inverse Gaussian).

    F(s; nu, lambda) = Phi(sqrt(lambda/s) * (s/nu - 1))
                     + exp(2*lambda/nu) * Phi(-sqrt(lambda/s) * (s/nu + 1))
    """
    if s <= 0:
        return 0.0
    sqrt_ls = np.sqrt(lambd / s)
    z1 = sqrt_ls * (s / nu - 1.0)
    z2 = -sqrt_ls * (s / nu + 1.0)

    log_correction = 2.0 * lambd / nu
    from scipy.special import ndtr, log_ndtr
    term1 = ndtr(z1)
    with np.errstate(over='ignore', under='ignore'):
        log_term2 = log_correction + float(log_ndtr(z2))
        term2 = float(np.exp(log_term2)) if log_term2 < 700 else 0.0
    return float(term1 + term2)


def solve_fpt_quantiles(
    delta_d: float,
    mu_tau: float,
    sigma_b_tau: float,
    tau: int,
    gamma: float = 1.75,
    quantiles: tuple[float, ...] = (0.05, 0.50, 0.95),
) -> FPTResult:
    """Giải các nghiệm phân vị RUL giải tích từ Wald CDF.

    Fail-closed: trả FPTResult với status != OK thay vì fallback.

    Parameters
    ----------
    delta_d : float
        Khoảng cách còn lại tới ngưỡng hỏng (RELATIVE_TO_ONSET). Phải > 0.
    mu_tau : float
        Kỳ vọng drift hiện tại. Phải > 0.
    sigma_b_tau : float
        Hệ số khuếch tán. Phải > 0.
    tau : int
        Bước thời gian suy thoái (>= 1).
    gamma : float
        Số mũ Paris Law.
    quantiles : tuple[float, ...]
        Các phân vị cần giải.

    Returns
    -------
    FPTResult
    """
    # Validate đầu vào
    if not (np.isfinite(delta_d) and delta_d > 0):
        diag = SolverDiagnostics(nu=0, lambd=0, p=0, error_message="delta_d <= 0")
        return FPTResult(q05=None, q50=None, q95=None,
                         solver_status=FPTSolverStatus.BRACKET_FAILED, diagnostics=diag)

    if not (np.isfinite(mu_tau) and mu_tau > 0):
        diag = SolverDiagnostics(nu=0, lambd=0, p=0, error_message="mu_tau <= 0")
        return FPTResult(q05=None, q50=None, q95=None,
                         solver_status=FPTSolverStatus.BRACKET_FAILED, diagnostics=diag)

    sigma_b = max(float(sigma_b_tau), 1e-8)

    # Tham số Wald CDF
    nu = delta_d / mu_tau
    lambd = (delta_d ** 2) / (sigma_b ** 2)

    # Bracket search cho brentq: tìm khoảng [low_dl, high_dl] bao tất cả quantiles
    lambda_tau = float(tau ** gamma)
    low_dl = 1e-8
    high_dl = max(nu * 50, 1e3)
    for _ in range(40):
        if wald_cdf(high_dl, nu, lambd) >= max(quantiles):
            break
        high_dl *= 4.0
    else:
        diag = SolverDiagnostics(nu=nu, lambd=lambd, p=max(quantiles),
                                  bracket_high=high_dl,
                                  error_message="Không tìm được khoảng bracket")
        return FPTResult(q05=None, q50=None, q95=None,
                         solver_status=FPTSolverStatus.BRACKET_FAILED, diagnostics=diag)

    # Giải từng phân vị
    rul_list: list[float | None] = []
    for p in quantiles:
        try:
            dl_p = brentq(lambda s: wald_cdf(s, nu, lambd) - p, low_dl, high_dl,
                          xtol=1e-6, maxiter=200)
            # Quy đổi DeltaLambda -> RUL thực (khi gamma=1.0: rul_p = dl_p)
            if abs(gamma - 1.0) < 1e-6:
                rul_p = dl_p
            else:
                rul_p = (lambda_tau + dl_p) ** (1.0 / gamma) - tau
            rul_list.append(max(float(rul_p), 0.0))
        except Exception as e:
            diag = SolverDiagnostics(nu=nu, lambd=lambd, p=p,
                                      error_message=str(e))
            return FPTResult(q05=None, q50=None, q95=None,
                             solver_status=FPTSolverStatus.SOLVER_FAILED, diagnostics=diag)

    q05, q50, q95 = rul_list[0], rul_list[1], rul_list[2]

    # Kiểm tra tính đơn điệu phân vị
    if not (q05 <= q50 <= q95):
        # Float precision issue — clamp
        q05 = min(q05, q50)
        q95 = max(q50, q95)

    return FPTResult(q05=q05, q50=q50, q95=q95, solver_status=FPTSolverStatus.OK)
