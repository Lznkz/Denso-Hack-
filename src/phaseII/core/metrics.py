"""
metrics.py -- ĐO LƯỜNG VÀ ĐÁNH GIÁ CHẤT LƯỢNG DỰ BÁO RUL CHO PHASE II

Hỗ trợ các chỉ số chuẩn PHM (Prognostics and Health Management):
  1. RMSE & MAE trên các chu kỳ đã suy thoái (tau >= 1).
  2. Coverage Probability: Tỷ lệ chu kỳ RUL thật nằm trong dải [RUL_0.05, RUL_0.95] (kỳ vọng >= 90%).
  3. Asymmetric PHM Penalty: Phạt nặng dự đoán trễ (overestimation) hơn dự đoán sớm.
  4. C-index: Đo mức độ xếp hạng rủi ro tương đối giữa các vòng bi.
"""

from __future__ import annotations
import numpy as np


def compute_rul_errors(
    predicted_ruls: list[float],
    true_ruls: list[float],
) -> dict[str, float]:
    """
    Tính RMSE, MAE và sai số trung bình giữa RUL dự báo và RUL thực tế.
    """
    preds = np.asarray(predicted_ruls, dtype=np.float64)
    trues = np.asarray(true_ruls, dtype=np.float64)

    if len(preds) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan")}

    diff = preds - trues
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    return {"rmse": rmse, "mae": mae, "bias": bias}


def compute_interval_coverage(
    lower_ruls: list[float],
    upper_ruls: list[float],
    true_ruls: list[float],
) -> float:
    """
    Tính tỷ lệ phần trăm (0 - 100%) mà RUL thực tế rơi vào khoảng tin cậy [RUL_lower, RUL_upper].
    """
    lows = np.asarray(lower_ruls, dtype=np.float64)
    highs = np.asarray(upper_ruls, dtype=np.float64)
    trues = np.asarray(true_ruls, dtype=np.float64)

    if len(trues) == 0:
        return float("nan")

    covered = (trues >= lows) & (trues <= highs)
    return float(np.mean(covered) * 100.0)


def asymmetric_phm_score(
    predicted_ruls: list[float],
    true_ruls: list[float],
    a1: float = 13.0,
    a2: float = 10.0,
) -> float:
    """
    Ham cham diem phi doi xung chuan PHM Data Challenge.

    d = pred - true
      d < 0: underestimation / early prediction (du doan som, conservative):
             penalty = exp(-d / a1) - 1   (a1 lon hon -> exponential doc nhe hon)
      d > 0: overestimation / late prediction (du doan tre - nguy hiem trong san xuat):
             penalty = exp(d / a2) - 1    (a2 nho hon -> exponential doc doc hon -> phat manh hon)

    Mac dinh a1=13, a2=10: voi cung |d|, overestimation bi phat nang hon underestimation.
    Luu y: a1 > a2, nguoc lai voi cach ki hieu trong mot so tai lieu dung convention nguoc.
    """
    preds = np.asarray(predicted_ruls, dtype=np.float64)
    trues = np.asarray(true_ruls, dtype=np.float64)

    if len(preds) == 0:
        return float("nan")

    diff = preds - trues
    scores = np.where(diff < 0, np.exp(-diff / a1) - 1.0, np.exp(diff / a2) - 1.0)
    return float(np.sum(scores))


def concordance_index(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
    higher_is_risk: bool = True,
) -> float:
    """
    Concordance Index (C-index) danh gia xep hang rui ro tuong doi giua cac vong bi.

    Parameters
    ----------
    event_times : np.ndarray
        Thoi diem hong thuc te (survival time).
    predicted_risk : np.ndarray
        Score du bao. Neu higher_is_risk=True (mac dinh), score cao = rui ro cao =
        survival time ngan hon. Ham tu dao dau truoc khi truyen cho lifelines.
    event_observed : np.ndarray
        1 neu su kien duoc quan sat.
    higher_is_risk : bool
        True (mac dinh): predicted_risk cao = rui ro cao -> can dao dau.
        False: predicted_risk cao = survival dai hon -> dung truc tiep.

    Notes
    -----
    lifelines.concordance_index(event_times, predicted_scores) ky vong predicted_scores
    cao tuong ung survival time dai. Neu truyen thang predicted_risk (cao = rui ro cao)
    ma khong dao dau, C-index se cho ket qua nguoc (duoi 0.5 thuc ra la model tot).
    """
    from lifelines.utils import concordance_index as _ci
    scores = -np.asarray(predicted_risk, dtype=np.float64) if higher_is_risk else np.asarray(predicted_risk, dtype=np.float64)
    return float(_ci(event_times, scores, event_observed))
