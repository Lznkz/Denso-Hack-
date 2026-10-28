"""
loss.py

Loss cho Phase I Smart AE với latent scalar Z.

Main objective:

    L = L_MAE + lambda_ng * L_NG

Trong đó:
    - L_MAE: reconstruction loss trên X da normalize.
    - L_NG: negative-gradient constraint tren Z theo thu tu measurement.

Quy uoc hien tai bam theo Bajarunas:
    Z khong duoc tang theo thoi gian.

Vi du hop le:
    Z = [1.0, 0.8, 0.8, 0.3]

Vi du bi phat:
    Z = [1.0, 0.8, 0.9, 0.3]

Plateau (delta Z = 0) khong bi phat.
"""

import torch


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Mean Absolute Error reconstruction loss."""

    if x_hat.shape != x.shape:
        raise ValueError(
            "x_hat va x phai cung shape: "
            f"x_hat={tuple(x_hat.shape)}, "
            f"x={tuple(x.shape)}"
        )

    return torch.abs(x_hat - x).mean()


def negative_gradient_loss(
    z_seq: torch.Tensor,
    same_engine_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Negative-gradient constraint cho scalar latent Z.

    Parameters
    ----------
    z_seq:
        Shape (N, 1). Moi row la latent cua mot measurement, da duoc sap
        theo thu tu thoi gian trong bearing.

    same_engine_mask:
        Boolean tensor shape (N - 1,). True tai vi tri i neu cap
        (measurement_i, measurement_{i+1}) thuc su lien tiep va duoc phep
        tham gia temporal loss.

    Constraint
    ----------
    Theo Bajarunas:

        dZ/dt <= 0

    Voi cac measurement lien tiep va delta_t = 1:

        dZ ~= Z[i+1] - Z[i]

    Nen chi phat khi Z tang:

        L_NG = mean(relu(Z[i+1] - Z[i]))

    Plateau (Z[i+1] == Z[i]) co loss bang 0.
    """

    if z_seq.ndim != 2 or z_seq.shape[1] != 1:
        raise ValueError(
            "z_seq phai co shape (N, 1) vi Phase I co dinh Z=1, "
            f"got {tuple(z_seq.shape)}"
        )

    n = z_seq.shape[0]

    if same_engine_mask.ndim != 1:
        raise ValueError(
            "same_engine_mask phai la tensor 1D, "
            f"got {tuple(same_engine_mask.shape)}"
        )

    expected = max(n - 1, 0)
    if same_engine_mask.numel() != expected:
        raise ValueError(
            "same_engine_mask phai co N-1 phan tu: "
            f"expected={expected}, got={same_engine_mask.numel()}"
        )

    if n <= 1:
        return torch.zeros(
            (),
            device=z_seq.device,
            dtype=z_seq.dtype,
        )

    mask = same_engine_mask.to(
        device=z_seq.device,
        dtype=torch.bool,
    )

    if not torch.any(mask):
        return torch.zeros(
            (),
            device=z_seq.device,
            dtype=z_seq.dtype,
        )

    z = z_seq[:, 0]
    dz = z[1:] - z[:-1]

    # Bajarunas negative-gradient constraint:
    # desired: dz <= 0
    penalties = torch.relu(dz)

    return penalties[mask].mean()


def smart_ae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    z_seq: torch.Tensor,
    same_engine_mask: torch.Tensor,
    lambda_ng: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Smart AE objective:

        L = L_MAE + lambda_ng * L_NG

    Khong con baseline-distance loss, trendability, variance,
    covariance hay smoothness trong main training objective.
    """

    if lambda_ng < 0:
        raise ValueError(
            f"lambda_ng phai >= 0, got {lambda_ng}"
        )

    l_recon = reconstruction_loss(
        x_hat=x_hat,
        x=x,
    )

    l_ng = negative_gradient_loss(
        z_seq=z_seq,
        same_engine_mask=same_engine_mask,
    )

    total = (
        l_recon
        + float(lambda_ng) * l_ng
    )

    return {
        "total": total,
        "recon": l_recon,
        "ng": l_ng,
    }
