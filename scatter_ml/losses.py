"""
losses.py -- L2 (default) and Poisson NLL.

Both are Bregman divergences, so both have the SAME minimizer over predictions:
the conditional mean E[label | input] (Banerjee et al. 2005). They differ only in
how they weight errors -- L2 penalizes absolute error, Poisson NLL penalizes
relative error (low-count bins matter more) -- which is exactly what the
loss ablation measures. Neither rescues correlated labels; that is the job of the
independent split in dataset.py.

Predictions are in a per-sample SCALED count space (see dataset.py), so they are
non-negative reals, not integers. The Poisson NLL minimizer proof uses only
linearity in the target, so pred=target is still the minimizer even though the
scaled target is not integer-valued -- what is lost is only the likelihood
interpretation, not the regression target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def poisson_nll(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    """ybar - y*log(ybar), mean-reduced. pred must be >= 0 (softplus output)."""
    pred = pred.clamp_min(eps)
    return (pred - target * torch.log(pred)).mean()


LOSSES = {"mse": mse, "poisson": poisson_nll}