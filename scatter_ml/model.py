"""
model.py -- 2.5D U-Net for sinogram scatter prediction.

2.5D means: predict ONE (angular, radial) plane from a WINDOW of k neighbouring
planes along zbar at the same ring difference d (fed as input channels), plus d
itself as a constant extra channel. This captures axial correlation at ~2D cost;
scatter is smooth and low-frequency, so near-neighbour context carries most of
the signal. 3D convolution is the ablation against this, not the baseline.

Output passes through softplus so predictions are non-negative -- required by the
Poisson NLL and physically correct for a count rate. The network predicts scatter
in the per-sample SCALED space (dataset.py normalises by the input window mean);
predict.py multiplies the scale back.

Input planes are (A=160, R=193). R is padded up to a multiple of 16 for the
U-Net's down/up-sampling and the output is cropped back, so the plane size never
constrains the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to(x, mult=16):
    """Pad last two dims up to a multiple of `mult`; return (x, (ph, pw))."""
    h, w = x.shape[-2:]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))     # pad right/bottom only (keeps origin)
    return x, (h, w)


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2p5D(nn.Module):
    """Small 3-level U-Net. in_ch = window_k + 1 (the +1 is the d channel)."""

    def __init__(self, window_k: int = 5, base: int = 32):
        super().__init__()
        cin = window_k + 1
        self.d1 = _DoubleConv(cin, base)
        self.d2 = _DoubleConv(base, base * 2)
        self.d3 = _DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = _DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, window, d_norm):
        B, k, A, R = window.shape                 # R = 193
        d_chan = d_norm.view(B, 1, 1, 1).expand(B, 1, A, R)
        x = torch.cat([window, d_chan], dim=1)    # (B, k+1, 160, 193)
        x = x[..., :192]                          # -> 160x192 = 16*10 x 16*12

        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bott(self.pool(c3))
        x = self.u3(torch.cat([self.up3(b), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        x = self.out(x)
        x = F.pad(x, (0, 1))                       # restore column -> 193, copy-safe
        return F.softplus(x)