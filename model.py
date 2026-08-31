"""Best-performing dual-branch EEG classifier from the branch-fusion ablation.

Winner of the ablation study: ``A1_fusion_wsum_power``
    top-5 seed test accuracy 84.85% / all-run 83.72%
    (vs. the previous SimAM per-scale-concat baseline ~82-83%).

This is the Stage-7 dual-branch model with a single change to the multi-scale
summary: instead of concatenating the four per-scale log-powers, each scale is
independently attended (SimAM) and log-power pooled, then the four log-powers
are combined by a **learnable weighted sum in the log-power domain**.

Weighting in the log-power domain matters: log-power turns a conv rescaling
into an additive shift, so the per-scale weights are genuinely learned and
cannot be absorbed back into the convolution weights (which is what makes a
feature-domain weighted sum ineffective).

Structure (input (B, 64, T) -> logits (B, num_classes)):
    temporal branch : 1x1 electrode mix -> depthwise multi-scale conv
                      -> per-scale [SimAM -> log-power] -> weighted sum  -> (B, 64)
    spatial  branch : full multi-scale conv
                      -> per-scale [SimAM -> log-power] -> weighted sum  -> (B, 64)
    concat (B, 128) -> Linear -> logits
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Parameter-free SimAM applied along the time dimension."""

    def __init__(self, lambda_: float = 1e-3):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        n = x.shape[1] - 1
        if n <= 0:
            return x

        squared_deviation = (x - x.mean(dim=1, keepdim=True)).pow(2)
        variance = squared_deviation.sum(dim=1, keepdim=True) / n
        importance = squared_deviation / (4 * (variance + self.lambda_)) + 0.5
        return x * torch.sigmoid(importance)


class PowerWeightedBranch(nn.Module):
    """Multi-scale branch with a learnable weighted sum of per-scale log-powers."""

    def __init__(
        self,
        channels: int = 64,
        branch_type: str = "temporal",
        kernel_size: int = 3,
        dilations=(1, 2, 3, 4),
        simam_lambda: float = 1e-3,
    ):
        super().__init__()
        if branch_type not in ("temporal", "spatial"):
            raise ValueError("branch_type must be 'temporal' or 'spatial'.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must contain positive integers.")

        # Temporal: channel-wise/depthwise temporal convolution.
        # Spatial: full convolution that mixes all EEG channels.
        groups = channels if branch_type == "temporal" else 1
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                    groups=groups,
                    bias=True,
                )
                for dilation in dilations
            ]
        )
        self.simam = SimAM(lambda_=simam_lambda)
        # Per-(scale, channel) fusion logits; zero logits -> uniform average.
        self.scale_logits = nn.Parameter(torch.zeros(len(dilations), channels))
        self.output_dim = channels

    def _log_power(self, feature: torch.Tensor, eps: float) -> torch.Tensor:
        # SimAM expects (B, T, C), while Conv1d returns (B, C, T).
        feature = self.simam(feature.transpose(1, 2)).transpose(1, 2)
        return torch.log(torch.mean(feature.pow(2), dim=2) + eps)

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        powers = [self._log_power(conv(x), eps) for conv in self.convs]  # each (B, C)
        stacked = torch.stack(powers, dim=1)                    # (B, S, C)
        weights = torch.softmax(self.scale_logits, dim=0)[None]  # (1, S, C)
        return (stacked * weights).sum(dim=1)                    # (B, C)


class DualBranchSimAMNet(nn.Module):
    """Dual-branch model with power-domain weighted-sum scale fusion (ablation winner)."""

    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 5,
        kernel_size: int = 3,
        dilations=(1, 2, 3, 4),
        simam_lambda: float = 1e-3,
    ):
        super().__init__()

        # Electrode mixing before channel-wise temporal filtering.
        self.temporal_spatial = nn.Conv1d(
            input_dim, input_dim, kernel_size=1, bias=True
        )

        self.temporal = PowerWeightedBranch(
            channels=input_dim,
            branch_type="temporal",
            kernel_size=kernel_size,
            dilations=dilations,
            simam_lambda=simam_lambda,
        )
        self.spatial = PowerWeightedBranch(
            channels=input_dim,
            branch_type="spatial",
            kernel_size=kernel_size,
            dilations=dilations,
            simam_lambda=simam_lambda,
        )
        self.classifier = nn.Linear(
            self.temporal.output_dim + self.spatial.output_dim,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_feature = self.temporal(self.temporal_spatial(x))
        spatial_feature = self.spatial(x)
        feature = torch.cat([temporal_feature, spatial_feature], dim=1)
        return self.classifier(feature)


if __name__ == "__main__":
    model = DualBranchSimAMNet()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    sample = torch.randn(2, 64, 795)
    print(model(sample).shape)
    print(f"Trainable parameters: {trainable_parameters:,}")
