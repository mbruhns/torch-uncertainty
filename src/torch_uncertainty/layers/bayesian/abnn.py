import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BatchNormAdapter2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        alpha: float = 0.1,
        momentum: float | None = 0.1,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        r"""Bayesian normalization layer of ABNN for 2D inputs.

        Batch normalization whose scale is perturbed by a Gaussian noise, both during training
        and evaluation: :math:`y = \gamma \, \text{BN}(x) (1 + \alpha \epsilon) + \beta` with
        :math:`\epsilon \sim \mathcal{N}(0, 1)`.

        Args:
            num_features: Number of channels of the input.
            alpha: Scale of the noise. Defaults to ``0.1``.
            momentum: Momentum of the running statistics. ``None`` corresponds to a cumulative
                moving average. Defaults to ``0.1``.
            eps: Value added to the denominator for numerical stability. Defaults to ``1e-5``.
            device: Device of the parameters and buffers. Defaults to ``None``.
            dtype: Data type of the parameters and buffers. Defaults to ``None``.

        References:
            [1] `Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from
            Pre-trained Models. In CVPR, 2024 <https://arxiv.org/abs/2312.15297>`_.
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_features, device=device, dtype=dtype),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.zeros(num_features, device=device, dtype=dtype),
            requires_grad=True,
        )

        self.register_buffer(
            "running_mean",
            torch.zeros(num_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "running_var",
            torch.ones(num_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long, device=device),
        )
        self.alpha = alpha
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Mirror nn.BatchNorm2d: ``momentum=None`` means a cumulative moving average.
        momentum = 0.0 if self.momentum is None else self.momentum
        if self.training:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                momentum = 1.0 / float(self.num_batches_tracked)

        out = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training,
            momentum,
            self.eps,
        )
        return self.weight.unsqueeze(-1).unsqueeze(-1) * out * (
            torch.randn_like(x) * self.alpha + 1
        ) + self.bias.unsqueeze(-1).unsqueeze(-1)
