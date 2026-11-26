import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _NormAdapter(nn.Module):
    """Base class of the Bayesian normalization layers of ABNN.

    The subclasses normalize their input without affine transformation and call :meth:`_adapt`
    to apply the noisy scale and the shift.
    """

    alpha: float
    weight: nn.Parameter
    bias: nn.Parameter

    def _adapt(self, out: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """Scale the normalized input with a noisy weight and shift it.

        Args:
            out: Normalized input.
            weight: Scale, broadcastable to the normalized input.
            bias: Shift, broadcastable to the normalized input.

        Returns:
            Tensor: The output of the layer.
        """
        return weight * out * (torch.randn_like(out) * self.alpha + 1) + bias


class BatchNormAdapter(_NormAdapter):
    def __init__(
        self,
        num_features: int,
        alpha: float = 0.1,
        momentum: float | None = 0.1,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        r"""Bayesian normalization layer of ABNN based on batch normalization.

        Batch normalization whose scale is perturbed by a Gaussian noise, both during training
        and evaluation: :math:`y = \gamma \, \text{BN}(x) (1 + \alpha \epsilon) + \beta` with
        :math:`\epsilon \sim \mathcal{N}(0, 1)`.

        This layer accepts inputs of shape :math:`(B, C)` or :math:`(B, C, *)`. See
        :class:`BatchNormAdapter1d`, :class:`BatchNormAdapter2d`, and :class:`BatchNormAdapter3d`
        for the variants checking the number of dimensions of their input.

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
        self.num_features = num_features
        self.alpha = alpha
        self.momentum = momentum
        self.eps = eps

    def _check_input_dim(self, x: Tensor) -> None:
        if x.dim() < 2:
            raise ValueError(f"expected at least 2D input (got {x.dim()}D input)")

    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)

        # Mirror nn.BatchNorm: ``momentum=None`` means a cumulative moving average.
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
        # The channels are on the second dimension, whatever the number of dimensions
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return self._adapt(out, self.weight.view(shape), self.bias.view(shape))


class BatchNormAdapter1d(BatchNormAdapter):
    r"""Bayesian normalization layer of ABNN for 2D or 3D inputs.

    See :class:`BatchNormAdapter` for the details and the arguments. The input is of shape
    :math:`(B, C)` or :math:`(B, C, L)`.
    """

    def _check_input_dim(self, x: Tensor) -> None:
        if x.dim() not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {x.dim()}D input)")


class BatchNormAdapter2d(BatchNormAdapter):
    r"""Bayesian normalization layer of ABNN for 4D inputs.

    See :class:`BatchNormAdapter` for the details and the arguments. The input is of shape
    :math:`(B, C, H, W)`.
    """

    def _check_input_dim(self, x: Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"expected 4D input (got {x.dim()}D input)")


class BatchNormAdapter3d(BatchNormAdapter):
    r"""Bayesian normalization layer of ABNN for 5D inputs.

    See :class:`BatchNormAdapter` for the details and the arguments. The input is of shape
    :math:`(B, C, D, H, W)`.
    """

    def _check_input_dim(self, x: Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"expected 5D input (got {x.dim()}D input)")


class LayerNormAdapter(_NormAdapter):
    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        alpha: float = 0.1,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        r"""Bayesian normalization layer of ABNN based on layer normalization.

        Layer normalization whose scale is perturbed by a Gaussian noise, both during training
        and evaluation: :math:`y = \gamma \, \text{LN}(x) (1 + \alpha \epsilon) + \beta` with
        :math:`\epsilon \sim \mathcal{N}(0, 1)`.

        Args:
            normalized_shape: Shape of the last dimensions of the input, over which the
                normalization is computed.
            alpha: Scale of the noise. Defaults to ``0.1``.
            eps: Value added to the denominator for numerical stability. Defaults to ``1e-5``.
            device: Device of the parameters. Defaults to ``None``.
            dtype: Data type of the parameters. Defaults to ``None``.

        References:
            [1] `Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from
            Pre-trained Models. In CVPR, 2024 <https://arxiv.org/abs/2312.15297>`_.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(
            torch.ones(self.normalized_shape, device=device, dtype=dtype),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.zeros(self.normalized_shape, device=device, dtype=dtype),
            requires_grad=True,
        )
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        out = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        # The parameters have the shape of the last dimensions: no reshaping is needed
        return self._adapt(out, self.weight, self.bias)
