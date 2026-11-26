import copy
import logging
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.layers.bayesian.abnn import (
    BatchNormAdapter,
    BatchNormAdapter1d,
    BatchNormAdapter2d,
    BatchNormAdapter3d,
    LayerNormAdapter,
    _NormAdapter,
)
from torch_uncertainty.methods.deep_ensembles import _DeepEnsembles
from torch_uncertainty.optim_recipes import optim_abnn
from torch_uncertainty.utils import TUTrainer

from .abstract import PostProcessing


class ABNN(PostProcessing):
    def __init__(
        self,
        num_classes: int,
        random_prior: float,
        alpha: float,
        num_models: int,
        num_samples: int,
        base_lr: float,
        device: torch.device | str,
        max_epochs: int = 5,
        use_original_model: bool = False,
        precision: str = "32",
        model: nn.Module | None = None,
        task: Literal["classification", "segmentation"] = "classification",
    ) -> None:
        r"""Adaptable Bayesian Neural Network (ABNN) post-processing.

        Replaces the normalization layers of a pre-trained model with their stochastic
        counterparts (:class:`~torch_uncertainty.layers.bayesian.BatchNormAdapter` and
        :class:`~torch_uncertainty.layers.bayesian.LayerNormAdapter`) and finetunes
        :attr:`num_models` copies of the model, each with its own random prior on the classes.
        Only the weights and biases of these layers are finetuned: the other parameters keep
        their pre-trained values. At test time, each finetuned model is sampled
        :attr:`num_samples` times.

        Args:
            num_classes: Number of classes of the inner model.
            random_prior: Random prior value to specialize estimators on certain classes.
            alpha: Alpha value for ABNN to control the diversity of the predictions.
            num_models: Number of stochastic models.
            num_samples: Number of samples per model.
            base_lr: Base learning rate.
            device: Device used for the finetuning. Either a torch device (``"cuda:1"``,
                ``torch.device("cpu")``, ...) or a Lightning accelerator name (``"gpu"``,
                ``"auto"``). The finetuning always runs on a single device.
            max_epochs: Number of training epochs. Defaults to ``5``.
            use_original_model: Add the original model to the ensemble, with the same weight as
                each finetuned model. The original model is deterministic: it is not finetuned
                and keeps its batch normalization layers. Defaults to ``False``, as in the
                original paper.
            precision: Machine precision for training & eval. Defaults to ``"32"``.
            model: Model to use. Defaults to ``None``.
            task: Task of the model, either ``"classification"`` or ``"segmentation"``. Defaults
                to ``"classification"``.

        Note:
            The forward pass returns the logits of all the estimators, with shape
            :math:`(N \times B, C)`, or :math:`(N \times B, C, H, W)` for segmentation, where
            :math:`N` is the number of estimators, i.e., the number of models (including the
            original one if :attr:`use_original_model`) times :attr:`num_samples`. The prediction
            of the original model is computed once and repeated :attr:`num_samples` times.

        Note:
            The models of the ensemble, including the original one, are copies made by :meth:`fit`
            and put in evaluation mode. The model given to the post-processing method is left
            untouched and its later changes do not affect the ensemble.

        Note:
            The supported normalization layers are ``BatchNorm1d``, ``BatchNorm2d``,
            ``BatchNorm3d``, and ``SyncBatchNorm`` tracking their running statistics, and
            ``LayerNorm``. Their subclasses and the other normalization layers are left
            unchanged and reported by a warning during the fit.

        Note:
            In evaluation mode, ``torch.nn.TransformerEncoderLayer`` may use a fast path that
            does not call its normalization layers and would ignore their noise. This fast path
            is disabled for the converted layers. As a consequence, a ``TransformerEncoder`` no
            longer sets its outputs to zero at the padded positions.

        References:
            [1] `Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from
            Pre-trained Models. In CVPR, 2024 <https://arxiv.org/abs/2312.15297>`_.
        """
        super().__init__(model)
        _abnn_checks(
            num_classes=num_classes,
            random_prior=random_prior,
            alpha=alpha,
            max_epochs=max_epochs,
            num_models=num_models,
            num_samples=num_samples,
            base_lr=base_lr,
            task=task,
        )
        self.task = task
        self.num_classes = num_classes
        self.alpha = alpha
        self.base_lr = base_lr
        self.num_models = num_models
        self.num_samples = num_samples
        self.use_original_model = use_original_model
        self.max_epochs = max_epochs

        self.precision = precision
        self.device = device

        self.final_model = None
        self.original_model = None

        # Build random prior
        num_rp_classes = int(num_classes**0.5)
        self.weights = []
        for _ in range(num_models):
            weight = torch.ones([num_classes])
            weight[torch.randperm(num_classes)[:num_rp_classes]] += random_prior - 1
            self.weights.append(weight)

    def fit(self, dataloader: DataLoader) -> None:
        """Finetune the stochastic models on the dataloader.

        Args:
            dataloader: Dataloader of the finetuning set.

        Raises:
            ValueError: If the model is not set or does not contain any supported normalization
                layer.
        """
        # Imported here to avoid a circular import with the routines
        from torch_uncertainty.routines import ClassificationRoutine, SegmentationRoutine

        routine = SegmentationRoutine if self.task == "segmentation" else ClassificationRoutine

        if self.model is None:
            raise ValueError("Model must be set before fitting.")

        source_model = copy.deepcopy(self.model)
        unsupported = _replace_norm_layers(source_model, self.alpha)
        if not any(isinstance(module, _NormAdapter) for module in source_model.modules()):
            raise ValueError(
                "ABNN requires a model with batch normalization layers tracking their running "
                "statistics or LayerNorm layers, but none was found."
            )
        if unsupported:
            logging.warning(
                "ABNN does not support the following normalization layers, which are left "
                "unchanged: %s.",
                ", ".join(sorted(set(unsupported))),
            )

        models = [copy.deepcopy(source_model) for _ in range(self.num_models)]
        for mod in models:
            _freeze_all_but_adapters(mod)

        baselines = [
            routine(
                num_classes=self.num_classes,
                model=mod,
                # The class weights are a buffer of the loss: Lightning moves them to the device.
                loss=nn.CrossEntropyLoss(weight=self.weights[i]),
                optim_recipe=optim_abnn(mod, lr=self.base_lr),  # type: ignore[arg-type]
            )
            for i, mod in enumerate(models)
        ]

        accelerator, devices = _parse_device(self.device)
        for baseline in baselines:
            trainer = TUTrainer(
                max_epochs=self.max_epochs,
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=False,
                precision=self.precision,  # type: ignore[arg-type]
                enable_checkpointing=False,
                logger=False,
                enable_model_summary=False,
            )
            # Lightning keeps the current mode of the modules: finetune in training mode even if
            # the original model was in evaluation mode.
            baseline.train()
            trainer.fit(model=baseline, train_dataloaders=dataloader)

        # Each model is stored once: the samples only differ by the noise of the adapters
        final_models = []
        for baseline in baselines:
            model = copy.deepcopy(source_model)
            model.load_state_dict(baseline.model.state_dict())
            final_models.append(model)
        # The ensemble is only meant for inference: its models must use their running statistics,
        # whatever the mode of the original model.
        self.final_model = _DeepEnsembles(final_models).eval()
        # Snapshot of the original model: the ensemble does not depend on its later changes
        self.original_model = copy.deepcopy(self.model).eval() if self.use_original_model else None
        self.trained = True

    def forward(self, inputs: Tensor) -> Tensor:
        if self.model is None:
            raise ValueError("Model must be set before calling forward.")
        if self.final_model is None:
            return self.model(inputs)

        preds = []
        if self.original_model is not None:
            # The original model is deterministic: compute its prediction once and repeat it to
            # give it the same weight as each finetuned model.
            original_pred = self.original_model(inputs)
            preds.append(original_pred.repeat(self.num_samples, *[1] * (original_pred.ndim - 1)))
        preds.extend(self.final_model(inputs) for _ in range(self.num_samples))
        return torch.cat(preds, dim=0)


def _abnn_checks(
    num_classes,
    random_prior,
    alpha,
    max_epochs,
    num_models,
    num_samples,
    base_lr,
    task,
) -> None:
    if task not in ("classification", "segmentation"):
        raise ValueError(f"task must be either 'classification' or 'segmentation'. Got {task}.")
    if random_prior < 0:
        raise ValueError(f"random_prior must be non-negative. Got {random_prior}.")
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be greater than 0. Got {max_epochs}.")
    if num_models < 1:
        raise ValueError(f"num_models must be greater than 0. Got {num_models}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be greater than 0. Got {num_samples}.")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative. Got {alpha}.")
    if base_lr < 0:
        raise ValueError(f"base_lr must be non-negative. Got {base_lr}.")
    if num_classes < 1:
        raise ValueError(f"num_classes must be greater than 0. Got {num_classes}.")


def _freeze_all_but_adapters(model: nn.Module) -> None:
    """Freeze all the parameters of the model except the weights and biases of the ABNN layers.

    Args:
        model: Model whose normalization layers have been replaced with ABNN layers.
    """
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, _NormAdapter):
            module.requires_grad_(True)


def _parse_device(device: torch.device | str) -> tuple[str, list[int] | int]:
    """Convert a device into the ``accelerator`` and ``devices`` arguments of a trainer.

    Args:
        device: A torch device or a Lightning accelerator name.

    Returns:
        The accelerator name and the device(s) to finetune on.
    """
    if isinstance(device, str) and device in ("auto", "gpu"):
        return device, 1
    device = torch.device(device)
    return device.type, 1 if device.index is None else [device.index]


# Exact types only: the subclasses of the normalization layers may change the forward pass, for
# instance to normalize another dimension or to apply an activation.
_BATCH_NORM_ADAPTERS: dict[type[nn.Module], type[BatchNormAdapter]] = {
    nn.BatchNorm1d: BatchNormAdapter1d,
    nn.BatchNorm2d: BatchNormAdapter2d,
    nn.BatchNorm3d: BatchNormAdapter3d,
    # The finetuning runs on a single device: no need to synchronize the statistics
    nn.SyncBatchNorm: BatchNormAdapter,
}

# TODO: support the remaining normalization layers. Until then, they are left unchanged (and
# reported by a warning during the fit):
#   - nn.GroupNorm and nn.InstanceNorm1d/2d/3d (instance normalization is part of the original
#     paper): adapters similar to LayerNormAdapter, based on F.group_norm and F.instance_norm.
#     The instance normalization may track running statistics and is not affine by default.
#   - batch normalization layers with track_running_stats=False: BatchNormAdapter needs to
#     work without running statistics, always normalizing with the statistics of the batch.
#   - subclasses changing the forward pass, which need a dedicated conversion: LayerNorm2d of
#     timm and torchvision (normalization over the channels of NCHW inputs), BatchNormAct2d of
#     timm (batch normalization followed by an activation), and ChannelLayerNorm,
#     PackedLayerNorm, and the MCBatchNorm layers of this library.
#   - normalization layers that do not derive from the PyTorch ones (FrozenBatchNorm2d of
#     torchvision, the FilterResponseNorm layers of this library, custom implementations) and
#     functional calls (F.batch_norm, F.layer_norm). They cannot be detected by looking at the
#     types of the modules and are therefore not reported by the warning either.
_UNSUPPORTED_NORM_LAYERS = (
    nn.modules.batchnorm._NormBase,  # batch and instance normalizations
    nn.LayerNorm,
    nn.GroupNorm,
)


def _disable_fast_path(module: nn.Module, args: tuple) -> None:
    """Forward pre-hook doing nothing but disabling the fast path of PyTorch's transformers.

    In evaluation mode, ``nn.TransformerEncoderLayer`` may use a fast path that reads the
    parameters of its normalization layers without calling them, which would silently discard
    the noise of the ABNN layers. PyTorch does not use this fast path when hooks are attached
    to the submodules of the layer.
    """


def _to_adapter(
    module: nn.Module, alpha: float, reference: Tensor | None = None
) -> _NormAdapter | None:
    """Build the ABNN layer corresponding to a normalization layer.

    Args:
        module: Module to convert.
        alpha: Alpha value for ABNN.
        reference: Tensor giving the device and data type of the ABNN layer when the module has
            no parameter. Defaults to ``None``.

    Returns:
        The ABNN layer initialized from the normalization layer, in the same mode, or ``None``
        if the module is not a supported normalization layer.
    """
    batch_norm_adapter = _BATCH_NORM_ADAPTERS.get(type(module))
    if batch_norm_adapter is not None and module.track_running_stats:
        assert module.running_mean is not None
        assert module.running_var is not None
        assert module.num_batches_tracked is not None
        adapter = batch_norm_adapter(
            module.num_features,
            alpha=alpha,
            momentum=module.momentum,
            eps=module.eps,
            device=module.running_mean.device,
            dtype=module.running_mean.dtype,
        )
        adapter.running_mean = module.running_mean
        adapter.running_var = module.running_var
        adapter.num_batches_tracked = module.num_batches_tracked
    elif type(module) is nn.LayerNorm:
        # Without affine parameters, the layer does not have a device nor a data type
        param = module.weight if module.weight is not None else reference
        adapter = LayerNormAdapter(
            module.normalized_shape,
            alpha=alpha,
            eps=module.eps,
            device=None if param is None else param.device,
            dtype=None if param is None else param.dtype,
        )
    else:
        return None

    # Without affine parameters, keep the identity initialization of the adapter.
    if module.weight is not None:
        adapter.weight.data = module.weight.data
    if module.bias is not None:
        adapter.bias.data = module.bias.data
    adapter.train(module.training)
    return adapter


def _replace_norm_layers(
    model: nn.Module, alpha: float, reference: Tensor | None = None
) -> list[str]:
    """Recursively replace the supported normalization layers with ABNN layers.

    The batch normalization layers tracking their running statistics (``BatchNorm1d``,
    ``BatchNorm2d``, ``BatchNorm3d``, and ``SyncBatchNorm``) and the ``LayerNorm`` layers are
    replaced. Their subclasses and the other normalization layers are left unchanged.

    Args:
        model: Model to replace normalization layers.
        alpha: Alpha value for ABNN.
        reference: Tensor giving the device and data type of the ABNN layers replacing layers
            without parameters. Defaults to ``None``, which corresponds to the first parameter
            of the model.

    Returns:
        The names of the classes of the normalization layers that were left unchanged.
    """
    if reference is None:
        reference = next(model.parameters(), None)

    unsupported = []
    for name, module in model.named_children():
        adapter = _to_adapter(module, alpha, reference)
        if adapter is not None:
            if isinstance(model, nn.TransformerEncoderLayer):
                adapter.register_forward_pre_hook(_disable_fast_path)
            setattr(model, name, adapter)
        elif isinstance(module, _UNSUPPORTED_NORM_LAYERS):
            unsupported.append(type(module).__name__)
        else:
            unsupported.extend(_replace_norm_layers(module, alpha, reference))

    # Without the fast path, the layers of the encoder cannot process the nested tensors that
    # it builds from the padding masks. The outputs only differ at the padded positions, which
    # are no longer set to zero.
    if isinstance(model, nn.TransformerEncoder) and any(
        isinstance(module, _NormAdapter) for module in model.modules()
    ):
        model.use_nested_tensor = False
    return unsupported
