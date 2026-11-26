import copy

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.layers.bayesian.abnn import BatchNormAdapter2d
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
        use_original_model: bool = True,
        precision: str = "32",
        model: nn.Module | None = None,
    ) -> None:
        r"""Adaptable Bayesian Neural Network (ABNN) post-processing.

        Replaces the ``BatchNorm2d`` layers of a pre-trained model with stochastic
        :class:`~torch_uncertainty.layers.bayesian.BatchNormAdapter2d` layers and finetunes
        :attr:`num_models` copies of the model, each with its own random prior on the classes.
        At test time, each finetuned model is sampled :attr:`num_samples` times.

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
                and keeps its batch normalization layers. Defaults to ``True``.
            precision: Machine precision for training & eval. Defaults to ``"32"``.
            model: Model to use. Defaults to ``None``.

        Note:
            The forward pass returns the logits of all the estimators, with shape
            :math:`(N \times B, C)`, where :math:`N` is the number of estimators, i.e., the number
            of models (including the original one if :attr:`use_original_model`) times
            :attr:`num_samples`. The prediction of the original model is computed once and
            repeated :attr:`num_samples` times.

        Note:
            The models of the ensemble, including the original one, are copies made by :meth:`fit`
            and put in evaluation mode. The model given to the post-processing method is left
            untouched and its later changes do not affect the ensemble.

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
        )
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
            ValueError: If the model is not set or does not contain any ``BatchNorm2d`` layer
                with running statistics.
        """
        # Imported here to avoid a circular import with the routines
        from torch_uncertainty.routines import ClassificationRoutine

        if self.model is None:
            raise ValueError("Model must be set before fitting.")

        source_model = copy.deepcopy(self.model)
        _replace_bn_layers(source_model, self.alpha)
        if not any(isinstance(module, BatchNormAdapter2d) for module in source_model.modules()):
            raise ValueError(
                "ABNN requires a model with BatchNorm2d layers tracking their running statistics, "
                "but none was found."
            )

        models = [copy.deepcopy(source_model) for _ in range(self.num_models)]

        baselines = [
            ClassificationRoutine(
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
            preds.append(self.original_model(inputs).repeat(self.num_samples, 1))
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
) -> None:
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


def _replace_bn_layers(model: nn.Module, alpha: float) -> None:
    """Recursively replace batch normalization layers with ABNN layers.

    Args:
        model: Model to replace batch normalization layers.
        alpha: Alpha value for ABNN.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d) and module.track_running_stats:
            assert module.running_mean is not None
            assert module.running_var is not None
            assert module.num_batches_tracked is not None
            new_module = BatchNormAdapter2d(
                module.num_features,
                alpha=alpha,
                momentum=module.momentum,
                eps=module.eps,
                device=module.running_mean.device,
                dtype=module.running_mean.dtype,
            )
            new_module.running_mean = module.running_mean
            new_module.running_var = module.running_var
            new_module.num_batches_tracked = module.num_batches_tracked

            # Without affine parameters, keep the identity initialization of the adapter.
            if module.affine:
                new_module.weight.data = module.weight.data
                new_module.bias.data = module.bias.data
            new_module.train(module.training)
            setattr(model, name, new_module)
        else:
            _replace_bn_layers(module, alpha)
