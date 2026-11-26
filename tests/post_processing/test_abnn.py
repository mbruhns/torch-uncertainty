import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests._dummies.datamodule import DummyClassificationDataModule, DummySegmentationDataModule
from torch_uncertainty.layers.bayesian import (
    BatchNormAdapter,
    BatchNormAdapter1d,
    BatchNormAdapter2d,
    BatchNormAdapter3d,
    LayerNormAdapter,
)
from torch_uncertainty.optim_recipes import optim_abnn
from torch_uncertainty.post_processing import ABNN
from torch_uncertainty.post_processing.abnn import _parse_device, _replace_norm_layers
from torch_uncertainty.routines import ClassificationRoutine, SegmentationRoutine
from torch_uncertainty.utils import TUTrainer

NUM_CLASSES = 4


def _model(num_classes: int = NUM_CLASSES, **bn_kwargs) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4, **bn_kwargs),
        nn.ReLU(),
        nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4, **bn_kwargs)),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, num_classes),
    )


def _segmentation_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, num_classes, 1),
    )


def _dataloader(num_images: int = 16) -> DataLoader:
    inputs = torch.randn(num_images, 1, 4, 4)
    targets = torch.arange(num_images) % NUM_CLASSES
    return DataLoader(TensorDataset(inputs, targets), batch_size=8)


def _abnn(**kwargs) -> ABNN:
    args = {
        "num_classes": NUM_CLASSES,
        "random_prior": 2.0,
        "alpha": 0.1,
        "num_models": 2,
        "num_samples": 2,
        "base_lr": 0.01,
        "device": "cpu",
        "max_epochs": 1,
    }
    return ABNN(**(args | kwargs))


def _randomize(model: nn.Module) -> None:
    """Give non-trivial values to the parameters and statistics of the normalization layers."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if module.running_mean is not None:
                    module.running_mean.normal_()
                    module.running_var.uniform_(1e-4, 1e-3)
            elif not isinstance(module, nn.LayerNorm):
                continue
            if module.weight is not None:
                module.weight.normal_()
            if module.bias is not None:
                module.bias.normal_()


class _LayerNorm2d(nn.LayerNorm):
    """Layer normalization over the channels of NCHW inputs, as in timm and torchvision."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _BatchNormAct2d(nn.BatchNorm2d):
    """Batch normalization followed by an activation, as in timm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).relu()


class TestNormAdapters:
    """Testing the Bayesian normalization layers of ABNN."""

    @pytest.mark.parametrize(
        ("layer", "shape"),
        [
            (BatchNormAdapter(4, alpha=0.5), (2, 4)),
            (BatchNormAdapter1d(4, alpha=0.5), (2, 4, 3)),
            (BatchNormAdapter2d(4, alpha=0.5), (2, 4, 3, 3)),
            (BatchNormAdapter3d(4, alpha=0.5), (2, 4, 3, 3, 3)),
            (LayerNormAdapter(4, alpha=0.5), (2, 3, 4)),
            (LayerNormAdapter((3, 4), alpha=0.5), (2, 3, 4)),
        ],
    )
    def test_stochastic(self, layer: nn.Module, shape: tuple[int, ...]) -> None:
        inputs = torch.randn(shape)
        for mode in (True, False):
            layer.train(mode)
            out = layer(inputs)
            assert out.shape == shape
            assert not torch.equal(out, layer(inputs))

    @pytest.mark.parametrize(
        ("layer", "reference", "shape"),
        [
            (BatchNormAdapter(4, alpha=0.0), nn.BatchNorm1d(4), (2, 4)),
            (BatchNormAdapter1d(4, alpha=0.0), nn.BatchNorm1d(4), (2, 4, 3)),
            (BatchNormAdapter2d(4, alpha=0.0), nn.BatchNorm2d(4), (2, 4, 3, 3)),
            (BatchNormAdapter3d(4, alpha=0.0), nn.BatchNorm3d(4), (2, 4, 3, 3, 3)),
            (LayerNormAdapter(4, alpha=0.0), nn.LayerNorm(4), (2, 3, 4)),
            (LayerNormAdapter([3, 4], alpha=0.0), nn.LayerNorm([3, 4]), (2, 3, 4)),
        ],
    )
    def test_default_initialization(
        self, layer: nn.Module, reference: nn.Module, shape: tuple[int, ...]
    ) -> None:
        """Without noise, a new layer behaves as the corresponding new PyTorch layer."""
        inputs = torch.randn(shape)
        for mode in (True, False):
            layer.train(mode)
            reference.train(mode)
            assert torch.allclose(layer(inputs), reference(inputs), atol=1e-6)

    @pytest.mark.parametrize(
        ("layer", "shape"),
        [
            (BatchNormAdapter(4), (4,)),
            (BatchNormAdapter1d(4), (2, 4, 3, 3)),
            (BatchNormAdapter2d(4), (2, 4, 3)),
            (BatchNormAdapter3d(4), (2, 4, 3, 3)),
        ],
    )
    def test_input_dim(self, layer: nn.Module, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="input"):
            layer(torch.randn(shape))

    @pytest.mark.parametrize("momentum", [0.1, 0.5, None])
    def test_running_stats_match_batch_norm(self, momentum: float | None) -> None:
        """The running statistics follow those of ``nn.BatchNorm2d``."""
        batch_norm = nn.BatchNorm2d(4, momentum=momentum)
        model = nn.Sequential(nn.BatchNorm2d(4, momentum=momentum))
        _replace_norm_layers(model, alpha=0.1)
        layer = model[0]
        assert isinstance(layer, BatchNormAdapter2d)

        for _ in range(3):
            inputs = torch.randn(8, 4, 3, 3)
            batch_norm(inputs)
            layer(inputs)
        assert torch.allclose(layer.running_mean, batch_norm.running_mean, atol=1e-6)
        assert torch.allclose(layer.running_var, batch_norm.running_var, atol=1e-6)
        assert layer.num_batches_tracked == batch_norm.num_batches_tracked == 3

        layer.eval()
        layer(torch.randn(8, 4, 3, 3))
        assert layer.num_batches_tracked == 3


class TestReplaceNormLayers:
    """Testing the conversion of the normalization layers."""

    @pytest.mark.parametrize(
        "bn_kwargs",
        [{}, {"eps": 1e-3}, {"momentum": 0.01}, {"momentum": None}, {"affine": False}],
    )
    def test_equivalence_without_noise(self, bn_kwargs: dict) -> None:
        """With alpha=0, the converted model is the original model."""
        # Double precision: the small running variances amplify the rounding errors
        model = _model(**bn_kwargs).double().eval()
        _randomize(model)
        inputs = torch.randn(3, 1, 4, 4, dtype=torch.double)
        expected = model(inputs)

        assert _replace_norm_layers(model, alpha=0.0) == []
        adapters = [m for m in model.modules() if isinstance(m, BatchNormAdapter2d)]
        assert len(adapters) == 2
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        reference = nn.BatchNorm2d(4, **bn_kwargs)
        for adapter in adapters:
            assert adapter.eps == reference.eps
            assert adapter.momentum == reference.momentum
        assert torch.allclose(model(inputs), expected)

    @pytest.mark.parametrize(
        ("layer", "adapter_type", "shape"),
        [
            (nn.BatchNorm1d(4), BatchNormAdapter1d, (3, 4)),
            (nn.BatchNorm1d(4, eps=1e-3), BatchNormAdapter1d, (3, 4, 5)),
            (nn.BatchNorm3d(4, affine=False), BatchNormAdapter3d, (3, 4, 2, 2, 2)),
            (nn.SyncBatchNorm(4), BatchNormAdapter, (3, 4, 2, 2)),
            (nn.SyncBatchNorm(4, momentum=None), BatchNormAdapter, (3, 4)),
            (nn.LayerNorm(4), LayerNormAdapter, (3, 5, 4)),
            (nn.LayerNorm([5, 4], eps=1e-3), LayerNormAdapter, (3, 5, 4)),
            (nn.LayerNorm(4, elementwise_affine=False), LayerNormAdapter, (3, 5, 4)),
            (nn.LayerNorm(4, bias=False), LayerNormAdapter, (3, 5, 4)),
        ],
    )
    def test_equivalence_of_other_layers(
        self, layer: nn.Module, adapter_type: type, shape: tuple[int, ...]
    ) -> None:
        """With alpha=0, each supported layer is converted into an equivalent layer."""
        # The linear layer gives a data type to the layers without parameters
        model = nn.Sequential(copy.deepcopy(layer), nn.Linear(shape[-1], 2)).double().eval()
        _randomize(model)
        inputs = torch.randn(shape, dtype=torch.double)
        expected = model(inputs)

        assert _replace_norm_layers(model, alpha=0.0) == []
        assert type(model[0]) is adapter_type
        assert model[0].eps == layer.eps
        assert model[0].weight.dtype == torch.double
        assert model[0].bias.dtype == torch.double
        assert torch.allclose(model(inputs), expected)

        model[0].alpha = 0.5
        assert not torch.equal(model(inputs), model(inputs))

    @pytest.mark.parametrize("training", [True, False])
    def test_keeps_training_mode(self, training: bool) -> None:
        model = nn.Sequential(_model(), nn.LayerNorm(4), nn.BatchNorm1d(4)).train(training)
        _replace_norm_layers(model, alpha=0.1)
        assert all(module.training is training for module in model.modules())

    def test_deeply_nested(self) -> None:
        """The number of visits is linear in the depth of the model."""
        model = nn.Sequential(nn.BatchNorm2d(4))
        for _ in range(64):
            model = nn.Sequential(model, nn.BatchNorm2d(4))
        _replace_norm_layers(model, alpha=0.1)
        assert sum(isinstance(m, BatchNormAdapter2d) for m in model.modules()) == 65
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

    def test_unsupported_layers(self) -> None:
        """The unsupported normalization layers are reported and left as is."""
        layers = {
            "no_stats": nn.BatchNorm2d(4, track_running_stats=False),
            "group_norm": nn.GroupNorm(2, 4),
            "instance_norm": nn.InstanceNorm2d(4),
            # The subclasses may change the forward pass: they are not converted
            "layer_norm_2d": _LayerNorm2d(4),
            "batch_norm_act": _BatchNormAct2d(4),
            "nested": nn.Sequential(nn.Conv2d(4, 4, 1), nn.GroupNorm(1, 4)),
        }
        model = nn.ModuleDict(layers)
        unsupported = _replace_norm_layers(model, alpha=0.1)
        assert sorted(unsupported) == [
            "BatchNorm2d",
            "GroupNorm",
            "GroupNorm",
            "InstanceNorm2d",
            "_BatchNormAct2d",
            "_LayerNorm2d",
        ]
        assert all(model[name] is layer for name, layer in layers.items())
        assert isinstance(model["nested"][1], nn.GroupNorm)

    @pytest.mark.parametrize("norm_first", [False, True])
    def test_transformer_encoder(self, norm_first: bool) -> None:
        """The fast path of PyTorch's transformers does not bypass the converted layers."""
        layer = nn.TransformerEncoderLayer(
            8, 2, 16, dropout=0.0, batch_first=True, norm_first=norm_first
        )
        encoder = nn.TransformerEncoder(
            layer, 2, norm=nn.LayerNorm(8), enable_nested_tensor=not norm_first
        ).eval()
        _randomize(encoder)
        inputs = torch.randn(2, 5, 8)
        padding_mask = torch.tensor([[False, False, False, True, True], [False] * 5])

        model = copy.deepcopy(encoder)
        assert _replace_norm_layers(model, alpha=0.0) == []
        assert sum(isinstance(m, LayerNormAdapter) for m in model.modules()) == 5
        with torch.inference_mode():
            assert torch.allclose(model(inputs), encoder(inputs), atol=1e-5)
            # The outputs may only differ at the padded positions
            out = model(inputs, src_key_padding_mask=padding_mask)
            expected = encoder(inputs, src_key_padding_mask=padding_mask)
            assert torch.allclose(out[~padding_mask], expected[~padding_mask], atol=1e-5)

        model = copy.deepcopy(encoder)
        _replace_norm_layers(model, alpha=0.5)
        # The copies made during the fit keep the layers stochastic
        for stochastic_model in (model, copy.deepcopy(model)):
            with torch.inference_mode():
                for kwargs in ({}, {"src_key_padding_mask": padding_mask}):
                    out = stochastic_model(inputs, **kwargs)
                    assert not torch.equal(out, stochastic_model(inputs, **kwargs))


class TestABNN:
    """Testing the ABNN post-processing method."""

    @pytest.mark.parametrize(("device", "training"), [("cpu", False), (torch.device("cpu"), True)])
    def test_fit(
        self, device: torch.device | str, training: bool, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        model = _model().train(training)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        abnn = _abnn(model=model, device=device, use_original_model=True)
        assert not abnn.trained

        abnn.fit(_dataloader())
        assert abnn.trained

        # The finetuning does not write logs or checkpoints
        assert not list(tmp_path.iterdir())
        # The original model is left untouched, including its mode
        assert all(torch.equal(v, state_dict[k]) for k, v in model.state_dict().items())
        assert not any(isinstance(m, BatchNormAdapter2d) for m in model.modules())
        assert all(module.training is training for module in model.modules())

        # Each model is stored once, whatever the number of samples
        assert abnn.final_model is not None
        assert abnn.original_model is not None
        assert len(abnn.final_model.core_models) == 2
        num_params = sum(p.numel() for p in model.parameters())
        assert sum(p.numel() for p in abnn.final_model.parameters()) == 2 * num_params
        # The ensemble is in evaluation mode, whatever the mode of the original model
        assert not any(module.training for module in abnn.final_model.modules())
        assert not any(module.training for module in abnn.original_model.modules())

        inputs = torch.randn(5, 1, 4, 4)
        expected = copy.deepcopy(model).eval()(inputs)
        buffers = {k: v.clone() for k, v in abnn.named_buffers()}
        # (num_models + original model) * num_samples estimators
        out = abnn(inputs)
        assert out.shape == (6 * 5, NUM_CLASSES)
        # Inference does not update the running statistics
        assert all(torch.equal(v, buffers[k]) for k, v in abnn.named_buffers())

        # The original model comes first: it is deterministic and repeated num_samples times
        original_out = out[: 2 * 5].reshape(2, 5, NUM_CLASSES)
        assert torch.equal(original_out[0], expected)
        assert torch.equal(original_out[1], expected)
        # The samples of a given finetuned model are different
        finetuned_out = out[2 * 5 :].reshape(2, 2, 5, NUM_CLASSES)
        for model_id in range(2):
            assert not torch.equal(finetuned_out[0, model_id], finetuned_out[1, model_id])

        # Only the weights and biases of the adapters are finetuned
        for finetuned in abnn.final_model.core_models:
            adapters = [m for m in finetuned.modules() if isinstance(m, BatchNormAdapter2d)]
            assert len(adapters) == 2
            adapter_params = {id(param) for adapter in adapters for param in adapter.parameters()}
            assert len(adapter_params) == 4
            for name, param in finetuned.named_parameters():
                is_finetuned = not torch.equal(param, state_dict[name])
                assert is_finetuned is (id(param) in adapter_params), name
            # The finetuning runs in training mode, whatever the mode of the original model
            assert not torch.equal(model[1].running_mean, finetuned[1].running_mean)

        # The ensemble is a snapshot: later changes of the original model do not affect it
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        assert torch.equal(abnn(inputs)[:5], expected)

    def test_forward_before_fit(self) -> None:
        """Before fitting, the post-processing falls back on the original model."""
        model = _model().eval()
        abnn = _abnn(model=model)
        inputs = torch.randn(5, 1, 4, 4)
        assert torch.equal(abnn(inputs), model(inputs))

    def test_fit_without_original_model(self) -> None:
        """By default, the ensemble only contains the finetuned models, as in the paper."""
        abnn = _abnn(model=_model(), num_models=1, num_samples=3)
        assert not abnn.use_original_model
        abnn.fit(_dataloader())
        assert abnn.final_model is not None
        assert len(abnn.final_model.core_models) == 1
        assert abnn.original_model is None
        inputs = torch.randn(5, 1, 4, 4)
        out = abnn(inputs).reshape(3, 5, NUM_CLASSES)
        # All the estimators are stochastic
        assert not torch.equal(out[0], out[1])
        assert not torch.equal(out[1], out[2])

    def test_fit_without_supported_norm_layer(self) -> None:
        """Fitting a model that cannot be made stochastic is an error."""
        model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.GroupNorm(2, 8), nn.Linear(8, 4))
        abnn = _abnn(model=model)
        with pytest.raises(ValueError, match="none was found"):
            abnn.fit(_dataloader())
        assert abnn.final_model is None
        assert not abnn.trained

    def test_fit_layer_norm(self, caplog) -> None:
        """Models with layer normalization are supported, the other layers are reported."""
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.BatchNorm1d(8),
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Linear(8, NUM_CLASSES),
        )
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        abnn = _abnn(model=model, num_models=1, num_samples=2)
        with caplog.at_level("WARNING"):
            abnn.fit(_dataloader())
        assert "left unchanged: GroupNorm." in caplog.text

        assert abnn.final_model is not None
        finetuned = abnn.final_model.core_models[0]
        assert isinstance(finetuned[2], LayerNormAdapter)
        assert isinstance(finetuned[3], BatchNormAdapter1d)
        assert isinstance(finetuned[4], nn.GroupNorm)
        # Only the weights and biases of the adapters are finetuned
        for name, param in finetuned.named_parameters():
            is_finetuned = not torch.equal(param, state_dict[name])
            assert is_finetuned is name.startswith(("2.", "3.")), name

        out = abnn(torch.randn(5, 1, 4, 4)).reshape(2, 5, NUM_CLASSES)
        assert not torch.equal(out[0], out[1])

    def test_optim_recipe_skips_frozen_parameters(self) -> None:
        model = _model()
        model[0].requires_grad_(False)
        optimizer = optim_abnn(model, lr=0.1)["optimizer"]
        optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
        assert optimized == {id(param) for param in model.parameters() if param.requires_grad}
        assert not optimized & {id(param) for param in model[0].parameters()}

    def test_fit_segmentation(self) -> None:
        inputs = torch.randn(8, 3, 4, 4)
        targets = torch.randint(0, NUM_CLASSES, (8, 4, 4))
        targets[0, 0, 0] = 255  # ignored pixel
        dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

        model = _segmentation_model().eval()
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        abnn = _abnn(model=model, task="segmentation", use_original_model=True)
        abnn.fit(dataloader)
        assert abnn.final_model is not None

        # (num_models + original model) * num_samples estimators of dense predictions
        out = abnn(inputs[:2])
        assert out.shape == (6 * 2, NUM_CLASSES, 4, 4)
        expected = model(inputs[:2])
        original_out = out[: 2 * 2].reshape(2, 2, NUM_CLASSES, 4, 4)
        assert torch.equal(original_out[0], expected)
        assert torch.equal(original_out[1], expected)
        finetuned_out = out[2 * 2 :].reshape(2, 2, 2, NUM_CLASSES, 4, 4)
        for model_id in range(2):
            assert not torch.equal(finetuned_out[0, model_id], finetuned_out[1, model_id])

        # Only the weights and biases of the adapters are finetuned
        for finetuned in abnn.final_model.core_models:
            assert isinstance(finetuned[1], BatchNormAdapter2d)
            for name, param in finetuned.named_parameters():
                is_finetuned = not torch.equal(param, state_dict[name])
                assert is_finetuned is name.startswith("1."), name

    @pytest.mark.parametrize("ood_criterion", [None, "msp", "mutual_information"])
    def test_segmentation_routine(
        self, ood_criterion: str | None, tmp_path: Path, monkeypatch
    ) -> None:
        """ABNN can be used as the post-processing method of the segmentation routine."""
        monkeypatch.chdir(tmp_path)
        eval_ood = ood_criterion is not None
        dm = DummySegmentationDataModule(
            root=Path(),
            batch_size=4,
            num_classes=NUM_CLASSES,
            num_images=8,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            eval_ood=eval_ood,
        )
        abnn = _abnn(task="segmentation", num_models=1)
        routine = SegmentationRoutine(
            model=_segmentation_model(),
            num_classes=NUM_CLASSES,
            loss=None,
            post_processing=abnn,
            # Keep all the pixels: the images are too small to be subsampled
            metric_subsampling_rate=1,
            log_plots=True,
            eval_ood=eval_ood,
            ood_criterion=ood_criterion or "msp",
        )
        # Keep the default logger to cover the plots of the post-processed predictions
        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        results = {k: v for result in trainer.test(routine, dm) for k, v in result.items()}
        assert abnn.trained
        for metric in ("seg/mIoU", "seg/pixAcc", "seg/NLL", "cal/ECE", "cal/PAvPU"):
            assert f"test/{metric}" in results
            assert f"test/post/{metric}" in results
        # The OOD metrics of the post-processed predictions are computed with the same criterion
        for metric in ("AUROC", "AUPR", "FPR95"):
            assert (f"ood/{metric}" in results) is eval_ood
            assert (f"ood/post/{metric}" in results) is eval_ood

    def test_random_prior(self) -> None:
        abnn = _abnn(random_prior=3.0)
        assert len(abnn.weights) == 2
        for weight in abnn.weights:
            # int(sqrt(num_classes)) classes are reweighted
            assert sorted(weight.tolist()) == [1.0, 1.0, 3.0, 3.0]

    def test_routine(self, tmp_path: Path, monkeypatch) -> None:
        """ABNN can be used as the post-processing method of the classification routine."""
        monkeypatch.chdir(tmp_path)
        dm = DummyClassificationDataModule(
            root=Path(),
            batch_size=8,
            num_classes=NUM_CLASSES,
            num_images=16,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        abnn = _abnn(num_models=1)
        routine = ClassificationRoutine(
            model=_model(),
            loss=None,
            num_classes=NUM_CLASSES,
            post_processing=abnn,
        )

        # Record the modes during the test: Lightning restores the previous ones afterwards
        training_modes = []
        forward = abnn.forward

        def spy_forward(inputs: torch.Tensor) -> torch.Tensor:
            training_modes.append(any(module.training for module in abnn.modules()))
            return forward(inputs)

        monkeypatch.setattr(abnn, "forward", spy_forward)

        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        results = trainer.test(routine, dm)
        assert "test/post/cls/Acc" in results[0]

        assert abnn.trained
        assert abnn.final_model is not None
        # The predictions are made in evaluation mode, including the adapters
        assert training_modes == [False, False]

    @pytest.mark.parametrize(
        ("device", "expected"),
        [
            ("cpu", ("cpu", 1)),
            (torch.device("cpu"), ("cpu", 1)),
            ("cuda", ("cuda", 1)),
            ("cuda:1", ("cuda", [1])),
            (torch.device("cuda", 2), ("cuda", [2])),
            ("mps", ("mps", 1)),
            ("gpu", ("gpu", 1)),
            ("auto", ("auto", 1)),
        ],
    )
    def test_parse_device(self, device: torch.device | str, expected: tuple) -> None:
        assert _parse_device(device) == expected

    def test_errors(self) -> None:
        for kwargs in (
            {"random_prior": -1},
            {"alpha": -1},
            {"max_epochs": 0},
            {"num_models": 0},
            {"num_samples": 0},
            {"base_lr": -1},
            {"num_classes": 0},
            {"task": "regression"},
        ):
            with pytest.raises(ValueError):
                _abnn(**kwargs)

        abnn = _abnn()
        with pytest.raises(ValueError, match="Model must be set before fitting"):
            abnn.fit(_dataloader())
        with pytest.raises(ValueError, match="Model must be set before calling forward"):
            abnn(torch.randn(1, 1, 4, 4))
