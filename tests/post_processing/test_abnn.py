import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests._dummies.datamodule import DummyClassificationDataModule
from torch_uncertainty.layers.bayesian import BatchNormAdapter2d
from torch_uncertainty.post_processing import ABNN
from torch_uncertainty.post_processing.abnn import _parse_device, _replace_bn_layers
from torch_uncertainty.routines import ClassificationRoutine
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


class TestBatchNormAdapter2d:
    """Testing the BatchNormAdapter2d layer."""

    def test_stochastic(self) -> None:
        layer = BatchNormAdapter2d(4, alpha=0.5)
        inputs = torch.randn(2, 4, 3, 3)
        for mode in (True, False):
            layer.train(mode)
            assert not torch.equal(layer(inputs), layer(inputs))

    def test_default_statistics(self) -> None:
        """Without noise, a new layer behaves as a new ``nn.BatchNorm2d``."""
        layer = BatchNormAdapter2d(4, alpha=0.0).eval()
        inputs = torch.randn(2, 4, 3, 3)
        assert torch.allclose(layer(inputs), nn.BatchNorm2d(4).eval()(inputs))

    @pytest.mark.parametrize("momentum", [0.1, 0.5, None])
    def test_running_stats_match_batch_norm(self, momentum: float | None) -> None:
        """The running statistics follow those of ``nn.BatchNorm2d``."""
        batch_norm = nn.BatchNorm2d(4, momentum=momentum)
        model = nn.Sequential(nn.BatchNorm2d(4, momentum=momentum))
        _replace_bn_layers(model, alpha=0.1)
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


class TestReplaceBNLayers:
    """Testing the conversion of the batch normalization layers."""

    @pytest.mark.parametrize(
        "bn_kwargs",
        [{}, {"eps": 1e-3}, {"momentum": 0.01}, {"momentum": None}, {"affine": False}],
    )
    def test_equivalence_without_noise(self, bn_kwargs: dict) -> None:
        """With alpha=0, the converted model is the original model."""
        # Double precision: the small running variances amplify the rounding errors
        model = _model(**bn_kwargs).double().eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.normal_()
                    module.running_var.uniform_(1e-4, 1e-3)
                    if module.affine:
                        module.weight.normal_()
                        module.bias.normal_()
        inputs = torch.randn(3, 1, 4, 4, dtype=torch.double)
        expected = model(inputs)

        _replace_bn_layers(model, alpha=0.0)
        adapters = [m for m in model.modules() if isinstance(m, BatchNormAdapter2d)]
        assert len(adapters) == 2
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        reference = nn.BatchNorm2d(4, **bn_kwargs)
        for adapter in adapters:
            assert adapter.eps == reference.eps
            assert adapter.momentum == reference.momentum
        assert torch.allclose(model(inputs), expected)

    @pytest.mark.parametrize("training", [True, False])
    def test_keeps_training_mode(self, training: bool) -> None:
        model = _model().train(training)
        _replace_bn_layers(model, alpha=0.1)
        assert all(module.training is training for module in model.modules())

    def test_deeply_nested(self) -> None:
        """The number of visits is linear in the depth of the model."""
        model = nn.Sequential(nn.BatchNorm2d(4))
        for _ in range(64):
            model = nn.Sequential(model, nn.BatchNorm2d(4))
        _replace_bn_layers(model, alpha=0.1)
        assert sum(isinstance(m, BatchNormAdapter2d) for m in model.modules()) == 65
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())

    def test_ignored_layers(self) -> None:
        """Layers without running statistics and other normalizations are left as is."""
        model = nn.ModuleDict(
            {
                "no_stats": nn.BatchNorm2d(4, track_running_stats=False),
                "bn1d": nn.BatchNorm1d(4),
                "layer_norm": nn.LayerNorm(4),
            }
        )
        _replace_bn_layers(model, alpha=0.1)
        assert not any(isinstance(m, BatchNormAdapter2d) for m in model.modules())


class TestABNN:
    """Testing the ABNN post-processing method."""

    @pytest.mark.parametrize(("device", "training"), [("cpu", False), (torch.device("cpu"), True)])
    def test_fit(
        self, device: torch.device | str, training: bool, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        model = _model().train(training)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        abnn = _abnn(model=model, device=device)
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

        # The finetuned models differ from the original one
        for finetuned in abnn.final_model.core_models:
            assert isinstance(finetuned[1], BatchNormAdapter2d)
            assert not torch.equal(model[-1].weight, finetuned[-1].weight)
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
        abnn = _abnn(model=_model(), use_original_model=False, num_models=1, num_samples=3)
        abnn.fit(_dataloader())
        assert abnn.final_model is not None
        assert len(abnn.final_model.core_models) == 1
        assert abnn.original_model is None
        inputs = torch.randn(5, 1, 4, 4)
        out = abnn(inputs).reshape(3, 5, NUM_CLASSES)
        # All the estimators are stochastic
        assert not torch.equal(out[0], out[1])
        assert not torch.equal(out[1], out[2])

    def test_fit_without_batch_norm(self) -> None:
        """Fitting a model that cannot be made stochastic is an error."""
        model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.LayerNorm(8), nn.Linear(8, 4))
        abnn = _abnn(model=model)
        with pytest.raises(ValueError, match="BatchNorm2d"):
            abnn.fit(_dataloader())
        assert abnn.final_model is None
        assert not abnn.trained

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
        ):
            with pytest.raises(ValueError):
                _abnn(**kwargs)

        abnn = _abnn()
        with pytest.raises(ValueError, match="Model must be set before fitting"):
            abnn.fit(_dataloader())
        with pytest.raises(ValueError, match="Model must be set before calling forward"):
            abnn(torch.randn(1, 1, 4, 4))
