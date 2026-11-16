"""Dataset loading and PyTorch MLP model definition."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from activations import activation_float, build_pytorch_activation, canonical_activation_name
from config import DatasetBundle, ModelBundle, QuantConfig

class FeedforwardNN(nn.Module):
    """Configurable fully connected network used by training and quantization."""
    def __init__(self, input_dim: int, layers: tuple[int, ...], activation: str,
                 output_dim: int = 2, activation_negative_slope: float = 1 / 64) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive integers")
        if not layers or any(units <= 0 for units in layers):
            raise ValueError("layers must contain at least one positive integer")
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else layers[i - 1], units) for i, units in enumerate(layers)]
        )
        self.out = nn.Linear(layers[-1], output_dim)
        self.activation_name = canonical_activation_name(activation)
        self.activation = build_pytorch_activation(self.activation_name, negative_slope=activation_negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.out(x)


def build_model(config: dict[str, Any], *, activation_negative_slope: float = 1 / 64) -> FeedforwardNN:
    """Build the MLP from the shared configuration dictionary."""
    return FeedforwardNN(
        input_dim=int(config["input_dim"]),
        layers=tuple(int(units) for units in config["layers"]),
        activation=str(config["activation"]),
        output_dim=int(config.get("output_dim", 2)),
        activation_negative_slope=activation_negative_slope,
    )


def prepare_mnist_dataset(x: np.ndarray, y: np.ndarray, *, normalize: bool = True) -> DatasetBundle:
    """Prepare MNIST for a basic MLP: flatten images and keep labels as classes."""
    x = np.asarray(x)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    x = x.astype(np.float32)
    if normalize and x.size and float(x.max()) > 1.0:
        x = x / 255.0
    return DatasetBundle(x=x, y=np.asarray(y).astype(np.int64), task="classification")


def build_activation_plan(config: dict[str, Any], n_hidden: int) -> list[dict[str, Any]]:
    activation_spec = config.get("activations_per_layer")
    if activation_spec is None:
        activation_spec = [config["activation"]] * n_hidden
    if len(activation_spec) != n_hidden:
        raise ValueError("activations_per_layer is inconsistent with the number of hidden layers")
    return [{"layer_index": i, "name": canonical_activation_name(name)} for i, name in enumerate(activation_spec)]


def load_model_bundle_from_dataset(model_path: Path, dataset: DatasetBundle,
                                   qcfg: QuantConfig, device: str = "cpu") -> ModelBundle:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = {**config, "input_dim": int(dataset.x.shape[1])}
    model = build_model(model_config, activation_negative_slope=qcfg.activation_negative_slope)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError("No state_dict found in the checkpoint.")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    activation_plan = build_activation_plan(config, len(model.hidden_layers))
    return ModelBundle(model=model, checkpoint=checkpoint, config=config, activation_plan=activation_plan)


def load_fashion_mnist_dataset(path: str | Path, *, split: str = "train") -> DatasetBundle:
    """Load only the requested Fashion-MNIST split, never silently substitute another.

    'train' is used for quantization calibration; 'test' is held out for evaluation.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    x_key, y_key = f"x_{split}", f"y_{split}"
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in (x_key, y_key) if key not in data.files]
        if missing:
            raise KeyError(f"Dataset {path} has no {missing}; available keys: {data.files}")
        x = data[x_key]
        y = data[y_key]
    return prepare_mnist_dataset(x, y)
