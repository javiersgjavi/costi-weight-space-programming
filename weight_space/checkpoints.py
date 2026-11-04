"""Load tensor weights without importing or executing training metadata."""

from pathlib import Path
import hashlib
import typing

import numpy as np
import torch


class MetadataOnly:
    """Inert substitute for serialized metric, optimizer, and config objects."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.serialized_state = state


METADATA_NAMES = (
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.nodes.AnyNode",
    "omegaconf.base.ContainerMetadata",
    "omegaconf.base.Metadata",
    "tsl.metrics.torch.metrics.MaskedMAE",
    "tsl.metrics.torch.metrics.MaskedMSE",
    "tsl.metrics.torch.metrics.MaskedMRE",
    "schedulefree.adamw_schedulefree.AdamWScheduleFree",
    "torchmetrics.utilities.data.dim_zero_sum",
    "torchmetrics.metric.jit_distributed_available",
    "torch.nn.functional.mse_loss",
    "torch.nn.functional.l1_loss",
    "functools.partial",
)


def metadata_defaultdict(*args):
    return {}


def load_checkpoint(path):
    """Use PyTorch's restricted unpickler, never the training object loader."""
    allowed = [
        dict, slice, typing.Any, np.ndarray, np.dtype,
        (metadata_defaultdict, "collections.defaultdict"),
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        type(np.dtype("float32")), type(np.dtype("float64")),
        type(np.dtype("int64")), type(np.dtype("bool")),
        *[(MetadataOnly, name) for name in METADATA_NAMES],
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def model_state(path):
    checkpoint = load_checkpoint(path)
    state = checkpoint.get("state_dict", checkpoint)
    model = {k.removeprefix("model."): v for k, v in state.items()
             if k.startswith("model.")}
    if not model:
        model = state
    if not model or not all(torch.is_tensor(v) for v in model.values()):
        raise ValueError("Expected a tensor-only model state dictionary")
    return model


def plain_metadata(value):
    """Read primitive values from inert OmegaConf records, without resolving code."""
    if isinstance(value, MetadataOnly):
        state = value.serialized_state
        if "_content" in state:
            return plain_metadata(state["_content"])
        if "_val" in state:
            return plain_metadata(state["_val"])
        raise ValueError(f"Unsupported metadata fields: {list(state)}")
    if isinstance(value, dict):
        return {k: plain_metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_metadata(v) for v in value]
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
