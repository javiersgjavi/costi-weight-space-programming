"""Checks against the repository's independent reference equations."""

import ast
from pathlib import Path
import types

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from einops import rearrange, repeat

from .data import sample_mask
from .reference_mamba import selective_scan, mamba_inner_no_out_proj


def source_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("seed", [1, 7])
def test_scan_matches_upstream_reference(seed):
    torch.manual_seed(seed)
    upstream = source_function(
        "src/nn/blocks/bimamba/ops/selective_scan_interface.py", "selective_scan_ref",
        {"torch": torch, "F": F, "rearrange": rearrange, "repeat": repeat})
    u, delta, z = [torch.randn(3, 8, 13) for _ in range(3)]
    A = -torch.rand(8, 4)
    B, C = torch.randn(3, 4, 13), torch.randn(3, 4, 13)
    D, bias = torch.randn(8), torch.randn(8)
    torch.testing.assert_close(
        selective_scan(u, delta, A, B, C, D, z, bias),
        upstream(u, delta, A, B, C, D, z, bias, delta_softplus=True),
        rtol=2e-6, atol=2e-6)


def test_inner_matches_upstream_reference():
    torch.manual_seed(17)
    scan = source_function(
        "src/nn/blocks/bimamba/ops/selective_scan_interface.py", "selective_scan_ref",
        {"torch": torch, "F": F, "rearrange": rearrange, "repeat": repeat})
    def conv(x, w, b, activation):
        return F.silu(F.conv1d(x, w[:, None, :], b, padding=w.shape[-1]-1,
                              groups=x.shape[1])[..., :x.shape[-1]])
    upstream = source_function(
        "src/nn/blocks/bimamba/ops/selective_scan_interface.py", "mamba_inner_ref",
        {"torch": torch, "F": F, "rearrange": rearrange,
         "causal_conv1d_fn": conv, "selective_scan_fn": scan})
    xz, w, b = torch.randn(3, 16, 13), torch.randn(8, 1, 4), torch.randn(8)
    xp, dp, A = torch.randn(10, 8) * .1, torch.randn(8, 2) * .1, -torch.rand(8, 4)
    D, bias = torch.randn(8), torch.randn(8)
    out = mamba_inner_no_out_proj(xz, w, b, xp, dp, A, D=D, delta_bias=bias)
    expected = upstream(xz, w, b, xp, dp, torch.eye(8), None, A,
                        D=D, delta_bias=bias).transpose(1, 2)
    torch.testing.assert_close(out, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("p,p_noise,seed", [(0., .25, 42), (.0015, .05, 9101112)])
def test_missing_masks_match_tsl(p, p_noise, seed):
    source = Path("/home/javier/miniconda3/envs/conda/lib/python3.12/site-packages/tsl/ops/imputation.py")
    if not source.exists():
        pytest.skip("TSL source not present")
    reference = source_function(source, "sample_mask", {"np": np})
    expected = reference((2000, 12, 1), p=p, p_noise=p_noise, min_seq=12,
                         max_seq=48, rng=np.random.default_rng(seed), verbose=False)
    actual = sample_mask((2000, 12, 1), p, p_noise, 12, 48, seed)
    np.testing.assert_array_equal(actual, expected)


def test_predict_chunking_and_no_gradients():
    from .run import Inference
    from .data import load_data
    torch.set_num_threads(2)
    path = next(Path("../weights/ctimba_aqi-36_point/version_0").glob("*.ckpt"))
    data = load_data("aqi-36_point")
    ids = data.indices("calibration", 2)
    batch = data.batch(ids)
    runner = Inference(path, 0)
    a = runner.predict(batch, ids, 3, [80, 20.92], 73, chunk=1)
    b = runner.predict(batch, ids, 3, [80, 20.92], 73, chunk=3)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a[batch["mask"]], batch["x"][batch["mask"]], rtol=0, atol=0)
    assert not a.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in runner.model.parameters())
    assert all(not p.requires_grad for p in runner.model.input_sfe.support[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dataset", ["aqi-36_point", "metr-la_point"])
def test_cuda_matches_cpu_predictions(dataset):
    from .run import Inference
    from .data import load_data
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = next((Path("../weights") / ("ctimba_" + dataset) / "version_0").glob("*.ckpt"))
    data = load_data(dataset)
    ids = data.indices("calibration", 1)
    batch = data.batch(ids)
    sigmas = [80, .821] if dataset.startswith("metr") else [80, 20.92]
    cpu = Inference(path, 0)
    gpu = Inference(path, 0, device="cuda")
    a = cpu.predict(batch, ids, 3, sigmas, 73, chunk=3)
    b = gpu.predict(batch, ids, 3, sigmas, 73, chunk=3)
    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    assert all(p.grad is None for p in gpu.model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("length", [12, 24, 36, 48])
def test_fused_scan_matches_reference(length):
    from .cuda_scan import selective_scan_cuda
    torch.manual_seed(89)
    u, delta, z = [torch.randn(3, 128, length, device="cuda") for _ in range(3)]
    A = -torch.rand(128, 16, device="cuda") * 16
    B, C = [torch.randn(3, 16, length, device="cuda") for _ in range(2)]
    D, bias = [torch.randn(128, device="cuda") for _ in range(2)]
    expected = selective_scan(u, delta, A, B, C, D, z, bias)
    actual = selective_scan_cuda(u, delta, A, B, C, D, z, bias)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dataset", ["aqi-36_point", "metr-la_point", "mimic-iii-challenge_point"])
def test_fused_model_matches_reference(dataset):
    from .run import Inference
    from .data import load_data
    torch.set_num_threads(4)
    path = next((Path("../weights") / ("ctimba_" + dataset) / "version_0").glob("*.ckpt"))
    data = load_data(dataset)
    ids = data.indices("calibration", 1)
    batch = data.batch(ids)
    sigmas = [80, .821] if dataset.startswith("metr") else [80, 20.92]
    reference = Inference(path, 0, device="cuda")
    fused = Inference(path, 0, device="cuda", backend="triton")
    a = reference.predict(batch, ids, 3, sigmas, 73, chunk=3)
    b = fused.predict(batch, ids, 3, sigmas, 73, chunk=3)
    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)


def test_effective_column_edit_isolates_selected_neuron():
    from .run import apply_edit
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 1))
    before = (model[1].weight @ model[0].weight).detach().clone()
    apply_edit(model, {"operations": [{"key": "0.weight", "kind": "scale_effective_columns",
                                       "readout_key": "1.weight", "columns": [3], "value": .9}]})
    expected = before.clone()
    expected[:, 3] *= .9
    torch.testing.assert_close(model[1].weight @ model[0].weight, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_graph_replays_new_inputs_without_changing_predictions():
    from .run import Inference
    from .data import load_data
    torch.set_num_threads(4)
    path = next(Path("../weights/ctimba_aqi-36_point/version_0").glob("*.ckpt"))
    data = load_data("aqi-36_point")
    eager = Inference(path, 0, device="cuda", backend="triton")
    graph = Inference(path, 0, device="cuda", backend="triton", execution="graph")
    for index in data.indices("calibration", 2):
        batch = data.batch([index])
        a = eager.predict(batch, [index], 3, [80, 20.92], 73, 3)
        b = graph.predict(batch, [index], 3, [80, 20.92], 73, 3)
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_zero_graph_edit_removes_unpersisted_random_dependency():
    import json
    from .run import Inference
    from .data import load_data
    torch.set_num_threads(4)
    path = next(Path("../weights/ctimba_metr-la_point/version_0").glob("*.ckpt"))
    edit = json.loads(Path("weight_space/edits/metr_point_v0_random_graph_zero.json").read_text())
    data = load_data("metr-la_point")
    ids = data.indices("calibration", 1)
    batch = data.batch(ids)
    runner = Inference(path, 0, edit)
    a = runner.predict(batch, ids, 3, [80, .821], 73, 3)
    generator = torch.Generator().manual_seed(9583)
    runner.model.input_sfe.support[-1] = [torch.randn(v.shape, generator=generator)
                                        for v in runner.model.input_sfe.support[-1]]
    b = runner.predict(batch, ids, 3, [80, .821], 73, 3)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
