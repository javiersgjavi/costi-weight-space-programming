"""Device-independent reference of the released BiMamba-v2 fused inner kernel.

Uses the same convolution, selective state recurrence, gate and reverse pass.
Installed only by the inference harness; the production CUDA path is unchanged.
"""

import torch
from torch.nn import functional as F
from typing import Optional


def selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                   B: torch.Tensor, C: torch.Tensor,
                   D: Optional[torch.Tensor] = None,
                   z: Optional[torch.Tensor] = None,
                   delta_bias: Optional[torch.Tensor] = None,
                   delta_softplus: bool = True):
    if A.is_complex() or B.ndim != 3 or C.ndim != 3:
        raise ValueError("CoSTI reference supports real A and input-dependent B/C")
    dtype = u.dtype
    u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    transition = torch.exp(delta[..., None] * A[None, :, None, :])
    drive = (delta * u)[..., None] * B.transpose(1, 2)[:, None, :, :]
    state = torch.zeros((u.shape[0], u.shape[1], A.shape[1]),
                        dtype=A.dtype, device=u.device)
    outputs = []
    for t in range(u.shape[-1]):
        state = transition[:, :, t] * state + drive[:, :, t]
        outputs.append((state * C[:, None, :, t]).sum(-1))
    out = torch.stack(outputs, dim=-1)
    if D is not None:
        out = out + u * D[None, :, None]
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype)


def mamba_inner_no_out_proj(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                           delta_proj_weight, A, B=None, C=None, D=None,
                           delta_bias=None, B_proj_bias=None, C_proj_bias=None,
                           delta_softplus=True, scan_impl=selective_scan):
    x, z = xz.chunk(2, dim=1)
    length = x.shape[-1]
    x = F.silu(F.conv1d(x, conv1d_weight, conv1d_bias,
                       padding=conv1d_weight.shape[-1] - 1,
                       groups=x.shape[1])[..., :length])
    projection = F.linear(x.transpose(1, 2), x_proj_weight)
    rank, state_size = delta_proj_weight.shape[1], A.shape[-1]
    delta = F.linear(projection[..., :rank], delta_proj_weight).transpose(1, 2)
    if B is None:
        B = projection[..., rank:rank + state_size]
        if B_proj_bias is not None:
            B = B + B_proj_bias
        B = B.transpose(1, 2).contiguous()
    if C is None:
        C = projection[..., -state_size:]
        if C_proj_bias is not None:
            C = C + C_proj_bias
        C = C.transpose(1, 2).contiguous()
    return scan_impl(x, delta, A, B, C, D, z,
                          delta_bias, delta_softplus)


def install_reference_backend(backend="reference"):
    from src.nn.blocks.bimamba.modules import mamba_simple
    if backend == "reference":
        mamba_simple.mamba_inner_fn_no_out_proj = mamba_inner_no_out_proj
    elif backend == "triton":
        from functools import partial
        from .cuda_scan import selective_scan_cuda
        mamba_simple.mamba_inner_fn_no_out_proj = partial(
            mamba_inner_no_out_proj, scan_impl=selective_scan_cuda)
    else:
        raise ValueError(backend)
