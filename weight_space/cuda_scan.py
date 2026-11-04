"""Forward-only fused implementation of the existing selective recurrence.

This changes execution, not model weights or equations. No backward kernel is
provided. API reference: https://triton-lang.org/main/python-api/triton.language.html
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _scan(U, Delta, A, B, C, D, Z, Bias, Out,
          CHANNELS: tl.constexpr, LENGTH: tl.constexpr, STATES: tl.constexpr,
          BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    states = tl.arange(0, BLOCK_S)
    valid = channels < CHANNELS
    a = tl.load(A + channels[:, None] * STATES + states[None, :],
                valid[:, None] & (states[None, :] < STATES), 0)
    d = tl.load(D + channels, valid, 0)
    bias = tl.load(Bias + channels, valid, 0)
    state = tl.full((BLOCK_D, BLOCK_S), 0, tl.float32)
    for t in range(LENGTH):
        offset = (batch * CHANNELS + channels) * LENGTH + t
        u = tl.load(U + offset, valid, 0)
        raw_delta = tl.load(Delta + offset, valid, 0) + bias
        delta = tl.where(raw_delta > 20., raw_delta,
                         tl.log(1. + tl.exp(tl.minimum(raw_delta, 20.))))
        b = tl.load(B + (batch * STATES + states) * LENGTH + t, states < STATES, 0)
        c = tl.load(C + (batch * STATES + states) * LENGTH + t, states < STATES, 0)
        state = tl.exp(delta[:, None] * a) * state + (delta * u)[:, None] * b[None, :]
        out = tl.sum(state * c[None, :], axis=1) + d * u
        z = tl.load(Z + offset, valid, 0)
        out = out * (z / (1. + tl.exp(-z)))
        tl.store(Out + offset, out, valid)


def selective_scan_cuda(u, delta, A, B, C, D=None, z=None,
                        delta_bias=None, delta_softplus=True):
    values = (u, delta, A, B, C, D, z, delta_bias)
    if any(v is None or v.device.type != "cuda" or v.dtype != torch.float32 for v in values):
        raise ValueError("Fused CoSTI scan requires float32 CUDA tensors, D, z and delta bias")
    if not delta_softplus or A.is_complex() or B.ndim != 3 or C.ndim != 3:
        raise ValueError("Unsupported selective-scan variant")
    if torch.is_grad_enabled() and any(v.requires_grad for v in values):
        raise RuntimeError("This backend is inference-only; gradients are prohibited")
    u, delta, A, B, C, D, z, delta_bias = [v.contiguous() for v in values]
    batch, channels, length = u.shape
    states = A.shape[1]
    output = torch.empty_like(u)
    _scan[(batch, triton.cdiv(channels, 16))](
        u, delta, A, B, C, D, z, delta_bias, output,
        channels, length, states, 16, triton.next_power_of_2(states),
        num_warps=4, enable_fp_fusion=False)
    return output
