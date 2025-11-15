import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _quantize_global(w_ptr, scale_ptr, zp_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    w_even_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2) * 2
    w_odd_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2) * 2 + 1
    mask_even = w_even_off < n_elements
    mask_odd = w_odd_off < n_elements

    w_even = tl.load(w_ptr + w_even_off, mask=mask_even)
    w_odd = tl.load(w_ptr + w_odd_off, mask=mask_odd)

    scale = tl.load(scale_ptr)
    zp = tl.load(zp_ptr)

    # q = round(w/scale + zp)
    wq_even = tl.extra.cuda.libdevice.rint(w_even / scale + zp).to(tl.int8)
    wq_odd = tl.extra.cuda.libdevice.rint(w_odd / scale + zp).to(tl.int8)

    # clamp
    wq_even = tl.clip(wq_even, -8, 7)
    wq_odd = tl.clip(wq_odd, -8, 7)

    wq_even = (wq_even + 8).to(tl.uint8)
    wq_odd = (wq_odd + 8).to(tl.uint8)

    packed = (wq_odd << 4) | wq_even

    out_off = pid * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    mask_out = out_off < n_elements // 2

    tl.store(out_ptr + out_off, packed.to(tl.uint8), mask=mask_out)


@triton.jit
def _matmul_int4_bf16(
    x_ptr,
    w_ptr,
    w_scale_ptr,
    w_zp_ptr,
    out_ptr,
    B,
    IN,
    OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(B, BLOCK_M)
    grid_n = tl.cdiv(OUT, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start in range(0, tl.cdiv(IN, BLOCK_K)):
        off_k = tl.arange(0, BLOCK_K // 2)

        # X even
        x_off = off_m[:, None] * IN + (off_k[None, :] + start * (BLOCK_K // 2)) * 2
        mask_x = (off_m[:, None] < B) & (x_off < (B * IN))
        x_even = tl.load(x_ptr + x_off, mask=mask_x)

        # X odd
        x_odd = tl.load(x_ptr + x_off + 1, mask=mask_x)

        x = tl.interleave(x_even, x_odd)

        # load packed weights
        w_off = off_n[:, None] * (IN // 2) + (off_k[None, :] + start * (BLOCK_K // 2))
        mask_w = off_n[:, None] < OUT
        w_packed = tl.load(w_ptr + w_off, mask=mask_w)

        w_even = w_packed & 0xF
        w_odd = (w_packed >> 4) & 0xF

        # unsigned to signed int4
        w_even = (w_even - 8).to(tl.int8)
        w_odd = (w_odd - 8).to(tl.int8)

        w = tl.interleave(w_even, w_odd)
        w = tl.trans(w)

        acc += tl.dot(x, w, out_dtype=tl.float32)

    # load scale + zp
    if PER_CHANNEL:
        w_scale = tl.load(w_scale_ptr + off_n, mask=off_n < OUT)
        w_zp = tl.load(w_zp_ptr + off_n, mask=off_n < OUT)
    else:
        w_scale = tl.load(w_scale_ptr)
        w_zp = tl.load(w_zp_ptr)

    # dequant:  real = scale * (q - zp)
    acc = acc * w_scale[None, :]

    acc -= w_zp[None, :] * w_scale[None, :]

    out = acc.to(tl.bfloat16)

    out_off = off_m[:, None] * OUT + off_n[None, :]
    mask_out = (off_m[:, None] < B) & (off_n[None, :] < OUT)
    tl.store(out_ptr + out_off, out, mask=mask_out)


class GlobalAsymQuantLinearTriton(nn.Module):
    def __init__(self, l):
        super().__init__()

        W = l.weight
        OUT, IN = W.shape

        w_min = W.min()
        w_max = W.max()

        qmin, qmax = -8, 7

        # scale and zero-point
        scale = (w_max - w_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zp = qmin - torch.round(w_min / scale)
        zp = torch.clamp(zp, qmin, qmax)

        self.scale = nn.Parameter(scale.to(W.device), requires_grad=False)
        self.zp = nn.Parameter(zp.to(W.device), requires_grad=False)

        weight_quant = torch.empty((OUT, IN // 2), dtype=torch.uint8, device=W.device)

        BLOCK_SIZE = 512
        grid = (triton.cdiv(OUT * IN, BLOCK_SIZE),)

        _quantize_global[grid](W, self.scale, self.zp, weight_quant, OUT * IN, BLOCK_SIZE)

        self.weight = nn.Parameter(weight_quant, requires_grad=False)

    def forward(self, x):
        B, L, IN = x.size()
        OUT = self.weight.size(0)

        x_flat = x.view(B * L, IN).contiguous()
        out = torch.empty((B * L, OUT), dtype=x.dtype, device=x.device)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 64
        GROUP_M = 4

        grid = lambda META: (
            triton.cdiv(B * L, META["BLOCK_M"]) * triton.cdiv(OUT, META["BLOCK_N"]),
        )

        _matmul_int4_bf16[grid](
            x_flat,
            self.weight,
            self.scale,
            self.zp,
            out,
            B * L,
            IN,
            OUT,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_M,
            num_warps=2,
            num_stages=2,
            PER_CHANNEL=False,
        )

        return out.view(B, L, OUT)
