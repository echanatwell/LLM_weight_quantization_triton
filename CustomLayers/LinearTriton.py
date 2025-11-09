import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _matmul(
    x_ptr,
    w_ptr,
    out_ptr,
    B,
    IN,
    OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.prorgam_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, IN, BLOCK_K):
        off_k = start + tl.arange(0, BLOCK_K)

        off_x = off_m[:, None] * IN + off_k[None, :]
        mask_x = (off_m[:, None] < B) & (off_k[None, :] < IN)
        x = tl.load(x_ptr + off_x, mask=mask_x)

        off_w = off_k[:, None] * OUT + off_n
        mask_w = (off_k[:, None] < IN) & (off_n[None, :] < OUT)
        w = tl.load(w_ptr + off_w, mask=mask_w)

        acc += tl.dot(x, w, out_dtype=tl.float32)

    off_out = off_m[:, None] * OUT + off_n[None, :]
    mask_out = (off_m[:, None] < B) & (off_n[None, :] < OUT)
    tl.store(out_ptr + off_out, acc, mask=mask_out)


class LinearTriton(nn.Module):
    def __init__(self, l):
        super(LinearTriton, self).__init__()
        self.weight = nn.Parameter(l.weight.clone())

    def forward(self, x):
        B, IN = x.size()
        OUT, _ = self.weight

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(OUT, BLOCK_N))
        out = torch.empty((B, OUT), dtype=torch.float32, device=x.device)
        _matmul[grid](x, self.weight.T, out, B, IN, OUT, BLOCK_M, BLOCK_N, BLOCK_K)
        out = out.to(x.dtype)

        return out
