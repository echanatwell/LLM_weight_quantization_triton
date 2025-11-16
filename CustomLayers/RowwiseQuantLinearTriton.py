import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _quantize_rowwise(w_ptr, scale_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(0)
    
    scale = tl.load(scale_ptr + pid0)
    for start in range(0, tl.cdiv(n_elements, BLOCK_SIZE)):
        w_even_off = pid0*n_elements + start*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)*2
        w_odd_off = pid0*n_elements + start*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)*2 + 1
        mask_w_even = w_even_off < pid0*n_elements + n_elements
        mask_w_odd = w_odd_off < pid0*n_elements + n_elements
    
        w_even = tl.load(w_ptr + w_even_off, mask=mask_w_even)
        w_odd = tl.load(w_ptr + w_odd_off, mask=mask_w_odd)

        w_quant_even = tl.extra.cuda.libdevice.rint(7. * w_even * scale).to(tl.int8)
        w_quant_odd = tl.extra.cuda.libdevice.rint(7. * w_odd * scale).to(tl.int8)

        # To unsigned
        w_quant_even += 8
        w_quant_odd += 8

        packed = (w_quant_odd << 4) | w_quant_even
        
        off_out = pid0 * (n_elements // 2) + start*(BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
        mask_out = off_out < pid0 * (n_elements // 2) + (n_elements // 2)
        tl.store(out_ptr + off_out, packed.to(tl.int8), mask_out)


@triton.jit
def _matmul_int4_bf16(x_ptr, w_ptr, 
                     w_scale_ptr, out_ptr, 
                     B, IN, OUT, 
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr,
                     GROUP_M: tl.constexpr,
                     PER_CHANNEL: tl.constexpr):
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

        # x_even
        x_off = off_m[:, None]*IN + (off_k[None, :] + start*(BLOCK_K // 2)) * 2
        mask_x = (off_m[:, None] < B) & ((off_k[None, :] + start*(BLOCK_K // 2)) * 2 < IN)
        x_even = tl.load(x_ptr + x_off, mask=mask_x)

        # x_odd
        x_off_odd = x_off + 1
        mask_x_odd = (off_m[:, None] < B) & ((off_k[None, :] + start*(BLOCK_K // 2)) * 2 + 1 < IN)
        x_odd = tl.load(x_ptr + x_off_odd, mask=mask_x_odd)
        
        x = tl.interleave(x_even, x_odd)

        # load w
        w_off = off_n[:, None]*(IN // 2) + (off_k[None, :] + start*(BLOCK_K // 2))
        mask_w = (off_n[:, None] < OUT) & (off_k[None, :] + start*(BLOCK_K // 2) < IN // 2)
        w_packed = tl.load(w_ptr + w_off, mask=mask_w)

        # unpacking
        w_even = w_packed & 0xF
        w_odd = (w_packed >> 4) & 0xF
        
        # unsigned to signed
        # w_even = (w_even ^ 0x8) - 0x8
        # w_odd = (w_odd ^ 0x8) - 0x8
        w_even = tl.where(mask_w, w_even - 8, 0.)
        w_odd = tl.where(mask_w, w_odd - 8, 0.)

        w = tl.interleave(w_even, w_odd)
        w = tl.trans(w)

        acc += tl.dot(x, w, out_dtype=tl.float32)

    # scaling
    if PER_CHANNEL:
        mask_scale = off_n < OUT
        w_scale = tl.load(w_scale_ptr + off_n, mask=mask_scale)
    else:
        w_scale = tl.load(w_scale_ptr)
    
    acc = acc / w_scale[None, :] / 7.

    out = acc.to(tl.bfloat16) # acc.to(tl.float32)
    
    out_off = off_m[:, None]*OUT + off_n[None, :]
    mask_out = (off_m[:, None] < B) & (off_n[None, :] < OUT)
    tl.store(out_ptr + out_off, out, mask=mask_out)

class RowwiseQuantLinearTriton(nn.Module):
    def __init__(self, l):
        super(RowwiseQuantLinearTriton, self).__init__()

        OUT, IN = l.weight.size()
        weight_quant = torch.empty((OUT, IN // 2), dtype=torch.int8, device=l.weight.device)
        self.w_scale = 1 / (l.weight.clone().detach().abs().max(dim=1)[0] + 1e-7)
        BLOCK_SIZE = 512
        grid = (OUT, )

        _quantize_rowwise[grid](l.weight.clone(), self.w_scale, weight_quant, IN, BLOCK_SIZE)

        self.weight_quant = weight_quant

        self.weight_quant = nn.Parameter(self.weight_quant, requires_grad=False)
        self.w_scale = nn.Parameter(self.w_scale, requires_grad=False)

    def forward(self, x):
        B, L, IN = x.size()
        OUT, _ = self.weight_quant.size()
        x_flatten = x.view(B*L, IN).contiguous()
        BLOCK_M = 32 #64
        BLOCK_N = 32 #64
        BLOCK_K = 64 #64
        GROUP_M = 4

        grid = lambda meta: (triton.cdiv(B*L, meta['BLOCK_M'])*triton.cdiv(OUT, meta['BLOCK_N']),) 
        out = torch.empty((B*L, OUT), dtype=x.dtype, device=x.device)

        _matmul_int4_bf16[grid](x_flatten, self.weight_quant, 
                               self.w_scale,
                               out, 
                               B*L, IN, OUT,
                               BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
                               PER_CHANNEL=(self.w_scale.numel() > 1))
        
        out = out.view(B, L, OUT).contiguous()
        
        return out
