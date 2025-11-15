import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymQuantizedLinearGlobalTorch(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        device = original_layer.weight.device

        weight = original_layer.weight.data
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone().to(device))
        else:
            self.register_parameter("bias", None)

        self.weight_scale, self.weight_zero_point, self.packed_weight = self.quantize_weight(
            weight, device
        )

    def quantize_weight(self, weight, device):
        w_min = weight.min()
        w_max = weight.max()

        qmin = -8
        qmax = 7

        # scale and zero-point
        scale = (w_max - w_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - torch.round(w_min / scale)

        # clamp ZP
        zero_point = torch.clamp(zero_point, qmin, qmax)

        # quantize
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)

        packed = self.pack_int4(quantized)

        return (
            nn.Parameter(scale.to(device), requires_grad=False),
            nn.Parameter(zero_point.to(device), requires_grad=False),
            nn.Parameter(packed.to(device), requires_grad=False),
        )

    def pack_int4(self, tensor):
        # Convert to unsigned
        t = tensor.clone()
        t = torch.where(t < 0, t + 16, t).to(torch.int8)

        high = (t[:, 0::2] & 0x0F) << 4
        low = t[:, 1::2] & 0x0F

        return (high | low).to(torch.uint8)

    def unpack_int4(self, packed):
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F

        # Convert back to signed int4
        high = torch.where(high > 7, high - 16, high)
        low = torch.where(low > 7, low - 16, low)

        unpacked = torch.zeros(
            self.out_features, self.in_features, dtype=torch.float32, device=packed.device
        )
        unpacked[:, 0::2] = high.float()
        unpacked[:, 1::2] = low.float()

        return unpacked

    def forward(self, x):
        q = self.unpack_int4(self.packed_weight)

        w = self.weight_scale * (q - self.weight_zero_point)

        return F.linear(x, w, self.bias)
