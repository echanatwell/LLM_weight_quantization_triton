import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinearRowwiseTorch(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        device = original_layer.weight.device

        weight = original_layer.weight.data
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone().to(device))
        else:
            self.register_parameter('bias', None)

        self.weight_row_scales, self.packed_weight = self.quantize_weight(weight, device)

    def quantize_weight(self, weight, device):
        max_row_vals = torch.max(torch.abs(weight), axis=1).values[:, None]  # w(R, C) -> m(R,1)
        row_scales = max_row_vals / 7.0  # m(R,1) -> m(R,1)

        quantized = torch.clamp(torch.round(weight / row_scales), -8, 7).to(torch.int8)

        packed = self.pack_int4(quantized)

        return (nn.Parameter(row_scales.to(device), requires_grad=False), 
                nn.Parameter(packed.to(device), requires_grad=False))

    def pack_int4(self, tensor):
        tensor = tensor.to(torch.int8)
        
        high = (tensor[:, 0::2] & 0x0F) << 4
        low = tensor[:, 1::2] & 0x0F
        
        packed = high | low
        return packed

    def unpack_int4(self, packed_tensor):
        high = (packed_tensor >> 4) & 0x0F
        low = packed_tensor & 0x0F
        
        high = torch.where(high > 7, high - 16, high)
        low = torch.where(low > 7, low - 16, low)
        
        unpacked = torch.zeros(self.out_features, self.in_features, 
                               dtype=torch.float32, device=packed_tensor.device)
        unpacked[:, 0::2] = high.float()
        unpacked[:, 1::2] = low.float()
        
        return unpacked

    def forward(self, x):
        unpacked_weight = self.unpack_int4(self.packed_weight)

        restored_weight = unpacked_weight * self.weight_row_scales

        return F.linear(x, restored_weight, self.bias)
