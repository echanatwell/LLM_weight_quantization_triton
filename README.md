# Quantization bf16 to int4 for Llama inference

## Memory effiency
| Layer                                      | Size, MB |
|--------------------------------------------|----------|
| Original (BFloat16)                        | 8.000    |
| GlobalQuantInt4 (Triton)                   | 2.000    |
| RowwiseQuantInt4 (Triton)                  | 2.004    |
| GlobalQuantInt4 (Torch)                    | 2.000    |
| RowwiseQuantInt4 (Torch)                   | 2.004    |
| Llama-3.2-1B-Instruct (Original)           | 4714.258 |
| Llama-3.2-1B-Instruct (Rowwise Quant)      | 2027.383 |

- matrix shape - 2048x2048, 
- packed matrix number of elements = original_num_elements // 2

## Matmul speed

### Weight matrix scaling

<img width="4490" height="2990" alt="123" src="https://github.com/user-attachments/assets/ba94e566-6c2c-4ec3-85ad-03facbd641ad" />

**Matrix multiplication performance:** X @ W, X - (1, Len, IN), W - (IN, IN)

### Llama linear layer shapes
| Layer           | 2048x512 | 2048x2048 | 2048x8192 | 2048x128256 |
|-----------------|----------|-----------|-----------|-------------|
| BitsAndBytes    | 330.3    | 338.7     | 319.2     | 374.5       |
| TorchGlobal     | 518.3    | 518.4     | 498.6     | 559.5       |
| TorchRowwise    | 531.8    | 518.7     | 461.2     | 555.0       |
| TritonRowwise   | 614.6    | 588.0     | 562.3     | 641.3       |

*Inference time in microseconds*

## Llama inference benchmark

| Layer                              | Mean inference time per sample, s | Mean Perplexity |
|------------------------------------|-----------------------------------|-----------------|
| Original                           | 0.028                             | 345.057         |
| Per-tensor quantization (Pytorch)  | 0.105                             | 817501.25       |
| Per-tensor quantization (Triton)   | 0.089                             | 820965.56       |
| Rowwise quantization (Pytorch)     | 0.105                             | 393.717         |
| Rowwise quantization (Triton)      | 0.090                             | 386.303         |

## Summary

- Triton quant implementation outperforms PyTorch quant implementation, but it loses classic PyTorch matmul implementation;
- Int4 quantization with weight packing reduces weight costs by 4 times;
- Rowwise quantization performs significantly better for the static weight quantization than Per-tensor quantization;

