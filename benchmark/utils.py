import torch
import torch.nn as nn


def matrices_mse(m1: torch.tensor, m2: torch.tensor) -> float:
    assert m1.shape == m2.shape
    assert m1.dtype == m2.dtype

    return float(((m1 - m2) ** 2).mean())


def layer_memory(l: nn.Module) -> float:
    """Get total bytes for parameters and buffers in layer"""
    total = 0
    for t in l.parameters():
        total += t.numel() * t.element_size()
    for t in l.buffers():
        total += t.numel() * t.element_size()
    return total
