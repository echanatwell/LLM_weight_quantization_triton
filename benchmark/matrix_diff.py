import torch


def matrices_mse(m1: torch.tensor, m2: torch.tensor) -> float:
    assert m1.shape == m2.shape
    assert m1.dtype == m2.dtype

    return float(((m1 - m2) ** 2).mean())
