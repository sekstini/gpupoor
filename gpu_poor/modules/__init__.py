import torch
import torch.nn as nn

from gpu_poor.kernels import split_k_sequential


class LowPrecisionLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return split_k_sequential(input, self.weight.T, self.bias, 32)
