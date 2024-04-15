import os
from typing import Literal
import torch
import torch.nn as nn

import gpupoor.kernels.matmul.split_k_sequential


K_ACC_DIV_MAX = int(os.getenv("K_ACC_DIV_MAX", "32"))

class LowPrecisionLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias, out, k_acc_div_max, activation):
        return torch.ops.gpu_poor.split_k_sequential(
            input,
            weight.T,
            bias,
            out,
            dump_ptx=False,
            k_acc_div_max=k_acc_div_max,
            activation=activation,
        )

    @staticmethod
    def setup_context(ctx, inputs: tuple, output):
        input, weight, bias, _, k_acc_div_max, activation = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.k_acc_div_max = k_acc_div_max
        ctx.activation = activation

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        mm = lambda x, y: torch.ops.gpu_poor.split_k_sequential(
            x, y, bias=None, out=None, dump_ptx=False, k_acc_div_max=ctx.k_acc_div_max, activation=None
        )

        if ctx.needs_input_grad[0]:
            grad_input = mm(grad_output, weight)

        if ctx.needs_input_grad[1]:
            grad_weight = mm(grad_output.T, input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None


class LowPrecisionLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = kwargs.get("activation", None)
        assert self.activation is None, "activation not supported yet"

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value: Literal[None, "relu", "gelu", "silu"]):
        assert value in [None, "relu", "gelu", "silu"]
        self._activation = value

    @torch.compiler.disable()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = LowPrecisionLinearFunction.apply(
            input.flatten(0, -2),
            self.weight,
            self.bias,
            None,  # out
            K_ACC_DIV_MAX,  # k_acc_div_max
            self.activation,  # activation
        )
        return out.reshape(*input.shape[:-1], self.out_features)
