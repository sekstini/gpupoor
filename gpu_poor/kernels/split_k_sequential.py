from typing import Optional

import torch
import triton
import triton.language as tl
from torch.utils.flop_counter import flop_registry, register_flop_formula

VALUES_BLOCK_M = [128, 256]
VALUES_BLOCK_N = [64, 128]
VALUES_BLOCK_K = [32, 64]  # Do not change this value
VALUES_GROUP_M = [2, 8]
VALUES_NUM_STAGES = [4, 5]
VALUES_NUM_WARPS = [4, 8]


def calculate_k_acc_div(k: int, block_k: int, k_acc_div_max: int) -> int:
    k_div = triton.next_power_of_2(int((k / block_k) ** 0.5))
    return max(1, min(k_div, k_acc_div_max))


@triton.autotune(
    configs=[
        *[
            triton.Config(
                {
                    "BLOCK_M": BLOCK_M,
                    "BLOCK_N": BLOCK_N,
                    "GROUP_M": GROUP_M,
                },
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_M in VALUES_BLOCK_M
            for BLOCK_N in VALUES_BLOCK_N
            for GROUP_M in VALUES_GROUP_M
            for num_stages in VALUES_NUM_STAGES
            for num_warps in VALUES_NUM_WARPS
        ],
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    compute_dtype: tl.constexpr,
    inner_acc_dtype: tl.constexpr,
    outer_acc_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_ACC_DIV: tl.constexpr,
    K_IS_DIVISIBLE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)  # type: ignore
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=outer_acc_dtype)

    k_inner = tl.cdiv(K, BLOCK_K * K_ACC_DIV)

    for k_i in range(0, K_ACC_DIV):
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=inner_acc_dtype)
        for k in range(0, k_inner):
            if K_IS_DIVISIBLE:
                a = tl.load(a_block_ptr)
                b = tl.load(b_block_ptr)
            else:
                a = tl.load(a_block_ptr, boundary_check=(0, 1))
                b = tl.load(b_block_ptr, boundary_check=(0, 1))

            a, b = a.to(compute_dtype), b.to(compute_dtype)
            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=inner_acc_dtype)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        out += accumulator.to(outer_acc_dtype)

    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(
            base=bias_ptr,
            shape=(1, BLOCK_N),
            strides=(0, 1),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(1, BLOCK_N),
            order=(0, 1),
        )
        out += tl.load(bias_block_ptr, boundary_check=(1,)).to(outer_acc_dtype)

    out = out.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1),
    )

    tl.store(c_block_ptr, out, boundary_check=(0, 1))


@torch.library.custom_op(
    "gpu_poor::_split_k_sequential",
    mutates_args={},
)
def _split_k_sequential(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    k_acc_div_max: int,
) -> torch.Tensor:
    # a = a.contiguous()
    # b = b.contiguous()
    assert a.ndim == 2 and b.ndim == 2, "Input tensors must have 2 dimensions"
    assert a.shape[-1] == b.shape[-2], f"Incompatible dimensions: {a.shape} and {b.shape}"

    batch_dims, K = a.shape[:-1], a.shape[-1]
    M = batch_dims.numel()
    N = b.shape[-1]

    a = a.reshape(M, K)

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta: dict):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    K_ACC_DIV = calculate_k_acc_div(K, VALUES_BLOCK_K[0], k_acc_div_max)
    K_IS_DIVISIBLE = K % (VALUES_BLOCK_K[0] * K_ACC_DIV) == 0
    K_IS_DIVISIBLE &= (M % max(VALUES_BLOCK_M) == 0) and (N % max(VALUES_BLOCK_N) == 0)

    _matmul_kernel[grid](
        a,
        b,
        out,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        compute_dtype=tl.float16,
        inner_acc_dtype=tl.float16,
        outer_acc_dtype=tl.float16,
        K_ACC_DIV=K_ACC_DIV,
        K_IS_DIVISIBLE=K_IS_DIVISIBLE,
        BLOCK_K=VALUES_BLOCK_K[0],
    )

    return out.reshape(*batch_dims, N)


@_split_k_sequential.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    k_acc_div_max: int,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "Input tensors must have 2 dimensions"
    m, k1 = a.shape
    k2, n = b.shape
    assert k1 == k2, f"Incompatible contraction dimensions {a.shape} and {b.shape}"
    assert a.device == b.device, "Input tensors must be on the same device"
    if bias is not None:
        assert bias.ndim == 2, "Bias tensor must be of shape (1, N)"
        assert bias.shape[0] == 1
        assert bias.shape[1] == n, f"Incompatible bias dimensions {bias.shape} and {b.shape}"
    return torch.empty((m, n), dtype=a.dtype, device=a.device)


def setup_context(ctx, inputs: tuple, output):
    input, weight, bias, k_acc_div_max = inputs
    ctx.save_for_backward(input, weight, bias)
    ctx.k_acc_div_max = k_acc_div_max


def backward(ctx, grad_output):
    input, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    print(input.shape, weight.shape, grad_output.shape)

    if ctx.needs_input_grad[0]:
        grad_input = _split_k_sequential(
            grad_output,
            weight.T,
            bias=None,
            k_acc_div_max=ctx.k_acc_div_max,
        )

    if ctx.needs_input_grad[1]:
        grad_weight = _split_k_sequential(
            input.T,
            grad_output,
            bias=None,
            k_acc_div_max=ctx.k_acc_div_max,
        )

    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum(0)

    return grad_input, grad_weight, grad_bias, None


_split_k_sequential.register_autograd(backward, setup_context=setup_context)


@register_flop_formula(torch.ops.gpu_poor._split_k_sequential)
def _split_k_sequential_flop(a_shape, b_shape, bias_shape, *args, **kwargs) -> int:
    """Count flops for matmul."""
    m, k, n = *a_shape, b_shape[-1]
    flops = 2 * m * n * (k - 1)
    if bias_shape is not None:
        flops += m * n
    return flops


flop_registry[torch.ops.gpu_poor._split_k_sequential] = _split_k_sequential_flop

_split_k_sequential = torch.ops.gpu_poor._split_k_sequential


def split_k_sequential(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    max_k_splits: int = 32,
) -> torch.Tensor:
    assert weight.ndim == 2
    out = _split_k_sequential(x.flatten(0, -2), weight, bias, max_k_splits)
    return out.view(*x.shape[:-1], weight.shape[1])
