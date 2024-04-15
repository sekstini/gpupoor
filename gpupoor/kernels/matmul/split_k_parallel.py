from typing import Optional

import torch
import triton
import triton.language as tl
from torch.utils.flop_counter import register_flop_formula, flop_registry

K_ACC_DIV_MIN = 1

VALUES_BLOCK_M = [128, 256]
VALUES_BLOCK_N = [128]
VALUES_BLOCK_K = [16, 32]
VALUES_GROUP_M = [4, 8]
VALUES_NUM_STAGES = [3, 4, 5]
VALUES_NUM_WARPS = [4, 8]

M_MIN, N_MIN = 512, 256
assert M_MIN >= min(VALUES_BLOCK_M)
assert N_MIN >= min(VALUES_BLOCK_N)


@triton.jit
def relu(x: tl.tensor) -> tl.tensor:
    return tl.where(x > 0, x, tl.zeros(x.shape, dtype=x.dtype))


@triton.jit
def silu(x: tl.tensor) -> tl.tensor:
    return x * tl.sigmoid(x)


@triton.autotune(
    configs=[
        *[
            triton.Config(
                {
                    "BLOCK_M": BLOCK_M,
                    "BLOCK_N": BLOCK_N,
                    "BLOCK_K": BLOCK_K,
                    "GROUP_M": GROUP_M,
                },
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_M in VALUES_BLOCK_M
            for BLOCK_N in VALUES_BLOCK_N
            for BLOCK_K in VALUES_BLOCK_K
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
    bias_ptr: Optional[tl.tensor],
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    inner_acc_dtype: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    # TODO: Replace with tl.swizzle2d
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)  # type: ignore
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    K_BLOCKS = tl.cdiv(K, BLOCK_K * SPLIT_K)

    pid_k = tl.program_id(axis=1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, pid_k * K_BLOCKS),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(pid_k * K_BLOCKS, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=inner_acc_dtype)

    if bias_ptr is not None:
        if pid_k == 0:
            bias_block_ptr = tl.make_block_ptr(
                base=bias_ptr,
                shape=(1, BLOCK_N),
                strides=(0, 1),
                offsets=(0, pid_n * BLOCK_N),
                block_shape=(1, BLOCK_N),
                order=(0, 1),
            )
            out += tl.load(bias_block_ptr, boundary_check=(0, 1)).to(inner_acc_dtype)

    # Needed?
    # tl.debug_barrier()

    for k in range(0, K_BLOCKS):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(1, 0))

        out = tl.dot(a, b, acc=out, out_dtype=inner_acc_dtype)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    if ACTIVATION == "relu":
        out = relu(out)
    # elif ACTIVATION == "gelu":
    #    out = gelu(out)
    elif ACTIVATION == "silu":
        out = silu(out)

    out = out.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.atomic_add(c_ptrs, out, mask=c_mask)


torch.library.Library("gpu_poor", "DEF")


@torch.library.custom_op(
    "gpu_poor::split_k_parallel",
    mutates_args={"out"},
    # device_types=("cuda",),
)
def _split_k_parallel(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    dump_ptx: bool,
    k_acc_div_max: Optional[int],
    activation: Optional[str],
) -> torch.Tensor:
    # a = a.contiguous()
    # b = b.contiguous()
    assert a.ndim == 2 and b.ndim == 2, "Input tensors must have 2 dimensions"
    assert a.shape[-1] == b.shape[-2], f"Incompatible dimensions: {a.shape} and {b.shape}"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"

    batch_dims, K = a.shape[:-1], a.shape[-1]
    M = batch_dims.numel()
    N = b.shape[-1]

    # Dimension is smaller than the minimum block size, fallback to torch.matmul
    if M < M_MIN or N < N_MIN:
        return torch.matmul(a, b, out=out)

    a = a.reshape(M, K)

    if out is None:
        out = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    else:
        raise NotImplementedError("Output tensor is not supported yet")

    if k_acc_div_max is None:
        k_acc_div_max = triton.next_power_of_2(int((K // 32) ** 0.5))

    def grid(meta: dict):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            k_acc_div_max,
        )

    kernel = _matmul_kernel[grid](
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
        inner_acc_dtype=tl.float16,
        SPLIT_K=k_acc_div_max,
        ACTIVATION=activation,
    )

    if dump_ptx:
        open("dump.ptx", "w").write(kernel.asm["ptx"])

    return out.reshape(*batch_dims, N)


@_split_k_parallel.register_fake
def _split_k_parallel_abstract(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    dump_ptx: bool,
    k_acc_div_max: int,
    activation: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "Input tensors must have 2 dimensions"
    m, k1 = a.shape
    k2, n = b.shape
    assert k1 == k2, f"Incompatible contraction dimensions {a.shape} and {b.shape}"
    assert a.device == b.device, "Input tensors must be on the same device"
    if bias is not None:
        assert bias.ndim == 1, "Bias tensor must have 1 dimension"
        assert bias.shape[0] == n, f"Incompatible bias dimensions {bias.shape} and {b.shape}"

    if out is not None:
        assert out.ndim == 2, "Output tensor must have 2 dimensions"
        assert out.shape == (m, n), f"Incompatible output dimensions {out.shape} and {(m, n)}"
        assert out.device == a.device, "Output tensor must be on the same device"

    return torch.empty((m, n), dtype=a.dtype, device=a.device)


@register_flop_formula(torch.ops.gpu_poor.split_k_parallel)
def split_k_parallel_flop(a_shape, b_shape, bias_shape, *args, **kwargs) -> int:
    """Count flops for matmul."""
    m, k, n = *a_shape, b_shape[-1]
    flops = 2 * m * n * (k - 1)
    if bias_shape is not None:
        flops += m * n
    return flops


flop_registry[torch.ops.gpu_poor.split_k_parallel] = split_k_parallel_flop

split_k_parallel = torch.ops.gpu_poor.split_k_parallel
