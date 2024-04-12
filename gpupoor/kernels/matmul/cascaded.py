from typing import Optional

import torch
import triton
import triton.language as tl


K_ACC_DIV_MIN = 1

VALUES_BLOCK_M = [128, 256]
VALUES_BLOCK_N = [128]
VALUES_BLOCK_K = [16, 32]
VALUES_GROUP_M = [8]
VALUES_NUM_STAGES = [4]
VALUES_NUM_WARPS = [8]

M_MIN, N_MIN = 512, 256
assert M_MIN >= min(VALUES_BLOCK_M)
assert N_MIN >= min(VALUES_BLOCK_N)


def calculate_k_acc_div(args: dict) -> int:
    K, BLOCK_K, K_ACC_DIV_MAX = args["K"], args["BLOCK_K"], args["K_ACC_DIV"]
    ret = triton.next_power_of_2(int((K // BLOCK_K) ** 0.5))
    return min(max(ret, K_ACC_DIV_MIN), K_ACC_DIV_MAX)


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
@triton.heuristics({"K_ACC_DIV": calculate_k_acc_div})
# @triton.heuristics({"K_IS_DIVISIBLE": lambda args: args["K"] % (args["BLOCK_K"] * args["K_ACC_DIV"]) == 0})
@triton.jit
def _matmul_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
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
    outer_acc_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    #    K_IS_DIVISIBLE: tl.constexpr,
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
        order=(1, 0),
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
    k_inner = tl.cdiv(K, BLOCK_K) // K_ACC_DIV

    for k_i in range(0, K_ACC_DIV):
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=inner_acc_dtype)
        for k in range(0, k_inner):
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(1, 0))

            accumulator = tl.dot(a, b, acc=accumulator, out_dtype=inner_acc_dtype)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        out += accumulator.to(outer_acc_dtype)

    out = out.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, out, boundary_check=(0, 1))


def cascaded(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    dump_ptx: bool = False,
    group_size: int = 256,
    inner_acc_dtype: tl.dtype = tl.float16,
    outer_acc_dtype: tl.dtype = tl.float32,
) -> torch.Tensor:
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    batch_dims, K = a.shape[:-1], a.shape[-1]
    M = batch_dims.numel()
    N = b.shape[-1]

    # Dimension is smaller than the minimum block size, fallback to torch.matmul
    if M < M_MIN or N < N_MIN:
        return torch.matmul(a, b, out=out)

    a = a.reshape(M, K)

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta: dict):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    kernel = _matmul_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        inner_acc_dtype=inner_acc_dtype,
        outer_acc_dtype=outer_acc_dtype,
        GROUP_SIZE=group_size,
    )

    if dump_ptx:
        open("dump.ptx", "w").write(kernel.asm["ptx"])

    return out.reshape(*batch_dims, N)
