from functools import partial

import torch
import triton

import gpupoor.kernels.matmul as mm

try:
    import gemm_streamk_transpose
except:
    gemm_streamk_transpose = None

try:
    import gemm_splitk_transpose
except:
    gemm_splitk_transpose = None

def to_col_major(x: torch.Tensor) -> torch.Tensor:
    K, N = x.shape
    x = x.T.contiguous()
    x = x.set_(x.untyped_storage(), x.storage_offset(), (K, N), (N, 1))
    return x

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[256 * i for i in range(1, 32 + 1, 2)],  # Different possible values for `x_name`
        # x_vals=[4096, 6144, 8192],
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=[
            "cublas",
            "triton_16",
            "triton_4",
            "triton_1",
            *([
            "cutlass_splitK_16",
            "cutlass_splitK_4",
            "cutlass_splitK_1",
            ] * (gemm_splitk_transpose is not None)),
            *(["cutlass_streamK"] * (gemm_streamk_transpose is not None)),
        ],
        # Label name for the lines
        line_names=[
            "cuBLAS (fp32 acc)",
            "Triton (fp16 acc, kdiv=32)",
            "Triton (fp16 acc, kdiv=8)",
            "Triton (fp16 acc, kdiv=1)",
            *([
            "CUTLASS (fp16 acc, splitK=32)",
            "CUTLASS (fp16 acc, splitK=8)",
            "CUTLASS (fp16 acc, splitK=1)",
            ] * (gemm_splitk_transpose is not None)),
            *(["CUTLASS (fp16 acc, streamK)"] * (gemm_streamk_transpose is not None)),
        ],
        # Line styles
        styles=[
            ("green", "--"),

            ("orange", "-"),
            ("purple", "-"),
            ("blue", "-"),

            *([
            ("orange", ":"),
            ("purple", ":"),
            ("blue", ":"),

            ] * (gemm_splitk_transpose is not None)),

            *([("red", ":")] * (gemm_streamk_transpose is not None)),
        ],
        ylabel="TFLOP/s",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
        xlabel="Matrix size (M = N = K)",
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    do_bench = partial(
        triton.testing.do_bench,
        quantiles=[0.5, 0.1, 0.9],
        return_mode="min",
    )

    if provider == "cublas":
        ms, min_ms, max_ms = do_bench(lambda: torch.matmul(a, b))
    elif provider.startswith("triton"):
        k_acc_div_max = provider.split("_")[-1]
        k_acc_div_max = int(k_acc_div_max)

        ms, min_ms, max_ms = do_bench(
            lambda: mm.split_k_sequential(
                a,
                b,
                bias=None,
                out=None,
                dump_ptx=False,
                k_acc_div_max=k_acc_div_max,
                activation=None,
            )
        )
    elif provider.startswith("cutlass"):
        split_k_slices = provider.split("_")[-1]
        b_col_major = to_col_major(b)

        if split_k_slices == "streamK":
            ms, min_ms, max_ms = do_bench(lambda: gemm_streamk_transpose.run(a, b_col_major))
        else:
            split_k_slices = int(split_k_slices)
            ms, min_ms, max_ms = do_bench(
                lambda: gemm_splitk_transpose.run(a, b_col_major, split_k_slices=split_k_slices)
            )
    else:
        raise

    def ms_to_tflops(ms: float) -> float:
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return ms_to_tflops(ms), ms_to_tflops(max_ms), ms_to_tflops(min_ms)


benchmark.run(save_path="benchmarks/triton", print_data=True)
