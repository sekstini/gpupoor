from functools import partial

import torch
import triton
import triton.language as tl

import gpupoor.kernels.matmul as mm


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
        ],
        # Label name for the lines
        line_names=[
            "cuBLAS (fp32 acc)",
            "Triton (fp16 acc, kdiv=16)",
            "Triton (fp16 acc, kdiv=4)",
            "Triton (fp16 acc, kdiv=1)",
        ],
        # Line styles
        styles=[
            ("green", "-"),
            ("blue", "-"),
            ("red", "-"),
            ("black", "-"),
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
        k_acc_div_max = int(provider.split("_")[-1])
        ms, min_ms, max_ms = do_bench(lambda: mm.split_k_sequential(a, b, k_acc_div_max=k_acc_div_max))
    else:
        raise

    def ms_to_tflops(ms: float) -> float:
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return ms_to_tflops(ms), ms_to_tflops(max_ms), ms_to_tflops(min_ms)


benchmark.run(save_path="benchmarks/triton", print_data=True)
