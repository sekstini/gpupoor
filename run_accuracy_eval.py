from collections import defaultdict
import torch
import fire
import torch.nn.functional as F
import triton.language as tl
import numpy as np
import matplotlib.pyplot as plt

import gpupoor.kernels.matmul as mm

try:
    import gemm_streamk_transpose
except:
    pass

try:
    import gemm_splitk_transpose
except:
    pass


def to_col_major(x: torch.Tensor) -> torch.Tensor:
    K, N = x.shape
    x = x.T.contiguous()
    x = x.set_(x.untyped_storage(), x.storage_offset(), (K, N), (N, 1))
    return x


EPS: float = 1e-6
M_MIN: int = 512
N_MIN: int = 256

KERNEL_MAP = {
    "split_k_sequential": mm.split_k_sequential,
    # "split_k_parallel": mm.split_k_parallel,
    # "cascaded": mm.cascaded,
    "cutlass_stream_k": gemm_streamk_transpose.run if gemm_streamk_transpose else None,
    "cutlass_split_k": gemm_splitk_transpose.run if gemm_splitk_transpose else None,
}


def get_rand_fn(name: str):
    assert name in ["randn", "rand"], f"Invalid rand_fn: {name}"
    return getattr(torch, name)


def main(
    kernel_name: str,
    metric: str = "delta",
    plot_type: str = "box",
    rand_fn: str = "randn",
    M: int = 1,
    N: int = 4096,
    K: int = 14336,
    seed: int | None = None,
    # split k sequential options
    k_acc_div_max: list[int] = [1, 2, 4, 8, 16, 32],
    # cascaded options
    group_size: list[int] = [32, 128, 512],
):
    kernel = KERNEL_MAP[kernel_name]
    assert kernel is not None, f"Kernel {kernel_name} failed to load."

    if seed is not None:
        torch.manual_seed(seed)

    M_, N_ = max(M, M_MIN), max(N, N_MIN)

    _rand_fn = get_rand_fn(rand_fn)
    A = _rand_fn(M_, K, device="cuda", dtype=torch.float16)
    B = _rand_fn(K, N_, device="cuda", dtype=torch.float16)
    fp64_out = A.double() @ B.double()

    print(f"{kernel_name} :: {M}x{N}x{K}")
    fp16_outs = []
    match kernel_name:
        case "split_k_sequential":
            for kdiv in k_acc_div_max:
                kwargs = dict(k_acc_div_max=kdiv, bias=None, out=None, activation=None, dump_ptx=False)
                print(f"\t{kdiv=}")
                fp16_outs.append(kernel(A, B, **kwargs))

        case "cascaded":
            for grp_sz in group_size:
                kwargs = dict(grp_sz=grp_sz)
                print(f"\t{grp_sz=}")
                fp16_outs.append(kernel(A, B, **kwargs))

        case "cutlass_stream_k":
            kwargs = dict()
            fp16_outs.append(kernel(A, to_col_major(B), **kwargs))

        case "cutlass_split_k":
            for kdiv in k_acc_div_max:
                kwargs = dict(split_k_slices=kdiv)
                fp16_outs.append(kernel(A, to_col_major(B), **kwargs))

    fp32_out = A @ B

    fp16_outs = [out[:M].to("cpu", torch.float64) for out in fp16_outs]
    fp32_out = fp32_out[:M].to("cpu", torch.float64)
    fp64_out = fp64_out[:M].to("cpu", torch.float64)

    samples = []
    match kernel_name:
        case "split_k_sequential":
            samples += [(f"fp16 ({kdiv=})", out) for kdiv, out in zip(k_acc_div_max, fp16_outs)]
        case "cascaded":
            samples += [(f"fp16 ({grp_sz=})", out) for grp_sz, out in zip(group_size, fp16_outs)]
        case "cutlass_stream_k":
            samples += [("fp16 (streamK)", fp16_outs[0])]
        case "cutlass_split_k":
            samples += [(f"fp16 (splitK={kdiv})", out) for kdiv, out in zip(k_acc_div_max, fp16_outs)]
    samples.append(("fp32", fp32_out))

    metrics = defaultdict(dict)
    for acc_type, out in samples:
        delta = out - fp64_out
        rel_err = delta.div(fp64_out)
        rel_err[torch.isnan(rel_err)] = 0
        rel_err.abs_()

        metrics[acc_type]["delta"] = delta.ravel()
        metrics[acc_type]["rel_err"] = rel_err.ravel()

        # l1, mse = F.l1_loss(fp64_out, out), F.mse_loss(fp64_out, out)
        # metrics[acc_type]["l1"] = l1.item()
        # metrics[acc_type]["mse"] = mse.item()

    labels = list(metrics.keys())

    plt.figure(figsize=(12, 9))
    plt.title(f"{kernel_name} - ({M}, {K}) x ({K}, {N})")
    plt.ylabel(f"{metric} vs fp64")

    match metric:
        case "delta":
            pass
            # plt.yscale("symlog", base=2)
            # yticks = np.geomspace(2**-3, 2**3, num=7)
            # plt.yticks(np.concatenate((-yticks[::-1], [0], yticks)))
        case "rel_err":
            plt.yscale("log")

    match plot_type:
        case "violin":
            plt.violinplot(
                [metrics[acc_type][metric] for acc_type in labels],
                showmedians=True,
            )
        case "box":
            plt.boxplot(
                [metrics[acc_type][metric] for acc_type in labels],
                # showfliers=False,
                showcaps=True,
                autorange=True,
            )

    plt.xticks(range(1, len(labels) + 1), labels)

    plt.grid(alpha=0.25)
    plt.tight_layout()
    filename = f"{kernel_name}__error_{M}x{N}x{K}.png"
    plt.savefig(filename, dpi=200)
    print(f"Saved {filename}")


if __name__ == "__main__":
    fire.Fire(main)
