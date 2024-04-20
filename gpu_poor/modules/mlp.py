import torch
import torch.nn as nn


from gpu_poor.modules.lowp_linear import LowPrecisionLinear


if __name__ == "__main__":

    def seed(k: int = 0):
        torch.manual_seed(k)
        torch.cuda.manual_seed_all(k)

    dim = 4096
    intermediate_size = 14336

    bsz = 8
    seq_len = 512

    with torch.device("cuda"):
        seed(42)
        mlp_orig = nn.Linear(dim, intermediate_size).half()

        seed(42)
        mlp_lowp = LowPrecisionLinear(dim, intermediate_size).half()

        seed(69)
        x_orig = torch.randn(bsz, seq_len, dim, dtype=torch.float16, requires_grad=True)
        x_lowp = x_orig.clone()

        y_orig = mlp_orig(x_orig)
        y_lowp = mlp_lowp(x_lowp)

        loss_orig = y_orig.sum()
        loss_lowp = y_lowp.sum()

        loss_orig.backward()
        loss_lowp.backward()

        for (name, p_orig), (_, p_lowp) in zip(mlp_orig.named_parameters(), mlp_lowp.named_parameters()):
            # print(name, torch.allclose(p_orig.grad, p_lowp.grad))#, atol=1e-2, rtol=1e-3))
            print(name)
            delta = p_orig.grad - p_lowp.grad
            print(f"L1         :: {delta.abs().mean()}")
            print(f"L2         :: {delta.square().mean().sqrt()}")
            print(f"DIFF MAX   :: {delta.abs().max()}")
