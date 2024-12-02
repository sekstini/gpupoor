import pytest
import torch
import torch.nn.functional as F

from gpu_poor.kernels import _split_k_sequential, split_k_sequential


def get_matmul_functions():
    """Returns a list of matmul implementations to test."""

    return [
        ("split_k_sequential", split_k_sequential),
        ("torch_matmul", lambda x, w, b=None, **kwargs: F.linear(x, w.mT, b)),
    ]


class TestMatmul:
    @pytest.fixture(params=get_matmul_functions())
    def matmul_fn(self, request):
        return request.param

    @pytest.fixture
    def device(self):
        return "cuda"

    def _compare_outputs(self, actual: torch.Tensor, expected: torch.Tensor, rtol=1e-3, atol=1e-1):
        """Compare two tensors with relaxed tolerances for GPU float16 computations."""
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((32, 64), (64, 32)),  # Basic matrix multiplication
            ((1, 64), (64, 32)),  # Batch size 1
            ((128, 64), (64, 256)),  # Larger matrices
            ((256, 512), (512, 128)),  # Even larger matrices
            ((2, 32, 64), (64, 32)),  # Batched input
            ((3, 4, 32, 64), (64, 32)),  # Multi-batch input
        ],
    )
    def test_shapes(self, matmul_fn, device, shape1, shape2):
        name, fn = matmul_fn
        x = torch.randn(*shape1, device=device, dtype=torch.float16)
        w = torch.randn(*shape2, device=device, dtype=torch.float16)

        actual = fn(x, w)
        expected = F.linear(x.reshape(-1, shape1[-1]), w.T).reshape(*x.shape[:-1], shape2[-1])
        self._compare_outputs(actual, expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_dtypes(self, matmul_fn, device, dtype):
        name, fn = matmul_fn
        x = torch.randn(32, 64, device=device, dtype=dtype)
        w = torch.randn(64, 32, device=device, dtype=dtype)

        actual = fn(x, w)
        expected = F.linear(x, w.T)
        self._compare_outputs(actual, expected)

    def test_bias_addition(self, matmul_fn, device):
        name, fn = matmul_fn
        x = torch.randn(32, 64, device=device, dtype=torch.float16)
        w = torch.randn(64, 32, device=device, dtype=torch.float16)
        bias = torch.randn(32, device=device, dtype=torch.float16)

        actual = fn(x, w, bias)
        expected = F.linear(x, w.T, bias)
        self._compare_outputs(actual, expected)

    def test_backward(self, matmul_fn, device):
        name, fn = matmul_fn
        x = torch.randn(32, 64, device=device, dtype=torch.float16, requires_grad=True)
        w = torch.randn(64, 32, device=device, dtype=torch.float16, requires_grad=True)
        bias = torch.randn(32, device=device, dtype=torch.float16, requires_grad=True)

        # Forward pass
        output = fn(x, w, bias)
        loss = output.sum()
        loss.backward()

        # Compare with PyTorch's gradients
        x_torch = x.detach().clone().requires_grad_()
        w_torch = w.detach().clone().requires_grad_()
        bias_torch = bias.detach().clone().requires_grad_()

        output_torch = F.linear(x_torch, w_torch.T, bias_torch)
        loss_torch = output_torch.sum()
        loss_torch.backward()

        self._compare_outputs(x.grad, x_torch.grad)
        self._compare_outputs(w.grad, w_torch.grad)
        self._compare_outputs(bias.grad, bias_torch.grad)

    def test_zero_size_dimension(self, matmul_fn, device):
        name, fn = matmul_fn
        x = torch.randn(0, 64, device=device, dtype=torch.float16)
        w = torch.randn(64, 32, device=device, dtype=torch.float16)

        actual = fn(x, w)
        expected = F.linear(x, w.T)
        self._compare_outputs(actual, expected)

    @pytest.mark.parametrize(
        "input_shape, weight_shape",
        [
            ((32, 64), (32, 64)),  # Mismatched inner dimensions
            ((32, 64), (64, 32, 16)),  # Wrong number of dimensions
        ],
    )
    def test_invalid_shapes(self, matmul_fn, device, input_shape, weight_shape):
        name, fn = matmul_fn
        x = torch.randn(*input_shape, device=device, dtype=torch.float16)
        w = torch.randn(*weight_shape, device=device, dtype=torch.float16)

        with pytest.raises(Exception):
            fn(x, w)

    def test_deterministic(self, matmul_fn, device):
        """Test that the function produces the same output given the same input."""
        name, fn = matmul_fn
        x = torch.randn(32, 64, device=device, dtype=torch.float16)
        w = torch.randn(64, 32, device=device, dtype=torch.float16)

        result1 = fn(x, w)
        result2 = fn(x, w)
        self._compare_outputs(result1, result2, rtol=0, atol=0)  # Exact match expected


class TestSplitKSequentialOpcheck:
    def test_opcheck_validation(self):
        """
        Tests the split_k_sequential operation using PyTorch's operator checking functionality.
        This validates both forward and backward pass correctness, gradient computation,
        and ensures the operation adheres to PyTorch's operator requirements.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        with torch.device("cuda"):
            test_cases = [
                {
                    "input": torch.randn(512, 1024, dtype=torch.float16),
                    "weight": torch.randn(512, 1024, dtype=torch.float16),
                    "requires_grad": False,
                    "description": "Forward pass without gradients",
                },
                {
                    "input": torch.randn(512, 1024, dtype=torch.float16, requires_grad=True),
                    "weight": torch.randn(512, 1024, dtype=torch.float16, requires_grad=True),
                    "requires_grad": True,
                    "description": "Forward and backward pass with gradients",
                },
            ]

            for case in test_cases:
                # Prepare inputs
                x = case["input"]
                weight = case["weight"]

                # Run opcheck
                try:
                    torch.library.opcheck(_split_k_sequential, (x, weight.T, None, 32))
                except Exception as e:
                    pytest.fail(f"Opcheck failed for {case['description']}: {str(e)}")

    @pytest.mark.parametrize(
        "batch_size,in_features,out_features",
        [
            (512, 1024, 512),
            (256, 2048, 1024),
            (1024, 512, 256),
        ],
    )
    def test_opcheck_different_sizes(self, batch_size, in_features, out_features):
        """
        Tests the split_k_sequential operation with different matrix sizes.
        This ensures the operation works correctly across various practical dimensions.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        with torch.device("cuda"):
            x = torch.randn(batch_size, in_features, dtype=torch.float16, requires_grad=True)
            weight = torch.randn(out_features, in_features, dtype=torch.float16, requires_grad=True)

            try:
                torch.library.opcheck(_split_k_sequential, (x, weight.T, None, 32))
            except Exception as e:
                pytest.fail(
                    f"Opcheck failed for size (batch_size={batch_size}, "
                    f"in_features={in_features}, out_features={out_features}): {str(e)}"
                )


if __name__ == "__main__":
    pytest.main([__file__])
