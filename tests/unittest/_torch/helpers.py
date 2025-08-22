import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
        torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax /
                                                             448.0).view(
                                                                 x_view.size(0),
                                                                 x_view.size(2))


def per_token_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n), sf


def per_block_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((align(m, 128), align(n, 128)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2))


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def calc_woq_tolerence(x: torch.Tensor, weight_dtype: torch.dtype):
    # align with woq_assert_near_eq function in tests/unittest/trt/quantization/_utils.py
    if weight_dtype == torch.int8:
        bits_in_type = 8
    elif weight_dtype == torch.quint4x2:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))
    max_val = torch.max(abs(x)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding

    return atol


def reference_moe_torch(x: torch.Tensor, selected_experts: torch.Tensor,
                        final_scales: torch.Tensor, num_experts: int,
                        weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    # cast back to the input dtype
    results = torch.zeros_like(x)

    # naive looping over experts
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        w1_weight = weights[f"{expert_id}.w1.weight"]
        w2_weight = weights[f"{expert_id}.w2.weight"]
        w3_weight = weights[f"{expert_id}.w3.weight"]
        expert_inputs = x[batch_idx]
        output = (F.silu(expert_inputs @ w1_weight.t()) *
                  (expert_inputs @ w3_weight.t())) @ w2_weight.t()
        results[batch_idx] += final_scales[batch_idx, nth_expert, None] * output

    return results.view_as(x)


def reference_block_scale_moe_torch(
        x: torch.Tensor, selected_experts: torch.Tensor,
        final_scales: torch.Tensor, num_experts: int,
        weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    results = torch.zeros_like(x)

    # naive looping over experts
    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        w1 = weights[f"{expert_id}.w1.weight"]
        w2 = weights[f"{expert_id}.w2.weight"]
        w3 = weights[f"{expert_id}.w3.weight"]

        w1_fp8, w1_scale = per_block_cast_to_fp8(w1)
        w2_fp8, w2_scale = per_block_cast_to_fp8(w2)
        w3_fp8, w3_scale = per_block_cast_to_fp8(w3)

        x_fp8, x_scale = per_token_cast_to_fp8(x[batch_idx])

        def block_scale_gemm(mat_a: torch.Tensor, mat_scale_a: torch.Tensor,
                             mat_b: torch.Tensor, mat_scale_b: torch.Tensor):
            shape_m, shape_k = mat_a.shape
            shape_n = mat_b.shape[0]
            result = torch.zeros((shape_m, shape_n), dtype=torch.float32).cuda()

            for m in range(shape_m):
                for n in range(shape_n):
                    for k in range(0, shape_k, 128):
                        scale_factor = mat_scale_a[m, k //
                                                   128] * mat_scale_b[n // 128,
                                                                      k // 128]
                        tile_a = mat_a[m, k:k + 128]
                        tile_b = mat_b[n, k:k + 128]
                        tile_d = torch.dot(tile_a.float(), tile_b.float())
                        result[
                            m,
                            n] += scale_factor.cuda() * tile_d.cuda().float()

            result_bf16 = result.bfloat16()

            return result_bf16

        # gemm1
        fc3_output = block_scale_gemm(x_fp8, x_scale, w1_fp8, w1_scale)
        gate_output = F.silu(fc3_output)
        fc1_output = block_scale_gemm(x_fp8, x_scale, w3_fp8, w3_scale)
        act_output = gate_output * fc1_output
        # gemm2
        act_fp8, act_scale = per_token_cast_to_fp8(act_output)
        output = block_scale_gemm(act_fp8, act_scale, w2_fp8, w2_scale)

        results[batch_idx] += final_scales[batch_idx, nth_expert, None] * output

    return results.view_as(x)


class MockPytorchBackendConfig:

    def __init__(self, use_cuda_graph, cuda_graph_padding_enabled):
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_padding_enabled = cuda_graph_padding_enabled


class MockSpecConfig:

    def __init__(self, max_draft_len):
        self.max_draft_len = max_draft_len


class MockEngine:
    """A replacement for SimpleNamespace that supports weak references."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CUDAGraphRunnerTester:
    """
    A testing helper class that simplifies the use of CUDAGraphRunner for
    single-batch decoding steps in a unit test environment.

    It wraps an instance of CUDAGraphRunner and provides a simplified interface
    for capturing and replaying a CUDA graph. It internally creates a mock engine
    object with the necessary attributes required by CUDAGraphRunner.
    """

    def __init__(self, batch_size: int, device: str, attn_metadata: Any):
        """
        Initializes the helper and the underlying CUDAGraphRunner.

        Args:
            batch_size: The batch size for which the CUDA graph will be captured.
            device: The torch device to use (e.g., "cuda").
            attn_metadata: An instance of attention metadata that is
                already prepared for CUDA graph capture (i.e., the output of
                `metadata.create_cuda_graph_metadata(...)`).
        """
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.attn_metadata = attn_metadata
        self.captured = False

        # 1. Create a mock engine object
        mock_engine = MockEngine(
            pytorch_backend_config=MockPytorchBackendConfig(
                use_cuda_graph=True, cuda_graph_padding_enabled=False),
            _cuda_graph_batch_sizes=[batch_size],
            _max_cuda_graph_batch_size=batch_size,
            max_beam_width=1,
            is_spec_decode=False,
            spec_config=MockSpecConfig(max_draft_len=0),
            _cuda_graph_mem_pool=None,
            use_mrope=False,
        )
        self.engine = mock_engine

        # 2. Instantiate the actual CUDAGraphRunner with the mock engine.
        self.runner = CUDAGraphRunner(self.engine)

    def capture(self, forward_fn: Callable[[Dict[str, Any]], torch.Tensor]):
        """
        Captures the CUDA graph for the provided model's forward function.

        This method prepares the initial inputs required by CUDAGraphRunner
        and calls its `capture` method. It should only be called once.

        Args:
            forward_fn: A callable (e.g., a lambda) that takes a dictionary of
                keyword arguments and executes the model's forward pass.
        """
        if self.captured:
            raise RuntimeError("Graph has already been captured.")

        # For capturing, CUDAGraphRunner needs a dictionary of initial inputs.
        # In this decoding-only test case, this dictionary primarily needs to
        # contain the graph-compatible attention metadata.
        initial_inputs = {
            "attn_metadata": self.attn_metadata,
            "spec_metadata": None,
        }

        self.runner.capture(self.batch_size, forward_fn, initial_inputs)
        self.captured = True

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Replays the previously captured CUDA graph with new dynamic inputs.

        Args:
            inputs: A dictionary containing the dynamic inputs for the model,
                    which must include 'input_ids', 'position_ids', and the
                    same 'attn_metadata' object used for capture.

        Returns:
            The output tensor from the model's forward pass.
        """
        if not self.captured:
            raise RuntimeError("Graph must be captured before it can be run.")

        output_tensor = self.runner.replay(self.batch_size, inputs)
        return output_tensor


class graph_capturing_local(threading.local):

    def __init__(self):
        self.is_graph_capturing = False


_local = graph_capturing_local()


def set_graph_capturing(enable: bool):
    _local.is_graph_capturing = enable


def is_graph_capturing() -> bool:
    return _local.is_graph_capturing


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata,
    ) -> None:
        self.batch_size = batch_size
        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones((batch_size, ),
                                    device=device,
                                    dtype=torch.int32)
        self.position_ids = torch.zeros((1, batch_size),
                                        device=device,
                                        dtype=torch.int32)

        self.attn_metadata = attn_metadata
        self._output = None
        self._graph = None

    def __del__(self):
        self._graph.reset()

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        self._graph = torch.cuda.CUDAGraph()
        inputs = {
            "attn_metadata": self.attn_metadata,
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "inputs_embeds": None,
        }

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        set_graph_capturing(True)
        for _ in range(2):
            forward_fn(inputs)
        with torch.cuda.graph(self._graph, pool=pool):
            output = forward_fn(inputs)
        set_graph_capturing(False)
        # Mark weak ref here. The output tensor should be freed properly.
        from tensorrt_llm._torch.utils import make_weak_ref
        self._output = make_weak_ref(output)
        return self._graph.pool()

    def needs_capture(self) -> bool:
        return self._output is None

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        assert "input_ids" in inputs
        assert "position_ids" in inputs
        assert "attn_metadata" in inputs

        attn_metadata = inputs["attn_metadata"]
        assert attn_metadata is self.attn_metadata, (
            "attn_metadata does not match the attn_metadata instance that was used to "
            "capture this graph.")

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        seqlen = input_ids.shape[0]
        self.input_ids[:seqlen].copy_(input_ids)
        self.position_ids[:, :seqlen].copy_(position_ids)

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output
