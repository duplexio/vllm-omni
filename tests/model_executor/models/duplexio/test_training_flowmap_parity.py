"""Cross-repository parity; put the DuplexIO checkout on PYTHONPATH to run."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.flowmap import FlowMap

training = pytest.importorskip("duplexio.modules.flowmap")
pytestmark = [pytest.mark.core_model]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0])
def test_matches_training_forward_and_sample(steps: int, temperature: float, device: str, autocast: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    torch.manual_seed(71)
    reference = training.FlowMap(4, 16, 12, 2, inference_steps=steps).to(device)
    # Fresh FlowMap initialization predicts zero velocity. Exercise learned,
    # nonzero modulation and velocity, not that degenerate identity map.
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, 0.1)
    native = FlowMap(4, 16, 12, 2, inference_steps=steps, sampling_temperature=temperature).to(device)
    state = reference.state_dict()
    del state["log_precision"]  # Training-only loss weighting.
    native.load_state_dict(state, strict=True)

    conditioning = torch.randn(5, 12, device=device)
    x = torch.randn(5, 4, device=device)
    s, t = torch.rand(5, device=device), torch.rand(5, device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        torch.testing.assert_close(native(x, conditioning, s, t), reference(x, conditioning, s, t), atol=0, rtol=0)
        torch.manual_seed(32)
        expected = reference.sample(conditioning, temperature=temperature)
        torch.manual_seed(32)
        noise = torch.randn(5, 4, device=device)
        torch.testing.assert_close(native.sample(conditioning, noise), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("batch_size", [1, 8, 32, 257])
@torch.inference_mode()
def test_fixed_conditioning_and_noise_match_training_exactly(batch_size: int) -> None:
    torch.manual_seed(71)
    reference = training.FlowMap(32, 512, 2560, 6).cuda()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, 0.03)
    native = FlowMap(32, 512, 2560, 6, sampling_temperature=0.3).cuda()
    state = reference.state_dict()
    del state["log_precision"]
    native.load_state_dict(state, strict=True)
    conditioning = torch.randn(batch_size, 2560, device="cuda")
    noise = torch.randn(batch_size, 32, device="cuda")
    start = torch.zeros(batch_size, device="cuda")
    end = torch.ones(batch_size, device="cuda")
    initial = 0.3**0.5 * noise
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = initial + reference(initial, conditioning, start, end)
        actual = native.sample(conditioning, noise)
        individual = torch.cat([
            native.sample(conditioning[row : row + 1], noise[row : row + 1])
            for row in range(batch_size)
        ])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, individual, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@torch.inference_mode()
def test_compiled_fp32_sampling_matches_eager_after_slot_replacement() -> None:
    native = FlowMap(32, 512, 2560, 6).cuda()
    conditioning = torch.randn(8, 2560, device="cuda")
    noise = torch.randn(8, 32, device="cuda")
    compiled = torch.compile(native.sample, fullgraph=True, mode="reduce-overhead")
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        expected = native.sample(conditioning, noise)
        torch.testing.assert_close(compiled(conditioning, noise), expected, atol=2e-5, rtol=2e-4)
        conditioning[3].normal_()
        noise[3].normal_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("autocast", [False, True])
@torch.inference_mode()
def test_cuda_graph_replays_exact_eager_kernels(autocast: bool) -> None:
    native = FlowMap(32, 512, 2560, 6).cuda()
    conditioning = torch.randn(8, 2560, device="cuda")
    noise = torch.randn(8, 32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast, cache_enabled=False):
        for _ in range(3):
            native.sample(conditioning, noise)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast, cache_enabled=False):
        captured = native.sample(conditioning, noise)
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            expected = native.sample(conditioning, noise)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        conditioning[3].normal_()
        noise[3].normal_()
