"""Numerical and batching contracts for the native continuous audio head."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.flowmap import FlowMap, PocketRMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_flow(*, steps: int = 1, temperature: float = 0.3) -> FlowMap:
    torch.manual_seed(71)
    return FlowMap(4, 16, 12, 2, inference_steps=steps, sampling_temperature=temperature)


def test_time_normalization_uses_sample_variance() -> None:
    hidden = torch.tensor([[1.0, 2.0, 5.0, 9.0]])
    expected = hidden / torch.sqrt(hidden.var(dim=-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(PocketRMSNorm(4)(hidden), expected)


def test_time_normalization_preserves_bf16_activations() -> None:
    norm = PocketRMSNorm(4)
    hidden = torch.tensor([[1.0, 2.0, 5.0, 9.0]], dtype=torch.bfloat16)
    expected = hidden * torch.rsqrt(hidden.var(dim=-1, keepdim=True) + 1e-5)
    actual = norm(hidden)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_flow_batch_slots_are_independent(steps: int) -> None:
    flow = make_flow(steps=steps)
    conditioning, noise = torch.randn(5, 12), torch.randn(5, 4)
    batched = flow.sample(conditioning, noise)
    individual = torch.cat([flow.sample(conditioning[i : i + 1], noise[i : i + 1]) for i in range(5)])
    torch.testing.assert_close(batched, individual, atol=2e-6, rtol=2e-5)

    order = torch.tensor([4, 0, 2, 1, 3])
    torch.testing.assert_close(flow.sample(conditioning[order], noise[order]), batched[order])
    conditioning[1] = 100
    noise[1] = -100
    unchanged = torch.tensor([0, 2, 3, 4])
    torch.testing.assert_close(flow.sample(conditioning, noise)[unchanged], batched[unchanged])


@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0])
def test_temperature_is_noise_variance(temperature: float) -> None:
    flow = make_flow(temperature=temperature)
    with torch.no_grad():
        flow.final_layer.linear.weight.zero_()
        flow.final_layer.linear.bias.zero_()
    noise = torch.randn(3, 4)
    torch.testing.assert_close(flow.sample(torch.randn(3, 12), noise), temperature**0.5 * noise)


@pytest.mark.parametrize("steps", [1, 3])
def test_sampling_integrates_from_zero_to_one(steps: int) -> None:
    flow = make_flow(steps=steps)
    conditioning, noise = torch.randn(3, 12), torch.randn(3, 4)
    expected = 0.3**0.5 * noise
    for step in range(steps):
        expected = (
            expected
            + flow(
                expected,
                conditioning,
                torch.full((3,), step / steps),
                torch.full((3,), (step + 1) / steps),
            )
            / steps
        )
    torch.testing.assert_close(flow.sample(conditioning, noise), expected)


def test_sampling_captures_as_one_graph_without_random_state() -> None:
    flow = make_flow(steps=2)
    conditioning, noise = torch.randn(3, 12), torch.randn(3, 4)
    random_state = torch.random.get_rng_state()
    compiled = torch.compile(flow.sample, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(conditioning, noise), flow.sample(conditioning, noise))
    torch.testing.assert_close(torch.random.get_rng_state(), random_state)


def test_flowmap_keeps_fp32_parameters_under_bf16_model_initialization() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        flow = make_flow()
    finally:
        torch.set_default_dtype(previous)
    assert all(parameter.dtype == torch.float32 for parameter in flow.parameters())
    assert flow.start_time_embedding.frequencies.dtype == torch.float32
