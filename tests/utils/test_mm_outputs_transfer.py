import pytest
import torch

from vllm_omni.utils.mm_outputs import build_mm_cpu


def test_cpu_payload_preserves_nested_values():
    source = torch.arange(12).view(3, 4).T
    result = build_mm_cpu({"nested": {"values": [source, 7]}, "empty": []})
    torch.testing.assert_close(result["nested"]["values"][0], source)
    assert result["nested"]["values"][0].is_contiguous()
    assert result["nested"]["values"][1] == 7
    assert result["empty"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_payload_is_ready_and_independent_on_return():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.arange(96, device="cuda").view(8, 12).T
        expected = torch.arange(96).view(8, 12).T
        result = build_mm_cpu({"audio": [source, source[:, :0]], "meta": {"ids": source[0]}})
        source.fill_(-1)
    torch.testing.assert_close(result["audio"][0], expected, rtol=0, atol=0)
    torch.testing.assert_close(result["meta"]["ids"], expected[0], rtol=0, atol=0)
    assert result["audio"][0].is_contiguous()
    assert result["audio"][1].shape == (12, 0)
    assert result["audio"][0].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(1,), (3, 4), (0,)])
def test_uniform_cuda_list_preserves_rows_and_snapshot(shape):
    expected = [torch.full(shape, index, dtype=torch.float32) for index in range(4)]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        sources = [value.cuda() for value in expected]
        result = build_mm_cpu({"nested": {"values": sources}})
        for source in sources:
            source.fill_(-1)
    values = result["nested"]["values"]
    assert len(values) == len(expected)
    for actual, reference in zip(values, expected, strict=True):
        assert actual.device.type == "cpu"
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if values[0].numel():
        values[0].fill_(-2)
        torch.testing.assert_close(values[1], expected[1], rtol=0, atol=0)
