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
