"""Fused frame construction preserves prefix masks and running text ordinals."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.modeling_duplexio import frame_inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_compiled_frame_inputs_preserve_layout() -> None:
    torch.manual_seed(43)
    compiled = torch.compile(
        frame_inputs, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True},
    )
    for frames, live in ((1, True), (7, False), (7, True), (1, False)):
        ids = torch.randint(0, 8, (frames, 4), device="cuda")
        text = torch.randn(frames, 4, 128, device="cuda", dtype=torch.bfloat16)
        channels = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        user = torch.randn(frames, 128, device="cuda", dtype=torch.bfloat16)
        agent = torch.randn_like(user)
        for start in (0, 1, 9, 1057):
            # Audio time is frozen on a text-only append, which `live` selects.
            arguments = (ids, text, channels, user, agent, 0, 1, start, start // 4, 3, live)
            expected = frame_inputs(*arguments)
            actual = compiled(*arguments)
            for output, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
            ordinals = []
            counter = start
            for row in ids.tolist():
                for token in row:
                    counter += token not in (0, 1)
                    ordinals.append(counter if token not in (0, 1) else 0)
                ordinals.extend((0, 0))
            assert actual[2].tolist() == ordinals
            assert actual[1].view(frames, 6)[:, 4:].eq(live).all()
