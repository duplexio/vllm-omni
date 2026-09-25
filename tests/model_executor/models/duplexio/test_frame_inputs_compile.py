"""Fused frame construction preserves prefix masks and running persistent ordinals."""

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
    for frames, live, prompt_count in (
        (1, True, 0),
        (7, False, 0),
        (7, True, 0),
        (1, False, 0),
        (4, False, 4),
        (7, False, 3),
    ):
        ids = torch.randint(0, 8, (frames, 4), device="cuda")
        text = torch.randn(frames, 4, 128, device="cuda", dtype=torch.bfloat16)
        channels = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        user = torch.randn(frames, 128, device="cuda", dtype=torch.bfloat16)
        agent = torch.randn_like(user)
        prompt = torch.arange(frames, device="cuda") < prompt_count
        for start in (0, 1, 9, 1057):
            metadata = torch.tensor([
                (start, start // 4 + (row if live else 0), live, row < prompt_count, row)
                for row in range(frames)
            ], dtype=torch.int32, device="cuda")
            # Audio time is frozen on a text-only append, which `live` selects.
            arguments = (
                ids, text, channels, user, agent, 0, 1, metadata, 3,
            )
            expected = frame_inputs(*arguments)
            actual = compiled(*arguments)
            for output, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
            # Voice-prompt keys and emitted text share one dense ordinal.
            ordinals, last = [], []
            counter = start
            for row, is_prompt in zip(ids.tolist(), prompt.tolist(), strict=True):
                last.extend([counter] * 6)
                for token in row:
                    counter += token not in (0, 1)
                    ordinals.append(counter if token not in (0, 1) else 0)
                counter += is_prompt
                ordinals.extend((0, counter if is_prompt else 0))
            assert actual[2].tolist() == ordinals
            # A row sees only the keys written before it.
            assert actual[3].tolist() == last
            assert actual[1].view(frames, 6)[:, 4].eq(live).all()
            assert actual[1].view(frames, 6)[:, 5].eq(prompt | live).all()
