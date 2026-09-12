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
    for frames, live, prompt in (
        (1, True, False),
        (7, False, False),
        (7, True, False),
        (1, False, False),
        # A pinned voice-prompt burst: not live, but its audio cells are keys.
        (4, False, True),
    ):
        ids = torch.randint(0, 8, (frames, 4), device="cuda")
        text = torch.randn(frames, 4, 128, device="cuda", dtype=torch.bfloat16)
        channels = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        user = torch.randn(frames, 128, device="cuda", dtype=torch.bfloat16)
        agent = torch.randn_like(user)
        for start in (0, 1, 9, 1057):
            # Audio time is frozen on a text-only append, which `live` selects.
            arguments = (
                ids, text, channels, user, agent, 0, 1, start, start // 4, 3, live,
                0, prompt,
            )
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
            assert actual[1].view(frames, 6)[:, 4:].eq(live or prompt).all()
            prompt_ordinal = actual[6].view(frames, 6)
            assert not prompt_ordinal[:, :4].any()
            expected_ordinals = (
                torch.arange(1, frames + 1, device="cuda", dtype=torch.int32)
                if prompt
                else torch.zeros(frames, device="cuda", dtype=torch.int32)
            )
            assert prompt_ordinal[:, 4:].eq(expected_ordinals[:, None]).all()
            # A prompt row sees only the prompt frames before it.
            assert actual[7].view(frames, 6).eq(
                (expected_ordinals - 1 if prompt else expected_ordinals)[:, None]
            ).all()
