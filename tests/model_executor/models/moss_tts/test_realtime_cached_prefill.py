"""A prefix-cache hit must align the remaining text with its audio rows."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSRealtimeTalkerForGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cached_prefill_uses_audio_at_computed_token_offset() -> None:
    model = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    nn.Module.__init__(model)
    model.n_vq = 2
    model.audio_vocab_size = 16
    model.audio_pad_token = 15
    model._stacked_audio_emb_w = None
    model.embed_tokens = nn.ModuleList([nn.Embedding(16, 4) for _ in range(3)])
    audio = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    text_ids = torch.tensor([9, 10])
    _, actual, update = model.preprocess(
        text_ids,
        None,
        codes={"ref": audio},
        _omni_num_computed_tokens=2,
        _omni_prompt_len=4,
    )
    expected = model._build_input_embeds(text_ids, audio[2:].clone())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(update["audio_codes"]["current"], audio[-1])
    assert update["ref_offset"] == 4
