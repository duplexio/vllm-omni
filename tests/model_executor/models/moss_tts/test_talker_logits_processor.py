# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MOSS-TTS talker logits processing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm")
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _RecordingLogitsProcessor:
    def __init__(self, vocab_size: int = 4) -> None:
        self.args: tuple[object, ...] | None = None
        self.kwargs: dict[str, object] | None = None
        self.vocab_size = vocab_size

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.args = args
        self.kwargs = kwargs
        return torch.zeros((1, self.vocab_size))


class _ZeroEmbeddingModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], self.hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )


def test_moss_tts_delay_compute_logits_does_not_forward_sampling_metadata() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSDelayTalkerForGeneration,
    )

    model = MossTTSDelayTalkerForGeneration.__new__(MossTTSDelayTalkerForGeneration)
    nn.Module.__init__(model)
    model.text_lm_head = object()
    model.logits_processor = _RecordingLogitsProcessor()
    model._batch_state = None
    hidden_states = torch.randn(1, 4)
    sampling_metadata = object()

    logits = model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    assert logits.shape == (1, 4)
    assert model.logits_processor.args == (model.text_lm_head, hidden_states)
    assert model.logits_processor.kwargs == {}


def test_moss_tts_delay_max_frames_excludes_teacher_forced_prefix() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSDelayTalkerForGeneration,
    )

    model = MossTTSDelayTalkerForGeneration.__new__(MossTTSDelayTalkerForGeneration)
    nn.Module.__init__(model)
    model.text_lm_head = object()
    model.logits_processor = _RecordingLogitsProcessor(vocab_size=8)
    model.n_vq = 2
    model.audio_assistant_gen_slot_token_id = 1
    model.audio_assistant_delay_slot_token_id = 2
    model.im_end_token_id = 3
    model.audio_end_token_id = 4
    model.pad_token_id = 5
    model._audio_keep_text_ids = (1, 2)
    model._pre_exclude_text_ids = (5, 1, 2, 4)
    model._batch_state_spans = None
    model._batch_state = [
        {
            "audio_lengths": 900,
            "generated_audio_frames": 0,
            "delayed_lengths": -1,
            "is_audio": True,
            "step": 20,
            "max_new_frames": 640,
        }
    ]

    logits = model.compute_logits(torch.randn(1, 8))

    assert torch.isfinite(logits[0, 1:3]).all()
    assert torch.isneginf(logits[0, 3])

    model._batch_state[0]["generated_audio_frames"] = 640
    capped_logits = model.compute_logits(torch.randn(1, 8))

    assert capped_logits[0, 3] == 0
    assert torch.isneginf(capped_logits[0, :3]).all()
    assert torch.isneginf(capped_logits[0, 4:]).all()


def test_moss_tts_delay_prefill_codes_resume_at_cached_token_offset() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSDelayTalkerForGeneration,
    )

    model = MossTTSDelayTalkerForGeneration.__new__(MossTTSDelayTalkerForGeneration)
    nn.Module.__init__(model)
    model.model = _ZeroEmbeddingModel(hidden_size=1)
    model.n_vq = 2
    model.audio_vocab_size = 9
    model.audio_pad_code = 9
    model.audio_start_token_id = 7
    model.audio_assistant_gen_slot_token_id = 8
    model._stacked_audio_emb_w = torch.arange(10, dtype=torch.float32).view(1, 10, 1).expand(2, -1, -1)
    model.audio_embeddings = nn.ModuleList()
    ref_codes = torch.tensor(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
        ],
        dtype=torch.long,
    )

    _, embeds, update = model.preprocess(
        input_ids=torch.tensor([6, 8], dtype=torch.long),
        input_embeds=None,
        codes={"ref": ref_codes},
        _omni_num_computed_tokens=3,
        _omni_prompt_len=5,
    )

    torch.testing.assert_close(embeds[:, 0], torch.tensor([6.0, 8.0]))
    assert update["ref_offset"] == 5
