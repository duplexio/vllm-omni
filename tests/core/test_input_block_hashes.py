"""KV reuse must identify the entire text-and-audio prefix."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import init_none_hash

from vllm_omni.core.input_block_hashes import get_omni_request_block_hasher
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
HASH_FN = get_hash_fn_by_name("sha256")
init_none_hash(HASH_FN)


def make_request(request_id: str, audio: torch.Tensor, salt: str = "speaker-a") -> OmniRequest:
    return OmniRequest(
        request_id=request_id,
        prompt_token_ids=[1] * len(audio),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        cache_salt=salt,
        additional_information=serialize_additional_information({"prefix_cache_input_ids": audio}),
        block_hasher=get_omni_request_block_hasher(2, HASH_FN),
    )


def test_audio_change_invalidates_that_block_and_all_later_blocks() -> None:
    audio = torch.arange(12).reshape(6, 2)
    changed = audio.clone()
    changed[2, 0] += 1
    original = make_request("original", audio)
    different = make_request("different", changed)
    assert original.block_hashes[0] == different.block_hashes[0]
    assert original.block_hashes[1] != different.block_hashes[1]
    assert original.block_hashes[2] != different.block_hashes[2]


def test_longer_prompt_reuses_identical_audio_prefix() -> None:
    audio = torch.arange(12).reshape(6, 2)
    short = make_request("short", audio[:4])
    long = make_request("long", audio)
    assert short.block_hashes == long.block_hashes[:2]


def test_speakers_cannot_share_kv_with_different_cache_namespaces() -> None:
    audio = torch.arange(8).reshape(4, 2)
    user = make_request("user", audio, "user")
    agent = make_request("agent", audio, "assistant")
    assert all(a != b for a, b in zip(user.block_hashes, agent.block_hashes))


def test_generated_audio_without_explicit_inputs_is_not_reused() -> None:
    audio = torch.arange(8).reshape(4, 2)
    first = make_request("first", audio)
    second = make_request("second", audio)
    first.append_output_token_ids([1, 1])
    second.append_output_token_ids([1, 1])
    assert first.block_hashes[:2] == second.block_hashes[:2]
    assert first.block_hashes[2] != second.block_hashes[2]
