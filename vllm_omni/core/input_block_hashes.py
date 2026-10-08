"""Hash extra per-position input IDs alongside the language-model token IDs."""

from collections.abc import Callable
from typing import Any

import numpy as np
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    get_request_block_hasher,
    hash_block_tokens,
)

from vllm_omni.request import OmniRequest


def get_omni_request_block_hasher(
    block_size: int,
    hash_fn: Callable[[Any], bytes],
) -> Callable[[OmniRequest], list[BlockHash]]:
    """Include all conditioning columns in each chained KV-cache block hash.

    Unknown generated inputs are request-specific: they cannot be reused by
    another request until supplied explicitly as a completed audio prefix.
    """
    text_hasher = get_request_block_hasher(block_size, hash_fn)

    def hash_request(request: OmniRequest) -> list[BlockHash]:
        inputs = request.prefix_cache_input_ids
        if inputs is None:
            return text_hasher(request)
        assert inputs.tensor_shape is not None
        assert inputs.tensor_data is not None
        assert inputs.tensor_dtype is not None
        assert inputs.tensor_shape[0] == request.num_prompt_tokens
        row_bytes = len(inputs.tensor_data) // inputs.tensor_shape[0]
        assert row_bytes == np.prod(inputs.tensor_shape[1:]) * np.dtype(inputs.tensor_dtype).itemsize
        start = len(request.block_hashes) * block_size
        parent = request.block_hashes[-1] if request.block_hashes else None
        hashes: list[BlockHash] = []
        mm_index = -1 if start else 0
        while start + block_size <= request.num_tokens:
            end = start + block_size
            extra_keys, mm_index = generate_block_hash_extra_keys(request, start, end, mm_index)
            if end <= request.num_prompt_tokens:
                input_key = (
                    inputs.tensor_dtype,
                    tuple(inputs.tensor_shape[1:]),
                    inputs.tensor_data[start * row_bytes : end * row_bytes],
                )
            else:
                input_key = (request.request_id, start)
            parent = hash_block_tokens(
                hash_fn, parent, request.all_token_ids[start:end],
                (*extra_keys, input_key) if extra_keys else (input_key,),
            )
            hashes.append(parent)
            start = end
        return hashes

    return hash_request
