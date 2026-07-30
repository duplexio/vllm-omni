# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

import torch
from transformers import MimiConfig as TransformersMimiConfig
from transformers import MimiModel as TransformersMimiModel

from vllm_omni.model_executor.models.duplexio.mimi import (
    MimiConv1dState,
    MimiConvTranspose1dState,
    MimiModel,
    MimiStreamingState,
)


def _transformers_model() -> TransformersMimiModel:
    config = TransformersMimiConfig(
        sampling_rate=16,
        audio_channels=1,
        hidden_size=8,
        num_filters=2,
        num_residual_layers=1,
        upsampling_ratios=[2, 2],
        codebook_size=8,
        codebook_dim=4,
        num_quantizers=3,
        vector_quantization_hidden_dimension=4,
        num_semantic_quantizers=1,
        upsample_groups=8,
        num_hidden_layers=2,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=100,
        sliding_window=5,
    )
    config._attn_implementation = "eager"
    model = TransformersMimiModel(config).eval()
    for name, tensor in model.state_dict().items():
        if name.endswith("embed_sum"):
            tensor.copy_(torch.randn_like(tensor))
    return model


def test_native_mimi_strict_load_and_offline_parity() -> None:
    torch.manual_seed(3)
    reference = _transformers_model()
    native = MimiModel(reference.config.to_dict()).eval()

    native.load_state_dict(reference.state_dict(), strict=True)
    assert set(native.state_dict()) == set(reference.state_dict())

    audio = torch.randn(2, 1, 64)
    with torch.no_grad():
        reference_latent = reference.encoder(audio)
        reference_latent = reference.encoder_transformer(
            reference_latent.transpose(1, 2),
            use_cache=False,
            return_dict=False,
        )[0].transpose(1, 2)
        assert reference.downsample is not None
        reference_latent = reference.downsample(reference_latent)
        native_latent = native.encode_latent(audio)
        reference_codes = reference.quantizer.encode(
            reference_latent,
            3,
        ).transpose(0, 1)
        native_codes = native.encode(audio, 3)
        reference_audio = reference.decode(
            reference_codes,
            return_dict=False,
        )[0]
        native_audio = native.decode(native_codes)

    torch.testing.assert_close(native_latent, reference_latent)
    torch.testing.assert_close(native_codes, reference_codes)
    torch.testing.assert_close(native_audio, reference_audio)


def test_native_mimi_streaming_matches_whole_sequence() -> None:
    torch.manual_seed(4)
    reference = _transformers_model()
    native = MimiModel(reference.config.to_dict()).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    audio = torch.randn(2, 1, 64)

    with torch.no_grad():
        offline_codes = native.encode(audio, 3)
        offline_audio = native.decode(offline_codes)

        encoder_state = native.new_streaming_state()
        streaming_codes = torch.cat(
            [
                native.encode(chunk, 3, encoder_state)
                for chunk in audio.split(native.config.frame_size, dim=-1)
            ],
            dim=-1,
        )
        decoder_state = native.new_streaming_state()
        streaming_audio = torch.cat(
            [
                native.decode(chunk, decoder_state)
                for chunk in offline_codes.split(1, dim=-1)
            ],
            dim=-1,
        )

    torch.testing.assert_close(streaming_codes, offline_codes)
    torch.testing.assert_close(streaming_audio, offline_audio)


def test_native_mimi_streaming_state_stays_bounded() -> None:
    torch.manual_seed(5)
    reference = _transformers_model()
    native = MimiModel(reference.config.to_dict()).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    state = native.new_streaming_state()
    frame = torch.randn(1, 1, native.config.frame_size)

    with torch.no_grad():
        for _ in range(native.config.sliding_window + 2):
            codes = native.encode(frame, 3, state)
            native.decode(codes, state)
        saturated_shapes = _streaming_state_shapes(state)

        for _ in range(100):
            codes = native.encode(frame, 3, state)
            native.decode(codes, state)

    assert _streaming_state_shapes(state) == saturated_shapes
    assert state.encoder_transformer is not None
    assert state.decoder_transformer is not None
    for transformer in (
        state.encoder_transformer,
        state.decoder_transformer,
    ):
        assert all(
            key is not None and key.shape[2] <= native.config.sliding_window
            for key in transformer.keys
        )
        assert all(
            value is not None and value.shape[2] <= native.config.sliding_window
            for value in transformer.values
        )


def _streaming_state_shapes(
    state: MimiStreamingState,
) -> tuple[tuple[int, ...], ...]:
    tensors = []
    for convolution_states in (state.encoder_convs, state.decoder_convs):
        for index in sorted(convolution_states):
            convolution = convolution_states[index]
            if isinstance(convolution, MimiConv1dState):
                tensor = convolution.previous
            else:
                assert isinstance(convolution, MimiConvTranspose1dState)
                tensor = convolution.partial
            tensors.append(tensor)
    for transformer in (state.encoder_transformer, state.decoder_transformer):
        assert transformer is not None
        tensors.extend(key for key in transformer.keys if key is not None)
        tensors.extend(value for value in transformer.values if value is not None)
    return tuple(tuple(tensor.shape) for tensor in tensors)
