"""Compare native streaming codec with the training checkpoint implementation."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.audio_representation import ContinuousAudioRepresentation
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import DuplexIOForConditionalGeneration
from vllm_omni.model_executor.models.duplexio.pocket_mimi import PocketMimi

reference = pytest.importorskip("duplexio.modules.continuous_mimi")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA codec parity")
@torch.inference_mode()
@torch.backends.cudnn.flags(allow_tf32=False)
def test_real_silence_matches_training_in_acoustic_frame_order() -> None:
    torch.manual_seed(71)
    training = reference.ContinuousMimiModel().cuda().eval()
    native = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    torch.nn.Module.__init__(native)
    native.audio_codec = PocketMimi().cuda().eval()
    native.audio_codec.load_state_dict(training.state_dict(), strict=True)
    native.audio_representation = ContinuousAudioRepresentation(32).cuda()
    native.vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    # Real audio and listening pauses; text-only context never reaches a codec.
    chunks = [
        torch.randn(2 * 1920, device="cuda") * 0.1,
        torch.zeros(3 * 1920, device="cuda"),
        torch.randn(2 * 1920, device="cuda") * 0.1,
        torch.zeros(2 * 1920, device="cuda"),
    ]
    waveform = torch.cat(chunks)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected, _ = training.encode_to_latent(waveform[None, None])
    training_state = reference.ContinuousMimiState(
        encoder=training.encoder.get_initial_state(),
        decoder=training.decoder.get_initial_state(),
        downsample=training.downsample.get_initial_state(),
        upsample=training.upsample.get_initial_state(),
        encoder_transformer=training.encoder_transformer.get_initial_state(waveform[None, None]),
        decoder_transformer=training.decoder_transformer.get_initial_state(waveform[None, None]),
    )
    state = native.audio_codec.new_state(1)
    encoded = []
    for chunk in chunks:
        rows, state = native.encode_agent_audio(chunk, state, None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            streamed, training_state = training.step_encode(chunk[None, None], training_state)
        torch.testing.assert_close(rows, streamed[0].T.float(), atol=0, rtol=0)
        encoded.append(rows)
    # Training's packed varlen attention and incremental SDPA have different
    # BF16 reductions. Require exact native/streaming parity above, and bound
    # the full-sequence difference with training's varlen attention tolerance.
    actual = torch.cat(encoded)
    torch.testing.assert_close(actual, expected[0].T.float(), atol=0.02, rtol=0.01)
    assert (actual - expected[0].T.float()).norm() / expected.float().norm() < 0.01
    assert encoded[1].abs().max() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA codec parity")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
@torch.backends.cudnn.flags(allow_tf32=False)
def test_streaming_codec_matches_training_and_independent_requests(dtype):
    torch.manual_seed(31)
    training = reference.ContinuousMimiModel().to(device="cuda", dtype=dtype).eval()
    native = PocketMimi().to(device="cuda", dtype=dtype).eval()
    native.load_state_dict(training.state_dict(), strict=True)
    waveform = torch.randn(2, 1, 3 * 1920, device="cuda", dtype=dtype) * 0.1
    training_state = reference.ContinuousMimiState(
        encoder=training.encoder.get_initial_state(),
        decoder=training.decoder.get_initial_state(),
        downsample=training.downsample.get_initial_state(),
        upsample=training.upsample.get_initial_state(),
        encoder_transformer=training.encoder_transformer.get_initial_state(waveform),
        decoder_transformer=training.decoder_transformer.get_initial_state(waveform),
    )
    native_state = native.new_state(2)
    encoded = []
    for index, chunk in enumerate(waveform.split(1920, dim=-1)):
        expected, training_state = training.step_encode(chunk, training_state)
        actual, native_state = native.encode(chunk, native_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        encoded.append(actual)

    latent = torch.cat(encoded, dim=-1)
    native_state = native.new_state(2)
    decoded = []
    for index, chunk in enumerate(latent.split(1, dim=-1)):
        expected, training_state = training.step_decode(chunk, training_state)
        actual, native_state = native.decode(chunk, native_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        decoded.append(actual)
    batched = torch.cat(decoded, dim=-1)

    for row in range(2):
        state = native.new_state(1)
        for index, chunk in enumerate(latent[row : row + 1].split(1, dim=-1)):
            actual, state = native.decode(chunk, state)
            torch.testing.assert_close(
                actual,
                batched[row : row + 1, :, index * 1920 : (index + 1) * 1920],
                rtol=2e-4 if dtype == torch.float32 else 0.04,
                atol=2e-6 if dtype == torch.float32 else 0.003,
            )

    # A fork must not mutate its parent's convolution or attention history.
    before, _ = native.decode(latent[..., :1], native_state)
    native.decode(latent[..., 1:2], native_state)
    after, _ = native.decode(latent[..., :1], native_state)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
