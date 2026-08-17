import torch
from torch import nn

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import MossAudioTokenizerDecoderOutput
from vllm_omni.model_executor.models.moss_tts.moss_codec_cudagraph import MossTTSCUDAGraphCodecWrapper
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import _MossCodecStreamSession


class FakeCodec:
    def _decode(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> MossAudioTokenizerDecoderOutput:
        return self.batch_decode([codes[:, index, : lengths[index]] for index in range(codes.shape[1])])

    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        lengths = torch.tensor([codes.shape[-1] * 3 for codes in codes_list], dtype=torch.long)
        audio = torch.zeros(len(codes_list), 1, int(lengths.max()))
        for index, length in enumerate(lengths.tolist()):
            audio[index, 0, :length] = index + 1
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=lengths)


def test_eager_batch_decode_preserves_request_order_and_lengths() -> None:
    wrapper = MossTTSCUDAGraphCodecWrapper(
        model=FakeCodec(),
        capture_sizes=[4, 8],
        num_quantizers=2,
        capture_batch_sizes=[1],
        enabled=False,
    )
    codes = [
        torch.zeros(2, 3, dtype=torch.long),
        torch.zeros(2, 6, dtype=torch.long),
        torch.zeros(2, 2, dtype=torch.long),
    ]

    outputs = wrapper.decode_batch(codes)

    assert [tuple(output.audio.shape) for output in outputs if output.audio is not None] == [
        (1, 1, 9),
        (1, 1, 18),
        (1, 1, 6),
    ]
    assert [int(output.audio[0, 0, 0]) for output in outputs if output.audio is not None] == [1, 2, 3]


def test_single_request_decode_uses_batch_path() -> None:
    wrapper = MossTTSCUDAGraphCodecWrapper(
        model=FakeCodec(),
        capture_sizes=[4],
        num_quantizers=2,
        capture_batch_sizes=[1],
        enabled=False,
    )

    output = wrapper.decode(torch.zeros(2, 3, dtype=torch.long))

    assert output.audio is not None
    assert tuple(output.audio.shape) == (1, 1, 9)


class FakeV1Codec(nn.Module):
    downsample_rate = 3

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.batch_sizes: list[int] = []

    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        del num_quantizers
        self.batch_sizes.append(len(codes_list))
        lengths = torch.tensor([codes.shape[-1] * self.downsample_rate for codes in codes_list])
        audio = torch.zeros(len(codes_list), 1, int(lengths.max()))
        for index, length in enumerate(lengths.tolist()):
            audio[index, 0, :length] = index + 1
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=lengths)


def test_v1_stream_session_batches_histories_and_returns_delta_tail() -> None:
    codec = FakeV1Codec()
    session = _MossCodecStreamSession(codec, stream_slots=2, n_vq=2)
    first = session.step({0: torch.zeros(2, 2, dtype=torch.long)})
    second = session.step({0: torch.zeros(2, 1, dtype=torch.long), 1: torch.zeros(2, 1, dtype=torch.long)})

    assert tuple(first[0].shape) == (1, 6)
    assert tuple(second[0].shape) == (1, 3)
    assert tuple(second[1].shape) == (1, 3)
    assert codec.batch_sizes == [1, 2]
