"""Replay a fixed-size batch of the unchanged audio input projections."""

import torch
from torch import Tensor, nn

from vllm_omni.model_executor.models.duplexio.audio_adapters import AgentAudioInputAdapter, AudioInputAdapter


class AudioInputGraph:
    """The engine owns weights; this graph owns only its input/output buffers.

    Outputs are borrowed until the next replay of this batch size. The runner
    consumes them in the current forward before preparing another batch.
    """

    def __init__(
        self,
        user_adapter: AudioInputAdapter,
        agent_embedding: nn.Module,
        agent_adapter: AgentAudioInputAdapter,
        features: Tensor,
        codes: Tensor,
        speakers: Tensor,
        dtype: torch.dtype,
    ) -> None:
        self.features = features.clone()
        self.codes = codes.clone()
        self.speakers = speakers.clone()

        def project() -> tuple[Tensor, Tensor]:
            return user_adapter(self.features), agent_adapter(agent_embedding(self.codes), self.speakers)

        stream = torch.cuda.Stream(device=features.device)
        stream.wait_stream(torch.cuda.current_stream())
        with (
            torch.cuda.stream(stream),
            torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False),
        ):
            project()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(self.graph),
            torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False),
        ):
            self.user_hidden, self.agent_hidden = project()

    def __call__(self, features: Tensor, codes: Tensor, speakers: Tensor) -> tuple[Tensor, Tensor]:
        self.features.copy_(features)
        self.codes.copy_(codes)
        self.speakers.copy_(speakers)
        self.graph.replay()
        return self.user_hidden, self.agent_hidden
