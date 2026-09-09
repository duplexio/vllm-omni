"""Cross-repository checks of the existing quantized audio head."""

import pytest
import torch

from vllm_omni.model_executor.models.duplexio.depth_sampler import (
    DepthAutoregressiveSampler,
    DepthSamplerConfig,
)

training = pytest.importorskip("duplexio.modules.depth_sampler")
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("codebooks", [1, 3, 8])
@pytest.mark.parametrize("rank", [None, 4])
@pytest.mark.parametrize("top_k", [1, 5])
def test_batched_depth_sampling_matches_training(codebooks: int, rank: int | None, top_k: int) -> None:
    torch.manual_seed(81)
    reference = training.DepthAutoregressiveSampler(
        conditioning_dim=12,
        speaker_embedding_dim=6,
        text_vocab_size=32,
        low_rank_embeddings=rank,
        codebook_size=16,
        num_codebooks=codebooks,
        dim=16,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
        codebook_loss_weights=[1.0] * codebooks,
        sampling_temperature=0.8,
        sampling_top_k=top_k,
        semantic_sampling_top_k=None,
        gradient_checkpointing=False,
    )
    native = DepthAutoregressiveSampler(
        DepthSamplerConfig(
            conditioning_dim=12,
            speaker_embedding_dim=6,
            text_vocab_size=32,
            low_rank_embeddings=rank,
            codebook_size=16,
            num_codebooks=codebooks,
            dim=16,
            num_layers=2,
            num_heads=4,
            feedforward_dim=64,
            sampling_temperature=0.8,
            sampling_top_k=top_k,
        )
    )
    native.load_state_dict(reference.state_dict(), strict=True)
    conditioning, speakers = torch.randn(4, 12), torch.randn(4, 6)
    text = torch.tensor([3, 7, 1, 5])
    torch.manual_seed(31)
    expected = reference.sample(conditioning, text, speakers)
    torch.manual_seed(31)
    actual = native.sample(conditioning, text, native.prepare_speaker(speakers))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if top_k == 1:
        independent = torch.cat(
            [
                native.sample(
                    conditioning[i : i + 1],
                    text[i : i + 1],
                    native.prepare_speaker(speakers[i : i + 1]),
                )
                for i in range(4)
            ]
        )
        torch.testing.assert_close(actual, independent, atol=0, rtol=0)
