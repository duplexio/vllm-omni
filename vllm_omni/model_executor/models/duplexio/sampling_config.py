# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampling configuration of one DuplexIO session, fixed when it starts."""

from pydantic import BaseModel, ConfigDict, Field


class EmissionPolicy(BaseModel):
    """Temperature of a stream's speak-versus-silence decision; zero is greedy."""

    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=0.0, ge=0)


class ContentPolicy(BaseModel):
    """How a stream draws a token once it speaks; zero temperature is greedy."""

    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=0.0, ge=0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)


class SamplingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emission: EmissionPolicy = Field(default_factory=EmissionPolicy)
    content: ContentPolicy = Field(default_factory=ContentPolicy)


class AudioPolicy(BaseModel):
    """Temperature of the agent-audio head; absent, the checkpoint's."""

    model_config = ConfigDict(extra="forbid")
    temperature: float | None = Field(default=None, gt=0)


class SamplingConfig(BaseModel):
    """Per-stream sampling: the agent's text and tool calls share its policy; the user stream transcribes."""

    model_config = ConfigDict(extra="forbid")
    agent: SamplingPolicy = Field(
        default_factory=lambda: SamplingPolicy(
            emission=EmissionPolicy(temperature=1.0),
            content=ContentPolicy(temperature=0.6, top_k=20, top_p=0.95),
        )
    )
    user: SamplingPolicy = Field(default_factory=SamplingPolicy)
    audio: AudioPolicy = Field(default_factory=AudioPolicy)
