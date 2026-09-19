"""Portable per-stream sampling contract shared by offline and online serving."""

from pydantic import BaseModel, ConfigDict, Field


class EmissionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=0.0, ge=0)


class ContentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(default=0.0, ge=0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)


class SamplingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emission: EmissionPolicy = Field(default_factory=EmissionPolicy)
    content: ContentPolicy = Field(default_factory=ContentPolicy)


class SamplingConfig(BaseModel):
    """Sampling belongs to a request, not to the checkpoint or execution mode."""

    model_config = ConfigDict(extra="forbid")
    agent: SamplingPolicy = Field(default_factory=lambda: SamplingPolicy(
        emission=EmissionPolicy(temperature=1.0),
        content=ContentPolicy(temperature=0.6, top_k=20, top_p=0.95),
    ))
    user: SamplingPolicy = Field(default_factory=SamplingPolicy)


def sampling_runtime(sampling: SamplingConfig) -> dict:
    """Resolve explicit policies once, at the request-construction boundary."""
    agent = sampling.agent.model_dump()
    user = sampling.user.model_dump()
    emissions = {"agent": agent["emission"], "tool_call": agent["emission"], "user": user["emission"]}
    return {
        "duplexio_text_sampling": agent["content"],
        "duplexio_user_sampling": user,
        "duplexio_emit_temperatures": {
            stream: policy["temperature"]
            for stream, policy in emissions.items()
        },
    }
