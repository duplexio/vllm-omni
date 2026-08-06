# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Server-owned state for turnwise MOSS-TTSD continuation."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Literal

import torch

SpeakerRole = Literal["user", "assistant"]


def speaker_tag(role: SpeakerRole) -> str:
    return "S1" if role == "user" else "S2"


@dataclass(frozen=True)
class MossTTSDTranscriptSegment:
    """One contiguous speaker segment in the synthesized dialogue."""

    role: SpeakerRole
    text: str

    def render(self) -> str:
        return f"[{speaker_tag(self.role)}] {self.text}"


@dataclass(frozen=True)
class MossTTSDTurn:
    """Immutable prompt view reserved for one in-flight session turn."""

    session_id: str
    revision: int
    full_text: str
    role: SpeakerRole
    text: str
    continues_previous: bool
    reference_codes: tuple[torch.Tensor, torch.Tensor]
    audio_prefix_codes: torch.Tensor
    cache_salt: str


@dataclass
class MossTTSDSession:
    """The transcript and raw RVQ prefix for one dialogue."""

    session_id: str
    reference_text: tuple[str, str]
    reference_codes: tuple[torch.Tensor, torch.Tensor]
    audio_prefix_codes: torch.Tensor
    cache_salt: str
    completed_segments: list[MossTTSDTranscriptSegment] = field(default_factory=list)
    revision: int = 0
    in_flight: bool = False
    updated_at: float = field(default_factory=time.monotonic)


class MossTTSDSessionStore:
    """Bounded in-memory continuation state owned by one API server."""

    def __init__(self, max_sessions: int = 2048, ttl_seconds: float = 3600.0) -> None:
        if max_sessions <= 0:
            raise ValueError(f"max_sessions must be positive, got {max_sessions}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, MossTTSDSession] = {}

    def create(
        self,
        session_id: str,
        reference_text: tuple[str, str],
        reference_codes: tuple[torch.Tensor, torch.Tensor],
        prompt_codes: torch.Tensor,
    ) -> MossTTSDSession:
        """Create a session with the continuous prompt encoding as its prefix."""
        self.prune()
        normalized_id = session_id.strip()
        if not normalized_id:
            raise ValueError("session_id cannot be empty")
        if normalized_id in self.sessions:
            raise ValueError(f"MOSS-TTSD session already exists: {normalized_id}")
        if len(self.sessions) >= self.max_sessions:
            idle = [session for session in self.sessions.values() if not session.in_flight]
            if not idle:
                raise RuntimeError("MOSS-TTSD session capacity is exhausted")
            oldest = min(idle, key=lambda session: session.updated_at)
            del self.sessions[oldest.session_id]

        refs = tuple(code.detach().to("cpu", torch.long).contiguous() for code in reference_codes)
        if len(refs) != 2:
            raise ValueError(f"MOSS-TTSD requires two reference code tensors, got {len(refs)}")
        for index, code in enumerate(refs):
            if code.ndim != 2 or code.shape[0] == 0:
                raise ValueError(
                    f"reference_codes[{index}] must have shape (frames, codebooks), got {tuple(code.shape)}"
                )
        if refs[0].shape[1] != refs[1].shape[1]:
            raise ValueError(
                "MOSS-TTSD reference codebooks differ: "
                f"{refs[0].shape[1]} != {refs[1].shape[1]}"
            )
        prompt = prompt_codes.detach().to("cpu", torch.long).contiguous()
        if prompt.ndim != 2 or prompt.shape[0] == 0:
            raise ValueError(
                "prompt_codes must have shape (frames, codebooks), "
                f"got {tuple(prompt.shape)}"
            )
        if prompt.shape[1] != refs[0].shape[1]:
            raise ValueError(
                "MOSS-TTSD prompt and reference codebooks differ: "
                f"{prompt.shape[1]} != {refs[0].shape[1]}"
            )

        session = MossTTSDSession(
            session_id=normalized_id,
            reference_text=(
                f"[S1] {reference_text[0].strip()}",
                f"[S2] {reference_text[1].strip()}",
            ),
            reference_codes=(refs[0], refs[1]),
            audio_prefix_codes=prompt,
            cache_salt=f"moss-ttsd:{secrets.token_hex(16)}",
        )
        self.sessions[normalized_id] = session
        return session

    def begin_turn(self, session_id: str, role: SpeakerRole, text: str) -> MossTTSDTurn:
        """Reserve one turn and return the exact continuation prompt state."""
        self.prune()
        session = self.get(session_id)
        if session.in_flight:
            raise RuntimeError(f"MOSS-TTSD session already has an in-flight turn: {session_id}")
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("MOSS-TTSD turn text cannot be empty")

        continues_previous = bool(
            session.completed_segments
            and session.completed_segments[-1].role == role
        )
        prompt_segments = list(session.completed_segments)
        if continues_previous:
            previous = prompt_segments[-1]
            prompt_segments[-1] = MossTTSDTranscriptSegment(
                role=role,
                text=f"{previous.text} {normalized_text}",
            )
        else:
            prompt_segments.append(
                MossTTSDTranscriptSegment(role=role, text=normalized_text)
            )
        full_text = " ".join(
            [
                *session.reference_text,
                *(segment.render() for segment in prompt_segments),
            ]
        )
        session.in_flight = True
        session.updated_at = time.monotonic()
        return MossTTSDTurn(
            session_id=session.session_id,
            revision=session.revision,
            full_text=full_text,
            role=role,
            text=normalized_text,
            continues_previous=continues_previous,
            reference_codes=session.reference_codes,
            audio_prefix_codes=session.audio_prefix_codes,
            cache_salt=session.cache_salt,
        )

    def commit_turn(
        self,
        turn: MossTTSDTurn,
        generated_codes: torch.Tensor,
    ) -> int:
        """Commit generated raw RVQ codes and return the new revision."""
        session = self.get(turn.session_id)
        self.check_pending_turn(session, turn)
        generated = generated_codes.detach().to("cpu", torch.long).contiguous()
        if generated.ndim != 2 or generated.shape[0] == 0:
            raise ValueError(
                "generated_codes must have shape (frames, codebooks), "
                f"got {tuple(generated.shape)}"
            )
        if turn.continues_previous:
            assert (
                session.completed_segments
                and session.completed_segments[-1].role == turn.role
            )
        session.audio_prefix_codes = torch.cat(
            [session.audio_prefix_codes, generated],
            dim=0,
        )
        if turn.continues_previous:
            previous = session.completed_segments[-1]
            session.completed_segments[-1] = MossTTSDTranscriptSegment(
                role=turn.role,
                text=f"{previous.text} {turn.text}",
            )
        else:
            session.completed_segments.append(
                MossTTSDTranscriptSegment(role=turn.role, text=turn.text)
            )
        session.revision += 1
        session.in_flight = False
        session.updated_at = time.monotonic()
        return session.revision

    def abort_turn(self, turn: MossTTSDTurn) -> None:
        """Release an in-flight reservation without changing its prefix."""
        session = self.sessions.get(turn.session_id)
        if session is None or session.revision != turn.revision:
            return
        session.in_flight = False
        session.updated_at = time.monotonic()

    def delete(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            return False
        if session.in_flight:
            raise RuntimeError(f"Cannot delete an in-flight MOSS-TTSD session: {session_id}")
        del self.sessions[session_id]
        return True

    def get(self, session_id: str) -> MossTTSDSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown or expired MOSS-TTSD session: {session_id}") from exc

    def prune(self) -> int:
        expires_before = time.monotonic() - self.ttl_seconds
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if not session.in_flight and session.updated_at < expires_before
        ]
        for session_id in expired:
            del self.sessions[session_id]
        return len(expired)

    @staticmethod
    def check_pending_turn(session: MossTTSDSession, turn: MossTTSDTurn) -> None:
        if not session.in_flight or session.revision != turn.revision:
            raise RuntimeError(
                "Stale MOSS-TTSD turn commit: "
                f"session={turn.session_id} expected_revision={session.revision} "
                f"turn_revision={turn.revision}"
            )
