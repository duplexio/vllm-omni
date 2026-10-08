"""Speak a JSON list of questions with Kokoro and save DuplexIO's answers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from client import SAMPLE_RATE, run
from kokoro import KModel, KPipeline


def load_kokoro():
    assets = os.environ.get("KOKORO_ASSETS")
    if not assets:
        return KPipeline(lang_code="a", device="cpu"), "af_heart"
    root = Path(assets)
    model = (
        KModel(repo_id="hexgrad/Kokoro-82M", config=str(root / "config.json"), model=str(root / "kokoro-v1_0.pth"))
        .to("cpu")
        .eval()
    )
    return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model), str(root / "voices/af_heart.pt")


def centered_score(correct: int, total: int, choices: int = 4) -> float:
    """Accuracy rescaled so random guessing over `choices` options is 0 and perfect is 1 (as in nanochat)."""
    chance = 1 / choices
    return (correct / total - chance) / (1 - chance)


def answer_letter_in(text: str) -> str | None:
    """The first option letter the reply commits to: a leading or bold letter, or 'answer/option is X'."""
    patterns = (
        r"^\W*\(?([ABCD])(?:[).:,]|\s*\n|\s*$)",
        r"\*\*\(?([ABCD])[.)]?\*\*",
        r"(?i:answer|option)(?:\s+(?i:is))?\s*[:\-]?\s*\**\(?([ABCD])\b",
        r"\b(?i:is)\s+\**\(?([ABCD])\b(?!\w)",
        r"\b([ABCD])\**\s+is\s+(?i:the\s+)?(?i:correct|right)",
    )
    matches = [match for pattern in patterns if (match := re.search(pattern, text))]
    return min(matches, key=lambda match: match.start(1)).group(1) if matches else None


def synthesize(pipeline, voice: str, text: str, path: Path) -> float:
    chunks = [result.audio.numpy() for result in pipeline(text, voice=voice, speed=1.0)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    audio = np.concatenate(chunks).astype(np.float32)
    if not np.isfinite(audio).all() or not audio.size:
        raise RuntimeError("Kokoro produced invalid audio")
    sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
    return audio.size / SAMPLE_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="duplexio/duo-4b")
    parser.add_argument("--url", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-seconds", type=float, default=30)
    parser.add_argument("--text-only", action="store_true", help="Type each question into a text-only session instead of speaking it")
    parser.add_argument("--greedy", action="store_true", help="Greedy agent text instead of the trained sampling")
    parser.add_argument(
        "--instructions",
        default="You are a helpful voice assistant. Answer the user's question clearly and briefly.",
    )
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    if not isinstance(questions, list) or not questions:
        parser.error("Questions must be a nonempty JSON array")
    questions = [{"text": q} if isinstance(q, str) else q for q in questions]
    if any(not isinstance(q, dict) or not isinstance(q.get("text"), str) or not q["text"].strip() for q in questions):
        parser.error("Each question must be a nonempty string or an object with nonempty 'text'")
    if args.response_seconds <= 0:
        parser.error("Response seconds must be positive")
    pipeline, voice = load_kokoro()
    reference = args.output_dir / "reference.wav"
    synthesize(pipeline, voice, "Hello. I am ready to help you today.", reference)
    results = []
    for index, item in enumerate(questions, 1):
        question = item["text"]
        case = args.output_dir / f"{index:02d}"
        case.mkdir()
        (case / "question.txt").write_text(question + "\n")
        print(f"[{index}/{len(questions)}] {question}", flush=True)
        result = {
            **item, "index": index, "question": question, "seed": 42, "instructions": args.instructions,
            "mode": "text" if args.text_only else "audio",
            "sampling": "greedy" if args.greedy else "training",
        }
        del result["text"]
        try:
            duration = 0.0
            if not args.text_only:
                duration = synthesize(pipeline, voice, question, case / "input.wav")
                result["question_audio_seconds"] = duration
            request = argparse.Namespace(
                model=args.model,
                url=args.url,
                ref_audio=reference,
                user_audio=None if args.text_only else case / "input.wav",
                user_text=question if args.text_only else None,
                seconds=duration + args.response_seconds,
                instructions=args.instructions,
                seed=42,
                greedy=args.greedy,
                output=case / "answer.wav",
            )
            result.update(asyncio.run(run(request)))
            if not result["agent_text"]:
                raise RuntimeError("The agent transcript is empty")
            result["status"] = "ok"
            print(f"  Heard: {result['user_text']}\n  Answer: {result['agent_text']}", flush=True)
            if "answer" in item:
                result["predicted"] = answer_letter_in(result["agent_text"])
                result["correct"] = result["predicted"] == item["answer"]
                print(f"  Predicted: {result['predicted']}  Expected: {item['answer']}", flush=True)
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"  ERROR: {result['error']}", flush=True)
        (case / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = sum(result["status"] != "ok" for result in results)
    scored = [result for result in results if "correct" in result]
    if scored:
        correct = sum(result["correct"] for result in scored)
        print(
            f"Correct letters: {correct}/{len(scored)}; centered score {centered_score(correct, len(scored)):.3f}",
            flush=True,
        )
    print(
        f"Saved {len(results)} cases to {args.output_dir}; {failed} request errors. Correctness requires review.",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
