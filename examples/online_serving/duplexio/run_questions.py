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


def answer_letter_in(text: str) -> str | None:
    text = text.replace("*", "")
    match = re.search(r"answer(?:\s+is)?\s*[:\-]?\s*\(?([ABCD])\b", text, re.IGNORECASE) or re.match(
        r"\W*\(?([ABCD])(?:[).:,]|\s*$)", text
    )
    return match.group(1).upper() if match else None


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
        result = {**item, "index": index, "question": question, "seed": 42, "instructions": args.instructions}
        del result["text"]
        try:
            duration = synthesize(pipeline, voice, question, case / "input.wav")
            result["question_audio_seconds"] = duration
            request = argparse.Namespace(
                model=args.model,
                url=args.url,
                ref_audio=reference,
                user_audio=case / "input.wav",
                seconds=duration + args.response_seconds,
                instructions=args.instructions,
                seed=42,
                output=case / "answer.wav",
            )
            result.update(asyncio.run(run(request)))
            if not result["agent_text"]:
                raise RuntimeError("Audio was returned but the agent transcript is empty")
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
        print(f"Correct letters: {sum(result['correct'] for result in scored)}/{len(scored)}", flush=True)
    print(
        f"Saved {len(results)} cases to {args.output_dir}; {failed} request errors. Correctness requires review.",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
