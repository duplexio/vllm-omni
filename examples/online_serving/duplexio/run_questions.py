"""Type a JSON list of questions into text-only DuplexIO sessions and score the option letters."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from client import run


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="duplexio/duo-4b")
    parser.add_argument("--url", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # 12 s is about 150 agent tokens at one token per 80 ms frame; the forced reply never ends on its own.
    parser.add_argument("--response-seconds", type=float, default=12)
    parser.add_argument("--limit", type=int, help="Run only the first N questions")
    parser.add_argument("--instructions", default="You are a helpful voice assistant.")
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    if not isinstance(questions, list) or not questions:
        parser.error("Questions must be a nonempty JSON array")
    questions = [{"text": q} if isinstance(q, str) else q for q in questions][: args.limit]
    if any(not isinstance(q, dict) or not isinstance(q.get("text"), str) or not q["text"].strip() for q in questions):
        parser.error("Each question must be a nonempty string or an object with nonempty 'text'")
    if args.response_seconds <= 0:
        parser.error("Response seconds must be positive")
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
            request = argparse.Namespace(
                model=args.model,
                url=args.url,
                ref_audio=None,
                user_audio=None,
                user_text=question,
                seconds=args.response_seconds,
                instructions=args.instructions,
                seed=42,
                output=None,
            )
            result.update(asyncio.run(run(request)))
            if not result["agent_text"]:
                raise RuntimeError("The agent transcript is empty")
            result["status"] = "ok"
            print(f"  Answer: {result['agent_text']}", flush=True)
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
    print(f"Saved {len(results)} cases to {args.output_dir}; {failed} request errors.", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
