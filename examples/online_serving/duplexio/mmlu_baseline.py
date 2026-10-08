"""Score a plain chat model (DuplexIO's backbone) on the same lettered questions as the text-only test."""

import argparse
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from run_questions import answer_letter_in, centered_score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Hugging Face ID or local snapshot path")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instructions", default="You are a helpful voice assistant.")
    # 150 tokens matches a 12 s DuplexIO text reply at one agent token per 80 ms frame.
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    llm = LLM(model=args.model, max_model_len=4096, limit_mm_per_prompt={"image": 0, "video": 0})
    conversations = [
        [{"role": "system", "content": args.instructions}, {"role": "user", "content": question["text"]}]
        for question in questions
    ]
    outputs = llm.chat(
        conversations,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        chat_template_kwargs={"enable_thinking": False},
    )
    results = []
    for question, output in zip(questions, outputs, strict=True):
        text = output.outputs[0].text
        predicted = answer_letter_in(text)
        results.append({**question, "agent_text": text, "predicted": predicted, "correct": predicted == question["answer"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    correct = sum(result["correct"] for result in results)
    print(
        f"Correct letters: {correct}/{len(results)}; centered score {centered_score(correct, len(results)):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
