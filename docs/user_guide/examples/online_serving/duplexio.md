# DuplexIO: Online serving

DuplexIO is a full duplex speech model. It listens and speaks in the same realtime
session. The native `duplexio/duo-4b` Hugging Face checkpoint is a version-seven
export with a Qwen3.5 backbone, streaming input ASR, and Pocket Mimi audio.

## Install

Use Python 3.13 and a CUDA GPU. From this checkout:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install vllm==0.26.0 --torch-backend=cu130
uv pip install -e .
uv pip install flash-linear-attention==0.5.1 fla-core==0.5.1 \
  quack-kernels==0.6.3 nvidia-cutlass-dsl==4.6.0 websockets
```

The input ASR needs a Transformers build that includes
`NemotronAsrStreamingModel`. If your release does not include it, install the
verified source revision:

```bash
uv pip install 'transformers @ git+https://github.com/huggingface/transformers.git@b3d7e8c9d4e078a5e6c09a9d67e22dcadc2df4b8'
```

This repository contains all of the inference code. You do not need the DuplexIO
training package. If your account needs access to the checkpoint, run
`hf auth login` first.

## Serve

```bash
vllm serve duplexio/duo-4b --omni
```

The model registry selects `vllm_omni/deploy/duplexio.yaml` automatically. This
config runs one session on GPU 0 with BF16 backbone weights and CUDA graph decode.
For a reproducible v7 export, add the revision:

```bash
vllm serve duplexio/duo-4b --omni \
  --revision 80de418a31334c737100afe767a08b1a27f6ee1a
```

A local export directory also works. For two concurrent sessions, use the bundled
realtime preset:

```bash
vllm serve duplexio/duo-4b --omni \
  --deploy-config vllm_omni/deploy/duplexio-realtime.yaml
```

## Run an inference

Give the agent a short reference recording of its voice. The checkpoint has no
named voices. Reference and user recordings must be mono WAV files at 24 kHz. The
server pins up to ten seconds of reference audio per session.

```bash
python examples/online_serving/duplexio/client.py \
  --ref-audio reference.wav --output agent.wav
```

Without a user recording, the agent speaks first. To ask a spoken question:

```bash
python examples/online_serving/duplexio/client.py \
  --ref-audio reference.wav --user-audio question.wav \
  --seconds 30 --output answer.wav
```

The client prints the user and agent transcripts and saves the received PCM. Then
it closes the session to release its state. `--seconds` sets how long the client
streams input, including the silence after the question. It does not end the answer.

## Run the MMLU benchmark

This benchmark scores the DuplexIO text channel on 100 MMLU questions. It types the
questions into text-only sessions with greedy agent text and no voice prompt. You
need only the [serving environment](#install).

1. Score DuplexIO. The script starts the server, runs every question and stops the
   server:

   ```bash
   bash examples/online_serving/duplexio/run_mmlu.sh outputs/mmlu100-duplexio
   ```

2. Optionally, score the backbone on the same questions as a reference:

   ```bash
   python examples/online_serving/duplexio/mmlu_baseline.py \
       --questions examples/online_serving/duplexio/mmlu100_questions.json --output outputs/mmlu100-qwen.json
   ```

3. Read the last line of each run. It shows `Correct letters: N/100; centered score S`.
   The replies and letters for each question are in
   `outputs/mmlu100-duplexio/results.json` and `outputs/mmlu100-qwen.json`.

On the DCAI cluster, submit step 1 from the repository root instead:

```bash
sbatch examples/online_serving/duplexio/mmlu.sbatch
```

The job log is `outputs/duplexio-inference/<job id>.log`. The results go to
`outputs/duplexio-inference/<job id>/`.

Both scripts take a question file as an argument and pass further arguments to
`run_questions.py`. Use `--limit N` to run only the first N questions. A question
is an object with `text` and an `answer` letter. Set `DUPLEXIO_PORT` to use another
server port.

## How the MMLU benchmark works

Spoken questions mix several problems:

- The agent takes its turn at the first pause and misses the options.
- The user ASR garbles symbols.
- The agent audio can drift away from its text.

The benchmark types the questions, so it measures only what the model knows. Each
question runs in a [text-only session](#text-only-sessions):

- The prefix holds only the system prompt. There is no voice prompt, because
  text-only training had none. A pinned voice clip is context that every answer
  token can attend to, and it changed some answers in tests.
- The question goes into the user text cell, one token per frame, with audio off.
  DuplexIO's text-only chat training uses the same layout.
- The agent text is greedy. The model always takes its most likely token.
- The server forces the agent to emit on every frame. This is a workaround. The
  speak/wait head never trained on text-only frames, and its probability follows
  word pacing, not the end of an answer. It is low before a new word and high inside
  a word. So the model cannot end the reply itself.
- `--response-seconds` cuts the reply instead. At one token per 80 ms frame, 12 s
  is about 150 tokens. Most replies commit to a letter within 15 tokens, but a
  reply that reasons first can get cut before it answers. Replies that run longer
  tend to loop.
- `answer_letter_in` in `run_questions.py` finds the first option letter that the
  reply commits to.
- The centered score is `(accuracy - 1/4) / (1 - 1/4)`. Random guessing gives 0
  and a perfect run gives 1.

`mmlu_baseline.py` asks the backbone through vLLM. It uses the same system prompt,
questions and 150-token budget, with thinking off and greedy sampling.
`mmlu100_questions.json` is a seeded sample of the `cais/mmlu` test split.

### Reply length in audio sessions

In audio sessions the speak/wait head works as trained, and the model ends its own
reply. A test ran `client.py --ref-audio --user-audio --seconds 90` with three
spoken prompts:

| Prompt | Agent text | Ended by itself |
|---|---|---|
| Tell me a long story about a lighthouse keeper. | 314 tokens, 0.5 to 89.5 s | No. The story continued at 90 s. |
| Explain how photosynthesis works. | 172 tokens, 1.3 to 89.1 s | No. The list continued at 90 s. |
| What is the capital of France? | 57 tokens, 3.9 to 26.4 s | Yes. |

The agent text arrives in bursts while the speech catches up. Inside one answer, the
gaps between bursts were up to 8 s. After the short answer, no text came for 17 s.
So a stop rule has to watch the text, not the audio. After the answer ended, the
agent audio kept making speech-like sound with no text behind it. The agent produced
about 3.5 tokens per second, against 12.5 in a forced text-only session.

The MMLU benchmark stays text-only for now. Audio sessions need a voice reference
and run about 3.5 times slower. Spoken questions also make the agent take its turn
at the first pause, before it hears the options.

## Realtime protocol

Connect to `/v1/realtime?duplex=1&model=duplexio/duo-4b&autostart=0`. Then send
`session.update` with audio and text modalities and an `extra_body` that contains:

- `full_duplex: true`
- `auto_response: true`
- `start_role: "user"` or `"agent"`
- the voice, as base64 little-endian float32 PCM in `ref_audio_data`, with
  `ref_audio_format: "pcm_f32le"` and `ref_audio_sample_rate: 24000`

Wait for `session.updated`. Then send `input_audio_buffer.append` events with 1,920
samples each. One event is one 80 ms frame at 24 kHz.

Agent output arrives in `response.audio.delta` and
`response.audio_transcript.delta`. User text arrives in
`conversation.item.input_audio_transcription.delta`. Each audio delta states its
format and sample rate. Acknowledge played audio with `playback.ack`. To finish,
send `session.close` and wait for `session.closed`.

The client sets sampling overrides in `extra_body.duplexio_sampling`, with separate
emission and content settings for `agent` and `user`. Realtime user transcription
uses greedy sampling by default. The
[native inference notes](https://github.com/duplexio/vllm-omni/blob/duplexio/examples/offline_inference/duplexio/README.md)
describe the numerical and state contracts.

### Text-only sessions

Set `extra_body.text_only: true` and `start_role: "user"`, and send no
`ref_audio_data`. The server rejects a voice reference in a text-only session, so
the prefix holds only the system prompt, as in text-only training. Then send
`{"type": "input_text.append", "text": ...}`. The server writes the text into the
user cell, one token per frame, with audio off.

Keep sending `input_audio_buffer.append` frames to move the reply forward, one frame
per event. Silent frames are enough. The server ignores their audio and decodes no
agent audio. The reply arrives only as `response.audio_transcript.delta`.

Training never fit the agent's speak/wait head on text-only frames. So the server
forces the agent to emit on every text-only frame, including the silence frames
that the server adds itself. The reply continues until you stop sending frames.
Each transcript delta carries `agent_emit_logprob`, the head's own log probability
of speaking on that frame.

To try it, run `client.py --user-text "..."`. The client asks for greedy agent text
in text-only sessions. `run_questions.py` does the same for every question in a file.
