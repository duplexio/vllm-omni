# DuplexIO: Online serving

DuplexIO is a full duplex speech model: it listens and speaks in the same realtime
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

The input ASR requires a Transformers build that includes
`NemotronAsrStreamingModel`. If your release does not include it, install the
verified source revision:

```bash
uv pip install 'transformers @ git+https://github.com/huggingface/transformers.git@b3d7e8c9d4e078a5e6c09a9d67e22dcadc2df4b8'
```

Inference is owned by this repository; the DuplexIO training package is not
required. Authenticate with `hf auth login` if your account needs access to the
checkpoint.

## Serve

```bash
vllm serve duplexio/duo-4b --omni
```

The model registry selects `vllm_omni/deploy/duplexio.yaml` automatically. This
runs one session on GPU 0 with BF16 backbone weights and CUDA graph decode. For a
reproducible v7 export, add:

```bash
vllm serve duplexio/duo-4b --omni \
  --revision 80de418a31334c737100afe767a08b1a27f6ee1a
```

A local export directory also works. Two concurrent sessions use the bundled
realtime preset:

```bash
vllm serve duplexio/duo-4b --omni \
  --deploy-config vllm_omni/deploy/duplexio-realtime.yaml
```

## Run an inference

Supply a short reference recording for the agent's voice. The checkpoint carries
no named voices. Reference and user recordings must be mono WAV files at 24 kHz;
the server pins up to ten seconds of reference audio per session.

```bash
python examples/online_serving/duplexio/client.py \
  --ref-audio reference.wav --output agent.wav
```

The agent speaks first when no user recording is supplied. To ask a spoken
question:

```bash
python examples/online_serving/duplexio/client.py \
  --ref-audio reference.wav --user-audio question.wav \
  --seconds 30 --output answer.wav
```

The client prints the user and agent transcripts, saves the received PCM, and
closes the session to release its state. `--seconds` controls the input streaming
duration, including silence after the question; it is not an answer-ending rule.

## Test ten spoken questions

The inference test starts the server, synthesizes ten questions with Kokoro on
CPU, and opens a fresh user-first realtime session for each question. Questions
progress from simple arithmetic to reasoning and an MMLU-style biology question.

Install Kokoro and its English tokenizer assets in the serving environment:

```bash
uv pip install kokoro==0.9.4 'misaki[en]==0.9.4'
uv pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
bash examples/online_serving/duplexio/test_inference.sh
```

Results go to `outputs/duplexio-inference/<timestamp>/`: `server.log`, the voice
reference, `server-command.txt`, `results.json`, and one numbered directory per question
with `question.txt`, `input.wav`, `answer.wav`, and `result.json`. The script stops
the server when it finishes. It continues after individual request errors and
returns a nonzero exit status if any fail. A successful request means audio and
text were returned; answer correctness and audio quality still require review.

Edit `examples/online_serving/duplexio/questions.json` to change the prompts, or
pass an output directory and custom question file as the two script arguments.
Further arguments go to `run_questions.py`, for example `--instructions`. A question
can also be an object with `text` and an `answer` letter, as in `mmlu_questions.json`;
the script then records the predicted letter and prints the number correct.
Set `KOKORO_ASSETS=/path/to/kokoro` for local Kokoro assets and `DUPLEXIO_PORT`
to choose another server port. Each question receives thirty seconds of silence
after its complete audio, and the v7 checkpoint revision is pinned. Pass
`--text-only` to type each question instead of speaking it (see below).

## Measure the text channel on MMLU

Spoken questions mix several problems: the agent takes its turn at the first pause
and misses the options, user ASR garbles symbols, and agent audio can drift from
its text. To test only what the model knows, type the questions instead:

```bash
bash examples/online_serving/duplexio/test_inference.sh outputs/mmlu100 \
    examples/online_serving/duplexio/mmlu100_questions.json \
    --text-only --greedy --response-seconds 12 --instructions "You are a helpful voice assistant."
```

This runs each question in a [text-only session](#text-only-sessions):

- The question goes one token per frame into the user text cell, with audio off,
  as in DuplexIO's text-only chat training.
- The agent is forced to emit on every frame. Its speak/wait head never trained on
  text-only frames, and its probability tracks word pacing (low before a new word,
  high inside one), not the end of an answer. So the model cannot end the reply.
- `--response-seconds` cuts the reply instead: 12 s is about 150 tokens at one
  token per 80 ms frame. Replies left running tend to loop.
- `--greedy` takes the most likely agent token instead of the trained sampling.
- `answer_letter_in` in `run_questions.py` scores the first option letter the reply
  commits to. The summary also prints a centered score, `(accuracy - 1/4) / (1 - 1/4)`,
  so random guessing is 0 and a perfect score is 1.

For a reference, `mmlu_baseline.py` asks the backbone the same questions with vLLM,
using the same system prompt and token budget, with thinking off and greedy sampling:

```bash
python examples/online_serving/duplexio/mmlu_baseline.py \
    --questions examples/online_serving/duplexio/mmlu100_questions.json --output qwen-mmlu100.json
```

The question files are seeded samples of the `cais/mmlu` test split.
`examples/online_serving/duplexio/questions.sbatch` runs the DuplexIO side on one
H100 of the DCAI cluster; split the file into chunks to run several jobs at once.

## Realtime protocol

Connect to `/v1/realtime?duplex=1&model=duplexio/duo-4b&autostart=0`.
Send `session.update` with audio/text modalities and `extra_body` containing
`full_duplex: true`, `auto_response: true`, and `start_role: "user"` or `"agent"`.
The voice is `ref_audio_data` (base64 little-endian float32 PCM),
`ref_audio_format: "pcm_f32le"`, and `ref_audio_sample_rate: 24000`.
Wait for `session.updated`, then send `input_audio_buffer.append` events with
1,920 samples each: one 80 ms frame at 24 kHz.

Agent output arrives in `response.audio.delta` and
`response.audio_transcript.delta`; user text arrives in
`conversation.item.input_audio_transcription.delta`. Audio deltas describe their
format and sample rate. Acknowledge played audio with `playback.ack`. Finish with
`session.close` and wait for `session.closed`.

Sampling overrides belong to the client under `extra_body.duplexio_sampling`,
with independent `agent` and `user` emission/content settings. Realtime user
transcription defaults to greedy sampling. See the
[native inference notes](https://github.com/duplexio/vllm-omni/blob/duplexio/examples/offline_inference/duplexio/README.md)
for the numerical and state contracts.

### Text-only sessions

Set `extra_body.text_only: true` and `start_role: "user"`, then send
`{"type": "input_text.append", "text": ...}`. The server writes the text one token
per frame into the user cell with audio switched off, the layout DuplexIO uses for
text-only chat training. Keep sending `input_audio_buffer.append` frames (silence is
fine) to advance the reply one frame each. Their audio is ignored and no agent audio
is decoded, so the reply arrives only as `response.audio_transcript.delta`. Training
never fit the agent's speak/wait head on text-only frames, so the server forces the
agent to emit on every text-only frame (the server's own silence frames included)
and the reply runs until you stop sending frames. Each transcript delta carries
`agent_emit_logprob`, the head's own log probability of speaking on that frame. Try it
with `client.py --user-text "..."`; add `--greedy` for greedy agent text.
