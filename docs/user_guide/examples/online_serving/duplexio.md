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
after its complete audio, and the v7 checkpoint revision is pinned.

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
