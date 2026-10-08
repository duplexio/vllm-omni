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

## Test ten spoken questions

The inference test starts the server and makes ten spoken questions with Kokoro on
the CPU. It opens a new user-first realtime session for each question. The
questions go from simple arithmetic to reasoning, and the last one is an MMLU-style
biology question.

Install Kokoro and its English tokenizer assets in the serving environment, then
run the test:

```bash
uv pip install kokoro==0.9.4 'misaki[en]==0.9.4'
uv pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
bash examples/online_serving/duplexio/test_inference.sh
```

The test writes to `outputs/duplexio-inference/<timestamp>/`:

- `server.log`, `server-command.txt`, `results.json` and the voice reference.
- One numbered folder per question, with `question.txt`, `input.wav`, `answer.wav`
  and `result.json`.

The script stops the server when it finishes. If a request fails, the script
continues with the next question and exits with a nonzero status at the end. A
request counts as successful when the server returns audio and text. You still
have to check answer correctness and audio quality yourself.

To change the prompts, edit `examples/online_serving/duplexio/questions.json`. You
can also give an output folder and a question file as the first two script
arguments. The script passes all further arguments to `run_questions.py`, for
example `--instructions`. A question can also be an object with `text` and an
`answer` letter, as in `mmlu_questions.json`. The script then records the predicted
letter and prints the number of correct answers.

- `KOKORO_ASSETS=/path/to/kokoro` uses local Kokoro assets.
- `DUPLEXIO_PORT` sets another server port.
- `--text-only` types each question instead of speaking it. See
  [Text-only sessions](#text-only-sessions).

Each question gets thirty seconds of silence after its audio ends. The script pins
the v7 checkpoint revision.

## Run the MMLU benchmark

This benchmark scores the DuplexIO text channel on 100 MMLU questions. It types the
questions instead of speaking them. You need the [serving environment](#install)
and Kokoro from the [spoken-question test](#test-ten-spoken-questions). Kokoro still
makes the speaker reference.

1. Score DuplexIO. The script starts the server, runs every question in a text-only
   session and stops the server:

   ```bash
   bash examples/online_serving/duplexio/test_inference.sh outputs/mmlu100-duplexio \
       examples/online_serving/duplexio/mmlu100_questions.json \
       --text-only --greedy --response-seconds 12 --instructions "You are a helpful voice assistant."
   ```

2. Optional: score the backbone on the same questions as a reference:

   ```bash
   python examples/online_serving/duplexio/mmlu_baseline.py \
       --questions examples/online_serving/duplexio/mmlu100_questions.json --output outputs/mmlu100-qwen.json
   ```

3. Read the last line of each run. It shows `Correct letters: N/100; centered score S`.
   The replies and letters for each question are in
   `outputs/mmlu100-duplexio/results.json` and `outputs/mmlu100-qwen.json`.

On the DCAI cluster, submit step 1 from the repository root instead:

```bash
sbatch examples/online_serving/duplexio/questions.sbatch \
    "$PWD/examples/online_serving/duplexio/mmlu100_questions.json" \
    --text-only --greedy --response-seconds 12 --instructions "You are a helpful voice assistant."
```

The job log is `outputs/duplexio-inference/<job id>.log`. The results go to
`outputs/duplexio-inference/<job id>/`. For a different sample, use your own
question file.

## How the MMLU benchmark works

Spoken questions mix several problems:

- The agent takes its turn at the first pause and misses the options.
- The user ASR garbles symbols.
- The agent audio can drift away from its text.

The benchmark types the questions, so it measures only what the model knows. Each
question runs in a [text-only session](#text-only-sessions):

- The question goes into the user text cell, one token per frame, with audio off.
  DuplexIO's text-only chat training uses the same layout.
- The server forces the agent to emit on every frame. The speak/wait head never
  trained on text-only frames. Its probability follows word pacing, not the end of
  an answer. It is low before a new word and high inside a word. So the model
  cannot end the reply itself.
- `--response-seconds` ends the reply instead. At one token per 80 ms frame, 12 s
  is about 150 tokens. Replies that run longer tend to loop.
- `--greedy` takes the most likely agent token instead of the trained sampling.
- `answer_letter_in` in `run_questions.py` finds the first option letter that the
  reply commits to.
- The centered score is `(accuracy - 1/4) / (1 - 1/4)`. Random guessing gives 0
  and a perfect run gives 1.

`mmlu_baseline.py` asks the backbone through vLLM. It uses the same system prompt,
questions and 150-token budget, with thinking off and greedy sampling.
`mmlu100_questions.json` is a seeded sample of the `cais/mmlu` test split.

## Realtime protocol

Connect to `/v1/realtime?duplex=1&model=duplexio/duo-4b&autostart=0`. Then send
`session.update` with audio and text modalities and an `extra_body` that contains:

- `full_duplex: true`
- `auto_response: true`
- `start_role: "user"` or `"agent"`
- the voice: `ref_audio_data` (base64 little-endian float32 PCM),
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

Set `extra_body.text_only: true` and `start_role: "user"`. Then send
`{"type": "input_text.append", "text": ...}`. The server writes the text into the
user cell, one token per frame, with audio off. DuplexIO's text-only chat training
uses the same layout.

Keep sending `input_audio_buffer.append` frames to move the reply forward, one frame
per event. Silent frames are enough. The server ignores their audio and decodes no
agent audio. The reply arrives only as `response.audio_transcript.delta`.

Training never fit the agent's speak/wait head on text-only frames. So the server
forces the agent to emit on every text-only frame, including the silence frames
that the server adds itself. The reply continues until you stop sending frames.
Each transcript delta carries `agent_emit_logprob`, the head's own log probability
of speaking on that frame.

To try it, run `client.py --user-text "..."`. Add `--greedy` for greedy agent text.
