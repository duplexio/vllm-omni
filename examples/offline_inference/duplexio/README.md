# DuplexIO batched OPD implementation

Worktree branch: `codex/duplexio-opd-batching`.

## Status

Native vLLM now matches packed training and cached prefill/decode exactly at the
recorded-feature boundary on validated H100 and earlier RTX 5090 runtimes. Eight
concurrent H100 trajectories match one packed learner batch. See [PARITY.md](PARITY.md)
for the exact scope, required compiler stack, results, and reproduction commands.
One independent frontend/engine actor per H100 measures 714 combined live frames/s
over three synchronized rounds (696–729), using eight conversations per GPU.
All 256 trajectories matched the reference exactly. See PARITY.md for the fixture
and measurement boundary; this is not an end-to-end OPD throughput claim.
Raw-audio end-to-end validation, other runtime
combinations, and the complete OPD update loop remain uncertified. Updates stay off.
New exports require version five, which preserves the model's actual RoPE
frequencies; the version-four results below are historical.

Implemented and checked:

- A native FP32 FlowMap matching the current training parameter layout and math.
  It accepts an entire batch of conversation conditions and explicit Gaussian
  noise; session RNG ownership belongs to the eventual scheduler.
- Training-versus-native FlowMap forward/sample checks, including nonzero learned
  modulation, noise-variance temperature, multi-step integration, batch permutation,
  slot replacement, compiled FP32 sampling, and exact FP32/BF16 CUDA-graph replay.
- Cross-repository parity for the existing native depth sampler, including
  batched stochastic sampling and independent greedy conversations.
- The training repository's version-four export now describes either
  Pocket-Mimi/FlowMap or quantized-Mimi/depth, never both. It includes the frozen
  FastConformer/RNN-T weights and processor, latent normalization, optional speaker
  LDA tensors, and the current full-frame emit/direct audio-row contracts.

The native model now accepts version four and constructs the current full-frame
emit heads, direct audio-cell conditioning, FastConformer/RNN-T, paired codec/head
variants, and optional speaker projection. It no longer constructs a user emit
head, user Mimi embedding, output adapter, or audio skip. Prefix audio inputs now
use training's masked initial representations rather than encoded silence.

A real 2k checkpoint was exported in job 501755 to
`checkpoints/flowmap-2k-v4`. Strict native GPU reload now passes on the RTX 5090:
4,891,587,940 parameters, 9,875,837,440 peak allocated bytes during loading.
The frozen ASR subsystem also passes separate strict reload (618,084,865 parameters).
Its exported mel/window buffers are persistent in the serving module so vLLM can
load them. H100 verification remains queued as job 501762.

Further checks: 40 migrated configuration/component/prefix/state tests pass.
Additional CPU checks pass for text sampling and bounded Pocket codec
attention. The serving argmax path previously sampled emission stochastically;
a failing parity test reproduced this and the fixed sampler matches training for
argmax/top-k/top-p. Pocket's native streaming kernels matched the training
implementation exactly in FP32/BF16, but the FP32 batch-vs-individual comparison
hit convolution rounding differences (maximum observed 4.06e-5). Both FP32 and
BF16 codec comparisons pass on the 5090 with TF32 convolution disabled. The 5090
passed 70 focused sampler, codec, replay, and conditional-KL tests.

Prepared ASR frames use the existing typed `embed.speech_feat` transport field and
are transferred together before per-request preprocessing. Real streaming encoder
outputs are FP32, even when its weights are BF16; no output cast is introduced.
The offline driver continuously refills independent sessions, recording actual
consumed inputs and explicit predictor rows. Its full-engine execution is still
under validation. Live frames carry their user token explicitly. The old PCM-only
realtime client has not yet been migrated to provide that user-token contract.

The training-side `duplexio/opd.py` packs recorded trajectories without padding
and selects only emitted-agent predictor rows. Its conditional forward-KL helper
passes dense-reference value/gradient tests, including student-only vocabulary
entries. Teacher context construction and a complete optimizer/update loop remain
unimplemented; this is not yet an end-to-end OPD trainer.

The synchronous self-play runner now opens two native DuplexIO sessions per case:
one with the Convogen agent prompt and one with the Convogen user prompt. After
prefill, both sessions advance on the same frame clock and exchange the previous
80-ms decoded PCM frame. The agent trace is directly replayable by the training
package, while the user trace and per-frame token events are retained for
diagnostics. Tool calls currently stop a pair explicitly because a tool-result
simulator is a third participant; no fabricated result is inserted.

Validation on 2026-09-07: 63 tests passed on an H100 (job 501752); 15 export
tests passed in the training repository. The real 2k checkpoint from training
job 501620 passed 27 FP32 and 27 BF16-autocast sampling cases (jobs 501749 and
501751). Native-versus-training maximum error was zero in both; batched-versus-
individual maximum error was 6.32e-6 in FP32 and zero in BF16.

Execution choice: full-graph mode now compiles deterministic FlowMap sampling
with `emulate_precision_casts=True` before CUDA-graph capture. Explicit noise
remains request-owned. Native sampling matches training with the same compiler
precision policy; this is not a claim of bitwise equality to uncompiled BF16.
Text filtering and frame-input construction are also compiled. See PARITY.md
for the numerical checks, measured scope, and remaining CPU overhead.

## Remaining work, in order

1. Extend whole-engine checks to a current quantized-Mimi/depth export and raw
   audio feedback, including acoustic-delay state and longer tool conversations.
   The H100 throughput result uses prepared causal user-ASR features.
2. Reduce CPU-bound preprocessing and improve batch occupancy beyond the validated
   independent actors. Input-projection graphs and larger batches are undergoing
   full-engine parity and throughput checks; they are not yet accepted results.
3. Connect the synchronous pair artifacts to the trainer's SFT/OPD update round,
   then add versioned GPU weight refresh between rounds. Never change weights
   during a conversation. Keep the sampling and KL-direction contracts explicit.
4. Add a separate tool-result participant for scenarios with tools; do not hide
   tool calls from replay.
5. Measure complete OPD throughput, including encoding, teacher work, and policy
   refresh, on broader workloads and two nodes. Preserve exact replay checks and
   upload complete traces, stdout/stderr, and telemetry to W&B.

## Checks

Prepare Convogen prompts with the same Qwen system-content extraction used by
training, then run the two-model synchronous clock (split the GPUs so each
engine has its own replica):

```bash
PYTHONPATH=/path/to/duplexio:$PYTHONPATH \
python examples/offline_inference/duplexio/prepare_convogen_pairs.py \
  /path/to/checkpoint /path/to/conversations.jsonl /tmp/pairs.pt

python examples/offline_inference/duplexio/run_self_play.py \
  /path/to/checkpoint /tmp/pairs.pt /tmp/self-play \
  --policy-version checkpoint-2k \
  --agent-devices 0 1 2 3 --user-devices 4 5 6 7 \
  --concurrency 8 --max-frames 128
```

The runner intentionally refuses an emitted tool call until a third-party tool
simulator is supplied. That keeps the replay context honest while the first
synchronous speech-policy pipeline is validated.

Use a working vLLM-Omni environment and put both checkouts on `PYTHONPATH`:

```bash
python -m pytest -q \
  tests/model_executor/models/duplexio/test_flowmap.py \
  tests/model_executor/models/duplexio/test_training_flowmap_parity.py \
  tests/model_executor/models/duplexio/test_training_depth_parity.py

python examples/offline_inference/duplexio/check_flowmap_parity.py \
  /path/to/training/checkpoint --autocast
```

The cross-repository tests explicitly skip if the training package is unavailable;
CUDA checks skip without a GPU. The checkpoint checker requires CUDA and loads
only the FlowMap weights. Neither command is a throughput benchmark.
