# Native/training parity

Semantic equality is required at the **recorded-feature boundary**; small measured
numerical differences are accepted. Earlier exact-equality results below predate
the recurrent-decode simplification and the paged full-attention simplification
(2026-09-10), which reduces full attention to one paged FlexAttention call per
layer plus a query-local diagonal merge and no longer rounds like training's
dense matmul. OPD optimizer updates remain
disabled. This does not certify raw-audio encoding, other compiler stacks,
tensor parallelism, or the complete rollout/teacher/learner update loop.

Both worktrees contain uncommitted integration changes. The checks use the
actual DuplexIO training code on main (base commit
`751cf4f24d9c4f7df3b1271f8f4364bda1c679bc`), not the former isolated prototype.

## Paged full attention, 2026-09-10

Full attention is now one paged FlexAttention call per layer over the native
cache, plus a query-local diagonal merged with the softmax normalizer. The
fork-local attention path is gone: the hand-written self/history merge module,
the runner hook, cache epochs, the K-vector position tail, four model-forward
arguments and four graph buffers were deleted (28 files, 1,062 insertions,
2,003 deletions). Sessions address the cache themselves — audio in a ring at
`frame % audio_ring_frames`, text appended one slot per emitted cell — so
vLLM's own block tables, batching and CUDA graphs run unmodified.

Three real-input trajectories (882 predictions) gave mean conditional agent KL
0.00029-0.00036, maximum KL 0.00217-0.00369, top-1 agreement 0.979-0.986 and
worst emitted-token log-probability difference 0.077-0.099 across repeated runs.
The pre-refactor path on the same conversations measured mean 0.00032, maximum
0.00402, top-1 0.993 and 0.114; its first prediction was exact, this one is not.
Packed replay of the same trajectories gives relative L2 0.0133-0.0149 per
conversation. Bit-exactness with training's dense matmul was given up
deliberately; these differences are accepted, and no long-context OPD stability
claim is made. Cached prefill/decode still matches packed training exactly for
the recurrent layers.

Throughput is unchanged. Alternating passes on one eight-H100 node with the
256-conversation fixture, three warmed rounds each, measured 2,469/2,497/2,508
and 2,517/2,538/2,526 live frames/s before versus 2,433/2,470/2,468 and
2,488/2,500/2,496 after: 1.1% lower, within the spread between nodes. The
compact scan bound still assumes four emitted text slots per frame where
sessions emit roughly one, which is the remaining page-scan headroom; tightening
it would reinstate the per-request slot state this change removed. The GPU suite
passes 211 tests with 2 skips.

Logs: `/dcai/users/thuand/duplexio-vllm-simplify-20260910/run_503325/`.

## Recurrent decode, 2026-09-08

Serving now uses ordinary packed FLA prefill and fused recurrence for six-cell
appends. Only convolution history and FP32 recurrent state remain cached; the
unfinished 64-cell input block and its custom state kernel are removed. Training
GDN is unchanged. OPD is evaluated only after audio warmup, at full influence.

Eight real-input trajectories (2,385 predictions) gave mean conditional agent KL
0.00032–0.00049, maximum KL 0.00585, worst emitted-token log-probability difference
0.1038, and maximum emission-probability difference 0.0134. These numerical
differences are accepted; they are not bit-perfect agreement or a long-context
OPD stability claim. Fixed conditioning/noise FlowMap sampling is unchanged.

On the same eight-H100 node and 256-conversation fixture, three warmed rounds
measured 2,441/2,457/2,451 live frames/s versus 2,130/2,111/2,084 before this change.
This is approximately 16% higher throughput, excluding profiling and startup.
The input is precomputed user features; waveform encoding/decoding is not included.

[All eight traces, stdout/stderr and GPU telemetry](https://wandb.ai/andersthuesen/duplexio/artifacts/profile/h100-recurrent-gdn-rollout-20260908/v0).

The isolated batch-32 GDN append improved from 0.870 ms to 0.217 ms with CUDA
graphs. Against literal FP64 recurrence, relative L2 error was 0.00166 versus
0.00335 for training's chunked kernel. Tests retain exact cache-slot reset/reorder
and eager/graph equality; mathematical accuracy uses the FP64 reference.

Logs and isolated candidate sources:
`/dcai/users/thuand/duplexio-h100-performance-20260908-5MU71p/`.

## Earlier exact H100 results, 2026-09-08

The matched compiler stack below was rebuilt on H100. Training/native equality
required two additional fixes: a fixed per-query Triton self/history merge (both
the self-dot reduction and pointwise merge), and a CTA barrier before shifting
GDN convolution history in place. The latter race appeared with 7/8 requests and
time-major cache strides; the real-width 100-step history regression covers it.

Eight concurrent trajectories match all 2,385 packed-training predictor states
exactly. Single-request full CUDA-graph execution matches all 882 predictor
states from the three original conversations. No tolerance was relaxed.

On one eight-H100 node, 64 independently seeded repetitions of those three real
conversations produced exactly identical text, audio feedback, inputs, masks,
and predictor-row placement across eager/graph execution and repeated rounds.
This is a throughput fixture, not a 64-example quality evaluation.

- Eight eager replicas, eight requests each: 125.6–148.2 live frames/s.
- Eight graph replicas, one request each: 194.2–196.2 live frames/s.
- Eight graph replicas, eight requests each: 304.8–416.9 live frames/s after
  batched heads/input projection and masked cache gathers. Combined-round rate:
  352.1 frames/s, up 80.4% from the one-request graph baseline. The large round
  spread means this is not yet a stable-rate guarantee.
- One H100 with eight requests: 77.26/77.28 frames/s. Larger tested request
  counts (16/32) and a deeper frontend queue did not improve this earlier version.
- After sparse query masks and nonblocking replica polling: 373.079 combined
  frames/s over five rounds through the shared frontend.
- Independent frontend/engine actors: 713.820 combined frames/s over three
  synchronized rounds (729.075/696.142/717.026), on the same 64 conversations.
  All 256 trajectories match exactly; packed replay of the first actor's eight
  conversations matches all 2,316 predictor states exactly.

Use `vllm_omni/deploy/duplexio_opd_h100.yaml`, driver `--concurrency 8`,
`--devices 0 1 2 3 4 5 6 7`, and `--init-timeout 1200`. Each device owns an
independent frontend and engine; concurrency is per actor. These rates include
prefix processing but exclude engine startup and assume prepared causal user-ASR
features and do not include PCM decode or teacher/learner work. Two-node scaling
is not validated here. All 64 trajectories in all three rounds match the eager
reference exactly; packed-training replay of the first eight matches all 2,385
predictor states exactly. Same actual FlowMap conditioning/noise must match;
identical RNG streams across batching schedules are not a required contract.

Independent-actor benchmark: job 502571, `sharded-node8.stdout`,
`sharded-node8.stderr`, `sharded-node8-gpu.csv` in
`/dcai/users/thuand/duplexio-h100-batched-20260908-Lu5Lck/`.

[All-eight-actor profile, logs, source snapshot and telemetry](https://wandb.ai/andersthuesen/duplexio/artifacts/profile/h100-independent-actors-profile-20260908/v0).
The normal multi-actor CLI also passed 192 exact trajectory comparisons. This
profile attributes 943 ms to preprocessing over 64 engine steps, with only 26 ms
of overlapping device work. Batched input-projection graphs are being evaluated
to remove that dispatch bottleneck; larger-batch throughput is not yet certified.

Artifacts and logs reside in
`/dcai/users/thuand/duplexio-h100-throughput-20260908-01LODi/`:
`conv-fixed-scale.stdout`, `node-graph2.stdout`, and their telemetry/trajectory
files. Jobs 502054 and 502109 completed with exact replay checks.

[Benchmark logs, configurations and telemetry](https://wandb.ai/andersthuesen/duplexio/artifacts/benchmark/h100-rollout-throughput-20260908/v0).
[Complete graph profile, stdout/stderr and GPU telemetry](https://wandb.ai/andersthuesen/duplexio/artifacts/profile/profile-graph/v0).
[Earlier eager profile bundle](https://wandb.ai/andersthuesen/duplexio/artifacts/profile/profile-fixed/v0).

## Compiled sampling and training JVP, 2026-09-08

On H100 job 502606, deterministic FlowMap sampling is compiled with
`emulate_precision_casts=True`, then captured per batch size. Noise generation
stays outside the graph. Changed inputs, retained outputs, and in-place weight
updates are covered by regression tests. Real-weight native sampling matches
the training forward under the same compiler precision policy exactly at batch
sizes 1/8/32/64. Compared with uncompiled BF16, the tested maximum difference was
0.0078125; compilation is not bitwise equivalent to every eager configuration.

Text top-k/top-p filtering is compiled separately from request-owned RNG.
Tests check candidate support, probabilities, 100 seeded draws and generator
states. Frame-input fusion preserves embeddings, prefix/live masks and text
ordinals exactly. CPU output transfer waits once per device after submitting
all tensor copies; noncontiguous and non-default-stream cases are tested.

The full engine with these changes matches packed training exactly for all
2,385 saved prediction states across eight concurrent conversations. This
checks replay on the actual recorded audio/text feedback, not regeneration of
that feedback by an uncompiled sampler.

[Complete optimized single-actor trace, stdout/stderr, source snapshot and GPU telemetry](https://wandb.ai/andersthuesen/duplexio/artifacts/profile/h100-fused-frame-sampling-profile-20260908/v0).
The trace covers 64 engine steps / 256 live frames. GPU busy time is 1.210 s
over a 2.154 s span: substantial host-side gaps remain. The separate unprofiled
round measured 108.436 frames/s, not a full-node throughput result.

Training's JVP compilation is validated on the complete LSD objective,
conditioning gradients and every parameter gradient, including masked batches.
Disabling Inductor buffer reuse/in-place buffers for the JVP prevents reuse of
immutable ZeroTensor storage. This is scoped to the tested PyTorch 2.13/cu130
build and should be rechecked after compiler upgrades; no PyTorch patch is used.
For 8,192 frames with batch multiplier four, BF16 loss forward/backward measured
49.793 ms eager, 35.935 ms with the previous compiled-forward/eager-JVP path,
and 20.008 ms with compiled JVP. Worst per-tensor relative L2 error versus eager
was 0.001204. These are isolated FlowMap timings, not whole training steps.
Logs and scripts are in
`/dcai/users/thuand/duplexio-compile-sampling-20260908-IRqNI4/`.

## Earlier RTX 5090 results, 2026-09-07

The real FlowMap 2k checkpoint was loaded by native vLLM, which generated and
recorded its own agent tokens and audio-latent feedback. Training replay consumed
those exact recorded tensors, including user tokens and input-only prefix masks.
Replay uses evaluation mode and full-strength audio influence; OPD must not
silently apply a different influence schedule or corrupt these recorded inputs.

- Short diagnostic: 724 prefix frames plus 12 live audio frames. Every captured
  layer (all 32), all six final cells, agent/tool vocabulary logits and emission
  logits match bit-for-bit. A single cached training prefill plus frame-by-frame
  decode also matches packed training and native chunked prefill/decode exactly.
- Three concurrent conversations: 879 live audio frames, 4,063 total input frames,
  882 prediction rows and 105 emitted agent tokens. Each trajectory matches its
  separate training replay exactly. Mean/max conditional agent KL and maximum
  emitted-token log-probability error are all **zero**.
- The same three trajectories packed into one learner batch also match every
  native prediction row exactly. No sequence padding or regenerated feedback.
- Actual checkpoint emission heads also match batched versus one-row execution
  over the first complete conversation's 1,014 frames.

Conversation IDs and emitted-token counts:
`a2ee9e7f-d6f3-4545-b495-a315a0f5c259` (32),
`bdf144f8-8753-4ab9-abad-b0fbdc53b36d` (38),
`5e9b09a9-3c13-4def-bf16-de295081d99e` (35).

No tolerance was relaxed. Comparing only argmax would not establish this result.
The distribution check is temperature-one conditional content KL; sampling's
top-k/top-p truncation is a separate contract.

The final focused regression run passed **153 tests** (124.77 seconds), covering
adapter and KL gradients, compiled backbone/FSDP2 backward, export, exact numerical
parity, independent request caches, slot resets, tool-only frames, audio eviction,
unowned/NaN cache storage and kernel CUDA-graph replay. This is not a full-engine
CUDA-graph acceptance test. CLI export help and lint/diff checks also passed.

## Reproduced causes and fixes

- GDN: replaying the unfinished 64-cell block previously preserved training's
  reduction order. The faster recurrent path above supersedes that requirement.
  State writes remain outside AOT because differently typed cache views share storage.
- Full attention: preserve chronological text/audio key order, block alignment,
  request ownership, query-local self attention and bounded audio eviction.
  Unowned cache bytes must never become keys or padding values.
- Projection arithmetic: fixed Quack GEMMs for QKV, adapters and vocabulary heads,
  including the learner's BF16 KL projection. Scratch alignment does not change
  vocabulary size. Explicit compiler boundaries preserve gate/rotation rounding.
- RoPE: the training loader casts its inverse-frequency buffer to BF16; vLLM
  previously rebuilt FP32 frequencies. Using the BF16 frequencies reproduced the
  first full-attention difference exactly. Export v5 carries the actual runtime
  frequencies and scaling. Native loading does not regenerate them or rely on
  vLLM's intentionally discarded HF `rotary_emb.inv_freq` convention.
- Audio normalization: Inductor's RMS reduction varied between one row and a
  batch at the checkpoint's 1,280-wide adapter. A sub-micro FP32 difference crossed
  BF16 rounding boundaries and propagated through Qwen. Both paths now use
  row-local Quack FP32 normalization; CPU retains the ordinary PyTorch operation.
- Compiler versions: identical FLA source differed at the GDN output kernel under
  Torch 2.11/Triton 3.6 versus Torch 2.13/Triton 3.7.1. Matched-stack actual-weight
  probes agree at all 22 captured intermediates/outputs.

## Validated runtime and artifacts

PC root: `/home/anders/duplexio-opd-notvTp`.
Native environment: `vllm-aligned-QICk4l/venv`.
Training snapshot: `main-integration-tfQjo5`; reference environment:
`.venv-reference`.

vLLM source tag v0.26.0 (`568afb3a1`) was built against Torch 2.13.0+cu130,
Triton 3.7.1, Quack 0.5.3, Cutlass DSL 4.6.0.dev0 and FLA/core 0.5.1.
Transformers is commit `b3d7e8c9d4e078a5e6c09a9d67e22dcadc2df4b8`.
All nine packaged extension hashes match the completed source build; dependency
validation passes. The wheel is in `vllm-aligned-QICk4l/source/dist/`.
The stock Torch 2.11 vLLM wheel is **not** an accepted substitute.

Checkpoint: `checkpoints/flowmap-2k-v5`.
Native outputs: `rollout_parity_20260907_v16` (short),
`rollout_parity_20260907_long_v1` (concurrent).
Logs: `reference_parity_20260907_v16.log`,
`reference_parity_long_v1_00000{0,1,2}.log`,
`packed_parity_20260907_long_v1.log`.
Logs are also archived on the login node at
`/dcai/users/thuand/duplexio-native-parity-20260907-9ChhSL/`, including
`parity_regressions_20260907_final.log` and the reproduced failing normalization
test. No Slurm or training job was started or stopped for this integration check.

## Reproduce

Put the training checkout and this repository on PYTHONPATH. Use the matched
native environment for generation and the training environment for replay.

```bash
python -m examples.offline_inference.duplexio.run \
  EXPORT PREPARED_INPUTS OUTPUT_DIR \
  --policy-version VERSION --record-hiddens --concurrency 3

python -m examples.offline_inference.duplexio.check_backbone_parity \
  EXPORT OUTPUT_DIR/trajectory_000000.pt --cached

python -m examples.offline_inference.duplexio.check_packed_replay \
  EXPORT OUTPUT_DIR/trajectory_000000.pt \
  OUTPUT_DIR/trajectory_000001.pt OUTPUT_DIR/trajectory_000002.pt
```

Add `--layer-capture CAPTURE.pt` to a single-conversation generation and replay
check for the first sixteen engine forwards. This is opt-in diagnostic tensor
capture, not a throughput measurement.
