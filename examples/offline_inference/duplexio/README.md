# DuplexIO native inference

## Ownership

The application runners now live in the DuplexIO training repository:
`duplexio.scripts.run_opd_actor`, `duplexio.scripts.run_rollouts`, and
`duplexio.scripts.run_self_play`. Dataset preparation, simulated tool execution,
trajectory assembly, and trainer-specific weight exchange live there too.
The inference implementation in this repository is shared by deployed serving
and rollout callers; it does not implement OPD/RL objectives or eligibility rules.
Actor deployment presets are in DuplexIO's `configs/rollout/`.

Worktree branch: `codex/duplexio-opd-batching`.

## Current training alignment, 2026-09-18

Serving targets version-seven exports and the current training model. Voice prompts
are pinned reference-audio rows, distinct from live audio. Both user and agent
encoders consume context silence in frame order, including text prefixes and tool
results. Generated agent PCM advances the input encoder's history; sampled audio
representations remain the model's live feedback.

The entire initial prefix is one append: speaker-reference frames followed by
system-token frames. The user encoder consumes silence for that whole timeline;
the agent encoder consumes the reference waveform followed by system-frame
silence. The scheduler splits this prefix into whole six-cell frames under its
token budget. Each chunk passes through audio encoding and the backbone, carrying
codec, ASR, KV, and recurrent state into the next chunk. Sampling begins only when
the complete prefix has been consumed; trajectory recording retains every chunk.
Only its last row is sampled, then rollout proceeds one live frame at a time.
Offline rollouts, online sessions, and self-play use the same prefix payload.

Offline rollouts submit one input per conversation and collect its output before
submitting the next. Conversations run concurrently, and tool requests run
asynchronously. There is no per-conversation input prefetch queue or wall-clock
audio pacing; prerecorded frames advance as soon as each prediction is collected.

Recordings retain the actual user features and separate `audio_mask` (live rows)
from `prompt_frames` (pinned reference rows). The actor and trainer must be updated
together; older recordings without `prompt_frames` must be regenerated. Replay
means feeding these recorded inputs through the training model to recompute its
predictions. Live weight refresh excludes only the unused system/user per-cell output
projections. The trained full-frame user projection and user emit head are loaded
and refreshed along with the other policy weights.

See [PARITY.md](PARITY.md) for current tests and their scope, including the
remaining BF16 ASR numerical/quality caveat. Historical throughput numbers below
predate this change. Maintaining encoder history requires decoding agent audio
even when output PCM is suppressed. The complete OPD update loop remains
unvalidated with the new prefix path.

## Learned user-token feedback

Sampling settings belong to the caller, not the model export. Offline entry points
require `--sampling-config` with a JSON object containing independent `agent` and
`user` settings:

```json
{
  "agent": {
    "emission": {"temperature": 1.0},
    "content": {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
  },
  "user": {
    "emission": {"temperature": 1.0},
    "content": {"temperature": 1.0}
  }
}
```

This explores both user actions during joint ASR RL/OPD. Omitted content filters
mean full-vocabulary sampling. Temperature zero is greedy, independently for
emission and content. For deterministic ASR evaluation or serving, set both user
temperatures to zero. Realtime sessions accept the same object under
`duplexio_sampling` (with optional `audio` settings and `seed`); their default user
temperatures are zero. There is no sampling `mode` or adjustable emission threshold.

The user stream uses `user_emit_head(h[t].flatten())` and the shared vocabulary
head applied to `user_token_projection(h[t].flatten())`. Each predictor consumes
all six cells of frame `t`; its sampled user ID (or silence for a wait) is the
input at frame `t+1`, matching training's `target_row - 1` alignment. Prompt and
intermediate text-context rows do not sample. The last text-context row can
predict the first following live frame. A later context burst replaces any
pending user feedback.

RNN-T transcript commits and externally supplied transcript IDs no longer drive
feedback. FastConformer still encodes user audio, including context silence, and
its features still feed the backbone. Prepared conversation transcripts remain
available for dataset metadata and first-utterance stopping, but are not injected
as model inputs. Exports must include both user heads; older artifacts missing
these parameters must be exported again. This change adds no RL objective.

Rollouts retain the following aligned vectors, one entry per `prediction_rows`:

| Field | Meaning |
| --- | --- |
| `sampled_user_emits` | Sampled emit (`true`) or wait (`false`) |
| `sampled_user_ids` | Emitted token ID, or silence on wait frames |
| `user_emit_logprobs` | Log probability of the sampled Bernoulli decision, including waits |
| `sampled_user_logprobs` | Conditional content log probability; zero for waits |
| `user_action_logprobs` | Emit log probability plus conditional content log probability |
| `user_action_eligible` | The following row exists, is live, and consumes this decision |
| `user_token_eligible` | Eligible action that emitted a content token |
| `row_versions` | Active worker policy version when all decisions on this predictor row were sampled |

Probabilities use the actual sampling distribution: temperature, top-k, top-p,
and silence exclusion are reflected in content probabilities; emit probabilities
use the exact FP32 Bernoulli probabilities. Greedy decisions have log probability
zero. Context-replaced and terminal predictions retain their sampled decisions and
probabilities with eligibility false. The worker publishes its new policy version
only after staged weight copies finish; neither user-head parameter storage nor
CUDA graph addresses change.

Tool decisions also record `tool_emit_logprobs` and `sampled_tool_logprobs`, aligned
with `sampled_tool_ids`, `prediction_rows`, and `row_versions`. `tool_emit_logprobs`
always scores the chosen emit/wait decision using the raw head: `logsigmoid(z)`
for emit and `logsigmoid(-z)` for wait. This includes forced continuations,
required starts, disabled-tool waits, and greedy decisions, without applying
sampling temperature. It is a model score, not the behavior log probability.
Content log probabilities include the grammar mask and temperature/top-k/top-p
filtering. Wait frames have zero content log probability, and greedy content
choices have zero behavior log probability. No RL objective is added.

Validation used one full eight-H100 node per Slurm allocation, submitted with
`--export=NIL` and explicit `srun` exports. Step `507641.1` passed 31 training-side
export/transport/schema tests; job `507643` passed 128 serving tests (one CPU-only
compile case was skipped). FP32/BF16 head logits and the training loss reductions
match; tests cover CUDA sampling, causal feedback across commits, masks/versions,
NCCL overlap, and unchanged CUDA graph parameter addresses. These are small-model
checks; no full-checkpoint rollout or RL objective was run.

For numerical parity, generate reference cases with the training environment's
Python using `tests/model_executor/models/duplexio/user_head_reference.py OUTPUT.pt`
from this repository. Then run the serving tests with
`DUPLEXIO_USER_HEAD_REFERENCE=OUTPUT.pt` in the serving environment. GPU execution
must remain inside a full-node allocation with the export settings above.

## Staged policy updates

The OPD actor receives each trainer push into a reusable GPU staging buffer on a
background thread and a separate CUDA stream. Existing sessions continue decoding
under the current policy while NCCL transfers the next version. Budget additional
GPU memory equal to the transmitted tensors (frozen modules are not transmitted).
The first push allocates this storage; subsequent pushes with the same metadata
reuse it.

After receipt finishes, the actor closes its rollout gate, drains outstanding
rows, and commits through the native weight loader. This is atomic with respect
to decoding: no prediction sees an intermediate set of weights. Commit copies
into existing parameter storage to preserve CUDA graph addresses; it is not a
zero-copy pointer swap. Sessions retain their caches, and the policy version is
advanced only after the commit finishes. A load failure leaves decoding stopped
and requires restarting the actor; partial copies are not rolled back.

The worker RPCs are `start_policy_weight_update(version, weights)`,
`policy_weight_update_ready(version)`, and `commit_policy_weight_update(version)`.
Only one version can be pending. The actor sends the existing `READY` message
after staging starts and `UPDATED` after commit. Consequently the trainer's
`push_weights` duration includes overlapped transfer time, not just actor pause
time. The blocking `receive_policy_weights` RPC remains available for older
callers that drain before receipt.

Validation: Slurm job `507639` on `dgx087` passed all 23 focused tests using two
H100s. The two-GPU test uses the training repository's actual NCCL `PolicyGroup`,
replays a captured graph during receipt, verifies both committed versions through
that same graph, and checks staging-buffer reuse. These tests exercise transport
and commit with a small model; a full-checkpoint OPD rollout remains unvalidated.

## Historical implementation notes, 2026-09-07 to 2026-09-10

The following results describe older exports and training implementations. Claims
about speaker embeddings, masked initial audio representations, version-four/five
exports, exact equality, and throughput do not describe the current contract.

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
The DuplexIO harness continuously refills independent sessions, recording actual
consumed inputs and explicit predictor rows. Serving samples user tokens and feeds
them into the following live frame; callers do not inject reference transcripts.

The training-side `duplexio/opd.py` packs recorded trajectories without padding
and selects only emitted-agent predictor rows. Its conditional forward-KL helper
passes dense-reference value/gradient tests, including student-only vocabulary
entries. Teacher context construction and the optimizer/update loop live in the
training repository, not in the inference implementation.

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

## Historical integration checklist (2026-09-07)

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
python -m duplexio.scripts.prepare_self_play \
  /path/to/checkpoint /path/to/conversations.jsonl /tmp/pairs.pt \
  --agent-voice /path/to/agent.wav --user-voice /path/to/user.wav

python -m duplexio.scripts.run_self_play \
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
