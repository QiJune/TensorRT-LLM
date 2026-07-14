# E5b exit report (gate ①c)

Status: cut-over delivered on branch `prototype_1`; V0-scope eligible OpenAI
streaming traffic runs contract-native. This report records the exit
evidence and enumerates the remaining Step-② (remote detach) work honestly —
the "transport-only" posture below is claimed for the **request path only**.

## Cut-over evidence

- **Zero rank-0 `GenerationResult` for contract traffic**: asserted
  programmatically (`test_e5b_cutover.py::TestContractNativeSubmit::
  test_no_rank0_generation_result`); the router binding is the rank-0
  request state. Legacy traffic on the same proxy is untouched
  (`TestExclusiveRouting`, `TestConcurrentMixedTraffic`).
- **Exclusive routing**: contract-owned client ids are claimed before the
  legacy `_results` lookup and never enter legacy delivery or the
  pop-on-final logic; ownership survives tombstone eviction, so late frames
  are absorbed, never legacy-delivered.
- **Shutdown symmetry**: every `pre_shutdown` entry poisons contract streams
  typed AND issues best-effort runtime cancellation alongside the legacy
  abort-all (`TestShutdownSymmetry`).
- **Flag**: `TorchLlmArgs.experimental_engine_client` (prototype status,
  `api_stability` suite green with the reference update); the env var
  overrides in both directions, malformed values fail closed
  (`TestFlagResolver`, four-row precedence matrix).

## Real-GPU validation (the approved reproducible job)

Artifact: **one merged** `gpu-e2e-report.json` across both pytest
invocations (main suite + destructive worker-crash test), containing EVERY
case result (plain streaming, stop-token, stop-string boundary, logprobs,
prompt logprobs, abort, stream close, usage, shadow overhead, measurement,
worker crash) AND the per-test pytest verdicts (recorded by the suite's
`conftest.py` hook); the loop's copy sits in the run directory. The
artifact's `commit` field records the **committed** code the suite ran on;
any later commits touch only documentation regenerated from this artifact.
Pinned configuration recorded in the artifact: commit SHA, model
`Qwen2.5-0.5B-Instruct` with a content-pinned revision
(`model_revision: sha256:0d1a2291cb93371d` over config + tokenizer
manifest), NVIDIA H100 (SM90 build, `cuda_architectures=90-real`),
execution backend `pytorch` and attention backend TRTLLM (recorded as
separate fields), greedy sampling (`top_k=1`), TP1 over the IPC proxy, KV
block reuse disabled (with partial block reuse, a repeated prompt gets
truncated context logits and therefore truncated prompt logprobs — a known
runtime limitation that the legacy logprobs harness also works around by
disabling reuse; the contract path surfaces that truncation as a typed
`logprob_mismatch` failure instead of delivering silently short values).
Reproducible commands are in the `test_gpu_e2e.py` module docstring.

Results (11/11 green across the two invocations, per-test verdicts in the
artifact):

- plain-streaming parity, stop-token, stop-string crossing a token
  boundary, logprobs, **prompt logprobs** (position-by-position parity with
  the legacy map-shaped output), abort mid-stream, usage — all parity
  cases green;
- **stream-close cancellation is observed directly**: a spy on the
  router's abort hook records the runtime cancellation issued by
  `aclose()`, and the engine stays healthy and serviceable afterwards;
- **worker crash**: the test first gives the executor's own detector a
  120 s window after SIGKILL with **no** test interference. Recorded
  evidence (artifact `worker_crash` case): `mpi_futures` never complete
  and `_error_queue` stays empty (`native_signal_observed: false`). A
  dedicated 300 s probe (`crash-native-detection-probe.{py,log}` in the
  run directory) confirms this definitively: the worker process is
  observed dead at t=1.0 s, the mpi4py `_manager_spawn` pool thread stays
  alive, and for the full 300 s every future reports `done=False` with an
  empty error queue while the proxy's error monitor polls with nothing to
  observe — the propagation gap is in the mpi4py-futures layer beneath
  the executor, and a companion probe
  (`crash-legacy-gap-probe.{py,log}`) shows the LEGACY path has the
  identical gap (a post-kill legacy request receives no result or typed
  failure either), i.e. this is pre-existing runtime behavior, not a
  contract-path deficiency, and its remediation belongs to the Step-②
  process-lifecycle work already inventoried below. The test then
  exercises everything the executor CAN own — the failure channel
  (`_error_queue` → monitor → `pre_shutdown` → typed poisoning) — via a
  **recorded, justified** injection: typed ending in **4.85 s**, inside
  the frozen DEC-4 bound measured from the point the failure channel
  observes the crash. The mode, window, and justification are stored in
  the artifact so this fallback is auditable.
  (Harness note: after the test passes and the artifact is written, the
  pytest interpreter can hang at exit on the SIGKILLed MPI runtime's
  teardown; the documented outer `timeout` reaps it and its 124/143 exit
  code is expected — the verdict is the pytest summary and the artifact.)

## Shadow-overhead report (draft DEC-5, report-only)

Post-cutover, routing is exclusive — a dual-consumption shadow tap no
longer exists — so the shadow number that still exists is the cost the
attached contract machinery imposes on **legacy** traffic (the per-response
contract-population check in the dispatch loop). Measured at concurrency 8
(2 warm-ups, 10 reps per arm, arms sequential because attach is
irreversible): **+2.79 %** legacy rank-0 CPU per streamed token in the
final run (+0.02 % in the earlier run — the sequential-arm design makes
this metric drift-sensitive, which is why it is report-only per DEC-5 and
carries no claim; artifact `shadow_overhead` case has the raw per-rep
values for both arms).

## Rank-0 measurement (contract DEC-5 / plan DEC-3 rule)

The pre-registered loop protocol of `E5B_CUTOVER_DESIGN.md` §7 was
executed: streaming concurrencies {1, 8, 32}, two discarded warm-up pairs,
alternating legacy/contract pairs per concurrency (protocol minimum 10; the
final run uses 30), metric = rank-0 process CPU seconds per emitted token
id, decision statistic = median paired delta with a percentile-bootstrap
two-sided 95 % CI (2000 resamples, fixed seed). Full disclosure: the
protocol ran twice — a first 10-pair run (c=1 −19.3 % [−50.5, −27.5] µs;
c=8 −1.25 % [−2.54, +0.19] µs; c=32 −5.5 % [−4.53, −2.58] µs) and this
final 30-pair run on the committed final code; the c=8 upper bound crossed
zero in BOTH runs, so more repetitions did not (and were not used to)
manufacture significance. Raw per-repetition values live in the artifact
(`rank0_measurement.rows`); the table below is **generated from the
artifact** by the test itself (`rank0_measurement.md`) and inserted here
mechanically, never hand-copied:

<!-- BEGIN generated from rank0_measurement.md -->
| concurrency | pairs | legacy median | contract median | Δ median | Δ% | 95% CI of Δ (bootstrap) |
|---|---|---|---|---|---|---|
| 1 | 30 | 189.18 | 157.45 | -33.81 | -17.87% | [-36.82, -28.80] |
| 8 | 30 | 60.80 | 60.01 | -1.16 | -1.91% | [-2.31, +0.30] |
| 32 | 30 | 63.56 | 60.79 | -3.06 | -4.82% | [-3.60, -2.28] |
<!-- END generated from rank0_measurement.md -->

Gate rule application ("upper 95 % CI bound must not show a regression,
otherwise an explicit written acceptance"):

- concurrency 1: upper bound −28.80 µs/token → **pass** (clear
  improvement, CI does not cross zero);
- concurrency 32: upper bound −2.28 µs/token → **pass**;
- concurrency 8: median delta −1.16 µs/token (−1.91 %) but the upper bound
  is **+0.30 µs/token (+0.49 %)** — the CI marginally crosses zero, so the
  automatic gate does not pass at this point and, per the pre-registered
  rule, this concurrency **requires an explicit written acceptance**
  (PENDING — to be granted or refused by the plan owner; not
  self-accepted here). The point estimate is an improvement in both runs
  and no regression is detectable at any concurrency.

No performance *improvement claim* is made where the CI crosses zero
(concurrency 8). The structural change this loop stands on is the removed
rank-0 `GenerationResult` construction/registration per contract request.

## Flag-off sweep (AC-8 final evidence, Round 1)

Flag-off (`TLLM_EXPERIMENTAL_ENGINE_CLIENT` unset), `LLM_MODELS_ROOT`
populated (Qwen2.5-0.5B-Instruct + TinyLlama-1.1B-Chat-v1.0):

- `tests/unittest/executor/` (excluding `engine_client/`, **including**
  `test_rpc.py`, `test_rpc_proxy.py`, `test_rpc_worker.py`): **206 passed /
  1 failed / 8 skipped**. The single failure
  (`test_rpc.py::TestRpcCorrectness::test_incremental_task_async`) fails
  identically at the base commit with base `tensorrt_llm/` sources —
  pre-existing, base-equivalence recorded.
- `tests/unittest/executor/engine_client/` (CPU): **370 passed** clean.
- `tests/unittest/llmapi/test_llm_args.py`: **190 passed / 3 failed** when
  run concurrently with the GPU-holding executor sweep; the 3 failures
  (`test_build_config_from_engine`, `test_runtime_sizes`,
  `test_model_kwargs_with_num_hidden_layers`) all **pass in an isolated
  rerun at head** with the GPU idle (and also pass at the base commit) —
  GPU-contention flakes of the concurrent run, not code.
- `tests/unittest/api_stability/`: **64 passed**.

## Remaining Step-② work (the honest inventory)

Request path — enumerable as transport only:

- Socket + codec-on-wire delivery (`encode`/`decode` are frozen and
  round-trip-tested; nothing is "in-process only").
- ROUTER/DEALER ingress, `(client_identity, request_id)` identity and the
  three-ID mapping; `event_seq` end-to-end delivery semantics (stamped from
  V0; e2e duplicate/gap detection is ② scope per contract draft DEC-3).
- Ready-handshake delivery of `FrontendModelContext` (built locally today;
  consumers already read only through it on the contract path).
- Heartbeat/liveness, bounded queues + credit backpressure, and
  disconnect/crash semantics symmetric across frontend / engine server /
  worker.
- Process lifecycle: engine process owns the MPI session; shutdown
  ordering; the serve entrypoint.

NOT transport-only — server-wide surfaces still coupled to the in-process
`LLM` (unchanged for legacy and fallback traffic):

- Engine-side sampling normalization (V0 deliberately runs it
  frontend-side; the remote target moves it after receipt).
- The serve layer's remaining reaches: ~29 `generator.args.*` reads, the
  `generator._executor.resource_governor_queue` reach, `_hf_model_dir`,
  and the `_torch.pyexecutor.config_utils` import — all on the legacy/
  control paths, not the contract request path.
- Control endpoints and model lifecycle: health/health_generate, metrics,
  KV-cache events surface, weight-update/memory operations, cluster
  registration, disagg role endpoints.
- Non-streaming and the public `LLM` API (stay legacy in V0 by decision),
  ineligible-request fallback (requires the in-process legacy path by
  design), chat templating and input preprocessing (server-side).

## ②-readiness guardrails (held throughout V0)

- Every wire type round-trips the strict codec (golden fixtures checked in).
- The `EngineClient`/`EngineService` surfaces expose no proxy or executor
  internals; frame payloads and binding state are primitives-only.
- `EngineService` is factored so a remote transport terminator fronts it
  without internal changes (see `E5B_CUTOVER_DESIGN.md` §5).
