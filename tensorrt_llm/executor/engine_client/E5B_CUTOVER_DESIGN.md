# E5b cut-over: contract-native submission and rank-0 result removal

Status: design for gate ①c on branch `prototype_1`.

This change cuts eligible engine-client requests over from the proxy's legacy
`GenerationResult` path to a contract-native path. After the cut-over, a contract request has
no rank-0 `GenerationResult` and no entry in `GenerationExecutorProxy._results`. Legacy requests
on the same proxy continue to use `GenerationExecutorProxy.submit`, rank-0 `GenerationResult`,
and the existing queue delivery behavior without semantic changes.

The current path is:

```text
LocalProcessEngineClient.submit
  -> engine_request_to_generation_request
  -> EngineFrameRouter.register_pending
  -> GenerationExecutorProxy.submit
       -> allocate client id
       -> EngineFrameRouter.observe_submit
       -> construct GenerationResult
       -> proxy._results[client_id] = result
       -> enqueue GenerationRequest
  -> dispatch_result_task
       -> EngineFrameRouter.on_response (observe)
       -> proxy._results delivery (legacy)
```

The E5b path is:

```text
LocalProcessEngineClient.submit
  -> GenerationExecutorProxy.submit_contract
  -> EngineService.submit_contract
       -> engine_request_to_generation_request
       -> allocate client id from the proxy's shared allocator
       -> bind ContractRequestState
       -> enqueue GenerationRequest
  -> dispatch_result_task
       -> EngineService.route_response (claim and consume contract ids)
       -> legacy proxy._results delivery only for non-contract ids
```

`GenerationExecutorProxy.submit` remains the legacy entry point. Contract traffic must not call it
after E5b.

## 1. Contract request state

### Decision

`EngineFrameRouter.RequestBinding` already is the rank-0 contract request state. E5b should not add
a second `ContractRequestState` registry beside it. Either rename `RequestBinding` to
`ContractRequestState` and keep `RequestBinding` as a private compatibility alias, or retain the
name and document that it is the canonical state. The important constraint is one state object and
one router-owned registry per contract request.

The existing binding already owns request/delivery lifecycle state under
`EngineFrameRouter._lock`: frontend request id, proxy client id, prompt token ids, stop-reason
associations, delivery, runtime-started/terminal/ended flags, abort state, event sequence,
completion and cached-token accounting, prompt-logprob delivery state, recent tokens, final
metrics/status, and stream-open state.

The canonical field model for E5b is:

| Field | Required representation and purpose | Current `RequestBinding` status |
|---|---|---|
| Frontend request id | `request_id: str`; the contract-visible identity and stream lookup key | Present |
| Proxy client id | `client_id: int`; allocated before binding and used for worker responses and abort | Present but initially optional because of the pending/observe hook; E5b binds it at construction |
| Prompt | `prompt_token_ids: tuple[int, ...]` plus a derived `prompt_length`; the ids are required by `normalize_response` for map-shaped prompt logprobs, so storing only the length is insufficient | Token ids present; length is currently repeatedly derived with `len(...)` |
| Per-sequence state | `sequence_states: dict[int, SequenceState]`, where each state owns `runtime_started`, `terminal_emitted`, `completion_tokens`, `recent_tokens`, prompt-logprob sent/held state, cached tokens, pending final metrics, and final status | These are currently scalar fields on the binding. That is correct only because V0 rejects `n > 1`; E5b may initialize sequence `0` eagerly or lazily, but must not imply that the current scalars support multiple sequences |
| Stop reason state | Ordered `(stop_token_sequence, visible_reason)` pairs used by `_resolve_stop_reason` | Present as `stop_reasons` |
| Delivery state | Bounded `_Delivery`, stream-open flag, request-ended flag, and request-level ending state | Present |
| Abort state | `abort_requested: bool` | Present |
| Event ordering | Request-wide `event_seq: int`, incremented only through `next_seq()` while holding the router lock | Present |
| Routing classification | `_by_client` for active or delivery-retained bindings, plus a bounded ordered recently-retired proxy-client-id structure used only for late/duplicate accounting | `_by_client` is present, but the current service-lifetime `_owned_client_ids` set must be replaced by the bounded structure |
| Request-id lifecycle | The frontend id remains reserved while its binding is active or its delivery is retained; it becomes reusable after the request has ended and that delivery is retired | The active and delivery indexes exist, but the current service-lifetime `_seen_request_ids` set must be removed and stream consumption/close must retire the delivery |

Delivery retirement is an explicit lifecycle transition under `EngineFrameRouter._lock`. It occurs
when the consumer reads through the request-ending frame, explicitly closes the stream, or the
bounded tombstone policy evicts the retained delivery. Duplicate frontend ids are rejected only
while the prior binding is active or its delivery remains in `_delivery_index`; after retirement,
reusing that frontend id is a required positive replay case. The recently-retired proxy-client-id
structure is independent of frontend-id reuse and is capped by `tombstone_limit`
(`DEFAULT_TOMBSTONE_LIMIT`, currently 4096). No per-request service-lifetime ownership set is part
of the canonical state.

The per-sequence refactor is structurally preferable because the wire contract says `Terminal` is
exactly once per started sequence. It is not a scope expansion: V0 still admits exactly one
sequence. If E5b keeps the scalar representation to minimize the gate, the state must be named and
documented explicitly as sequence-0 state, and any envelope with `sequence_index != 0` must fail the
request rather than mix token/terminal state across sequences.

### Sampling and logprob state

Do **not** put a second `SamplingParams` or `LogprobParams` copy in the rank-0 contract state.
`engine_request_to_generation_request` constructs a `GenerationRequest` whose synthetic
`SamplingParams` is what the worker receives. For `_get_logprob_params`, the relevant values are
`logprobs`, `prompt_logprobs`, `logprobs_simple_format`,
`prompt_logprobs_simple_format`, `_need_return_context_logits`, and
`_need_return_generation_logits` (plus the worker's postprocess-worker count). The contract
translation explicitly restores the two requested logprob counts. The other four values use their
synthetic `SamplingParams` defaults: both simple-format flags are false and both explicit
return-logit flags are false. That is correct for the current eligible surface: raw logits are
rejected and `num_postprocess_workers` must be zero, while `normalize_response` accepts the
map-shaped logprobs produced when simple format is false. If the contract later promises a specific
worker logprob representation, those booleans must be added to `EngineSamplingConfig`; they still
must not be stored in a rank-0 `GenerationResult`.

The rank-0 router needs only the prompt token ids to normalize map-shaped prompt logprobs and the
contract bookkeeping listed above. It must not call `_get_logprob_params`; doing so would preserve
rank-0-only state that E5b is intended to remove.

## 2. `submit_contract` on the proxy

### Interface and ownership

Add a distinct internal entry point:

```python
GenerationExecutorProxy.submit_contract(
    engine_request: EngineRequest,
    *,
    stop_reasons: tuple = (),
) -> int
```

It returns the allocated proxy `client_id`, not a `GenerationResult`.
`LocalProcessEngineClient.submit` continues to return the frontend `request_id`; it uses the integer
only as an internal submission result. The proxy method is a narrow facade over
`EngineService.submit_contract`, described in section 5.

Move the call to `engine_request_to_generation_request` behind this contract-native entry point.
This makes the service boundary accept the contract request rather than an in-process
`GenerationRequest`, and keeps later transport detachment from changing the service internals.

### Required order

Submission and shutdown need a proxy-level submission/lifecycle lock. Both legacy `submit` and
contract `submit_contract` use it. This lock serializes allocation plus enqueue with
`pre_shutdown`; it does not own router state. The lock order is always proxy lifecycle lock, then
`EngineFrameRouter._lock`. The dispatch thread never takes the proxy lifecycle lock.

The contract flow is:

1. Perform type/capability checks and `engine_request_to_generation_request` before allocating an
   id or mutating a registry. A conversion rejection therefore has no engine-side state.
2. Start the result dispatch thread as `GenerationExecutorProxy.submit` does today.
3. Acquire the proxy submission/lifecycle lock. Reject if `doing_shutdown`, `_fatal_error`, or the
   service fatal latch is set.
4. Allocate the proxy client id with the same `_get_next_client_id()` used by legacy submission,
   under this lock, and call `generation_request.set_id(client_id)`.
5. Under the router's single `_lock`, reject a duplicate frontend request id and install the fully
   bound state in the request-id, client-id, delivery, and routing-ownership registries. There is no
   `_pending[id(generation_request)]` interval and no `observe_submit` matching by Python object
   identity.
6. Still under the proxy lifecycle lock, enqueue the `GenerationRequest`. Binding is visible before
   the worker can return a response, while shutdown cannot put its sentinel between binding and
   enqueue.
7. If enqueue raises, call `EngineService.on_submit_enqueue_failed(client_id)` before releasing
   the lifecycle lock. It transitions the binding to a standalone
   `ErrorFrame(error_code="enqueue_failed")`. The request was never sent, so no runtime abort is
   needed. Re-raise the enqueue exception.
8. Release the lifecycle lock, run the existing background-error check, and return `client_id`.
   If that check discovers a fatal error after enqueue, the normal `pre_shutdown` path poisons the
   binding.

The binding replaces `proxy._results[client_id] = result`. In particular, the contract path must
not construct `GenerationResult`, call the proxy's `_get_logprob_params`, allocate a result queue,
or insert anything into `GenerationExecutorProxy._results`.

The current `register_pending` -> `observe_submit` protocol can then be removed from the contract
path. Keeping it available only for transitional tests is acceptable temporarily, but production
contract submission must have one explicit method so it cannot accidentally fall back through
legacy `submit`.

### Why worker-side response wrapping still works

The rank-0 and worker registries are independent. In `worker_main`, rank 0 dequeues the
`GenerationRequest` and calls `BaseWorker.submit`. That method:

1. preserves the proxy-assigned `request.id` as `client_id`;
2. calls `BaseWorker._get_logprob_params(request)`;
3. constructs a worker-local `GenerationResult` with those parameters;
4. inserts it into `BaseWorker._results[client_id]`; and
5. only then calls `_enqueue_request`.

The await-response path consults this worker-local registry in `_get_logprobs`,
`_compute_pytorch_prompt_logprobs`, and `_get_params_for_first_rsp`; `_maybe_wrap_response` can
therefore continue to build `ResponseWrapper` with prompt logprobs and request metrics. `_send_rsp`
also continues to pop the worker-local result on final/error. This is V0 behavior and must remain.

Tests for E5b must spy on the two processes/objects separately: a contract request is absent from
`GenerationExecutorProxy._results` while present in `BaseWorker._results` during execution, and
prompt-logprob `ResponseWrapper` output remains identical to the pre-cut-over path.

## 3. Unified response routing in `dispatch_result_task`

The router must change from an observer to an exclusive route claimant. Add an operation such as
`EngineService.route_response(raw) -> bool`, where `True` means the client id is owned by the
contract population and the response has been consumed or deliberately absorbed.

For every unbatched response item, `GenerationExecutorProxy.dispatch_result_task` must use this
exact order:

```python
client_id = raw.client_id

if engine_service is not None and engine_service.route_response(raw):
    continue

result = self._results.get(client_id)
if result is None:
    continue

# Existing legacy queue delivery, async notification, and final/error pop.
deliver_to_generation_result(result, raw)
if is_final_or_error(raw):
    self._results.pop(client_id, None)
```

Contract routing is checked before `proxy._results.get`. A claimed contract response is never
delivered to a legacy queue and never enters the legacy final-pop block. Legacy ids take the exact
existing path. An id that belongs to neither population is dropped as today.

`EngineFrameRouter.on_response` currently returns no ownership result and the proxy always calls
the legacy `process_res` afterward. E5b must replace that tap behavior; merely relying on the
absence of a contract id from `_results` is not a sufficient registry contract.

### Atomic classification and concurrency

`route_response` uses the router's existing single `_lock` for the initial client-id lookup and
claim. Contract-owned classification consists of a binding in `_by_client` plus the bounded
recently-retired proxy-client-id structure. If it finds active or delivery-retained state, it may
release the lock while `normalize_response` snapshots the raw response, but it has already claimed
the population. On reacquisition it either advances that state or absorbs the response if another
thread ended it. If it finds a recently retired id, it counts and absorbs the late response and
returns `True`. It returns `False` when the id is in neither structure.

This preserves the concurrency specification in `engine_client/router.py`:

- The submit thread binds under the router lock before enqueue.
- The dispatch thread claims and transitions under the same lock.
- Consumer threads mark abort/close state under the same lock, and issue runtime abort outside it.
- The shutdown thread latches failure and ends states under the same lock.
- `_Delivery` keeps its inner lock and never calls back into the router. Router state is never
  accessed while holding a delivery lock.

The relevant races all have one outcome:

- If dispatch wins before shutdown, it emits the response transition; shutdown sees the updated or
  ended state.
- If shutdown wins, `fail_all` emits the typed ending; dispatch recognizes the retained contract
  ownership and absorbs the late frame.
- If abort and final race, `abort_requested` and the final transition still meet at the existing
  idempotent ending functions.
- A worker response cannot beat registration because the request is not enqueued until after the
  client-id binding is visible.

The shared allocator, rather than unbounded router ownership state, makes bounded classification
safe. Proxy client ids increase monotonically across both legacy and contract submission and are
never reused within a proxy lifetime; wrap or any collision is a submission error. Therefore, once
an id ages out of the recently-retired structure, a very late frame may return `False`, fall through
to the legacy `_results.get`, find no result, and be dropped exactly as an unknown legacy late frame
is dropped today. It cannot be delivered to a newer legacy request because no newer request can
receive that id.

## 4. Abort, shutdown, and crash behavior across both populations

### Per-request abort

Legacy `GenerationResult.abort()` remains unchanged. Contract abort continues to look up the
frontend request id, set `abort_requested` under the router lock, and call
`proxy.abort_request(client_id)` outside that lock. A final response, a cancellation response, or a
later service failure resolves through the same exactly-once state machine.

### Abort-all and typed poisoning

`GenerationExecutorProxy._abort_all_requests` currently enumerates only `proxy._results`, so it
sees only legacy requests after E5b. Change the shutdown operation to cover both populations:

1. Snapshot legacy `GenerationResult` objects as today.
2. Through `EngineService.fail_all(reason)`, atomically latch the service failure, mark every active
   contract binding aborted, emit its appropriate typed ending, and return the active contract
   client ids that need runtime cancellation.
3. Outside the router lock, send `CancellingRequest` for those contract client ids on a best-effort
   basis.
4. Call `result.abort()` for the legacy snapshot.
5. Put the worker shutdown sentinel only after the submission/lifecycle lock has excluded new
   enqueues.

Poisoning is guaranteed; runtime cancellation is best effort because the worker may already have
crashed. `fail_all` must be idempotent, and the list of ids to cancel must be captured before the
bindings are pruned. Calling the current `router.fail_all` and then trying to enumerate
`_by_request` is too late.

The current `EngineFrameRouter.fail_all` only emits endings; it does not call the abort function.
That must change at the `EngineService` level. It also fixes the current mismatch in
`LocalProcessEngineClient.close_client`: its docstring promises to abort this client's in-flight
requests, but the implementation only calls `router.fail_all("client closed")`, allowing engine
work to continue.

### Every shutdown/crash entry

`GenerationExecutorProxy.pre_shutdown` is the common non-blocking transition and already calls
`router.fail_all("executor shutdown")`. The implementation currently reaches `pre_shutdown` from:

- explicit `GenerationExecutorProxy.shutdown`;
- `check_health` through `_drain_error_queue` or `_check_mpi_futures`;
- the background error monitor through those same helpers;
- the MPI-future done path after the monitor observes its queued/future error;
- `_handle_background_error`, which calls `shutdown` for a system error; and
- the registered interpreter/threading atexit callback.

E5b replaces the direct router call with the combined EngineService fail-and-abort operation above.
The first `pre_shutdown` invocation owns the transition under the lifecycle lock; later invocations
are no-ops. This gives check-health-driven, monitor-driven, explicit, and exit-driven shutdown the
same behavior for both request populations.

Client detach is separate from engine shutdown. `LocalProcessEngineClient.close_client` must call
an EngineService operation that poisons and aborts this client's active bindings without shutting
down a shared engine. V0 attaches only one service/client to a proxy; attachment of a second service
must be rejected rather than silently overwriting `_engine_frame_router` as
`attach_engine_frame_router` does today.

### Enqueue and submission failures

- A failure before binding (validation, conversion, dispatch-thread startup, shutdown check, or
  duplicate request id) creates no state and no stream.
- A failure after binding but before a successful queue put calls
  `on_submit_enqueue_failed(client_id)`, producing exactly one standalone `ErrorFrame`; no
  `GenerationResult` exists to leak and no abort is sent for a request that was not enqueued.
- A system error discovered after queue put flows through `pre_shutdown` and `fail_all`.
- A per-request worker admission error returns as `ErrorResponse` and is exclusively routed by the
  contract state machine.

The current generic `GenerationExecutorProxy.submit` creates and registers its rank-0 result before
`request_queue.put` and does not remove it if put raises. E5b must not reproduce that stale-entry
behavior in `submit_contract`; changing the legacy behavior is a separate cleanup.

## 5. `EngineService` factoring and remote-detach seam

Introduce one server-side `EngineService` component in the engine-client package. It owns the
contract execution endpoint, not frontend rendering and not the legacy population.

Its cohesive responsibilities are:

- validate/translate `EngineRequest` with `engine_request_to_generation_request`;
- submit through the proxy's shared allocator and request queue without a rank-0
  `GenerationResult`;
- own `ContractRequestState`/`RequestBinding`, delivery, tombstones, durable id ownership, and the
  router counters;
- exclusively route contract responses;
- implement request abort, client detach, and `fail_all`;
- expose capabilities/health and typed `IterationStatsBatch` / `KvCacheEventsBatch` query results;
  and
- hold no tokenizer, HTTP request, formatter callback, event loop, or other frontend object.

The current pieces map as follows:

| Existing piece | E5b mapping |
|---|---|
| `EngineFrameRouter` | Becomes the state-machine core owned by `EngineService`; its lock remains the sole contract-state lock |
| `LocalProcessEngineClient` | Becomes a thin in-process client/transport adapter: pre-submit contract checks, calls into the service-facing proxy methods, and exposes `FrameStream`; direct executor stats/KV calls move behind the service |
| `engine_request_to_generation_request` call in `LocalProcessEngineClient.submit` | Moves into `EngineService.submit_contract` |
| `attach_engine_frame_router` | Replaced by a one-time `attach_engine_service` (or equivalent constructor injection); duplicate attachment is an error |
| `observe_submit` and pending object-identity matching | Removed from production submission; explicit `submit_contract` binds the assigned id directly |
| `on_submit_enqueue_failed` | Retained as an internal EngineService/router transition |
| `dispatch_result_task` response tap | Replaced by exclusive `EngineService.route_response` before legacy lookup |
| `pre_shutdown` direct `router.fail_all` | Replaced by `EngineService.fail_all` plus best-effort runtime cancellation |
| `LocalProcessEngineClient.get_stats/get_kv_events/health` direct proxy calls | Delegate to EngineService query methods that return the existing typed contract results |

Removing the pending/object-identity protocol also fixes the service's active-count definition.
Today `EngineFrameRouter.active_request_count` adds `len(_by_request)` and `len(_pending)`, even
though `register_pending` inserts the same binding into both, so a pending submit is counted twice.
EngineService should count each non-ended binding once.

The service should depend on a narrow runtime adapter supplied by the proxy: start dispatch,
allocate id, enqueue, abort by client id, fetch stats/KV events, and inspect health. It must not know
about ZMQ socket construction or MPI process management.

This produces the later detach seam:

```text
frontend EngineClient
        |
        | local calls now; encoded transport later
        v
transport terminator
        |
        v
EngineService  ->  proxy runtime adapter  ->  worker
```

For V0, `LocalProcessEngineClient` is effectively the pass-through terminator. A later remote
transport terminator validates/decodes the same `EngineRequest`, submit metadata, abort, stats, KV,
and health operations and invokes the same EngineService methods. Nothing inside EngineService or
the worker-facing path changes. The ordered `stop_reasons` submit metadata must remain
primitive/callable-free and explicitly encoded by that terminator; `FrontendOutputConfig` itself
stays frontend-owned and is not passed wholesale into the engine.

## 6. `LlmArgs` option and environment precedence

Add a user-facing PyTorch-only field to `TorchLlmArgs`:

```python
experimental_engine_client: bool = Field(
    default=False,
    description="Enable the experimental engine-client path for eligible serving requests.",
    status="prototype",
)
```

The environment variable remains `TLLM_EXPERIMENTAL_ENGINE_CLIENT`. Resolve it once with this
precedence:

| Environment | `TorchLlmArgs.experimental_engine_client` | Effective value |
|---|---:|---:|
| unset | `False` | `False` |
| unset | `True` | `True` |
| `0` | either | `False` |
| `1` | either | `True` |

Thus the environment wins in both directions. Presence, not truthiness, decides whether it is an
override. Values other than the documented `0` and `1` should fail closed with a clear
configuration error rather than silently acting as false.

`EngineClientServing.create_if_enabled` and `EngineClientConfig` must consume the same resolved
value. The current split behavior is unsafe for this requirement:
`engine_client_flag_enabled()` reads only the environment, while
`EngineClientConfig._validate_config` ignores the environment whenever `flag_enabled` is not
`None`. In particular, an explicit `flag_enabled=True` currently defeats an environment value of
`0`. Remove that ambiguity; `EngineClientConfig.flag_enabled` should be the already-resolved value
or should use the shared resolver unconditionally.

Adding the field changes the generated `LLM.__init__` surface. The same change must update
`tests/unittest/api_stability/references/llm.yaml` with annotation `bool`, default `False`, and
status `prototype`. It must not be added to the committed/stable reference because it is a new
prototype option. API-stability and env-precedence tests must cover all four rows above, including
explicit-true/env-zero and explicit-false/env-one.

## 7. Risks, required safeguards, and measurement plan

### Client-id collision between contract and legacy traffic

Both current populations ultimately use `GenerationExecutor._get_next_client_id` on the same
`GenerationExecutorProxy`; `BaseWorker.submit` preserves the id that the proxy assigned. E5b must
keep this single allocator. It must not add a router-local counter.

The current allocator mutation is not explicitly synchronized and wraps at unsigned 64-bit.
Because legacy and contract submit threads can run concurrently, the proxy lifecycle/submission
lock must cover allocation for both paths. Make the allocator monotonic for the proxy lifetime and
reject unsigned-64 wrap rather than returning the wrapped value. Before binding, also reject any
collision with active legacy results, `_by_client`, or the bounded recently-retired id structure.
Proxy client ids are never reused in one proxy lifetime. The practical wrap interval is enormous,
but correctness must not depend on probability.

### Late frames after cut-over

Final/error removes detailed execution state. While the binding or delivery is retained,
`_by_client` still classifies late contract frames; after delivery retirement, move the client id
to the bounded recently-retired structure and use it only to count and absorb late/duplicate
frames. That structure is capped by `tombstone_limit` (`DEFAULT_TOMBSTONE_LIMIT` is 4096), evicting
oldest entries. Once an id is evicted, a very late frame may fall through to legacy lookup and be
dropped because the shared allocator never reuses the id during the proxy lifetime. Record claimed
late/duplicate frames and unknown-id legacy drops separately enough to diagnose runtime behavior.

The submission cut-over itself must occur before any contract request is admitted. Do not switch a
request that already has a rank-0 `GenerationResult` to contract ownership mid-flight. Flag changes
apply when the LLM/proxy/service is constructed, not dynamically while it is serving.

### Frontend request-id reuse

Reject a frontend request id only while its prior binding is active or its delivery is retained.
Once that request has ended and the delivery is retired by consuming through the ending frame,
explicitly closing the stream, or bounded tombstone eviction, remove its request-id indexes and
allow the id to be submitted again. AC-4 requires this reuse-after-retirement path as a positive
replay case, not merely as an eviction side effect. A stale stream cannot overlap the reused id
because consumption or close retired it; a late runtime frame carries the old, never-reused proxy
client id and therefore cannot bind to the replayed request. Remove `_seen_request_ids`; retaining
frontend-id ownership for the service lifetime violates the bounded-state requirement.

### Bounded retirement and memory baseline

Router classification and delivery retention must remain bounded in request count. `_by_request`
contains only active bindings; `_by_client` contains active or delivery-retained bindings;
`_delivery_index` and any request-id tombstones contain only retained deliveries; the ordered
recently-retired proxy-client-id structure contains at most
`tombstone_limit` entries (4096 by default) and carries no delivery or request state. Neither
`_owned_client_ids` nor `_seen_request_ids`, nor an equivalent unbounded per-request history, is
permitted.

Add a stress assertion with `N` substantially greater than `tombstone_limit`. After all `N`
requests complete and their deliveries retire, `_pending`, `_by_request`, `_by_client`,
`_delivery_index`, and request-id tombstones must return to their empty baseline; buffered delivery
memory must be released; and every recently-retired/accounting structure must be at or below its
configured bound. Repeating the stress cycle must not increase any per-request structure beyond
those baselines/bounds. This is the AC-4 memory-baseline assertion.

### Rank-0 CPU and streamed-token measurement

The performance claim for E5b is specifically rank-0 overhead removal, not GPU throughput. Measure
legacy and contract-native modes on the same commit, model, GPU, tokenizer, sampling parameters,
prompt/output lengths, `stream_interval=1`, and server configuration. Disable unrelated telemetry
and pin the rank-0 proxy process to the same CPU set. Alternate the mode order by repetition
(`A/B`, then `B/A`) so thermal/load drift is paired rather than confounded.

Use at least these streaming concurrency points: 1, 8, 32, and 128, capped only if a validated
configuration has a lower maximum. At each point:

- run at least 5 warm-up repetitions that are discarded;
- run at least 30 paired measured repetitions per mode;
- make each measured repetition contain at least 10,000 emitted token ids and last at least 30
  seconds, increasing request count if necessary; and
- reject/repeat a pair if request counts, emitted tokens, output-length distribution, or errors
  differ between modes.

Collect:

- rank-0 process user plus system CPU seconds from process-scoped counters;
- rank-0 CPU seconds per completed request;
- rank-0 CPU microseconds per emitted token id (CPU delta divided by the sum of
  `TokenDelta.new_token_ids`, not by frame count);
- proxy dispatch/service routing CPU time per response and per emitted token, with p50/p95/p99;
- peak and steady-state rank-0 RSS at each concurrency; and
- proof counters: zero rank-0 `GenerationResult` constructions/`proxy._results` entries for
  contract requests, with legacy counts unchanged.

Report the paired median difference and a two-sided 95% bootstrap confidence interval for every
concurrency point and primary CPU metric. Pre-register the acceptance threshold before running the
experiment; for gate ①c, use an upper 95% confidence bound of no more than a 5% regression in
rank-0 CPU seconds per emitted token at any concurrency, while expecting an improvement in CPU
and/or RSS from removing rank-0 `GenerationResult`. Do not claim an improvement when the 95%
interval crosses zero. Publish raw per-repetition data, environment/configuration, commit SHA, and
the analysis script so the result is repeatable.

### Pre-registered plan-DEC-3 gate protocol for this loop

For the plan-DEC-3 gate in this implementation loop, this subsection is authoritative. The fuller
protocol above remains the standard for publishable performance claims.

- Measure streaming concurrencies `{1, 8, 32}`.
- At each concurrency, discard two warm-up pairs, then record at least 10 paired repetitions. Each
  pair runs both legacy and contract-native modes, alternating order across pairs (`A/B`, then
  `B/A`).
- The metric is rank-0 CPU seconds per emitted token id: rank-0 process user-plus-system CPU seconds
  divided by the number of emitted token ids, never by response or frame count.
- For each concurrency, compute the per-pair contract-minus-legacy delta and use its median as the
  decision statistic. Compute a percentile-bootstrap two-sided 95% confidence interval for that
  paired median with at least 2000 resamples of the paired observations.
- The gate passes only when the upper confidence bound does not show a regression (that is, it is
  less than or equal to zero for a contract-minus-legacy delta). Otherwise, the result requires an
  explicit written acceptance.
- Generate the exit-report table directly from the recorded artifact values; never hand-copy
  measurement values into the table.

### Cut-over safety conditions

E5b is not safe to enable until all of the following current behaviors are changed and tested:

- contract submission no longer calls `GenerationExecutorProxy.submit` and never constructs or
  registers a rank-0 `GenerationResult`;
- contract ownership is checked before the legacy `_results.get` path and claimed responses never
  enter legacy delivery/pop logic;
- submission/id allocation is serialized with shutdown for both populations;
- contract classification uses `_by_client` plus only the bounded recently-retired id structure,
  and very late untracked ids fall through safely because proxy client ids are never reused;
- frontend request-id reuse after request ending and delivery retirement is a tested positive replay
  case, while reuse during active or retained delivery is rejected;
- after stress beyond the configured retirement bound, all per-request structures satisfy the AC-4
  empty-baseline/configured-bound assertions;
- `fail_all` both poisons contract streams and snapshots ids for best-effort runtime abort;
- `close_client` actually aborts its in-flight engine work;
- every `pre_shutdown` entry treats legacy and contract populations symmetrically;
- enqueue failure produces one typed ending without leaving rank-0 state; and
- worker-local `_results` registration and logprob/first-response behavior are retained and covered
  by an IPC worker-path test.
