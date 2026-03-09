# SimpleUnifiedScheduler Refactor: Design Document

## 1. Background

TensorRT-LLM has two scheduler implementations:

- **SimpleScheduler** (C++ bindings): The default scheduler. Uses C++ `BindCapacityScheduler`
  + `BindMicroBatchScheduler` via nanobind.
- **SimpleUnifiedScheduler** (pure Python): A Python mirror of SimpleScheduler, introduced
  for extensibility and experimentation. On main branch it follows the same two-pass
  structure as SimpleScheduler but implemented in Python.

The original two-pass Python implementation was slower due to Python interpreter overhead
and excessive Python→C++ boundary crossings. This refactor optimizes
`SimpleUnifiedScheduler` with a fused single-pass design, keeping scheduling intent and
major outputs aligned, with explicit intentional semantic differences documented in
Section 4.

## 2. Optimizations

### 2.1 Fused Single-Pass Scheduling

**Old**: Two sequential passes — capacity first, then microbatch (token budget + chunking).

```
PyCapacityScheduler.schedule_request(active_requests)
    → fitting_requests
PyMicroBatchScheduler.schedule(fitting_requests, inflight_ids)
    → context_requests, generation_requests
```

**New**: `TokenBudgetTracker` is passed into the capacity policy loop. Each request is
checked for both KV-block capacity AND token budget in one iteration. Chunking and sorting
are still performed in `tracker.finalize()`, but the separate microbatch iteration over
`fitting_requests` is eliminated.

**Impact**: Eliminates one full iteration over the fitting list.

### 2.2 Batched Block Decrements

**Old**: `decrement_reserved_blocks(req)` called per-request in first-pass loop →
O(N × W) C++ calls (N requests, W window sizes).

**New**: Deferred to `batch_decrement_list(scheduled_requests)` after the loop →
O(W) batch C++ calls using `get_remaining_blocks_to_completion_batch()`.

**Correctness**: `available_blocks` is not read during the first pass. `sync_to_dict()`
is called before the second pass starts.

### 2.3 Preview/Commit Block Reservation

**Old**: Second pass calls `enough_available_blocks(req)` then
`decrement_reserved_blocks(req)` → 2 × W C++ calls per request.

**New**: `preview_reserve(req)` checks AND caches needed blocks (1 × W C++ calls).
`commit_preview()` applies the cached decrement in pure Python.

### 2.4 Cached C++ Property Calls

| Property | Old (per request) | New |
|----------|------------------|-----|
| `req.is_disagg_generation_init_state` | Called 2× (guard + elif) | Cached as `is_disagg` once |
| `req.state_value` | Called each pass | Cached as `sv` once |
| `req.is_generation_in_progress_state` | 1 C++ call | `sv == _gen_in_progress` (Python int compare) |

### 2.5 Split Second-Pass Loops

**Old**: Combined loop over `[disagg_requests, context_requests]` with per-request
`is_disagg_generation_init_state` checks and routing.

**New**: Two typed loops — disagg loop skips `beneficial_to_skip` (never applies to
disagg) and routes directly to `fitting_disagg`; context loop skips disagg checks.

### 2.6 Single-Window Fast Path

`NoEvictScheduledBlocksManager` and `MaxUtilizationScheduledBlocksManager` detect
the common single-window case and use scalar arithmetic instead of dict iteration.

## 3. mypyc Compilation

### 3.1 Overview

`unified_scheduler.py` can be compiled with [mypyc](https://mypyc.readthedocs.io/) to
produce a native C extension (`.so`), eliminating Python interpreter overhead (attribute
lookups, frame creation, type dispatch) from the scheduling hot path.

mypyc compilation is optional and controlled by the `--mypyc` flag in `build_wheel.py`.
When not compiled, the module runs as normal Python.

### 3.2 What Gets Compiled

Only `unified_scheduler.py` is compiled — it contains all hot-path classes:
- `TokenBudgetTracker`
- `GuaranteedNoEvictPolicy`, `MaxUtilizationPolicy`
- `NoEvictScheduledBlocksManager`, `MaxUtilizationScheduledBlocksManager`
- `PyCapacityScheduler`
- `SimpleUnifiedScheduler`

Other scheduler files (`scheduler.py`, `adp_router.py`, `waiting_queue.py`) are thin
wrappers or C++ bindings that don't benefit from compilation.

### 3.3 Type Annotation Fixes for mypyc

mypyc enforces type annotations at runtime (unlike CPython which ignores them). Several
annotations were widened for compatibility:

| Change | Reason |
|--------|--------|
| `inflight_request_ids: set[int]` → `object = None` | Callers pass C++ `ReqIdsSet` (nanobind type), not Python `set` |
| `uniq_task_ids: set[int]` → `Optional[set[int]]` | Assigned `None` when PEFT is disabled |

### 3.4 Build Integration

```bash
# Standalone build (from pyexecutor/ directory):
python scheduler/setup_mypyc.py build_ext --inplace

# Via build_wheel.py:
python scripts/build_wheel.py --mypyc
```

`build_wheel.py` calls `build_pyexecutor_scheduler()` which invokes `setup_mypyc.py`.
When `--mypyc` is not set, stale `.so` artifacts are cleaned up to prevent accidental
use.

### 3.5 Profiling mypyc-Compiled Code

mypyc-compiled functions lack `__code__`, so `line_profiler` cannot hook them. The host
profiler automatically falls back to function-level timing wrappers for these targets.
Use `TLLM_LINE_PROFILER_PRESET=scheduler_hotpath` to profile the scheduler hot path.

## 4. Behavior Changes vs Main Branch

### 4.1 Intentional Semantic Changes

#### 4.1.1 Fused first-pass break produces a lighter resource state

When token budget is exhausted in the first pass, the fused path breaks the
loop immediately. Requests after the break point — generation, context,
and disagg alike — are never evaluated. This affects both
`MaxUtilizationPolicy` (token failure returns `None` → break) and
`GuaranteedNoEvictPolicy` (generation token failure → break classification
loop).

Because the failing generation request is never admitted, it does not consume
KV blocks, request slots, or PEFT pages. The second pass (in
GuaranteedNoEvict) and downstream scheduling therefore see a **lighter
resource state** than the old two-pass path, where capacity admitted all
generation unconditionally and microbatch dropped the excess afterward.

This produces three kinds of differences vs the old path:

**a) `paused_requests` may have fewer entries (MaxUtilization only).**
The old path could pause requests to make room for later requests that
microbatch would then drop anyway — wasted work. The fused path avoids this.

**b) `context_requests` may have additional entries (GuaranteedNoEvict).**
Context classified before the break gets a chance in the second pass against
the lighter state. In the old path, microbatch saw all generation before
context (generation-first ordering), so generation consumed budget first and
context after a failing generation was unreachable.

**c) `fitting_disagg_gen_init_requests` may differ in both directions.**
Disagg after the break point is never reached (fewer). Disagg before the
break point is evaluated against a lighter state — fewer KV blocks consumed,
fewer PEFT pages claimed — so it may be admitted where the old path would
have blocked it (more).

These differences all result in **equal or better token budget utilization**
than the old path. The old path's behavior was an artifact of the two-pass
ordering (capacity admits everything, microbatch iterates generation-first),
not a deliberate scheduling priority.

**Example — MaxUtilization pause avoidance (token_budget=100):**

```
Old two-pass pipeline:

  Capacity (MaxUtil): iterates ALL requests, admits/pauses based on KV blocks only
    → Request A: KV ok → admit
    → Request B: KV ok → admit
    → Request C: KV fail → pause older request, retry → admit
    → Request D: KV ok → admit
    Result: fitting_requests = [A, B, C, D], paused = [old_req]

  Microbatch: iterates fitting_requests with token budget
    → A: 30 tokens → ok (30/100)
    → B: 80 tokens → 30+80=110 > 100 → break
    Result: scheduled = [A], B/C/D dropped silently

Fused single-pass pipeline:

  Capacity + Token (MaxUtil): iterates with fused check
    → Request A: KV ok, 30 tokens ok → admit
    → Request B: KV ok, 30+80=110 > 100 → token fail → None → BREAK
    → Request C: NEVER REACHED
    → Request D: NEVER REACHED
    Result: fitting = [A], paused = []
```

Paused requests differ ([] vs [old_req]). Scheduled output is the same.

**Example — GuaranteedNoEvict admits context from lighter state
(token_budget=100, active_requests = [Gen_A(60), Ctx_C(30), Gen_B(50)]):**

```
Old two-pass pipeline:

  Capacity first pass: no token budget — classifies ALL requests
    → Gen_A: generation → scheduled, decrement blocks
    → Ctx_C: context → pending_requests
    → Gen_B: generation → scheduled, decrement blocks

  Capacity second pass: Ctx_C → blocks ok → added to scheduled
    Result: fittingRequests = [Gen_A, Gen_B, Ctx_C]

  Microbatch: iterates fittingRequests (generation-first order)
    → Gen_A: 60 tokens → ok (60/100)
    → Gen_B: 50 tokens → 60+50=110 > 100 → break
    → Ctx_C: NEVER REACHED
    Result: scheduled = [Gen_A]

Fused single-pass pipeline:

  First pass: token budget checked inline
    → Gen_A: generation, 60 tokens ok → admitted
    → Ctx_C: context → pending_requests (classified, not token-checked)
    → Gen_B: generation, 60+50=110 > 100 → break

  Second pass: processes pending_requests with remaining budget
    → Ctx_C: 60+30=90 ≤ 100 → admitted
    Result: scheduled = [Gen_A, Ctx_C]
```

The fused path schedules [Gen_A, Ctx_C] (90/100 tokens) vs the old path's
[Gen_A] (60/100 tokens). Gen_B is not scheduled in either path.

**Example — disagg-init diverges in both directions (token_budget=100):**

```
Old two-pass pipeline:

  Capacity first pass: no token budget
    → Gen_A (50 tokens): → scheduled, decrement blocks
    → Disagg_X:          → pending_dis_gen_init
    → Gen_B (60 tokens): → scheduled, decrement blocks
    → Disagg_Y:          → pending_dis_gen_init

  batch_decrement_list([Gen_A, Gen_B]) → both consume KV blocks

  Capacity second pass: evaluates Disagg_X, Disagg_Y against remaining
    blocks (after Gen_A + Gen_B consumed)

  Microbatch: Gen_A(50) ok, Gen_B(60) → 110>100 → break
    Result: scheduled = [Gen_A], fitting_disagg = [Disagg_X, Disagg_Y]
            (if both fit blocks)

Fused single-pass pipeline:

  First pass:
    → Gen_A (50 tokens): admitted
    → Disagg_X:          → pending_dis_gen_init
    → Gen_B (60 tokens): 50+60=110>100 → break
    → Disagg_Y:          NEVER REACHED

  batch_decrement_list([Gen_A]) → only Gen_A consumes KV blocks

  Second pass: evaluates Disagg_X against remaining blocks
    (after only Gen_A consumed — lighter state)
    Result: scheduled = [Gen_A], fitting_disagg = [Disagg_X]
```

Disagg_Y was deferred (fewer). Disagg_X was evaluated against a lighter
block state (Gen_B's blocks not consumed), so it may fit where the old path
would have blocked it (more). Both effects are benign — deferred requests
retry next iteration, and extra admissions reflect genuinely available
resources.

#### 4.1.2 `num_fitting_requests` semantics

Now counts requests admitted by the fused capacity + token-budget path
(`TokenBudgetTracker._num_fitting`), not just capacity-fitting. In
`SimpleScheduler`, `num_fitting_requests` was `len(fitting_requests)` from
the capacity pass only.

Note: this count is computed before late pruning. It can overcount in two ways:

1. **Chunking**: `_num_fitting` is incremented when `try_add_context()`
   accepts a request, but `finalize()` may later drop requests with
   `context_chunk_size == 0` without decrementing.
2. **Post-scheduler filters**: `schedule_active_requests()` passes
   `num_fitting_requests` through unchanged after `balance_adp_requests()`
   or `check_batch_waiting()` may have shrunk the context batch.

The field therefore means "requests admitted by the fused capacity/token path
before late pruning," not "final scheduled batch size."

#### 4.1.3 `speculation_permanently_disabled`

New monotonic `False→True` flag. Set by executor when spec-decode acceptance
rate drops below threshold.

### 4.2 Bug Fixes vs Main

#### 4.2.1 MaxUtilization PEFT page accumulation

Fixes a pre-existing bug in main's Python `MaxUtilizationPolicy` where
`num_scheduled_peft_pages` was passed by value to `_try_scheduling_request()`
and never accumulated across requests. Every request saw
`num_scheduled_peft_pages = 0`, so cumulative PEFT page limits were not
enforced. The same bug exists on main's `scheduler.py`.

Now returns the updated total from `_try_scheduling_request()`, matching the
C++ reference (`capacityScheduler.cpp` `trySchedulingRequestMaxUtilization`)
which passes by reference. `GuaranteedNoEvictPolicy` was already correct
(accumulates `claimed_peft_pages` locally).

This can change `context_requests` and `generation_requests` vs main on
workloads that use LoRA with MaxUtilization scheduling, because the old path
would over-admit requests that exceed the cumulative PEFT page budget.

### 4.3 Internal Refactoring (no external semantic change)

| Change | Details |
|--------|---------|
| **Disagg request return path** | Capacity policy returns 3-tuple `(scheduled, fitting_disagg, paused)` instead of 2-tuple. `fitting_disagg` was already a separate output in `SchedulerOutput` — this is an internal plumbing change, not a new external behavior. |
| **Scheduling orchestration** | Validation, ADP routing, and drafter setup consolidated into `schedule_step()` instead of being scattered across `py_executor.py`. |

### 4.4 Preserved Behavior

| Area | Why Equivalent |
|------|----------------|
| State range check | Same conditions: disagg bypasses range, others check `_until <= sv < _after` |
| Block reservation | Same check-then-decrement logic, batched/cached |
| PEFT checks (`GuaranteedNoEvictPolicy`) | Identical to main (accumulates `claimed_peft_pages` locally) |
| `beneficial_to_skip` | Disagg always skipped it (old code had `not req.is_disagg` guard) |
| Context chunking | Same `EQUAL_PROGRESS` / `FIRST_COME_FIRST_SERVED` policies |
| Request sorting | Same LoRA-based sort in `finalize()` |

Note: `MaxUtilizationPolicy` PEFT behavior changed vs main — see Section 4.2.1.

### 4.5 KV Allocation Semantics

`prepare_resources()` runs on the final scheduled batch only — requests
filtered by token budget never allocate real KV blocks in either path.

However, the fused path's lighter resource state (Section 4.1.1) means:
- The main scheduled batch may contain additional context requests that the
  old path would have dropped (GuaranteedNoEvict). KV allocation for these
  extra requests is correct — they passed KV block checks in the second pass.
- `fitting_disagg_gen_init_requests` may differ. Those requests are fed into
  `_prepare_disagg_gen_init()` which prepares KV resources outside the main
  `prepare_resources()` batch.

## 5. Performance Results

**Experiment setting**: Llama 8B, B200 single GPU, 411 scheduling iterations.
Measured with the host profiler.

| Configuration | Total | Per-Iteration | vs Main |
|--------------|-------|---------------|---------|
| main branch | 7.16s | 17.4ms | baseline |
| Refactored (Python) | 4.33s | 10.5ms | **-39.6%** |
| Refactored (mypyc)* | 1.19s | 2.89ms | **-83.4%** |

\* mypyc measurement covers `schedule_step` (includes fetch/validate/drafter overhead beyond `schedule_request`). Approximate comparison only.

### Speedup Attribution (rough hypothesis, not precisely measured)

| Source | Estimated Contribution | Mechanism |
|--------|----------------------|-----------|
| Eliminate separate microbatch pass | Major | One fewer O(N) iteration; chunking/sorting still runs in `finalize()` |
| Reduce C++ boundary crossings | Moderate | Caching, batching, preview/commit |
| Python micro-optimizations | Minor | Local variable caching, int counters, `__slots__` |

## 6. Files Changed

| File | Change |
|------|--------|
| `scheduler/unified_scheduler.py` | Refactored TokenBudgetTracker, capacity policies, NoEvictScheduledBlocksManager, SimpleUnifiedScheduler |
| `scheduler/scheduler.py` | Removed old Python scheduling classes (moved to unified_scheduler.py) |
| `pyexecutor/py_executor.py` | Added `_prepare_and_schedule_batch_unified()` path |
| `pyexecutor/request_utils.py` | Extracted validation, ADP routing, drafter utilities |
| `pyexecutor/_util.py` | Instantiation gate: enabled when `TLLM_USE_PYTHON_SCHEDULER` is set to `1` |
| `scheduler/setup_mypyc.py` | mypyc build script for compiling `unified_scheduler.py` to native C extension |
| `scheduler/mypy_mypyc.ini` | mypy configuration for mypyc compilation (error suppressions for external types) |
| `scripts/build_wheel.py` | Added `build_pyexecutor_scheduler()` for mypyc integration via `--mypyc` flag |
| `tools/profiler/host_profile_tools/host_profiler.py` | Preset system (`scheduler_hotpath`) + timer fallback for mypyc-compiled functions |

## 7. Validation

### Recommended correctness checks

Compare scheduling outputs between `SimpleScheduler` (default) and
`SimpleUnifiedScheduler` on the same workload.

**All policies — must match:**
- Request ordering (verify LoRA sort and chunk partitioning match)
- Key state transitions (requests entering/leaving scheduled batch)

**All policies — expected to differ:**
- `num_fitting_requests`: now counts capacity + token-budget admissions, not
  just capacity (Section 4.1.2)

**GuaranteedNoEvict — must match:**
- `len(generation_requests)`
- `len(paused_requests)` (this policy does not pause)

**GuaranteedNoEvict — expected to differ when token budget is the bottleneck:**
- `len(context_requests)`: may have more entries — the second pass admits
  context against the lighter resource state left by the first-pass break
  (Section 4.1.1b)
- `len(fitting_disagg_gen_init_requests)`: may have more entries — disagg
  classified before the break is evaluated against a lighter state where the
  failing generation did not consume KV/PEFT (Section 4.1.1c). Disagg after
  the break is deferred (fewer).

**MaxUtilization — must match:**
- `len(context_requests)`, `len(generation_requests)` (single loop breaks,
  no second pass to admit extra work)

**MaxUtilization — expected to differ when token budget is the bottleneck:**
- `len(paused_requests)`: fewer — the fused path avoids wasted pause/backtrack
  (Section 4.1.1a)
- `len(fitting_disagg_gen_init_requests)`: may have fewer entries — disagg
  after the break point is deferred (Section 4.1.1c)

**LoRA + MaxUtilization — expected to differ:**
- `len(context_requests)`, `len(generation_requests)`: may differ due to
  PEFT page accumulation bug fix (Section 4.2.1) — the old path over-admitted
  requests that exceed the cumulative PEFT page budget

### Enable the refactored scheduler
```bash
export TLLM_USE_PYTHON_SCHEDULER=1
```

### Profile with scheduler preset
```bash
TLLM_USE_PYTHON_SCHEDULER=1 \
TLLM_LINE_PROFILER_PATH=./profile.txt \
TLLM_LINE_PROFILER_PRESET=scheduler_hotpath \
trtllm-bench --model <model> throughput --dataset <dataset>
```
