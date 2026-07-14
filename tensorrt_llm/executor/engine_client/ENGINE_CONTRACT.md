# Engine Client Contract (V0, experimental)

Typed, versioned, iteration-level contract between a serving frontend and the
generation engine. Gated by `TLLM_EXPERIMENTAL_ENGINE_CLIENT=1`; with the flag
off, legacy behavior is byte-identical (see Divergence notes for the single
approved exception). Protocol version: `ENGINE_CONTRACT_VERSION = 1`.

This document is the living specification companion: it tracks the scope
matrix, the rejection→test table, and every divergence from the design draft
(Part V of `serve → runtime layering`, frozen decisions draft DEC-1..9). It is
updated by every change to this package.

## Package layout

| Module | Owns | Status |
|---|---|---|
| `contract.py` | wire types + construction-time validation + `validate_no_callables` | delivered |
| `codec.py` | strict msgpack encode/decode for every wire type | delivered |
| `conversion.py` | legacy inputs → `EngineRequest` + `FrontendOutputConfig`; first capability gate; eligibility matrix; compositional runtime translation | delivered |
| `envelope.py` | raw worker response → normalized envelope; typed rejection of out-of-scope shapes | delivered |
| `router.py` | proxy-attached fork target; per-request exactly-once terminal state machine; concurrency spec | delivered |
| `local_client.py` | in-process `EngineClient`; setup gates; pre-submit checks; async `FrameStream` | delivered |
| `assembler.py` | frontend response assembly: accumulation, detok, stop handling, presentation | delivered |
| `invariants.py` | pass-through stream-shape checker (test utility) | delivered |
| `../../serve/engine_client_serving.py` | flag-on serving path: eligibility fallback + counters, model context, SSE via the real formatters | delivered |
| `service.py` | contract execution endpoint: contract-native submit (no rank-0 `GenerationResult`), exclusive routing, abort/detach/fail_all, typed control plane | delivered |
| `E5B_CUTOVER_DESIGN.md` / `E5B_EXIT_REPORT.md` | cut-over design (Codex-reviewed) and the gate-①c exit evidence + Step-② remaining-work inventory | delivered |

## Scope matrix (V0)

**Supported:** PyTorch backend · IPC-proxy transport (`GenerationExecutorProxy`)
· `num_postprocess_workers=0` · streaming · n=1 · pre-tokenized input · plain
text · chosen-token logprobs (`logprobs <= 1`, `prompt_logprobs <= 1`) ·
stop-token / stop-token-sequence / stop-string (incl. stop-string ⇒
engine-abort control edge) · abort · adapter-observed usage (incl. cached-token
accounting) · opt-in flag with legacy byte-identical when off.

**Config-gated at client construction (typed config error):** non-PyTorch
backends (`_autodeploy` explicitly rejected — do not gate on
`_is_pytorch_backend`) · RPC / Ray / single-process transports ·
`num_postprocess_workers > 0` · global `post_processor_hook` ·
`speculative_config` · early-first-token response mode · topology beyond the
validated set (TP1 over the IPC proxy) · `trust_remote_code` tokenizers · flag
off.

**Rejected pre-submit (typed request rejection):** multimodal · n>1 / best_of ·
beam search · LoRA · disaggregated serving · scheduling params · per-request
logits-processor callables · response-formatter callables (`PostprocParams`) ·
`logprobs > 1` · `prompt_logprobs > 1` · non-streaming mode · guided decoding
(schema field exists, capability-gated until the engine-path validation) ·
`ignore_eos` · bad words · `min_p` · prompt-ignore-length · embedding/logit
bias · output flags beyond logprobs · trace-header-bearing requests (avoids
silent telemetry loss) · `cache_salt` · thinking-token-budget · BART-family
models (normalization would create a callable) · `truncate_prompt_tokens` ·
`echo` · `top_logprobs`.

The machine-readable eligibility matrix (delivered with `conversion.py`) is
the authoritative per-field classification; this section is its prose summary.

## Wire types

Defined in `contract.py`; every type is a frozen, primitives-only,
callable-free dataclass carrying `protocol_version`, and every type
round-trips `codec.encode`/`codec.decode` (nothing is "in-process only").
`OutputFrame = TokenDelta | Terminal | RequestComplete | ErrorFrame`; every
frame carries per-request monotonic `event_seq` starting at 0 (draft DEC-3).
`Terminal` is the sole carrier of finish/stop reasons; `TokenDelta` is
non-empty by construction (a no-token response produces no frame, draft
DEC-6); `RequestComplete` usage is adapter-observed (draft DEC-2) with
`cached_tokens` as the final cached-token accounting (divergence note 5).
`FrontendOutputConfig` never crosses the contract; its
`stop_sequence_reasons` is an ordered association list (divergence note 11).
`FrontendModelContext`/`TokenizerSpec` are frozen from V0 (divergence note
10) even though their delivery is local until the remote detach.

## Codec rules

Strict msgpack (§2.4 of the draft): any value outside
`None | bool | int | float | str | list | dict` is a typed encode error — no
pickle fallback under any flag. Canonical encoding: dataclass fields in
declaration order, mapping keys sorted — equal values produce identical bytes
(golden fixtures in `tests/unittest/executor/engine_client/test_codec.py`).
Decode rejects with typed errors (`DecodeError.reason`): `unknown_kind`,
`unknown_field`, `missing_field`, `invalid_content`, `version_unsupported`,
`limit_exceeded`, `malformed`. Resource limits: 16 MiB message, depth 16,
2 Mi array items, 4 Ki map entries, 4 Mi-char strings, signed 64-bit ints,
finite floats, msgpack ext/bin rejected. Limits live in `contract.py` /
`codec.py` as constants.

## Draft frozen decisions (DEC-1..9) — implementation status

| Decision | Summary | Status |
|---|---|---|
| draft DEC-1 | chosen-token logprobs only; `logprobs > 1` rejected pre-submit; both runtime shapes normalized by exact token-id lookup | implemented (`conversion`, `envelope`; typed `logprob_mismatch` errors) |
| draft DEC-2 | usage adapter-observed; runtime-native usage is a later additive field | implemented (router-observed counts; user-visible usage counts assembled tokens — divergence note 14) |
| draft DEC-3 | exactly-once = server-side emission; `event_seq` stamped from V0 | implemented (router stamps per-request monotonic `event_seq` from 0) |
| draft DEC-4 | failure ownership matrix (worker crash ≤ ~5 s; per-stream bounds; early close ⇒ abort) | implemented (fail_all latch on `pre_shutdown`; `FrameStream.aclose()` ⇒ abort-if-incomplete); mechanism amended — see divergence note 7 |
| draft DEC-5 | cut-over at E5b; shadow-mode numbers are not performance claims | implemented (contract-native submit via `EngineService`; measurement + verdict in `E5B_EXIT_REPORT.md`) |
| draft DEC-6 | runtime-started predicate; `ErrorFrame` iff nothing runtime-started | implemented (router state machine; replay-matrix tested) |
| draft DEC-7 | native stop-token-sequence carrier on the worker-facing request path | implemented (`GenerationRequest.stop_token_sequences` merged by the worker's stop-word derivation) |
| draft DEC-8 | worker metrics enum keys convert via `.value` to `Mapping[str, float]` | implemented (`envelope`, typed `metrics_mismatch` errors) |
| draft DEC-9 | `TIMED_OUT` crosses as `Terminal(error, stop_reason="timeout")`; assembler renders legacy-visible reasons | implemented (router table + assembler presentation map; wire markers not presented as user stop_reason) |

## Divergence notes (amendments to the draft, all converged at plan time)

1. ①c cut-over scope = eligible OpenAI streaming chat/completions only; the
   public LLM API reroute is future work.
2. Ineligible requests under the flag fall back to legacy transparently, with
   per-axis observability counters; conversion still rejects typed internally.
3. V0 sampling normalization runs frontend-side exactly once (equivalent of
   `LLM._prepare_sampling_params` minus callable-producing steps);
   engine-side normalization is the remote-detach target.
4. `pad_id` added to `EngineSamplingConfig` (additive to draft §2.2) so the
   runtime request is reproducible from the encoded `EngineRequest` alone.
5. Cached-token accounting: reserved `TokenDelta.metrics` key
   `cached_tokens` pre-completion + typed `RequestComplete.cached_tokens`
   final field. Absent = never reported; 0 = reported zero; final must equal
   the last per-delta value when any delta carried the key; a zero-token
   request may report the value only on `RequestComplete`.
6. `FrameStream` is async-first with explicit `aclose()`.
7. Fatal-error propagation is a non-consuming sticky latch triggering
   `router.fail_all()`; the router never drains the shared `_error_queue`
   (draft DEC-4 semantics and bounds unchanged).
8. The worker's `_compute_pytorch_prompt_logprobs` gains a guard so a
   no-token `CANCELLED` final degrades gracefully instead of asserting.
   **This is the sole approved flag-off behavior divergence** (the affected
   path crashes today); it has its own regression test.
9. Config gates (client construction) are distinct from per-request
   rejections; see the scope matrix.
10. `FrontendModelContext` + `TokenizerSpec` schemas frozen in V0 with full
    provenance; `trust_remote_code` tokenizers are config-gated out.
11. `FrontendOutputConfig.stop_sequence_reasons` is an ordered association
    list: configuration order, first match wins, collisions resolved by
    order.
12. "Transport-only remote detach" is claimed for the request path only; the
    exit report enumerates all remaining reaches and control surfaces.
13. The post-verification ZMQ transport spike is out of scope.
14. User-visible usage counts the assembled (post-trim) tokens for legacy
    parity; `RequestComplete.completion_tokens` on the wire stays the raw
    adapter-observed engine count (the draft DEC-2 posture).
15. The legacy cross-chunk stop-string prefix leak is preserved (parity by
    oracle); a deliberate semantic fix is future work.
16. Prompt logprobs under KV block reuse: with (partial) block reuse
    enabled, the runtime returns context logits — and therefore prompt
    logprobs — only for the non-reused prompt suffix (a known runtime
    limitation; `llm_return_logprobs_test_harness` disables reuse for the
    same reason). Legacy delivers the truncated list silently; the contract
    path ends the request with a typed `logprob_mismatch` failure instead
    of delivering silently short values (the truncation check is
    unit-tested in `test_envelope.py`). The GPU suite therefore pins
    `enable_block_reuse=False` as the supported configuration for
    prompt-logprob traffic (`test_gpu_e2e.py` fixture note).

## Rejection → test table

One row per scope-matrix rejection axis; every row must have a passing test
and every rejection test must have a row (bidirectional check at audit time).

| Axis | Rejection point | Typed error | Test |
|---|---|---|---|
| unknown wire kind | codec decode | `DecodeError(unknown_kind)` | `test_codec.py::TestDecodeRejects::test_unknown_kind` |
| unknown field | codec decode | `DecodeError(unknown_field)` | `test_codec.py::TestDecodeRejects::test_unknown_field` |
| missing required field | codec decode | `DecodeError(missing_field)` | `test_codec.py::TestDecodeRejects::test_missing_required_field` |
| wrong item type / bool-as-int | codec decode | `DecodeError(invalid_content)` | `test_codec.py::TestDecodeRejects::test_wrong_item_type`, `test_bool_is_not_int` |
| newer protocol version (top-level + nested) | codec decode | `DecodeError(version_unsupported)` | `test_codec.py::TestDecodeRejects::test_newer_protocol_version`, `test_nested_version_check` |
| NaN/Inf | construction + codec | `ContractConstructionError` / `EncodeError(not_finite)` / `DecodeError(invalid_content)` | `test_contract.py::TestConstructionRejects`, `test_codec.py::TestEncodeRejects::test_smuggled_nan_rejected`, `TestDecodeRejects::test_nan_in_payload` |
| msgpack ext / bin / malformed | codec decode | `DecodeError` | `test_codec.py::TestDecodeRejects::test_extension_type_rejected`, `test_bytes_payload_rejected`, `test_malformed_bytes` |
| oversized message / nesting depth | codec | `EncodeError/DecodeError(limit_exceeded)` | `test_codec.py::TestDecodeRejects::test_oversized_message`, `test_nesting_depth_limit` |
| callable / object / bytes on the wire | construction + codec encode | `ContractConstructionError` / `EncodeError(non_primitive)` | `test_contract.py::TestNoCallables`, `test_codec.py::TestEncodeRejects` |
| non-registered type encode | codec encode | `EncodeError(unknown_type)` | `test_codec.py::TestEncodeRejects::test_unregistered_type` |
| `n_gt_1` (`n`, `best_of`) | conversion pre-submit | `RequestIneligibleError("n_gt_1")` | `test_conversion.py::TestRejections` case `n_gt_1`; facade fallback + counter via `test_serving_sse_full.py::TestEveryFallbackAxisThroughFacade` |
| `beam_search` (`use_beam_search`, `beam_width_array`, diversity/length/early-stopping) | conversion pre-submit | `RequestIneligibleError("beam_search")` | `TestRejections` case `beam_search`; facade recipe |
| `top_logprobs` (`logprobs > 1`) | conversion pre-submit | `RequestIneligibleError("top_logprobs")` | `TestRejections` case `top_logprobs`; facade recipe |
| `prompt_top_logprobs` (`prompt_logprobs > 1`) | conversion pre-submit | `RequestIneligibleError("prompt_top_logprobs")` | `TestRejections` case `prompt_top_logprobs`; facade recipe |
| `logprobs_mode` (non-RAW) | conversion pre-submit | `RequestIneligibleError("logprobs_mode")` | `test_conversion.py::TestRejections::test_logprobs_mode_rejected`; facade recipe |
| `logits_processor` (incl. batched) | conversion pre-submit | `RequestIneligibleError("logits_processor")` | `TestRejections` case `logits_processor`; facade recipe |
| `embedding_bias` | conversion pre-submit | `RequestIneligibleError("embedding_bias")` | `TestRejections` case `embedding_bias`; facade recipe |
| `bad_words` (`bad`, `bad_token_ids`) | conversion pre-submit | `RequestIneligibleError("bad_words")` | `TestRejections` case `bad_words`; facade recipe |
| `ignore_eos` | conversion pre-submit | `RequestIneligibleError("ignore_eos")` | `TestRejections` case `ignore_eos`; facade recipe |
| `min_p` | conversion pre-submit | `RequestIneligibleError("min_p")` | `TestRejections` case `min_p`; facade recipe |
| `top_p_extras` (`top_p_min`/`top_p_reset_ids`/`top_p_decay`) | conversion pre-submit | `RequestIneligibleError("top_p_extras")` | `TestRejections` case `top_p_extras`; facade recipe |
| `no_repeat_ngram` | conversion pre-submit | `RequestIneligibleError("no_repeat_ngram")` | `TestRejections` case `no_repeat_ngram`; facade recipe |
| `prompt_ignore_length` | conversion pre-submit | `RequestIneligibleError("prompt_ignore_length")` | `TestRejections` case `prompt_ignore_length`; facade recipe |
| `return_logits` (context/generation/encoder/additional outputs) | conversion pre-submit | `RequestIneligibleError("return_logits")` | `TestRejections` case `return_logits`; facade recipe |
| `exclude_input_from_output` (non-default) | conversion pre-submit | `RequestIneligibleError("exclude_input_from_output")` | `TestRejections` case `exclude_input_from_output`; facade recipe |
| `truncate_prompt_tokens` | conversion pre-submit | `RequestIneligibleError("truncate_prompt_tokens")` | `TestRejections` case `truncate_prompt_tokens`; facade recipe |
| `lookahead_config` | conversion pre-submit | `RequestIneligibleError("lookahead_config")` | `TestRejections` (direct conversion test); facade recipe |
| `thinking_token_budget` | context-only normalization | `RequestIneligibleError("thinking_token_budget")` | `TestRejections` case; facade recipe |
| `bart_forced_tokens` (`model_type == "bart"`) | context-only normalization | `RequestIneligibleError("bart_forced_tokens")` | `test_serving_e5a.py::TestContextOnlyBoundary` (facade, counter) |
| `echo` | conversion pre-submit | `RequestIneligibleError("echo")` | `TestRejections` case `echo`; facade recipe |
| `non_streaming` | conversion pre-submit | `RequestIneligibleError("non_streaming")` | `TestRejections` case `non_streaming`; facade recipe (`streaming=False`) |
| `multimodal` | conversion pre-submit | `RequestIneligibleError("multimodal")` | `TestRejections` case `multimodal`; facade recipe |
| `lora` | conversion pre-submit | `RequestIneligibleError("lora")` | `TestRejections` case `lora`; facade recipe |
| `prompt_adapter` | conversion pre-submit | `RequestIneligibleError("prompt_adapter")` | `TestRejections` case `prompt_adapter`; facade recipe |
| `disaggregated` | conversion pre-submit | `RequestIneligibleError("disaggregated")` | `TestRejections` case `disaggregated`; facade recipe |
| `scheduling_params` | conversion pre-submit | `RequestIneligibleError("scheduling_params")` | `TestRejections` case `scheduling_params`; facade recipe |
| `conversation_params` | conversion pre-submit | `RequestIneligibleError("conversation_params")` | `TestRejections` case `conversation_params`; facade recipe |
| `postproc_params` | conversion pre-submit (defense in depth — unreachable from the facade: both endpoints construct `PostprocParams` strictly after the contract branch) | `RequestIneligibleError("postproc_params")` | `TestRejections` case `postproc_params` |
| `trace_headers` (requests actually carrying one) | conversion pre-submit | `RequestIneligibleError("trace_headers")` | `TestRejections` case `trace_headers`; facade recipe; empty-mapping normalization tested in `test_serving_e5a.py` |
| `cache_salt` | conversion pre-submit | `RequestIneligibleError("cache_salt")` | `TestRejections` case `cache_salt`; facade recipe |
| `query_token_ids` | conversion pre-submit | `RequestIneligibleError("query_token_ids")` | `TestRejections` case `query_token_ids`; facade recipe |
| `encoder_input` | conversion pre-submit | `RequestIneligibleError("encoder_input")` | `TestRejections` case `encoder_input`; facade recipe |
| `priority` | conversion pre-submit | `RequestIneligibleError("priority")` | `TestRejections` case `priority`; facade recipe |
| `kv_cache_retention` | conversion pre-submit | `RequestIneligibleError("kv_cache_retention")` | `TestRejections` case `kv_cache_retention`; facade recipe |
| config gate: non-PyTorch backend (`_autodeploy` explicit) | client construction | `EngineClientConfigError` | `test_router_client.py::TestSetupGates` case `backend` |
| config gate: non-IPC transport (RPC/Ray/single-process) | client construction | `EngineClientConfigError` | `TestSetupGates` case `transport` |
| config gate: `num_postprocess_workers > 0` | client construction | `EngineClientConfigError` | `TestSetupGates` case `postproc` |
| config gate: global `post_processor_hook` | client construction | `EngineClientConfigError` | `TestSetupGates` case `post_processor_hook` |
| config gate: `speculative_config` | client construction | `EngineClientConfigError` | `TestSetupGates` case `speculative` |
| config gate: early-first-token response mode | client construction | `EngineClientConfigError` | `TestSetupGates` case `early_first_token` |
| config gate: topology beyond TP1-over-IPC | client construction | `EngineClientConfigError` | `TestSetupGates` case `topology` |
| config gate: `trust_remote_code` tokenizer | client construction | `EngineClientConfigError` | `TestSetupGates` case `trust_remote_code`; reload refusal in `load_tokenizer_from_spec` |
| config gate: flag off | client construction | `EngineClientConfigError` | `TestSetupGates` case `flag`; precedence matrix in `test_e5b_cutover.py::TestFlagResolver` |

The machine-readable sources are `conversion.ELIGIBILITY_MATRIX` and the
setup-gate list in `local_client._validate_config`; bidirectional
completeness between the matrix and the parametrized rejection cases is
enforced by `test_every_ineligible_axis_has_a_rejection_case`,
`TestEligibilityMatrixCompleteness`, and (facade side)
`test_every_request_level_axis_is_driven_through_the_facade`.
| unsupported response shapes (C++-engine, postproc-parallel, unknown) | envelope | `EnvelopeError(reason)` | `test_envelope.py::TestRejectedShapes` |
| pre-submit request checks (newer protocol, forged required_features, capability subset, duplicate/reused id) | client submit | `RequestRejectedError` | `test_router_client.py::TestPreSubmitChecks`, `TestStreamLifecycle` |
| serving fallback (ineligible under the flag ⇒ legacy + per-axis counter) | serving facade | counted fallback; the request is served by the unchanged legacy path (user-visible equivalence is inherited from legacy, exercised end-to-end by the GPU suite) | `test_serving_e5a.py::TestFallbackCounters` |
| submission during shutdown / after service close | service submit | `EngineServiceError` | `test_e5b_cutover.py::TestContractNativeSubmit::test_submission_rejected_during_shutdown` |
| duplicate engine-service attachment | proxy attach | `RuntimeError` | `test_e5b_cutover.py::TestContractNativeSubmit::test_second_service_attachment_rejected` |
| abort unknown id / double stream open / unknown stream | client ops | `UnknownRequestError` / `RequestRejectedError` | `test_router_client.py::TestAbort`, `TestStreamLifecycle` |
| malformed stop-reason carrier (non-primitive) | router bind | `RouterError` | `test_audit_remediations.py::TestRouterRemediations::test_malformed_stop_reasons_rejected_at_bind` |
| unprepared / untokenized conversion inputs | conversion | `ConversionError` | `test_conversion.py::TestConvertRequest::test_unprepared_params_rejected`, `test_untokenized_stop_strings_rejected` |
| multi-sequence response (`sequence_index != 0`) | router | request fails typed (`router_error`) | `test_audit_remediations.py::TestRouterRemediations::test_nonzero_sequence_index_fails_request` |
| mixed/non-string mapping keys on encode; non-finite metrics at the envelope | codec / envelope | `EncodeError(non_primitive)` / `EnvelopeError(metrics_mismatch)` | `test_audit_remediations.py::TestCodecRemediation`, `TestEnvelopeRemediation` |

## Normalization posture (V0)

Sampling normalization for the contract path runs frontend-side, exactly
once, at the CONTEXT-ONLY boundary — never via live model-config reaches:

- `conversion.prepare_sampling_params_from_context` is the sole normalizer
  on the serving contract path: EOS/PAD defaulting, generation-config stop
  ids, and model-type gates come from the frozen `FrontendModelContext`;
  stop strings are tokenized exactly once with the spec-reloaded
  tokenizer. Callable-producing legacy steps (BART forced tokens, thinking
  budget) raise `RequestIneligibleError`.
- Both endpoints capture a PRISTINE `SamplingParams` copy before any live
  `LLM.preprocess` / `input_processor` call and hand that copy to
  `try_stream`, which normalizes it unconditionally through the
  context-only boundary. The live-prepared original continues to serve the
  legacy/fallback path unchanged. A negative test poisons
  `prepare_sampling_params` and `LLM._prepare_sampling_params` and runs a
  full contract stream
  (`test_serving_sse_full.py::TestNoLiveNormalizationReaches`).
- `conversion.prepare_sampling_params` (live-input variant) remains only
  for direct library callers outside serving (e.g. the GPU suite driving
  the client directly); the serving glue never calls it.
- Downstream of conversion, the compositional translation
  (`engine_request_to_generation_request`) rebuilds the worker-facing
  request purely from the encoded `EngineRequest`; the synthetic params
  carry no stop strings, so nothing can re-run `_setup` (guard-tested).
- Tokenizer provenance is enforced at reload: `load_tokenizer_from_spec`
  verifies every manifest content hash and the recorded added/special-token
  configuration, failing typed
  (`test_serving_sse_full.py::TestTokenizerSpecProvenance`).

Engine-side normalization remains the remote-detach target (future work).
