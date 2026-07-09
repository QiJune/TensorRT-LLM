# Engine-Client Pipeline Capability Matrix

Routing of every endpoint/feature/backend/topology when the prototype
`enable_engine_client_pipeline` flag is on, plus the serve→runtime reach
inventory and its replacements. Consistent with the eligibility predicate in
`tensorrt_llm/serve/frontend/eligibility.py` and the handshake capability
set advertised by `EngineSocketServer` (`LegacyEngineClientAdapter`
default capabilities). `FUT-*` ids refer to the follow-up items of the
serve→runtime layering program.

Routing values:

- **new-path** — served by the engine-client pipeline (frontend-owned
  tokenize/detokenize/formatting over the token-level boundary).
- **co-located fallback** — with the flag on, silently served by the
  historical in-process path (debug log, never an error).
- **headless reject** — on a detached frontend there is no in-process path;
  the request receives a typed `unsupported_capability` error and nothing
  crosses the boundary.
- **untouched** — the flag does not affect this path at all.

## Endpoints

| Endpoint | Co-located (flag on) | Detached frontend | Crosses seam | Deferral |
|---|---|---|---|---|
| `/v1/chat/completions` (text-only) | new-path | new-path | token ids + `RuntimeSamplingConfig` (msgpack) | — |
| `/v1/completions` (single prompt) | new-path | new-path | token ids + `RuntimeSamplingConfig` | — |
| `/v1/completions` (multi-prompt batch) | co-located fallback | headless reject | — | FUT-10 |
| `/v1/completions` (echo with pre-tokenized prompt) | co-located fallback | headless reject | — | FUT-10 |
| Chat for harmony/gpt-oss models | untouched (separate route) | headless reject (route absent) | — | FUT-6 |
| `/v1/responses` (Responses API) | untouched (separate route) | headless reject (route absent) | — | FUT-6 |
| LLM API text path (`LLM.generate_async`) | new-path | n/a — the LLM API never runs detached (DEC-1) | in-process contract objects | — |
| LLM API non-text inputs (multimodal, queries, encoder ids) | co-located fallback | n/a | — | FUT-7 |
| `/health` | untouched | new-path (`RuntimeControl.check_health` over the socket) | control message | — |
| `/version`, `/v1/models` | untouched | new-path (handshake `model_context`) | handshake | — |
| `/metrics` (Prometheus) | untouched | co-located-only (raw stats exposed as `/iteration_stats` detached) | — | FUT-5 |
| `/kv_cache_events` | untouched | new-path (`RuntimeControl.get_kv_events`) | control message | — |
| `/iteration_stats` (detached only) | n/a | new-path (`RuntimeControl.get_stats`) | control message | — |
| `/release_memory`, `/resume_memory`, `/update_weights` (collective-RPC-backed) | untouched | headless reject (typed `unsupported_capability`, HTTP 501) | — | FUT-1 |
| `/perf_metrics`, `/energy_metrics`, `/steady_clock_offset`, `/server_info`, `/health_generate` | untouched | co-located-only (not served detached) | — | FUT-5 |
| Disaggregated server/coordinator (`openai_disagg_server`, `openai_client`) | untouched (files untouched; HTTP role protocol unchanged) | not offered | — | FUT-3 |
| gRPC serving mode | untouched | not offered | — | FUT-4 |
| Multimodal encoder serving (`mm_embedding_serve`) | untouched | not offered | — | FUT-7 |

## Request features

Per-request half of the predicate (`check_request` / `check_sampling_params`).

| Feature | Co-located (flag on) | Detached | Crosses seam | Deferral |
|---|---|---|---|---|
| Plain text generation, streaming + non-streaming | new-path | new-path | token ids, sampling data | — |
| `n`/`best_of` multi-sequence sampling | new-path | new-path | `sequence_index` per event | — |
| Stop strings (incl. `include_stop_str_in_output`) | new-path | new-path | tokenized stop sequences cross; strings stay frontend-side (`FrontendOutputConfig`) | — |
| `stop_token_ids` | new-path | new-path | crosses (runtime stops early) | — |
| Logprobs / prompt logprobs | new-path | new-path | plain-data logprob payloads | — |
| Usage / `stream_options` | new-path | new-path | frontend-computed | — |
| Reasoning parser, tool parser (text tool calls) | new-path | new-path | frontend-only (stateful formatter params) | — |
| Beam search | new-path (cumulative events) | new-path | `cumulative` token prefixes | oracle coverage deferred with FUT-9 |
| Multimodal content | co-located fallback | headless reject | — | FUT-7 |
| Guided decoding (`response_format`, structural tags) | co-located fallback (always, this loop) | headless reject | — | FUT-9 |
| Python `logits_processor` (incl. batched) | co-located fallback | headless reject | declared `PythonExtension` side channel, never the neutral wire | FUT-8 |
| `thinking_token_budget` (attaches a logits processor) | co-located fallback | headless reject | — | FUT-8 |
| `logit_bias` / `embedding_bias` (tensor) | co-located fallback | headless reject | — | FUT-8 |
| `return_context_logits` / `return_generation_logits` / `return_encoder_output` / `additional_model_outputs` | co-located fallback | headless reject | reserved `TensorAuxiliaryPayload` side channel | FUT-7 |
| LoRA / prompt adapter | co-located fallback | headless reject | — | FUT-9 |
| Lookahead / speculative per-request config | co-located fallback | headless reject | — | FUT-9 |
| `disaggregated_params` on a request | co-located fallback | headless reject | adapter passes disagg metadata through opaquely for the old path | FUT-3 |
| Chat tool-call conversation history (assistant `tool_calls`/`function_call`, `tool` messages) | co-located fallback | headless reject | the import-light detached parser cannot reproduce `parse_chat_messages_coroutines`' tool handling; `request.tools` for the current turn stays eligible | FUT-9 |
| `return_perf_metrics` enabled on the LLM API | pipeline never constructed — deployment-level fallback: the EngineRequest/output lack arrival_time and metrics_dict | n/a | — | FUT-10 |
| Chat/completion `conversation_params` (multi-turn routing) | co-located fallback | headless reject | not mapped to the engine request; served in-process | FUT-9 |
| Chat `agent_hierarchy` (hierarchy-aware scheduling) | co-located fallback | headless reject | not mapped to the engine request; served in-process | FUT-9 |
| `return_perf_metrics` enabled on the deployment | pipeline never constructed — deployment-level fallback: the pipeline bypasses the server's `_extract_metrics`/Prometheus, so the in-process path serves everything | not offered | — | FUT-10 |
| KV retention / scheduling / conversation params, `_postproc_params` (LLM API kwargs) | co-located fallback | n/a | — | FUT-9 |
| Post-processor hook (`post_processor_hook` configured) | pipeline never constructed — deployment-level fallback: the hook runs at the in-process detokenization chokepoint, so the in-process path serves everything | not offered | — | FUT-10 |

## Backends and topologies

Deployment half of the predicate (`check_deployment`,
`validate_headless_launch_args`).

| Deployment | Flag on, co-located | Headless engine launch |
|---|---|---|
| PyTorch backend, default MPI/IPC orchestration (in-process worker or IPC proxy, single- or multi-rank) | pipeline active | supported |
| TensorRT backend | pipeline never constructed (in-process path serves everything) | fails fast (`ValueError`) |
| Ray orchestrator | pipeline never constructed | fails fast |
| RPC orchestrator | pipeline never constructed | fails fast |
| `num_postprocess_workers > 0` | pipeline never constructed (postproc workers carry formatter callables engine-side) | fails fast at startup | 
| Encoder-decoder models | pipeline never constructed (BART forced-token processors) | not offered |
| Multi-rank headless engines + collective-RPC control endpoints | n/a | typed `unsupported_capability` (see Endpoints); data-plane generation unaffected |

Deferral for the postproc-worker row: FUT-10. Encoder-decoder: FUT-9.

## Reach-cutting inventory

Serve→runtime reaches on the historical `OpenAIServer` and their status. The
detached frontend (`serve/frontend/detached_app.py`,
`commands/serve_frontend.py`) performs none of these; the static and
process-level audits in
`tests/unittest/serve/frontend/test_detached_frontend.py` enforce that.
Counts reflect the current source (the design draft counted 29 at survey
time; the tree has since changed).

| Reach (in `serve/openai_server.py`) | Count | Replacement on the detached path |
|---|---|---|
| `generator.args.reasoning_parser` | 7 | `FrontendModelContext.reasoning_parser` (handshake) |
| `generator.args.return_perf_metrics` | 3 | `FrontendModelContext.return_perf_metrics` (handshake) |
| `generator.args.perf_metrics_max_requests` | 2 | co-located-only (`/perf_metrics` not served detached, FUT-5) |
| `generator.args.max_beam_width` | 2 | co-located-only (request validation stays engine-side; engine rejects typed) |
| `generator.args.backend` | 2 | deployment predicate input; detached is pytorch-only by launch validation |
| `generator.args.num_postprocess_workers` | 1 | deployment predicate input; headless launch fails fast |
| `generator.args.trust_remote_code` / `parallel_config` / misc | 4 | co-located-only (tokenizer/processor loading happens from `FrontendModelContext.tokenizer_dir` detached) |
| `generator._hf_model_dir` | 1 | `FrontendModelContext.tokenizer_dir` (handshake) |
| `generator._executor.resource_governor_queue` | 1 | co-located-only (resource governor endpoint not served detached, FUT-1) |
| `from tensorrt_llm._torch.pyexecutor.config_utils import ...` | 1 | not imported detached; model-type resolution comes from the handshake |
| Module-level import chain (`commands/serve.py`, `serve/openai_server.py` import torch/LLM at import time) | — | separate light entrypoint `trtllm-serve-frontend` + `TLLM_LIGHTWEIGHT_IMPORT=1` package gates; smoke test asserts no `torch`/`tensorrt_llm._torch`/LLM/executor modules load |
| `getattr(generator, "_generation_config")` (co-located pipeline construction) | 1 | `FrontendModelContext`/handshake carries stop-token defaults detached (engine-side setup already applied them) |

Worker-side dynamically attached Python-only request fields (the `py_*`
inventory on `BaseWorker`) remain engine-internal; they never cross the
boundary in either direction this loop (FUT-1, FUT-8).

## Consistency checks

- The predicate conditions in `eligibility.py` map 1:1 to the
  fallback/reject rows above; `tests/unittest/serve/frontend/test_eligibility.py`
  exercises each row's fallback (co-located) and typed rejection (detached)
  with no-partial-submission assertions.
- The handshake capability set advertised by the adapter
  (`generation.streaming/beam_search/num_sequences/logprobs`,
  `control.health/stats/kv_events`) matches the new-path rows;
  `SocketEngineClient.require_capability` rejects anything not advertised.
- Golden-output equality on all three new-path variants (in-process, socket
  co-located, headless) is enforced by
  `tests/unittest/serve/frontend/test_golden_output_oracle.py`.
