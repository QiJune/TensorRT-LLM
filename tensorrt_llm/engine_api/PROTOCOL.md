# Engine Boundary Wire Protocol (seam between serving frontend and engine)

**Protocol version: 1** (`tensorrt_llm.engine_api.protocol.PROTOCOL_VERSION`)

This document specifies the language-neutral wire protocol spoken between a
serving frontend (HTTP server, tokenization, detokenization, output
formatting) and a generation engine. The transport is ZMQ; the codec is
msgpack with **no pickle fallback and no Python-object serialization**: any
value that is not plain data (callables, tensors, arbitrary objects) fails
encoding with a typed error. A frontend written in any language can speak
this protocol.

## Transport topology

- One `ROUTER` socket on the engine side (`EngineServer`), bound to a
  `tcp://127.0.0.1:<port>` or `ipc://` endpoint.
- One `DEALER` socket per frontend connection.
- Every frame is exactly one msgpack-encoded message (below). Requests and
  events for any number of in-flight requests are multiplexed over one
  connection and correlated by `request_id`.

## Envelope

Every message is one msgpack map:

| Field | Type | Meaning |
|---|---|---|
| `protocol_version` | int | Protocol revision. Mismatch fails the handshake with a typed error — never a hang or crash. |
| `message_type` | str | One of the declared types below. Undeclared types are rejected as protocol violations. |
| `request_id` | str \| nil | Correlation id, frontend-assigned, unique per connection. |
| `payload` | map | Message-type-specific body. |

## Message types

| `message_type` | Direction | Payload |
|---|---|---|
| `handshake` | frontend → engine | `{client_info}` |
| `handshake` | engine → frontend | `{capabilities, readiness_state, model_context}` |
| `submit` | frontend → engine | Flattened `EngineRequest`: `request_id`, `prompt_token_ids`, `sampling` (the `RuntimeSamplingConfig` map), `streaming`, `priority`, `cache_salt`, `arrival_time`, `trace_context`, `disaggregated_metadata`. Declared side channels (`PythonExtension`, `TensorAuxiliaryPayload`) **never** cross this wire; a request carrying them fails encoding. |
| `event` | engine → frontend | Flattened `EngineEvent`: `sequence_index`, `event_index`, `token_ids` (delta; full prefix when `cumulative`), `logprobs`, `cumulative_logprob`, first-event-only `prompt_token_ids`/`prompt_logprobs`, `finish_reason`, `stop_kind`, `stop_reason`, `terminal_kind`, `error_code`+`error_message` (on error terminals), `disaggregated_metadata`, `metrics`. |
| `abort` | frontend → engine | `{}` — abort the request named by `request_id`. |
| `abort_ack` | engine → frontend | `{known: bool}` — always sent in response to `abort`. |
| `control_request` | frontend → engine | `{method, kwargs}` — control plane (`get_capabilities`, `check_health`, `get_stats`, `get_kv_events`). `control_id` in payload correlates responses. |
| `control_response` | engine → frontend | `{control_id, result}` or `{control_id, error_code, error_message}`. |
| `error` | engine → frontend | `{error_code, error_message}` — connection- or request-scoped typed error. Never a pickled exception or stack trace. |

`terminal_kind` ∈ {`finished`, `aborted`, `error`}. `readiness_state` ∈
{`initializing`, `ready`, `unhealthy`, `shutting_down`}. Typed
`error_code` values are the `EngineErrorCode` enum: `invalid_request`,
`unsupported_capability`, `unknown_request`, `request_failed`,
`engine_unavailable`, `engine_shutdown`, `slow_consumer`,
`protocol_version_mismatch`, `protocol_violation`, `internal_error`.

## Handshake

1. On connect, the frontend sends `handshake` with its `protocol_version`.
2. The engine replies with `handshake` carrying its `capabilities` map, its
   `readiness_state`, and the `model_context` map (tokenizer source, model
   name and limits, perf-metrics flags) from which a detached frontend
   builds its model context.
3. If the versions differ, the receiving side replies with `error`
   (`protocol_version_mismatch`) and closes; neither side hangs.
4. The frontend must not `submit` before receiving a handshake reply with
   `readiness_state == ready`; the engine answers early submissions with a
   typed `engine_unavailable` error.
5. The frontend must treat the advertised `capabilities` as exhaustive: a
   request needing a capability the engine did not advertise is rejected
   frontend-side with a typed capability error and never submitted.

## Per-request state machine

```
                    submit
   (frontend) ────────────────▶ ACTIVE
   ACTIVE:   event(non-terminal)*  →  ACTIVE
             event(terminal_kind=finished|aborted|error, per sequence)
             all sequences terminal → COMPLETE (engine forgets request_id)
   abort:    ACTIVE   → engine sends abort_ack{known: true},
                        then exactly one terminal event (aborted) per
                        open sequence, then COMPLETE
             COMPLETE / unknown id → abort_ack{known: false}; idempotent
```

Ordering invariants (enforced by client-side checkers; violations are
protocol errors):

- `event_index` is monotonically increasing per `(request_id,
  sequence_index)`, starting at 0, without gaps.
- Exactly one terminal event per sequence; no events after it.
- Prompt metadata (`prompt_token_ids` echo, `prompt_logprobs`) appears only
  on an `event_index == 0` event.

## Abort-race matrix

| Race | Behavior |
|---|---|
| abort before submit | `abort_ack{known: false}`; typed `unknown_request` semantics client-side |
| abort during stream | `abort_ack{known: true}`, then a terminal `aborted` event per open sequence; no further events |
| abort after terminal | idempotent no-op: `abort_ack{known: false}` |
| duplicate abort | idempotent: second ack `known: false` (or `true` if still draining); never an error |
| abort for unknown id | `abort_ack{known: false}`; never silence |

## Errors

Engine-side exceptions surface as typed `event` terminals
(`terminal_kind=error`, `error_code`, single-line `error_message`) for
request-scoped failures, or as `error` messages for connection-scoped
failures. Stack traces and language-specific exception objects never cross
the wire.

## Slow-consumer policy

Per-connection engine-side send buffers are bounded. When a frontend
connection stops draining events:

1. The engine blocks the send for that connection up to a grace timeout.
2. On timeout, the engine disconnects that connection and aborts its
   in-flight requests engine-side, recording a typed `slow_consumer` error.
3. Other connections are unaffected.

Slow *HTTP end-clients* are a frontend concern: the frontend buffers
bounded amounts per HTTP response and uses the normal `abort` path when the
HTTP client disconnects.

## Security

This protocol carries no authentication in version 1 and defaults to
localhost transports (`tcp://127.0.0.1` / `ipc://`). Threat note: anyone
who can connect to the endpoint can submit generation work, abort other
requests on the same engine, and read model outputs. Do not expose the
endpoint beyond localhost or a trusted network segment; cross-host
deployments require an authenticated tunnel until socket auth lands.
