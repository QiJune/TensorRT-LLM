# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank 0's half of the executor's wire: the engine's requests from frames, its rows to frontends.

``submit`` builds the executor's requests from a ``SubmitFrame``; ``publish`` sends each frontend
(or postprocess worker) one ``StepOutput`` per step, by column, and tensor bytes beside it. A
response the wire cannot say ends its request with an error.
"""

import functools
import struct
from typing import TYPE_CHECKING, Callable, Optional

import torch

from ..bindings import executor as tllm
from ..logger import logger
from ..sampling_params import LogprobParams
from .logprobs import (
    SimpleTokenLogprobs,
    TokenLogprobs,
    compute_prompt_logprobs,
    prompt_logprobs_from_reduced,
    reduce_prompt_logprobs,
)
from .wire import (
    DISAGG_PHASES,
    LOGPROB_MODES,
    StepFrame,
    SubmitFrame,
    decode_disagg,
    decode_multimodal,
    decode_prefill_usage,
    device_tensor,
    encode_first_gen_logprobs,
    encode_prefill_usage,
    native,
    tensor_ref,
)

if TYPE_CHECKING:
    from .._torch.pyexecutor.llm_request import LlmResponse, LlmResult
    from .base_worker import BaseWorker
    from .utils import ErrorResponse

# the extras of a request whose rows rank 0 prepares in Python
_OUTPUTS_EXTRAS = frozenset(
    ("prompt_logprobs", "context_logits", "tensor_outputs", "disagg", "multimodal")
)

# a result's groups of additional outputs; a row sends each output as the tensor "<group>/<name>"
_OUTPUT_GROUPS = ("additional_context_outputs", "additional_generation_outputs")

_NO_DEFAULT_MAX_TOKENS = "`default_max_tokens` cannot be deduced"

# a warning or an error, worded from an ingress() status' two values
_Message = Optional[Callable[[int, int], str]]


class _Unsendable(ValueError):
    """A row the wire cannot say; it ends its request with an error."""


class IncompleteContextLogitsError(RuntimeError):
    """Context logits do not cover the whole prompt, so prompt logprobs cannot be aligned."""


class RequestEntry:
    """Rank 0's record of a request that came in a ``SubmitFrame``, kept in ``worker._results``.

    ``kind`` is how its sequences come back (``native()``'s ``KIND_SINGLE``: one,
    ``KIND_CHILDREN``: n > 1 sampled, ``KIND_BEAMS``: beam search); ``decoding_stats`` is whether
    its rows say the decoding counters (OutputOptions.decoding_stats) or its sequences' ends do;
    ``first`` is cleared once its first row is published; ``frontend_context`` is the request's
    (Request.frontend_context) until its first row carries it.
    """

    __slots__ = ("first", "kind", "decoding_stats", "frontend_context")

    def __init__(self, kind: int, decoding_stats: bool, frontend_context: Optional[bytes]):
        self.first = True
        self.kind = kind
        self.decoding_stats = decoding_stats
        self.frontend_context = frontend_context


class OutputsEntry(RequestEntry):
    """The ``RequestEntry`` of a request whose rows rank 0 prepares in Python.

    Rank 0 prepares prompt logprobs, logits, additional outputs and handoffs. ``logprob_params``
    are the prompt logprobs the request asked for, until its last row; ``prompt_token_ids`` is
    the prompt they score; ``cached_prompt_logprobs`` holds a streaming request's once computed;
    ``prompt_first_token`` is the output token the cached prompt logprobs' last position was
    scored against, which every later row repeats; ``ctx_usage_sent`` is the ctx_usage the
    frontend has.
    """

    __slots__ = (
        "logprob_params",
        "streaming",
        "prompt_token_ids",
        "cached_prompt_logprobs",
        "prompt_first_token",
        "context_logits",
        "tensor_outputs",
        "handoffs",
        "ctx_usage_sent",
    )

    def __init__(
        self,
        kind: int,
        decoding_stats: bool,
        frontend_context: Optional[bytes],
        *,
        streaming: bool,
        logprob_params: Optional[LogprobParams],
        prompt_token_ids: Optional[list],
        context_logits: bool,
        tensor_outputs: bool,
        handoffs: bool,
        ctx_usage_sent: Optional[dict],
    ):
        super().__init__(kind, decoding_stats, frontend_context)
        self.logprob_params = logprob_params
        self.streaming = streaming
        self.prompt_token_ids = prompt_token_ids
        self.cached_prompt_logprobs = None
        self.prompt_first_token = None
        self.context_logits = context_logits
        self.tensor_outputs = tensor_outputs
        self.handoffs = handoffs
        self.ctx_usage_sent = ctx_usage_sent


@functools.cache
def _max_tokens_messages() -> dict[int, tuple[_Message, _Message]]:
    """``ingress``'s statuses: the warning and the error ``_deduce_max_tokens`` gives for each."""
    from .utils import context_length_exceeded_message

    n = native()
    return {
        n.STATUS_WARN_NO_MAX_SEQ_LEN: (lambda asked, limit: _NO_DEFAULT_MAX_TOKENS, None),
        n.STATUS_WARN_TRUNCATED: (
            lambda asked, limit: (
                f"User-specified `max_tokens` ({asked}) is greater than deduced "
                f"`default_max_tokens` ({limit}), using default_max_tokens instead."
            ),
            None,
        ),
        n.STATUS_ERR_NEED_MAX_TOKENS: (
            lambda asked, limit: _NO_DEFAULT_MAX_TOKENS,
            lambda asked,
            limit: "`max_tokens` must be set when `default_max_tokens` cannot be deduced",
        ),
        n.STATUS_ERR_CONTEXT_EXCEEDED: (
            None,
            lambda asked, limit: context_length_exceeded_message(
                max_context_length=limit, num_prompt_tokens=asked
            ),
        ),
        n.STATUS_ERR_MAX_TOKENS: (
            None,
            lambda asked, limit: f"`max_tokens` ({asked}) must be greater than 0",
        ),
    }


def _is_float32(value) -> bool:
    return isinstance(value, float) and struct.unpack("f", struct.pack("f", value))[0] == value


def _compute_prompt_logprobs(
    params: LogprobParams, prompt_token_ids: list, result: "LlmResult"
) -> Optional[TokenLogprobs | SimpleTokenLogprobs]:
    """The prompt logprobs of a request's first row, ``result``, from its context logits.

    None when the request was cancelled before it ran, so no context logits exist.

    Raises:
        IncompleteContextLogitsError: the context logits do not cover the whole prompt.
    """
    context_logits = result.context_logits
    compact_rows = result.context_prompt_logprob_rows
    executor_result = result.get_result()
    assert executor_result is not None, "result must not be None when prompt_logprobs is requested."
    if (
        context_logits is None
        and compact_rows is None
        and executor_result.is_sequence_final
        and executor_result.finish_reasons
        and all(reason == tllm.FinishReason.CANCELLED for reason in executor_result.finish_reasons)
    ):
        # cancellation before execution leaves no prompt logits to report
        return None
    assert context_logits is not None or compact_rows is not None, (
        "context_logits must not be None when prompt_logprobs is requested."
    )
    # Single element list
    first_generation_token = executor_result.output_token_ids[0][:1]
    assert first_generation_token, (
        "first generation token must not be empty when prompt_logprobs is requested."
    )
    # Pass prompt_token_ids with an offset of 1 for correct mapping to the context logits
    targets = prompt_token_ids[1:] + first_generation_token
    if compact_rows is None:
        return compute_prompt_logprobs(
            params.prompt_logprobs,
            context_logits,
            targets,
            simple_prompt_logprobs=params.prompt_logprobs_simple_format,
        ).prompt
    rows = torch.cat(compact_rows)
    if rows.shape[0] + 1 != len(targets):
        # Row t scores prompt token t + 1; a reused KV cache prefix produces
        # no rows, so the remaining rows cannot be aligned with the prompt.
        raise IncompleteContextLogitsError(
            f"prompt logprobs need context logits for all "
            f"{len(targets)} prompt tokens, got {rows.shape[0] + 1} "
            "(a reused KV cache prefix is not rescored)"
        )
    tail = result.context_prompt_last_logprobs
    assert tail is not None, "Completed prompt must retain its last prediction row"
    # The executor already normalized the last row on the logits' device,
    # matching the other rows; normalizing it again on the host would not.
    rows = torch.cat(
        [
            rows,
            reduce_prompt_logprobs(
                tail, first_generation_token, params.prompt_logprobs, normalized=True
            ),
        ]
    )
    return prompt_logprobs_from_reduced(
        rows, targets, params.prompt_logprobs, params.prompt_logprobs_simple_format
    )


def _prompt_columns(prompt: list, entry: OutputsEntry) -> tuple:
    """Prompt logprobs as engine.proto's TokenLogprobs columns.

    The columns are ids, logprobs, ranks, top-k ends, top-k ids and top-k logprobs. Each position
    predicts the next prompt token, the last ``entry.prompt_first_token``, the output token it was
    scored against; its dict has the top-k by rank, then that token when it is not among them.

    Raises:
        _Unsendable: the logprobs do not fit the columns.
    """
    targets = (entry.prompt_token_ids[1:] + entry.prompt_first_token)[: len(prompt)]
    if len(targets) != len(prompt):
        raise _Unsendable("prompt logprobs for another prompt")
    k = entry.logprob_params.prompt_logprobs
    ids, values, ranks, ends, topk_ids, topk_values = [], [], [], [], [], []
    for target, logprob in zip(targets, prompt):
        ids.append(target)
        if isinstance(logprob, float):
            if not _is_float32(logprob):
                raise _Unsendable("a prompt logprob that is not a float32")
            values.append(logprob)
            ranks.append(0)
            ends.append(len(topk_ids))
            continue
        items = list(logprob.items())
        for rank, (token, candidate) in enumerate(items[:k], 1):
            if candidate.rank != rank or not _is_float32(candidate.logprob):
                raise _Unsendable("prompt top-k logprobs whose ranks are not 1 to k")
            topk_ids.append(token)
            topk_values.append(candidate.logprob)
        tail = items[k:]
        if tail and (len(tail) != 1 or tail[0][0] != target):
            raise _Unsendable("a prompt logprob dict of an unknown shape")
        chosen = tail[0][1] if tail else logprob.get(target)
        if chosen is None or not _is_float32(chosen.logprob):
            raise _Unsendable("a prompt token without a float32 logprob")
        values.append(chosen.logprob)
        # 0: a rank the sampler did not compute
        ranks.append(chosen.rank or 0)
        ends.append(len(topk_ids))
    return ids, values, ranks, ends, topk_ids, topk_values


def _prepare_handoffs(entry: OutputsEntry, result, starts: bool, out: dict) -> None:
    """What a disaggregated or multimodal request's row hands over (sealed natively).

    That is a prefill part's remote tail and first-token logprobs, and embedding and mRoPE
    handles; and the prefill part's usage when the engine learned it with the KV, on the first
    and last row.

    Raises:
        _Unsendable: that usage changed on a middle row, which no row can say.
    """
    usage = result.ctx_usage
    if usage != entry.ctx_usage_sent:
        counts = encode_prefill_usage(usage)
        if (
            counts is None
            or usage != decode_prefill_usage(*counts)
            or not (starts or result.is_final)
        ):
            raise _Unsendable("a prefill usage that changed on a middle row")
        out["prefill_usage"] = counts
        entry.ctx_usage_sent = usage
    kv = {}
    if (tail := result.remote_tail_start) is not None:
        kv["remote_tail_start"] = tail
    if (first_gen := result.first_gen_log_probs) is not None:
        kv["first_gen_log_probs"] = encode_first_gen_logprobs(first_gen)
    if kv:
        out["kv_export"] = kv
    encoder = {}
    if (handles := result.mm_embedding_handles) is not None:
        encoder["mm_embedding_handles"] = handles
    if (ids := result.mrope_position_ids_handle) is not None:
        encoder["mrope_position_ids_handle"] = ids
        encoder["mrope_position_deltas_handle"] = result.mrope_position_deltas_handle
    if encoder:
        out["encoder_export"] = encoder


def _prepare_row(entry: OutputsEntry, response: "LlmResponse", starts: bool) -> Optional[dict]:
    """What Python computes for one row of ``entry``'s request, or None when there is none.

    That is prompt logprobs on the row that starts a sequence, tensors, and handoffs.

    Raises:
        IncompleteContextLogitsError: the prompt logprobs cannot be computed.
        _Unsendable: the wire cannot say the row.
    """
    result = response.result
    out = {}
    if entry.logprob_params is not None and starts:
        prompt = entry.cached_prompt_logprobs
        if prompt is None:
            prompt = _compute_prompt_logprobs(entry.logprob_params, entry.prompt_token_ids, result)
            entry.prompt_first_token = result.get_result().output_token_ids[0][:1]
            if entry.streaming:
                entry.cached_prompt_logprobs = prompt
        if prompt is not None:
            out["prompt_logprobs"] = _prompt_columns(prompt, entry)
    tensors = {}
    if entry.context_logits and starts and (logits := result.context_logits) is not None:
        tensors["context_logits"] = tensor_ref(logits)
    if entry.tensor_outputs:
        if (logits := result.generation_logits) is not None:
            tensors["generation_logits"] = tensor_ref(logits)
        groups = []
        for group in _OUTPUT_GROUPS:
            if (outputs := getattr(result, group)) is None:
                continue
            groups.append(group)
            for name, tensor in outputs.items():
                tensors[f"{group}/{name}"] = tensor_ref(tensor)
        out["groups"] = groups
    if any(ref is None for ref in tensors.values()):
        raise _Unsendable("a tensor of a dtype engine.proto has no name for")
    if entry.handoffs:
        _prepare_handoffs(entry, result, starts, out)
    if tensors:
        out["tensors"] = tensors
    return out or None


def _refuse_request(
    worker: "BaseWorker", response: "LlmResponse", error: Exception | str
) -> "ErrorResponse":
    """Answer ``response``'s request with ``error``, failing only this request.

    The engine stops generating for the client, which has been answered;
    ``drop_refused_response`` drops what it still emits.
    """
    from .utils import ErrorResponse

    if not response.result.is_final:
        worker._refused_client_ids.add(response.client_id)
        worker.abort_request(response.client_id)
    return ErrorResponse(response.client_id, str(error), response.request_id)


def drop_refused_response(worker: "BaseWorker", response: "LlmResponse") -> bool:
    """Whether ``response`` belongs to a request already answered with an error.

    Such a request was aborted, so the engine may still emit steps that were in
    flight and a final cancellation; the client must see none of them.
    """
    from .._torch.pyexecutor.llm_request import release_undelivered_response

    if response.client_id not in worker._refused_client_ids:
        return False
    if response.has_error() or response.result.is_final:
        worker._refused_client_ids.discard(response.client_id)
    release_undelivered_response(response)
    return True


def _prepare(worker: "BaseWorker", responses: list, prepared_client_ids: set[int]) -> list:
    """What Python computes for the rows of ``prepared_client_ids``' requests, by response.

    See ``_prepare_row``. A response whose prompt logprobs cannot be computed, or that the wire
    cannot say, is replaced by its error, and its request aborted.

    A streaming request computes its prompt logprobs once; the sequences of an n > 1 request
    that start later repeat them, scored against the first token of the sequence that computed
    them.
    """
    children = native().KIND_CHILDREN
    prepared = [None] * len(responses)
    seen = set()
    for i, response in enumerate(responses):
        client_id = getattr(response, "client_id", None)
        if client_id not in prepared_client_ids:
            continue
        entry = worker._results.get(client_id)
        if not isinstance(entry, OutputsEntry):
            prepared_client_ids.discard(client_id)
            continue
        result = getattr(response, "result", None)
        if result is None or response.has_error():
            continue
        starts = (entry.first and client_id not in seen) or entry.kind == children
        seen.add(client_id)
        try:
            prepared[i] = _prepare_row(entry, response, starts)
        except IncompleteContextLogitsError as error:
            responses[i] = _refuse_request(worker, response, error)
            continue
        except _Unsendable as error:
            responses[i] = _refuse_request(worker, response, f"cannot send the response: {error}")
            continue
        if result.is_final:
            entry.logprob_params = None
    return prepared


def _load_lora(worker: "BaseWorker", lora: tuple) -> Optional[tuple]:
    """(lora_config, path, ckpt source) of a request's adapter, which rank 0 loads first.

    None without a LoRA manager; ``RequestError`` when the adapter cannot be loaded.
    """
    if worker._lora_manager is None:
        return None
    adapter_id, path, source = lora
    return worker._load_lora_config(adapter_id, path, source), path, source


def _apply_extra(extra: dict, request, inputs, lora: Optional[tuple]) -> None:
    """Set what ``ingress`` leaves to Python on the request and its inputs."""
    if lora is not None:
        request.lora_config, inputs.lora_path, inputs.lora_ckpt_source = lora
    if (scheduling := extra.get("scheduling")) is not None:
        from ..scheduling_params import SchedulingParams

        rank, relax, hierarchy, affinity = scheduling
        inputs.scheduling_params = SchedulingParams(
            attention_dp_rank=rank,
            attention_dp_relax=relax,
            agent_hierarchy=hierarchy,
            subagent_affinity_id=affinity,
        )
    if (conversation_id := extra.get("conversation_id")) is not None:
        from ..conversation_params import ConversationParams

        inputs.conversation_params = ConversationParams(conversation_id=conversation_id)
    if (hints := extra.get("kv_hints")) is not None:
        from ..kv_hints.protocol import KvHintsSnapshot

        inputs.kv_hints = KvHintsSnapshot(hints)
    if "watermark" in extra:
        inputs.watermarking = extra["watermark"]
    if "logprobs_values_only" in extra:
        inputs.logprobs_simple_format = True
    if (prompt := extra.get("prompt_logprobs")) is not None:
        inputs.prompt_logprobs, inputs.prompt_logprobs_simple_format = prompt
    if "context_logits" in extra:
        inputs.context_logits_explicit = True
    if "routed_experts" in extra:
        inputs.return_routed_experts = True
    if (disagg := extra.get("disagg")) is not None:
        inputs.disaggregated_params = decode_disagg(disagg)
    if (trimmed := extra.get("prompt_trimmed_prefix")) is not None:
        inputs.prompt_trimmed_prefix = trimmed
    if (bias := extra.get("embedding_bias")) is not None:
        request.embedding_bias = device_tensor(*bias)
    if (pairs := extra.get("token_bias")) is not None:
        inputs.sparse_embedding_bias = pairs
    if (budget := extra.get("thinking_budget")) is not None:
        from ..llmapi.thinking_budget import ThinkingBudgetLogitsProcessor

        inputs.logits_post_processors = [ThinkingBudgetLogitsProcessor(*budget)]
    if (processors := extra.get("logits_processors")) is not None:
        inputs.logits_post_processors = processors


def _multimodal_params(extra: dict, request):
    """A multimodal request's MultimodalParams, or None without multimodal data.

    Its input layout is set on ``request``, which the C++ runtime reads.
    """
    if (multimodal := extra.get("multimodal")) is None:
        return None
    params = decode_multimodal(multimodal)
    if params.has_content():
        if params.multimodal_input is not None:
            request.multimodal_input = params.multimodal_input.to_binding(tllm)
        params.multimodal_input = None
    return params


def _entry(kind: int, decoding_stats: bool, request, inputs, extra: Optional[dict]) -> RequestEntry:
    """Rank 0's record of a request.

    An ``OutputsEntry`` for one whose rows are prepared in Python.
    """
    frontend_context = None if extra is None else extra.get("frontend_context")
    if extra is None or not extra.keys() & _OUTPUTS_EXTRAS:
        return RequestEntry(kind, decoding_stats, frontend_context)
    logprob_params = None
    if inputs.prompt_logprobs is not None:
        logprob_params = LogprobParams(
            prompt_logprobs=inputs.prompt_logprobs,
            drop_context_logits=not inputs.context_logits_explicit,
            prompt_logprobs_simple_format=inputs.prompt_logprobs_simple_format,
        )
    return OutputsEntry(
        kind,
        decoding_stats,
        frontend_context,
        streaming=request.streaming,
        logprob_params=logprob_params,
        prompt_token_ids=list(request.input_token_ids) if logprob_params is not None else None,
        context_logits=inputs.context_logits_explicit,
        tensor_outputs="tensor_outputs" in extra,
        handoffs="disagg" in extra or "multimodal" in extra,
        ctx_usage_sent=None
        if inputs.disaggregated_params is None
        else inputs.disaggregated_params.ctx_usage,
    )


def submit(
    worker: "BaseWorker", frame: SubmitFrame, *, raise_errors: bool = False, result_wait_queue=None
) -> Optional[int]:
    """Enqueue a ``SubmitFrame``'s requests; the engine's id of the last one enqueued.

    A request rank 0 refuses gets its error as an ErrorResponse, or raised with ``raise_errors``,
    which takes a frame of one request. With ``result_wait_queue``, the engine delivers the
    requests' responses there, and rank 0 keeps no record of them.
    """
    from .._torch.pyexecutor.request_inputs import ExecutorRequest, PythonRequestInputs
    from .utils import ErrorResponse, RequestError

    assert not raise_errors or len(frame.client_ids) == 1, "raise_errors takes one request"

    def reject(client_id: int, error: RequestError) -> None:
        if raise_errors:
            raise error
        logger.error(f"submit request failed: {error}")
        worker._await_response_helper.temp_error_responses.put(
            ErrorResponse(client_id, str(error), client_id)
        )

    request_id = None

    worker.start()
    if frame.refusal is not None:
        for client_id in frame.client_ids:
            reject(client_id, RequestError(frame.refusal))
        return None
    cp_size = 1
    if hasattr(worker, "mapping") and worker.mapping.cp_size is not None:
        cp_size = worker.mapping.cp_size
    try:
        requests = native().ingress(
            frame.data,
            frame.objects or [],
            getattr(worker, "max_seq_len", None),
            cp_size,
            worker.engine.is_generation_model,
        )
    except (RuntimeError, TypeError, ValueError) as e:  # a frame rank 0 cannot read
        for client_id in frame.client_ids:
            reject(client_id, RequestError(f"cannot read the request: {e}"))
        return None
    for (
        client_id,
        request,
        status,
        asked,
        limit,
        num_logprobs,
        mode,
        end_id,
        arrival_time,
        extra,
        kind,
        decoding_stats,
    ) in requests:
        lora = None
        if extra is not None and "lora" in extra:
            try:
                lora = _load_lora(worker, extra["lora"])
            except RequestError as e:
                reject(client_id, e)
                continue
        if extra is not None and "disagg" in extra:
            assert (
                worker.engine.kv_cache_transceiver is not None
                or DISAGG_PHASES[extra["disagg"][0]] == "context_and_generation"
            ), (
                "kv_cache_transceiver is disabled, please set 'cache_transceiver_config: "
                "backend:<backend_type>` in config file for disaggregated serving"
            )
        warning, error = _max_tokens_messages().get(status, (None, None))
        if warning is not None:
            logger.warning(warning(asked, limit))
        if error is not None:
            reject(client_id, RequestError(error(asked, limit)))
            continue
        if request is None:
            reject(client_id, RequestError(extra["error"]))
            continue
        # any failure from here on is this request's error
        try:
            inputs = PythonRequestInputs(
                num_logprobs=num_logprobs,
                context_logits_explicit=False,
                logprobs_mode=LOGPROB_MODES[mode],
                arrival_time=arrival_time,
                original_end_id=end_id,
            )
            if extra is not None:
                _apply_extra(extra, request, inputs, lora)
                multimodal = _multimodal_params(extra, request)
                if multimodal is not None:
                    # the handles the frontend shared the tensors through
                    multimodal.to_tensor("multimodal_data")
                    inputs.multimodal_data = multimodal.multimodal_data
            if result_wait_queue is not None:
                request_id = worker.engine.enqueue_request(
                    ExecutorRequest(request, inputs), result_wait_queue=result_wait_queue
                )
                continue
            entry = _entry(kind, decoding_stats, request, inputs, extra)
            worker._results[client_id] = entry
            if isinstance(entry, OutputsEntry):
                worker._prepared_client_ids.add(client_id)
            request_id = worker.engine.enqueue_request(ExecutorRequest(request, inputs))
            worker._client_id_to_request_id[client_id] = request_id
        except Exception as e:
            worker._results.pop(client_id, None)
            worker._prepared_client_ids.discard(client_id)
            reject(client_id, RequestError(str(e)))
    return request_id


def _publish(worker: "BaseWorker", responses: list, lanes: int, postprocess: bool) -> tuple:
    """The rows of ``responses`` as one StepOutput per lane, and its tensor buffers.

    Returns (frames, buffers, rest): ``rest`` the responses no RequestEntry waits for. Rank 0
    drops its record of each request whose last row goes out.
    """
    from .utils import FRONTEND_ID_SHIFT

    prepared_client_ids = worker._prepared_client_ids
    if prepared_client_ids:
        # requests that ended without a row of theirs going out
        prepared_client_ids.intersection_update(worker._results.keys())
    prepared = _prepare(worker, responses, prepared_client_ids) if prepared_client_ids else None
    frames, buffers, rest, finals, aborts = native().publish(
        responses, worker._results, lanes, postprocess, FRONTEND_ID_SHIFT, prepared
    )
    for client_id in aborts:
        # a response the wire could not say ended the request with an error row
        worker._refused_client_ids.add(client_id)
        worker.abort_request(client_id)
    for client_id in finals:
        worker._results.pop(client_id, None)
        worker._client_id_to_request_id.pop(client_id, None)
    prepared_client_ids.difference_update(finals)
    return frames, buffers, rest


def _unclaimed(worker: "BaseWorker", rest: list) -> list:
    """The ErrorResponses among the responses no RequestEntry waits for.

    The others are steps of requests that already ended, or never started; their buffers are
    released.
    """
    from .._torch.pyexecutor.llm_request import release_undelivered_response
    from .utils import ErrorResponse

    errors = []
    for response in rest:
        if isinstance(response, ErrorResponse):
            errors.append(response)
        elif not drop_refused_response(worker, response):
            release_undelivered_response(response)
    return errors


def publish(worker: "BaseWorker", responses: list, postprocess: bool) -> list:
    """Send the rows of ``responses`` as one ``StepFrame`` per frontend or worker.

    The lanes are the frontends, or the postprocess workers with ``postprocess``; each gets a
    list holding its frame. Returns the ErrorResponses among the rest, for the caller to send.
    """
    lanes = worker.postproc_queues if postprocess else worker.frontend_result_queues
    frames, buffers, rest = _publish(
        worker, responses, len(lanes) if lanes is not None else 1, postprocess
    )
    for lane, data in enumerate(frames):
        if data:
            queue = lanes[lane] if lanes is not None else worker.result_queue
            queue.put([StepFrame(data, buffers[lane] or None)])
    return _unclaimed(worker, rest)


def publish_here(worker: "BaseWorker", responses: list) -> list:
    """The rows of ``responses`` as a ``StepFrame`` for the frontend in rank 0's process.

    Returns that frame, when there are rows, then the ErrorResponses among the rest.
    """
    (data,), (buffers,), rest = _publish(worker, responses, 1, False)
    frames = [StepFrame(data, buffers)] if data else []
    return frames + _unclaimed(worker, rest)
