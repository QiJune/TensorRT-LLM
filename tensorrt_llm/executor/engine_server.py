# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank 0's half of the executor's proto wire: the engine's requests from frames, its rows to frontends.

``submit`` builds the executor's requests from a ``SubmitFrame``; ``publish`` sends each frontend
(or postprocess worker) one ``StepOutput`` per step, by column. A response the proto cannot carry
keeps the pickled path.
"""

import functools
import struct
import types
from typing import TYPE_CHECKING, Callable, Optional

from ..logger import logger
from ..sampling_params import LogprobParams
from .wire import (
    DISAGG_PHASES,
    LOGPROB_MODES,
    StepFrame,
    SubmitFrame,
    decode_disagg,
    decode_frontend_context,
    decode_multimodal,
    decode_prefill_usage,
    device_tensor,
    encode_first_gen_logprobs,
    encode_prefill_usage,
    native,
    tensor_ref,
)

if TYPE_CHECKING:
    from .base_worker import BaseWorker

_NO_PERF_METRICS = types.SimpleNamespace(return_perf_metrics=False)

# the extras of a request whose rows rank 0 prepares in Python or that asks for perf metrics
_OUTPUTS_EXTRAS = frozenset(
    ("prompt_logprobs", "context_logits", "tensor_outputs", "perf_metrics", "disagg", "multimodal")
)

# a result's groups of additional outputs; a row sends each output as the tensor "<group>/<name>"
_OUTPUT_GROUPS = ("additional_context_outputs", "additional_generation_outputs")

# Tensors one publish sends inline at most, which keeps each StepOutput well below protobuf's 2 GiB
_TENSOR_BYTES = 1 << 30

_NO_DEFAULT_MAX_TOKENS = "`default_max_tokens` cannot be deduced"

# a warning or an error, worded from an ingress() status' two values
_Message = Optional[Callable[[int, int], str]]


class RequestEntry:
    """Rank 0's record of a request that came in a ``SubmitFrame``.

    It is kept in ``worker._results``, where a ``GenerationResult`` is kept for a pickled one.
    ``kind`` is how its sequences come back (``native()``'s ``KIND_SINGLE``: one,
    ``KIND_CHILDREN``: n > 1 sampled, ``KIND_BEAMS``: beam search); ``decoding_stats`` is whether
    its rows say the decoding counters (OutputOptions.decoding_stats) or its sequences' ends do;
    ``first`` is cleared once its first row is published; ``frontend_context`` is the request's
    (Request.frontend_context) until its first row, or its first pickled response, carries it.
    ``_streaming`` and the class attributes answer what the pickled path reads when one of its
    responses falls back to it.
    """

    __slots__ = ("first", "kind", "decoding_stats", "_streaming", "frontend_context")
    _logprob_params = None
    sampling_params = _NO_PERF_METRICS

    def __init__(
        self, kind: int, decoding_stats: bool, streaming: bool, frontend_context: Optional[bytes]
    ):
        self.first = True
        self.kind = kind
        self.decoding_stats = decoding_stats
        self._streaming = streaming
        self.frontend_context = frontend_context

    def clear_logprob_params(self) -> None:
        pass


def first_params(entry: RequestEntry) -> tuple:
    """Rank 0: the params a postprocess worker builds a record from, for a pickled response.

    They are (sampling_params, postproc_params, disaggregated_params) from the request's frontend
    context when no row has carried it yet, else Nones: what ``_get_params_for_first_rsp`` gives
    for a pickled request.
    """
    data = entry.frontend_context
    if data is None:
        return None, None, None
    entry.frontend_context = None
    context = decode_frontend_context(data)
    return context.sampling_params, context.postproc_params, context.disaggregated_params


class OutputsEntry(RequestEntry):
    """The ``RequestEntry`` of a request whose rows rank 0 prepares in Python, or that asks for perf metrics.

    Rank 0 prepares prompt logprobs, logits, additional outputs and handoffs. The entry holds what
    ``_compute_pytorch_prompt_logprobs`` reads, and what the pickled path reads when one of its
    responses falls back to it; ``prompt_first_token``, the output token the cached prompt
    logprobs' last position was scored against, which every later row repeats; and
    ``ctx_usage_sent``, the ctx_usage the frontend has.
    """

    __slots__ = (
        "_logprob_params",
        "_generation_request",
        "_cached_prompt_logprobs",
        "prompt_first_token",
        "sampling_params",
        "context_logits",
        "tensor_outputs",
        "handoffs",
        "ctx_usage_sent",
    )

    def __init__(
        self,
        kind: int,
        decoding_stats: bool,
        streaming: bool,
        frontend_context: Optional[bytes],
        *,
        logprob_params: Optional[LogprobParams],
        prompt_token_ids: Optional[list],
        perf_metrics: bool,
        context_logits: bool,
        tensor_outputs: bool,
        handoffs: bool,
        ctx_usage_sent: Optional[dict],
    ):
        super().__init__(kind, decoding_stats, streaming, frontend_context)
        self._logprob_params = logprob_params
        self._generation_request = types.SimpleNamespace(prompt_token_ids=prompt_token_ids)
        self._cached_prompt_logprobs = None
        self.prompt_first_token = None
        self.sampling_params = types.SimpleNamespace(return_perf_metrics=perf_metrics)
        self.context_logits = context_logits
        self.tensor_outputs = tensor_outputs
        self.handoffs = handoffs
        self.ctx_usage_sent = ctx_usage_sent

    @property
    def prepares_rows(self) -> bool:
        """Whether rank 0 prepares the request's rows in Python, beyond its perf metrics."""
        return (
            self._logprob_params is not None
            or self.context_logits
            or self.tensor_outputs
            or self.handoffs
        )

    def clear_logprob_params(self) -> None:
        self._logprob_params = None


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


def _prompt_columns(prompt: list, entry: OutputsEntry) -> Optional[tuple]:
    """Prompt logprobs as engine.proto's TokenLogprobs columns, or None when they do not fit them.

    The columns are ids, logprobs, ranks, top-k ends, top-k ids and top-k logprobs. Each position
    predicts the next prompt token, the last ``entry.prompt_first_token``, the output token it was
    scored against; its dict has the top-k by rank, then that token when it is not among them.
    """
    prompt_ids = entry._generation_request.prompt_token_ids
    targets = (prompt_ids[1:] + entry.prompt_first_token)[: len(prompt)]
    if len(targets) != len(prompt):
        return None
    k = entry._logprob_params.prompt_logprobs
    ids, values, ranks, ends, topk_ids, topk_values = [], [], [], [], [], []
    for target, logprob in zip(targets, prompt):
        ids.append(target)
        if isinstance(logprob, float):
            if not _is_float32(logprob):
                return None
            values.append(logprob)
            ranks.append(0)
            ends.append(len(topk_ids))
            continue
        items = list(logprob.items())
        for rank, (token, candidate) in enumerate(items[:k], 1):
            if candidate.rank != rank or not _is_float32(candidate.logprob):
                return None
            topk_ids.append(token)
            topk_values.append(candidate.logprob)
        tail = items[k:]
        if tail and (len(tail) != 1 or tail[0][0] != target):
            return None
        chosen = tail[0][1] if tail else logprob.get(target)
        if chosen is None or not _is_float32(chosen.logprob):
            return None
        values.append(chosen.logprob)
        ranks.append(chosen.rank)
        ends.append(len(topk_ids))
    return ids, values, ranks, ends, topk_ids, topk_values


def _prepare_handoffs(entry: OutputsEntry, result, starts: bool, out: dict) -> bool:
    """Rank 0: what a disaggregated or multimodal request's row hands over (sealed natively).

    That is a prefill part's remote tail and first-token logprobs, and embedding and mRoPE
    handles; and the prefill part's usage when the engine learned it with the KV, on the first
    and last row. False when the row must keep the pickled path.
    """
    usage = result.ctx_usage
    if usage != entry.ctx_usage_sent:
        counts = encode_prefill_usage(usage)
        if (
            counts is None
            or usage != decode_prefill_usage(*counts)
            or not (starts or result.is_final)
        ):
            return False
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
    return True


def _prepare_row(
    entry: OutputsEntry, response, starts: bool, tensor_budget: int
) -> tuple[dict | bool | None, int]:
    """Rank 0: what Python computes for one row of ``entry``'s request, and its tensors' bytes.

    That is prompt logprobs on the row that starts a sequence, tensors, and handoffs; None when
    there are none, and False when the columns cannot hold the row or its tensors exceed
    ``tensor_budget`` bytes, which keeps it on the pickled path.

    Raises:
        IncompleteContextLogitsError: the prompt logprobs cannot be computed.
    """
    from .base_worker import _compute_pytorch_prompt_logprobs

    result = response.result
    out = {}
    if entry._logprob_params is not None and starts:
        cached = entry._cached_prompt_logprobs is not None
        logprobs = _compute_pytorch_prompt_logprobs(entry, response)
        if not cached:
            entry.prompt_first_token = result.get_result().output_token_ids[0][:1]
        if logprobs is not None and logprobs.prompt is not None:
            columns = _prompt_columns(logprobs.prompt, entry)
            if columns is None:
                return False, 0
            out["prompt_logprobs"] = columns
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
        return False, 0
    tensor_bytes = sum(len(ref[2]) for ref in tensors.values())
    if tensor_bytes > tensor_budget:
        return False, 0
    if entry.handoffs and not _prepare_handoffs(entry, result, starts, out):
        return False, 0
    if tensors:
        out["tensors"] = tensors
    return out or None, tensor_bytes


def _prepare(worker: "BaseWorker", responses: list, prepared_client_ids: set[int]) -> list:
    """Rank 0: what Python computes for the rows of the requests in ``prepared_client_ids``, by response.

    See ``_prepare_row``. A response whose prompt logprobs cannot be computed is replaced by its
    error, as ``_maybe_wrap_response`` does.

    A streaming request computes its prompt logprobs once; the sequences of an n > 1 request
    that start later repeat them, scored against the first token of the sequence that computed
    them.
    """
    from .base_worker import IncompleteContextLogitsError, _refuse_request

    children = native().KIND_CHILDREN
    prepared = [None] * len(responses)
    seen = set()
    tensor_bytes = 0
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
            prepared[i], row_bytes = _prepare_row(
                entry, response, starts, _TENSOR_BYTES - tensor_bytes
            )
        except IncompleteContextLogitsError as error:
            responses[i] = _refuse_request(worker, response, error)
            continue
        if prepared[i] is False:
            continue
        tensor_bytes += row_bytes
        if result.is_final:
            entry.clear_logprob_params()
    return prepared


def _load_lora(worker: "BaseWorker", lora: tuple) -> Optional[tuple]:
    """(lora_config, path, ckpt source) of a request's adapter, as ``_enqueue_request`` sets LoRA up.

    None without a LoRA manager; ``RequestError`` when the adapter cannot be loaded.
    """
    if worker._lora_manager is None:
        return None
    from .request import LoRARequest

    adapter_id, path, source = lora
    config = worker._load_lora_config(LoRARequest(str(adapter_id), adapter_id, path, source))
    return config, path, source


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


def _multimodal_params(extra: dict, request):
    """Rank 0: a multimodal request's MultimodalParams, or None without multimodal data.

    Its input is set on ``request`` as ``_enqueue_request`` sets it.
    """
    if (multimodal := extra.get("multimodal")) is None:
        return None
    from ..bindings import executor as tllm

    params = decode_multimodal(multimodal)
    if params.has_content():
        if params.multimodal_input is not None:
            request.multimodal_input = params.multimodal_input.to_binding(tllm)
        params.multimodal_input = None
    return params


def _entry(kind: int, decoding_stats: bool, request, inputs, extra: Optional[dict]) -> RequestEntry:
    """Rank 0's record of a request.

    An ``OutputsEntry`` for one whose rows are prepared in Python or that asked for perf metrics.
    """
    frontend_context = None if extra is None else extra.get("frontend_context")
    if extra is None or not extra.keys() & _OUTPUTS_EXTRAS:
        return RequestEntry(kind, decoding_stats, request.streaming, frontend_context)
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
        request.streaming,
        frontend_context,
        logprob_params=logprob_params,
        prompt_token_ids=list(request.input_token_ids) if logprob_params is not None else None,
        perf_metrics="perf_metrics" in extra,
        context_logits=inputs.context_logits_explicit,
        tensor_outputs="tensor_outputs" in extra,
        handoffs="disagg" in extra or "multimodal" in extra,
        ctx_usage_sent=None
        if inputs.disaggregated_params is None
        else inputs.disaggregated_params.ctx_usage,
    )


def submit(worker: "BaseWorker", frame: SubmitFrame) -> None:
    """Rank 0: enqueue a ``SubmitFrame``'s requests as ``BaseWorker.submit`` does a pickled request's."""
    from .._torch.pyexecutor.request_inputs import ExecutorRequest, PythonRequestInputs
    from .utils import ErrorResponse, RequestError

    def reject(client_id: int, error: RequestError) -> None:
        logger.error(f"submit request failed: {error}")
        worker._await_response_helper.temp_error_responses.put(
            ErrorResponse(client_id, error, client_id)
        )

    worker.start()
    # from now on rank 0 publishes these requests' rows as StepOutputs
    worker._engine_wire_active = True
    cp_size = 1
    if hasattr(worker, "mapping") and worker.mapping.cp_size is not None:
        cp_size = worker.mapping.cp_size
    try:
        requests = native().ingress(
            frame.data,
            getattr(worker, "max_seq_len", None),
            cp_size,
            worker.engine.is_generation_model,
        )
    except (RuntimeError, TypeError, ValueError) as e:  # a frame rank 0 cannot read
        for client_id in frame.client_ids:
            reject(client_id, RequestError(f"cannot read the request: {e}"))
        return
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
        # any failure from here on is this request's error, as _enqueue_request makes it
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
            entry = _entry(kind, decoding_stats, request, inputs, extra)
            worker._results[client_id] = entry
            if isinstance(entry, OutputsEntry) and entry.prepares_rows:
                worker._prepared_client_ids.add(client_id)
            worker._client_id_to_request_id[client_id] = worker.engine.enqueue_request(
                ExecutorRequest(request, inputs)
            )
        except Exception as e:
            worker._results.pop(client_id, None)
            worker._prepared_client_ids.discard(client_id)
            reject(client_id, RequestError(str(e)))


def publish(worker: "BaseWorker", responses: list, postprocess: bool) -> list:
    """Rank 0: send the rows of ``SubmitFrame`` requests as one ``StepFrame`` per frontend or worker.

    The lanes are the frontends, or the postprocess workers with ``postprocess``. Returns the
    responses that keep the pickled path.
    """
    from .utils import FRONTEND_ID_SHIFT

    lanes = worker.postproc_queues if postprocess else worker.frontend_result_queues
    prepared_client_ids = worker._prepared_client_ids
    if prepared_client_ids:
        # requests whose last response took the pickled path
        prepared_client_ids.intersection_update(worker._results.keys())
    prepared = _prepare(worker, responses, prepared_client_ids) if prepared_client_ids else None
    frames, rest, finals = native().publish(
        responses,
        worker._results,
        RequestEntry,
        len(lanes) if lanes is not None else 1,
        postprocess,
        FRONTEND_ID_SHIFT,
        prepared,
    )
    for response in rest:
        entry = worker._results.get(getattr(response, "client_id", None))
        if isinstance(entry, RequestEntry):
            # a pickled response carries what the request's first row would
            entry.first = False
    for lane, data in enumerate(frames):
        if data:
            queue = lanes[lane] if lanes is not None else worker.result_queue
            queue.put(StepFrame(data))
    for client_id in finals:
        worker._pop_result(client_id)
        prepared_client_ids.discard(client_id)
    return rest
