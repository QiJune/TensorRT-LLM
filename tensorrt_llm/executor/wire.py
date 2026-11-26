# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The executor's proto wire (the engine contract, tensorrt_llm/contracts/engine/v1/engine.proto).

Off by default; with ``TLLM_PROTO_WIRE=1``, a request the proto carries goes from the proxy to
rank 0 as a ``SubmitBatch``, and rank 0 sends its rows back as one ``StepOutput`` per step and
frontend instead of one pickled ``LlmResponse`` per row. Any other request, and any response the
proto cannot carry, keeps the pickled path. The proto bytes ride in the existing queues, which
still sign every message.

This module is what both sides use: the switch, the frames, the native codec
(tensorrt_llm.bindings.internal.engine_wire, which holds the GIL only to read and build Python
objects), and the conversions both sides make, as ``encode_*``/``decode_*`` pairs. The frontend's
half is llmapi/engine_client/proto_wire.py, rank 0's engine_server.py.
"""

import dataclasses
import functools
import os
import pickle  # nosec B403
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from ..disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from ..sampling_params import LogprobMode, SamplingParams

if TYPE_CHECKING:
    from ..inputs.multimodal import MultimodalParams
    from .request import PostprocParams

ENABLED = os.environ.get("TLLM_PROTO_WIRE", "0") == "1"

# engine.proto's DType values
DTYPES = {
    torch.float32: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float8_e4m3fn: 4,
    torch.int64: 5,
    torch.int32: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
_TORCH_DTYPES = {code: dtype for dtype, code in DTYPES.items()}

# engine.proto's DisaggPhase values, in order
DISAGG_PHASES = ("context_and_generation", "context_only", "generation_only")

# engine.proto's LogprobMode values, in order
LOGPROB_MODES = (LogprobMode.RAW, LogprobMode.PROCESSED)

# a decode part's DisaggregatedParams fields that its prefill part hands over (Disagg.kv_import)
_KV_IMPORT_FIELDS = (
    "ctx_request_id",
    "ctx_info_endpoint",
    "ctx_dp_rank",
    "opaque_state",
    "first_gen_tokens",
    "draft_tokens",
    "first_gen_log_probs",
    "first_gen_logits",
    "remote_tail_start",
)

# a prefill part's that its encoder part hands over, in EPD (Disagg.encoder_import)
_ENCODER_IMPORT_FIELDS = (
    "multimodal_embedding_handles",
    "multimodal_hashes",
    "multimodal_positions",
    "multimodal_lengths",
    "multimodal_input_layout",
)

# the mRoPE handles: they go with the KV to a decode part, with the embeddings otherwise
_MROPE_IMPORT_FIELDS = ("mrope_position_ids_handle", "mrope_position_deltas_handle")


@functools.cache
def native() -> ModuleType:
    """The native codec, with this module's tensor conversions registered."""
    from ..bindings.internal import engine_wire

    engine_wire.set_tensor_codec(torch.Tensor, tensor_ref, device_tensor)
    return engine_wire


class SubmitFrame:
    """A ``SubmitBatch`` from a proxy to rank 0, and the client ids of its requests."""

    __slots__ = ("data", "client_ids")

    def __init__(self, data: bytes, client_ids: tuple):
        self.data = data
        self.client_ids = client_ids

    def __reduce__(self):
        return SubmitFrame, (self.data, self.client_ids)


class StepFrame:
    """One step's ``StepOutput`` for one frontend."""

    __slots__ = ("data",)

    def __init__(self, data: bytes):
        self.data = data

    def __reduce__(self):
        return StepFrame, (self.data,)


class FrontendContext(NamedTuple):
    """What a postprocess worker builds a request's record from.

    The frontend sends it with the request (Request.frontend_context), and the request's first
    row brings it back, as the first Input of a pickled request carries these params.
    """

    sampling_params: SamplingParams
    postproc_params: Optional["PostprocParams"]
    streaming: bool
    disaggregated_params: Optional[DisaggregatedParams]


def encode_frontend_context(context: FrontendContext) -> bytes:
    """``context`` as Request.frontend_context's bytes."""
    return pickle.dumps(context)


def decode_frontend_context(data: bytes) -> FrontendContext:
    """The inverse of ``encode_frontend_context``."""
    return pickle.loads(data)  # nosec B301


def to_tensor(dtype: int, shape: list, data: bytes) -> torch.Tensor:
    """A CPU tensor from engine.proto's (dtype, shape, bytes)."""
    torch_dtype = _TORCH_DTYPES.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"engine.proto has no DType {dtype}")
    if not data:
        return torch.empty(shape, dtype=torch_dtype)
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return raw.view(torch_dtype).reshape(shape)


def device_tensor(dtype: int, shape: list, data: bytes, device: Optional[int]) -> torch.Tensor:
    """A tensor from engine.proto's TensorRef, on the device it came from."""
    tensor = to_tensor(dtype, shape, data)
    return tensor if device is None else tensor.to(f"cuda:{device}")


def tensor_ref(tensor: torch.Tensor) -> Optional[tuple]:
    """engine.proto's (dtype, shape, bytes, device) of ``tensor``; None for a dtype it has no name for.

    The bytes go inline, with the CUDA device the tensor was on, where the frontend puts it back.
    """
    dtype = DTYPES.get(tensor.dtype)
    if dtype is None:
        return None
    device = tensor.device.index if tensor.is_cuda else None
    t = tensor.detach().contiguous().cpu()
    return (dtype, list(t.shape), t.reshape(-1).view(torch.uint8).numpy().tobytes(), device)


def encode_prefill_usage(usage) -> Optional[tuple]:
    """engine.proto's PrefillUsage of a ctx_usage dict: (prompt_tokens, cached_tokens).

    They are the two counts the engine reads from it. None when ``usage`` is not such a dict.
    """
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details") or {}
    prompt, cached = (
        usage.get("prompt_tokens", 0),
        (details.get("cached_tokens", 0) if isinstance(details, dict) else None),
    )
    if type(prompt) is not int or type(cached) is not int:
        return None
    return prompt, cached


def decode_prefill_usage(prompt_tokens: int, cached_tokens: int) -> dict:
    """The ctx_usage dict of a PrefillUsage, as the engine builds one from the KV transfer."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "total_tokens": prompt_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def encode_first_gen_logprobs(entries: list) -> list:
    """First-token logprobs as plain values.

    A float, or the dict's (token, logprob, rank) in order.
    """
    return [
        entry
        if isinstance(entry, float)
        else [(token, lp.logprob, lp.rank) for token, lp in entry.items()]
        for entry in entries
    ]


def decode_first_gen_logprobs(entries: list) -> list:
    """The inverse of ``encode_first_gen_logprobs``."""
    # result.py imports this module
    from .logprobs import Logprob

    return [
        entry
        if isinstance(entry, float)
        else {token: Logprob(logprob=lp, rank=rank) for token, lp, rank in entry}
        for entry in entries
    ]


def _set_fields(params: DisaggregatedParams, names: tuple) -> dict:
    """The fields ``names`` of ``params`` that are set."""
    return {name: value for name in names if (value := getattr(params, name)) is not None}


def encode_disagg(params: DisaggregatedParams) -> tuple:
    """engine.proto's Disagg of a request's DisaggregatedParams.

    That is (phase, schedule, disagg_request_id, orig_elapsed_s, kv_import, prefill_usage,
    encoder_import). The imports are dicts of DisaggregatedParams' fields, which the native side
    seals; the mRoPE handles go with the KV to a decode part, with the embeddings otherwise.
    """
    kv = _set_fields(params, _KV_IMPORT_FIELDS)
    encoder = _set_fields(params, _ENCODER_IMPORT_FIELDS)
    if "first_gen_log_probs" in kv:
        kv["first_gen_log_probs"] = encode_first_gen_logprobs(kv["first_gen_log_probs"])
    if "multimodal_input_layout" in encoder:
        encoder["multimodal_input_layout"] = dataclasses.asdict(encoder["multimodal_input_layout"])
    mrope = kv if params.request_type == "generation_only" else encoder
    mrope.update(_set_fields(params, _MROPE_IMPORT_FIELDS))
    return (
        DISAGG_PHASES.index(params.request_type),
        None if params.schedule_style is None else int(params.schedule_style),
        params.disagg_request_id,
        params.orig_elapsed_s,
        kv or None,
        None if params.ctx_usage is None else encode_prefill_usage(params.ctx_usage),
        encoder or None,
    )


def decode_disagg(disagg: tuple) -> DisaggregatedParams:
    """The inverse of ``encode_disagg``.

    Its ctx_usage is built as the engine builds one, from the two counts the engine reads.
    """
    phase, schedule, disagg_request_id, orig_elapsed_s, kv, usage, encoder = disagg
    imported = {**(kv or {}), **(encoder or {})}
    if "first_gen_log_probs" in imported:
        imported["first_gen_log_probs"] = decode_first_gen_logprobs(imported["first_gen_log_probs"])
    return DisaggregatedParams(
        request_type=DISAGG_PHASES[phase],
        disagg_request_id=disagg_request_id,
        orig_elapsed_s=orig_elapsed_s,
        schedule_style=None if schedule is None else DisaggScheduleStyle(schedule),
        ctx_usage=None if usage is None else decode_prefill_usage(*usage),
        **imported,
    )


def encode_multimodal(params: "MultimodalParams") -> tuple:
    """engine.proto's Multimodal of a request's MultimodalParams: (items, model inputs).

    An item is (hash, uuid, position, length, run positions, run lengths). The frontend sends
    only a layout the items can say: one item at least, uuids when one is set, runs when there
    is one.
    """
    mm = params.multimodal_input
    if mm is None:
        return [], params.multimodal_data
    uuids = mm.multimodal_uuids
    offsets = mm.multimodal_item_run_cu_offsets
    run_positions = mm.multimodal_run_positions or []
    run_lengths = mm.multimodal_run_lengths or []
    items = []
    for i, hashes in enumerate(mm.multimodal_hashes):
        runs = slice(offsets[i], offsets[i + 1]) if offsets is not None else slice(0)
        items.append(
            (
                hashes,
                None if uuids is None else uuids[i],
                mm.multimodal_positions[i],
                mm.multimodal_lengths[i],
                run_positions[runs],
                run_lengths[runs],
            )
        )
    return items, params.multimodal_data


def decode_multimodal(multimodal: tuple) -> "MultimodalParams":
    """The inverse of ``encode_multimodal``."""
    from ..inputs.multimodal import MultimodalInput, MultimodalParams

    items, model_inputs = multimodal
    if not items:
        return MultimodalParams(multimodal_input=None, multimodal_data=model_inputs)
    uuids = [item[1] for item in items]
    runs = any(item[4] for item in items)
    offsets = [0]
    for item in items:
        offsets.append(offsets[-1] + len(item[4]))
    return MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=[list(item[0]) for item in items],
            multimodal_positions=[item[2] for item in items],
            multimodal_lengths=[item[3] for item in items],
            multimodal_uuids=uuids if any(u is not None for u in uuids) else None,
            multimodal_item_run_cu_offsets=offsets if runs else None,
            multimodal_run_positions=[p for item in items for p in item[4]] if runs else None,
            multimodal_run_lengths=[n for item in items for n in item[5]] if runs else None,
        ),
        multimodal_data=model_inputs,
    )
