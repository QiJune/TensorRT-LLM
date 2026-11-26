# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Snapshots of what the executor gives, as plain JSON: one snapshot per line of a file.

``users_view`` is what users read from each ``GenerationResult``, by field name, and
``step_changes`` turns the views after every step of a flow into the first view and what each
later step changed. ``request_state`` is a request rank 0 built and enqueued.

``encode`` gives the JSON values they hold: an enum by its name, a ``Logprob`` as
``[logprob, rank]``, a tensor or an array as ``{"dtype", "shape", "sha256"}`` (of its bytes),
bytes as hex, a bound (native) object as its pickled state, a non-finite float as its ``repr``,
and a record (a dataclass, a named tuple, a view) as an object of its fields that are not None.

``dump`` writes a snapshot file and ``load`` reads it: JSON Lines, a ``{key: snapshot}`` object
per line. ``assert_recorded`` compares a snapshot with its recording, dict order included.

The module imports nothing from TensorRT-LLM and reads only long-standing attributes of results
and requests, so it takes the same snapshots of different versions of the executor.
"""

import dataclasses
import datetime
import enum
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "assert_recorded",
    "dump",
    "encode",
    "load",
    "request_state",
    "step_changes",
    "users_view",
]


def _array(dtype: str, shape, data: bytes) -> dict:
    return {"dtype": dtype, "shape": list(shape), "sha256": hashlib.sha256(data).hexdigest()}


def _key(key) -> str:
    """A dict key as a JSON object's."""
    if isinstance(key, enum.Enum):
        return key.name
    if isinstance(key, (str, int, float)):
        return str(key)
    raise TypeError(f"no snapshot encoding for a {type(key).__qualname__} dict key")


def _record(fields: dict) -> dict:
    """A record: its fields that are not None."""
    return {name: encode(value) for name, value in fields.items() if value is not None}


def encode(value: Any) -> Any:
    """``value`` as JSON values; see the module docstring.

    Raises:
        TypeError: ``value`` holds something snapshots have no encoding for.
    """
    if isinstance(value, enum.Enum):
        return value.name
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous().reshape(-1)
        return _array(str(value.dtype), value.shape, t.view(torch.uint8).numpy().tobytes())
    if isinstance(value, np.ndarray):
        return _array(str(value.dtype), value.shape, np.ascontiguousarray(value).tobytes())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {_key(k): encode(v) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return _record(value._asdict())
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        names = [f.name for f in dataclasses.fields(value)]
        if names == ["logprob", "rank"]:
            return [encode(value.logprob), encode(value.rank)]
        return _record({name: getattr(value, name) for name in names})
    if type(value).__module__.startswith("tensorrt_llm.bindings"):
        return encode(value.__getstate__())
    raise TypeError(f"no snapshot encoding for a {type(value).__qualname__}")


def _output_view(output) -> dict:
    """What users read from one of a result's outputs (a ``CompletionOutput``)."""
    return _record(
        {
            "token_ids": list(output.token_ids),
            "logprobs": output.logprobs,
            "cumulative_logprob": output.cumulative_logprob,
            "finish_reason": output.finish_reason,
            "prompt_logprobs": output.prompt_logprobs,
            "disaggregated_params": output.disaggregated_params,
            "spec_dec_counters": output._spec_dec_counters,
            "request_perf_metrics": output.request_perf_metrics,
            "generation_logits": output.generation_logits,
            "additional_context_outputs": output.additional_context_outputs,
            "additional_generation_outputs": output.additional_generation_outputs,
        }
    )


def users_view(results: dict) -> dict:
    """What users read from each ``GenerationResult`` of ``results``, by client id."""
    view = {}
    for client_id, result in sorted(results.items()):
        view[str(client_id)] = _record(
            {
                "outputs": [_output_view(output) for output in result._outputs],
                "output_indices": [output.index for output in result.outputs],
                "context_logits": result.context_logits,
                "disaggregated_params": result.disaggregated_params,
                "done": result._done,
                "error": result._error_msg,
                "decoding_iter": result.decoding_iter,
                "avg_decoded_tokens_per_iter": result.avg_decoded_tokens_per_iter,
                "cached_tokens": result.cached_tokens,
                "ctx_computed_tokens": result.ctx_computed_tokens,
                "ctx_first_begin": result.ctx_first_begin,
                "ctx_num_chunks": result.ctx_num_chunks,
                "per_pos_drafted": result.per_pos_drafted,
                "per_pos_accepted": result.per_pos_accepted,
                "spec_dec_totals": result.spec_dec_totals,
                "time_breakdown_metrics": result.time_breakdown_metrics,
                "metrics": dict(result.metrics_dict),
            }
        )
    return view


def _text(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def _changed(before: dict, after: dict) -> dict:
    """The fields whose values ``after`` changed from ``before``'s; null for one it leaves out."""
    names = [*after, *(name for name in before if name not in after)]
    return {
        name: after.get(name) for name in names if _text(before.get(name)) != _text(after.get(name))
    }


def step_changes(views: list) -> list:
    """The users' views after every step: the first view, then what each later step changed.

    A step's changes hold, by client id, the fields of the view that changed and, for the outputs,
    a list of each output's changed fields; a field that became None is null. Unlike the views,
    they hold what stays the same from step to step only once.
    """
    changes = views[:1]
    for before, after in zip(views, views[1:]):
        step = {}
        for client_id, view in after.items():
            earlier = before.get(client_id, {})
            changed = _changed(earlier, view)
            before_outputs, outputs = earlier.get("outputs"), view.get("outputs")
            if "outputs" in changed and len(before_outputs or ()) == len(outputs or ()) > 0:
                changed["outputs"] = [_changed(b, a) for b, a in zip(before_outputs, outputs)]
            if changed:
                step[client_id] = changed
        changes.append(step)
    return changes


def _processor(processor) -> Any:
    """A logits processor: its function's name, or its class's name and its attributes."""
    if inspect.isfunction(processor) or inspect.isbuiltin(processor):
        return processor.__qualname__
    return {type(processor).__qualname__: encode(vars(processor))}


def _prefill_usage(params) -> dict:
    """A request's disaggregated params, of whose ctx_usage the engine reads two counts.

    They are the prompt and cached token counts.
    """
    state = encode(params)
    usage = params.ctx_usage
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
        state["ctx_usage"] = [usage.get("prompt_tokens", 0), cached]
    return state


def request_state(enqueued) -> dict:
    """What rank 0 enqueued, as the engine reads it: the bound request and its Python inputs.

    ``enqueued`` is an ``ExecutorRequest``; ``request`` is its bound request's pickled state.
    """
    inputs = {}
    for field in dataclasses.fields(enqueued.inputs):
        value = getattr(enqueued.inputs, field.name)
        if field.name == "logits_post_processors" and value is not None:
            value = [_processor(processor) for processor in value]
        elif field.name == "disaggregated_params" and value is not None:
            value = _prefill_usage(value)
        inputs[field.name] = value
    return {"request": encode(enqueued.request.__getstate__()), "inputs": _record(inputs)}


def _short(value) -> str:
    text = _text(value)
    return text if len(text) <= 300 else f"{text[:300]}..."


def _first_difference(got, want, path: str = "") -> str:
    """Where two JSON values first differ, as a path into them, and what each has there."""
    if isinstance(got, dict) and isinstance(want, dict):
        if list(got) != list(want):
            return f"{path or '.'}: keys {list(got)} != {list(want)}"
        for key in got:
            if _text(got[key]) != _text(want[key]):
                return _first_difference(got[key], want[key], f"{path}[{key!r}]")
    if isinstance(got, list) and isinstance(want, list):
        if len(got) != len(want):
            return f"{path or '.'}: {len(got)} items != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            if _text(g) != _text(w):
                return _first_difference(g, w, f"{path}[{i}]")
    return f"{path or '.'}: {_short(got)} != {_short(want)}"


def assert_recorded(got, recorded, what: str) -> None:
    """Assert that ``got``, JSON values, is the snapshot ``recorded``, in dict order too.

    The order is what users see: a token's logprobs come in the sampler's order.
    """
    if _text(got) != _text(recorded):
        raise AssertionError(
            f"{what} differs from its snapshot at {_first_difference(got, recorded)}"
        )


def load(path: Path) -> dict:
    """The snapshots of the file ``path``, by key."""
    entries = {}
    for line in Path(path).read_text().splitlines():
        if line:
            ((key, snapshot),) = json.loads(line).items()
            entries[key] = snapshot
    return entries


def dump(path: Path, entries: dict) -> None:
    """Write the snapshots ``entries`` (JSON values, by key) to ``path``, one per line, in order."""
    lines = (_text({key: snapshot}) for key, snapshot in entries.items())
    Path(path).write_text("".join(f"{line}\n" for line in lines))
