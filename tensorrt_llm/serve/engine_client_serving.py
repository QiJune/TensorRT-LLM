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
"""Experimental engine-client serving path (TLLM_EXPERIMENTAL_ENGINE_CLIENT).

With the flag on, ELIGIBLE OpenAI streaming requests route through
``conversion`` → ``LocalProcessEngineClient`` → ``FrontendResponseAssembler``
and the existing OpenAI stream formatters, never calling
``LLM.generate_async``, never constructing ``PostprocParams``, and never
consuming a ``GenerationResult``. Ineligible requests transparently fall
back to the legacy path, counted per eligibility axis.

Model/tokenizer information on this path is read only through a
locally-built, data-only ``FrontendModelContext``: the assembler and the
formatters use a tokenizer **reloaded from the context's TokenizerSpec**,
not the server's live tokenizer object (the reload-parity test guards
token-for-token equivalence). The context is built locally in V0; a later
remote detach swaps only its delivery.
"""

import asyncio
import hashlib
import os
import uuid
from typing import List, Optional

from tensorrt_llm.executor.engine_client.assembler import FrontendResponseAssembler
from tensorrt_llm.executor.engine_client.contract import (EngineCapabilities,
                                                          FrontendModelContext,
                                                          TokenizerSpec)
from tensorrt_llm.executor.engine_client.conversion import (
    ConversionError, RequestIneligibleError, convert_request,
    prepare_sampling_params_from_context)
from tensorrt_llm.executor.engine_client.local_client import (
    ENGINE_CLIENT_FLAG_ENV, EngineClientConfig, EngineClientConfigError,
    LocalProcessEngineClient, RequestRejectedError,
    resolve_engine_client_flag)
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.logger import logger

__all__ = ["ContractStreamView", "EngineClientServing", "build_model_context"]

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "special_tokens_map.json",
                    "generation_config.json")


class _ViewOutput:
    """Duck-typed CompletionOutput view consumed by the stream formatters."""

    def __init__(self):
        self.index = 0
        self.token_ids_diff: List[int] = []
        self.text_diff: str = ""
        self.logprobs_diff: List[float] = []
        self.finish_reason: Optional[str] = None
        self.stop_reason = None
        self.length: int = 0
        self.disaggregated_params = None
        self.prompt_logprobs = None
        self.token_ids: List[int] = []
        self.text: str = ""
        self.logprobs: List[float] = []
        self.request_perf_metrics = None
        # Character length of the accumulated text BEFORE the current batch
        # (the completions logprob formatter uses it as the text offset).
        self._last_text_len: int = 0


class ContractStreamView:
    """Duck-typed ``GenerationResultBase`` view fed from assembler updates.

    NOT a ``GenerationResult``: no queue, no executor, no runtime state —
    only the read surface the existing OpenAI stream formatters consume.
    """

    def __init__(self, view_id: int, prompt_token_ids):
        self.id = view_id
        self.prompt_token_ids = list(prompt_token_ids)
        self._outputs = [_ViewOutput()]
        self._done = False
        self.cached_tokens = 0
        self.avg_decoded_tokens_per_iter = None
        self.metrics_dict = {}
        self.time_breakdown_metrics = None
        self._disaggregated_params = None

    @property
    def outputs(self):
        return self._outputs

    @property
    def disaggregated_params(self):
        return self._disaggregated_params

    def apply_updates(self, assembler: FrontendResponseAssembler, updates) -> bool:
        """Fold one batch of assembly updates into the view.

        Returns True when the request reached its end (complete or error).
        """
        output = self._outputs[0]
        output.token_ids_diff = []
        output.text_diff = ""
        output.logprobs_diff = []
        output._last_text_len = len(output.text)
        ended = False
        for update in updates:
            if update.kind == "delta":
                output.token_ids_diff.extend(update.new_token_ids)
                output.text_diff += update.text_diff
                if update.logprobs:
                    output.logprobs_diff.extend(update.logprobs)
                if update.finish_reason is not None:
                    output.finish_reason = update.finish_reason
                    output.stop_reason = update.stop_reason
                if update.metrics_cached_tokens is not None:
                    self.cached_tokens = update.metrics_cached_tokens
            elif update.kind == "finish":
                output.finish_reason = update.finish_reason
                output.stop_reason = update.stop_reason
            elif update.kind == "complete":
                self._done = True
                ended = True
                if update.usage and "cached_tokens" in update.usage:
                    self.cached_tokens = update.usage["cached_tokens"]
            elif update.kind == "error":
                self._done = True
                ended = True
                output.finish_reason = output.finish_reason or "error"
                raise _ContractStreamError(update.error_code or "error",
                                           update.error_message or "")
        output.token_ids = assembler.token_ids
        output.text = assembler.text
        output.logprobs = assembler.logprobs
        output.length = len(assembler.token_ids)
        return ended


class _ContractStreamError(RuntimeError):

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _file_manifest(model_dir: str):
    manifest = []
    for name in _TOKENIZER_FILES:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            manifest.append((name, digest.hexdigest()))
    return tuple(manifest)


def _dump_json_or_none(value) -> Optional[str]:
    if value is None:
        return None
    try:
        import json
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


def build_model_context(llm, capabilities: EngineCapabilities) -> FrontendModelContext:
    """Build the data-only model context locally from the in-process engine.

    This is the ONLY point where the contract path touches live model
    objects; everything downstream (normalization, detokenization,
    formatting) consumes the frozen context plus the spec-reloaded
    tokenizer. The remote detach later swaps the delivery (ready-handshake)
    without changing any consumer.
    """
    import json
    model_dir = str(getattr(llm, "_hf_model_dir", None) or "")
    tokenizer = llm.tokenizer
    chat_template = None
    inner = getattr(tokenizer, "tokenizer", None)
    added_tokens_json = special_tokens_json = None
    normalizer_json = pre_tokenizer_json = None
    if inner is not None:
        chat_template = getattr(inner, "chat_template", None)
        try:
            added_tokens_json = _dump_json_or_none(inner.get_added_vocab())
        except Exception:
            pass
        special_tokens_json = _dump_json_or_none(
            getattr(inner, "special_tokens_map", None))
        backend = getattr(inner, "backend_tokenizer", None)
        if backend is not None:
            try:
                backend_config = json.loads(backend.to_str())
                normalizer_json = _dump_json_or_none(
                    backend_config.get("normalizer"))
                pre_tokenizer_json = _dump_json_or_none(
                    backend_config.get("pre_tokenizer"))
            except Exception:
                pass
    hf_config = getattr(llm, "_hf_model_config", None)
    revision = getattr(hf_config, "_commit_hash", None)
    spec = TokenizerSpec(
        uri=model_dir or "unknown",
        files_manifest=_file_manifest(model_dir) if model_dir else (),
        revision=revision if isinstance(revision, str) and revision else None,
        fast=bool(getattr(tokenizer, "is_fast", True)),
        trust_remote_code=bool(getattr(llm.args, "trust_remote_code", False)),
        added_tokens_json=added_tokens_json,
        special_tokens_json=special_tokens_json,
        normalizer_json=normalizer_json,
        pre_tokenizer_json=pre_tokenizer_json,
    )
    generation_config = getattr(llm, "_generation_config", None)
    generation_stop_ids = getattr(generation_config, "eos_token_id", None)
    if isinstance(generation_stop_ids, int):
        generation_stop_ids = [generation_stop_ids]
    chat_template_version = None
    if chat_template:
        chat_template_version = hashlib.sha256(
            chat_template.encode()).hexdigest()[:12]
    return FrontendModelContext(
        tokenizer=spec,
        capabilities=capabilities,
        chat_template_source=chat_template,
        chat_template_version=chat_template_version,
        eos_id=getattr(tokenizer, "eos_token_id", None),
        pad_id=getattr(tokenizer, "pad_token_id", None),
        max_context_length=getattr(llm.args, "max_seq_len", None),
        generation_stop_token_ids=tuple(
            int(t) for t in (generation_stop_ids or ())),
        model_type=getattr(hf_config, "model_type", None) or None,
    )


def load_tokenizer_from_spec(spec: TokenizerSpec):
    """Reload the frontend tokenizer purely from the spec (no live handle).

    Provenance is ENFORCED, not advisory: every manifest entry must exist
    on disk with a matching content hash before the reload (the manifest is
    the content pin — it subsumes a revision string), and the reloaded
    tokenizer's added/special-token configuration must match the recorded
    source. Every failure is a typed config error (fail closed).
    """
    import hashlib
    from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer
    if spec.trust_remote_code:
        raise EngineClientConfigError(
            "trust_remote_code_tokenizer: cannot reload a trust_remote_code "
            "tokenizer from a data-only spec")
    for rel_path, expected_hash in spec.files_manifest:
        candidate = os.path.join(spec.uri, rel_path)
        try:
            with open(candidate, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError as e:
            raise EngineClientConfigError(
                f"tokenizer_manifest: {rel_path!r} unreadable during spec "
                f"reload: {e}")
        if digest != expected_hash:
            raise EngineClientConfigError(
                f"tokenizer_manifest: {rel_path!r} content does not match "
                "the recorded manifest hash; refusing a divergent reload")
    tokenizer = TransformersTokenizer.from_pretrained(spec.uri,
                                                      use_fast=spec.fast)
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        if spec.added_tokens_json is not None:
            try:
                reloaded = _dump_json_or_none(inner.get_added_vocab())
            except Exception:
                reloaded = None
            if reloaded != spec.added_tokens_json:
                raise EngineClientConfigError(
                    "tokenizer_manifest: reloaded added-token vocabulary "
                    "does not match the recorded spec")
        if spec.special_tokens_json is not None:
            reloaded = _dump_json_or_none(
                getattr(inner, "special_tokens_map", None))
            if reloaded != spec.special_tokens_json:
                raise EngineClientConfigError(
                    "tokenizer_manifest: reloaded special-token map does "
                    "not match the recorded spec")
    return tokenizer


async def wrap_sse_with_done(generator, raw_request=None):
    """Wrap a contract SSE generator with first-token timing and ``[DONE]``.

    Mirrors the legacy stream wrappers: the first piece stamps
    ``server_first_token_time`` on the request state, and the stream always
    ends with the ``[DONE]`` sentinel (even when empty).
    """
    try:
        first = await generator.__anext__()
    except StopAsyncIteration:
        yield "data: [DONE]\n\n"
        return
    if raw_request is not None:
        from tensorrt_llm.serve.responses_utils import \
            get_steady_clock_now_in_seconds
        raw_request.state.server_first_token_time = \
            get_steady_clock_now_in_seconds()
    yield first
    async for piece in generator:
        yield piece
    yield "data: [DONE]\n\n"


class EngineClientServing:
    """Serving-side owner of the experimental engine-client path."""

    def __init__(self, llm):
        config = EngineClientConfig(
            backend=llm.args.backend or "pytorch",
            num_postprocess_workers=getattr(llm.args, "num_postprocess_workers",
                                            0) or 0,
            post_processor_hook_set=getattr(llm.args, "post_processor_hook", None)
            is not None,
            speculative_config_set=getattr(llm.args, "speculative_config", None)
            is not None,
            early_first_token_mode=bool(
                getattr(llm.args, "enable_early_first_token_response", False)),
            world_size=getattr(getattr(llm.args, "parallel_config", None),
                               "world_size", 1) or 1,
            tokenizer_trust_remote_code=bool(
                getattr(llm.args, "trust_remote_code", False)),
            flag_enabled=bool(
                getattr(llm.args, "experimental_engine_client", False)),
        )
        if not isinstance(llm._executor, GenerationExecutorProxy):
            raise EngineClientConfigError(
                "transport: the executor is not the IPC GenerationExecutorProxy")
        self.client = LocalProcessEngineClient(llm._executor, config)
        self.model_context = build_model_context(llm, self.client.capabilities())
        # The contract path reads model/tokenizer info only through the
        # context: the tokenizer below is reloaded from the spec, not the
        # server's live handle.
        self.tokenizer = load_tokenizer_from_spec(self.model_context.tokenizer)
        self.counters = {"contract_requests": 0, "capability_rejections": 0}
        self._stream_interval = getattr(llm.args, "stream_interval", 1) or 1
        # Plain config scalar (not a model reach), captured once.
        self._force_return_perf_metrics = bool(
            getattr(llm.args, "return_perf_metrics", False))

    @classmethod
    def create_if_enabled(cls, llm) -> Optional["EngineClientServing"]:
        """Build the serving path when the resolved flag is on; None when off.

        Raises typed errors (fail closed) for malformed flag values and for
        config-gate violations while enablement was requested.
        """
        # Fail CLOSED: a malformed flag value or a config-gate violation
        # while the resolved flag requests enablement is a typed startup
        # error, never a silent fall back to legacy. Only a flag that
        # resolves OFF returns None.
        enabled = resolve_engine_client_flag(
            bool(getattr(llm.args, "experimental_engine_client", False)))
        if not enabled:
            return None
        serving = cls(llm)
        logger.info(
            f"{ENGINE_CLIENT_FLAG_ENV}: experimental engine-client serving "
            "path enabled for eligible streaming requests")
        return serving

    # ------------------------------------------------------------------ #

    def _count_fallback(self, axis: str) -> None:
        key = f"fallback:{axis}"
        self.counters[key] = self.counters.get(key, 0) + 1

    def get_counters(self) -> dict:
        merged = dict(self.counters)
        merged.update(self.client._router.counters)
        merged["active_requests"] = self.client._router.active_request_count()
        return merged

    @staticmethod
    def _normalize_scheduling(scheduling_params):
        if scheduling_params is None:
            return None
        if getattr(scheduling_params, "agent_hierarchy", None) is None:
            return None  # an all-default SchedulingParams carries nothing
        return scheduling_params

    def try_stream(self, *, preprocessed, sampling_params, post_processor,
                   postproc_args, raw_request=None, lora_request=None,
                   prompt_adapter_request=None, disaggregated_params=None,
                   conversation_params=None, scheduling_params=None,
                   cache_salt=None, trace_headers=None,
                   kv_cache_retention_config=None, priority=None,
                   echo=False, streaming=True):
        """Attempt the contract path; returns an async SSE generator or None.

        None means "fall back to legacy" (the eligibility axis was counted).
        Never raises for eligibility reasons.

        ``sampling_params`` MUST be pristine (never live-prepared): the
        endpoints capture a copy before any live ``LLM.preprocess`` call so
        that EVERY contract request is normalized here, through the
        context-only boundary, and never by live model-config reaches.
        """
        if getattr(preprocessed, "prompt_token_ids", None) is None:
            self._count_fallback("no_preprocessed_inputs")
            return None
        if trace_headers is not None and not trace_headers:
            # A tracing-enabled server extracts {} when no trace header is
            # present; only requests actually CARRYING trace headers are
            # ineligible.
            trace_headers = None
        try:
            # Context-only normalization — unconditional, so a request that
            # arrived live-prepared can never smuggle live-derived sampling
            # state past this boundary.
            prepare_sampling_params_from_context(
                sampling_params,
                context=self.model_context,
                tokenizer=self.tokenizer,
                stream_interval=self._stream_interval,
                force_return_perf_metrics=self._force_return_perf_metrics)
        except RequestIneligibleError as e:
            self._count_fallback(e.axis)
            return None
        except ConversionError as e:
            self._count_fallback("conversion_error")
            logger.warning(
                f"engine-client normalization error; falling back: {e}")
            return None
        try:
            engine_request, output_config = convert_request(
                f"chatcmpl-ec-{uuid.uuid4().hex}",
                preprocessed.prompt_token_ids,
                sampling_params,
                streaming=streaming,
                multimodal_params=getattr(preprocessed, "multimodal_params", None),
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                disaggregated_params=disaggregated_params,
                scheduling_params=self._normalize_scheduling(scheduling_params),
                conversation_params=conversation_params,
                trace_headers=trace_headers,
                cache_salt=cache_salt,
                query_token_ids=getattr(preprocessed, "query_token_ids", None),
                encoder_input_token_ids=getattr(preprocessed,
                                                "encoder_input_token_ids", None),
                kv_cache_retention_config=kv_cache_retention_config,
                priority=priority,
                echo=echo,
            )
        except RequestIneligibleError as e:
            self._count_fallback(e.axis)
            return None
        except ConversionError as e:
            self._count_fallback("conversion_error")
            logger.warning(f"engine-client conversion error; falling back: {e}")
            return None

        try:
            request_id = self.client.submit(engine_request,
                                            output_config=output_config)
        except RequestRejectedError as e:
            self.counters["capability_rejections"] += 1
            logger.warning(f"engine-client pre-submit rejection; falling back: {e}")
            return None

        self.counters["contract_requests"] += 1
        assembler = FrontendResponseAssembler(request_id, output_config,
                                              tokenizer=self.tokenizer,
                                              abort_callback=self.client.abort,
                                              stream_interval=self._stream_interval)
        view = ContractStreamView(abs(hash(request_id)) % (1 << 31),
                                  engine_request.prompt_token_ids)
        postproc_args.tokenizer = self.tokenizer
        postproc_args.num_prompt_tokens = len(engine_request.prompt_token_ids)
        return self._sse_generator(request_id, assembler, view, post_processor,
                                   postproc_args, raw_request)

    async def _watch_disconnect(self, raw_request, request_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                if await raw_request.is_disconnected():
                    # Client went away: end background generation via the
                    # same abort control edge as an explicit cancel.
                    try:
                        self.client.abort(request_id)
                    except Exception:
                        pass
                    return
        except asyncio.CancelledError:
            pass

    async def _sse_generator(self, request_id, assembler, view, post_processor,
                             postproc_args, raw_request):
        stream = self.client.stream(request_id)
        watcher = None
        if raw_request is not None:
            watcher = asyncio.create_task(
                self._watch_disconnect(raw_request, request_id))
        try:
            ended = False
            while not ended:
                frame = await stream.__anext__()
                batch = [frame] + stream.pop_ready()
                updates = assembler.process_frames(batch)
                try:
                    ended = view.apply_updates(assembler, updates)
                except _ContractStreamError as e:
                    logger.error(f"engine-client stream {request_id} failed: {e}")
                    raise
                for piece in post_processor(view, postproc_args):
                    yield piece
        finally:
            if watcher is not None:
                watcher.cancel()
            await stream.aclose()
