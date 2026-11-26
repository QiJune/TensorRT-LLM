/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The executor's wire (engine.proto): a frontend sends a SubmitBatch to rank 0, and rank 0 sends
// each frontend one StepOutput per step.
//
// Frontend (proto_wire.py): encode_requests, decode_rows, and frontend_contexts for the
// postprocess workers. Rank 0 (engine_server.py): ingress, publish. Both sides (wire.py):
// set_tensor_codec. Python objects a Value cannot say travel beside a SubmitBatch, in the objects
// list encode_requests fills and ingress reads (Value.object, LogitsProcessor.object); a sealed
// handoff holds none. A StepOutput's tensor bytes travel beside it (TensorRef.buffer).
// encode_requests raises NotCarried (a ValueError) for a request that holds a value engine.proto
// cannot carry, which the frontend refuses; ingress gives a request it cannot take as None with its
// error; publish ends a request whose response it cannot carry with an error row. KIND_* and
// STATUS_* are the values of Kind and Status below.

#include "engineWire.h"
#include "engine/v1/engine.pb.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serialization.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <list>
#include <optional>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <unordered_set>
#include <vector>

namespace tle = tensorrt_llm::executor;
namespace ev1 = trtllm::engine::v1;

namespace
{

using Tokens = google::protobuf::RepeatedField<int32_t>;

// A value engine.proto cannot carry: Python's NotCarried, a ValueError. The frontend refuses a request
// that holds one; in publish it becomes a RowError.
struct NotCarried : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// Throws std::runtime_error(what), Python's RuntimeError, unless ok: input this codec cannot read (a
// malformed message, data it does not read), or a message it cannot write.
void check(bool ok, char const* what)
{
    if (!ok)
    {
        throw std::runtime_error(what);
    }
}

struct MemBuf : std::streambuf
{
    MemBuf(char const* p, size_t n)
    {
        auto* q = const_cast<char*>(p);
        setg(q, q, q + n);
    }
};

// Owns a new reference the C API returned; raises the pending Python error when it is null.
nb::object own(PyObject* o)
{
    if (o == nullptr)
    {
        throw nb::python_error();
    }
    return nb::steal(o);
}

nb::object getattrOrThrow(nb::handle h, nb::handle name)
{
    return own(PyObject_GetAttr(h.ptr(), name.ptr()));
}

// An attribute some responses lack (getattr(obj, name, default) in Python).
nb::object getattrOr(nb::handle h, nb::handle name, nb::handle fallback)
{
    PyObject* v = PyObject_GetAttr(h.ptr(), name.ptr());
    if (v == nullptr)
    {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
        {
            throw nb::python_error();
        }
        PyErr_Clear();
        return nb::borrow(fallback);
    }
    return nb::steal(v);
}

void setAttr(nb::handle o, nb::handle name, nb::handle value)
{
    if (PyObject_SetAttr(o.ptr(), name.ptr(), value.ptr()) != 0)
    {
        throw nb::python_error();
    }
}

bool isTrue(nb::handle h)
{
    int const truth = PyObject_IsTrue(h.ptr());
    if (truth < 0)
    {
        throw nb::python_error();
    }
    return truth == 1;
}

// An int that fits int32_t, or nullopt; sets no Python error.
std::optional<int32_t> int32Of(PyObject* o)
{
    if (!PyLong_Check(o))
    {
        return std::nullopt;
    }
    int overflow = 0;
    long long const v = PyLong_AsLongLongAndOverflow(o, &overflow);
    if (overflow != 0 || v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
    {
        return std::nullopt;
    }
    return static_cast<int32_t>(v);
}

// A class the caller passed, for PyObject_TypeCheck and tp_new.
PyTypeObject* asType(nb::handle h)
{
    if (!PyType_Check(h.ptr()))
    {
        throw nb::type_error("expected a class");
    }
    return reinterpret_cast<PyTypeObject*>(h.ptr());
}

// ----------------------------------------------------------------------------------------------
// Frontend: requests to a SubmitBatch

void putTokens(Tokens* out, nb::handle ids)
{
    if (PyList_Check(ids.ptr()))
    {
        Py_ssize_t const n = PyList_GET_SIZE(ids.ptr());
        out->Reserve(out->size() + static_cast<int>(n));
        for (Py_ssize_t i = 0; i < n; ++i)
        {
            int overflow = 0;
            long long const v = PyLong_AsLongLongAndOverflow(PyList_GET_ITEM(ids.ptr(), i), &overflow);
            if (v == -1 && PyErr_Occurred())
            {
                throw nb::python_error();
            }
            if (overflow != 0 || v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
            {
                throw NotCarried("a token id does not fit in int32");
            }
            out->AddAlreadyReserved(static_cast<int32_t>(v));
        }
        return;
    }
    // GenerationRequest's int32 wire buffer
    Py_buffer view;
    if (PyObject_GetBuffer(ids.ptr(), &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0)
    {
        throw nb::python_error();
    }
    bool const ok = view.itemsize == 4 && view.format != nullptr
        && (std::string(view.format) == "i" || std::string(view.format) == "<i" || std::string(view.format) == "=i");
    if (ok)
    {
        auto const* p = static_cast<int32_t const*>(view.buf);
        out->Add(p, p + view.len / 4);
    }
    PyBuffer_Release(&view);
    if (!ok)
    {
        throw nb::type_error("token ids must be a list of int or an int32 array");
    }
}

void putWords(nb::handle words, google::protobuf::RepeatedPtrField<ev1::TokenSeq>* out)
{
    for (nb::handle w : words)
    {
        putTokens(out->Add()->mutable_ids(), w);
    }
}

void putSampling(tle::SamplingConfig const& c, ev1::Sampling* s, ev1::Stopping* stop)
{
    s->set_beam_width(static_cast<uint32_t>(c.getBeamWidth()));
    if (auto v = c.getNumReturnSequences())
    {
        s->set_num_sequences(static_cast<uint32_t>(*v));
    }
    if (auto v = c.getBeamWidthArray())
    {
        for (auto w : *v)
        {
            s->add_beam_width_array(static_cast<uint32_t>(w));
        }
    }
    if (auto v = c.getTemperature())
    {
        s->set_temperature(*v);
    }
    if (auto v = c.getTopK())
    {
        s->set_top_k(static_cast<uint32_t>(*v));
    }
    if (auto v = c.getTopP())
    {
        s->set_top_p(*v);
    }
    if (auto v = c.getTopPMin())
    {
        s->set_top_p_min(*v);
    }
    if (auto v = c.getTopPResetIds())
    {
        s->set_top_p_reset_ids(*v);
    }
    if (auto v = c.getTopPDecay())
    {
        s->set_top_p_decay(*v);
    }
    if (auto v = c.getMinP())
    {
        s->set_min_p(*v);
    }
    if (auto v = c.getSeed())
    {
        s->set_seed(*v);
    }
    if (auto v = c.getRepetitionPenalty())
    {
        s->set_repetition_penalty(*v);
    }
    if (auto v = c.getPresencePenalty())
    {
        s->set_presence_penalty(*v);
    }
    if (auto v = c.getFrequencyPenalty())
    {
        s->set_frequency_penalty(*v);
    }
    if (auto v = c.getPromptIgnoreLength())
    {
        s->set_prompt_ignore_length(static_cast<uint32_t>(*v));
    }
    if (auto v = c.getNoRepeatNgramSize())
    {
        s->set_no_repeat_ngram_size(static_cast<uint32_t>(*v));
    }
    if (auto v = c.getLengthPenalty())
    {
        s->set_length_penalty(*v);
    }
    if (auto v = c.getEarlyStopping())
    {
        s->set_early_stopping(*v);
    }
    if (auto v = c.getBeamSearchDiversityRate())
    {
        s->set_beam_search_diversity_rate(*v);
    }
    if (auto v = c.getMinTokens())
    {
        stop->set_min_new_tokens(static_cast<uint32_t>(*v));
    }
}

// A key of the cold-field dict, or a null handle.
nb::handle key(nb::handle dict, char const* name)
{
    return PyDict_GetItemString(dict.ptr(), name);
}

void putGuided(tle::GuidedDecodingParams const& g, ev1::GuidedDecoding* out)
{
    using Type = tle::GuidedDecodingParams::GuideType;
    std::string guide = g.getGuide().value_or("");
    switch (g.getGuideType())
    {
    case Type::kJSON: out->set_json_object(true); break;
    case Type::kJSON_SCHEMA: out->set_json_schema(std::move(guide)); break;
    case Type::kREGEX: out->set_regex(std::move(guide)); break;
    case Type::kEBNF_GRAMMAR: out->set_ebnf_grammar(std::move(guide)); break;
    case Type::kSTRUCTURAL_TAG: out->set_structural_tag(std::move(guide)); break;
    }
}

void putRetention(tle::KvCacheRetentionConfig const& c, ev1::KvRetention* out)
{
    for (auto const& r : c.getTokenRangeRetentionConfigs())
    {
        auto* range = out->add_ranges();
        range->set_token_start(static_cast<uint32_t>(r.tokenStart));
        if (r.tokenEnd)
        {
            range->set_token_end(static_cast<uint32_t>(*r.tokenEnd));
        }
        range->set_priority(r.priority);
        if (r.durationMs)
        {
            range->set_duration_ms(static_cast<uint32_t>(r.durationMs->count()));
        }
    }
    out->set_decode_priority(c.getDecodeRetentionPriority());
    if (auto d = c.getDecodeDurationMs())
    {
        out->set_decode_duration_ms(static_cast<uint32_t>(d->count()));
    }
    out->set_transfer_mode(static_cast<ev1::KvTransferMode>(c.getTransferMode()));
    out->set_directory(c.getDirectory());
}

// TokenLogprobs columns from Python's (ids, logprobs, ranks, top-k ends, top-k ids, top-k logprobs).
void putColumns(nb::handle v, ev1::TokenLogprobs* lp)
{
    for (nb::handle x : v[0])
    {
        lp->add_ids(nb::cast<int32_t>(x));
    }
    for (nb::handle x : v[1])
    {
        lp->add_logprobs(static_cast<float>(nb::cast<double>(x)));
    }
    for (nb::handle x : v[2])
    {
        lp->add_ranks(nb::cast<uint32_t>(x));
    }
    lp->add_topk_offsets(0);
    for (nb::handle x : v[3])
    {
        lp->add_topk_offsets(nb::cast<uint32_t>(x));
    }
    for (nb::handle x : v[4])
    {
        lp->add_topk_ids(nb::cast<int32_t>(x));
    }
    for (nb::handle x : v[5])
    {
        lp->add_topk_logprobs(static_cast<float>(nb::cast<double>(x)));
    }
}

template <class Field>
nb::list listOf(Field const& f)
{
    nb::list l;
    for (auto v : f)
    {
        l.append(v);
    }
    return l;
}

// A TensorRef from Python's (dtype, shape, bytes, device); device None: host memory. Its data goes
// inline, or, given buffers, beside the message: appended to buffers, and TensorRef.buffer numbers it
// where the message is put together (appendRow).
void putTensorRef(nb::handle t, ev1::TensorRef* ref, std::vector<nb::object>* buffers = nullptr)
{
    ref->set_dtype(static_cast<ev1::DType>(nb::cast<int>(t[0])));
    for (nb::handle s : t[1])
    {
        ref->add_shape(nb::cast<uint64_t>(s));
    }
    nb::object const data = t[2];
    if (!PyBytes_Check(data.ptr()))
    {
        throw nb::type_error("a tensor's data must be bytes");
    }
    if (buffers != nullptr)
    {
        buffers->push_back(data);
    }
    else
    {
        ref->set_inline_data(PyBytes_AS_STRING(data.ptr()), static_cast<size_t>(PyBytes_GET_SIZE(data.ptr())));
    }
    nb::object const device = t[3];
    if (!device.is_none())
    {
        ref->set_device(nb::cast<int32_t>(device));
    }
}

// The inverse of putTensorRef. Reads inline data, and given buffers (the ones beside the message) a
// TensorRef.buffer; a TensorRef in shared memory or CUDA IPC raises.
nb::tuple tensorRefOf(ev1::TensorRef const& ref, nb::list const* buffers = nullptr)
{
    nb::object data;
    if (ref.data_case() == ev1::TensorRef::kBuffer)
    {
        check(buffers != nullptr && ref.buffer() < buffers->size(),
            "a TensorRef names a buffer the message does not carry");
        data = (*buffers)[ref.buffer()];
    }
    else
    {
        check(ref.data_case() == ev1::TensorRef::kInlineData,
            "a TensorRef whose data is neither inline nor a buffer: this codec reads no other");
        data = nb::bytes(ref.inline_data().data(), ref.inline_data().size());
    }
    return nb::make_tuple(static_cast<int>(ref.dtype()), listOf(ref.shape()), data,
        ref.has_device() ? nb::cast(ref.device()) : nb::none());
}

// What wire.py registers (set_tensor_codec) for tensors in a Value: torch.Tensor, a function from a
// tensor to (dtype, shape, bytes, device) or None, and one back.
struct TensorCodec
{
    nb::object type;
    nb::object toRef;
    nb::object fromRef;
};

TensorCodec& tensorCodec()
{
    static TensorCodec* codec = new TensorCodec();
    return *codec;
}

void setTensorCodec(nb::handle tensorType, nb::handle toRef, nb::handle fromRef)
{
    auto& c = tensorCodec();
    c.type = nb::borrow(tensorType);
    c.toRef = nb::borrow(toRef);
    c.fromRef = nb::borrow(fromRef);
}

// An object put in objects, the list beside the message (Value.object, LogitsProcessor.object); its
// index there. objects nullptr: the value cannot refer to an object, and NotCarried.
uint32_t putObject(nb::handle v, nb::list* objects)
{
    if (objects == nullptr)
    {
        throw NotCarried("a value engine.proto's Value has no kind for");
    }
    objects->append(v);
    return static_cast<uint32_t>(objects->size() - 1);
}

// The object at index in objects, the list beside the message; nullptr: no objects.
nb::object objectAt(uint32_t index, nb::list const* objects)
{
    check(objects != nullptr && index < objects->size(), "an object index the message does not carry");
    return (*objects)[index];
}

// A Python value as a Value: None, and bool, int, float, str, bytes, list, tuple and dict of
// exactly those types, and tensors; any other object goes in objects (Value.object), or raises
// NotCarried when objects is nullptr.
void putValue(nb::handle v, ev1::Value* out, nb::list* objects)
{
    PyObject* o = v.ptr();
    if (o == Py_None)
    {
        out->set_none(true);
    }
    else if (PyBool_Check(o))
    {
        out->set_flag(o == Py_True);
    }
    else if (PyLong_CheckExact(o))
    {
        int overflow = 0;
        long long const i = PyLong_AsLongLongAndOverflow(o, &overflow);
        if (overflow)
        {
            out->set_object(putObject(v, objects));
        }
        else
        {
            out->set_integer(i);
        }
    }
    else if (PyFloat_CheckExact(o))
    {
        out->set_number(PyFloat_AS_DOUBLE(o));
    }
    else if (PyUnicode_CheckExact(o))
    {
        Py_ssize_t size = 0;
        char const* text = PyUnicode_AsUTF8AndSize(o, &size);
        if (text == nullptr)
        {
            throw nb::python_error();
        }
        out->set_text(text, static_cast<size_t>(size));
    }
    else if (PyBytes_CheckExact(o))
    {
        out->set_raw(PyBytes_AS_STRING(o), static_cast<size_t>(PyBytes_GET_SIZE(o)));
    }
    else if (PyList_CheckExact(o) || PyTuple_CheckExact(o))
    {
        auto* items = PyList_CheckExact(o) ? out->mutable_list() : out->mutable_tuple();
        for (nb::handle x : v)
        {
            putValue(x, items->add_items(), objects);
        }
    }
    else if (PyDict_CheckExact(o))
    {
        auto* fields = out->mutable_map();
        PyObject* key;
        PyObject* value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(o, &pos, &key, &value))
        {
            auto* f = fields->add_fields();
            putValue(key, f->mutable_key(), objects);
            putValue(value, f->mutable_value(), objects);
        }
    }
    else
    {
        auto& c = tensorCodec();
        // a tensor of a dtype engine.proto has no name for goes beside the message too
        nb::object ref = c.type.is_valid() && PyObject_TypeCheck(o, asType(c.type)) ? c.toRef(v) : nb::none();
        if (ref.is_none())
        {
            out->set_object(putObject(v, objects));
        }
        else
        {
            putTensorRef(ref, out->mutable_tensor());
        }
    }
}

// The inverse of putValue. An unset Value is None.
nb::object valueOf(ev1::Value const& v, nb::list const* objects)
{
    switch (v.kind_case())
    {
    case ev1::Value::kFlag: return nb::bool_(v.flag());
    case ev1::Value::kInteger: return nb::int_(v.integer());
    case ev1::Value::kNumber: return nb::float_(v.number());
    case ev1::Value::kText: return nb::str(v.text().data(), v.text().size());
    case ev1::Value::kRaw: return nb::bytes(v.raw().data(), v.raw().size());
    case ev1::Value::kTensor:
    {
        auto const& fromRef = tensorCodec().fromRef;
        check(fromRef.is_valid(), "a tensor in a Value before set_tensor_codec");
        return own(PyObject_CallObject(fromRef.ptr(), tensorRefOf(v.tensor()).ptr()));
    }
    case ev1::Value::kList:
    {
        nb::list l;
        for (auto const& x : v.list().items())
        {
            l.append(valueOf(x, objects));
        }
        return l;
    }
    case ev1::Value::kTuple:
    {
        nb::object t = own(PyTuple_New(v.tuple().items_size()));
        for (int i = 0; i < v.tuple().items_size(); ++i)
        {
            PyTuple_SET_ITEM(t.ptr(), i, valueOf(v.tuple().items(i), objects).release().ptr());
        }
        return t;
    }
    case ev1::Value::kMap:
    {
        nb::dict d;
        for (auto const& f : v.map().fields())
        {
            d[valueOf(f.key(), objects)] = valueOf(f.value(), objects);
        }
        return d;
    }
    case ev1::Value::kObject: return objectAt(v.object(), objects);
    default: return nb::none();
    }
}

// A handoff sealed for serve to pass on: a Value, serialized. It holds no objects, since serve passes
// it on apart from the message it came with; a value that would need one raises NotCarried.
std::string seal(nb::handle v)
{
    ev1::Value value;
    putValue(v, &value, nullptr);
    std::string s;
    check(value.SerializeToString(&s), "cannot serialize a handoff");
    return s;
}

nb::object unseal(std::string const& s)
{
    ev1::Value value;
    check(value.ParseFromString(s), "malformed handoff");
    return valueOf(value, nullptr);
}

// A string-keyed field of a map Value, or null.
ev1::Value const* field(ev1::Value const& map, std::string const& name)
{
    for (auto const& f : map.map().fields())
    {
        if (f.key().kind_case() == ev1::Value::kText && f.key().text() == name)
        {
            return &f.value();
        }
    }
    return nullptr;
}

ev1::Value* addField(ev1::Value& map, char const* name)
{
    auto* f = map.mutable_map()->add_fields();
    f->mutable_key()->set_text(name);
    return f->mutable_value();
}

// The fields few requests set, from the dict proto_wire.encode builds (None when the request
// sets none). objects: the list beside the message.
void putCold(nb::handle cold, ev1::Request* q, nb::list* objects)
{
    if (cold.is_none())
    {
        return;
    }
    if (nb::handle v = key(cold, "guided"))
    {
        putGuided(nb::cast<tle::GuidedDecodingParams const&>(v), q->mutable_guided());
    }
    if (nb::handle v = key(cold, "retention"))
    {
        putRetention(nb::cast<tle::KvCacheRetentionConfig const&>(v), q->mutable_kv()->mutable_retention());
    }
    if (nb::handle v = key(cold, "encoder_input"))
    {
        putTokens(q->mutable_encoder_input(), v);
    }
    if (nb::handle v = key(cold, "embedding_bias")) // (dtype, shape, bytes, device)
    {
        putTensorRef(v, q->mutable_sampling()->mutable_embedding_bias());
    }
    if (nb::handle v = key(cold, "token_bias")) // ((id, bias), ...)
    {
        for (nb::handle pair : v)
        {
            auto* b = q->mutable_sampling()->add_token_bias();
            b->set_id(nb::cast<int32_t>(pair[0]));
            b->set_bias(nb::cast<double>(pair[1]));
        }
    }
    if (nb::handle v = key(cold, "scheduling")) // (attention_dp_rank, relax, hierarchy, affinity id)
    {
        auto* s = q->mutable_scheduling();
        if (!v[0].is_none())
        {
            s->set_attention_dp_rank(nb::cast<uint32_t>(v[0]));
        }
        s->set_attention_dp_strict(!nb::cast<bool>(v[1]));
        if (!v[2].is_none())
        {
            for (nb::handle hop : v[2])
            {
                auto* h = s->add_agent_hierarchy();
                h->set_name(nb::cast<std::string>(hop[0]));
                h->set_index(nb::cast<int32_t>(hop[1]));
            }
        }
        if (!v[3].is_none())
        {
            s->set_subagent_affinity_id(nb::cast<std::string>(v[3]));
        }
    }
    if (nb::handle v = key(cold, "conversation_id"))
    {
        q->mutable_kv()->set_conversation_id(nb::cast<std::string>(v));
    }
    if (nb::handle v = key(cold, "kv_hints"))
    {
        nb::bytes data = nb::borrow<nb::bytes>(v);
        q->mutable_kv()->set_kv_hints(data.c_str(), data.size());
    }
    if (nb::handle v = key(cold, "lora")) // (id, path, ckpt source)
    {
        auto* l = q->mutable_lora();
        l->set_id(nb::cast<uint64_t>(v[0]));
        l->set_path(nb::cast<std::string>(v[1]));
        l->set_format(nb::cast<std::string>(v[2]) == "nemo" ? ev1::LORA_FORMAT_NEMO : ev1::LORA_FORMAT_HF);
    }
    if (nb::handle v = key(cold, "watermark"))
    {
        q->set_watermark(nb::cast<bool>(v));
    }
    auto* output = q->mutable_output();
    if (key(cold, "perf_metrics"))
    {
        output->set_perf_metrics(true);
    }
    if (key(cold, "logprobs_values_only"))
    {
        output->set_logprobs_values_only(true);
    }
    if (nb::handle v = key(cold, "prompt_logprobs"))
    {
        output->set_prompt_logprobs(nb::cast<uint32_t>(v));
    }
    if (key(cold, "prompt_logprobs_values_only"))
    {
        output->set_prompt_logprobs_values_only(true);
    }
    if (key(cold, "context_logits"))
    {
        output->set_context_logits(true);
    }
    if (key(cold, "generation_logits"))
    {
        output->set_generation_logits(true);
    }
    if (nb::handle v = key(cold, "additional_outputs"))
    {
        for (nb::handle name : v)
        {
            output->add_additional_outputs(nb::cast<std::string>(name));
        }
    }
    if (key(cold, "routed_experts"))
    {
        output->set_routed_experts(true);
    }
    if (key(cold, "encoder_output"))
    {
        output->set_encoder_output(true);
    }
    if (nb::handle v = key(cold, "frontend_context"))
    {
        nb::bytes data = nb::borrow<nb::bytes>(v);
        q->set_frontend_context(data.c_str(), data.size());
    }
    if (nb::handle v = key(cold, "prompt_trimmed_prefix"))
    {
        q->set_prompt_trimmed_prefix(nb::cast<uint32_t>(v));
    }
    // (phase, schedule, disagg_request_id, orig_elapsed_s, kv_import, prefill_usage, encoder_import)
    if (nb::handle v = key(cold, "disagg"))
    {
        auto* d = q->mutable_disagg();
        d->set_phase(static_cast<ev1::DisaggPhase>(nb::cast<int>(v[0])));
        if (!v[1].is_none())
        {
            d->set_schedule(static_cast<ev1::DisaggScheduleStyle>(nb::cast<int>(v[1])));
        }
        if (!v[2].is_none())
        {
            d->set_disagg_request_id(nb::cast<uint64_t>(v[2]));
        }
        if (!v[3].is_none())
        {
            d->set_orig_elapsed_s(nb::cast<double>(v[3]));
        }
        if (!v[4].is_none())
        {
            d->set_kv_import(seal(v[4]));
        }
        if (!v[5].is_none())
        {
            auto* usage = d->mutable_prefill_usage();
            usage->set_prompt_tokens(nb::cast<uint32_t>(v[5][0]));
            usage->set_cached_tokens(nb::cast<uint32_t>(v[5][1]));
        }
        if (!v[6].is_none())
        {
            d->set_encoder_import(seal(v[6]));
        }
    }
    if (nb::handle v = key(cold, "multimodal")) // (items, model_inputs)
    {
        auto* m = q->mutable_multimodal();
        for (nb::handle item : v[0]) // (hash, uuid, position, length, run positions, run lengths)
        {
            auto* it = m->add_items();
            putTokens(it->mutable_hash(), item[0]);
            if (!item[1].is_none())
            {
                it->set_uuid(nb::cast<std::string>(item[1]));
            }
            it->set_position(nb::cast<uint32_t>(item[2]));
            it->set_length(nb::cast<uint32_t>(item[3]));
            for (nb::handle x : item[4])
            {
                it->add_run_positions(nb::cast<uint32_t>(x));
            }
            for (nb::handle x : item[5])
            {
                it->add_run_lengths(nb::cast<uint32_t>(x));
            }
        }
        putValue(v[1], m->mutable_model_inputs(), objects);
    }
    if (nb::handle v = key(cold, "thinking_budget")) // (budget, start ids, end ids)
    {
        auto* b = q->mutable_thinking_budget();
        b->set_budget(nb::cast<uint32_t>(v[0]));
        putTokens(b->mutable_start_ids(), v[1]);
        putTokens(b->mutable_end_ids(), v[2]);
    }
    if (nb::handle v = key(cold, "logits_processors")) // Python callables, beside the message
    {
        for (nb::handle p : v)
        {
            q->add_logits_processors()->set_object(putObject(p, objects));
        }
    }
}

// encode_requests([(handle, prompt, stream, sampling_config, max_tokens, end_id, ignore_eos,
//                   stop_words, bad_words, logprobs, logprobs_mode, priority, arrival_time,
//                   cache_salt, cold)], objects) -> SubmitBatch bytes
// objects: a list the Python objects the messages refer to are appended to.
nb::bytes encodeRequests(nb::list items, nb::list objects)
{
    ev1::SubmitBatch batch;
    for (nb::handle item : items)
    {
        auto t = nb::borrow<nb::tuple>(item);
        auto* q = batch.add_requests();
        q->set_handle(nb::cast<uint64_t>(t[0]));
        putTokens(q->mutable_prompt(), t[1]);
        q->set_stream(nb::cast<bool>(t[2]));
        auto* stop = q->mutable_stopping();
        putSampling(nb::cast<tle::SamplingConfig const&>(t[3]), q->mutable_sampling(), stop);
        if (!t[4].is_none())
        {
            stop->set_max_new_tokens(nb::cast<int32_t>(t[4]));
        }
        if (!t[5].is_none())
        {
            stop->set_end_id(nb::cast<int32_t>(t[5]));
        }
        stop->set_ignore_eos(nb::cast<bool>(t[6]));
        putWords(t[7], stop->mutable_stop());
        putWords(t[8], stop->mutable_bad());
        auto* out = q->mutable_output();
        if (!t[9].is_none())
        {
            out->set_logprobs(nb::cast<uint32_t>(t[9]));
        }
        out->set_logprobs_mode(static_cast<ev1::LogprobMode>(nb::cast<int>(t[10])));
        // the LLM API shows the counters on every response; a non-streaming request has one row per
        // sequence, at its end, where FinalInfo has them
        out->set_decoding_stats(q->stream());
        auto* sched = q->mutable_scheduling();
        sched->set_priority(nb::cast<float>(t[11]));
        if (!t[12].is_none())
        {
            // seconds on the steady clock, as a float or as the timedelta bindings.steady_clock_now() gives
            double const arrival = nb::cast<std::chrono::duration<double>>(t[12]).count();
            sched->set_arrival_time_ns(static_cast<uint64_t>(std::llround(arrival * 1e9)));
        }
        if (!t[13].is_none())
        {
            q->mutable_kv()->set_cache_salt(nb::cast<std::string>(t[13]));
        }
        putCold(t[14], q, &objects);
    }
    std::string s;
    check(batch.SerializeToString(&s), "cannot serialize SubmitBatch");
    return nb::bytes(s.data(), s.size());
}

// ----------------------------------------------------------------------------------------------
// Rank 0: a SubmitBatch to executor requests

template <class T, class P>
std::optional<T> opt(bool has, P v)
{
    return has ? std::optional<T>(static_cast<T>(v)) : std::nullopt;
}

tle::SamplingConfig toSampling(ev1::Sampling const& s, ev1::Stopping const& stop)
{
    std::optional<std::vector<tle::SizeType32>> widths;
    if (s.beam_width_array_size())
    {
        widths.emplace(s.beam_width_array().begin(), s.beam_width_array().end());
    }
    return tle::SamplingConfig(s.beam_width() ? static_cast<tle::SizeType32>(s.beam_width()) : 1,
        opt<tle::SizeType32>(s.has_top_k(), s.top_k()), opt<tle::FloatType>(s.has_top_p(), s.top_p()),
        opt<tle::FloatType>(s.has_top_p_min(), s.top_p_min()),
        opt<tle::TokenIdType>(s.has_top_p_reset_ids(), s.top_p_reset_ids()),
        opt<tle::FloatType>(s.has_top_p_decay(), s.top_p_decay()), opt<tle::RandomSeedType>(s.has_seed(), s.seed()),
        opt<tle::FloatType>(s.has_temperature(), s.temperature()),
        opt<tle::SizeType32>(stop.min_new_tokens() > 0, stop.min_new_tokens()),
        opt<tle::FloatType>(s.has_beam_search_diversity_rate(), s.beam_search_diversity_rate()),
        opt<tle::FloatType>(s.has_repetition_penalty(), s.repetition_penalty()),
        opt<tle::FloatType>(s.has_presence_penalty(), s.presence_penalty()),
        opt<tle::FloatType>(s.has_frequency_penalty(), s.frequency_penalty()),
        opt<tle::SizeType32>(s.has_prompt_ignore_length(), s.prompt_ignore_length()),
        opt<tle::FloatType>(s.has_length_penalty(), s.length_penalty()),
        opt<tle::SizeType32>(s.has_early_stopping(), s.early_stopping()),
        opt<tle::SizeType32>(s.has_no_repeat_ngram_size(), s.no_repeat_ngram_size()),
        opt<tle::SizeType32>(s.num_sequences() > 0, s.num_sequences()), opt<tle::FloatType>(s.has_min_p(), s.min_p()),
        std::move(widths));
}

std::list<tle::VecTokens> wordLists(google::protobuf::RepeatedPtrField<ev1::TokenSeq> const& words)
{
    std::list<tle::VecTokens> out;
    for (auto const& w : words)
    {
        out.emplace_back(w.ids().begin(), w.ids().end());
    }
    return out;
}

// The max_tokens a request runs with, from max_seq_len and its prompt. The statuses name its
// warnings and errors, which engine_server.submit words. An error refuses the request.
enum class Status : int
{
    kOk = 0,
    kWarnNoMaxSeqLen = 1,
    kWarnTruncated = 2,
    kErrNeedMaxTokens = 10,
    kErrContextExceeded = 11,
    kErrMaxTokens = 12,
};

bool isError(Status status)
{
    return status >= Status::kErrNeedMaxTokens;
}

struct MaxTokens
{
    Status status = Status::kOk;
    int64_t value = 0; // the max_tokens the request runs with
    // For the warnings and errors that quote them: what the request asked for (its max_tokens, or
    // with kErrContextExceeded its prompt length), and the bound it crossed (the deduced max_tokens,
    // max_seq_len * cp_size with kErrContextExceeded, 0 with kErrMaxTokens).
    int64_t asked = 0;
    int64_t limit = 0;
};

MaxTokens deduceMaxTokens(int64_t promptLen, std::optional<int64_t> maxTokens, std::optional<int64_t> maxSeqLen,
    int64_t cpSize, bool isGenerationModel)
{
    if (!maxSeqLen)
    {
        return maxTokens ? MaxTokens{.status = Status::kWarnNoMaxSeqLen, .value = *maxTokens}
                         : MaxTokens{.status = Status::kErrNeedMaxTokens};
    }
    MaxTokens const contextExceeded{
        .status = Status::kErrContextExceeded, .asked = promptLen, .limit = *maxSeqLen * cpSize};
    if (!isGenerationModel)
    {
        return promptLen > *maxSeqLen * cpSize ? contextExceeded : MaxTokens{.value = 1};
    }
    int64_t const deduced = *maxSeqLen - promptLen / cpSize;
    if (deduced <= 0)
    {
        return contextExceeded;
    }
    if (!maxTokens)
    {
        return {.value = deduced};
    }
    if (*maxTokens > deduced)
    {
        return {.status = Status::kWarnTruncated, .value = deduced, .asked = *maxTokens, .limit = deduced};
    }
    if (*maxTokens <= 0)
    {
        return {.status = Status::kErrMaxTokens, .asked = *maxTokens};
    }
    return {.value = *maxTokens};
}

std::optional<tle::GuidedDecodingParams> toGuided(ev1::Request const& q)
{
    if (!q.has_guided())
    {
        return std::nullopt;
    }
    using Type = tle::GuidedDecodingParams::GuideType;
    auto const& g = q.guided();
    switch (g.guide_case())
    {
    case ev1::GuidedDecoding::kJsonObject: return tle::GuidedDecodingParams(Type::kJSON);
    case ev1::GuidedDecoding::kJsonSchema: return tle::GuidedDecodingParams(Type::kJSON_SCHEMA, g.json_schema());
    case ev1::GuidedDecoding::kRegex: return tle::GuidedDecodingParams(Type::kREGEX, g.regex());
    case ev1::GuidedDecoding::kEbnfGrammar: return tle::GuidedDecodingParams(Type::kEBNF_GRAMMAR, g.ebnf_grammar());
    case ev1::GuidedDecoding::kStructuralTag:
        return tle::GuidedDecodingParams(Type::kSTRUCTURAL_TAG, g.structural_tag());
    default: return std::nullopt;
    }
}

std::optional<tle::KvCacheRetentionConfig> toRetention(ev1::KvOptions const& kv)
{
    if (!kv.has_retention())
    {
        return std::nullopt;
    }
    auto const& r = kv.retention();
    using Ms = std::chrono::milliseconds;
    std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig> ranges;
    for (auto const& range : r.ranges())
    {
        ranges.emplace_back(static_cast<tle::SizeType32>(range.token_start()),
            opt<tle::SizeType32>(range.has_token_end(), range.token_end()), range.priority(),
            range.has_duration_ms() ? std::optional<Ms>(Ms(range.duration_ms())) : std::nullopt);
    }
    return tle::KvCacheRetentionConfig(ranges,
        r.has_decode_priority() ? r.decode_priority() : tle::KvCacheRetentionConfig::kDefaultRetentionPriority,
        r.has_decode_duration_ms() ? std::optional<Ms>(Ms(r.decode_duration_ms())) : std::nullopt,
        static_cast<tle::KvCacheTransferMode>(r.transfer_mode()), r.directory());
}

// The tokens a Result carries are always the new ones; a frontend that wants the prompt in front
// adds it (exclude_input_from_output=False).
tle::OutputConfig toOutput(ev1::OutputOptions const& o)
{
    std::optional<std::vector<tle::AdditionalModelOutput>> additional;
    if (o.additional_outputs_size())
    {
        additional.emplace();
        for (auto const& name : o.additional_outputs())
        {
            additional->emplace_back(name, /*gatherContext=*/false);
        }
    }
    // prompt logprobs are computed from the context logits
    return tle::OutputConfig(/*returnLogProbs=*/o.has_logprobs(),
        /*returnContextLogits=*/o.context_logits() || o.has_prompt_logprobs(),
        /*returnGenerationLogits=*/o.generation_logits(), /*excludeInputFromOutput=*/true,
        /*returnEncoderOutput=*/o.encoder_output(), /*returnPerfMetrics=*/o.perf_metrics(), std::move(additional));
}

tle::RequestType requestType(ev1::Request const& q)
{
    if (!q.has_disagg())
    {
        return tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION;
    }
    switch (q.disagg().phase())
    {
    case ev1::DISAGG_PHASE_CONTEXT_ONLY: return tle::RequestType::REQUEST_TYPE_CONTEXT_ONLY;
    case ev1::DISAGG_PHASE_GENERATION_ONLY: return tle::RequestType::REQUEST_TYPE_GENERATION_ONLY;
    default: return tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION;
    }
}

// The token ids in a list Value; nullopt when the field is missing or not a list.
std::optional<tle::VecTokens> tokensOf(ev1::Value const* v)
{
    if (v == nullptr || v->kind_case() != ev1::Value::kList)
    {
        return std::nullopt;
    }
    tle::VecTokens tokens;
    for (auto const& x : v->list().items())
    {
        tokens.push_back(static_cast<tle::TokenIdType>(x.integer()));
    }
    return tokens;
}

// What DisaggregatedParams.get_context_phase_params gives a decode part's request, from the
// handoff the prefill part sealed (a map of DisaggregatedParams' fields).
std::optional<tle::ContextPhaseParams> toContextPhase(ev1::Request const& q)
{
    if (!q.has_disagg() || q.disagg().phase() != ev1::DISAGG_PHASE_GENERATION_ONLY)
    {
        return std::nullopt;
    }
    auto const& d = q.disagg();
    ev1::Value h;
    check(h.ParseFromString(d.kv_import()), "malformed kv_import");
    auto integer = [&](char const* name) -> std::optional<int64_t>
    {
        auto const* f = field(h, name);
        return f != nullptr && f->kind_case() == ev1::Value::kInteger ? std::optional<int64_t>(f->integer())
                                                                      : std::nullopt;
    };
    auto const reqId = static_cast<tle::ContextPhaseParams::RequestIdType>(
        d.has_disagg_request_id() ? d.disagg_request_id() : integer("ctx_request_id").value_or(0));
    tle::VecTokens firstGen = tokensOf(field(h, "first_gen_tokens")).value_or(tle::VecTokens{});
    std::optional<tle::VecTokens> draft = tokensOf(field(h, "draft_tokens"));
    std::optional<tle::SizeType32> dpRank;
    if (auto rank = integer("ctx_dp_rank"))
    {
        dpRank = static_cast<tle::SizeType32>(*rank);
    }
    std::optional<std::string> endpoint;
    if (auto const* f = field(h, "ctx_info_endpoint"); f != nullptr && f->kind_case() == ev1::Value::kText)
    {
        endpoint = f->text();
    }
    if (auto const* f = field(h, "opaque_state"); f != nullptr && f->kind_case() == ev1::Value::kRaw)
    {
        return tle::ContextPhaseParams(std::move(firstGen), reqId, std::vector<char>(f->raw().begin(), f->raw().end()),
            std::move(draft), dpRank, std::move(endpoint));
    }
    return tle::ContextPhaseParams(std::move(firstGen), reqId, std::move(draft), dpRank, std::move(endpoint));
}

// RequestEntry.kind: how a request's sequences come back.
enum class Kind : int
{
    kSingle = 0,   // one sequence
    kChildren = 1, // n > 1 sampled sequences: one response per sequence, each with its sequence index
    kBeams = 2,    // beam search: each response holds every beam
};

// RequestEntry.kind from the sampling a request asked for.
Kind requestKind(ev1::Sampling const& s)
{
    if (s.beam_width() > 1)
    {
        return Kind::kBeams;
    }
    return s.num_sequences() > 1 ? Kind::kChildren : Kind::kSingle;
}

// What only the Python side builds (PythonRequestInputs and LoRA), for engine_server.submit; None
// when the request has none of it. objects: the list beside the message. Raises for a request this
// engine cannot take (a logits processor plugin, a malformed handoff, an object index the message
// does not carry), which ingress makes that request's error.
nb::object pythonExtras(ev1::Request const& q, nb::list const* objects)
{
    nb::dict extra;
    auto const& s = q.scheduling();
    if (s.has_attention_dp_rank() || s.attention_dp_strict() || s.agent_hierarchy_size()
        || !s.subagent_affinity_id().empty())
    {
        nb::object hierarchy = nb::none();
        if (s.agent_hierarchy_size())
        {
            nb::list hops;
            for (auto const& h : s.agent_hierarchy())
            {
                hops.append(nb::make_tuple(h.name(), h.index()));
            }
            hierarchy = hops;
        }
        extra["scheduling"] = nb::make_tuple(s.has_attention_dp_rank() ? nb::cast(s.attention_dp_rank()) : nb::none(),
            !s.attention_dp_strict(), hierarchy,
            s.subagent_affinity_id().empty() ? nb::none() : nb::cast(s.subagent_affinity_id()));
    }
    auto const& kv = q.kv();
    if (!kv.conversation_id().empty())
    {
        extra["conversation_id"] = kv.conversation_id();
    }
    if (!kv.kv_hints().empty())
    {
        extra["kv_hints"] = nb::bytes(kv.kv_hints().data(), kv.kv_hints().size());
    }
    if (q.has_lora())
    {
        extra["lora"] = nb::make_tuple(
            q.lora().id(), q.lora().path(), q.lora().format() == ev1::LORA_FORMAT_NEMO ? "nemo" : "hf");
    }
    if (q.has_watermark())
    {
        extra["watermark"] = q.watermark();
    }
    auto const& o = q.output();
    if (o.logprobs_values_only())
    {
        extra["logprobs_values_only"] = true;
    }
    if (o.has_prompt_logprobs())
    {
        extra["prompt_logprobs"] = nb::make_tuple(o.prompt_logprobs(), o.prompt_logprobs_values_only());
    }
    if (o.context_logits())
    {
        extra["context_logits"] = true;
    }
    if (o.generation_logits() || o.additional_outputs_size() || o.routed_experts())
    {
        extra["tensor_outputs"] = true;
    }
    if (o.routed_experts())
    {
        extra["routed_experts"] = true;
    }
    if (o.perf_metrics())
    {
        extra["perf_metrics"] = true;
    }
    if (!q.frontend_context().empty())
    {
        extra["frontend_context"] = nb::bytes(q.frontend_context().data(), q.frontend_context().size());
    }
    if (q.prompt_trimmed_prefix())
    {
        extra["prompt_trimmed_prefix"] = q.prompt_trimmed_prefix();
    }
    if (q.has_disagg())
    {
        auto const& d = q.disagg();
        extra["disagg"] = nb::make_tuple(static_cast<int>(d.phase()),
            d.has_schedule() ? nb::cast(static_cast<int>(d.schedule())) : nb::none(),
            d.has_disagg_request_id() ? nb::cast(d.disagg_request_id()) : nb::none(),
            d.has_orig_elapsed_s() ? nb::cast(d.orig_elapsed_s()) : nb::none(),
            d.kv_import().empty() ? nb::none() : unseal(d.kv_import()),
            d.has_prefill_usage()
                ? nb::object(nb::make_tuple(d.prefill_usage().prompt_tokens(), d.prefill_usage().cached_tokens()))
                : nb::none(),
            d.encoder_import().empty() ? nb::none() : unseal(d.encoder_import()));
    }
    if (q.has_multimodal())
    {
        auto const& m = q.multimodal();
        nb::list items;
        for (auto const& it : m.items())
        {
            items.append(nb::make_tuple(listOf(it.hash()), it.has_uuid() ? nb::cast(it.uuid()) : nb::none(),
                it.position(), it.length(), listOf(it.run_positions()), listOf(it.run_lengths())));
        }
        extra["multimodal"] = nb::make_tuple(items, valueOf(m.model_inputs(), objects));
    }
    auto const& sampling = q.sampling();
    if (sampling.has_embedding_bias())
    {
        extra["embedding_bias"] = tensorRefOf(sampling.embedding_bias());
    }
    if (sampling.token_bias_size())
    {
        nb::list pairs;
        for (auto const& b : sampling.token_bias())
        {
            pairs.append(nb::make_tuple(b.id(), b.bias()));
        }
        extra["token_bias"] = nb::tuple(pairs);
    }
    if (q.has_thinking_budget())
    {
        auto const& b = q.thinking_budget();
        extra["thinking_budget"]
            = nb::make_tuple(b.budget(), std::vector<int32_t>(b.start_ids().begin(), b.start_ids().end()),
                std::vector<int32_t>(b.end_ids().begin(), b.end_ids().end()));
    }
    if (q.logits_processors_size())
    {
        nb::list processors;
        for (auto const& p : q.logits_processors())
        {
            switch (p.processor_case())
            {
            case ev1::LogitsProcessor::kObject: processors.append(objectAt(p.object(), objects)); break;
            case ev1::LogitsProcessor::kName:
                throw std::runtime_error("unsupported: the logits processor plugin \"" + p.name() + "\"");
            case ev1::LogitsProcessor::PROCESSOR_NOT_SET:
                throw std::runtime_error("a logits processor with neither a name nor an object");
            }
        }
        extra["logits_processors"] = processors;
    }
    if (extra.size() == 0)
    {
        return nb::none();
    }
    return extra;
}

// ingress(SubmitBatch bytes, objects, max_seq_len, cp_size, is_generation_model) ->
//   [(client_id, tllm.Request or None, status, asked, limit, logprobs, logprobs_mode, end_id, arrival_time,
//     extra, kind, decoding_stats)]
// objects: the Python objects that came beside the message. The requests are built as the
// executor's Request takes them; the rest of the tuple is what the caller needs for
// PythonRequestInputs and its warnings and errors (status, asked and limit: MaxTokens). A request
// that cannot be built or read is None, with its error in extra["error"].
nb::list ingress(
    nb::bytes data, nb::list objects, std::optional<int64_t> maxSeqLen, int64_t cpSize, bool isGenerationModel)
{
    struct Built
    {
        uint64_t handle;
        std::optional<tle::Request> request;
        MaxTokens maxTokens;
        std::optional<uint32_t> logprobs;
        int logprobsMode;
        std::optional<int32_t> endId;
        std::optional<double> arrival;
        std::string error; // building or reading the request failed
    };

    ev1::SubmitBatch batch;
    std::vector<Built> built;
    {
        char const* p = data.c_str();
        size_t const n = data.size();
        nb::gil_scoped_release nogil;
        check(batch.ParseFromArray(p, static_cast<int>(n)), "malformed SubmitBatch");
        built.reserve(batch.requests_size());
        for (auto const& q : batch.requests())
        {
            auto const& stop = q.stopping();
            Built b{q.handle(), std::nullopt,
                deduceMaxTokens(static_cast<int64_t>(q.prompt_size()) + q.prompt_trimmed_prefix(),
                    stop.has_max_new_tokens() ? std::optional<int64_t>(stop.max_new_tokens()) : std::nullopt, maxSeqLen,
                    cpSize, isGenerationModel),
                q.output().has_logprobs() ? std::optional<uint32_t>(q.output().logprobs()) : std::nullopt,
                static_cast<int>(q.output().logprobs_mode()),
                stop.has_end_id() ? std::optional<int32_t>(stop.end_id()) : std::nullopt,
                // a division, as the frontend's seconds came from one
                q.scheduling().arrival_time_ns() ? std::optional<double>(q.scheduling().arrival_time_ns() / 1e9)
                                                 : std::nullopt};
            if (!isError(b.maxTokens.status))
            {
                try
                {
                    std::optional<tle::SizeType32> endId = stop.ignore_eos()
                        ? std::optional<tle::SizeType32>(-1)
                        : opt<tle::SizeType32>(b.endId.has_value(), b.endId.value_or(0));
                    tle::OutputConfig output = toOutput(q.output());
                    std::optional<tle::VecTokens> encoderInput;
                    if (q.encoder_input_size())
                    {
                        encoderInput.emplace(q.encoder_input().begin(), q.encoder_input().end());
                    }
                    b.request.emplace(tle::VecTokens(q.prompt().begin(), q.prompt().end()),
                        static_cast<tle::SizeType32>(b.maxTokens.value), q.stream(), toSampling(q.sampling(), stop),
                        output, endId, std::nullopt, wordLists(stop.bad()), wordLists(stop.stop()), std::nullopt,
                        std::nullopt, std::nullopt, std::nullopt, toRetention(q.kv()), std::move(encoderInput),
                        static_cast<tle::IdType>(q.handle()),
                        /*returnAllGeneratedTokens=*/false,
                        q.scheduling().has_priority() ? q.scheduling().priority() : tle::Request::kDefaultPriority,
                        requestType(q), toContextPhase(q), std::nullopt, std::nullopt, toGuided(q), std::nullopt,
                        q.has_disagg()
                            ? opt<tle::IdType>(q.disagg().has_disagg_request_id(), q.disagg().disagg_request_id())
                            : std::optional<tle::IdType>(0),
                        q.kv().has_cache_salt() ? std::optional<std::string>(q.kv().cache_salt()) : std::nullopt);
                }
                catch (std::exception const& e)
                {
                    b.error = e.what();
                }
            }
            built.push_back(std::move(b));
        }
    }

    nb::list out;
    for (size_t i = 0; i < built.size(); ++i)
    {
        auto& b = built[i];
        auto const& q = batch.requests(static_cast<int>(i));
        nb::object extra;
        try
        {
            extra = pythonExtras(q, &objects);
        }
        catch (std::exception const& e)
        {
            b.error = e.what();
            b.request.reset();
            extra = nb::none();
        }
        nb::object req = b.request ? nb::cast(std::move(*b.request), nb::rv_policy::move) : nb::none();
        if (!b.error.empty())
        {
            if (extra.is_none())
            {
                extra = nb::dict();
            }
            extra["error"] = b.error;
        }
        out.append(nb::make_tuple(b.handle, req, static_cast<int>(b.maxTokens.status), b.maxTokens.asked,
            b.maxTokens.limit, b.logprobs ? nb::cast(*b.logprobs) : nb::none(), b.logprobsMode,
            b.endId ? nb::cast(*b.endId) : nb::none(), b.arrival ? nb::cast(*b.arrival) : nb::none(), extra,
            static_cast<int>(requestKind(q.sampling())), q.output().decoding_stats()));
    }
    return out;
}

// ----------------------------------------------------------------------------------------------
// Rank 0: one step's responses to one StepOutput per frontend

// The attribute and key names the codec reads and sets on Python objects, each member named after
// its string (a leading underscore as "private").
struct Names
{
    nb::object clientId = nb::str("client_id");
    nb::object errorMsg = nb::str("error_msg");
    nb::object result = nb::str("result");
    nb::object privateResult = nb::str("_result");
    nb::object isFinal = nb::str("is_final");
    nb::object privatePyResult = nb::str("_py_result");
    nb::object privateLogProbs = nb::str("_log_probs");
    nb::object logProbs = nb::str("log_probs");
    nb::object cumLogProbs = nb::str("cum_log_probs");
    nb::object logprob = nb::str("logprob");
    nb::object rank = nb::str("rank");
    nb::object first = nb::str("first");
    nb::object kind = nb::str("kind");
    nb::object decodingStats = nb::str("decoding_stats");
    nb::object frontendContext = nb::str("frontend_context");
    nb::object cachedTokens = nb::str("cached_tokens");
    nb::object ctxComputedTokens = nb::str("ctx_computed_tokens");
    nb::object ctxFirstBegin = nb::str("ctx_first_begin");
    nb::object ctxNumChunks = nb::str("ctx_num_chunks");
    nb::object perPosDrafted = nb::str("per_pos_drafted");
    nb::object perPosAccepted = nb::str("per_pos_accepted");
    nb::object specDecTotals = nb::str("spec_dec_totals");
    nb::object timeBreakdownMetrics = nb::str("time_breakdown_metrics");
    nb::object samplingParams = nb::str("sampling_params");
    nb::object logprobs = nb::str("logprobs");
    nb::object logprobsSimpleFormat = nb::str("logprobs_simple_format");
    nb::object useBeamSearch = nb::str("use_beam_search");
    nb::object requestId = nb::str("request_id");
    nb::object decodingIter = nb::str("decoding_iter");
    nb::object avgDecodedTokensPerIter = nb::str("avg_decoded_tokens_per_iter");
    nb::object finishReasons = nb::str("finish_reasons");
    nb::object outputTokenIds = nb::str("output_token_ids");
    nb::object sequenceIndex = nb::str("sequence_index");
    nb::object counters = nb::str("counters");
    nb::object spec = nb::str("spec");
    nb::object requestPerfMetrics = nb::str("request_perf_metrics");
    nb::object promptLogprobs = nb::str("prompt_logprobs");
    nb::object promptLogprobsSimpleFormat = nb::str("prompt_logprobs_simple_format");
    nb::object tensors = nb::str("tensors");
    nb::object outputGroups = nb::str("output_groups");
    nb::object contextLogits = nb::str("context_logits");
    nb::object generationLogits = nb::str("generation_logits");
    nb::object additionalContextOutputs = nb::str("additional_context_outputs");
    nb::object additionalGenerationOutputs = nb::str("additional_generation_outputs");
    nb::object kvExport = nb::str("kv_export");
    nb::object encoderExport = nb::str("encoder_export");
    nb::object prefillUsage = nb::str("prefill_usage");
    // time_breakdown_metrics, in the order the executor writes its keys
    nb::object stepMetrics = nb::str("step_metrics");
    nb::object ctxChunkMetrics = nb::str("ctx_chunk_metrics");
    nb::object ctxGpuForwardTime = nb::str("ctx_gpu_forward_time");
    nb::object ctxGpuSampleTime = nb::str("ctx_gpu_sample_time");
    nb::object forwardStartTime = nb::str("forward_start_time");
    nb::object forwardEndTime = nb::str("forward_end_time");
    nb::object sampleStartTime = nb::str("sample_start_time");
    nb::object sampleEndTime = nb::str("sample_end_time");
    nb::object gpuForwardTime = nb::str("gpu_forward_time");
    nb::object gpuSampleTime = nb::str("gpu_sample_time");
    nb::object tokenTime = nb::str("token_time");
    nb::object iter = nb::str("iter");
    // getattr defaults, and the executor's "not measured yet" GPU time
    nb::object zero = nb::int_(0);
    nb::object minusOne = nb::int_(-1);
};

Names const& names()
{
    static Names const* n = new Names();
    return *n;
}

constexpr uint32_t kAppends = 0xFFFFFFFFU; // replace_from of a row that appends

int64_t toInt(nb::handle h)
{
    return h.is_none() ? 0 : nb::cast<int64_t>(h);
}

bool isFloat32(double v)
{
    return static_cast<double>(static_cast<float>(v)) == v || std::isnan(v);
}

struct Candidate
{
    int32_t id;
    float logprob;
    int64_t rank;
};

// One beam's logprobs in the columns' layout: per token the chosen token, then its top-k candidates.
struct BeamLogprobs
{
    std::vector<int32_t> chosenIds;   // to check against the tokens
    std::vector<float> chosen;
    std::vector<uint32_t> chosenRank; // 0 with values only
    std::vector<uint32_t> topkEnds;   // per token, where its candidates end
    std::vector<int32_t> topkIds;
    std::vector<float> topkLogprobs;
};

// One response: one row per beam (one row unless beam search).
struct Row
{
    nb::handle response; // borrowed from the caller's list
    nb::object entry;    // its RequestEntry
    uint64_t handle = 0;
    int lane = 0;
    bool takesFirst = false;           // the request's first row: clears RequestEntry.first once it goes out
    bool takesFrontendContext = false; // carries RequestEntry.frontend_context: clears it once it goes out
    Kind kind = Kind::kSingle;
    bool stats = true;                 // OutputOptions.decoding_stats
    char const* data = nullptr;
    size_t size = 0;
    bool isFinal = false;
    std::optional<std::string> error; // an error row, which says only this and ends its request
    bool hasLogprobs = false;
    bool valuesOnly = false;
    std::vector<BeamLogprobs> logprobs;
    std::vector<double> cum;
    std::optional<ev1::PromptInfo> prompt;
    std::optional<ev1::SpecDecodeStats> spec;
    std::optional<ev1::FinalInfo> final;
    std::vector<std::pair<std::string, ev1::TensorRef>> tensors;
    std::vector<nb::object> tensorData;         // the tensors' bytes, which go beside the message
    std::vector<std::string> groups;
    std::optional<std::string> frontendContext; // RequestEntry.frontend_context, on the request's first row
    std::optional<ev1::Value> kvExtra;        // prepared: what a prefill part hands over besides its ContextPhaseParams
    std::optional<std::string> encoderExport; // prepared: an encoder part's sealed handoff
    // filled without the GIL
    std::optional<std::string> kvExport; // a prefill part's sealed handoff
    bool unclaimed = false;              // not this wire's row, for rest: no RequestEntry's, or after its error row
    bool abortRequest = false;           // failRow ended a request that had not ended: rank 0 aborts it
    std::vector<tle::VecTokens> tokens;  // per beam
    std::vector<int> finish;             // per beam
    bool sequenceEnd = false;            // the row ends its sequence
    uint32_t seq = 0;                    // Kind::kChildren: the sequence index
    uint32_t decodingIter = 0;
    float avg = 0.F;
    std::optional<ev1::RequestTiming> timing;
};

// A response the wire cannot say: rank 0 ends its request with this error instead.
struct RowError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

void require(bool ok, char const* what)
{
    if (!ok)
    {
        throw RowError(what);
    }
}

// The sampler's dict[int, Logprob] for one token: the sampled token first, then the top-k by
// rank, where a sampled token that is among the top-k carries its top-k entry. A sampled token's
// rank of None (not computed) goes as 0.
void takeLogprobs(nb::handle dict, BeamLogprobs& out)
{
    auto const& n = names();
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    std::vector<Candidate> entries;
    while (PyDict_Next(dict.ptr(), &pos, &key, &value))
    {
        std::optional<int32_t> const id = int32Of(key);
        nb::object rank = getattrOrThrow(value, n.rank);
        require(id.has_value(), "a logprob of a token id that does not fit in int32");
        require(!rank.is_none() || entries.empty(), "a top-k logprob without a rank");
        double const lp = nb::cast<double>(getattrOrThrow(value, n.logprob));
        require(isFloat32(lp), "a logprob that is not a float32");
        entries.push_back({*id, static_cast<float>(lp), rank.is_none() ? 0 : nb::cast<int64_t>(rank)});
    }
    require(!entries.empty(), "a token without logprobs");
    int64_t maxOther = 0;
    for (size_t i = 1; i < entries.size(); ++i)
    {
        maxOther = std::max(maxOther, entries[i].rank);
    }
    std::vector<Candidate> topk(entries.begin() + 1, entries.end());
    if (entries[0].rank > 0 && entries[0].rank < maxOther)
    {
        topk.push_back(entries[0]);
    }
    std::sort(topk.begin(), topk.end(), [](Candidate const& a, Candidate const& b) { return a.rank < b.rank; });
    for (size_t i = 0; i < topk.size(); ++i)
    {
        require(topk[i].rank == static_cast<int64_t>(i) + 1, "top-k logprobs whose ranks are not 1 to k");
    }
    out.chosenIds.push_back(entries[0].id);
    out.chosen.push_back(entries[0].logprob);
    out.chosenRank.push_back(static_cast<uint32_t>(entries[0].rank));
    for (auto const& c : topk)
    {
        out.topkIds.push_back(c.id);
        out.topkLogprobs.push_back(c.logprob);
    }
    out.topkEnds.push_back(static_cast<uint32_t>(out.topkIds.size()));
}

// The logprobs a response carries, per beam. A storage nothing was appended to yet (a request
// that ended before its first token) has none, and the row says so.
void takeBeamLogprobs(nb::handle storage, Row& row)
{
    auto const& n = names();
    nb::object perBeam = getattrOr(storage, n.logProbs, Py_None);
    nb::object cum = getattrOr(storage, n.cumLogProbs, Py_None);
    if (perBeam.is_none() || cum.is_none())
    {
        return;
    }
    size_t const beams = nb::len(perBeam);
    require(beams > 0 && nb::len(cum) == beams && (row.kind == Kind::kBeams || beams == 1),
        "logprobs for other beams than the response's");
    row.hasLogprobs = true;
    row.logprobs.resize(beams);
    for (size_t b = 0; b < beams; ++b)
    {
        auto& out = row.logprobs[b];
        for (nb::handle item : perBeam[b])
        {
            if (PyDict_Check(item.ptr()))
            {
                require(!row.valuesOnly, "logprobs of both formats");
                takeLogprobs(item, out);
            }
            else
            {
                require(PyFloat_Check(item.ptr()), "a logprob of an unknown type");
                double const lp = PyFloat_AS_DOUBLE(item.ptr());
                require(row.valuesOnly || out.chosen.empty(), "logprobs of both formats");
                require(isFloat32(lp), "a logprob that is not a float32");
                row.valuesOnly = true;
                out.chosen.push_back(static_cast<float>(lp));
                out.chosenRank.push_back(0);
                out.topkEnds.push_back(static_cast<uint32_t>(out.topkIds.size()));
            }
        }
        row.cum.push_back(nb::cast<double>(cum[b]));
    }
}

// The speculative decoding counters (LlmResult.per_pos_drafted etc.).
void takeSpec(nb::handle result, nb::handle totals, Row& row)
{
    auto const& n = names();
    nb::object drafted = getattrOr(result, n.perPosDrafted, Py_None);
    nb::object accepted = getattrOr(result, n.perPosAccepted, Py_None);
    if (drafted.is_none() && accepted.is_none() && totals.is_none())
    {
        return;
    }
    require(PyList_Check(drafted.ptr()) && PyList_Check(accepted.ptr()),
        "speculative decoding counters that are not lists");
    ev1::SpecDecodeStats spec;
    for (nb::handle v : drafted)
    {
        spec.add_drafted_per_pos(nb::cast<uint32_t>(v));
    }
    for (nb::handle v : accepted)
    {
        spec.add_accepted_per_pos(nb::cast<uint32_t>(v));
    }
    if (!totals.is_none())
    {
        spec.set_total_accepted(nb::cast<uint64_t>(totals[0]));
        spec.set_total_drafted(nb::cast<uint64_t>(totals[1]));
    }
    row.spec = std::move(spec);
}

// A steady-clock time in seconds, as nanoseconds.
uint64_t toNs(nb::handle v)
{
    require(PyFloat_Check(v.ptr()) || PyLong_Check(v.ptr()), "a time that is not a number");
    double const s = nb::cast<double>(v);
    require(s >= 0., "a negative time");
    return static_cast<uint64_t>(std::llround(s * 1e9));
}

// A GPU time in milliseconds; an int 0 is the executor's "not measured yet".
std::optional<double> toMs(nb::handle v)
{
    if (PyFloat_Check(v.ptr()))
    {
        return PyFloat_AS_DOUBLE(v.ptr());
    }
    int overflow = 0;
    require(PyLong_Check(v.ptr()) && PyLong_AsLongAndOverflow(v.ptr(), &overflow) == 0 && overflow == 0,
        "a GPU time that is neither a float nor 0");
    return std::nullopt;
}

// One step_metrics or ctx_chunk_metrics entry, its keys in the order the executor writes them.
void takeStepTiming(nb::handle dict, bool generation, ev1::StepTiming* out)
{
    auto const& n = names();
    require(PyDict_Check(dict.ptr()), "a step timing that is not a dict");
    nb::handle const order[] = {n.forwardStartTime, n.forwardEndTime, n.sampleStartTime, n.sampleEndTime,
        n.gpuForwardTime, n.gpuSampleTime, n.tokenTime, n.iter};
    size_t const keys = generation ? 8 : 7;
    require(static_cast<size_t>(PyDict_GET_SIZE(dict.ptr())) == keys, "a step timing with other keys");
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    size_t i = 0;
    while (PyDict_Next(dict.ptr(), &pos, &key, &value))
    {
        int const same = PyObject_RichCompareBool(key, order[i].ptr(), Py_EQ);
        if (same < 0)
        {
            throw nb::python_error();
        }
        require(same, "a step timing with other keys");
        nb::handle v(value);
        switch (i++)
        {
        case 0: out->set_forward_start_ns(toNs(v)); break;
        case 1: out->set_forward_end_ns(toNs(v)); break;
        case 2: out->set_sample_start_ns(toNs(v)); break;
        case 3: out->set_sample_end_ns(toNs(v)); break;
        case 4:
            if (auto ms = toMs(v))
            {
                out->set_gpu_forward_ms(*ms);
            }
            break;
        case 5:
            if (auto ms = toMs(v))
            {
                out->set_gpu_sample_ms(*ms);
            }
            break;
        case 6: out->set_token_ns(toNs(v)); break;
        default:
            require(PyLong_Check(value), "a step timing whose iter is not an int");
            out->set_iter(nb::cast<uint32_t>(v));
        }
    }
}

// LlmResult.time_breakdown_metrics, its keys in the order the executor writes them.
void takeBreakdown(nb::handle dict, ev1::TimeBreakdown* out)
{
    auto const& n = names();
    require(PyDict_Check(dict.ptr()), "a time breakdown that is not a dict");
    nb::handle const order[] = {n.stepMetrics, n.ctxChunkMetrics, n.ctxGpuForwardTime, n.ctxGpuSampleTime};
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    size_t next = 0;
    while (PyDict_Next(dict.ptr(), &pos, &key, &value))
    {
        size_t i = next;
        for (; i < 4; ++i)
        {
            int const same = PyObject_RichCompareBool(key, order[i].ptr(), Py_EQ);
            if (same < 0)
            {
                throw nb::python_error();
            }
            if (same)
            {
                break;
            }
        }
        require(i < 4, "a time breakdown with other keys");
        next = i + 1;
        nb::handle v(value);
        if (i < 2)
        {
            require(PyList_Check(value) && PyList_GET_SIZE(value) > 0, "a time breakdown with an empty step list");
            for (nb::handle step : v)
            {
                takeStepTiming(step, i == 0, i == 0 ? out->add_steps() : out->add_ctx_chunks());
            }
        }
        else
        {
            require(PyFloat_Check(value), "a context GPU time that is not a float");
            if (i == 2)
            {
                out->set_ctx_gpu_forward_ms(PyFloat_AS_DOUBLE(value));
            }
            else
            {
                out->set_ctx_gpu_sample_ms(PyFloat_AS_DOUBLE(value));
            }
        }
    }
    require(next > 0, "an empty time breakdown");
}

// What engine_server.publish prepared in Python for a row: prompt logprobs as (ids, logprobs, ranks,
// top-k ends, top-k ids, top-k logprobs), tensors as {name: (dtype, shape, bytes, device)}, their
// bytes beside the message, and the tensor groups the row has.
void takePrepared(nb::handle prepared, Row& row)
{
    if (nb::handle v = key(prepared, "prompt_logprobs"))
    {
        require(row.prompt.has_value(), "prompt logprobs on a row that does not start its sequence");
        putColumns(v, row.prompt->mutable_prompt_logprobs());
    }
    if (nb::handle v = key(prepared, "prefill_usage")) // (prompt_tokens, cached_tokens), on the first and last row
    {
        require(row.prompt || row.final, "a prefill usage that changed on a middle row");
        for (auto* info : {row.prompt ? row.prompt->mutable_prefill_usage() : nullptr,
                 row.final ? row.final->mutable_prefill_usage() : nullptr})
        {
            if (info != nullptr)
            {
                info->set_prompt_tokens(nb::cast<uint32_t>(v[0]));
                info->set_cached_tokens(nb::cast<uint32_t>(v[1]));
            }
        }
    }
    try
    {
        if (nb::handle v = key(prepared, "kv_export"))    // a dict of what the handoff adds
        {
            putValue(v, &row.kvExtra.emplace(), nullptr); // sealed with the handoff, which holds no objects
        }
        if (nb::handle v = key(prepared, "encoder_export"))
        {
            row.encoderExport = seal(v);
        }
    }
    catch (NotCarried const&)
    {
        throw RowError("a handoff value engine.proto's Value has no kind for");
    }
    if (nb::handle v = key(prepared, "tensors"))
    {
        PyObject* name;
        PyObject* value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(v.ptr(), &pos, &name, &value))
        {
            ev1::TensorRef ref;
            putTensorRef(value, &ref, &row.tensorData); // its bytes beside the message; appendRow numbers them
            row.tensors.emplace_back(nb::cast<std::string>(nb::handle(name)), std::move(ref));
        }
    }
    if (nb::handle v = key(prepared, "groups"))
    {
        for (nb::handle g : v)
        {
            row.groups.push_back(nb::cast<std::string>(g));
        }
    }
}

// Reads one response under the GIL, leaving its RequestEntry as it is; it changes only once the
// row goes out. Returns false for a response no RequestEntry waits for. Raises RowError for one the
// wire cannot say, once row.entry, handle and lane are set. byWorker: the lanes are postprocess
// workers, a request's by its client id modulo their number; else frontends, by the client id's
// high bits. prepared: what engine_server.publish prepared in Python, by response index. seen: the
// client ids of this batch's earlier responses.
bool takeRow(nb::handle response, size_t index, nb::handle results, int numLanes, bool byWorker, int frontendShift,
    nb::handle prepared, std::unordered_set<uint64_t>& seen, Row& row)
{
    auto const& n = names();
    nb::object cid = getattrOr(response, n.clientId, Py_None);
    if (!PyLong_Check(cid.ptr()))
    {
        return false;
    }
    PyObject* entry = PyDict_GetItemWithError(results.ptr(), cid.ptr());
    if (entry == nullptr)
    {
        if (PyErr_Occurred())
        {
            throw nb::python_error();
        }
        return false;
    }
    row.response = response;
    row.entry = nb::borrow(entry);
    row.handle = nb::cast<uint64_t>(cid);
    bool const repeat = !seen.insert(row.handle).second;
    if (byWorker)
    {
        row.lane = static_cast<int>(row.handle % static_cast<uint64_t>(numLanes));
    }
    else
    {
        int const frontend = static_cast<int>(row.handle >> frontendShift);
        row.lane = frontend < numLanes ? frontend : 0;
    }
    row.kind = static_cast<Kind>(nb::cast<int>(getattrOrThrow(entry, n.kind)));
    row.stats = getattrOrThrow(entry, n.decodingStats).ptr() == Py_True;

    nb::object error = getattrOr(response, n.errorMsg, Py_None);
    if (!error.is_none())
    {
        row.error = nb::cast<std::string>(nb::str(error));
        row.isFinal = true;
        return true;
    }
    nb::object result = getattrOr(response, n.result, Py_None);
    require(!result.is_none(), "a response without a result");
    nb::object raw = getattrOr(result, n.privateResult, Py_None);
    require(PyBytes_Check(raw.ptr()), "a result without its serialized form");
    row.data = PyBytes_AS_STRING(raw.ptr());
    row.size = static_cast<size_t>(PyBytes_GET_SIZE(raw.ptr()));
    row.isFinal = isTrue(getattrOrThrow(result, n.isFinal));

    nb::object storage = getattrOrThrow(getattrOrThrow(result, n.privatePyResult), n.privateLogProbs);
    if (!storage.is_none())
    {
        takeBeamLogprobs(storage, row);
    }

    // The request-level counters: the executor's LlmResult carries them on every response. Children each have their
    // own, so their rows always say them; other rows say them on the first and the last row.
    // With decoding_stats the speculative decoding counters go on every row once the request has
    // drafted; without it, once, when the sequence ends.
    bool const children = row.kind == Kind::kChildren;
    bool const isFirst = !children && !repeat && getattrOrThrow(entry, n.first).ptr() == Py_True;
    row.takesFirst = isFirst;
    bool const counters = isFirst || row.isFinal || children;
    nb::object totals = getattrOr(result, n.specDecTotals, Py_None);
    bool const spec = row.stats ? counters || !totals.is_none() : row.isFinal || children;
    if (spec)
    {
        takeSpec(result, totals, row);
    }
    if (counters)
    {
        uint32_t const cached = static_cast<uint32_t>(toInt(getattrOr(result, n.cachedTokens, n.zero)));
        uint32_t const computed = static_cast<uint32_t>(toInt(getattrOr(result, n.ctxComputedTokens, n.zero)));
        int32_t const firstBegin = static_cast<int32_t>(toInt(getattrOr(result, n.ctxFirstBegin, n.minusOne)));
        uint32_t const chunks = static_cast<uint32_t>(toInt(getattrOr(result, n.ctxNumChunks, n.zero)));
        if (isFirst || children)
        {
            auto& p = row.prompt.emplace();
            p.set_cached_tokens(cached);
            p.set_ctx_computed_tokens(computed);
            p.set_ctx_first_begin(firstBegin);
            p.set_ctx_num_chunks(chunks);
        }
        if (row.isFinal)
        {
            auto& f = row.final.emplace();
            f.set_cached_tokens(cached);
            f.set_ctx_computed_tokens(computed);
            f.set_ctx_first_begin(firstBegin);
            f.set_ctx_num_chunks(chunks);
            nb::object breakdown = getattrOr(result, n.timeBreakdownMetrics, Py_None);
            if (!breakdown.is_none())
            {
                takeBreakdown(breakdown, f.mutable_breakdown());
            }
        }
    }
    if (!prepared.is_none())
    {
        PyObject* p = PyList_GET_ITEM(prepared.ptr(), static_cast<Py_ssize_t>(index));
        if (p != Py_None)
        {
            takePrepared(nb::handle(p), row);
        }
    }
    if (!repeat && (isFirst || children))
    {
        nb::object context = getattrOrThrow(entry, n.frontendContext);
        if (PyBytes_Check(context.ptr()))
        {
            row.frontendContext.emplace(PyBytes_AS_STRING(context.ptr()), PyBytes_GET_SIZE(context.ptr()));
            row.takesFrontendContext = true;
        }
    }
    return true;
}

uint64_t nsOf(tle::RequestPerfMetrics::TimePoint tp)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count());
}

ev1::RequestTiming toTiming(tle::RequestPerfMetrics const& m)
{
    ev1::RequestTiming t;
    auto const& tm = m.timingMetrics;
    t.set_arrival_ns(nsOf(tm.arrivalTime));
    t.set_first_scheduled_ns(nsOf(tm.firstScheduledTime));
    t.set_first_token_ns(nsOf(tm.firstTokenTime));
    t.set_last_token_ns(nsOf(tm.lastTokenTime));
    t.set_kv_transfer_start_ns(nsOf(tm.kvCacheTransferStart));
    t.set_kv_transfer_end_ns(nsOf(tm.kvCacheTransferEnd));
    t.set_kv_transfer_bytes(tm.kvCacheSize);
    auto const& kv = m.kvCacheMetrics;
    t.set_kv_blocks_allocated(static_cast<uint32_t>(kv.numTotalAllocatedBlocks));
    t.set_kv_blocks_new(static_cast<uint32_t>(kv.numNewAllocatedBlocks));
    t.set_kv_blocks_reused(static_cast<uint32_t>(kv.numReusedBlocks));
    t.set_kv_blocks_missed(static_cast<uint32_t>(kv.numMissedBlocks));
    t.set_kv_cache_hit_rate(kv.kvCacheHitRate);
    auto const& spec = m.speculativeDecoding;
    t.set_spec_acceptance_rate(spec.acceptanceRate);
    t.set_spec_accepted_tokens(static_cast<uint32_t>(spec.totalAcceptedDraftTokens));
    t.set_spec_draft_tokens(static_cast<uint32_t>(spec.totalDraftTokens));
    if (m.firstIter)
    {
        t.set_first_iter(static_cast<uint32_t>(*m.firstIter));
    }
    if (m.lastIter)
    {
        t.set_last_iter(static_cast<uint32_t>(*m.lastIter));
    }
    if (m.iter)
    {
        t.set_iter(static_cast<uint32_t>(*m.iter));
    }
    return t;
}

// Ends the row's request with an error instead of the response the wire cannot say: an error row,
// which says only its error, whatever else the row read before. Runs without the GIL.
void failRow(Row& row, std::string const& what)
{
    row.error = "cannot send the response: " + what;
    row.abortRequest = !row.isFinal;
    row.isFinal = true;
}

// Decodes a row's Result without the GIL.
void decodeRow(Row& row)
{
    if (row.error)
    {
        return;
    }
    MemBuf mb(row.data, row.size);
    std::istream is(&mb);
    tle::Result res = tle::Serialization::deserializeResult(is);
    size_t const beams = res.outputTokenIds.size();
    if (beams == 0 || res.finishReasons.size() != beams || res.isFinal != row.isFinal
        || (row.kind != Kind::kBeams && beams != 1) || (row.hasLogprobs && row.logprobs.size() != beams))
    {
        failRow(row, "a result whose beams do not match its response");
        return;
    }
    // a prefill part ends with its handoff and no finish reason
    bool const contextDone = res.contextPhaseParams.has_value() && res.isFinal;
    for (size_t b = 0; b < beams; ++b)
    {
        bool const finished = res.finishReasons[b] != tle::FinishReason::kNOT_FINISHED || contextDone;
        // a single sequence ends with the request; a child's own end is isSequenceFinal
        if ((row.kind == Kind::kSingle && finished != row.isFinal)
            || (row.kind == Kind::kChildren && finished != res.isSequenceFinal)
            || (row.hasLogprobs && !row.valuesOnly && row.logprobs[b].chosenIds != res.outputTokenIds[b])
            || (row.hasLogprobs && row.logprobs[b].chosen.size() != res.outputTokenIds[b].size()))
        {
            failRow(row, "a result whose finish or logprobs do not match its tokens");
            return;
        }
        row.finish.push_back(contextDone && res.finishReasons[b] == tle::FinishReason::kNOT_FINISHED
                ? static_cast<int>(ev1::FINISH_REASON_CONTEXT_DONE)
                : static_cast<int>(res.finishReasons[b]));
    }
    if (res.contextPhaseParams)
    {
        // the handoff, sealed: DisaggregatedParams' fields as a context-only request's params hold them
        auto const& c = *res.contextPhaseParams;
        ev1::Value h = row.kvExtra ? *row.kvExtra : ev1::Value();
        h.mutable_map();
        addField(h, "ctx_request_id")->set_integer(static_cast<int64_t>(c.getReqId()));
        auto* firstGen = addField(h, "first_gen_tokens")->mutable_list();
        for (auto t : c.getFirstGenTokens())
        {
            firstGen->add_items()->set_integer(t);
        }
        auto* draft = addField(h, "draft_tokens");
        if (auto const& tokens = c.getDraftTokens())
        {
            auto* list = draft->mutable_list();
            for (auto t : *tokens)
            {
                list->add_items()->set_integer(t);
            }
        }
        else
        {
            draft->set_none(true);
        }
        auto* rank = addField(h, "ctx_dp_rank");
        if (auto r = c.getCtxDpRank())
        {
            rank->set_integer(*r);
        }
        else
        {
            rank->set_none(true);
        }
        auto* endpoint = addField(h, "ctx_info_endpoint");
        if (auto const& e = c.getDisaggInfoEndpoint())
        {
            endpoint->set_text(*e);
        }
        else
        {
            endpoint->set_none(true);
        }
        auto* state = addField(h, "opaque_state");
        if (c.getState() != nullptr)
        {
            auto serialized = c.getSerializedState();
            state->set_raw(serialized.data(), serialized.size());
        }
        else
        {
            state->set_none(true);
        }
        check(h.SerializeToString(&row.kvExport.emplace()), "cannot serialize a handoff");
    }
    bool const children = row.kind == Kind::kChildren;
    row.tokens = std::move(res.outputTokenIds);
    row.seq = children ? static_cast<uint32_t>(res.sequenceIndex) : 0;
    row.sequenceEnd = children ? res.isSequenceFinal : row.isFinal;
    row.decodingIter = static_cast<uint32_t>(res.decodingIter);
    row.avg = res.avgDecodedTokensPerIter;
    if (res.requestPerfMetrics)
    {
        row.timing = toTiming(*res.requestPerfMetrics);
    }
}

struct LaneColumns
{
    bool logprobs = false;
    bool seq = false;
    bool replace = false;
    bool stats = false;
    uint32_t buffers = 0; // the buffers beside the lane's message so far
};

// An error row: the request's handle, no tokens, FINISH_REASON_ERROR, placeholders in the lane's
// other columns, and in its extras only the error.
void appendErrorRow(ev1::StepOutput& so, Row const& row, LaneColumns const& cols)
{
    int const index = so.handles_size();
    so.add_handles(row.handle);
    if (cols.seq)
    {
        so.add_seq(0);
    }
    if (cols.replace)
    {
        so.add_replace_from(kAppends);
    }
    so.add_offsets(static_cast<uint32_t>(so.tokens_size()));
    so.add_finish(ev1::FINISH_REASON_ERROR);
    if (cols.stats)
    {
        so.add_decoding_iter(0);
        so.add_avg_decoded_tokens_per_iter(0.F);
    }
    if (cols.logprobs)
    {
        so.add_cum_logprobs(std::numeric_limits<double>::quiet_NaN());
    }
    auto* extra = so.add_extras();
    extra->set_row(static_cast<uint32_t>(index));
    // an error response carries only its message, which names no ErrorCode
    auto* error = extra->mutable_error();
    error->set_code(ev1::ERROR_CODE_INTERNAL);
    error->set_message(*row.error);
    if (row.kind != Kind::kSingle)
    {
        extra->set_request_done(true);
    }
}

void appendRow(ev1::StepOutput& so, Row const& row, LaneColumns& cols)
{
    if (row.error)
    {
        appendErrorRow(so, row, cols);
        return;
    }
    size_t const beams = row.tokens.size();
    for (size_t b = 0; b < beams; ++b)
    {
        int const index = so.handles_size();
        so.add_handles(row.handle);
        bool const beam = row.kind == Kind::kBeams;
        if (cols.seq)
        {
            so.add_seq(beam ? static_cast<uint32_t>(b) : row.seq);
        }
        if (cols.replace)
        {
            so.add_replace_from(beam ? 0U : kAppends);
        }
        size_t const count = row.tokens[b].size();
        if (count)
        {
            so.mutable_tokens()->Add(row.tokens[b].begin(), row.tokens[b].end());
        }
        so.add_offsets(static_cast<uint32_t>(so.tokens_size()));
        so.add_finish(static_cast<ev1::FinishReason>(row.finish[b]));
        if (cols.stats)
        {
            so.add_decoding_iter(row.stats ? row.decodingIter : 0);
            so.add_avg_decoded_tokens_per_iter(row.stats ? row.avg : 0.F);
        }
        if (cols.logprobs)
        {
            auto* lp = so.mutable_logprobs();
            uint32_t const base = static_cast<uint32_t>(lp->topk_ids_size());
            if (row.hasLogprobs)
            {
                auto const& l = row.logprobs[b];
                lp->mutable_logprobs()->Add(l.chosen.begin(), l.chosen.end());
                lp->mutable_ranks()->Add(l.chosenRank.begin(), l.chosenRank.end());
                for (auto end : l.topkEnds)
                {
                    lp->add_topk_offsets(base + end);
                }
                lp->mutable_topk_ids()->Add(l.topkIds.begin(), l.topkIds.end());
                lp->mutable_topk_logprobs()->Add(l.topkLogprobs.begin(), l.topkLogprobs.end());
                so.add_cum_logprobs(row.cum[b]);
            }
            else
            {
                // a row without logprobs: aligned placeholders, and a NaN cumulative logprob
                for (size_t i = 0; i < count; ++i)
                {
                    lp->add_logprobs(0.F);
                    lp->add_ranks(0);
                    lp->add_topk_offsets(base);
                }
                so.add_cum_logprobs(std::numeric_limits<double>::quiet_NaN());
            }
        }
        if (b > 0)
        {
            continue;
        }
        // the response's extras go on its first row
        bool const multi = row.kind != Kind::kSingle;
        // FinalInfo where the sequence ends: the request's last row has it from takeRow, and a child
        // that ends before the others takes the counters of its PromptInfo
        bool const ends = row.final || (row.sequenceEnd && row.prompt);
        bool const spec = row.spec && row.stats;
        if (!(row.prompt || ends || spec || row.timing || multi || !row.tensors.empty() || !row.groups.empty()
                || row.frontendContext || row.kvExport || row.encoderExport))
        {
            continue;
        }
        auto* extra = so.add_extras();
        extra->set_row(static_cast<uint32_t>(index));
        if (row.prompt)
        {
            *extra->mutable_prompt_info() = *row.prompt;
        }
        if (ends)
        {
            auto* f = extra->mutable_final_info();
            if (row.final)
            {
                *f = *row.final;
            }
            else
            {
                f->set_cached_tokens(row.prompt->cached_tokens());
                f->set_ctx_computed_tokens(row.prompt->ctx_computed_tokens());
                f->set_ctx_first_begin(row.prompt->ctx_first_begin());
                f->set_ctx_num_chunks(row.prompt->ctx_num_chunks());
            }
            if (!row.stats)
            {
                f->set_decoding_iter(row.decodingIter);
                f->set_avg_decoded_tokens_per_iter(row.avg);
                if (row.spec)
                {
                    *f->mutable_spec() = *row.spec;
                }
            }
        }
        if (spec)
        {
            *extra->mutable_spec() = *row.spec;
        }
        if (row.timing)
        {
            *extra->mutable_timing() = *row.timing;
        }
        if (multi)
        {
            extra->set_request_done(row.isFinal);
        }
        for (auto const& [name, ref] : row.tensors)
        {
            auto& out = (*extra->mutable_tensors())[name];
            out = ref;
            out.set_buffer(cols.buffers++);
        }
        for (auto const& g : row.groups)
        {
            extra->add_output_groups(g);
        }
        if (row.frontendContext)
        {
            extra->set_frontend_context(*row.frontendContext);
        }
        if (row.kvExport)
        {
            extra->set_kv_export(*row.kvExport);
        }
        if (row.encoderExport)
        {
            extra->set_encoder_export(*row.encoderExport);
        }
    }
}

// publish(responses, results, num_lanes, by_worker, frontend_shift, prepared)
//   -> (frames, buffers, rest, finals, aborts)
// results: the RequestEntries by client id; by_worker: the lanes are postprocess workers;
// frontend_shift: where a client id's frontend id starts; prepared: what engine_server.publish
// computed in Python, a list aligned with responses, or None. frames: one StepOutput's bytes per
// lane (empty when the lane has no row); buffers: per lane, the tensor bytes its message names
// (TensorRef.buffer); rest: the unclaimed responses, those no RequestEntry waits for and those of a
// request after its error row, in their order; finals: the client ids whose last row went out here;
// aborts: the client ids of unfinished requests that ended with an error row because the wire
// could not say a response of theirs.
nb::tuple publish(
    nb::list responses, nb::dict results, int numLanes, bool byWorker, int frontendShift, nb::handle prepared)
{
    size_t const count = nb::len(responses);
    if (!prepared.is_none()
        && (!PyList_Check(prepared.ptr()) || static_cast<size_t>(PyList_GET_SIZE(prepared.ptr())) != count))
    {
        throw nb::type_error("prepared must be None or a list aligned with responses");
    }
    std::vector<Row> rows(count);
    std::unordered_set<uint64_t> seen;
    for (size_t i = 0; i < count; ++i)
    {
        Row& row = rows[i];
        try
        {
            row.unclaimed = !takeRow(responses[i], i, results, numLanes, byWorker, frontendShift, prepared, seen, row);
        }
        catch (RowError const& e)
        {
            failRow(row, e.what());
        }
    }

    std::vector<std::string> frames(static_cast<size_t>(std::max(numLanes, 1)));
    std::vector<LaneColumns> cols(frames.size());
    {
        nb::gil_scoped_release nogil;
        std::unordered_set<uint64_t> ended; // requests whose error row went out in this batch
        for (auto& row : rows)
        {
            if (row.unclaimed)
            {
                continue;
            }
            if (ended.count(row.handle))
            {
                row.unclaimed = true; // after its request's error row
                continue;
            }
            decodeRow(row);
            if (row.error)
            {
                ended.insert(row.handle);
            }
        }
        std::vector<ev1::StepOutput> outs(frames.size());
        for (auto const& row : rows)
        {
            if (row.unclaimed || row.error) // an error row has placeholders in the other rows' columns
            {
                continue;
            }
            auto& c = cols[row.lane];
            c.logprobs |= row.hasLogprobs;
            c.seq |= row.seq != 0 || (row.kind == Kind::kBeams && row.tokens.size() > 1);
            c.replace |= row.kind == Kind::kBeams;
            c.stats |= row.stats;
        }
        for (auto& so : outs)
        {
            so.add_offsets(0);
        }
        for (auto const& row : rows)
        {
            if (row.unclaimed)
            {
                continue;
            }
            auto& so = outs[row.lane];
            if (cols[row.lane].logprobs && !so.has_logprobs())
            {
                so.mutable_logprobs()->add_topk_offsets(0);
            }
            appendRow(so, row, cols[row.lane]);
        }
        for (size_t i = 0; i < outs.size(); ++i)
        {
            if (outs[i].handles_size())
            {
                check(outs[i].SerializeToString(&frames[i]), "cannot serialize StepOutput");
            }
        }
    }

    auto const& n = names();
    std::vector<nb::list> buffers(frames.size());
    nb::list rest;
    nb::list finals;
    nb::list aborts;
    for (size_t i = 0; i < count; ++i)
    {
        Row const& row = rows[i];
        if (row.unclaimed)
        {
            rest.append(responses[i]);
            continue;
        }
        if (row.isFinal)
        {
            finals.append(row.handle);
        }
        if (row.abortRequest)
        {
            aborts.append(row.handle);
        }
        if (row.error) // says only its error: it takes nothing from its RequestEntry
        {
            continue;
        }
        for (auto const& data : row.tensorData) // in the order appendRow numbered them
        {
            buffers[row.lane].append(data);
        }
        if (row.takesFirst)
        {
            setAttr(row.entry, n.first, Py_False);
        }
        if (row.takesFrontendContext)
        {
            setAttr(row.entry, n.frontendContext, Py_None);
        }
    }
    nb::list out;
    for (auto const& f : frames)
    {
        out.append(nb::bytes(f.data(), f.size()));
    }
    nb::list lanesBuffers;
    for (auto const& b : buffers)
    {
        lanesBuffers.append(b);
    }
    return nb::make_tuple(out, lanesBuffers, rest, finals, aborts);
}

// ----------------------------------------------------------------------------------------------
// Frontend: a StepOutput to rows for GenerationResult

// An instance of a row or result class, its __init__ not run: decodeRows sets its slots.
nb::object newObject(PyTypeObject* type, nb::handle empty)
{
    return own(type->tp_new(type, empty.ptr(), nullptr));
}

// Fills slot i of a list made by PyList_New, which takes the reference.
void fill(nb::handle list, Py_ssize_t i, nb::object value)
{
    PyList_SET_ITEM(list.ptr(), i, value.release().ptr());
}

nb::float_ seconds(uint64_t ns)
{
    return nb::float_(static_cast<double>(ns) / 1e9);
}

// tensorrt_llm.bindings.executor.RequestPerfMetrics, as the executor's Result gives it.
nb::object perfMetrics(ev1::RequestTiming const& t)
{
    using TimePoint = tle::RequestPerfMetrics::TimePoint;
    auto tp = [](uint64_t ns)
    { return TimePoint(std::chrono::duration_cast<TimePoint::duration>(std::chrono::nanoseconds(ns))); };
    tle::RequestPerfMetrics m;
    m.timingMetrics.arrivalTime = tp(t.arrival_ns());
    m.timingMetrics.firstScheduledTime = tp(t.first_scheduled_ns());
    m.timingMetrics.firstTokenTime = tp(t.first_token_ns());
    m.timingMetrics.lastTokenTime = tp(t.last_token_ns());
    m.timingMetrics.kvCacheTransferStart = tp(t.kv_transfer_start_ns());
    m.timingMetrics.kvCacheTransferEnd = tp(t.kv_transfer_end_ns());
    m.timingMetrics.kvCacheSize = t.kv_transfer_bytes();
    m.kvCacheMetrics.numTotalAllocatedBlocks = static_cast<tle::SizeType32>(t.kv_blocks_allocated());
    m.kvCacheMetrics.numNewAllocatedBlocks = static_cast<tle::SizeType32>(t.kv_blocks_new());
    m.kvCacheMetrics.numReusedBlocks = static_cast<tle::SizeType32>(t.kv_blocks_reused());
    m.kvCacheMetrics.numMissedBlocks = static_cast<tle::SizeType32>(t.kv_blocks_missed());
    m.kvCacheMetrics.kvCacheHitRate = t.kv_cache_hit_rate();
    m.speculativeDecoding.acceptanceRate = t.spec_acceptance_rate();
    m.speculativeDecoding.totalAcceptedDraftTokens = static_cast<tle::SizeType32>(t.spec_accepted_tokens());
    m.speculativeDecoding.totalDraftTokens = static_cast<tle::SizeType32>(t.spec_draft_tokens());
    if (t.has_first_iter())
    {
        m.firstIter = t.first_iter();
    }
    if (t.has_last_iter())
    {
        m.lastIter = t.last_iter();
    }
    if (t.has_iter())
    {
        m.iter = t.iter();
    }
    return nb::cast(std::move(m));
}

nb::dict stepTiming(ev1::StepTiming const& s)
{
    auto const& n = names();
    nb::dict d;
    d[n.forwardStartTime] = seconds(s.forward_start_ns());
    d[n.forwardEndTime] = seconds(s.forward_end_ns());
    d[n.sampleStartTime] = seconds(s.sample_start_ns());
    d[n.sampleEndTime] = seconds(s.sample_end_ns());
    d[n.gpuForwardTime] = s.has_gpu_forward_ms() ? nb::cast(s.gpu_forward_ms()) : n.zero;
    d[n.gpuSampleTime] = s.has_gpu_sample_ms() ? nb::cast(s.gpu_sample_ms()) : n.zero;
    d[n.tokenTime] = seconds(s.token_ns());
    if (s.has_iter())
    {
        d[n.iter] = s.iter();
    }
    return d;
}

// LlmResult.time_breakdown_metrics, its keys in the executor's order.
nb::dict timeBreakdown(ev1::TimeBreakdown const& t)
{
    auto const& n = names();
    nb::dict d;
    if (t.steps_size())
    {
        nb::list steps;
        for (auto const& s : t.steps())
        {
            steps.append(stepTiming(s));
        }
        d[n.stepMetrics] = steps;
    }
    if (t.ctx_chunks_size())
    {
        nb::list chunks;
        for (auto const& s : t.ctx_chunks())
        {
            chunks.append(stepTiming(s));
        }
        d[n.ctxChunkMetrics] = chunks;
    }
    if (t.has_ctx_gpu_forward_ms())
    {
        d[n.ctxGpuForwardTime] = t.ctx_gpu_forward_ms();
    }
    if (t.has_ctx_gpu_sample_ms())
    {
        d[n.ctxGpuSampleTime] = t.ctx_gpu_sample_ms();
    }
    return d;
}

// Checks a TokenLogprobs' columns against the number of tokens they are aligned with.
void checkTokenLogprobs(ev1::TokenLogprobs const& lp, int tokens, bool withIds)
{
    int const k = lp.topk_offsets_size();
    check(lp.logprobs_size() == tokens && lp.ranks_size() == tokens && k == tokens + 1
            && (!withIds || lp.ids_size() == tokens) && lp.topk_ids_size() == lp.topk_logprobs_size(),
        "malformed StepOutput: logprobs");
    check(lp.topk_offsets(0) == 0 && lp.topk_offsets(k - 1) == static_cast<uint32_t>(lp.topk_ids_size()),
        "malformed StepOutput: logprobs");
    for (int i = 0; i + 1 < k; ++i)
    {
        check(lp.topk_offsets(i) <= lp.topk_offsets(i + 1), "malformed StepOutput: logprobs");
    }
}

// Parses a StepOutput and checks every size and index the decoders read, so that a malformed one
// raises instead of reading out of bounds. finishReasons: how many finish reasons the reader knows.
void parseStepOutput(char const* data, size_t size, int finishReasons, ev1::StepOutput& so)
{
    check(so.ParseFromArray(data, static_cast<int>(size)), "malformed StepOutput");
    int const rows = so.handles_size();
    check(so.offsets_size() == rows + 1 && so.finish_size() == rows, "malformed StepOutput: row columns");
    check(so.offsets(0) == 0 && so.offsets(rows) == static_cast<uint32_t>(so.tokens_size()),
        "malformed StepOutput: offsets");
    for (int i = 0; i < rows; ++i)
    {
        check(so.offsets(i) <= so.offsets(i + 1), "malformed StepOutput: offsets");
        check(so.finish(i) >= 0 && so.finish(i) < finishReasons, "malformed StepOutput: finish");
    }
    for (int columnSize : {so.seq_size(), so.stop_index_size(), so.replace_from_size(), so.decoding_iter_size(),
             so.avg_decoded_tokens_per_iter_size()})
    {
        check(columnSize == 0 || columnSize == rows, "malformed StepOutput: optional columns");
    }
    check(so.decoding_iter_size() == so.avg_decoded_tokens_per_iter_size(), "malformed StepOutput: decoding counters");
    if (so.has_logprobs())
    {
        check(so.cum_logprobs_size() == rows, "malformed StepOutput: cum_logprobs");
        checkTokenLogprobs(so.logprobs(), so.tokens_size(), false);
    }
    for (auto const& e : so.extras())
    {
        check(e.row() < static_cast<uint32_t>(rows), "malformed StepOutput: extras");
        if (e.has_prompt_info() && e.prompt_info().has_prompt_logprobs())
        {
            checkTokenLogprobs(
                e.prompt_info().prompt_logprobs(), e.prompt_info().prompt_logprobs().logprobs_size(), true);
        }
    }
}

// decode_rows(StepOutput bytes, buffers, results, more_results, (row, error row) types,
//             (result, extras, outputs) types, finish_reasons, logprob_type)
//   -> [row]: the row and result objects of proto_wire.py, filled with what
//   GenerationResult._handle_step_row reads: a row with extras gets an extras result, one with
//   prompt logprobs, tensors or a handoff an outputs result, and an error the error row type. A
//   beam search response's rows become one row with a list per beam.
//   Logprobs are built as the sampler's dict[int, Logprob], or a float per token with
//   logprobs_simple_format, for the rows whose request asked for them.
nb::list decodeRows(nb::bytes data, nb::list buffers, nb::dict results, nb::handle moreResults, nb::tuple rowTypes,
    nb::tuple resultTypes, nb::tuple finishReasons, nb::handle logprobType)
{
    if (rowTypes.size() != 2 || resultTypes.size() != 3)
    {
        throw nb::type_error("decode_rows takes (row, error row) and (result, extras, outputs) types");
    }
    ev1::StepOutput so;
    {
        char const* p = data.c_str();
        size_t const size = data.size();
        int const reasons = static_cast<int>(finishReasons.size());
        nb::gil_scoped_release nogil;
        parseStepOutput(p, size, reasons, so);
    }
    auto const& n = names();
    PyTypeObject* const rowT = asType(rowTypes[0]);
    PyTypeObject* const errorRowT = asType(rowTypes[1]);
    PyTypeObject* const resultT = asType(resultTypes[0]);
    PyTypeObject* const extrasT = asType(resultTypes[1]);
    PyTypeObject* const outputsT = asType(resultTypes[2]);
    PyTypeObject* const logprobT = asType(logprobType);
    nb::object const empty = own(PyTuple_New(0));

    int const rowCount = so.handles_size();
    std::vector<ev1::RowExtra const*> extras(rowCount, nullptr);
    for (auto const& e : so.extras())
    {
        extras[e.row()] = &e;
    }
    bool const columnLogprobs = so.has_logprobs();
    bool const columnSeq = so.seq_size() == rowCount;
    bool const columnStats = so.decoding_iter_size() == rowCount;
    auto const& lp = so.logprobs();

    // dict[id] = Logprob(logprob=value, rank=rank); a rank of 0 is one the sampler did not compute (None)
    auto put = [&](nb::handle dict, int32_t id, float value, int64_t rank)
    {
        nb::object const o = newObject(logprobT, empty);
        setAttr(o, n.logprob, nb::float_(value));
        setAttr(o, n.rank, rank != 0 ? nb::object(nb::int_(rank)) : nb::none());
        if (PyDict_SetItem(dict.ptr(), nb::int_(id).ptr(), o.ptr()) != 0)
        {
            throw nb::python_error();
        }
    };
    // one token's logprobs: the chosen token first, then the top-k by rank, as the sampler builds it
    auto tokenLogprobs = [&](int j)
    {
        nb::object d = own(PyDict_New());
        put(d, so.tokens(j), lp.logprobs(j), lp.ranks(j));
        uint32_t const b = lp.topk_offsets(j), e = lp.topk_offsets(j + 1);
        for (uint32_t q = b; q < e; ++q)
        {
            put(d, lp.topk_ids(q), lp.topk_logprobs(q), q - b + 1);
        }
        return d;
    };
    // prompt logprobs, as the worker builds them: the top-k by rank, then the prompt token when it
    // is not among them; or a float per token
    auto promptLogprobs = [&](ev1::TokenLogprobs const& p, bool simple)
    {
        nb::object l = own(PyList_New(p.logprobs_size()));
        for (int t = 0; t < p.logprobs_size(); ++t)
        {
            if (simple)
            {
                fill(l, t, nb::float_(p.logprobs(t)));
                continue;
            }
            nb::object d = own(PyDict_New());
            uint32_t const b = p.topk_offsets(t), e = p.topk_offsets(t + 1);
            for (uint32_t q = b; q < e; ++q)
            {
                put(d, p.topk_ids(q), p.topk_logprobs(q), q - b + 1);
            }
            int const listed = PyDict_Contains(d.ptr(), nb::int_(p.ids(t)).ptr());
            if (listed < 0)
            {
                throw nb::python_error();
            }
            if (listed == 0)
            {
                put(d, p.ids(t), p.logprobs(t), p.ranks(t));
            }
            fill(l, t, std::move(d));
        }
        return l;
    };

    nb::list out;
    for (int i = 0; i < rowCount;)
    {
        uint64_t const handle = so.handles(i);
        nb::object const cid = nb::int_(handle);
        ev1::RowExtra const* extra = extras[i];

        bool const outputs = extra != nullptr
            && (extra->tensors_size() || extra->output_groups_size() || !extra->kv_export().empty()
                || !extra->encoder_export().empty()
                || (extra->has_prompt_info()
                    && (extra->prompt_info().has_prompt_logprobs() || extra->prompt_info().has_prefill_usage()))
                || (extra->has_final_info() && extra->final_info().has_prefill_usage()));
        bool const error = extra != nullptr && extra->has_error();
        nb::object const res = newObject(outputs ? outputsT : extra != nullptr ? extrasT : resultT, empty);
        nb::object const row = newObject(error ? errorRowT : rowT, empty);
        setAttr(row, n.clientId, cid);
        setAttr(row, n.requestId, cid);
        setAttr(row, n.result, res);
        out.append(row);

        if (error)
        {
            auto const& message = extra->error().message();
            setAttr(row, n.errorMsg, nb::str(message.data(), message.size()));
            setAttr(res, n.isFinal, Py_True);
            ++i;
            continue;
        }

        // the request's sampling params, read only for what few rows need
        nb::object sampling;
        auto samplingParams = [&]() -> nb::handle
        {
            if (!sampling.is_valid())
            {
                PyObject* gen = PyDict_GetItemWithError(results.ptr(), cid.ptr());
                if (gen == nullptr && !PyErr_Occurred() && !moreResults.is_none())
                {
                    gen = PyDict_GetItemWithError(moreResults.ptr(), cid.ptr());
                }
                if (gen == nullptr && PyErr_Occurred())
                {
                    throw nb::python_error();
                }
                sampling = gen == nullptr ? nb::none() : getattrOrThrow(gen, n.samplingParams);
            }
            return sampling;
        };

        // the rows of this response: a beam search response has one per beam
        int end = i + 1;
        bool const multi = extra != nullptr && extra->has_request_done();
        if (multi && !samplingParams().is_none() && isTrue(getattrOrThrow(samplingParams(), n.useBeamSearch)))
        {
            while (end < rowCount && so.handles(end) == handle
                && (extras[end] == nullptr || !extras[end]->has_request_done()))
            {
                ++end;
            }
        }
        int const beams = end - i;

        bool wantLogprobs = false;
        bool valuesOnly = false;
        // a NaN cumulative logprob: a row without logprobs (its request ended before its first token)
        if (columnLogprobs && !std::isnan(so.cum_logprobs(i)) && !samplingParams().is_none())
        {
            wantLogprobs = !getattrOrThrow(samplingParams(), n.logprobs).is_none();
            valuesOnly = wantLogprobs && isTrue(getattrOrThrow(samplingParams(), n.logprobsSimpleFormat));
        }

        nb::object const tokens = own(PyList_New(beams));
        nb::object const reasons = own(PyList_New(beams));
        nb::object const logProbs = wantLogprobs ? own(PyList_New(beams)) : nb::none();
        nb::object const cumLogProbs = wantLogprobs ? own(PyList_New(beams)) : nb::none();
        bool finishedAll = true;
        for (int b = 0; b < beams; ++b)
        {
            int const r = i + b;
            uint32_t const lo = so.offsets(r), hi = so.offsets(r + 1);
            nb::object ids = own(PyList_New(hi - lo));
            for (uint32_t j = lo; j < hi; ++j)
            {
                fill(ids, j - lo, nb::int_(so.tokens(j)));
            }
            fill(tokens, b, std::move(ids));
            int const finish = so.finish(r);
            finishedAll &= finish != 0;
            fill(reasons, b, nb::object(finishReasons[finish]));
            if (wantLogprobs)
            {
                nb::object perToken = own(PyList_New(hi - lo));
                for (uint32_t j = lo; j < hi; ++j)
                {
                    fill(
                        perToken, j - lo, valuesOnly ? nb::float_(lp.logprobs(j)) : tokenLogprobs(static_cast<int>(j)));
                }
                fill(logProbs, b, std::move(perToken));
                fill(cumLogProbs, b, nb::float_(so.cum_logprobs(r)));
            }
        }
        setAttr(res, n.outputTokenIds, tokens);
        setAttr(res, n.finishReasons, reasons);
        setAttr(res, n.logProbs, logProbs);
        setAttr(res, n.cumLogProbs, cumLogProbs);
        setAttr(res, n.isFinal, nb::bool_(multi ? extra->request_done() : finishedAll));
        // decoding_iter and avg_decoded_tokens_per_iter: the sequence's FinalInfo has them when its
        // request did not ask for decoding_stats, else the columns; None where neither says them.
        // This frontend asks for decoding_stats on its streaming requests, and the others have one
        // row per sequence, at its end
        ev1::FinalInfo const* fin = extra != nullptr && extra->has_final_info() ? &extra->final_info() : nullptr;
        if (fin != nullptr && fin->has_decoding_iter())
        {
            setAttr(res, n.decodingIter, nb::int_(fin->decoding_iter()));
            setAttr(res, n.avgDecodedTokensPerIter, nb::float_(fin->avg_decoded_tokens_per_iter()));
        }
        else if (columnStats)
        {
            setAttr(res, n.decodingIter, nb::int_(so.decoding_iter(i)));
            setAttr(res, n.avgDecodedTokensPerIter, nb::float_(so.avg_decoded_tokens_per_iter(i)));
        }
        else
        {
            setAttr(res, n.decodingIter, Py_None);
            setAttr(res, n.avgDecodedTokensPerIter, Py_None);
        }
        if (extra == nullptr)
        {
            // what the class says: no extras
            i = end;
            continue;
        }
        setAttr(res, n.sequenceIndex, nb::int_(columnSeq && beams == 1 ? so.seq(i) : 0U));

        // the request-level counters, when this row says them
        nb::object counters = nb::none();
        if (extra->has_final_info())
        {
            auto const& f = extra->final_info();
            counters
                = nb::make_tuple(f.cached_tokens(), f.ctx_computed_tokens(), f.ctx_first_begin(), f.ctx_num_chunks());
        }
        else if (extra->has_prompt_info())
        {
            auto const& p = extra->prompt_info();
            counters
                = nb::make_tuple(p.cached_tokens(), p.ctx_computed_tokens(), p.ctx_first_begin(), p.ctx_num_chunks());
        }
        setAttr(res, n.counters, counters);
        ev1::SpecDecodeStats const* specStats = extra->has_spec() ? &extra->spec()
            : fin != nullptr && fin->has_spec()                   ? &fin->spec()
                                                                  : nullptr;
        nb::object spec = nb::none();
        if (specStats != nullptr)
        {
            auto const& s = *specStats;
            // no totals before the sequence drafts
            nb::object const totals = s.total_drafted() > 0
                ? nb::object(nb::make_tuple(s.total_accepted(), s.total_drafted()))
                : nb::none();
            spec = nb::make_tuple(listOf(s.drafted_per_pos()), listOf(s.accepted_per_pos()), totals);
        }
        setAttr(res, n.spec, spec);
        setAttr(res, n.requestPerfMetrics, extra->has_timing() ? perfMetrics(extra->timing()) : nb::none());
        setAttr(res, n.timeBreakdownMetrics,
            extra->has_final_info() && extra->final_info().has_breakdown()
                ? nb::object(timeBreakdown(extra->final_info().breakdown()))
                : nb::none());

        if (outputs)
        {
            nb::object prompt = nb::none();
            if (extra->has_prompt_info() && extra->prompt_info().has_prompt_logprobs() && !samplingParams().is_none())
            {
                prompt = promptLogprobs(extra->prompt_info().prompt_logprobs(),
                    isTrue(getattrOrThrow(samplingParams(), n.promptLogprobsSimpleFormat)));
            }
            setAttr(res, n.promptLogprobs, prompt);
            nb::dict tensors;
            for (auto const& [name, ref] : extra->tensors())
            {
                tensors[nb::str(name.data(), name.size())] = tensorRefOf(ref, &buffers);
            }
            setAttr(res, n.tensors, tensors);
            nb::list groups;
            for (auto const& g : extra->output_groups())
            {
                groups.append(nb::str(g.data(), g.size()));
            }
            setAttr(res, n.outputGroups, groups);
            // until take_outputs builds them
            for (nb::handle name :
                {n.contextLogits, n.generationLogits, n.additionalContextOutputs, n.additionalGenerationOutputs})
            {
                setAttr(res, name, Py_None);
            }
            setAttr(res, n.kvExport, extra->kv_export().empty() ? nb::none() : unseal(extra->kv_export()));
            setAttr(
                res, n.encoderExport, extra->encoder_export().empty() ? nb::none() : unseal(extra->encoder_export()));
            ev1::PrefillUsage const* usage = nullptr;
            if (extra->has_final_info() && extra->final_info().has_prefill_usage())
            {
                usage = &extra->final_info().prefill_usage();
            }
            else if (extra->has_prompt_info() && extra->prompt_info().has_prefill_usage())
            {
                usage = &extra->prompt_info().prefill_usage();
            }
            setAttr(res, n.prefillUsage,
                usage != nullptr ? nb::object(nb::make_tuple(usage->prompt_tokens(), usage->cached_tokens()))
                                 : nb::none());
        }
        i = end;
    }
    return out;
}

// frontend_contexts(StepOutput bytes) -> [(client_id, bytes)]: the rows that carry their
// request's Request.frontend_context.
nb::list frontendContexts(nb::bytes data)
{
    ev1::StepOutput so;
    parseStepOutput(data.c_str(), data.size(), ev1::FinishReason_ARRAYSIZE, so);
    nb::list out;
    for (auto const& e : so.extras())
    {
        if (!e.frontend_context().empty())
        {
            out.append(nb::make_tuple(so.handles(static_cast<int>(e.row())),
                nb::bytes(e.frontend_context().data(), e.frontend_context().size())));
        }
    }
    return out;
}

} // namespace

namespace tensorrt_llm::nanobind::engine_wire
{

void initBindings(nb::module_& m)
{
    nb::exception<NotCarried>(m, "NotCarried", PyExc_ValueError);
    m.attr("KIND_SINGLE") = nb::int_(static_cast<int>(Kind::kSingle));
    m.attr("KIND_CHILDREN") = nb::int_(static_cast<int>(Kind::kChildren));
    m.attr("KIND_BEAMS") = nb::int_(static_cast<int>(Kind::kBeams));
    m.attr("STATUS_WARN_NO_MAX_SEQ_LEN") = nb::int_(static_cast<int>(Status::kWarnNoMaxSeqLen));
    m.attr("STATUS_WARN_TRUNCATED") = nb::int_(static_cast<int>(Status::kWarnTruncated));
    m.attr("STATUS_ERR_NEED_MAX_TOKENS") = nb::int_(static_cast<int>(Status::kErrNeedMaxTokens));
    m.attr("STATUS_ERR_CONTEXT_EXCEEDED") = nb::int_(static_cast<int>(Status::kErrContextExceeded));
    m.attr("STATUS_ERR_MAX_TOKENS") = nb::int_(static_cast<int>(Status::kErrMaxTokens));

    m.def("encode_requests", &encodeRequests, nb::arg("items"), nb::arg("objects"));
    m.def("ingress", &ingress, nb::arg("data"), nb::arg("objects"), nb::arg("max_seq_len").none(), nb::arg("cp_size"),
        nb::arg("is_generation_model"));
    m.def("publish", &publish, nb::arg("responses"), nb::arg("results"), nb::arg("num_lanes"), nb::arg("by_worker"),
        nb::arg("frontend_shift"), nb::arg("prepared").none());
    m.def("decode_rows", &decodeRows, nb::arg("data"), nb::arg("buffers"), nb::arg("results"),
        nb::arg("more_results").none(), nb::arg("row_types"), nb::arg("result_types"), nb::arg("finish_reasons"),
        nb::arg("logprob_type"));
    m.def("frontend_contexts", &frontendContexts, nb::arg("data"));
    m.def("set_tensor_codec", &setTensorCodec, nb::arg("tensor_type"), nb::arg("to_ref"), nb::arg("from_ref"));
}

} // namespace tensorrt_llm::nanobind::engine_wire
