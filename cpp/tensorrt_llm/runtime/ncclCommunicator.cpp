/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/ncclCommunicator.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

using namespace tensorrt_llm::runtime;

namespace
{
#if ENABLE_MULTI_DEVICE

ncclDataType_t toNcclType(tensorrt_llm::DataType dataType)
{
    switch (dataType)
    {
    case tensorrt_llm::DataType::kFLOAT: return ncclFloat32;
    case tensorrt_llm::DataType::kHALF: return ncclHalf;
    case tensorrt_llm::DataType::kINT8: return ncclInt8;
    case tensorrt_llm::DataType::kINT32: return ncclInt32;
    case tensorrt_llm::DataType::kUINT8: return ncclUint8;
    case tensorrt_llm::DataType::kINT64: return ncclInt64;
    case tensorrt_llm::DataType::kFP8: return ncclUint8;
#if ENABLE_BF16
    case tensorrt_llm::DataType::kBF16: return ncclBfloat16;
#endif // ENABLE_BF16
    default: TLLM_THROW("Unsupported data type: %d", static_cast<int>(dataType));
    }
}
#endif // ENABLE_MULTI_DEVICE
} // namespace

void NcclCommunicator::send(
    void const* sendbuff, size_t count, tensorrt_llm::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    TLLM_NCCL_CHECK(ncclSend(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicator::receive(
    void* sendbuff, size_t count, tensorrt_llm::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    TLLM_NCCL_CHECK(ncclRecv(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

ncclComm_t NcclCommunicator::createComm(int worldSize, int rank, mpi::MpiComm const& mpiComm)
{
#if ENABLE_MULTI_DEVICE

    ncclUniqueId id;
    if (rank == 0)
    {
        ncclGetUniqueId(&id);
    }
    mpiComm.bcastValue(id, 0);
    ncclComm_t comm;
// Need static connection initialization for accurate KV cache size estimation
#if defined(_WIN32)
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
#endif // _WIN32
    TLLM_NCCL_CHECK(ncclCommInitRank(&comm, worldSize, id, rank));
    return comm;
#else
    // Python runtime requires instantiation of a communicator even though it may never be used to enable
    // pipeline parallel code-path. To enable this, have an empty communicator with uninitialized state.
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

NcclCommunicator::~NcclCommunicator()
{
#if ENABLE_MULTI_DEVICE
    if (mComm && ncclCommDestroy(mComm) != ncclSuccess)
    {
        TLLM_LOG_WARNING("Failed to destroy NCCL communicator.");
    }
#endif // ENABLE_MULTI_DEVICE
}
