/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#include "compressed_reduce_scatter.h"
#include "common_kernel.h"
#include "checks.h"
#include "core.h"

namespace {

constexpr size_t PayloadOffset = (sizeof(ncclCompressedP2pSlot) + 15) & ~size_t(15);

__device__ __forceinline__ char* payload(void* slot) {
  return static_cast<char*>(slot) + PayloadOffset;
}

__device__ __forceinline__ ncclCompressedP2pSlot* control(void* slot) {
  return static_cast<ncclCompressedP2pSlot*>(slot);
}

__device__ __forceinline__ void setTaskData(
    ncclCompressionTask_t* task, const void* sendbuff, void* recvbuff,
    size_t recvcount, size_t blockOffset, size_t elementCount,
    int destination, int typeSize, size_t maxPayloadBytes) {
  size_t inputElement = static_cast<size_t>(destination) * recvcount + blockOffset;
  task->localInput = static_cast<const char*>(sendbuff) + inputElement * typeSize;
  task->output = recvbuff == nullptr ? nullptr : static_cast<char*>(recvbuff) + blockOffset * typeSize;
  task->elementCount = elementCount;
  task->compressedCapacity = maxPayloadBytes;
  task->reserved[0] = 0;
  task->reserved[1] = 0;
}

__global__ void prepareEncodeKernel(
    ncclCompressionTask_t* task, void* txSlot, const void* sendbuff,
    size_t recvcount, size_t blockOffset, size_t elementCount,
    int destination, int typeSize, size_t maxPayloadBytes) {
  if (threadIdx.x != 0) return;
  ncclCompressedP2pSlot* tx = control(txSlot);
  tx->frame = {};
  task->compressedInput = nullptr;
  task->inputHeader = nullptr;
  task->compressedOutput = payload(txSlot);
  task->outputHeader = &tx->frame;
  setTaskData(task, sendbuff, nullptr, recvcount, blockOffset, elementCount,
              destination, typeSize, maxPayloadBytes);
}

__global__ void prepareRoundKernel(
    ncclCompressionTask_t* task, void* rxSlot, void* txSlot, bool prepareSend,
    const void* sendbuff, void* recvbuff, size_t recvcount, size_t blockOffset,
    size_t elementCount, int destination, int typeSize, size_t maxPayloadBytes) {
  ncclCompressedP2pSlot* rx = control(rxSlot);
  if (threadIdx.x == 0) {
    task->compressedInput = nullptr;
    task->inputHeader = nullptr;
    task->compressedOutput = nullptr;
    task->outputHeader = nullptr;
    task->localInput = nullptr;
    task->output = nullptr;
    task->elementCount = 0;
    task->compressedCapacity = maxPayloadBytes;
    if (rx->frame.payloadBytes > maxPayloadBytes)
      rx->frame.flags |= NCCL_COMPRESSED_FRAME_FLAG_SIZE_ERROR;
    task->compressedInput = payload(rxSlot);
    task->inputHeader = &rx->frame;
    if (prepareSend) {
      ncclCompressedP2pSlot* tx = control(txSlot);
      tx->frame = {};
      task->compressedOutput = payload(txSlot);
      task->outputHeader = &tx->frame;
    } else {
      task->compressedOutput = nullptr;
      task->outputHeader = nullptr;
    }
    setTaskData(task, sendbuff, recvbuff, recvcount, blockOffset, elementCount,
                destination, typeSize, maxPayloadBytes);
  }
}

__global__ __launch_bounds__(256, 1) void sendSlicesKernel(
    void* txSlot, void* remoteRxSlot, size_t maxPayloadBytes) {
  __shared__ uint32_t bytes;
  ncclCompressedP2pSlot* tx = control(txSlot);
  ncclCompressedP2pSlot* rx = control(remoteRxSlot);

  if (threadIdx.x == 0) {
    bytes = tx->frame.payloadBytes;
    if (bytes > maxPayloadBytes) {
      bytes = 0;
      if (blockIdx.x == 0) {
        tx->frame.flags |= NCCL_COMPRESSED_FRAME_FLAG_SIZE_ERROR;
        tx->frame.payloadBytes = 0;
      }
    }
    if (blockIdx.x == 0) rx->frame = tx->frame;
  }
  __syncthreads();

  const uint4* src = reinterpret_cast<const uint4*>(payload(txSlot));
  uint4* dst = reinterpret_cast<uint4*>(payload(remoteRxSlot));
  size_t vectors = static_cast<size_t>(bytes) / sizeof(uint4);
  size_t vectorBegin = vectors * blockIdx.x / gridDim.x;
  size_t vectorEnd = vectors * (blockIdx.x + 1) / gridDim.x;
  constexpr size_t Unroll = 8;
  size_t i = vectorBegin + threadIdx.x;
  for (; i + (Unroll - 1) * blockDim.x < vectorEnd; i += Unroll * blockDim.x) {
    uint4 values[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; u++) values[u] = src[i + u * blockDim.x];
#pragma unroll
    for (int u = 0; u < Unroll; u++) dst[i + u * blockDim.x] = values[u];
  }
  for (; i < vectorEnd; i += blockDim.x) dst[i] = src[i];
  size_t tail = static_cast<size_t>(bytes) - vectors * sizeof(uint4);
  const char* srcTail = reinterpret_cast<const char*>(src + vectors);
  char* dstTail = reinterpret_cast<char*>(dst + vectors);
  if (blockIdx.x + 1 == gridDim.x)
    for (size_t i = threadIdx.x; i < tail; i += blockDim.x)
      dstTail[i] = srcTail[i];
}

} // namespace

ncclResult_t ncclLaunchCompressedPrepareEncode(
    ncclCompressionTask_t* task, void* txSlot, const void* sendbuff,
    size_t recvcount, size_t blockOffset, size_t elementCount,
    int destination, int typeSize, size_t maxPayloadBytes, cudaStream_t stream) {
  prepareEncodeKernel<<<1, 1, 0, stream>>>(task, txSlot, sendbuff, recvcount,
      blockOffset, elementCount, destination, typeSize, maxPayloadBytes);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclLaunchCompressedPrepareRound(
    ncclCompressionTask_t* task, void* rxSlot, void* txSlot, bool prepareSend,
    const void* sendbuff, void* recvbuff, size_t recvcount, size_t blockOffset,
    size_t elementCount, int destination, int typeSize, size_t maxPayloadBytes,
    cudaStream_t stream) {
  prepareRoundKernel<<<1, 1, 0, stream>>>(task, rxSlot, txSlot, prepareSend,
      sendbuff, recvbuff, recvcount, blockOffset, elementCount, destination,
      typeSize, maxPayloadBytes);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclLaunchCompressedSend(
    void* txSlot, void* remoteRxSlot, size_t maxPayloadBytes,
    int sliceCount, cudaStream_t stream) {
  if (sliceCount < 1) return ncclInvalidArgument;
  sendSlicesKernel<<<sliceCount, 256, 0, stream>>>(
      txSlot, remoteRxSlot, maxPayloadBytes);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
