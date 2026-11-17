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

__global__ void prepareEncodeAllKernel(
    ncclCompressionTask_t* tasks, void* encoded, size_t slotStride,
    const void* sendbuff, size_t recvcount, size_t blockElements,
    size_t blockCount, const int* destinations, int typeSize,
    size_t maxPayloadBytes) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < blockCount;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t blockOffset = index * blockElements;
    size_t elementCount = min(blockElements, recvcount - blockOffset);
    void* encodedSlot = static_cast<char*>(encoded) + index * slotStride;
    ncclCompressedP2pSlot* frame = control(encodedSlot);
    frame->frame = {};
    ncclCompressionTask_t* task = tasks + index;
    task->compressedInput = nullptr;
    task->inputHeader = nullptr;
    task->compressedOutput = payload(encodedSlot);
    task->outputHeader = &frame->frame;
    setTaskData(task, sendbuff, nullptr, recvcount, blockOffset, elementCount,
                destinations[index], typeSize, maxPayloadBytes);
  }
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

__device__ __forceinline__ void sendSegment(
    void* txBase, void* remoteRxBase, size_t slotStride,
    size_t maxPayloadBytes, int segment, int segmentSlice,
    int segmentSlices, uint32_t* sharedBytes) {
  void* txSlot = static_cast<char*>(txBase) + slotStride * segment;
  void* remoteRxSlot = static_cast<char*>(remoteRxBase) + slotStride * segment;
  ncclCompressedP2pSlot* tx = control(txSlot);
  ncclCompressedP2pSlot* rx = control(remoteRxSlot);

  if (threadIdx.x == 0) {
    *sharedBytes = tx->frame.payloadBytes;
    if (*sharedBytes > maxPayloadBytes) {
      *sharedBytes = 0;
      if (segmentSlice == 0) {
        tx->frame.flags |= NCCL_COMPRESSED_FRAME_FLAG_SIZE_ERROR;
        tx->frame.payloadBytes = 0;
      }
    }
    if (segmentSlice == 0) rx->frame = tx->frame;
  }
  __syncthreads();

  constexpr size_t PackBytes = 16;
  constexpr size_t Unroll = 8;
  uintptr_t src = cvta_to_global(payload(txSlot));
  uintptr_t dst = cvta_to_global(payload(remoteRxSlot));
  size_t packs = static_cast<size_t>(*sharedBytes) / PackBytes;
  size_t packBegin = packs * segmentSlice / segmentSlices;
  size_t packEnd = packs * (segmentSlice + 1) / segmentSlices;
  size_t i = packBegin + threadIdx.x;
  for (; i + (Unroll - 1) * blockDim.x < packEnd;
       i += Unroll * blockDim.x) {
    BytePack<PackBytes> values[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; u++)
      values[u] = ld_volatile_global<PackBytes>(
          src + (i + u * blockDim.x) * PackBytes);
#pragma unroll
    for (int u = 0; u < Unroll; u++)
      st_global<PackBytes>(dst + (i + u * blockDim.x) * PackBytes, values[u]);
  }
  for (; i < packEnd; i += blockDim.x) {
    BytePack<PackBytes> value =
        ld_volatile_global<PackBytes>(src + i * PackBytes);
    st_global<PackBytes>(dst + i * PackBytes, value);
  }
  size_t tail = static_cast<size_t>(*sharedBytes) - packs * PackBytes;
  const char* srcTail = reinterpret_cast<const char*>(src + packs * PackBytes);
  char* dstTail = reinterpret_cast<char*>(dst + packs * PackBytes);
  if (segmentSlice + 1 == segmentSlices)
    for (size_t i = threadIdx.x; i < tail; i += blockDim.x)
      dstTail[i] = srcTail[i];
  __syncthreads();
}

__global__ __launch_bounds__(512, 1) void sendSlicesKernel(
    void* txBase, void* remoteRxBase, size_t slotStride,
    size_t maxPayloadBytes, int segmentCount) {
  __shared__ uint32_t bytes;
  if (gridDim.x >= segmentCount) {
    int segment = blockIdx.x % segmentCount;
    int segmentSlice = blockIdx.x / segmentCount;
    int segmentSlices = (gridDim.x + segmentCount - 1 - segment) / segmentCount;
    sendSegment(txBase, remoteRxBase, slotStride, maxPayloadBytes,
                segment, segmentSlice, segmentSlices, &bytes);
  } else {
    for (int segment = blockIdx.x; segment < segmentCount; segment += gridDim.x)
      sendSegment(txBase, remoteRxBase, slotStride, maxPayloadBytes,
                  segment, 0, 1, &bytes);
  }
}

} // namespace

ncclResult_t ncclLaunchCompressedPrepareEncodeAll(
    ncclCompressionTask_t* tasks, void* encoded, size_t slotStride,
    const void* sendbuff, size_t recvcount, size_t blockElements,
    size_t blockCount, const int* destinations, int typeSize,
    size_t maxPayloadBytes, cudaStream_t stream) {
  if (blockCount == 0 || destinations == nullptr) return ncclInvalidArgument;
  constexpr int Threads = 256;
  size_t requiredBlocks = (blockCount + Threads - 1) / Threads;
  int blocks = static_cast<int>(requiredBlocks < 65535 ? requiredBlocks : 65535);
  prepareEncodeAllKernel<<<blocks, Threads, 0, stream>>>(tasks, encoded,
      slotStride, sendbuff, recvcount, blockElements, blockCount, destinations,
      typeSize, maxPayloadBytes);
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
    void* txSlot, void* remoteRxSlot, size_t slotStride,
    size_t maxPayloadBytes, int segmentCount, int ctaCount,
    cudaStream_t stream) {
  // A CTA may process multiple segments serially when the communication group
  // contains more segments than the configured CTA budget.
  if (segmentCount < 1 || ctaCount < 1) return ncclInvalidArgument;
  sendSlicesKernel<<<ctaCount, 512, 0, stream>>>(
      txSlot, remoteRxSlot, slotStride, maxPayloadBytes, segmentCount);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
