/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#ifndef NCCL_COMPRESSED_REDUCE_SCATTER_H_
#define NCCL_COMPRESSED_REDUCE_SCATTER_H_

#include "nccl.h"

struct ncclCompressedP2pSlot {
  ncclCompressedFrameHeader_t frame;
};

#define NCCL_COMPRESSED_PIPELINE_DEPTH 3

struct ncclComm;
ncclResult_t ncclCompressedRsDestroy(struct ncclComm* comm);

ncclResult_t ncclLaunchCompressedPrepareEncode(
    ncclCompressionTask_t* task, void* txSlot, const void* sendbuff,
    size_t recvcount, size_t blockOffset, size_t elementCount,
    int destination, int typeSize, size_t maxPayloadBytes, cudaStream_t stream);

ncclResult_t ncclLaunchCompressedPrepareRound(
    ncclCompressionTask_t* task, void* rxSlot, void* txSlot,
    bool prepareSend, const void* sendbuff, void* recvbuff,
    size_t recvcount, size_t blockOffset, size_t elementCount,
    int destination, int typeSize,
    size_t maxPayloadBytes, cudaStream_t stream);

ncclResult_t ncclLaunchCompressedSend(
    void* txSlot, void* remoteRxSlot, size_t slotStride,
    size_t maxPayloadBytes, int segmentCount, int ctaCount,
    cudaStream_t stream);

#endif
