/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#ifndef NCCL_COMPRESSED_REDUCE_SCATTER_H_
#define NCCL_COMPRESSED_REDUCE_SCATTER_H_

#include "nccl.h"

struct ncclCompressedP2pSlot {
  ncclCompressedFrameHeader_t frame;
};

#define NCCL_COMPRESSED_RX_DEPTH 3
#define NCCL_COMPRESSED_TX_DEPTH 3

struct ncclComm;
ncclResult_t ncclCompressedRsDestroy(struct ncclComm* comm);

ncclResult_t ncclLaunchCompressedPrepareEncodeAll(
    ncclCompressionTask_t* tasks, void* encoded, size_t slotStride,
    const void* sendbuff, size_t recvcount, size_t blockElements,
    size_t blockCount, const int* destinations, int typeSize,
    size_t maxPayloadBytes, cudaStream_t stream);

ncclResult_t ncclLaunchCompressedSend(
    void* txSlot, void* remoteRxSlot, size_t slotStride,
    size_t maxPayloadBytes, int segmentCount, int ctaCount,
    cudaStream_t stream);

#endif
