/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_REDUCE_SCATTER_RING_CODEC_H_
#define NCCL_REDUCE_SCATTER_RING_CODEC_H_

// Customization point for codecs carried by Ring ReduceScatter.
//
// A codec specialization owns the representation stored in the connection
// FIFO, while ReduceScatter continues to own rank/chunk scheduling. A
// specialization is expected to implement the following three operations:
//
//   encodeSend              local input -> encoded send FIFO
//   decodeReduceEncodeSend  encoded recv FIFO + local input -> encoded send FIFO
//   decodeReduceCopy        encoded recv FIFO + local input -> user output
//
// Implementations for ProtoSimple can use Primitives::process() to retain the
// existing wait/post and FIFO credit protocol. The callback must set dstSizes
// in units of T; consequently an encoded payload must be padded to sizeof(T).
// It must also fit in maxSliceSize*sizeof(T) bytes. Call syncCodecWorkers() or
// anyCodecWorker() from a process callback when the codec needs CTA-worker
// prefix scans or other multi-phase cooperation.
//
// Specialize this class for the supported (T, RedOp, Proto) tuple. No changes
// to the ReduceScatter ring scheduler are needed when a codec is added.
#include "nccl.h"
#include "primitives.h"
#include "reduce_kernel.h"
template <typename T, typename RedOp, typename Proto>
struct ReduceScatterRingCodec {
  static constexpr bool IsEncoded = false;

  // Return the number of source elements that one encoded ring chunk may
  // consume. Encoded formats should reduce this value when their worst-case
  // payload (including metadata) is larger than sizeof(T) per source element.
  __device__ __forceinline__ static size_t chunkElementLimit(size_t scheduledChunkElements) {
    return scheduledChunkElements;
  }

  template <typename Prims>
  __device__ __forceinline__ static void encodeSend(
    Prims& prims,
    struct ncclDevWorkColl* work,
    intptr_t inpIx,
    int eltN
  ) {
    prims.send(inpIx, eltN);
  }

  template <typename Prims>
  __device__ __forceinline__ static void decodeReduceEncodeSend(Prims& prims,
                                                                 struct ncclDevWorkColl* /*work*/, intptr_t inpIx,
                                                                 int eltN) {
    prims.recvReduceSend(inpIx, eltN);
  }

  template <typename Prims>
  __device__ __forceinline__ static void decodeReduceCopy(Prims& prims, struct ncclDevWorkColl* /*work*/,
                                                           intptr_t inpIx, intptr_t outIx, int eltN, bool postOp) {
    prims.recvReduceCopy(inpIx, outIx, eltN, postOp);
  }
};

// Project-specific codecs are provided as specializations in this extension
// header so adding a codec does not require another scheduler change.
#include "zipccl_ring_codec.h"

#endif // NCCL_REDUCE_SCATTER_RING_CODEC_H_
