/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_ZIPCCL_RING_CODEC_H_
#define NCCL_ZIPCCL_RING_CODEC_H_

#include "zipccl_frame_codec.h"

namespace ncclZipRing {
using namespace zipccl::device;


// Publish a consistent aggregate of all completed StatEncode slices. Each slot has one writer.
template<class P>
__device__ float publishStats(P& prims, int tid, int workers, ncclZipStats& stats, unsigned slot,
                              ncclZipMoments local) {
  auto* partial = static_cast<ncclZipMoments*>(prims.codecScratch());
  prims.syncCodecWorkers();
  for (int d = 16; d; d /= 2) {
    ncclZipMoments other{__shfl_down_sync(~0u, local.count, d), __shfl_down_sync(~0u, local.mean, d),
                         __shfl_down_sync(~0u, local.m2, d), __shfl_down_sync(~0u, local.exponent, d)};
    if ((tid % 32) < d) local.merge(other);
  }
  if (tid % 32 == 0) partial[tid / 32] = local;
  prims.syncCodecWorkers();
  if (tid != 0) return 0;
  ncclZipMoments slice{};
  for (int w = 0; w < workers / 32; ++w) slice.merge(partial[w]);
  return ncclZipPublishSlice(stats, slot, slice);
}

struct SliceEstimate { float sigma; unsigned slot; };

template<bool Encode, class P>
__device__ SliceEstimate sliceEstimate(P& prims, int tid, ncclZipStats& stats, const uint16_t* local, int n) {
  auto* shared = static_cast<SliceEstimate*>(prims.codecScratch());
  if (tid == 0) {
    auto snapshot = atomicCAS(&stats.snapshot, 0ull, 0ull);
    // The first frame needs a window before its encoder can produce moments.
    if (!snapshot && n) {
      ncclZipSums pilot{};
      int samples = min(n, 32);
      for (int i = 0; i < samples; ++i)
        pilot.add(__bfloat162float(__ushort_as_bfloat16(local[size_t(i) * n / samples])));
      auto initial = (1ull << 32) | __float_as_uint(pilot.moments().sigma());
      snapshot = atomicCAS(&stats.snapshot, 0ull, initial);
      if (!snapshot) snapshot = initial;
    }
    unsigned slot = ncclZipStatSlices;
    if (Encode && n && (snapshot >> 32) <= ncclZipStatSlices &&
        atomicAdd(&stats.claimed, 0u) < ncclZipStatSlices)
      slot = atomicAdd(&stats.claimed, 1u);
    *shared = {__uint_as_float(unsigned(snapshot)), slot};
  }
  prims.syncCodecWorkers();
  auto estimate = *shared;
  prims.syncCodecWorkers();
  return estimate;
}

using Frame = RingFrameFormat;

template<class P>
__device__ const RingFrameHeader* stageFrame(P& prims, int tid, int workers,
                                             const RingFrameHeader* src, uint8_t* dst, unsigned capacity) {
  auto* sharedBytes = static_cast<unsigned*>(prims.codecScratch());
  if (tid == 0) *sharedBytes = src->frameBytes;
  prims.syncCodecWorkers();
  unsigned bytes = *sharedBytes;
  if (bytes < sizeof(RingFrameHeader) || bytes > capacity) {
    prims.syncCodecWorkers();
    return src;
  }
  auto* source = reinterpret_cast<const uint8_t*>(src);
  unsigned vectors = bytes / sizeof(uint4);
  for (unsigned i = tid; i < vectors; i += workers)
    reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(source)[i];
  for (unsigned i = vectors * sizeof(uint4) + tid; i < bytes; i += workers)
    dst[i] = source[i];
  prims.syncCodecWorkers();
  return reinterpret_cast<const RingFrameHeader*>(dst);
}

template<bool Reduce, bool Statistics, Frame In, class P>
__device__ int dispatchOutput(Frame out, P& p, int tid, int workers, const PayloadArgs& args, ncclZipSums* sums) {
  if (out == Frame::Fixed) return processPayload<Reduce, Statistics, In, Frame::Fixed>(p, tid, workers, args, sums);
  if (out == Frame::Compressed) return processPayload<Reduce, Statistics, In, Frame::Compressed>(p, tid, workers, args, sums);
  return processPayload<Reduce, Statistics, In, Frame::Raw>(p, tid, workers, args, sums);
}

template<SegmentOp Stage, class P>
__device__ int run(P& prims, int tid, int workers, const uint16_t* local, int n, const float* suppliedSigma,
                   const RingFrameHeader* src, RingFrameHeader* dst, uint16_t* dense, bool fixedSend = false) {
  constexpr bool encode = Stage == SegmentOp::Encode;
  constexpr bool finalCopy = Stage == SegmentOp::Copy;
  Frame in = encode ? Frame::Raw : src->format;
  Frame out = finalCopy || n < ncclZipMinSliceBytes / sizeof(uint16_t) ? Frame::Raw :
              fixedSend ? Frame::Fixed : Frame::Compressed;
  auto& stats = reinterpret_cast<ncclKernelCommAndChannels*>(ncclShmem.args.comm)->zipStats;
  SliceEstimate estimate{0.0f, ncclZipStatSlices};
  if constexpr (!finalCopy) {
    if (suppliedSigma) {
      float sigma = *suppliedSigma;
      estimate.sigma = isfinite(sigma) && sigma >= 0.0f ? sigma : 0.0f;
    } else {
      estimate = sliceEstimate<encode>(prims, tid, stats, local, n);
    }
  }
  float sigma = estimate.sigma;
  if constexpr (Stage == SegmentOp::Dre) sigma = hypotf(src->sigma, sigma);
  int base = out == Frame::Raw ? 0 : ncclZipWindow(sigma);
  PayloadArgs args{local, encode ? nullptr : src + 1, finalCopy ? static_cast<void*>(dense) : dst + 1,
                   n, encode ? 0 : src->base, base};
  int bytes;
  if constexpr (encode) {
    if (estimate.slot < ncclZipStatSlices) {
      ncclZipSums sums{};
      bytes = dispatchOutput<false, true, Frame::Raw>(out, prims, tid, workers, args, &sums);
      sigma = publishStats(prims, tid, workers, stats, estimate.slot, sums.moments());
    } else {
      bytes = dispatchOutput<false, false, Frame::Raw>(out, prims, tid, workers, args, nullptr);
    }
  } else {
    if (in == Frame::Fixed)
      bytes = dispatchOutput<true, false, Frame::Fixed>(out, prims, tid, workers, args, nullptr);
    else if (in == Frame::Compressed)
      bytes = dispatchOutput<true, false, Frame::Compressed>(out, prims, tid, workers, args, nullptr);
    else
      bytes = dispatchOutput<true, false, Frame::Raw>(out, prims, tid, workers, args, nullptr);
  }
  if constexpr (!finalCopy) {
    unsigned frameBytes = sizeof(RingFrameHeader) + bytes;
    if (tid == 0) *dst = {unsigned(n), encode ? 1u : src->contributors + 1, uint16_t(base), out, sigma, frameBytes};
    return frameBytes;
  }
  return 0;
}
} // namespace ncclZipRing

template <typename Prims, int Stage>
struct ZipCclRingProcess {
  Prims* prims;
  ncclDevWorkColl* work;
  intptr_t inpIx, outIx;
  int eltN;
  template<int Slices, int, int, int, int, int, int>
  __device__ void operator()(int tid, int workers, int slice, int fifoElements, int nSrcs, void** srcs,
                             int nDsts, void** dsts, int32_t* sizes, uint32_t, uint32_t) {
    using namespace ncclZipRing;
    // Use the Simple slice partition in units of original BF16 elements.
    constexpr int rawSliceElements = ncclZipRawSliceBytes / sizeof(uint16_t);
    int sliceSize = max(divUp(eltN, 16 * Slices) * 16, rawSliceElements / 32);
    int begin = min(eltN, slice * sliceSize);
    int n = min(sliceSize, eltN - begin);
    if ((Stage != 0 && !nSrcs) || (Stage != 2 && !nDsts)) return;
    const auto* local = static_cast<const uint16_t*>(work->sendbuff) + inpIx + begin;
    auto* dense = Stage == 2 ? static_cast<uint16_t*>(work->recvbuff) + outIx + begin : nullptr;
    const RingFrameHeader* src = nullptr;
    if constexpr (Stage != 0) {
      src = static_cast<const RingFrameHeader*>(srcs[0]);
      if (prims->zipStageRecv()) {
        unsigned capacity = fifoElements * sizeof(uint16_t);
        auto* comm = reinterpret_cast<const ncclKernelCommAndChannels*>(ncclShmem.args.comm);
        auto* staged = comm->zipRecvFrames + size_t(ncclShmem.channelId) * capacity;
        src = stageFrame(*prims, tid, workers, src, staged, capacity);
      }
    }
    constexpr SegmentOp op = Stage == 0 ? SegmentOp::Encode : Stage == 1 ? SegmentOp::Dre : SegmentOp::Copy;
    int bytes = run<op>(*prims, tid, workers, local, n, work->zipSigma, src,
                        Stage == 2 ? nullptr : static_cast<RingFrameHeader*>(dsts[0]), dense,
                        Stage != 2 && prims->zipFixedSend());
    if (Stage != 2 && tid == 0) sizes[0] = (bytes + 1)/2;
  }
};

template <int SlicePerChunk, int StepPerSlice, int Unroll, int MultimemSrcs, int MultimemDsts>
struct ReduceScatterRingCodec<
  __nv_bfloat16, FuncSum<__nv_bfloat16>,
  ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>> {
  static constexpr bool IsEncoded = true;
  template <typename Prims> __device__ __forceinline__ static void encodeSend(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, int eltN) {
    ZipCclRingProcess<Prims, 0> fn{&prims, work, inpIx, 0, eltN};
    prims.template process<0, 1, true>(fn);
  }
  template <typename Prims> __device__ __forceinline__ static void decodeReduceEncodeSend(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, int eltN) {
    ZipCclRingProcess<Prims, 1> fn{&prims, work, inpIx, 0, eltN};
    prims.template process<1, 1, true>(fn);
  }
  template <typename Prims> __device__ __forceinline__ static void decodeReduceCopy(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, intptr_t outIx, int eltN, bool /*postOp*/) {
    ZipCclRingProcess<Prims, 2> fn{&prims, work, inpIx, outIx, eltN};
    prims.template process<1, 0, true>(fn);
  }
};

#endif
