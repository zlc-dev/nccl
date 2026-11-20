/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_ZIPCCL_RING_CODEC_H_
#define NCCL_ZIPCCL_RING_CODEC_H_

#include "../../zipccl/single_cta.cuh"

namespace ncclZipRing {
using namespace zipccl::device;

template<SegmentOp Op, class P>
__device__ int run(P& prims, int tid, int workers, const uint16_t* local, int n, float suppliedSigma,
                   const RingFrameHeader* src, RingFrameHeader* dst, uint16_t* dense) {
  bool raw = Op == SegmentOp::Encode ? n < ncclZipMinSliceBytes / sizeof(uint16_t)
                                     : src->format == RingFrameFormat::Raw;
  if (raw) {
    const auto* incoming = Op == SegmentOp::Encode ? nullptr : reinterpret_cast<const uint16_t*>(src + 1);
    auto* output = Op == SegmentOp::Copy ? dense : reinterpret_cast<uint16_t*>(dst + 1);
    for (int i = tid; i < n; i += workers) {
      uint16_t value = local[i];
      if (Op != SegmentOp::Encode)
        value = __bfloat16_as_ushort(__hadd(__ushort_as_bfloat16(incoming[i]), __ushort_as_bfloat16(value)));
      output[i] = value;
    }
    if (Op != SegmentOp::Copy && tid == 0) {
      unsigned count = Op == SegmentOp::Encode ? 1 : src->contributors + 1;
      *dst = {unsigned(n), count, 0, RingFrameFormat::Raw, 0.0f};
      return sizeof(RingFrameHeader) + n * sizeof(uint16_t);
    }
    return 0;
  }
  const auto layout = FrameLayout(n);
  EncodedInput input{};
  EncodedOutput output{};
  unsigned contributors = 1;
  const auto& stats = reinterpret_cast<const ncclKernelCommAndChannels*>(ncclShmem.args.comm)->zipStats;
  const float localSigma = suppliedSigma >= 0 ? suppliedSigma : stats.sigma;
  float sigma = 0;
  int base = 0;
  if (Op != SegmentOp::Encode) {
    const auto* data = reinterpret_cast<const uint8_t*>(src + 1);
    const auto* words = reinterpret_cast<const uint32_t*>(data + layout.bitmap[0]);
    input = {data, words, words + layout.bitmap_words, words + 2*layout.bitmap_words,
             reinterpret_cast<const uint32_t*>(data + layout.block_offsets),
             reinterpret_cast<const uint32_t*>(data + layout.warp_offsets), data + layout.outliers};
    base = src->base;
    contributors = src->contributors + 1;
    if (Op == SegmentOp::Dre) sigma = hypotf(src->sigma, localSigma);
  } else {
    sigma = localSigma;
  }
  if (Op != SegmentOp::Copy) {
    auto* data = reinterpret_cast<uint8_t*>(dst + 1);
    auto* words = reinterpret_cast<uint32_t*>(data + layout.bitmap[0]);
    output = {data, words, words + layout.bitmap_words, words + 2*layout.bitmap_words,
              reinterpret_cast<uint32_t*>(data + layout.block_offsets),
              reinterpret_cast<uint32_t*>(data + layout.warp_offsets), data + layout.outliers};
  }
  const int outBase = Op == SegmentOp::Encode ? (suppliedSigma >= 0 ? ncclZipWindow(localSigma) : stats.base) : Op == SegmentOp::Dre ? ncclZipWindow(sigma) : 0;
  // process() reserves control threads. Size each batch for the participating
  // warps and the existing NCCL CTA scratch allocation.
  constexpr int capacity = ncclShmemScratchWarpSize() * (NCCL_MAX_NTHREADS / 32);
  auto* shared = static_cast<uint8_t*>(prims.codecScratch());
  if constexpr (Op == SegmentOp::Copy) {
    // Copy only needs private warp scratch; there is no output to compact.
    int copyWarps = min(workers / 32, int(capacity / kDreWarpSharedBytes));
    int warp = tid / 32;
    if (warp < copyWarps) {
      unsigned full = unsigned(n) / kDreValuesPerWarp;
      unsigned segment = warp;
      for (; segment < full; segment += copyWarps)
        ProcessSegment<true, Op>(input, base, local, n, outBase, output, segment, nullptr, nullptr, shared, dense);
      if (segment == full && n % kDreValuesPerWarp)
        ProcessSegment<false, Op>(input, base, local, n, outBase, output, segment, nullptr, nullptr, shared, dense);
    }
    return 0;
  }
  constexpr int slotBytes = ncclZipRawSliceBytes / sizeof(uint16_t);
  constexpr int countBytesPerBlock = kSegmentsPerBlock * sizeof(uint16_t);
  constexpr int warpLimit = (capacity - countBytesPerBlock) / kDreWarpSharedBytes;
  static_assert(warpLimit >= 1, "ZipCCL requires warp scratch and segment counts");
  int warps = min(workers / 32, min(warpLimit, kWarpsPerCta));
  int blocks = min(slotBytes / int(zipccl::kElementsPerBlock),
                   (capacity - warps * kDreWarpSharedBytes) / countBytesPerBlock);
  auto* comm = reinterpret_cast<const ncclKernelCommAndChannels*>(ncclShmem.args.comm);
  auto* slots = comm->zipOutlierSlots + size_t(ncclShmem.channelId) * slotBytes;
  auto* counts = reinterpret_cast<uint16_t*>(shared + warps * kDreWarpSharedBytes);
  unsigned total = 0;
  int warp = tid / 32;
  prims.syncCodecWorkers();
  if constexpr (Op == SegmentOp::Dre) {
    // A full compaction group contains sixteen complete logical blocks.
    constexpr int groupElements = kWarpsPerCta * zipccl::kElementsPerBlock;
    if (warps == kWarpsPerCta && n % groupElements == 0 && layout.blocks <= unsigned(blocks)) {
      if (warp < kWarpsPerCta) {
        for (int segment = warp; segment < n / kDreValuesPerWarp; segment += kWarpsPerCta)
          ProcessSegment<true, Op>(input, base, local, n, outBase, output, segment, counts + segment,
                                   slots + segment * kDreValuesPerWarp, shared, dense);
      }
      prims.syncCodecWorkers();
      if (warp < kWarpsPerCta) {
        for (unsigned first = 0; first < layout.blocks; first += kWarpsPerCta)
          total = CompactSingleCta<true>(counts + first * kSegmentsPerBlock,
                                         slots + first * zipccl::kElementsPerBlock,
                                         output, first, kWarpsPerCta, total);
      }
      if (tid == 0) {
        *dst = {unsigned(n), contributors, uint16_t(outBase), RingFrameFormat::Compressed, sigma};
        return sizeof(RingFrameHeader) + layout.outliers + total;
      }
      return 0;
    }
  }
  for (unsigned first = 0; first < layout.blocks; first += blocks) {
    int batch = min(unsigned(blocks), unsigned(layout.blocks - first));
    if (warp < warps) {
      for (int i = warp; i < batch * kSegmentsPerBlock; i += warps) {
        unsigned segment = first * kSegmentsPerBlock + i;
        // Each warp owns its tile scratch. Slots and counts belong to segments.
        if (segment < unsigned(n) / kDreValuesPerWarp)
          ProcessSegment<true, Op>(input, base, local, n, outBase, output, segment,
                                      counts + i, slots + i * kDreValuesPerWarp, shared, dense);
        else
          ProcessSegment<false, Op>(input, base, local, n, outBase, output, segment,
                                       counts + i, slots + i * kDreValuesPerWarp, shared, dense);
      }
    }
    prims.syncCodecWorkers();
    if (warp < warps) {
      for (int i = 0; i < batch; i += warps) {
        int group = min(warps, batch - i);
        auto* groupCounts = counts + i * kSegmentsPerBlock;
        auto* groupSlots = slots + i * zipccl::kElementsPerBlock;
        if (group == kWarpsPerCta)
          total = CompactSingleCta<true>(groupCounts, groupSlots, output, first + i, group, total);
        else
          total = CompactSingleCta<false>(groupCounts, groupSlots, output, first + i, group, total);
      }
    }
    if (first + batch < layout.blocks) prims.syncCodecWorkers();
  }
  if (Op != SegmentOp::Copy && tid == 0) {
    *dst = {unsigned(n), contributors, uint16_t(outBase), RingFrameFormat::Compressed, sigma};
    return sizeof(RingFrameHeader) + layout.outliers + total;
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
  __device__ void operator()(int tid, int workers, int slice, int, int nSrcs, void** srcs,
                             int nDsts, void** dsts, int32_t* sizes, uint32_t, uint32_t) {
    using namespace ncclZipRing;
    int begin = int(int64_t(eltN)*slice/Slices);
    int n = int(int64_t(eltN)*(slice+1)/Slices) - begin;
    if ((Stage != 0 && !nSrcs) || (Stage != 2 && !nDsts)) return;
    const auto* local = static_cast<const uint16_t*>(work->sendbuff) + inpIx + begin;
    auto* dense = Stage == 2 ? static_cast<uint16_t*>(work->recvbuff) + outIx + begin : nullptr;
    constexpr SegmentOp op = Stage == 0 ? SegmentOp::Encode : Stage == 1 ? SegmentOp::Dre : SegmentOp::Copy;
    int bytes = run<op>(*prims, tid, workers, local, n, work->zipSigma, Stage ? static_cast<RingFrameHeader*>(srcs[0]) : nullptr,
                        Stage == 2 ? nullptr : static_cast<RingFrameHeader*>(dsts[0]), dense);
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
