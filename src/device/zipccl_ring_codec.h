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
  constexpr int perWarp = kDreWarpSharedBytes + kDreValuesPerWarp + 16;
  static_assert(capacity >= 4 * perWarp, "ZipCCL requires one logical block of scratch");
  int warps = min(min(workers / 32, capacity / perWarp), kWarpsPerCta);
  int blocks = min(warps, (capacity - warps*kDreWarpSharedBytes) /
                                    (kSegmentsPerBlock * (kDreValuesPerWarp + int(sizeof(uint16_t)))));
  auto* shared = static_cast<uint8_t*>(prims.codecScratch());
  auto* slots = shared + warps*kDreWarpSharedBytes;
  auto* counts = reinterpret_cast<uint16_t*>(slots + blocks * zipccl::kElementsPerBlock);
  unsigned total = 0;
  int warp = tid / 32;
  prims.syncCodecWorkers();
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
    if (Op != SegmentOp::Copy) {
      prims.syncCodecWorkers();
      if (warp < warps)
        total = CompactSingleCta<false>(counts, slots, output, first, batch, total);
      prims.syncCodecWorkers();
    }
  }
  if (Op != SegmentOp::Copy && tid == 0) {
    *dst = {unsigned(n), contributors, outBase, sigma};
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
    prims.template process<0, 1>(fn);
  }
  template <typename Prims> __device__ __forceinline__ static void decodeReduceEncodeSend(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, int eltN) {
    ZipCclRingProcess<Prims, 1> fn{&prims, work, inpIx, 0, eltN};
    prims.template process<1, 1>(fn);
  }
  template <typename Prims> __device__ __forceinline__ static void decodeReduceCopy(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, intptr_t outIx, int eltN, bool /*postOp*/) {
    ZipCclRingProcess<Prims, 2> fn{&prims, work, inpIx, outIx, eltN};
    prims.template process<1, 0>(fn);
  }
};

#endif
