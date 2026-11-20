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
__device__ int codec(P& prims, int tid, int workers, const uint16_t* local, int n, float sigma, int outBase,
                   const RingFrameHeader* src, RingFrameHeader* dst, uint16_t* dense, ncclZipMoments* moments = nullptr) {
  const auto layout = FrameLayout(n, kRingFrameAlignment);
  EncodedInput input{};
  EncodedOutput output{};
  unsigned contributors = 1;
  int base = 0;
  if constexpr (Op == SegmentOp::Dre || Op == SegmentOp::Copy) {
    const auto* data = reinterpret_cast<const uint8_t*>(src + 1);
    input = {data, reinterpret_cast<const uint32_t*>(data + layout.bitmap[0]),
             reinterpret_cast<const uint32_t*>(data + layout.bitmap[1]),
             reinterpret_cast<const uint32_t*>(data + layout.bitmap[2]),
             reinterpret_cast<const uint32_t*>(data + layout.block_offsets),
             reinterpret_cast<const uint32_t*>(data + layout.warp_offsets), data + layout.outliers};
    base = src->base;
    if constexpr (Op == SegmentOp::Dre) contributors = src->contributors + 1;
  }
  if (Op != SegmentOp::Copy) {
    auto* data = reinterpret_cast<uint8_t*>(dst + 1);
    output = {data, reinterpret_cast<uint32_t*>(data + layout.bitmap[0]),
              reinterpret_cast<uint32_t*>(data + layout.bitmap[1]),
              reinterpret_cast<uint32_t*>(data + layout.bitmap[2]),
              reinterpret_cast<uint32_t*>(data + layout.block_offsets),
              reinterpret_cast<uint32_t*>(data + layout.warp_offsets), data + layout.outliers};
  }
  int warps = min(workers / 32, ncclZipWorkerWarps);
  auto* shared = static_cast<uint8_t*>(prims.codecScratch());
  if constexpr (Op == SegmentOp::Copy) {
    // Copy only needs private warp scratch; there is no output to compact.
    int warp = tid / 32;
    if (warp < warps) {
      unsigned full = unsigned(n) / kDreValuesPerWarp;
      unsigned segment = warp;
      for (; segment < full; segment += warps)
        ProcessSegment<true, Op>(input, base, local, n, outBase, output, segment, nullptr, nullptr, shared, dense);
      if (segment == full && n % kDreValuesPerWarp)
        ProcessSegment<false, Op>(input, base, local, n, outBase, output, segment, nullptr, nullptr, shared, dense);
    }
    return 0;
  }
  constexpr int slotBytes = ncclZipRawSliceBytes / sizeof(uint16_t);
  int compactWarps = min(warps, kWarpsPerCta);
  auto* comm = reinterpret_cast<const ncclKernelCommAndChannels*>(ncclShmem.args.comm);
  auto* slots = comm->zipOutlierSlots + size_t(ncclShmem.channelId) * RingWorkspaceBytes(slotBytes);
  auto* counts = reinterpret_cast<uint16_t*>(slots + slotBytes);
  unsigned total = 0;
  int warp = tid / 32;
  auto* warpScratch = shared + warp * kDreWarpSharedBytes;
  prims.syncCodecWorkers();
  if constexpr (Op == SegmentOp::Dre) {
    // A full compaction group contains sixteen complete logical blocks.
    constexpr int groupElements = kWarpsPerCta * zipccl::kElementsPerBlock;
    if (warps == ncclZipWorkerWarps && n % groupElements == 0) {
      if (warp < ncclZipWorkerWarps) {
        for (int segment = warp; segment < n / kDreValuesPerWarp; segment += ncclZipWorkerWarps)
          ProcessSegment<true, Op>(input, base, local, n, outBase, output, segment, counts + segment,
                                   slots + segment * kDreValuesPerWarp, shared, dense);
      }
      prims.syncCodecWorkers();
      if (warp < kWarpsPerCta) {
        for (unsigned first = 0; first < layout.blocks; first += kWarpsPerCta)
          total = CompactSingleCta<true, true>(counts + first * kSegmentsPerBlock,
                                         slots + first * zipccl::kElementsPerBlock,
                                         output, first, kWarpsPerCta, total, warpScratch);
      }
      if (tid == 0) {
        *dst = {unsigned(n), contributors, uint16_t(outBase), RingFrameFormat::Compressed, sigma};
        return sizeof(RingFrameHeader) + layout.outliers + total;
      }
      return 0;
    }
  }
  if (warp < warps) {
    for (unsigned segment = warp; segment < layout.blocks * kSegmentsPerBlock; segment += warps) {
      auto* slot = slots + segment * kDreValuesPerWarp;
      if (segment < unsigned(n) / kDreValuesPerWarp)
        ProcessSegment<true, Op>(input, base, local, n, outBase, output, segment, counts + segment, slot, shared, dense, moments);
      else
        ProcessSegment<false, Op>(input, base, local, n, outBase, output, segment, counts + segment, slot, shared, dense, moments);
    }
  }
  prims.syncCodecWorkers();
  if (warp < compactWarps) {
    for (unsigned first = 0; first < layout.blocks; first += compactWarps) {
      unsigned group = min(unsigned(compactWarps), unsigned(layout.blocks - first));
      auto* groupCounts = counts + first * kSegmentsPerBlock;
      auto* groupSlots = slots + first * zipccl::kElementsPerBlock;
      if (group == kWarpsPerCta)
        total = CompactSingleCta<true, true>(groupCounts, groupSlots, output, first, group, total, warpScratch);
      else
        total = CompactSingleCta<false, true>(groupCounts, groupSlots, output, first, group, total, warpScratch);
    }
  }
  if (Op != SegmentOp::Copy && tid == 0) {
    *dst = {unsigned(n), contributors, uint16_t(outBase), RingFrameFormat::Compressed, sigma};
    return sizeof(RingFrameHeader) + layout.outliers + total;
  }
  return 0;
}

// Publish a consistent aggregate of all completed StatEncode slices. Each slot has one writer.
template<class P>
__device__ float publishStats(P& prims, int tid, int workers, ncclZipStats& stats, unsigned slot,
                              ncclZipMoments local) {
  auto* partial = static_cast<ncclZipMoments*>(prims.codecScratch());
  prims.syncCodecWorkers();
  for (int d = 16; d; d /= 2) {
    ncclZipMoments other{__shfl_down_sync(~0u, local.count, d), __shfl_down_sync(~0u, local.mean, d),
                         __shfl_down_sync(~0u, local.m2, d)};
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
      ncclZipMoments pilot{};
      int samples = min(n, 32);
      for (int i = 0; i < samples; ++i)
        pilot.add(__bfloat162float(__ushort_as_bfloat16(local[size_t(i) * n / samples])));
      auto initial = (1ull << 32) | __float_as_uint(pilot.sigma());
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

template<SegmentOp Stage, class P>
__device__ int run(P& prims, int tid, int workers, const uint16_t* local, int n, float suppliedSigma,
                   const RingFrameHeader* src, RingFrameHeader* dst, uint16_t* dense) {
  unsigned hop = Stage == SegmentOp::Encode ? 0 : src->contributors;
  bool compressedInput = Stage != SegmentOp::Encode && src->format == RingFrameFormat::Compressed;
  float sigma = 0;
  if constexpr (Stage != SegmentOp::Copy) {
    auto& stats = reinterpret_cast<ncclKernelCommAndChannels*>(ncclShmem.args.comm)->zipStats;
    SliceEstimate estimate{suppliedSigma, ncclZipStatSlices};
    if (suppliedSigma < 0) estimate = sliceEstimate<Stage == SegmentOp::Encode>(prims, tid, stats, local, n);
    sigma = estimate.sigma;
    if constexpr (Stage == SegmentOp::Encode) {
      if (estimate.slot < ncclZipStatSlices) {
        ncclZipMoments moments{};
        int bytes;
        if (n >= ncclZipMinSliceBytes / sizeof(uint16_t)) {
          bytes = codec<SegmentOp::StatEncode>(prims, tid, workers, local, n, sigma, ncclZipWindow(sigma),
                                               nullptr, dst, nullptr, &moments);
        } else {
          auto* payload = reinterpret_cast<uint16_t*>(dst + 1);
          for (int i = tid; i < n; i += workers) {
            uint16_t value = local[i];
            payload[i] = value;
            moments.add(__bfloat162float(__ushort_as_bfloat16(value)));
          }
          if (tid == 0) *dst = {unsigned(n), 1, 0, RingFrameFormat::Raw, sigma};
          bytes = sizeof(RingFrameHeader) + n * sizeof(uint16_t);
        }
        float updated = publishStats(prims, tid, workers, stats, estimate.slot, moments);
        if (tid == 0) dst->sigma = updated;
        return bytes;
      }
    }
    if constexpr (Stage == SegmentOp::Dre) sigma = hypotf(src->sigma, sigma);
    if (Stage == SegmentOp::Encode ? n >= ncclZipMinSliceBytes / sizeof(uint16_t) : compressedInput) {
      int base = ncclZipWindow(sigma);
      return codec<Stage>(prims, tid, workers, local, n, sigma, base, src, dst, nullptr);
    }
    dense = reinterpret_cast<uint16_t*>(dst + 1);
  }
  if (compressedInput) {
    codec<SegmentOp::Copy>(prims, tid, workers, local, n, 0, 0, src, nullptr, dense);
  } else {
    for (int i = tid; i < n; i += workers) {
      uint16_t value = local[i];
      if constexpr (Stage != SegmentOp::Encode) {
        const auto* incoming = reinterpret_cast<const uint16_t*>(src + 1);
        value = __bfloat16_as_ushort(__hadd(__ushort_as_bfloat16(incoming[i]), __ushort_as_bfloat16(value)));
      }
      dense[i] = value;
    }
  }
  if (Stage != SegmentOp::Copy && tid == 0) {
    *dst = {unsigned(n), hop + 1, 0, RingFrameFormat::Raw, sigma};
    return sizeof(RingFrameHeader) + n * sizeof(uint16_t);
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
