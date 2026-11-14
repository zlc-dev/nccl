/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_ZIPCCL_RING_CODEC_H_
#define NCCL_ZIPCCL_RING_CODEC_H_

// Each Simple-protocol slice is a self-contained frame. Only the first rank
// scans the data; intermediate ranks update the distribution analytically.
struct alignas(16) ZipCclRingFrameHeader {
  static constexpr uint32_t Magic = 0x5a43434c; // "ZCCL"
  static constexpr uint16_t Version = 5;
  uint32_t magic;
  uint16_t version;
  uint16_t headerBytes;
  uint32_t elementCount;
  uint32_t frameBytes;
  int32_t exponentBase;
  uint32_t groupCount;
  uint32_t outlierCount;
  uint32_t contributorCount;
  float mean;
  float variance;
  uint32_t reserved[2];
};
static_assert(sizeof(ZipCclRingFrameHeader) == 48, "ZipCCL wire header size changed");

template <typename T, typename RedOp, int Unroll>
struct ZipCclRingAlgorithm;

template <int Unroll>
struct ZipCclRingAlgorithm<__nv_bfloat16, FuncSum<__nv_bfloat16>, Unroll> {
  using T = __nv_bfloat16;
  static constexpr int Pack = 8;
  static constexpr int PacksPerGroup = WARP_SIZE;
  static constexpr int ElementsPerGroup = Pack * PacksPerGroup;
  static constexpr int MaxWorkerWarps = NCCL_SIMPLE_MAX_NTHREADS / WARP_SIZE;

  struct Scratch {
    float warpSums[MaxWorkerWarps];
    float warpSumSqs[MaxWorkerWarps];
    float sum;
    float sumSq;
    int exponentBase;
    int outlierCount;
    int frameBytes;
    int valid;
  };
  static_assert(sizeof(Scratch) <= ncclShmemScratchWarpSize(), "ZipCCL scratch exceeds one NCCL warp slot");

  struct Statistics { float mean, variance; };
  struct FrameView {
    const uint8_t* frame;
    const uint8_t* signMantissa;
    const uint8_t* bitmap0;
    const uint8_t* bitmap1;
    const uint8_t* bitmap2;
    const uint32_t* groupOffsets;
    const uint8_t* outlierExponents;
  };
  struct Pack8Registers { uint32_t pair[4]; };
  struct CodeBitmaps { uint8_t b0, b1, b2; };

  __device__ __forceinline__ static int alignUp(int value, int alignment) {
    return (value + alignment - 1) & -alignment;
  }
  __device__ __forceinline__ static int packCount(int n) { return (n + Pack - 1) / Pack; }
  __device__ __forceinline__ static int groupCount(int n) { return (n + ElementsPerGroup - 1) / ElementsPerGroup; }
  __device__ __forceinline__ static int groupOffsetOffset(int n) {
    return alignUp(sizeof(ZipCclRingFrameHeader) + n + packCount(n) * 3, int(alignof(uint32_t)));
  }
  __device__ __forceinline__ static int outlierOffset(int n) {
    return groupOffsetOffset(n) + groupCount(n) * int(sizeof(uint32_t));
  }
  __device__ __forceinline__ static int worstFrameBytes(int n) {
    return alignUp(outlierOffset(n) + n, int(sizeof(T)));
  }
  __device__ __forceinline__ static bool isAligned16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0;
  }
  __device__ __forceinline__ static uint8_t loadU8(const uint8_t* ptr) {
    uint16_t v;
    asm volatile("ld.volatile.global.u8 %0, [%1];" : "=h"(v) : "l"(cvta_to_global(ptr)) : "memory");
    return uint8_t(v);
  }
  __device__ __forceinline__ static uint32_t loadU32(const uint32_t* ptr) {
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(cvta_to_global(ptr)) : "memory");
    return v;
  }
  __device__ __forceinline__ static float loadFloat(const float* ptr) {
    return __uint_as_float(loadU32(reinterpret_cast<const uint32_t*>(ptr)));
  }

  __device__ __forceinline__ static void loadBf16Pack(const T* src, int count, Pack8Registers& pack) {
    if (count == Pack && isAligned16(src)) {
      uint4 v = *reinterpret_cast<const uint4*>(src);
      pack.pair[0] = v.x; pack.pair[1] = v.y; pack.pair[2] = v.z; pack.pair[3] = v.w;
    } else {
#pragma unroll
      for (int p = 0; p < 4; p++) pack.pair[p] = 0;
#pragma unroll
      for (int i = 0; i < Pack; i++)
        if (i < count) pack.pair[i / 2] |= uint32_t(__bfloat16_as_ushort(src[i])) << (16 * (i & 1));
    }
  }
  __device__ __forceinline__ static void storeBf16Pack(T* dst, int count, const Pack8Registers& pack) {
    if (count == Pack && isAligned16(dst)) {
      *reinterpret_cast<uint4*>(dst) = make_uint4(pack.pair[0], pack.pair[1], pack.pair[2], pack.pair[3]);
    } else {
#pragma unroll
      for (int i = 0; i < Pack; i++)
        if (i < count) dst[i] = __ushort_as_bfloat16(uint16_t(pack.pair[i / 2] >> (16 * (i & 1))));
    }
  }
  __device__ __forceinline__ static void loadSignMantissa(const uint8_t* src, int count, uint32_t sm[2]) {
    sm[0] = sm[1] = 0;
    if (count == Pack && (reinterpret_cast<uintptr_t>(src) & 3u) == 0) {
      sm[0] = loadU32(reinterpret_cast<const uint32_t*>(src));
      sm[1] = loadU32(reinterpret_cast<const uint32_t*>(src + 4));
    } else {
#pragma unroll
      for (int i = 0; i < Pack; i++) if (i < count) sm[i / 4] |= uint32_t(loadU8(src + i)) << (8 * (i & 3));
    }
  }
  __device__ __forceinline__ static void storeSignMantissa(uint8_t* dst, int count, const uint32_t sm[2]) {
    if (count == Pack && (reinterpret_cast<uintptr_t>(dst) & 3u) == 0) {
      *reinterpret_cast<uint32_t*>(dst) = sm[0];
      *reinterpret_cast<uint32_t*>(dst + 4) = sm[1];
    } else {
#pragma unroll
      for (int i = 0; i < Pack; i++) if (i < count) dst[i] = uint8_t(sm[i / 4] >> (8 * (i & 3)));
    }
  }
  __device__ __forceinline__ static uint16_t pairElement(uint32_t pair, int i) {
    return uint16_t(pair >> (16 * i));
  }
  __device__ __forceinline__ static uint8_t smElement(const uint32_t sm[2], int i) {
    return uint8_t(sm[i / 4] >> (8 * (i & 3)));
  }
  __device__ __forceinline__ static void setSmElement(uint32_t sm[2], int i, uint16_t raw) {
    uint8_t v = uint8_t(((raw >> 8) & 0x80u) | (raw & 0x7fu));
    sm[i / 4] |= uint32_t(v) << (8 * (i & 3));
  }
  __device__ __forceinline__ static uint32_t addBf16Pair(uint32_t lhs, uint32_t rhs) {
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&lhs);
    __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&rhs);
    __nv_bfloat162 result = __hadd2(a, b);
    return *reinterpret_cast<uint32_t*>(&result);
  }
  __device__ __forceinline__ static int warpExclusivePrefix(int value) {
    int inclusive = value;
#pragma unroll
    for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
      int preceding = __shfl_up_sync(0xffffffffu, inclusive, delta);
      if ((threadIdx.x & (WARP_SIZE - 1)) >= delta) inclusive += preceding;
    }
    return inclusive - value;
  }
  __device__ __forceinline__ static int allocateOutliers(
      Scratch* scratch, int lane, int localCount, uint32_t* groupOffset) {
    int prefix = warpExclusivePrefix(localCount);
    int total = __shfl_sync(0xffffffffu, prefix + localCount, WARP_SIZE - 1);
    int base = lane == 0 ? atomicAdd(&scratch->outlierCount, total) : 0;
    base = __shfl_sync(0xffffffffu, base, 0);
    if (lane == 0) *groupOffset = uint32_t(base);
    return base + prefix;
  }
  __device__ __forceinline__ static FrameView frameView(const void* encoded, int n) {
    FrameView v;
    const uint8_t* bytes = static_cast<const uint8_t*>(encoded);
    v.frame = bytes;
    v.signMantissa = bytes + sizeof(ZipCclRingFrameHeader);
    int packs = packCount(n);
    v.bitmap0 = v.signMantissa + n;
    v.bitmap1 = v.bitmap0 + packs;
    v.bitmap2 = v.bitmap1 + packs;
    v.groupOffsets = reinterpret_cast<const uint32_t*>(bytes + groupOffsetOffset(n));
    v.outlierExponents = bytes + outlierOffset(n);
    return v;
  }

  template <typename Prims>
  __device__ __forceinline__ static Statistics calculateStatistics(
      int tid, int nworkers, const T* input, int n, uint8_t* signMantissa, Prims& prims) {
    Scratch* scratch = reinterpret_cast<Scratch*>(prims.codecScratch());
    float localSum = 0.0f, localSumSq = 0.0f;
    for (int begin = tid * Pack; begin < n; begin += nworkers * Pack) {
      int count = min(Pack, n - begin);
      Pack8Registers pack;
      loadBf16Pack(input + begin, count, pack);
      uint32_t sm[2] = {};
#pragma unroll
      for (int i = 0; i < Pack; i++) if (i < count) {
        uint16_t raw = pairElement(pack.pair[i / 2], i & 1);
        float value = __bfloat162float(__ushort_as_bfloat16(raw));
        localSum += value;
        localSumSq = fmaf(value, value, localSumSq);
        setSmElement(sm, i, raw);
      }
      storeSignMantissa(signMantissa + begin, count, sm);
    }
#pragma unroll
    for (int delta = WARP_SIZE / 2; delta != 0; delta >>= 1) {
      localSum += __shfl_down_sync(0xffffffffu, localSum, delta);
      localSumSq += __shfl_down_sync(0xffffffffu, localSumSq, delta);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
      scratch->warpSums[tid / WARP_SIZE] = localSum;
      scratch->warpSumSqs[tid / WARP_SIZE] = localSumSq;
    }
    prims.syncCodecWorkers();
    if (tid == 0) {
      float sum = 0.0f, sumSq = 0.0f;
      for (int warp = 0; warp < (nworkers + WARP_SIZE - 1) / WARP_SIZE; warp++) {
        sum += scratch->warpSums[warp];
        sumSq += scratch->warpSumSqs[warp];
      }
      scratch->sum = sum;
      scratch->sumSq = sumSq;
    }
    prims.syncCodecWorkers();
    Statistics stats;
    float inverseCount = n == 0 ? 0.0f : 1.0f / float(n);
    stats.mean = scratch->sum * inverseCount;
    stats.variance = fmaxf(0.0f, scratch->sumSq * inverseCount - stats.mean * stats.mean);
    return stats;
  }
  __device__ __forceinline__ static int selectExponentBase(Statistics stats) {
    if (isfinite(stats.variance) && stats.variance > 0.0f) {
      constexpr float OptimalBaseOffset = -5.3606616965413565f;
      float optimum = 0.5f * log2f(stats.variance) + OptimalBaseOffset;
      int unbiased = max(-127, min(122, __float2int_rn(optimum)));
      return unbiased + 127;
    }
    if (isfinite(stats.mean) && stats.mean != 0.0f) {
      uint16_t raw = __bfloat16_as_ushort(__float2bfloat16(fabsf(stats.mean)));
      return min(249, int((raw >> 7) & 0xffu));
    }
    return 0;
  }
  __device__ __forceinline__ static uint16_t decodeElement(
      const FrameView& input, int bit, CodeBitmaps bitmaps, int exponentBase,
      const uint32_t sm[2], int& outlierCursor) {
    uint32_t code = ((bitmaps.b0 >> bit) & 1u) |
                    (((bitmaps.b1 >> bit) & 1u) << 1) |
                    (((bitmaps.b2 >> bit) & 1u) << 2);
    int exponent;
    if (code == 0) {
      exponent = int(loadU8(input.outlierExponents + outlierCursor));
      outlierCursor++;
    } else {
      exponent = exponentBase + int(code) - 1;
    }
    uint16_t signMantissa = smElement(sm, bit);
    return uint16_t((signMantissa & 0x80u) << 8) | uint16_t((exponent & 0xff) << 7) |
           uint16_t(signMantissa & 0x7fu);
  }
  template <bool StoreSignMantissa>
  __device__ __forceinline__ static int encodeRegisters(
      const Pack8Registers& values, int count, int exponentBase, uint32_t* sm, CodeBitmaps& bitmaps,
      uint8_t outlierExponents[Pack]) {
    if (StoreSignMantissa) sm[0] = sm[1] = 0;
    bitmaps.b0 = bitmaps.b1 = bitmaps.b2 = 0;
    int outliers = 0;
#pragma unroll
    for (int i = 0; i < Pack; i++) if (i < count) {
      uint16_t raw = pairElement(values.pair[i / 2], i & 1);
      uint8_t exponent = uint8_t((raw >> 7) & 0xffu);
      if (StoreSignMantissa) setSmElement(sm, i, raw);
      uint32_t code = exponent >= exponentBase && exponent < exponentBase + 7
                        ? uint32_t(exponent - exponentBase + 1) : 0u;
      bitmaps.b0 |= uint8_t((code & 1u) << i);
      bitmaps.b1 |= uint8_t(((code >> 1) & 1u) << i);
      bitmaps.b2 |= uint8_t(((code >> 2) & 1u) << i);
      if (code == 0) outlierExponents[outliers++] = exponent;
    }
    return outliers;
  }
  template <typename Prims>
  __device__ __forceinline__ static int finishFrame(
      Prims& prims, Scratch* scratch, int tid, int n, void* encoded,
      Statistics stats, uint32_t contributors) {
    prims.syncCodecWorkers();
    if (tid == 0) {
      int bytes = alignUp(outlierOffset(n) + scratch->outlierCount, int(sizeof(T)));
      ZipCclRingFrameHeader* h = reinterpret_cast<ZipCclRingFrameHeader*>(encoded);
      h->magic = ZipCclRingFrameHeader::Magic; h->version = ZipCclRingFrameHeader::Version;
      h->headerBytes = sizeof(ZipCclRingFrameHeader); h->elementCount = n; h->frameBytes = bytes;
      h->exponentBase = scratch->exponentBase; h->groupCount = groupCount(n);
      h->outlierCount = scratch->outlierCount; h->contributorCount = contributors;
      h->mean = stats.mean; h->variance = stats.variance;
      h->reserved[0] = h->reserved[1] = 0; scratch->frameBytes = bytes;
    }
    // process() executes a full CTA barrier immediately after the callback;
    // only worker 0 consumes the returned size before that barrier.
    return tid == 0 ? scratch->frameBytes : 0;
  }

  template <typename Prims>
  __device__ __forceinline__ static int encode(
      Prims& prims, int tid, int nworkers, const T* input, int n, void* encoded, int capacity) {
    Scratch* scratch = reinterpret_cast<Scratch*>(prims.codecScratch());
    if (tid == 0) { scratch->outlierCount = 0; scratch->frameBytes = 0; scratch->valid = worstFrameBytes(n) <= capacity; }
    prims.syncCodecWorkers();
    if (!scratch->valid) return 0;
    uint8_t* bytes = static_cast<uint8_t*>(encoded);
    uint8_t* smStream = bytes + sizeof(ZipCclRingFrameHeader);
    int packs = packCount(n);
    uint8_t* bitmap0 = smStream + n;
    uint8_t* bitmap1 = bitmap0 + packs;
    uint8_t* bitmap2 = bitmap1 + packs;
    uint32_t* offsets = reinterpret_cast<uint32_t*>(bytes + groupOffsetOffset(n));
    uint8_t* outlierStream = bytes + outlierOffset(n);

    Statistics stats = calculateStatistics(tid, nworkers, input, n, smStream, prims);
    if (tid == 0) scratch->exponentBase = selectExponentBase(stats);
    prims.syncCodecWorkers();

    int warp = tid / WARP_SIZE, lane = tid & (WARP_SIZE - 1), workerWarps = nworkers / WARP_SIZE;
    for (int group = warp; group < groupCount(n); group += workerWarps) {
      int pack = group * PacksPerGroup + lane, begin = pack * Pack;
      int count = min(Pack, max(0, n - begin));
      Pack8Registers values = {};
      if (count) loadBf16Pack(input + begin, count, values);
      CodeBitmaps bitmaps; uint8_t outliers[Pack];
      int outlierN = encodeRegisters<false>(
        values, count, scratch->exponentBase, nullptr, bitmaps, outliers);
      int out = allocateOutliers(scratch, lane, outlierN, offsets + group);
      if (count) {
        bitmap0[pack] = bitmaps.b0;
        bitmap1[pack] = bitmaps.b1;
        bitmap2[pack] = bitmaps.b2;
        for (int i = 0; i < outlierN; i++) outlierStream[out + i] = outliers[i];
      }
    }
    return finishFrame(prims, scratch, tid, n, encoded, stats, 1);
  }

  template <typename Prims>
  __device__ __forceinline__ static int decodeReduceEncode(
      Prims& prims, int tid, int nworkers, const void* encodedInput, const T* localInput, int n,
      void* encodedOutput, int capacity) {
    FrameView input = frameView(encodedInput, n);
    const ZipCclRingFrameHeader* header = reinterpret_cast<const ZipCclRingFrameHeader*>(input.frame);
    uint32_t contributors = max(1u, loadU32(&header->contributorCount));
    float growth = float(contributors + 1) / float(contributors);
    Statistics combined = {loadFloat(&header->mean) * growth,
                           loadFloat(&header->variance) * growth};
    Scratch* scratch = reinterpret_cast<Scratch*>(prims.codecScratch());
    if (tid == 0) {
      scratch->outlierCount = 0; scratch->frameBytes = 0; scratch->valid = worstFrameBytes(n) <= capacity;
      scratch->exponentBase = selectExponentBase(combined);
    }
    prims.syncCodecWorkers();
    if (!scratch->valid) return 0;

    uint8_t* bytes = static_cast<uint8_t*>(encodedOutput);
    uint8_t* smStream = bytes + sizeof(ZipCclRingFrameHeader);
    int packs = packCount(n);
    uint8_t* bitmap0 = smStream + n;
    uint8_t* bitmap1 = bitmap0 + packs;
    uint8_t* bitmap2 = bitmap1 + packs;
    uint32_t* offsets = reinterpret_cast<uint32_t*>(bytes + groupOffsetOffset(n));
    uint8_t* outlierStream = bytes + outlierOffset(n);
    int inputBase = int(loadU32(reinterpret_cast<const uint32_t*>(input.frame + offsetof(ZipCclRingFrameHeader, exponentBase))));
    int warp = tid / WARP_SIZE, lane = tid & (WARP_SIZE - 1), workerWarps = nworkers / WARP_SIZE;
    for (int group = warp; group < groupCount(n); group += workerWarps) {
      int pack = group * PacksPerGroup + lane, begin = pack * Pack;
      int count = min(Pack, max(0, n - begin));
      CodeBitmaps inBitmaps = {count ? loadU8(input.bitmap0 + pack) : uint8_t(0),
                              count ? loadU8(input.bitmap1 + pack) : uint8_t(0),
                              count ? loadU8(input.bitmap2 + pack) : uint8_t(0)};
      uint8_t validMask = count == Pack ? uint8_t(0xffu) : uint8_t((1u << count) - 1u);
      uint8_t inOutlierMask = uint8_t(~(inBitmaps.b0 | inBitmaps.b1 | inBitmaps.b2)) & validMask;
      int inOutliers = __popc(uint32_t(inOutlierMask));
      int inPrefix = warpExclusivePrefix(inOutliers);
      int inGroupBase = int(loadU32(input.groupOffsets + group));
      int outlierCursor = inGroupBase + inPrefix;
      uint32_t inSm[2] = {}; Pack8Registers local = {}, result = {};
      if (count) {
        loadSignMantissa(input.signMantissa + begin, count, inSm); loadBf16Pack(localInput + begin, count, local);
#pragma unroll
        for (int pair = 0; pair < 4; pair++) {
          uint16_t lo = decodeElement(input, pair * 2, inBitmaps, inputBase, inSm, outlierCursor);
          uint16_t hi = pair * 2 + 1 < count
                          ? decodeElement(input, pair * 2 + 1, inBitmaps, inputBase, inSm, outlierCursor) : 0;
          result.pair[pair] = addBf16Pair(uint32_t(lo) | (uint32_t(hi) << 16), local.pair[pair]);
        }
      }
      uint32_t outSm[2]; CodeBitmaps outBitmaps; uint8_t outliers[Pack];
      int outlierN = encodeRegisters<true>(
        result, count, scratch->exponentBase, outSm, outBitmaps, outliers);
      int out = allocateOutliers(scratch, lane, outlierN, offsets + group);
      if (count) {
        storeSignMantissa(smStream + begin, count, outSm);
        bitmap0[pack] = outBitmaps.b0;
        bitmap1[pack] = outBitmaps.b1;
        bitmap2[pack] = outBitmaps.b2;
        for (int i = 0; i < outlierN; i++) outlierStream[out + i] = outliers[i];
      }
    }
    return finishFrame(prims, scratch, tid, n, encodedOutput, combined, contributors + 1);
  }

  template <typename Prims>
  __device__ __forceinline__ static void decodeReduceCopy(
      Prims& /*prims*/, int tid, int nworkers, const void* encodedInput, const T* localInput, int n,
      T* output, bool /*postOp*/) {
    FrameView input = frameView(encodedInput, n);
    int inputBase = int(loadU32(reinterpret_cast<const uint32_t*>(input.frame + offsetof(ZipCclRingFrameHeader, exponentBase))));
    int warp = tid / WARP_SIZE, lane = tid & (WARP_SIZE - 1), workerWarps = nworkers / WARP_SIZE;
    for (int group = warp; group < groupCount(n); group += workerWarps) {
      int pack = group * PacksPerGroup + lane, begin = pack * Pack;
      int count = min(Pack, max(0, n - begin));
      CodeBitmaps bitmaps = {count ? loadU8(input.bitmap0 + pack) : uint8_t(0),
                            count ? loadU8(input.bitmap1 + pack) : uint8_t(0),
                            count ? loadU8(input.bitmap2 + pack) : uint8_t(0)};
      uint8_t validMask = count == Pack ? uint8_t(0xffu) : uint8_t((1u << count) - 1u);
      uint8_t inOutlierMask = uint8_t(~(bitmaps.b0 | bitmaps.b1 | bitmaps.b2)) & validMask;
      int inOutliers = __popc(uint32_t(inOutlierMask));
      int inPrefix = warpExclusivePrefix(inOutliers);
      int inGroupBase = int(loadU32(input.groupOffsets + group));
      int outlierCursor = inGroupBase + inPrefix;
      if (count) {
        uint32_t sm[2]; Pack8Registers local, result = {};
        loadSignMantissa(input.signMantissa + begin, count, sm); loadBf16Pack(localInput + begin, count, local);
#pragma unroll
        for (int pair = 0; pair < 4; pair++) {
          uint16_t lo = decodeElement(input, pair * 2, bitmaps, inputBase, sm, outlierCursor);
          uint16_t hi = pair * 2 + 1 < count
                          ? decodeElement(input, pair * 2 + 1, bitmaps, inputBase, sm, outlierCursor) : 0;
          result.pair[pair] = addBf16Pair(uint32_t(lo) | (uint32_t(hi) << 16), local.pair[pair]);
        }
        storeBf16Pack(output + begin, count, result);
      }
    }
  }

  __device__ __forceinline__ static size_t maxChunkElements(size_t transportChunkElements, int nSlices) {
    size_t capacity = transportChunkElements * sizeof(T) / nSlices;
    // 153/64 bytes per element conservatively covers sign/mantissa, 3-bit
    // bitmap streams, worst-case outliers, and one uint32 offset per 256 elements.
    size_t n = capacity > sizeof(ZipCclRingFrameHeader) + 32
                 ? (capacity - sizeof(ZipCclRingFrameHeader) - 32) * 64 / 153 : 0;
    while (n && size_t(worstFrameBytes(int(n))) > capacity) n--;
    return n * nSlices;
  }
};

template <typename T, typename RedOp, int Unroll, typename Prims, int Stage>
struct ZipCclRingProcess {
  using Algorithm = ZipCclRingAlgorithm<T, RedOp, Unroll>;
  Prims* prims; struct ncclDevWorkColl* work; intptr_t inpIx, outIx; int eltN; bool postOp;
  template <int Slices, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
  __device__ __forceinline__ void operator()(
      int tid, int nworkers, int slice, int maxSliceElts, int nSrcs, void** srcPtrs,
      int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t, uint32_t) {
    static_assert(MultimemSrcs == 0 && MultimemDsts == 0, "ZipCCL does not support multimem");
    int sliceBeg = int((int64_t(eltN) * slice) / Slices);
    int sliceEnd = int((int64_t(eltN) * (slice + 1)) / Slices);
    int sliceN = sliceEnd - sliceBeg, payloadBytes = 0;
    const T* local = static_cast<const T*>(work->sendbuff) + inpIx + sliceBeg;
    if (Stage == 0 && nDsts)
      payloadBytes = Algorithm::encode(*prims, tid, nworkers, local, sliceN, dstPtrs[0], maxSliceElts * int(sizeof(T)));
    else if (Stage == 1 && nSrcs && nDsts)
      payloadBytes = Algorithm::decodeReduceEncode(
        *prims, tid, nworkers, srcPtrs[0], local, sliceN, dstPtrs[0], maxSliceElts * int(sizeof(T)));
    else if (Stage == 2 && nSrcs) {
      T* output = static_cast<T*>(work->recvbuff) + outIx + sliceBeg;
      Algorithm::decodeReduceCopy(*prims, tid, nworkers, srcPtrs[0], local, sliceN, output, postOp);
    }
    if (Stage != 2 && tid == 0 && nDsts) dstSizes[0] = (payloadBytes + int(sizeof(T)) - 1) / int(sizeof(T));
  }
};

template <int SlicePerChunk, int StepPerSlice, int Unroll, int MultimemSrcs, int MultimemDsts>
struct ReduceScatterRingCodec<
  __nv_bfloat16, FuncSum<__nv_bfloat16>,
  ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>> {
  using T = __nv_bfloat16; using RedOp = FuncSum<T>; using Algorithm = ZipCclRingAlgorithm<T, RedOp, Unroll>;
  static constexpr bool IsEncoded = true;
  __device__ __forceinline__ static size_t chunkElementLimit(size_t scheduled) {
    size_t limit = Algorithm::maxChunkElements(scheduled, SlicePerChunk);
    return limit == 0 ? 1 : min(limit, scheduled);
  }
  template <typename Prims> __device__ __forceinline__ static void encodeSend(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, int eltN) {
    ZipCclRingProcess<T, RedOp, Unroll, Prims, 0> fn{&prims, work, inpIx, 0, eltN, false};
    prims.template process<0, 1>(fn);
  }
  template <typename Prims> __device__ __forceinline__ static void decodeReduceEncodeSend(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, int eltN) {
    ZipCclRingProcess<T, RedOp, Unroll, Prims, 1> fn{&prims, work, inpIx, 0, eltN, false};
    prims.template process<1, 1>(fn);
  }
  template <typename Prims> __device__ __forceinline__ static void decodeReduceCopy(
      Prims& prims, struct ncclDevWorkColl* work, intptr_t inpIx, intptr_t outIx, int eltN, bool postOp) {
    ZipCclRingProcess<T, RedOp, Unroll, Prims, 2> fn{&prims, work, inpIx, outIx, eltN, postOp};
    prims.template process<1, 0>(fn);
  }
};

// A future pre-compression kernel may feed Stage 0 through an alternate source
// while retaining work->sendbuff for local reductions. It must stay on NCCL's
// stream and preserve the same per-slice frame and FIFO-capacity contract.

#endif // NCCL_ZIPCCL_RING_CODEC_H_
