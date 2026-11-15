/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#ifndef NCCL_ZIPCCL_RING_CODEC_H_
#define NCCL_ZIPCCL_RING_CODEC_H_

// Each Simple-protocol slice is a self-contained frame. The first rank scans
// and sends one slice at a time so the first payload can enter the ring without
// waiting for statistics over the complete channel chunk. Intermediate ranks
// update the distribution analytically.
struct alignas(16) ZipCclRingFrameHeader {
  static constexpr uint32_t Magic = 0x5a43434c; // "ZCCL"
  static constexpr uint16_t Version = 6;
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
  static constexpr int Pack = 32;
  static constexpr int PairsPerPack = Pack / 2;
  static constexpr int WordsPerSignMantissa = Pack / 4;
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
    const uint32_t* bitmap0;
    const uint32_t* bitmap1;
    const uint32_t* bitmap2;
    const uint32_t* groupOffsets;
    const uint8_t* outlierExponents;
  };
  struct PackRegisters { uint32_t pair[PairsPerPack]; };
  struct CodeBitmaps { uint32_t b0, b1, b2; };

  __device__ __forceinline__ static int alignUp(int value, int alignment) {
    return (value + alignment - 1) & -alignment;
  }
  __device__ __forceinline__ static int packCount(int n) { return (n + Pack - 1) / Pack; }
  __device__ __forceinline__ static int bitmapBytes(int n) {
    return packCount(n) * int(sizeof(uint32_t));
  }
  __device__ __forceinline__ static int bitmapOffset(int n) {
    return alignUp(sizeof(ZipCclRingFrameHeader) + n, int(alignof(uint32_t)));
  }
  __device__ __forceinline__ static int groupCount(int n) { return (n + ElementsPerGroup - 1) / ElementsPerGroup; }
  __device__ __forceinline__ static int payloadEndOffset(int n) {
    return bitmapOffset(n) + bitmapBytes(n) * 3;
  }
  __device__ __forceinline__ static int groupOffsetOffset(int n) {
    return payloadEndOffset(n);
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
  __device__ __forceinline__ static uint4 loadU128(const void* ptr) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(cvta_to_global(ptr)) : "memory");
    return v;
  }
  __device__ __forceinline__ static float loadFloat(const float* ptr) {
    return __uint_as_float(loadU32(reinterpret_cast<const uint32_t*>(ptr)));
  }

  __device__ __forceinline__ static void loadBf16Pack(const T* src, int count, PackRegisters& pack) {
    if (count == Pack && isAligned16(src)) {
      const uint4* input = reinterpret_cast<const uint4*>(src);
#pragma unroll
      for (int v = 0; v < Pack * int(sizeof(T)) / int(sizeof(uint4)); v++) {
        uint4 values = input[v];
        pack.pair[v * 4 + 0] = values.x; pack.pair[v * 4 + 1] = values.y;
        pack.pair[v * 4 + 2] = values.z; pack.pair[v * 4 + 3] = values.w;
      }
    } else {
#pragma unroll
      for (int p = 0; p < PairsPerPack; p++) pack.pair[p] = 0;
#pragma unroll
      for (int i = 0; i < Pack; i++)
        if (i < count) pack.pair[i / 2] |= uint32_t(__bfloat16_as_ushort(src[i])) << (16 * (i & 1));
    }
  }
  __device__ __forceinline__ static void storeBf16Pack(T* dst, int count, const PackRegisters& pack) {
    if (count == Pack && isAligned16(dst)) {
      uint4* output = reinterpret_cast<uint4*>(dst);
#pragma unroll
      for (int v = 0; v < Pack * int(sizeof(T)) / int(sizeof(uint4)); v++)
        output[v] = make_uint4(pack.pair[v * 4 + 0], pack.pair[v * 4 + 1],
                              pack.pair[v * 4 + 2], pack.pair[v * 4 + 3]);
    } else {
#pragma unroll
      for (int i = 0; i < Pack; i++)
        if (i < count) dst[i] = __ushort_as_bfloat16(uint16_t(pack.pair[i / 2] >> (16 * (i & 1))));
    }
  }
  __device__ __forceinline__ static void loadSignMantissa(
      const uint8_t* src, int count, uint32_t sm[WordsPerSignMantissa]) {
#pragma unroll
    for (int word = 0; word < WordsPerSignMantissa; word++) sm[word] = 0;
    if (count == Pack && isAligned16(src)) {
#pragma unroll
      for (int v = 0; v < Pack / int(sizeof(uint4)); v++) {
        uint4 values = loadU128(src + v * sizeof(uint4));
        sm[v * 4 + 0] = values.x; sm[v * 4 + 1] = values.y;
        sm[v * 4 + 2] = values.z; sm[v * 4 + 3] = values.w;
      }
    } else {
#pragma unroll
      for (int i = 0; i < Pack; i++) if (i < count) sm[i / 4] |= uint32_t(loadU8(src + i)) << (8 * (i & 3));
    }
  }
  __device__ __forceinline__ static void storeSignMantissa(
      uint8_t* dst, int count, const uint32_t sm[WordsPerSignMantissa]) {
    if (count == Pack && isAligned16(dst)) {
      uint4* output = reinterpret_cast<uint4*>(dst);
#pragma unroll
      for (int v = 0; v < Pack / int(sizeof(uint4)); v++)
        output[v] = make_uint4(sm[v * 4 + 0], sm[v * 4 + 1], sm[v * 4 + 2], sm[v * 4 + 3]);
    } else {
#pragma unroll
      for (int i = 0; i < Pack; i++) if (i < count) dst[i] = uint8_t(sm[i / 4] >> (8 * (i & 3)));
    }
  }
  __device__ __forceinline__ static uint16_t pairElement(uint32_t pair, int i) {
    return uint16_t(pair >> (16 * i));
  }
  __device__ __forceinline__ static uint8_t smElement(const uint32_t sm[WordsPerSignMantissa], int i) {
    return uint8_t(sm[i / 4] >> (8 * (i & 3)));
  }
  __device__ __forceinline__ static void setSmElement(uint32_t sm[WordsPerSignMantissa], int i, uint16_t raw) {
    uint8_t v = uint8_t(((raw >> 8) & 0x80u) | (raw & 0x7fu));
    sm[i / 4] |= uint32_t(v) << (8 * (i & 3));
  }
  __device__ __forceinline__ static void replaceSmElement(
      uint32_t sm[WordsPerSignMantissa], int i, uint16_t raw) {
    int shift = 8 * (i & 3);
    uint32_t mask = 0xffu << shift;
    uint8_t value = uint8_t(((raw >> 8) & 0x80u) | (raw & 0x7fu));
    sm[i / 4] = (sm[i / 4] & ~mask) | (uint32_t(value) << shift);
  }
  __device__ __forceinline__ static uint32_t exponentCode(uint16_t raw, int exponentBase) {
    uint32_t exponent = (raw >> 7) & 0xffu;
    return exponent >= uint32_t(exponentBase) && exponent < uint32_t(exponentBase + 7)
             ? exponent - uint32_t(exponentBase) + 1u : 0u;
  }
  __device__ __forceinline__ static void replaceCodeElement(
      CodeBitmaps& bitmaps, int bit, uint32_t code) {
    uint32_t mask = 1u << bit;
    bitmaps.b0 = (bitmaps.b0 & ~mask) | ((code & 1u) << bit);
    bitmaps.b1 = (bitmaps.b1 & ~mask) | (((code >> 1) & 1u) << bit);
    bitmaps.b2 = (bitmaps.b2 & ~mask) | (((code >> 2) & 1u) << bit);
  }
  __device__ __forceinline__ static void replacePackElement(
      PackRegisters& values, int bit, uint16_t raw) {
    int half = bit & 1;
    uint32_t shift = 16 * half;
    uint32_t mask = 0xffffu << shift;
    values.pair[bit / 2] = (values.pair[bit / 2] & ~mask) | (uint32_t(raw) << shift);
  }
  __device__ __forceinline__ static uint16_t reduceOutlierElement(
      uint8_t exponent, const uint32_t sm[WordsPerSignMantissa], int bit,
      const PackRegisters& local) {
    uint16_t signMantissa = smElement(sm, bit);
    uint16_t inputRaw = uint16_t((signMantissa & 0x80u) << 8) |
                        uint16_t(uint16_t(exponent) << 7) |
                        uint16_t(signMantissa & 0x7fu);
    int half = bit & 1;
    uint32_t inputPair = uint32_t(inputRaw) << (16 * half);
    uint32_t reducedPair = addBf16Pair(inputPair, local.pair[bit / 2]);
    return pairElement(reducedPair, half);
  }
  __device__ __forceinline__ static uint32_t addBf16Pair(uint32_t lhs, uint32_t rhs) {
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&lhs);
    __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&rhs);
    __nv_bfloat162 result = __hadd2(a, b);
    return *reinterpret_cast<uint32_t*>(&result);
  }
  __device__ __forceinline__ static int warpExclusivePrefix(int value, int lane) {
    int inclusive = value;
#pragma unroll
    for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
      int preceding = __shfl_up_sync(0xffffffffu, inclusive, delta);
      if (lane >= delta) inclusive += preceding;
    }
    return inclusive - value;
  }
  __device__ __forceinline__ static FrameView frameView(const void* encoded, int n) {
    FrameView v;
    const uint8_t* bytes = static_cast<const uint8_t*>(encoded);
    v.frame = bytes;
    v.signMantissa = bytes + sizeof(ZipCclRingFrameHeader);
    int bitmapN = bitmapBytes(n);
    v.bitmap0 = reinterpret_cast<const uint32_t*>(bytes + bitmapOffset(n));
    v.bitmap1 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(v.bitmap0) + bitmapN);
    v.bitmap2 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(v.bitmap1) + bitmapN);
    v.groupOffsets = reinterpret_cast<const uint32_t*>(bytes + groupOffsetOffset(n));
    v.outlierExponents = bytes + outlierOffset(n);
    return v;
  }

  template <typename Prims>
  __device__ __forceinline__ static Statistics calculateStatistics(
      int tid, int nworkers, const T* input, int n, Prims& prims) {
    Scratch* scratch = reinterpret_cast<Scratch*>(prims.codecScratch());
    float localSum = 0.0f, localSumSq = 0.0f;
    for (int begin = tid * Pack; begin < n; begin += nworkers * Pack) {
      int count = min(Pack, n - begin);
      PackRegisters pack;
      loadBf16Pack(input + begin, count, pack);
#pragma unroll
      for (int i = 0; i < Pack; i++) if (i < count) {
        uint16_t raw = pairElement(pack.pair[i / 2], i & 1);
        float value = __bfloat162float(__ushort_as_bfloat16(raw));
        localSum += value;
        localSumSq = fmaf(value, value, localSumSq);
      }
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

  __device__ __forceinline__ static uint16_t decodeElementOrNan(
      int bit, CodeBitmaps bitmaps, int exponentBase,
      const uint32_t sm[WordsPerSignMantissa]) {
    uint32_t code = ((bitmaps.b0 >> bit) & 1u) |
                    (((bitmaps.b1 >> bit) & 1u) << 1) |
                    (((bitmaps.b2 >> bit) & 1u) << 2);
    uint16_t signMantissa = smElement(sm, bit);
    if (code == 0) {
      // Reserved code 0 stays on the dependency-free path as a quiet NaN.
      return uint16_t((signMantissa & 0x80u) << 8) | uint16_t(0x7fc0u);
    }
    int exponent = exponentBase + int(code) - 1;
    return uint16_t((signMantissa & 0x80u) << 8) | uint16_t((exponent & 0xff) << 7) |
           uint16_t(signMantissa & 0x7fu);
  }

  template <bool FullPack>
  __device__ __forceinline__ static void decodeReduceNonOutliersPackImpl(
      int count, CodeBitmaps inputBitmaps, int inputBase,
      const uint32_t inputSm[WordsPerSignMantissa], const PackRegisters& local,
      PackRegisters& result) {
#pragma unroll
    for (int pair = 0; pair < PairsPerPack; pair++) {
      int bit = pair * 2;
      if (FullPack || bit < count) {
        uint16_t lo = decodeElementOrNan(bit, inputBitmaps, inputBase, inputSm);
        uint16_t hi = (FullPack || bit + 1 < count)
          ? decodeElementOrNan(bit + 1, inputBitmaps, inputBase, inputSm)
          : uint16_t(0);
        result.pair[pair] = addBf16Pair(
          uint32_t(lo) | (uint32_t(hi) << 16), local.pair[pair]);
      }
    }
  }

  template <bool FullPack>
  __device__ __forceinline__ static void encodeNonOutliersImpl(
      const PackRegisters& values, int count, int exponentBase,
      uint32_t* sm, CodeBitmaps& bitmaps) {

    #pragma unroll
    for (int word = 0; word < WordsPerSignMantissa; word++) {
      sm[word] = 0;
    }

    bitmaps.b0 = bitmaps.b1 = bitmaps.b2 = 0;

    #pragma unroll
    for (int i = 0; i < Pack; i++) if (FullPack || i < count) {
      uint16_t raw = pairElement(values.pair[i / 2], i & 1);
      uint32_t code = exponentCode(raw, exponentBase);
      setSmElement(sm, i, raw);
      bitmaps.b0 |= (code & 1u) << i;
      bitmaps.b1 |= ((code >> 1) & 1u) << i;
      bitmaps.b2 |= ((code >> 2) & 1u) << i;
    }
  }

  __device__ __forceinline__ static void encodeNonOutliers(
      const PackRegisters& values, int count, int exponentBase,
      uint32_t* sm, CodeBitmaps& bitmaps) {
    if (count == Pack) {
      encodeNonOutliersImpl<true>(values, count, exponentBase, sm, bitmaps);
    } else {
      encodeNonOutliersImpl<false>(values, count, exponentBase, sm, bitmaps);
    }
  }

  __device__ __forceinline__ static void writeOutliers(
      Scratch* scratch, int lane, int group, uint32_t validMask,
      const PackRegisters& values, CodeBitmaps bitmaps,
      uint32_t* groupOffsets, uint8_t* outlierStream) {
    uint32_t outlierMask = ~(bitmaps.b0 | bitmaps.b1 | bitmaps.b2) & validMask;
    int threadOutliers = __popc(outlierMask);

    // Fixed streams must be visible before the group writes its variable stream.
    __syncwarp();
    int threadPrefix = warpExclusivePrefix(threadOutliers, lane);
    int groupOutliers = __shfl_sync(
      0xffffffffu, threadPrefix + threadOutliers, WARP_SIZE - 1);
    int groupBase = lane == 0 ? atomicAdd(&scratch->outlierCount, groupOutliers) : 0;
    groupBase = __shfl_sync(0xffffffffu, groupBase, 0);
    if (lane == 0) groupOffsets[group] = uint32_t(groupBase);

    int outputIndex = groupBase + threadPrefix;
    while (outlierMask) {
      int bit = __ffs(outlierMask) - 1;
      uint16_t raw = pairElement(values.pair[bit / 2], bit & 1);
      outlierStream[outputIndex++] = uint8_t((raw >> 7) & 0xffu);
      outlierMask &= outlierMask - 1;
    }
  }

  template <bool FullGroup>
  __device__ __forceinline__ static void decodeReduceEncodeGroup(
      Scratch* scratch, const FrameView& input, const T* localInput, int n, int group,
      int lane, int inputBase, int outputBase, uint8_t* smStream,
      uint32_t* bitmap0, uint32_t* bitmap1, uint32_t* bitmap2,
      uint32_t* groupOffsets, uint8_t* outlierStream) {
    int pack = group * PacksPerGroup + lane;
    int begin = pack * Pack;
    int count = FullGroup ? Pack : min(Pack, max(0, n - begin));
    uint32_t validMask = count == Pack ? 0xffffffffu : (1u << count) - 1u;

    CodeBitmaps bitmaps = {};
    uint32_t sm[WordsPerSignMantissa] = {};
    PackRegisters local = {}, result = {};
    if (FullGroup || count) {
      bitmaps.b0 = loadU32(input.bitmap0 + pack);
      bitmaps.b1 = loadU32(input.bitmap1 + pack);
      bitmaps.b2 = loadU32(input.bitmap2 + pack);
      loadSignMantissa(input.signMantissa + begin, count, sm);
      loadBf16Pack(localInput + begin, count, local);
    }

    uint32_t inputOutlierMask = ~(bitmaps.b0 | bitmaps.b1 | bitmaps.b2) & validMask;
    int inputOutliers = __popc(inputOutlierMask);
    int inputGroupBase = lane == 0 ? int(loadU32(input.groupOffsets + group)) : 0;

    uint32_t outputSm[WordsPerSignMantissa];
    if (FullGroup || count) {
      decodeReduceNonOutliersPackImpl<FullGroup>(
        count, bitmaps, inputBase, sm, local, result);
      // The input bitmap registers are no longer needed after inputOutlierMask.
      encodeNonOutliersImpl<FullGroup>(result, count, outputBase, outputSm, bitmaps);
    }

    int inputPrefix = warpExclusivePrefix(inputOutliers, lane);
    inputGroupBase = __shfl_sync(0xffffffffu, inputGroupBase, 0);
    int inputIndex = inputGroupBase + inputPrefix;
    while (inputOutlierMask) {
      int bit = __ffs(inputOutlierMask) - 1;
      uint8_t exponent = loadU8(input.outlierExponents + inputIndex++);
      uint16_t reducedRaw = reduceOutlierElement(exponent, sm, bit, local);
      replacePackElement(result, bit, reducedRaw);
      replaceSmElement(outputSm, bit, reducedRaw);
      replaceCodeElement(bitmaps, bit, exponentCode(reducedRaw, outputBase));
      inputOutlierMask &= inputOutlierMask - 1;
    }

    if (FullGroup || count) {
      storeSignMantissa(smStream + begin, count, outputSm);
      bitmap0[pack] = bitmaps.b0;
      bitmap1[pack] = bitmaps.b1;
      bitmap2[pack] = bitmaps.b2;
    }
    writeOutliers(
      scratch, lane, group, validMask, result, bitmaps, groupOffsets, outlierStream);
  }

  template <bool FullGroup>
  __device__ __forceinline__ static void decodeReduceCopyGroup(
      const FrameView& input, const T* localInput, int n, int group,
      int lane, int inputBase, T* output) {
    int pack = group * PacksPerGroup + lane;
    int begin = pack * Pack;
    int count = FullGroup ? Pack : min(Pack, max(0, n - begin));
    uint32_t validMask = count == Pack ? 0xffffffffu : (1u << count) - 1u;

    CodeBitmaps bitmaps = {};
    uint32_t sm[WordsPerSignMantissa] = {};
    PackRegisters local = {}, result = {};
    if (FullGroup || count) {
      bitmaps.b0 = loadU32(input.bitmap0 + pack);
      bitmaps.b1 = loadU32(input.bitmap1 + pack);
      bitmaps.b2 = loadU32(input.bitmap2 + pack);
      loadSignMantissa(input.signMantissa + begin, count, sm);
      loadBf16Pack(localInput + begin, count, local);
    }

    uint32_t inputOutlierMask = ~(bitmaps.b0 | bitmaps.b1 | bitmaps.b2) & validMask;
    int inputOutliers = __popc(inputOutlierMask);
    int inputPrefix = warpExclusivePrefix(inputOutliers, lane);
    int inputGroupBase = lane == 0 ? int(loadU32(input.groupOffsets + group)) : 0;
    inputGroupBase = __shfl_sync(0xffffffffu, inputGroupBase, 0);

    if (FullGroup || count)
      decodeReduceNonOutliersPackImpl<FullGroup>(
        count, bitmaps, inputBase, sm, local, result);

    int inputIndex = inputGroupBase + inputPrefix;
    while (inputOutlierMask) {
      int bit = __ffs(inputOutlierMask) - 1;
      uint8_t exponent = loadU8(input.outlierExponents + inputIndex++);
      replacePackElement(
        result, bit, reduceOutlierElement(exponent, sm, bit, local));
      inputOutlierMask &= inputOutlierMask - 1;
    }

    if (FullGroup || count)
      storeBf16Pack(output + begin, count, result);
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
      Prims& prims, int tid, int nworkers, const T* input, int n, void* encoded, int capacity,
      Statistics stats) {
    Scratch* scratch = reinterpret_cast<Scratch*>(prims.codecScratch());
    if (tid == 0) {
      scratch->outlierCount = 0; scratch->frameBytes = 0;
      scratch->valid = worstFrameBytes(n) <= capacity;
      scratch->exponentBase = selectExponentBase(stats);
    }
    prims.syncCodecWorkers();
    if (!scratch->valid) return 0;
    uint8_t* bytes = static_cast<uint8_t*>(encoded);
    uint8_t* smStream = bytes + sizeof(ZipCclRingFrameHeader);
    int bitmapN = bitmapBytes(n);
    uint32_t* bitmap0 = reinterpret_cast<uint32_t*>(bytes + bitmapOffset(n));
    uint32_t* bitmap1 = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(bitmap0) + bitmapN);
    uint32_t* bitmap2 = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(bitmap1) + bitmapN);
    uint32_t* groupOffsets = reinterpret_cast<uint32_t*>(bytes + groupOffsetOffset(n));
    uint8_t* outlierStream = bytes + outlierOffset(n);

    int warp = tid / WARP_SIZE, lane = tid & (WARP_SIZE - 1), workerWarps = nworkers / WARP_SIZE;
    for (int group = warp; group < groupCount(n); group += workerWarps) {
      int pack = group * PacksPerGroup + lane, begin = pack * Pack;
      int count = min(Pack, max(0, n - begin));
      PackRegisters values = {};
      CodeBitmaps bitmaps = {};
      if (count) {
        uint32_t sm[WordsPerSignMantissa];
        loadBf16Pack(input + begin, count, values);
        encodeNonOutliers(values, count, scratch->exponentBase, sm, bitmaps);
        storeSignMantissa(smStream + begin, count, sm);
        bitmap0[pack] = bitmaps.b0;
        bitmap1[pack] = bitmaps.b1;
        bitmap2[pack] = bitmaps.b2;
      }

      uint32_t validMask = count == Pack ? 0xffffffffu : (1u << count) - 1u;
      writeOutliers(
        scratch, lane, group, validMask, values, bitmaps, groupOffsets, outlierStream);
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
      scratch->outlierCount = 0; scratch->frameBytes = 0;
      scratch->valid = worstFrameBytes(n) <= capacity;
      scratch->exponentBase = selectExponentBase(combined);
    }
    prims.syncCodecWorkers();
    if (!scratch->valid) return 0;

    uint8_t* bytes = static_cast<uint8_t*>(encodedOutput);
    uint8_t* smStream = bytes + sizeof(ZipCclRingFrameHeader);
    int bitmapN = bitmapBytes(n);
    uint32_t* bitmap0 = reinterpret_cast<uint32_t*>(bytes + bitmapOffset(n));
    uint32_t* bitmap1 = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(bitmap0) + bitmapN);
    uint32_t* bitmap2 = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(bitmap1) + bitmapN);
    uint32_t* groupOffsets = reinterpret_cast<uint32_t*>(bytes + groupOffsetOffset(n));
    uint8_t* outlierStream = bytes + outlierOffset(n);
    int inputBase = int(loadU32(reinterpret_cast<const uint32_t*>(input.frame + offsetof(ZipCclRingFrameHeader, exponentBase))));
    int warp = tid / WARP_SIZE, lane = tid & (WARP_SIZE - 1), workerWarps = nworkers / WARP_SIZE;
    int fullGroups = n / ElementsPerGroup;
    for (int group = warp; group < fullGroups; group += workerWarps) {
      decodeReduceEncodeGroup<true>(
        scratch, input, localInput, n, group, lane, inputBase, scratch->exponentBase,
        smStream, bitmap0, bitmap1, bitmap2, groupOffsets, outlierStream);
    }

    int tailN = n - fullGroups * ElementsPerGroup;
    if (tailN && warp == fullGroups % workerWarps) {
      decodeReduceEncodeGroup<false>(
        scratch, input, localInput, n, fullGroups, lane, inputBase, scratch->exponentBase,
        smStream, bitmap0, bitmap1, bitmap2, groupOffsets, outlierStream);
    }
    return finishFrame(prims, scratch, tid, n, encodedOutput, combined, contributors + 1);
  }

  template <typename Prims>
  __device__ __forceinline__ static void decodeReduceCopy(
      Prims& prims, int tid, int nworkers, const void* encodedInput, const T* localInput, int n,
      T* output, bool /*postOp*/) {
    FrameView input = frameView(encodedInput, n);
    int inputBase = int(loadU32(reinterpret_cast<const uint32_t*>(input.frame + offsetof(ZipCclRingFrameHeader, exponentBase))));
    int warp = tid / WARP_SIZE, lane = tid & (WARP_SIZE - 1), workerWarps = nworkers / WARP_SIZE;
    int fullGroups = n / ElementsPerGroup;
    for (int group = warp; group < fullGroups; group += workerWarps) {
      decodeReduceCopyGroup<true>(
        input, localInput, n, group, lane, inputBase, output);
    }

    int tailN = n - fullGroups * ElementsPerGroup;
    if (tailN && warp == fullGroups % workerWarps) {
      decodeReduceCopyGroup<false>(
        input, localInput, n, fullGroups, lane, inputBase, output);
    }
  }

  __device__ __forceinline__ static size_t maxChunkElements(size_t transportChunkElements, int nSlices) {
    size_t capacity = transportChunkElements * sizeof(T) / nSlices;
    // Sign/mantissa, three bitmap bits, a worst-case outlier byte, and group
    // offsets fit under this conservative 153/64-byte bound per element.
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
    if (Stage == 0 && nDsts) {
      // Compute statistics for this slice immediately before encoding it. This
      // allows the first encoded slice to enter the ring without scanning the
      // rest of the channel chunk.
      typename Algorithm::Statistics stats =
        Algorithm::calculateStatistics(tid, nworkers, local, sliceN, *prims);
      payloadBytes = Algorithm::encode(
        *prims, tid, nworkers, local, sliceN, dstPtrs[0], maxSliceElts * int(sizeof(T)), stats);
    } else if (Stage == 1 && nSrcs && nDsts)
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
