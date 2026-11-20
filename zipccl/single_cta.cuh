#pragma once

#include "frame.cuh"
#include <cuda_bf16.h>
#include <cuda/pipeline>

namespace zipccl {
namespace device {
constexpr int kWarpsPerCta = 16;
constexpr int kThreads = kWarpSize * kWarpsPerCta;
constexpr std::size_t kWordsPerBlock = kElementsPerBlock / kWarpSize;
constexpr int kDreValuesPerThread = kDreValuesPerWarp / kWarpSize;
constexpr int kBlocksPerRound = kWarpsPerCta / kSegmentsPerBlock;
constexpr int kSignTilesPerWarp = kDreSignBytes / kBytesPerTile;
constexpr int kBf16TilesPerWarp = kDreBf16Bytes / kBytesPerTile;
constexpr int kDreRoundSlotsBytes = kWarpsPerCta * kDreValuesPerWarp;
constexpr int kDreRoundCountsBytes = kWarpsPerCta * sizeof(std::uint16_t);
constexpr int kDreFixedBytes = kWarpsPerCta * kDreWarpSharedBytes;

struct EncodedInput {
    const std::uint8_t* sign_mantissa;
    const std::uint32_t* bitmap0;
    const std::uint32_t* bitmap1;
    const std::uint32_t* bitmap2;
    const std::uint32_t* block_offsets;
    const std::uint32_t* warp_offsets;
    const std::uint8_t* outliers;
    const std::uint16_t* raw = nullptr;
};

struct EncodedOutput {
    std::uint8_t* sign_mantissa;
    std::uint32_t* bitmap0;
    std::uint32_t* bitmap1;
    std::uint32_t* bitmap2;
    std::uint32_t* block_offsets;
    std::uint32_t* warp_offsets;
    std::uint8_t* outliers;
};

__device__ __forceinline__ std::uint16_t JoinBf16(std::uint8_t sign_mantissa,
                                                  unsigned exponent) {
    return static_cast<std::uint16_t>(
        (static_cast<unsigned>(sign_mantissa & 0x80u) << 8) |
        ((exponent & 0xffu) << 7) | (sign_mantissa & 0x7fu));
}

__device__ __forceinline__ std::uint8_t DecodeExponent(
        unsigned code, std::uint8_t exponent_base, std::uint32_t exceptional,
        int bit, const std::uint8_t* outliers, std::uint32_t outlier_base) {
    return code == 0
        ? outliers[outlier_base + (bit == 0 ? 0u :
            __popc(exceptional & ((1u << bit) - 1u)))]
        : static_cast<std::uint8_t>(exponent_base + code - 1u);
}

__device__ __forceinline__ unsigned LanePrefix(std::uint32_t mask, int lane) {
    return lane == 0 ? 0u : __popc(mask & ((1u << lane) - 1u));
}

__device__ __forceinline__ std::uint32_t LoadBitmapWord(
        const std::uint32_t* bitmap, std::size_t word, int source_lane) {
    const std::uint32_t value = threadIdx.x % kWarpSize == source_lane
        ? bitmap[word] : 0u;
    return __shfl_sync(0xffffffffu, value, source_lane);
}

__device__ __forceinline__ int WarpExclusivePrefix(int value, int lane) {
    int inclusive = value;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += other;
    }
    return inclusive - value;
}

__device__ __forceinline__ unsigned SwizzleSignTile(unsigned tile) {
    return tile ^ (tile >> 1);
}

__device__ __forceinline__ unsigned SwizzleBf16Tile(unsigned tile) {
    return tile ^ (tile >> 2);
}

__device__ __forceinline__ std::uint32_t Uint4Word(const uint4& value,
                                                    int word) {
    return word == 0 ? value.x :
           word == 1 ? value.y :
           word == 2 ? value.z : value.w;
}

__device__ __forceinline__ std::uint8_t Uint4Byte(const uint4& value,
                                                  int byte) {
    return static_cast<std::uint8_t>(Uint4Word(value, byte >> 2) >>
                                     ((byte & 3) * 8));
}

__device__ __forceinline__ std::uint16_t Uint4Bf16(const uint4& value,
                                                   int element) {
    return static_cast<std::uint16_t>(Uint4Word(value, element >> 1) >>
                                      ((element & 1) * 16));
}

__device__ __forceinline__ std::uint32_t PackFourExponentCodes(
        std::uint32_t bitmap0, std::uint32_t bitmap1,
        std::uint32_t bitmap2, int first) {
    constexpr std::uint32_t kSpreadMultiplier = 0x00204081u;
    constexpr std::uint32_t kByteLowBits = 0x01010101u;
    const std::uint32_t code_bit0 =
        ((bitmap0 >> first) & 0xfu) * kSpreadMultiplier & kByteLowBits;
    const std::uint32_t code_bit1 =
        ((bitmap1 >> first) & 0xfu) * kSpreadMultiplier & kByteLowBits;
    const std::uint32_t code_bit2 =
        ((bitmap2 >> first) & 0xfu) * kSpreadMultiplier & kByteLowBits;
    return code_bit0 | (code_bit1 << 1) | (code_bit2 << 2);
}

__device__ __forceinline__ std::uint32_t GatherFourByteLowBits(
        std::uint32_t bytes) {
    return ((bytes & 0x01010101u) * 0x01020408u) >> 24;
}

__device__ __forceinline__ uint2 ReduceFullFourCodes(
        const uint4& sign_values, const uint4& b_values0,
        const uint4& b_values1, int j, int i, std::uint32_t packed_codes,
        std::uint32_t exceptional_a,
        std::uint8_t exponent_base_a, const std::uint8_t* shared_outliers,
        std::uint32_t outlier_base_a) {
    const unsigned code0 = packed_codes & 7u;
    const unsigned code1 = (packed_codes >> 8) & 7u;
    const unsigned code2 = (packed_codes >> 16) & 7u;
    const unsigned code3 = (packed_codes >> 24) & 7u;
    const auto exponent0 = DecodeExponent(
        code0, exponent_base_a, exceptional_a, i,
        shared_outliers, outlier_base_a);
    const auto exponent1 = DecodeExponent(
        code1, exponent_base_a, exceptional_a, i + 1,
        shared_outliers, outlier_base_a);
    const auto exponent2 = DecodeExponent(
        code2, exponent_base_a, exceptional_a, i + 2,
        shared_outliers, outlier_base_a);
    const auto exponent3 = DecodeExponent(
        code3, exponent_base_a, exceptional_a, i + 3,
        shared_outliers, outlier_base_a);
    const __nv_bfloat162 va0(
        __ushort_as_bfloat16(JoinBf16(Uint4Byte(sign_values, j), exponent0)),
        __ushort_as_bfloat16(JoinBf16(
            Uint4Byte(sign_values, j + 1), exponent1)));
    const __nv_bfloat162 va1(
        __ushort_as_bfloat16(JoinBf16(
            Uint4Byte(sign_values, j + 2), exponent2)),
        __ushort_as_bfloat16(JoinBf16(
            Uint4Byte(sign_values, j + 3), exponent3)));
    const uint4& b_values = j < 8 ? b_values0 : b_values1;
    const int b_element = j & 7;
    const __nv_bfloat162 vb0(
        __ushort_as_bfloat16(Uint4Bf16(b_values, b_element)),
        __ushort_as_bfloat16(Uint4Bf16(b_values, b_element + 1)));
    const __nv_bfloat162 vb1(
        __ushort_as_bfloat16(Uint4Bf16(b_values, b_element + 2)),
        __ushort_as_bfloat16(Uint4Bf16(b_values, b_element + 3)));
    const __nv_bfloat162 sum0 = __hadd2(va0, vb0);
    const __nv_bfloat162 sum1 = __hadd2(va1, vb1);
    return make_uint2(
        static_cast<std::uint32_t>(__bfloat16_as_ushort(sum0.x)) |
            (static_cast<std::uint32_t>(__bfloat16_as_ushort(sum0.y)) << 16),
        static_cast<std::uint32_t>(__bfloat16_as_ushort(sum1.x)) |
            (static_cast<std::uint32_t>(__bfloat16_as_ushort(sum1.y)) << 16));
}

enum class SegmentOp { Encode, Dre, Copy, RawReduceEncode };

__device__ __forceinline__ unsigned AddBf16Pair(unsigned a, unsigned b) {
    const auto lhs = __halves2bfloat162(__ushort_as_bfloat16(a), __ushort_as_bfloat16(a >> 16));
    const auto rhs = __halves2bfloat162(__ushort_as_bfloat16(b), __ushort_as_bfloat16(b >> 16));
    const auto sum = __hadd2(lhs, rhs);
    return __bfloat16_as_ushort(sum.x) | (unsigned(__bfloat16_as_ushort(sum.y)) << 16);
}

template<bool FullSegment, SegmentOp Op = SegmentOp::Dre>
__device__ __forceinline__ void ProcessSegment(
        EncodedInput input_a, std::uint8_t exponent_base_a,
        const std::uint16_t* __restrict__ input_b, std::uint32_t elements,
        std::uint8_t output_exponent_base, EncodedOutput output,
        unsigned segment, std::uint16_t* segment_count,
        std::uint8_t* segment_slot, unsigned char* shared_bytes,
        std::uint16_t* dense_output = nullptr) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const unsigned segment_first = segment * kDreValuesPerWarp;
    if (!FullSegment && segment_first >= elements) {
        if (Op != SegmentOp::Copy && lane == 0) *segment_count = 0;
        return;
    }
    auto* warp_shared = shared_bytes + warp * kDreWarpSharedBytes;
    auto* shared_sign_mantissa = warp_shared;
    auto* shared_b = reinterpret_cast<std::uint16_t*>(
        shared_sign_mantissa + kDreSignBytes);
    auto* shared_bitmap0 = reinterpret_cast<std::uint32_t*>(
        warp_shared + kDreSignBytes + kDreBf16Bytes);
    auto* shared_bitmap1 = reinterpret_cast<std::uint32_t*>(
        reinterpret_cast<unsigned char*>(shared_bitmap0) + kDreBitmapBytes);
    auto* shared_bitmap2 = reinterpret_cast<std::uint32_t*>(
        reinterpret_cast<unsigned char*>(shared_bitmap1) + kDreBitmapBytes);
    auto* warp_shared_outliers = reinterpret_cast<std::uint8_t*>(
        reinterpret_cast<unsigned char*>(shared_bitmap2) + kDreBitmapBytes);
    const unsigned segment_elements = FullSegment ? kDreValuesPerWarp :
        min(static_cast<unsigned>(kDreValuesPerWarp), elements - segment_first);
    const unsigned word = segment * kWarpSize + lane;
    const unsigned first = word * kDreValuesPerThread;
    const bool active_word = FullSegment || first < elements;
    const std::uint32_t valid = !active_word ? 0u :
        (FullSegment || elements - first >= 32 ? 0xffffffffu :
         ((1u << (elements - first)) - 1u));
    std::uint32_t warp_outlier_start = 0;
    std::uint32_t warp_outlier_end = 0;
    constexpr bool decode = Op == SegmentOp::Dre || Op == SegmentOp::Copy;
    if (decode && lane == 0) {
        const std::uint32_t offset_index =
            (segment / kSegmentsPerBlock) * kDreWarpOffsetsPerBlock + segment % kSegmentsPerBlock;
        warp_outlier_start = input_a.warp_offsets[offset_index];
        warp_outlier_end = input_a.warp_offsets[offset_index + 1];
    }
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    for (std::uint32_t tile = lane;
         decode && tile < kSignTilesPerWarp;
         tile += kWarpSize) {
        const std::uint32_t offset = tile * kBytesPerTile;
        const std::uint32_t shared_offset =
            SwizzleSignTile(static_cast<unsigned>(tile)) * kBytesPerTile;
        if (FullSegment || offset + kBytesPerTile <= segment_elements) {
            cuda::memcpy_async(shared_sign_mantissa + shared_offset,
                               input_a.sign_mantissa + segment_first + offset,
                               cuda::aligned_size_t<kBytesPerTile>(kBytesPerTile),
                               pipe);
        } else {
            for (std::uint32_t i = offset; i < segment_elements; ++i) {
                shared_sign_mantissa[shared_offset + i - offset] =
                    input_a.sign_mantissa[segment_first + i];
            }
        }
    }
    const std::uint32_t segment_b_bytes =
        segment_elements * sizeof(std::uint16_t);
    auto* shared_b_bytes = reinterpret_cast<unsigned char*>(shared_b);
    const auto* input_b_bytes = reinterpret_cast<const unsigned char*>(
        input_b + segment_first);
    for (std::uint32_t tile = lane;
         tile < kBf16TilesPerWarp;
         tile += kWarpSize) {
        const std::uint32_t offset = tile * kBytesPerTile;
        const std::uint32_t shared_offset =
            SwizzleBf16Tile(static_cast<unsigned>(tile)) * kBytesPerTile;
        if constexpr (Op == SegmentOp::RawReduceEncode) {
            // Reduce adjacent FIFO and local tiles before redistributing them to encoder lanes.
            const auto* raw = input_a.raw + segment_first;
            const unsigned elem = offset / sizeof(std::uint16_t);
            if ((FullSegment || offset + kBytesPerTile <= segment_b_bytes) &&
                (reinterpret_cast<uintptr_t>(input_b_bytes) & 15) == 0) {
                const uint4 a = *reinterpret_cast<const uint4*>(raw + elem);
                const uint4 b = *reinterpret_cast<const uint4*>(input_b_bytes + offset);
                *reinterpret_cast<uint4*>(shared_b_bytes + shared_offset) =
                    make_uint4(AddBf16Pair(a.x, b.x), AddBf16Pair(a.y, b.y),
                               AddBf16Pair(a.z, b.z), AddBf16Pair(a.w, b.w));
            } else {
                auto* dst = reinterpret_cast<std::uint16_t*>(shared_b_bytes + shared_offset);
                for (unsigned i = elem; i < min(elem + 8, segment_elements); ++i)
                    dst[i - elem] = __bfloat16_as_ushort(__hadd(__ushort_as_bfloat16(raw[i]),
                        __ushort_as_bfloat16(input_b[segment_first + i])));
            }
        } else if ((FullSegment || offset + kBytesPerTile <= segment_b_bytes) &&
            (reinterpret_cast<uintptr_t>(input_b_bytes) & 15) == 0) {
            cuda::memcpy_async(shared_b_bytes + shared_offset,
                               input_b_bytes + offset,
                               cuda::aligned_size_t<kBytesPerTile>(kBytesPerTile),
                               pipe);
        } else {
            for (std::uint32_t i = offset; i < min(offset + kBytesPerTile, segment_b_bytes); ++i) {
                shared_b_bytes[shared_offset + i - offset] = input_b_bytes[i];
            }
        }
    }
    if (decode && active_word) {
        cuda::memcpy_async(shared_bitmap0 + lane, input_a.bitmap0 + word,
                           cuda::aligned_size_t<sizeof(std::uint32_t)>(
                               sizeof(std::uint32_t)), pipe);
        cuda::memcpy_async(shared_bitmap1 + lane, input_a.bitmap1 + word,
                           cuda::aligned_size_t<sizeof(std::uint32_t)>(
                               sizeof(std::uint32_t)), pipe);
        cuda::memcpy_async(shared_bitmap2 + lane, input_a.bitmap2 + word,
                           cuda::aligned_size_t<sizeof(std::uint32_t)>(
                               sizeof(std::uint32_t)), pipe);
    } else {
        shared_bitmap0[lane] = 0u;
        shared_bitmap1[lane] = 0u;
        shared_bitmap2[lane] = 0u;
    }
    warp_outlier_start = __shfl_sync(
        0xffffffffu, warp_outlier_start, 0);
    warp_outlier_end = __shfl_sync(0xffffffffu, warp_outlier_end, 0);
    const unsigned outlier_alignment =
        warp_outlier_start & (kBytesPerTile - 1);
    const std::uint32_t aligned_outlier_start =
        warp_outlier_start - outlier_alignment;
    const std::uint32_t outlier_bytes =
        warp_outlier_end > warp_outlier_start
        ? outlier_alignment + warp_outlier_end - warp_outlier_start : 0;
    const std::uint32_t outlier_copy_bytes = outlier_bytes == 0 ? 0 :
        (outlier_bytes + kBytesPerTile - 1) & ~(kBytesPerTile - 1);
    for (std::uint32_t offset = lane * kBytesPerTile;
         offset < outlier_copy_bytes;
         offset += kWarpSize * kBytesPerTile) {
        cuda::memcpy_async(warp_shared_outliers + offset,
                           input_a.outliers + aligned_outlier_start + offset,
                           cuda::aligned_size_t<kBytesPerTile>(kBytesPerTile),
                           pipe);
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    pipe.consumer_release();
    __syncwarp();
    const std::uint32_t a0 = shared_bitmap0[lane];
    const std::uint32_t a1 = shared_bitmap1[lane];
    const std::uint32_t a2 = shared_bitmap2[lane];
    const std::uint32_t exceptional_a = valid & ~(a0 | a1 | a2);
    const int prefix_a = WarpExclusivePrefix(__popc(exceptional_a), lane);
    const std::uint32_t outlier_base_a = outlier_alignment + prefix_a;
    const auto* shared_outliers = warp_shared_outliers;
    const bool full_segment = segment_elements == kDreValuesPerWarp;
    uint3 bitmap_half[2];
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        std::uint32_t reduced_pairs[8];
        const unsigned sign_tile = lane * 2 + half;
        const unsigned b_tile = lane * 4 + half * 2;
        const uint4 sign_values = *reinterpret_cast<const uint4*>(
            shared_sign_mantissa +
            SwizzleSignTile(sign_tile) * kBytesPerTile);
        const uint4 b_values0 = *reinterpret_cast<const uint4*>(
            shared_b_bytes + SwizzleBf16Tile(b_tile) * kBytesPerTile);
        const uint4 b_values1 = *reinterpret_cast<const uint4*>(
            shared_b_bytes +
            SwizzleBf16Tile(b_tile + 1) * kBytesPerTile);
        if constexpr (Op == SegmentOp::RawReduceEncode || Op == SegmentOp::Encode) {
            const uint4 values[2] = {b_values0, b_values1};
            #pragma unroll
            for (int j = 0; j < 8; ++j) reduced_pairs[j] = Uint4Word(values[j / 4], j % 4);
        } else if (full_segment) {
            #pragma unroll
            for (int j = 0; j < 16; j += 4) {
                const int i = half * 16 + j;
                const std::uint32_t packed_codes =
                    PackFourExponentCodes(a0, a1, a2, i);
                const uint2 reduced = ReduceFullFourCodes(
                    sign_values, b_values0, b_values1, j, i,
                    packed_codes, exceptional_a, exponent_base_a,
                    shared_outliers, outlier_base_a);
                reduced_pairs[j / 2] = reduced.x;
                reduced_pairs[j / 2 + 1] = reduced.y;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j += 2) {
                const int i = half * 16 + j;
                const std::uint32_t index = first + i;
                if (index + 1 < elements) {
                    const unsigned code0 = ((a0 >> i) & 1u) |
                        (((a1 >> i) & 1u) << 1) |
                        (((a2 >> i) & 1u) << 2);
                    const unsigned code1 = ((a0 >> (i + 1)) & 1u) |
                        (((a1 >> (i + 1)) & 1u) << 1) |
                        (((a2 >> (i + 1)) & 1u) << 2);
                    const auto exponent0 = DecodeExponent(
                        code0, exponent_base_a, exceptional_a, i,
                        shared_outliers, outlier_base_a);
                    const auto exponent1 = DecodeExponent(
                        code1, exponent_base_a, exceptional_a, i + 1,
                        shared_outliers, outlier_base_a);
                    const __nv_bfloat162 va(
                        __ushort_as_bfloat16(JoinBf16(
                            Uint4Byte(sign_values, j), exponent0)),
                        __ushort_as_bfloat16(JoinBf16(
                            Uint4Byte(sign_values, j + 1), exponent1)));
                    const uint4& b_values =
                        j < 8 ? b_values0 : b_values1;
                    const int b_element = j & 7;
                    const __nv_bfloat162 vb(
                        __ushort_as_bfloat16(Uint4Bf16(
                            b_values, b_element)),
                        __ushort_as_bfloat16(Uint4Bf16(
                            b_values, b_element + 1)));
                    const __nv_bfloat162 sum = __hadd2(va, vb);
                    reduced_pairs[j / 2] =
                        static_cast<std::uint32_t>(
                            __bfloat16_as_ushort(sum.x)) |
                        (static_cast<std::uint32_t>(
                            __bfloat16_as_ushort(sum.y)) << 16);
                } else if (index < elements) {
                    const unsigned code = ((a0 >> i) & 1u) |
                        (((a1 >> i) & 1u) << 1) |
                        (((a2 >> i) & 1u) << 2);
                    const auto exponent = DecodeExponent(
                        code, exponent_base_a, exceptional_a, i,
                        shared_outliers, outlier_base_a);
                    const uint4& b_values =
                        j < 8 ? b_values0 : b_values1;
                    reduced_pairs[j / 2] =
                        __bfloat16_as_ushort(__hadd(
                            __ushort_as_bfloat16(JoinBf16(
                                Uint4Byte(sign_values, j), exponent)),
                            __ushort_as_bfloat16(Uint4Bf16(
                                b_values, j & 7))));
                } else {
                    reduced_pairs[j / 2] = 0;
                }
            }
        }
        if (Op == SegmentOp::Copy) {
            auto* pairs = reinterpret_cast<uint4*>(shared_b_bytes);
            pairs[SwizzleBf16Tile(b_tile)] = make_uint4(reduced_pairs[0], reduced_pairs[1],
                                                        reduced_pairs[2], reduced_pairs[3]);
            pairs[SwizzleBf16Tile(b_tile + 1)] = make_uint4(reduced_pairs[4], reduced_pairs[5],
                                                            reduced_pairs[6], reduced_pairs[7]);
            continue;
        }
        uint4 encoded_sign = make_uint4(0, 0, 0, 0);
        uint3 bitmap = make_uint3(0, 0, 0);
        auto* exponent_scratch = shared_b_bytes +
            SwizzleBf16Tile(lane * 4 + half) * kBytesPerTile;
        if (full_segment) {
            #pragma unroll
            for (int j = 0; j < 16; j += 4) {
                const int i = half * 16 + j;
                const std::uint32_t pair0 = reduced_pairs[j / 2];
                const std::uint32_t pair1 = reduced_pairs[j / 2 + 1];
                const std::uint32_t low_bytes =
                    __byte_perm(pair0, pair1, 0x6420);
                const std::uint32_t high_bytes =
                    __byte_perm(pair0, pair1, 0x7531);
                const std::uint32_t sign_mantissa =
                    (low_bytes & 0x7f7f7f7fu) |
                    (high_bytes & 0x80808080u);
                std::uint32_t* encoded_word = j < 4 ? &encoded_sign.x :
                    j < 8 ? &encoded_sign.y :
                    j < 12 ? &encoded_sign.z : &encoded_sign.w;
                *encoded_word = sign_mantissa;
                const std::uint32_t exponents = __byte_perm(
                    pair0 >> 7, pair1 >> 7, 0x6420);
                const std::uint32_t exponent_bases =
                    static_cast<std::uint32_t>(output_exponent_base) *
                    0x01010101u;
                const std::uint32_t deltas =
                    __vsub4(exponents, exponent_bases);
                const std::uint32_t in_window =
                    __vcmpltu4(deltas, 0x07070707u);
                const std::uint32_t codes =
                    __vadd4(deltas, 0x01010101u) & in_window;
                const std::uint32_t code_bits0 =
                    GatherFourByteLowBits(codes);
                const std::uint32_t code_bits1 =
                    GatherFourByteLowBits(codes >> 1);
                const std::uint32_t code_bits2 =
                    GatherFourByteLowBits(codes >> 2);
                const std::uint32_t exceptional =
                    ~(code_bits0 | code_bits1 | code_bits2) & 0xfu;
                if (exceptional != 0) {
                    *reinterpret_cast<std::uint32_t*>(
                        exponent_scratch + j) = exponents;
                }
                bitmap.x |= code_bits0 << i;
                bitmap.y |= code_bits1 << i;
                bitmap.z |= code_bits2 << i;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int i = half * 16 + j;
                const std::uint32_t index = first + i;
                if (index < elements) {
                    const std::uint16_t bits = static_cast<std::uint16_t>(
                        reduced_pairs[j / 2] >> ((j & 1) * 16));
                    const std::uint32_t sign_mantissa =
                        ((bits >> 8) & 0x80u) | (bits & 0x7fu);
                    std::uint32_t* encoded_word = j < 4 ? &encoded_sign.x :
                        j < 8 ? &encoded_sign.y :
                        j < 12 ? &encoded_sign.z : &encoded_sign.w;
                    *encoded_word |= sign_mantissa << ((j & 3) * 8);
                    const unsigned exponent = (bits >> 7) & 0xffu;
                    const unsigned delta = exponent - output_exponent_base;
                    const unsigned code = delta < 7 ? delta + 1 : 0;
                    if (code == 0) exponent_scratch[j] = exponent;
                    bitmap.x |= ((code >> 0) & 1u) << i;
                    bitmap.y |= ((code >> 1) & 1u) << i;
                    bitmap.z |= ((code >> 2) & 1u) << i;
                }
            }
        }
        bitmap_half[half] = bitmap;
        const unsigned output_tile = lane * 2 + half;
        *reinterpret_cast<uint4*>(shared_sign_mantissa +
            SwizzleSignTile(output_tile) * kBytesPerTile) = encoded_sign;
    }
    if (Op == SegmentOp::Copy) {
        __syncwarp();
        for (unsigned tile = lane; tile * 8 < segment_elements; tile += kWarpSize) {
            unsigned i = tile * 8;
            auto* dst = dense_output + segment_first + i;
            auto* src = shared_b + SwizzleBf16Tile(tile) * 8;
            if (i + 8 <= segment_elements && (reinterpret_cast<uintptr_t>(dst) & 15) == 0)
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            else
                for (unsigned j = 0; j < min(8u, segment_elements - i); ++j) dst[j] = src[j];
        }
        __syncwarp();
        return;
    }
    const std::uint32_t out0 = bitmap_half[0].x | bitmap_half[1].x;
    const std::uint32_t out1 = bitmap_half[0].y | bitmap_half[1].y;
    const std::uint32_t out2 = bitmap_half[0].z | bitmap_half[1].z;
    if (active_word) {
        output.bitmap0[word] = out0;
        output.bitmap1[word] = out1;
        output.bitmap2[word] = out2;
    }
    __syncwarp();
    for (std::uint32_t tile = lane;
         tile < kSignTilesPerWarp;
         tile += kWarpSize) {
        const std::uint32_t offset = tile * kBytesPerTile;
        const std::uint32_t shared_offset =
            SwizzleSignTile(static_cast<unsigned>(tile)) * kBytesPerTile;
        if (FullSegment || offset + kBytesPerTile <= segment_elements) {
            *reinterpret_cast<uint4*>(
                output.sign_mantissa + segment_first + offset) =
                *reinterpret_cast<const uint4*>(
                    shared_sign_mantissa + shared_offset);
        } else {
            for (std::uint32_t i = offset; i < segment_elements; ++i) {
                output.sign_mantissa[segment_first + i] =
                    shared_sign_mantissa[shared_offset + i - offset];
            }
        }
    }
    const std::uint32_t exceptional_out = valid & ~(out0 | out1 | out2);
    const int local_output_count = __popc(exceptional_out);
    int position = WarpExclusivePrefix(local_output_count, lane);
    if (lane == kWarpSize - 1) {
        *segment_count = position + local_output_count;
    }
    std::uint32_t pending = exceptional_out;
    while (pending != 0) {
        const int bit = __ffs(static_cast<int>(pending)) - 1;
        const auto* exponent_scratch = shared_b_bytes +
            SwizzleBf16Tile(lane * 4 + (bit >> 4)) * kBytesPerTile;
        segment_slot[position++] = exponent_scratch[bit & 15];
        pending &= pending - 1;
    }
    __syncwarp();
}

// A compactor warp owns one logical block. Eight lanes copy each segment.
template<bool PackInShared = false>
__device__ __forceinline__ void CopyDreBlock(
        const std::uint8_t* block_slots, EncodedOutput output,
        unsigned block, unsigned offset, int count, std::uint8_t* scratch = nullptr) {
    constexpr int lanes_per_segment = kWarpSize / kSegmentsPerBlock;
    const int lane = threadIdx.x % kWarpSize;
    const int segment = lane / lanes_per_segment;
    const int i = lane % lanes_per_segment;
    if (i == 0) {
        auto* offsets = output.warp_offsets + block * kDreWarpOffsetsPerBlock;
        offsets[segment] = offset;
        if (segment == 0) output.block_offsets[block] = offset;
        if (segment == kSegmentsPerBlock - 1) offsets[kSegmentsPerBlock] = offset + count;
    }
    const auto* slot = block_slots + segment * kDreValuesPerWarp;
    if constexpr (PackInShared) {
        static_assert(kDreWarpSharedBytes >= kElementsPerBlock + 3, "Block staging capacity");
        unsigned begin = __shfl_sync(0xffffffffu, offset, 0);
        unsigned end = __shfl_sync(0xffffffffu, offset + count, kWarpSize - 1);
        // Match the FIFO alignment so both ends of each word copy are aligned.
        auto* packed = scratch + (begin & 3);
        for (int j = i; j < count; j += lanes_per_segment)
            packed[offset - begin + j] = slot[j];
        __syncwarp();
        unsigned bytes = end - begin;
        unsigned head = min(bytes, (0u - begin) & 3u);
        auto* dst = output.outliers + begin;
        if (lane < head) dst[lane] = packed[lane];
        unsigned words = (bytes - head) / 4;
        for (unsigned j = lane; j < words; j += kWarpSize)
            reinterpret_cast<std::uint32_t*>(dst + head)[j] =
                reinterpret_cast<const std::uint32_t*>(packed + head)[j];
        unsigned tail = head + words * 4 + lane;
        if (tail < bytes) dst[tail] = packed[tail];
        __syncwarp(); // All lanes must finish reading before the next block reuses scratch.
    } else {
        for (int j = i; j < count; j += lanes_per_segment) output.outliers[offset + j] = slot[j];
    }
}

template<bool FourFullRounds, bool PackInShared = false>
__device__ __forceinline__ unsigned CompactSingleCta(
        const std::uint16_t* counts, const std::uint8_t* slots,
        EncodedOutput output, unsigned first_block, unsigned blocks, unsigned base,
        std::uint8_t* scratch = nullptr) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const bool valid_pair = FourFullRounds || 2 * lane < blocks * kSegmentsPerBlock;
    const int even = valid_pair ? counts[2 * lane] : 0;
    const int odd = valid_pair ? counts[2 * lane + 1] : 0;
    const int pair = even + odd;
    const int prefix = WarpExclusivePrefix(pair, lane);
    const int segment = lane / (kWarpSize / kSegmentsPerBlock);
    const int source = 2 * warp + segment / 2;
    const int first_count = __shfl_sync(0xffffffffu, even, source);
    const int second_count = __shfl_sync(0xffffffffu, odd, source);
    const int segment_base = __shfl_sync(0xffffffffu, prefix, source) +
        ((segment & 1) ? first_count : 0);
    const int count = (segment & 1) ? second_count : first_count;
    const int total = __shfl_sync(0xffffffffu, prefix + pair, 31);
    if (FourFullRounds || warp < blocks) {
        CopyDreBlock<PackInShared>(slots + warp * kElementsPerBlock, output, first_block + warp,
                                   base + segment_base, count, scratch);
    }
    return base + total;
}

} // namespace device
} // namespace zipccl
