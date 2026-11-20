#include "single_cta.cuh"

#include <cuda_bf16.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace zipccl {
namespace {

using namespace device;

enum class DreBatchMode { General, FullBatches, FullBatchesWithTail };

template<DreBatchMode Mode>
__global__ __launch_bounds__(kThreads, 1)
void DecodeReduceEncodeKernel(EncodedInput input_a, std::uint8_t exponent_base_a,
        const std::uint16_t* __restrict__ input_b, std::uint32_t elements,
        std::uint8_t output_exponent_base, EncodedOutput output,
        std::uint32_t* __restrict__ output_outlier_count, unsigned logical_blocks,
        int batch_rounds, std::uint8_t* __restrict__ slots) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto* counts = reinterpret_cast<std::uint16_t*>(shared_bytes + kDreFixedBytes);
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const unsigned stride = gridDim.x * kBlocksPerRound;
    unsigned running_base = 0;
    for (unsigned batch = blockIdx.x * kBlocksPerRound; batch < logical_blocks;
         batch += batch_rounds * stride) {
        const int rounds = min(static_cast<unsigned>(batch_rounds),
                               1u + (logical_blocks - 1 - batch) / stride);
        for (int r = 0; r < rounds; ++r) {
            const unsigned segment = (batch + r * stride) * kSegmentsPerBlock + warp;
            auto* round_counts = counts + r * kWarpsPerCta;
            auto* round_slots = slots + (std::size_t(r) * gridDim.x + blockIdx.x) *
                kDreRoundSlotsBytes;
            if (Mode == DreBatchMode::FullBatches ||
                segment < elements / kDreValuesPerWarp) {
                ProcessSegment<true>(input_a, exponent_base_a, input_b, elements,
                    output_exponent_base, output, segment,
                    round_counts + warp, round_slots + warp * kDreValuesPerWarp, shared_bytes);
            } else {
                ProcessSegment<false>(input_a, exponent_base_a, input_b, elements,
                    output_exponent_base, output, segment,
                    round_counts + warp, round_slots + warp * kDreValuesPerWarp, shared_bytes);
            }
        }
        __syncthreads();
        if constexpr (Mode == DreBatchMode::General) {
            // Across CTAs, each compactor warp reserves its own block's contiguous output.
            for (int index = warp; index < rounds * kBlocksPerRound; index += kWarpsPerCta) {
                const int r = index / kBlocksPerRound;
                const int local_block = index % kBlocksPerRound;
                const unsigned block = batch + r * stride + local_block;
                if (block >= logical_blocks) continue;
                const int value = lane < kSegmentsPerBlock ? counts[index * kSegmentsPerBlock + lane] : 0;
                const int prefix = WarpExclusivePrefix(value, lane);
                const int total = __shfl_sync(0xffffffffu, prefix, kSegmentsPerBlock);
                unsigned base = lane == 0 ? atomicAdd(output_outlier_count, total) : 0;
                base = __shfl_sync(0xffffffffu, base, 0);
                const int segment = lane / (kWarpSize / kSegmentsPerBlock);
                const int offset = __shfl_sync(0xffffffffu, prefix, segment);
                const int count = __shfl_sync(0xffffffffu, value, segment);
                const auto* block_slots = slots + (std::size_t(r) * gridDim.x + blockIdx.x) *
                    kDreRoundSlotsBytes + local_block * kElementsPerBlock;
                CopyDreBlock(block_slots, output, block, base + offset, count);
            }
        } else {
            const unsigned blocks = min(logical_blocks - batch, unsigned(rounds * kBlocksPerRound));
            unsigned i = 0;
            for (; i + kWarpsPerCta <= blocks; i += kWarpsPerCta) {
                running_base = CompactSingleCta<true>(counts + i * kSegmentsPerBlock,
                    slots + i * kElementsPerBlock, output, batch + i, kWarpsPerCta, running_base);
            }
            if (i < blocks) {
                running_base = CompactSingleCta<false>(counts + i * kSegmentsPerBlock,
                    slots + i * kElementsPerBlock, output, batch + i, blocks - i, running_base);
            }
        }
        // Other warps may still read the slots this warp will reuse in the next batch.
        if (batch + rounds * stride < logical_blocks) __syncthreads();
    }
    if (Mode != DreBatchMode::General && threadIdx.x == 0) {
        *output_outlier_count = running_base;
    }
}

__global__ __launch_bounds__(kThreads, 1)
void CompressKernel(const std::uint16_t* __restrict__ input,
                    std::size_t elements,
                    std::uint8_t exponent_base,
                    EncodedOutput output,
                    std::uint32_t* __restrict__ outlier_count,
                    std::size_t logical_blocks) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const std::size_t first_block =
        static_cast<std::size_t>(blockIdx.x) * kWarpsPerCta + warp;
    const std::size_t block_stride =
        static_cast<std::size_t>(gridDim.x) * kWarpsPerCta;

    for (std::size_t logical_block = first_block;
         logical_block < logical_blocks;
         logical_block += block_stride) {
        const std::size_t block_first = logical_block * kElementsPerBlock;
        const std::size_t block_elements =
            min(kElementsPerBlock, elements - block_first);
        const int words = static_cast<int>((block_elements + kWarpSize - 1) / kWarpSize);
        const std::size_t first_word = logical_block * kWordsPerBlock;
        unsigned block_count = 0;
        unsigned segment_counts[kSegmentsPerBlock] = {0, 0, 0, 0};

        for (int w = 0; w < words; ++w) {
            const std::size_t index = block_first +
                static_cast<std::size_t>(w) * kWarpSize + lane;
            const bool valid = index < elements;
            unsigned code = 0;
            if (valid) {
                const std::uint16_t bits = input[index];
                output.sign_mantissa[index] = static_cast<std::uint8_t>(
                    ((bits >> 8) & 0x80u) | (bits & 0x7fu));
                const unsigned exponent = (bits >> 7) & 0xffu;
                const unsigned delta = exponent - exponent_base;
                code = delta < 7 ? delta + 1 : 0;
            }
            const std::uint32_t b0 = __ballot_sync(0xffffffffu, code & 1u);
            const std::uint32_t b1 = __ballot_sync(0xffffffffu, code & 2u);
            const std::uint32_t b2 = __ballot_sync(0xffffffffu, code & 4u);
            if (lane == 0) output.bitmap0[first_word + w] = b0;
            if (lane == 1) output.bitmap1[first_word + w] = b1;
            if (lane == 2) output.bitmap2[first_word + w] = b2;
            const unsigned word_outliers =
                __popc(__ballot_sync(0xffffffffu, valid && code == 0));
            block_count += word_outliers;
            segment_counts[w / kWarpSize] += word_outliers;
        }

        std::uint32_t block_base = 0;
        if (lane == 0) {
            block_base = atomicAdd(outlier_count, block_count);
            output.block_offsets[logical_block] = block_base;
            std::uint32_t segment_base = block_base;
#pragma unroll
            for (int segment = 0; segment < kSegmentsPerBlock; ++segment) {
                output.warp_offsets[
                    logical_block * kDreWarpOffsetsPerBlock + segment] =
                    segment_base;
                segment_base += segment_counts[segment];
            }
            output.warp_offsets[
                logical_block * kDreWarpOffsetsPerBlock + kSegmentsPerBlock] =
                segment_base;
        }
        block_base = __shfl_sync(0xffffffffu, block_base, 0);

        unsigned running = 0;
        for (int w = 0; w < words; ++w) {
            const std::size_t index = block_first +
                static_cast<std::size_t>(w) * kWarpSize + lane;
            const bool valid = index < elements;
            unsigned exponent = 0;
            bool exceptional = false;
            if (valid) {
                exponent = (input[index] >> 7) & 0xffu;
                const unsigned delta = exponent - exponent_base;
                exceptional = delta >= 7;
            }
            const std::uint32_t mask = __ballot_sync(0xffffffffu, exceptional);
            if (exceptional) {
                output.outliers[block_base + running + LanePrefix(mask, lane)] =
                    static_cast<std::uint8_t>(exponent);
            }
            running += __popc(mask);
        }
    }
}

__global__ __launch_bounds__(kThreads, 1)
void DecompressKernel(EncodedInput input,
                      std::size_t elements,
                      std::uint8_t exponent_base,
                      std::uint16_t* __restrict__ output,
                      std::size_t logical_blocks) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const std::size_t first_block =
        static_cast<std::size_t>(blockIdx.x) * kWarpsPerCta + warp;
    const std::size_t block_stride =
        static_cast<std::size_t>(gridDim.x) * kWarpsPerCta;

    for (std::size_t logical_block = first_block;
         logical_block < logical_blocks;
         logical_block += block_stride) {
        const std::size_t block_first = logical_block * kElementsPerBlock;
        const std::size_t block_elements =
            min(kElementsPerBlock, elements - block_first);
        const int words = static_cast<int>((block_elements + kWarpSize - 1) / kWarpSize);
        const std::size_t first_word = logical_block * kWordsPerBlock;
        std::uint32_t block_base = lane == 0
            ? input.block_offsets[logical_block] : 0u;
        block_base = __shfl_sync(0xffffffffu, block_base, 0);
        unsigned running = 0;

        for (int w = 0; w < words; ++w) {
            const std::size_t word = first_word + w;
            const std::size_t index = block_first +
                static_cast<std::size_t>(w) * kWarpSize + lane;
            const bool valid_lane = index < elements;
            const std::uint32_t valid = __ballot_sync(0xffffffffu, valid_lane);
            const std::uint32_t b0 = LoadBitmapWord(input.bitmap0, word, 0);
            const std::uint32_t b1 = LoadBitmapWord(input.bitmap1, word, 1);
            const std::uint32_t b2 = LoadBitmapWord(input.bitmap2, word, 2);
            const std::uint32_t exceptional = valid & ~(b0 | b1 | b2);
            if (valid_lane) {
                const unsigned code = ((b0 >> lane) & 1u) |
                    (((b1 >> lane) & 1u) << 1) | (((b2 >> lane) & 1u) << 2);
                const unsigned exponent = code == 0
                    ? input.outliers[block_base + running + LanePrefix(exceptional, lane)]
                    : exponent_base + code - 1u;
                output[index] = JoinBf16(input.sign_mantissa[index], exponent);
            }
            running += __popc(exceptional);
        }
    }
}

EncodedInput MakeInput(const std::uint8_t* data, const Layout& layout) {
    return {data + layout.sign_mantissa,
            reinterpret_cast<const std::uint32_t*>(data + layout.bitmap[0]),
            reinterpret_cast<const std::uint32_t*>(data + layout.bitmap[1]),
            reinterpret_cast<const std::uint32_t*>(data + layout.bitmap[2]),
            reinterpret_cast<const std::uint32_t*>(data + layout.block_offsets),
            reinterpret_cast<const std::uint32_t*>(data + layout.warp_offsets),
            data + layout.outliers};
}

EncodedOutput MakeOutput(std::uint8_t* data, const Layout& layout) {
    return {data + layout.sign_mantissa,
            reinterpret_cast<std::uint32_t*>(data + layout.bitmap[0]),
            reinterpret_cast<std::uint32_t*>(data + layout.bitmap[1]),
            reinterpret_cast<std::uint32_t*>(data + layout.bitmap[2]),
            reinterpret_cast<std::uint32_t*>(data + layout.block_offsets),
            reinterpret_cast<std::uint32_t*>(data + layout.warp_offsets),
            data + layout.outliers};
}

}  // namespace

Layout GetLayout(std::size_t elements) { return device::FrameLayout(elements); }

std::size_t MaxCompressedBytes(std::size_t elements) {
    return GetLayout(elements).capacity;
}

std::size_t CompressedBytes(std::size_t elements, std::uint32_t outlier_count) {
    return GetLayout(elements).outliers + outlier_count;
}

std::uint8_t ComputeExponentBase(double standard_deviation) {
    const double kCorrection = 0.5 * std::log2(14.0 * 0.6931471805599453 / 16383.0);
    const long base = std::lround(std::log2(standard_deviation) + 127.0 + kCorrection);
    return static_cast<std::uint8_t>(std::clamp(base, 0l, 249l));
}

cudaError_t CompressAsync(const std::uint16_t* input,
                          std::size_t elements,
                          std::uint8_t exponent_base,
                          std::uint8_t* compressed,
                          std::size_t compressed_capacity,
                          std::uint32_t* outlier_count,
                          cudaStream_t stream,
                          std::uint32_t max_blocks) {
    if (!input || !compressed || !outlier_count || elements == 0) {
        return cudaErrorInvalidValue;
    }
    const Layout layout = GetLayout(elements);
    if (compressed_capacity < layout.capacity) return cudaErrorInvalidValue;
    cudaError_t error = cudaMemsetAsync(outlier_count, 0, sizeof(*outlier_count), stream);
    if (error != cudaSuccess) return error;
    const std::uint32_t required_ctas = static_cast<std::uint32_t>(
        (layout.blocks + kWarpsPerCta - 1) / kWarpsPerCta);
    const std::uint32_t launch_ctas = max_blocks == 0
        ? required_ctas : std::min(required_ctas, max_blocks);
    CompressKernel<<<launch_ctas, kThreads, 0, stream>>>(
        input, elements, exponent_base, MakeOutput(compressed, layout),
        outlier_count, layout.blocks);
    return cudaGetLastError();
}

cudaError_t DecompressAsync(const std::uint8_t* compressed,
                            std::size_t elements,
                            std::uint8_t exponent_base,
                            std::uint16_t* output,
                            cudaStream_t stream,
                            std::uint32_t max_blocks) {
    if (!compressed || !output || elements == 0) return cudaErrorInvalidValue;
    const Layout layout = GetLayout(elements);
    const std::uint32_t required_ctas = static_cast<std::uint32_t>(
        (layout.blocks + kWarpsPerCta - 1) / kWarpsPerCta);
    const std::uint32_t launch_ctas = max_blocks == 0
        ? required_ctas : std::min(required_ctas, max_blocks);
    DecompressKernel<<<launch_ctas, kThreads, 0, stream>>>(
        MakeInput(compressed, layout), elements, exponent_base, output, layout.blocks);
    return cudaGetLastError();
}

cudaError_t DecodeReduceEncodeAsync(const std::uint8_t* compressed_a,
                                    std::uint8_t exponent_base_a,
                                    const std::uint16_t* uncompressed_b,
                                    std::size_t elements,
                                    std::uint8_t output_exponent_base,
                                    std::uint8_t* compressed_output,
                                    std::size_t compressed_output_capacity,
                                    std::uint32_t* output_outlier_count,
                                    void* workspace, std::size_t workspace_bytes,
                                    cudaStream_t stream,
                                    std::uint32_t max_blocks) {
    if (!compressed_a || !uncompressed_b || !compressed_output ||
        !output_outlier_count || !workspace || workspace_bytes < kMinDreWorkspaceBytes ||
        elements == 0 ||
        elements > static_cast<std::size_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    const Layout layout = GetLayout(elements);
    if (compressed_output_capacity < layout.capacity) return cudaErrorInvalidValue;
    const std::uint32_t required_ctas = static_cast<std::uint32_t>(
        (layout.blocks + kBlocksPerRound - 1) / kBlocksPerRound);
    // Each concurrent CTA needs one full round of worst-case slots.
    const std::size_t workspace_rounds = workspace_bytes / kDreRoundSlotsBytes;
    const auto cta_limit = max_blocks == 0 ? required_ctas : std::min(required_ctas, max_blocks);
    const auto launch_ctas = static_cast<std::uint32_t>(
        std::min<std::size_t>(cta_limit, workspace_rounds));
    cudaError_t error = cudaSuccess;
    if (launch_ctas > 1) {
        error = cudaMemsetAsync(output_outlier_count, 0,
                                sizeof(*output_outlier_count), stream);
        if (error != cudaSuccess) return error;
    }
    int device, shared_limit;
    error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    error = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess) return error;
    const int max_rounds = (shared_limit - kDreFixedBytes) / kDreRoundCountsBytes;
    if (max_rounds < 1) return cudaErrorInvalidConfiguration;
    const int batch_rounds = static_cast<int>(std::min<std::size_t>({
        static_cast<std::size_t>(max_rounds), workspace_rounds / launch_ctas,
        (required_ctas + launch_ctas - 1) / launch_ctas}));
    const int shared_size = kDreFixedBytes + batch_rounds * kDreRoundCountsBytes;
    auto kernel = DecodeReduceEncodeKernel<DreBatchMode::General>;
    if (launch_ctas == 1) {
        kernel = elements % (kBlocksPerRound * kElementsPerBlock) == 0
            ? DecodeReduceEncodeKernel<DreBatchMode::FullBatches>
            : DecodeReduceEncodeKernel<DreBatchMode::FullBatchesWithTail>;
    }
    error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_limit);
    if (error != cudaSuccess) return error;
    kernel<<<launch_ctas, kThreads, shared_size, stream>>>(
        MakeInput(compressed_a, layout), exponent_base_a, uncompressed_b,
        static_cast<std::uint32_t>(elements),
        output_exponent_base, MakeOutput(compressed_output, layout),
        output_outlier_count, static_cast<std::uint32_t>(layout.blocks), batch_rounds,
        static_cast<std::uint8_t*>(workspace));
    return cudaGetLastError();
}

}  // namespace zipccl
