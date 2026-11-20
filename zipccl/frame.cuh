#pragma once

#include "zipccl.cuh"

namespace zipccl {
namespace device {
constexpr int kWarpSize = 32;
constexpr int kDreValuesPerWarp = 1024;
constexpr int kSegmentsPerBlock = kElementsPerBlock / kDreValuesPerWarp;
constexpr int kBytesPerTile = 16;
constexpr int kDreWarpOffsetsPerBlock = kSegmentsPerBlock + 1;
constexpr int kDreSignBytes = kDreValuesPerWarp;
constexpr int kDreBf16Bytes = kDreValuesPerWarp * sizeof(std::uint16_t);
constexpr int kDreBitmapBytes = kWarpSize * sizeof(std::uint32_t);
constexpr int kDreWarpSharedBytes = kDreSignBytes + kDreBf16Bytes +
    3 * kDreBitmapBytes + kDreValuesPerWarp + kBytesPerTile;

// Keep the 19-warp ring CTA within the SM86 opt-in shared memory limit.
// Larger slices reuse the same segment counts in successive batches.
__host__ __device__ constexpr int RingDreScratchBytes(int warps) {
    constexpr int elements = (1 << 20) / sizeof(std::uint16_t);
    return warps * kDreWarpSharedBytes +
        ((elements + kDreValuesPerWarp - 1) / kDreValuesPerWarp) * sizeof(std::uint16_t);
}

enum class RingFrameFormat : uint16_t { Compressed, Raw };

struct alignas(16) RingFrameHeader {
  unsigned elements;
  unsigned contributors;
  uint16_t base;
  RingFrameFormat format;
  float sigma;
};
static_assert(sizeof(RingFrameHeader) == 16, "Ring frame payload must stay 16-byte aligned");

__host__ __device__ constexpr Layout FrameLayout(std::size_t elements) {
    Layout layout{};
    layout.bitmap_words = (elements + 31) / 32;
    layout.blocks = (elements + kElementsPerBlock - 1) / kElementsPerBlock;
    layout.sign_mantissa = 0;
    layout.bitmap[0] = (elements + alignof(std::uint32_t) - 1) &
        ~(alignof(std::uint32_t) - 1);
    layout.bitmap[1] = layout.bitmap[0] + layout.bitmap_words * sizeof(std::uint32_t);
    layout.bitmap[2] = layout.bitmap[1] + layout.bitmap_words * sizeof(std::uint32_t);
    layout.block_offsets = layout.bitmap[2] + layout.bitmap_words * sizeof(std::uint32_t);
    layout.warp_offsets = layout.block_offsets +
        layout.blocks * sizeof(std::uint32_t);
    const std::size_t metadata_end = layout.warp_offsets + layout.blocks *
        kDreWarpOffsetsPerBlock * sizeof(std::uint32_t);
    layout.outliers = (metadata_end + kBytesPerTile - 1) &
        ~(kBytesPerTile - 1);
    layout.capacity = layout.outliers + elements + kBytesPerTile - 1;
    return layout;
}

} // namespace device
} // namespace zipccl
