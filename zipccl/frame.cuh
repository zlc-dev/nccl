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

__host__ __device__ constexpr int RingDreScratchBytes(int warps) {
    return warps * kDreWarpSharedBytes;
}

// Per-channel device memory: one outlier slot byte per element, then one count per segment.
__host__ __device__ constexpr size_t RingWorkspaceBytes(size_t elements) {
    return elements + (elements / kDreValuesPerWarp) * sizeof(std::uint16_t);
}

enum class RingFrameFormat : uint16_t { Compressed, Raw, Fixed };

constexpr size_t kRingFrameAlignment = 128;
struct alignas(kRingFrameAlignment) RingFrameHeader {
  unsigned elements;
  unsigned contributors;
  uint16_t base;
  RingFrameFormat format;
  float sigma;
};
static_assert(sizeof(RingFrameHeader) == kRingFrameAlignment, "Ring payload must start on a transaction boundary");

__host__ __device__ constexpr size_t AlignFrameOffset(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

__host__ __device__ constexpr Layout FrameLayout(std::size_t elements, size_t alignment = alignof(std::uint32_t)) {
    Layout layout{};
    layout.bitmap_words = (elements + 31) / 32;
    layout.blocks = (elements + kElementsPerBlock - 1) / kElementsPerBlock;
    layout.sign_mantissa = 0;
    layout.bitmap[0] = AlignFrameOffset(elements, alignment);
    const auto bitmapStride = AlignFrameOffset(layout.bitmap_words * sizeof(std::uint32_t), alignment);
    layout.bitmap[1] = layout.bitmap[0] + bitmapStride;
    layout.bitmap[2] = layout.bitmap[1] + bitmapStride;
    layout.block_offsets = layout.bitmap[2] + bitmapStride;
    layout.warp_offsets = AlignFrameOffset(layout.block_offsets + layout.blocks * sizeof(std::uint32_t), alignment);
    const std::size_t metadata_end = layout.warp_offsets + layout.blocks *
        kDreWarpOffsetsPerBlock * sizeof(std::uint32_t);
    layout.outliers = AlignFrameOffset(metadata_end, alignment < kBytesPerTile ? kBytesPerTile : alignment);
    layout.capacity = layout.outliers + elements + kBytesPerTile - 1;
    return layout;
}

} // namespace device
} // namespace zipccl
