#pragma once

#include "zipccl.cuh"

namespace zipccl {
namespace device {
constexpr int kWarpSize = 32;
constexpr int kDreValuesPerWarp = 1024;
constexpr int kSegmentsPerBlock = kElementsPerBlock / kDreValuesPerWarp;
constexpr int kBytesPerTile = 16;
constexpr int kDreWarpOffsetsPerBlock = kSegmentsPerBlock + 1;

struct alignas(16) RingFrameHeader {
  unsigned elements;
  unsigned contributors;
  int base;
  float sigma;
};

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
