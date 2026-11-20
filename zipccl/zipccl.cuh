#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace zipccl {

constexpr std::size_t kElementsPerBlock = 4096;

// Byte offsets of each stream in the compressed buffer. The outlier stream is
// provisioned for the worst case (one byte per input element).
struct Layout {
    std::size_t sign_mantissa;
    std::size_t bitmap[3];
    std::size_t block_offsets;
    std::size_t warp_offsets;
    std::size_t outliers;
    std::size_t capacity;
    std::size_t bitmap_words;
    std::size_t blocks;
};

Layout GetLayout(std::size_t elements);
std::size_t MaxCompressedBytes(std::size_t elements);

// Optimal lower exponent of the seven-value window from the Gaussian model in
// the paper. The returned value is the biased BF16 exponent E=x+127.
std::uint8_t ComputeExponentBase(double standard_deviation);

// input/output are raw BF16 bit patterns. No CUDA BF16 library or framework is
// required. outlier_count is a single uint32_t in device memory. max_blocks=0
// launches one CUDA block per 4096 values; a positive value caps the physical
// grid and makes each block process multiple chunks with a grid-stride loop.
cudaError_t CompressAsync(const std::uint16_t* input,
                          std::size_t elements,
                          std::uint8_t exponent_base,
                          std::uint8_t* compressed,
                          std::size_t compressed_capacity,
                          std::uint32_t* outlier_count,
                          cudaStream_t stream = nullptr,
                          std::uint32_t max_blocks = 0);

cudaError_t DecompressAsync(const std::uint8_t* compressed,
                            std::size_t elements,
                            std::uint8_t exponent_base,
                            std::uint16_t* output,
                            cudaStream_t stream = nullptr,
                            std::uint32_t max_blocks = 0);

constexpr std::size_t kMinDreWorkspaceBytes = 1u << 20;

// Decode, add raw BF16, and encode. Supply a separate device workspace of at
// least kMinDreWorkspaceBytes. The stream must finish using it before another
// stream reuses it. Larger workspaces allow more CTA rounds per batch.
cudaError_t DecodeReduceEncodeAsync(const std::uint8_t* compressed_a,
                                    std::uint8_t exponent_base_a,
                                    const std::uint16_t* uncompressed_b,
                                    std::size_t elements,
                                    std::uint8_t output_exponent_base,
                                    std::uint8_t* compressed_output,
                                    std::size_t compressed_output_capacity,
                                    std::uint32_t* output_outlier_count,
                                    void* workspace, std::size_t workspace_bytes,
                                    cudaStream_t stream = nullptr,
                                    std::uint32_t max_blocks = 0);

// Decode one compressed BF16 array, add a raw BF16 array with native BF16
// arithmetic, and write a raw BF16 result without an intermediate decode.
cudaError_t DecodeReduceCopyAsync(const std::uint8_t* compressed_a,
                                  std::uint8_t exponent_base_a,
                                  const std::uint16_t* uncompressed_b,
                                  std::size_t elements,
                                  std::uint16_t* output,
                                  cudaStream_t stream = nullptr,
                                  std::uint32_t max_blocks = 0);

std::size_t CompressedBytes(std::size_t elements, std::uint32_t outlier_count);

}  // namespace zipccl
