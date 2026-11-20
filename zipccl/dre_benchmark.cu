#include "zipccl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(error_));                               \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)

namespace {

constexpr int kSamples = 7;
constexpr int kTimingBatch = 64;

__global__ void EvictL2Kernel(const uint4* data, std::size_t values,
                              std::uint32_t* sink) {
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::uint32_t checksum = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
         i < values; i += stride) {
        const uint4 value = data[i];
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        checksum ^= __shfl_down_sync(0xffffffffu, checksum, offset);
    }
    if ((threadIdx.x & 31) == 0) atomicXor(sink, checksum);
}

std::uint16_t FloatToBf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float Bf16ToFloat(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

double Rate(std::size_t bytes, float milliseconds) {
    return static_cast<double>(bytes) / 1.0e6 / milliseconds;
}

int WarmupDre(const std::uint8_t* compressed_a,
              std::uint8_t input_base,
              const std::uint16_t* input_b,
              std::size_t elements,
              std::uint8_t output_base,
              std::uint8_t* compressed_output,
              std::size_t capacity,
              std::uint32_t* output_count,
              void* workspace, std::size_t workspace_bytes,
              std::uint32_t max_ctas,
              float minimum_ms) {
    if (minimum_ms <= 0.0f) return 0;
    constexpr int kBatchIterations = 1000;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    int launches = 0;
    while (total_ms < minimum_ms) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < kBatchIterations; ++i) {
            CUDA_CHECK(zipccl::DecodeReduceEncodeAsync(
                compressed_a, input_base, input_b, elements, output_base,
                compressed_output, capacity, output_count, workspace, workspace_bytes, nullptr, max_ctas));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float batch_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&batch_ms, start, stop));
        total_ms += batch_ms;
        launches += kBatchIterations;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return launches;
}

float MeasureDre(const std::uint8_t* compressed_a,
                 std::uint8_t input_base,
                 const std::uint16_t* input_b,
                 std::size_t elements,
                 std::uint8_t output_base,
                 std::uint8_t* compressed_output,
                 std::size_t capacity,
                 std::uint32_t* output_count,
                 void* workspace, std::size_t workspace_bytes,
                 uint4* cache_scrub,
                 std::size_t cache_scrub_values,
                 std::uint32_t* cache_scrub_sink,
                 int cache_scrub_blocks,
                 std::uint32_t max_ctas,
                 int iterations) {
    cudaEvent_t starts[kTimingBatch], stops[kTimingBatch];
    for (int i = 0; i < kTimingBatch; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
    }
    std::vector<float> samples;
    samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
        float sample_ms = 0.0f;
        for (int first = 0; first < iterations; first += kTimingBatch) {
            const int batch = std::min(kTimingBatch, iterations - first);
            for (int i = 0; i < batch; ++i) {
                EvictL2Kernel<<<cache_scrub_blocks, 256>>>(
                    cache_scrub, cache_scrub_values, cache_scrub_sink);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(starts[i]));
                CUDA_CHECK(zipccl::DecodeReduceEncodeAsync(
                    compressed_a, input_base, input_b, elements, output_base,
                    compressed_output, capacity, output_count, workspace, workspace_bytes, nullptr, max_ctas));
                CUDA_CHECK(cudaEventRecord(stops[i]));
            }
            CUDA_CHECK(cudaEventSynchronize(stops[batch - 1]));
            for (int i = 0; i < batch; ++i) {
                float elapsed = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, starts[i], stops[i]));
                sample_ms += elapsed;
            }
        }
        samples.push_back(sample_ms / iterations);
    }
    for (int i = 0; i < kTimingBatch; ++i) {
        CUDA_CHECK(cudaEventDestroy(starts[i]));
        CUDA_CHECK(cudaEventDestroy(stops[i]));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t elements = argc > 1
        ? std::strtoull(argv[1], nullptr, 10) : 131072;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 10000;
    const double standard_deviation = argc > 3 ? std::atof(argv[3]) : 0.02;
    const float warmup_ms = argc > 4 ? std::atof(argv[4]) : 1500.0f;
    const std::uint32_t max_ctas = argc > 5
        ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 1u;
    const std::size_t workspace_bytes = argc > 6
        ? std::strtoull(argv[6], nullptr, 10) : zipccl::kMinDreWorkspaceBytes;
    if (workspace_bytes < zipccl::kMinDreWorkspaceBytes || elements == 0 || iterations <= 0 || !(standard_deviation > 0.0) ||
        !std::isfinite(standard_deviation) || warmup_ms < 0.0f || max_ctas == 0) {
        std::fprintf(stderr,
                     "usage: %s [elements] [iterations] [std>0] "
                     "[warmup_ms>=0] [max_ctas>0] [workspace_bytes>=1048576]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const std::uint8_t input_base =
        zipccl::ComputeExponentBase(standard_deviation);
    const double output_standard_deviation =
        std::sqrt(2.0) * standard_deviation;
    const std::uint8_t output_base =
        zipccl::ComputeExponentBase(output_standard_deviation);

    std::vector<std::uint16_t> input_a(elements), input_b(elements),
        expected(elements), actual(elements);
    std::mt19937 generator(12345);
    std::normal_distribution<float> distribution(
        0.0f, static_cast<float>(standard_deviation));
    for (std::size_t i = 0; i < elements; ++i) {
        input_a[i] = FloatToBf16(distribution(generator));
        input_b[i] = FloatToBf16(distribution(generator));
        expected[i] = FloatToBf16(Bf16ToFloat(input_a[i]) +
                                  Bf16ToFloat(input_b[i]));
    }

    const std::size_t raw_bytes = elements * sizeof(std::uint16_t);
    const std::size_t capacity = zipccl::MaxCompressedBytes(elements);
    const std::size_t logical_blocks =
        (elements + zipccl::kElementsPerBlock - 1) / zipccl::kElementsPerBlock;
    const std::size_t cache_scrub_bytes = std::max<std::size_t>(
        8u << 20, static_cast<std::size_t>(properties.l2CacheSize) * 2);
    const std::size_t cache_scrub_values =
        (cache_scrub_bytes + sizeof(uint4) - 1) / sizeof(uint4);
    const int cache_scrub_blocks = properties.multiProcessorCount * 4;
    std::uint16_t *d_a = nullptr, *d_b = nullptr, *d_actual = nullptr;
    std::uint8_t *d_compressed_a = nullptr, *d_compressed_output = nullptr;
    std::uint32_t *d_input_count = nullptr, *d_output_count = nullptr;
    std::uint32_t* d_cache_scrub_sink = nullptr;
    uint4* d_cache_scrub = nullptr;
    void* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    CUDA_CHECK(cudaMalloc(&d_a, raw_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, raw_bytes));
    CUDA_CHECK(cudaMalloc(&d_actual, raw_bytes));
    CUDA_CHECK(cudaMalloc(&d_compressed_a, capacity));
    CUDA_CHECK(cudaMalloc(&d_compressed_output, capacity));
    CUDA_CHECK(cudaMalloc(&d_input_count, sizeof(*d_input_count)));
    CUDA_CHECK(cudaMalloc(&d_output_count, sizeof(*d_output_count)));
    CUDA_CHECK(cudaMalloc(&d_cache_scrub_sink, sizeof(*d_cache_scrub_sink)));
    CUDA_CHECK(cudaMalloc(&d_cache_scrub,
                          cache_scrub_values * sizeof(*d_cache_scrub)));
    CUDA_CHECK(cudaMemset(d_cache_scrub, 0,
                          cache_scrub_values * sizeof(*d_cache_scrub)));
    CUDA_CHECK(cudaMemset(d_cache_scrub_sink, 0, sizeof(*d_cache_scrub_sink)));
    CUDA_CHECK(cudaMemcpy(d_a, input_a.data(), raw_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, input_b.data(), raw_bytes, cudaMemcpyHostToDevice));

    // Input preparation is deliberately outside the timed DRE region.
    CUDA_CHECK(zipccl::CompressAsync(d_a, elements, input_base, d_compressed_a,
                                     capacity, d_input_count));
    CUDA_CHECK(cudaDeviceSynchronize());
    const int warmup_launches = WarmupDre(
        d_compressed_a, input_base, d_b, elements, output_base,
        d_compressed_output, capacity, d_output_count, workspace, workspace_bytes, max_ctas, warmup_ms);

    // Warm-up keeps clocks high but its input must not populate the timed
    // working set. Generate and compress an independent data set afterwards.
    generator.seed(67890);
    for (std::size_t i = 0; i < elements; ++i) {
        input_a[i] = FloatToBf16(distribution(generator));
        input_b[i] = FloatToBf16(distribution(generator));
        expected[i] = FloatToBf16(Bf16ToFloat(input_a[i]) +
                                  Bf16ToFloat(input_b[i]));
    }
    CUDA_CHECK(cudaMemcpy(d_a, input_a.data(), raw_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, input_b.data(), raw_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(zipccl::CompressAsync(d_a, elements, input_base, d_compressed_a,
                                     capacity, d_input_count));
    CUDA_CHECK(cudaDeviceSynchronize());

    const float dre_ms = MeasureDre(
        d_compressed_a, input_base, d_b, elements, output_base,
        d_compressed_output, capacity, d_output_count, workspace, workspace_bytes, d_cache_scrub,
        cache_scrub_values, d_cache_scrub_sink, cache_scrub_blocks, max_ctas,
        iterations);

    CUDA_CHECK(zipccl::DecompressAsync(d_compressed_output, elements, output_base,
                                       d_actual));
    CUDA_CHECK(cudaMemcpy(actual.data(), d_actual, raw_bytes, cudaMemcpyDeviceToHost));
    const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (mismatch.first != expected.end()) {
        const std::size_t index = static_cast<std::size_t>(mismatch.first - expected.begin());
        std::fprintf(stderr, "verification failed at %zu: expected=%04x actual=%04x\n",
                     index, expected[index], actual[index]);
        return EXIT_FAILURE;
    }

    std::uint32_t input_outliers = 0, output_outliers = 0;
    CUDA_CHECK(cudaMemcpy(&input_outliers, d_input_count, sizeof(input_outliers),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&output_outliers, d_output_count, sizeof(output_outliers),
                          cudaMemcpyDeviceToHost));

    const double chained_standard_deviation =
        std::sqrt(3.0) * standard_deviation;
    const std::uint8_t chained_base =
        zipccl::ComputeExponentBase(chained_standard_deviation);
    for (std::size_t i = 0; i < elements; ++i) {
        expected[i] = FloatToBf16(Bf16ToFloat(expected[i]) +
                                  Bf16ToFloat(input_b[i]));
    }
    CUDA_CHECK(zipccl::DecodeReduceEncodeAsync(
        d_compressed_output, output_base, d_b, elements, chained_base,
        d_compressed_a, capacity, d_input_count, workspace, workspace_bytes, nullptr, max_ctas));
    CUDA_CHECK(zipccl::DecompressAsync(d_compressed_a, elements, chained_base,
                                       d_actual));
    CUDA_CHECK(cudaMemcpy(actual.data(), d_actual, raw_bytes,
                          cudaMemcpyDeviceToHost));
    const auto chained_mismatch =
        std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (chained_mismatch.first != expected.end()) {
        const std::size_t index = static_cast<std::size_t>(
            chained_mismatch.first - expected.begin());
        std::fprintf(stderr,
                     "chained verification failed at %zu: expected=%04x actual=%04x\n",
                     index, expected[index], actual[index]);
        return EXIT_FAILURE;
    }
    const std::size_t compressed_input_bytes =
        zipccl::CompressedBytes(elements, input_outliers);
    const std::size_t compressed_output_bytes =
        zipccl::CompressedBytes(elements, output_outliers);
    const std::size_t physical_bytes =
        compressed_input_bytes + raw_bytes + compressed_output_bytes;

    std::printf("GPU:                    %s (%d SMs)\n", properties.name,
                properties.multiProcessorCount);
    std::printf("DRE launch cap:         %u CTA(s), workspace %zu bytes\n",
                max_ctas, workspace_bytes);
    std::printf("Clock warm-up:          %.0f ms minimum (%d DRE launches)\n",
                warmup_ms, warmup_launches);
    std::printf("Cache state:            cold per launch (%.1f MiB L2 scrub)\n",
                cache_scrub_values * sizeof(*d_cache_scrub) / 1048576.0);
    std::printf("BF16 elements:          %zu (%zu logical 4096-element blocks)\n",
                elements, logical_blocks);
    std::printf("Raw output size:        %.3f MiB\n", raw_bytes / 1048576.0);
    std::printf("Input distribution:     N(0, %.8g)\n", standard_deviation);
    std::printf("Input exponent window:  [%u, %u]\n", input_base, input_base + 6);
    std::printf("Output estimated std:   %.8g\n", output_standard_deviation);
    std::printf("Output exponent window: [%u, %u]\n", output_base, output_base + 6);
    std::printf("Input/output outliers:  %.3f%% / %.3f%%\n",
                100.0 * input_outliers / elements,
                100.0 * output_outliers / elements);
    std::printf("DRE median time:        %.3f us (%d samples x %d iterations)\n",
                dre_ms * 1000.0, kSamples, iterations);
    std::printf("DRE raw-equivalent:     %.3f GB/s (N * sizeof(BF16) / time)\n",
                Rate(raw_bytes, dre_ms));
    std::printf("DRE actual output:      %.3f GB/s (compressed bytes / time)\n",
                Rate(compressed_output_bytes, dre_ms));
    std::printf("DRE physical payload:   %.3f GB/s (compressed A + raw B + compressed output)\n",
                Rate(physical_bytes, dre_ms));
    std::printf("Verification:           bit-exact BF16 and chained DRE PASS\n");

    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_actual));
    CUDA_CHECK(cudaFree(d_compressed_a));
    CUDA_CHECK(cudaFree(d_compressed_output));
    CUDA_CHECK(cudaFree(d_input_count));
    CUDA_CHECK(cudaFree(d_output_count));
    CUDA_CHECK(cudaFree(d_cache_scrub_sink));
    CUDA_CHECK(cudaFree(d_cache_scrub));
    return EXIT_SUCCESS;
}
