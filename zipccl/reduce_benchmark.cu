#include <cuda_bf16.h>
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

constexpr int kThreads = 512;
constexpr int kElementsPerVector = sizeof(uint4) / sizeof(std::uint16_t);
constexpr int kSamples = 7;

__device__ __forceinline__ std::uint32_t AddBf16Pair(std::uint32_t a,
                                                     std::uint32_t b) {
    const __nv_bfloat162 va(
        __ushort_as_bfloat16(static_cast<std::uint16_t>(a)),
        __ushort_as_bfloat16(static_cast<std::uint16_t>(a >> 16)));
    const __nv_bfloat162 vb(
        __ushort_as_bfloat16(static_cast<std::uint16_t>(b)),
        __ushort_as_bfloat16(static_cast<std::uint16_t>(b >> 16)));
    const __nv_bfloat162 sum = __hadd2(va, vb);
    return static_cast<std::uint32_t>(__bfloat16_as_ushort(sum.x)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(sum.y)) << 16);
}

__global__ __launch_bounds__(kThreads, 1)
void ReduceKernel(const std::uint16_t* __restrict__ input_a,
                  const std::uint16_t* __restrict__ input_b,
                  std::uint16_t* __restrict__ output,
                  std::size_t elements) {
    const auto* vector_a = reinterpret_cast<const uint4*>(input_a);
    const auto* vector_b = reinterpret_cast<const uint4*>(input_b);
    auto* vector_output = reinterpret_cast<uint4*>(output);
    const std::size_t vectors = elements / kElementsPerVector;

    for (std::size_t index = threadIdx.x; index < vectors;
         index += blockDim.x) {
        const uint4 a = vector_a[index];
        const uint4 b = vector_b[index];
        vector_output[index] = make_uint4(
            AddBf16Pair(a.x, b.x), AddBf16Pair(a.y, b.y),
            AddBf16Pair(a.z, b.z), AddBf16Pair(a.w, b.w));
    }

    for (std::size_t index = vectors * kElementsPerVector + threadIdx.x;
         index < elements; index += blockDim.x) {
        output[index] = __bfloat16_as_ushort(__hadd(
            __ushort_as_bfloat16(input_a[index]),
            __ushort_as_bfloat16(input_b[index])));
    }
}

cudaError_t ReduceAsync(const std::uint16_t* input_a,
                        const std::uint16_t* input_b,
                        std::uint16_t* output,
                        std::size_t elements,
                        cudaStream_t stream = nullptr) {
    ReduceKernel<<<1, kThreads, 0, stream>>>(input_a, input_b, output, elements);
    return cudaGetLastError();
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

int Warmup(const std::uint16_t* input_a,
           const std::uint16_t* input_b,
           std::uint16_t* output,
           std::size_t elements,
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
            CUDA_CHECK(ReduceAsync(input_a, input_b, output, elements));
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

float Measure(const std::uint16_t* input_a,
              const std::uint16_t* input_b,
              std::uint16_t* output,
              std::size_t elements,
              int iterations) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            CUDA_CHECK(ReduceAsync(input_a, input_b, output, elements));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        samples.push_back(elapsed_ms / iterations);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t elements = argc > 1
        ? std::strtoull(argv[1], nullptr, 10) : 131072;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 5000;
    const double standard_deviation = argc > 3 ? std::atof(argv[3]) : 0.02;
    const float warmup_ms = argc > 4 ? std::atof(argv[4]) : 1500.0f;
    if (elements == 0 || iterations <= 0 || !(standard_deviation > 0.0) ||
        !std::isfinite(standard_deviation) || warmup_ms < 0.0f) {
        std::fprintf(stderr,
                     "usage: %s [elements] [iterations] [std>0] [warmup_ms>=0]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

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

    const std::size_t output_bytes = elements * sizeof(std::uint16_t);
    const std::size_t physical_bytes = output_bytes * 3;
    std::uint16_t *d_a = nullptr, *d_b = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, input_a.data(), output_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, input_b.data(), output_bytes,
                          cudaMemcpyHostToDevice));

    const int warmup_launches = Warmup(
        d_a, d_b, d_output, elements, warmup_ms);
    const float reduce_ms = Measure(
        d_a, d_b, d_output, elements, iterations);
    CUDA_CHECK(cudaMemcpy(actual.data(), d_output, output_bytes,
                          cudaMemcpyDeviceToHost));

    const auto mismatch = std::mismatch(
        expected.begin(), expected.end(), actual.begin());
    if (mismatch.first != expected.end()) {
        const std::size_t index = static_cast<std::size_t>(
            mismatch.first - expected.begin());
        std::fprintf(stderr,
                     "verification failed at %zu: expected=%04x actual=%04x\n",
                     index, expected[index], actual[index]);
        return EXIT_FAILURE;
    }

    std::printf("GPU:                    %s (%d SMs)\n", properties.name,
                properties.multiProcessorCount);
    std::printf("Reduce launch:          exactly 1 CTA, 512 threads\n");
    std::printf("Operation:              raw BF16 A + raw BF16 B -> raw BF16 C\n");
    std::printf("Clock warm-up:          %.0f ms minimum (%d Reduce launches)\n",
                warmup_ms, warmup_launches);
    std::printf("BF16 elements:          %zu\n", elements);
    std::printf("Raw output size:        %.3f MiB\n", output_bytes / 1048576.0);
    std::printf("Input distribution:     N(0, %.8g)\n", standard_deviation);
    std::printf("Reduce median time:     %.3f us (%d samples x %d iterations)\n",
                reduce_ms * 1000.0f, kSamples, iterations);
    std::printf("Reduce output rate:     %.3f GB/s (N * sizeof(BF16) / time)\n",
                Rate(output_bytes, reduce_ms));
    std::printf("Reduce physical rate:   %.3f GB/s (read A + read B + write C)\n",
                Rate(physical_bytes, reduce_ms));
    std::printf("Verification:           bit-exact BF16 PASS\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
    return EXIT_SUCCESS;
}
