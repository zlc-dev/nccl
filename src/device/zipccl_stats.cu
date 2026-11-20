#include "zipccl_stats.h"
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <algorithm>

namespace {
constexpr int threads = 256;

__device__ ncclZipMoments merge(ncclZipMoments a, ncclZipMoments b) {
  if (!b.count) return a;
  if (!a.count) return b;
  const uint64_t count = a.count + b.count;
  const double delta = b.mean - a.mean;
  const double weight = double(b.count) / double(count);
  return {count, a.mean + delta * weight, a.m2 + b.m2 + delta * delta * (a.count * weight)};
}

__device__ ncclZipMoments warpReduce(ncclZipMoments a) {
  for (int offset = 16; offset; offset /= 2) {
    ncclZipMoments b;
    b.count = __shfl_down_sync(0xffffffff, (unsigned long long)a.count, offset);
    b.mean = __shfl_down_sync(0xffffffff, a.mean, offset);
    b.m2 = __shfl_down_sync(0xffffffff, a.m2, offset);
    if (int(threadIdx.x % 32) < offset) a = merge(a, b);
  }
  return a;
}

__device__ ncclZipMoments blockReduce(ncclZipMoments state) {
  __shared__ ncclZipMoments warps[threads / 32];
  state = warpReduce(state);
  if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = state;
  __syncthreads();
  if (threadIdx.x < 32) {
    state = threadIdx.x < threads / 32 ? warps[threadIdx.x] : ncclZipMoments{};
    state = warpReduce(state);
  }
  return state;
}

__global__ void rankMoments(const __nv_bfloat16* input, size_t count, ncclZipStats* result) {
  ncclZipMoments a{};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    const double x = __bfloat162float(input[i]);
    const double delta = x - a.mean;
    ++a.count;
    a.mean += delta / double(a.count);
    a.m2 += delta * (x - a.mean);
  }
  a = blockReduce(a);
  __shared__ bool last;
  if (threadIdx.x == 0) {
    result->partial[blockIdx.x] = a;
    // The chain of device-scope acquire/release increments publishes every
    // partial to the final CTA. Other CTAs exit without spinning.
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> arrived(result->completed);
    last = arrived.fetch_add(1, cuda::memory_order_acq_rel) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  a = {};
  for (unsigned i = threadIdx.x; i < gridDim.x; i += blockDim.x) a = merge(a, result->partial[i]);
  a = blockReduce(a);
  if (threadIdx.x == 0) {
    result->sigma = float(sqrt(fmax(0.0, a.m2 / double(a.count))));
    result->base = ncclZipWindow(result->sigma);
    result->completed = 0;
  }
}
} // namespace

cudaError_t ncclZipStatsGridSize(int device, int* blocks) {
  int sms, blocksPerSm;
  cudaError_t status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, rankMoments, threads, 0);
  if (status == cudaSuccess) *blocks = sms * blocksPerSm;
  return status;
}

cudaError_t ncclZipComputeStats(const void* input, size_t count, int maxBlocks, ncclZipStats* result, cudaStream_t stream) {
  if (count == 0) return cudaErrorInvalidValue;
  int blocks = int(std::min<size_t>(maxBlocks, 1 + (count - 1) / threads));
  rankMoments<<<blocks, threads, 0, stream>>>(static_cast<const __nv_bfloat16*>(input), count, result);
  return cudaGetLastError();
}
