#include "zipccl_stats.h"
#include <cuda/atomic>

__device__ float ncclZipPublishSlice(ncclZipStats& stats, unsigned slot, ncclZipMoments slice) {
  stats.partial[slot] = slice;
  cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(stats.readyMask);
  unsigned bit = 1u << slot;
  unsigned mask = ready.fetch_or(bit, cuda::memory_order_acq_rel) | bit;
  ncclZipMoments merged{};
  for (unsigned i = 0; i < ncclZipStatSlices; ++i)
    if (mask & (1u << i)) merged.merge(stats.partial[i]);
  // Version ordering prevents a slower publisher from replacing a newer aggregate.
  unsigned long long snapshot = (static_cast<unsigned long long>(1 + __popc(mask)) << 32) |
                               __float_as_uint(merged.sigma());
  auto previous = atomicMax(&stats.snapshot, snapshot);
  return __uint_as_float(unsigned(previous > snapshot ? previous : snapshot));
}
