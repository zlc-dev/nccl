#ifndef NCCL_ZIPCCL_STATS_H_
#define NCCL_ZIPCCL_STATS_H_

#include <cuda_runtime.h>
#include <stdint.h>

struct ncclZipMoments {
  uint64_t count;
  double mean;
  double m2;
};

struct ncclZipStats {
  ncclZipMoments* partial;
  unsigned completed;
  float sigma;
  int base;
};

#ifdef __CUDACC__
__device__ inline int ncclZipWindow(float sigma) {
  if (!(sigma > 0) || !isfinite(sigma)) return 0;
  float logSigma = log2f(sigma);
  int lo = max(-127, min(121, __float2int_rd(logSigma - 5.3606617f)));
  float x = exp2f(lo - logSigma - 0.5f);
  return 127 + lo + (erff(256*x) - erff(2*x) > erff(128*x) - erff(x));
}

#endif

cudaError_t ncclZipStatsGridSize(int device, int* blocks);
cudaError_t ncclZipComputeStats(const void* input, size_t count, int maxBlocks, ncclZipStats* result, cudaStream_t stream);
#endif
