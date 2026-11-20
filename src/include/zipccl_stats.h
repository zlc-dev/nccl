#ifndef NCCL_ZIPCCL_STATS_H_
#define NCCL_ZIPCCL_STATS_H_

#include <cuda_runtime.h>
#include <stdint.h>

#include "../../zipccl/moments.cuh"

// One immutable partial per elected Encode; snapshot packs publication version and sigma bits.
constexpr unsigned ncclZipStatSlices = 4;
struct ncclZipStats {
  unsigned claimed, readyMask;
  unsigned long long snapshot;
  ncclZipMoments partial[ncclZipStatSlices];
};

#ifdef __CUDACC__
__device__ float ncclZipPublishSlice(ncclZipStats& stats, unsigned slot, ncclZipMoments slice);

__device__ inline int ncclZipWindow(float sigma) {
  if (!(sigma > 0) || !isfinite(sigma)) return 0;
  float logSigma = log2f(sigma);
  int lo = max(-127, min(121, __float2int_rd(logSigma - 5.3606617f)));
  float x = exp2f(lo - logSigma - 0.5f);
  return 127 + lo + (erff(256*x) - erff(2*x) > erff(128*x) - erff(x));
}

#endif

#endif
