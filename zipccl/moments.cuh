#pragma once

#include <stdint.h>

struct ncclZipMoments {
  unsigned count;
  double mean;
  double m2;

#ifdef __CUDACC__
  __device__ void add(double value) {
    if (!isfinite(value)) return;
    double delta = value - mean;
    mean += delta / ++count;
    m2 += delta * (value - mean);
  }
  __device__ void merge(ncclZipMoments other) {
    if (!other.count) return;
    double delta = other.mean - mean;
    unsigned combined = count + other.count;
    double weight = double(other.count) / combined;
    m2 += other.m2 + delta * delta * (count * weight);
    mean += delta * weight;
    count = combined;
  }
  __device__ float sigma() const { return count ? float(sqrt(fmax(0., m2 / count))) : 0.f; }
#endif
};
