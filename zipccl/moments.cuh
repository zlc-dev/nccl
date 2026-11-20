#pragma once

#include <stdint.h>

// Mean and m2 use units of 2^exponent and 2^(2*exponent), respectively.
struct ncclZipMoments {
  unsigned count;
  float mean;
  float m2;
  int exponent;

#ifdef __CUDACC__
  __device__ void merge(ncclZipMoments other) {
    if (!other.count) return;
    if (!count) { *this = other; return; }
    if (exponent != other.exponent) {
      int common = max(exponent, other.exponent);
      mean = scalbnf(mean, exponent - common);
      m2 = scalbnf(m2, 2 * (exponent - common));
      other.mean = scalbnf(other.mean, other.exponent - common);
      other.m2 = scalbnf(other.m2, 2 * (other.exponent - common));
      exponent = common;
    }
    unsigned combined = count + other.count;
    float weight = float(other.count) / float(combined);
    float delta = other.mean - mean;
    m2 += fmaf(delta * delta, count * weight, other.m2);
    mean = fmaf(delta, weight, mean);
    count = combined;
  }
  __device__ float sigma() const {
    return count ? scalbnf(sqrtf(fmaxf(0.f, m2 / float(count))), exponent) : 0.f;
  }
#endif
};

// Center on the first finite value to retain precision for nearly constant inputs.
// Three power-of-two ranges keep both BF16 extremes and full-slice sums within FP32.
struct ncclZipSums {
  unsigned count;
  float origin, sum, squares;
  int exponent;
#ifdef __CUDACC__
  __device__ void add(float x) {
    if (!isfinite(x)) return;
    float magnitude = fabsf(x);
    int range = magnitude < 0x1p-40f ? -80 : magnitude < 0x1p40f ? 0 : 96;
    if (!count) {
      exponent = range;
      origin = scalbnf(x, -range);
      count = 1;
      return;
    }
    if (range > exponent) {
      int shift = exponent - range;
      origin = scalbnf(origin, shift);
      sum = scalbnf(sum, shift);
      squares = scalbnf(squares, 2 * shift);
      exponent = range;
    }
    // The common exponent range uses the input unchanged.
    float value = exponent == 0 ? x : x * (exponent < 0 ? 0x1p80f : 0x1p-96f);
    float delta = value - origin;
    ++count;
    sum += delta;
    squares = fmaf(delta, delta, squares);
  }
  __device__ ncclZipMoments moments() const {
    if (!count) return {};
    float offset = sum / float(count);
    return {count, origin + offset, fmaxf(0.f, fmaf(-offset, sum, squares)), exponent};
  }
#endif
};
