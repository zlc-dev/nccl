// SPDX-License-Identifier: Apache-2.0
#pragma once

// Buffer the first eight slices per channel (one complete four-rank chunk).
// Each field has one writer. The Ring CTA flushes after its last DRC.
struct ZipSliceTrace {
  using Time = unsigned long long;
  static constexpr int capacity = 8;
  Time recvBegin, recvEnd, sendBegin, sendEnd;
  Time codecBegin, codecEnd, postBegin, postEnd;
};

__device__ __forceinline__ ZipSliceTrace* zipTraceBuffer() {
  __shared__ ZipSliceTrace records[ZipSliceTrace::capacity];
  return records;
}

__device__ __forceinline__ ZipSliceTrace::Time zipStageTime() {
  ZipSliceTrace::Time ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory");
  return ns;
}
