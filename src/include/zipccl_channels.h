#ifndef NCCL_ZIPCCL_CHANNELS_H_
#define NCCL_ZIPCCL_CHANNELS_H_

#include <algorithm>
#include <cstddef>
#include <cstring>

// ncclBuildRings lists every directed cycle starting at the calling rank.
// Equality of these complete lists is independent of the calling rank.
inline int ncclZipDistinctRings(const int* rings, int channels, int ranks, int* representatives) {
  int distinct = 0;
  for (int c = 0; c < channels; ++c) {
    int r = 0;
    while (r < distinct && std::memcmp(rings + size_t(c) * ranks,
            rings + size_t(representatives[r]) * ranks, size_t(ranks) * sizeof(int)) != 0) ++r;
    if (r == distinct) representatives[distinct++] = c;
  }
  return distinct;
}

constexpr int ncclZipMaxChannelsPerRing = 8;

// Assign channel IDs in rounds over the distinct cycles. Every prefix gives
// each cycle the same number of channels to within one.
inline int ncclZipChannelCount(size_t elements, size_t elementsPerSlice, int rings, int limit) {
  size_t elementsPerRound = elementsPerSlice * rings;
  size_t rounds = elements / elementsPerRound + (elements % elementsPerRound != 0);
  int perRing = int(std::min<size_t>(ncclZipMaxChannelsPerRing, std::max<size_t>(4, rounds)));
  return int(std::min<size_t>(elements, std::min(limit, perRing * rings)));
}
#endif
