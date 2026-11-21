#pragma once

#include "single_cta.cuh"

namespace zipccl::device {
// Each scan lane owns one block; each writer warp owns that lane's four segments.
__device__ inline unsigned CompactRingBlocks(const uint16_t* counts, const uint8_t* slots,
    EncodedOutput output, unsigned first, unsigned blocks, unsigned base, uint8_t* scratch) {
    unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    ushort4 c = lane < blocks ? reinterpret_cast<const ushort4*>(counts)[lane] : make_ushort4(0,0,0,0);
    unsigned sum = c.x + c.y + c.z + c.w;
    unsigned prefix = WarpExclusivePrefix(sum, lane);
    unsigned blockBase = base + __shfl_sync(~0u, prefix, warp);
    unsigned segment = lane / 8;
    unsigned a = __shfl_sync(~0u, unsigned(c.x), warp), b = __shfl_sync(~0u, unsigned(c.y), warp);
    unsigned d = __shfl_sync(~0u, unsigned(c.z), warp), e = __shfl_sync(~0u, unsigned(c.w), warp);
    unsigned offset = blockBase + (segment >= 1 ? a : 0) + (segment >= 2 ? b : 0) + (segment >= 3 ? d : 0);
    unsigned count = segment == 0 ? a : segment == 1 ? b : segment == 2 ? d : e;
    unsigned total = __shfl_sync(~0u, prefix + sum, 31);
    if (warp < blocks)
        CopyDreBlock<true>(slots + warp * kElementsPerBlock, output, first + warp, offset, count, scratch);
    return base + total;
}
}
