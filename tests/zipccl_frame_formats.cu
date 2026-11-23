#include "common.h"
#include "reduce_scatter_ring_codec.h"
#include <cstdio>
#include <cstdlib>

using namespace zipccl::device;
using Format = RingFrameFormat;
__shared__ ncclShmemData ncclShmem;

struct Workers {
  __device__ void syncCodecWorkers() { __syncthreads(); }
  __device__ void* codecScratch() { return ncclScratchForWarp(0); }
};

template<SegmentOp Op>
__global__ void processFrame(ncclKernelCommAndChannels* comm, uint16_t* local, int n,
                             RingFrameHeader* src, RingFrameHeader* dst, uint16_t* dense, Format target,
                             const float* sigma = nullptr, uint8_t* staged = nullptr, unsigned capacity = 0) {
  if (!threadIdx.x) {
    ncclShmem.args.comm = &comm->comm;
    ncclShmem.channelId = 0;
  }
  __syncthreads();
  Workers workers;
  if constexpr (Op == SegmentOp::Encode || Op == SegmentOp::StatEncode) {
    // Construct even a small compressed input to exercise compressed-to-raw transitions.
    const int base = ncclZipWindow(.02f);
    ncclZipRing::PayloadArgs args{local, nullptr, dst + 1, n, 0, base};
    ncclZipSums sums{};
    int bytes = ncclZipRing::dispatchOutput<false, Op == SegmentOp::StatEncode, Format::Raw>(
        target, workers, threadIdx.x, blockDim.x, args, &sums);
    if constexpr (Op == SegmentOp::StatEncode) {
      unsigned count = __syncthreads_count(sums.count != 0);
      if (count == 0) asm("trap;");
      atomicAdd(&comm->zipStats.claimed, sums.count);
    }
    if (!threadIdx.x) {
      if (target == Format::Raw && bytes != n * sizeof(uint16_t)) asm("trap;");
      *dst = {unsigned(n), 1, uint16_t(base), target, .02f, unsigned(sizeof(RingFrameHeader) + bytes)};
    }
  } else {
    const auto* incoming = staged ? ncclZipRing::stageFrame(workers, threadIdx.x, blockDim.x, src, staged, capacity)
                                  : src;
    if (staged && incoming == src) asm("trap;");
    ncclZipRing::run<Op>(workers, threadIdx.x, blockDim.x, local, n, sigma,
                         incoming, dst, dense, target == Format::Fixed);
  }
}

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(error));
    exit(1);
  }
}

template<class Operation, class = void>
constexpr bool hasOperation = false;
template<class Operation>
constexpr bool hasOperation<Operation, std::void_t<decltype(Operation::op)>> = true;

template<bool Reduce, Format In>
constexpr bool rejectsStats =
    !hasOperation<ncclZipRing::PayloadOperation<Reduce, true, In, Format::Raw>> &&
    !hasOperation<ncclZipRing::PayloadOperation<Reduce, true, In, Format::Compressed>> &&
    !hasOperation<ncclZipRing::PayloadOperation<Reduce, true, In, Format::Fixed>>;
static_assert(rejectsStats<false, Format::Compressed> && rejectsStats<false, Format::Fixed>);
static_assert(rejectsStats<true, Format::Compressed> && rejectsStats<true, Format::Fixed>);

template<class T> T* allocate(size_t count) {
  T* data;
  check(cudaMallocManaged(&data, count * sizeof(T)));
  return data;
}

__global__ void compact32(const uint16_t* counts, const uint8_t* slots, EncodedOutput output, unsigned* total) {
  // The scan is independent of the codec's shared staging. For this test only
  // one byte per segment is populated, so each warp needs four scratch bytes.
  __shared__ uint8_t scratch[32 * 16];
  unsigned size = CompactRingBlocks(counts, slots, output, 0, 32, 0, scratch + (threadIdx.x / 32) * 16);
  if (!threadIdx.x) *total = size;
}

static bool check32Warps() {
  constexpr unsigned segments = 32 * kSegmentsPerBlock;
  auto* counts = allocate<uint16_t>(segments);
  auto* slots = allocate<uint8_t>(segments * kDreValuesPerWarp);
  EncodedOutput output{};
  output.outliers = allocate<uint8_t>(segments);
  output.block_offsets = allocate<uint32_t>(32);
  output.warp_offsets = allocate<uint32_t>(32 * kDreWarpOffsetsPerBlock);
  auto* total = allocate<unsigned>(1);
  for (unsigned s = 0; s < segments; ++s) {
    counts[s] = s % 3 != 0;
    slots[s * kDreValuesPerWarp] = s;
  }
  compact32<<<1, 1024>>>(counts, slots, output, total);
  check(cudaDeviceSynchronize());
  unsigned pos = 0;
  for (unsigned s = 0; s < segments; ++s) {
    unsigned block = s / kSegmentsPerBlock, segment = s % kSegmentsPerBlock;
    if (segment == 0 && output.block_offsets[block] != pos) return false;
    if (output.warp_offsets[block * kDreWarpOffsetsPerBlock + segment] != pos) return false;
    if (counts[s] && output.outliers[pos++] != s) return false;
    if (segment == kSegmentsPerBlock - 1 && output.warp_offsets[(block + 1) * kDreWarpOffsetsPerBlock - 1] != pos)
      return false;
  }
  return pos == *total;
}

int main() {
  if (!check32Warps()) return 1;
  constexpr int elements = ncclZipRawSliceBytes / sizeof(uint16_t);
  constexpr int scratch = RingDreScratchBytes(ncclZipWorkerWarps);
  constexpr int workers = ncclZipWorkerWarps * kWarpSize;
  constexpr size_t capacity = sizeof(RingFrameHeader) + FrameLayout(elements, kRingFrameAlignment).capacity;
  auto* comm = allocate<ncclKernelCommAndChannels>(1);
  comm->zipOutlierSlots = allocate<uint8_t>(RingWorkspaceBytes(elements));
  auto* input = allocate<uint16_t>(3 * elements);
  auto* dense = allocate<uint16_t>(elements);
  RingFrameHeader *hostSrc, *src;
  check(cudaHostAlloc(&hostSrc, capacity, cudaHostAllocMapped));
  check(cudaHostGetDevicePointer(&src, hostSrc, 0));
  uint8_t* staged;
  check(cudaMalloc(&staged, capacity));
  auto* dst = reinterpret_cast<RingFrameHeader*>(allocate<uint8_t>(capacity));
  auto* sigma = allocate<float>(1);
  *sigma = .03f;
  check(cudaFuncSetAttribute(processFrame<SegmentOp::Encode>, cudaFuncAttributeMaxDynamicSharedMemorySize, scratch));
  check(cudaFuncSetAttribute(processFrame<SegmentOp::StatEncode>, cudaFuncAttributeMaxDynamicSharedMemorySize, scratch));
  check(cudaFuncSetAttribute(processFrame<SegmentOp::Dre>, cudaFuncAttributeMaxDynamicSharedMemorySize, scratch));
  check(cudaFuncSetAttribute(processFrame<SegmentOp::Copy>, cudaFuncAttributeMaxDynamicSharedMemorySize, scratch));
  for (int pattern = 0; pattern < 3; ++pattern) {
    for (int i = 0; i < 3 * elements; ++i) {
      float x = pattern == 0 ? (int(i * 31u % 199) - 99) * .0007f : pattern == 1 ? .02f : 1000.f;
      input[i] = __bfloat16_as_ushort(__float2bfloat16(x));
    }
    for (Format incoming : {Format::Raw, Format::Compressed, Format::Fixed}) {
      for (Format outgoing : {Format::Raw, Format::Compressed, Format::Fixed}) {
        for (int n : {outgoing == Format::Raw ? 4095 : 262145, outgoing == Format::Raw ? 1 : elements}) {
          if (pattern == 1) {
            check(cudaMemset(&comm->zipStats.claimed, 0, sizeof(unsigned)));
            processFrame<SegmentOp::StatEncode><<<1, workers, scratch>>>(comm, input, n, nullptr, src, nullptr, incoming);
            check(cudaDeviceSynchronize());
            if (comm->zipStats.claimed != unsigned(n)) return 6;
          } else {
            processFrame<SegmentOp::Encode><<<1, workers, scratch>>>(comm, input, n, nullptr, src, nullptr, incoming);
          }
          check(cudaMemset(dst, 0xa5, capacity));
          processFrame<SegmentOp::Dre><<<1, workers, scratch>>>(
              comm, input + elements, n, src, dst, nullptr, outgoing, sigma, staged, capacity);
          processFrame<SegmentOp::Copy><<<1, workers, scratch>>>(comm, input + 2*elements, n, dst, nullptr, dense, Format::Raw);
          check(cudaDeviceSynchronize());
          if (dst->format != outgoing || dst->contributors != 2) return 2;
          if (fabsf(dst->sigma - hypotf(.02f, *sigma)) > 1e-7f) return 7;
          int used = 0;
          for (int i = 0; i < n; ++i) {
            auto sum = __float2bfloat16(float(__ushort_as_bfloat16(input[i])) + float(__ushort_as_bfloat16(input[elements+i])));
            if (outgoing == Format::Fixed) {
              unsigned exp = (__bfloat16_as_ushort(sum) >> 7) & 255u;
              used += exp - dst->base >= 7;
              if ((i + 1) % kDreValuesPerWarp == 0 || i + 1 == n) {
                int begin = i / kDreValuesPerWarp * kDreValuesPerWarp;
                auto* slot = reinterpret_cast<uint8_t*>(dst+1) + FrameLayout(n,kRingFrameAlignment).block_offsets + begin;
                if (reinterpret_cast<uintptr_t>(slot) % kRingFrameAlignment) return 3;
                for (int j = used; j <= i - begin; ++j) if (slot[j] != 0xa5) return 4;
                used = 0;
              }
            }
            sum = __float2bfloat16(float(sum) + float(__ushort_as_bfloat16(input[2*elements+i])));
            if (dense[i] != __bfloat16_as_ushort(sum)) return 5;
          }
        }
      }
    }
  }
  check(cudaFree(sigma));
  check(cudaFree(staged));
  check(cudaFreeHost(hostSrc));
  puts("PASS: 32-warp compaction, all nine frame transitions with host staging, BF16 results, sparse slots and device sigma");
}
