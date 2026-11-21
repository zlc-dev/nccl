#pragma once

#include "../../zipccl/ring_compact.cuh"

namespace ncclZipRing {
using namespace zipccl::device;

struct PayloadArgs {
  const uint16_t* local;
  const void* input;
  void* output;
  int elements;
  int inputBase;
  int outputBase;
};

using PayloadFormat = RingFrameFormat;

template<SegmentOp Op, PayloadFormat In, PayloadFormat Out>
struct PayloadTransform {};

template<PayloadFormat In>
__device__ EncodedInput payloadInput(const PayloadArgs& args, const zipccl::Layout& layout) {
  const auto* data = static_cast<const uint8_t*>(args.input);
  return {data, reinterpret_cast<const uint32_t*>(data + layout.bitmap[0]),
          reinterpret_cast<const uint32_t*>(data + layout.bitmap[1]),
          reinterpret_cast<const uint32_t*>(data + layout.bitmap[2]),
          reinterpret_cast<const uint32_t*>(data + layout.block_offsets),
          reinterpret_cast<const uint32_t*>(data + layout.warp_offsets),
          data + (In == PayloadFormat::Fixed ? layout.block_offsets : layout.outliers)};
}

template<>
__device__ inline EncodedInput payloadInput<PayloadFormat::Raw>(const PayloadArgs& args, const zipccl::Layout&) {
  EncodedInput input{};
  input.raw = static_cast<const uint16_t*>(args.input);
  return input;
}

template<SegmentOp Op>
struct PayloadTransform<Op, PayloadFormat::Raw, PayloadFormat::Raw> {
  template<class P>
  static __device__ int run(P&, int tid, int workers, const PayloadArgs& args, ncclZipSums* sums) {
    auto* dst = static_cast<uint16_t*>(args.output);
    for (int i = tid; i < args.elements; i += workers) {
      uint16_t value = args.local[i];
      if constexpr (Op == SegmentOp::Copy)
        value = __bfloat16_as_ushort(__hadd(__ushort_as_bfloat16(static_cast<const uint16_t*>(args.input)[i]),
                                           __ushort_as_bfloat16(value)));
      if constexpr (Op == SegmentOp::StatEncode) sums->add(__ushort_as_bfloat16(value));
      dst[i] = value;
    }
    return args.elements * sizeof(uint16_t);
  }
};

template<SegmentOp Op, PayloadFormat In>
struct PayloadTransform<Op, In, PayloadFormat::Raw> {
  template<class P>
  static __device__ int run(P& prims, int tid, int workers, const PayloadArgs& args, ncclZipSums*) {
    static_assert(Op == SegmentOp::Copy);
    const auto input = payloadInput<In>(args, FrameLayout(args.elements, kRingFrameAlignment));
    auto* shared = static_cast<uint8_t*>(prims.codecScratch());
    auto* dst = static_cast<uint16_t*>(args.output);
    int warps = min(workers / 32, ncclZipWorkerWarps);
    unsigned segment = tid / 32, full = unsigned(args.elements) / kDreValuesPerWarp;
    if (segment < unsigned(warps)) {
      for (; segment < full; segment += warps)
        ProcessSegment<true, Op, In == PayloadFormat::Fixed>(input, args.inputBase, args.local, args.elements,
            0, {}, segment, nullptr, nullptr, shared, dst);
      if (segment == full && args.elements % kDreValuesPerWarp)
        ProcessSegment<false, Op, In == PayloadFormat::Fixed>(input, args.inputBase, args.local, args.elements,
            0, {}, segment, nullptr, nullptr, shared, dst);
    }
    return args.elements * sizeof(uint16_t);
  }
};

template<SegmentOp Op, PayloadFormat In, bool Fixed, class P>
__device__ EncodedOutput encodeSegments(P& prims, int tid, int workers, const PayloadArgs& args,
    const zipccl::Layout& layout, uint8_t* slots, uint16_t* counts, ncclZipSums* sums) {
  const auto input = payloadInput<In>(args, layout);
  auto* data = static_cast<uint8_t*>(args.output);
  EncodedOutput output{data, reinterpret_cast<uint32_t*>(data + layout.bitmap[0]),
      reinterpret_cast<uint32_t*>(data + layout.bitmap[1]), reinterpret_cast<uint32_t*>(data + layout.bitmap[2]),
      reinterpret_cast<uint32_t*>(data + layout.block_offsets), reinterpret_cast<uint32_t*>(data + layout.warp_offsets),
      data + (Fixed ? layout.block_offsets : layout.outliers)};
  int warps = min(workers / 32, ncclZipWorkerWarps);
  auto* shared = static_cast<uint8_t*>(prims.codecScratch());
  prims.syncCodecWorkers();
  if (tid / 32 < warps) {
    for (unsigned segment = tid / 32; segment < layout.blocks * kSegmentsPerBlock; segment += warps) {
      auto* slot = Fixed ? nullptr : slots + segment * kDreValuesPerWarp;
      auto* count = Fixed ? nullptr : counts + segment;
      if (segment < unsigned(args.elements) / kDreValuesPerWarp)
        ProcessSegment<true, Op, In == PayloadFormat::Fixed, Fixed>(input, args.inputBase, args.local,
            args.elements, args.outputBase, output, segment, count, slot, shared, nullptr, sums);
      else
        ProcessSegment<false, Op, In == PayloadFormat::Fixed, Fixed>(input, args.inputBase, args.local,
            args.elements, args.outputBase, output, segment, count, slot, shared, nullptr, sums);
    }
  }
  return output;
}

template<SegmentOp Op, PayloadFormat In>
struct PayloadTransform<Op, In, PayloadFormat::Fixed> {
  template<class P>
  static __device__ int run(P& prims, int tid, int workers, const PayloadArgs& args, ncclZipSums* sums) {
    const auto layout = FrameLayout(args.elements, kRingFrameAlignment);
    encodeSegments<Op, In, true>(prims, tid, workers, args, layout, nullptr, nullptr, sums);
    return layout.block_offsets + args.elements;
  }
};

template<SegmentOp Op, PayloadFormat In>
struct PayloadTransform<Op, In, PayloadFormat::Compressed> {
  template<class P>
  static __device__ int run(P& prims, int tid, int workers, const PayloadArgs& args, ncclZipSums* sums) {
    const auto layout = FrameLayout(args.elements, kRingFrameAlignment);
    constexpr int slotBytes = ncclZipRawSliceBytes / sizeof(uint16_t);
    auto* comm = reinterpret_cast<const ncclKernelCommAndChannels*>(ncclShmem.args.comm);
    auto* slots = comm->zipOutlierSlots + size_t(ncclShmem.channelId) * RingWorkspaceBytes(slotBytes);
    auto* counts = reinterpret_cast<uint16_t*>(slots + slotBytes);
    auto output = encodeSegments<Op, In, false>(prims, tid, workers, args, layout, slots, counts, sums);
    prims.syncCodecWorkers();
    int compactWarps = min(min(workers / 32, ncclZipWorkerWarps), kWarpSize);
    unsigned total = 0;
    if (tid / 32 < compactWarps) {
      auto* scratch = static_cast<uint8_t*>(prims.codecScratch()) + (tid / 32) * kDreWarpSharedBytes;
      for (unsigned first = 0; first < layout.blocks; first += compactWarps) {
        unsigned group = min(unsigned(compactWarps), unsigned(layout.blocks - first));
        total = CompactRingBlocks(counts + first * kSegmentsPerBlock, slots + first * zipccl::kElementsPerBlock,
                                  output, first, group, total, scratch);
      }
    }
    return layout.outliers + total;
  }
};

template<bool Reduce, bool Statistics, RingFrameFormat In, RingFrameFormat Out>
struct PayloadOperation {};

template<bool Statistics, RingFrameFormat Out>
struct PayloadOperation<false, Statistics, RingFrameFormat::Raw, Out> {
  static constexpr SegmentOp op = Statistics ? SegmentOp::StatEncode : SegmentOp::Encode;
};

template<RingFrameFormat In, RingFrameFormat Out>
struct PayloadOperation<true, false, In, Out> {
  static constexpr SegmentOp op = Out == RingFrameFormat::Raw ? SegmentOp::Copy :
      In == RingFrameFormat::Raw ? SegmentOp::RawReduceEncode : SegmentOp::Dre;
};

template<bool Reduce, bool Statistics, RingFrameFormat In, RingFrameFormat Out, class P>
__device__ int processPayload(P& prims, int tid, int workers, const PayloadArgs& args, ncclZipSums* sums = nullptr) {
  constexpr auto op = PayloadOperation<Reduce, Statistics, In, Out>::op;
  return PayloadTransform<op, In, Out>::run(prims, tid, workers, args, sums);
}

} // namespace ncclZipRing
