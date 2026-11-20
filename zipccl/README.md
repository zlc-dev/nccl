# Standalone ZipCCL core

This directory contains the single-buffer, single-work compression core from
ZipCCL without PyTorch or collective-communication code. It losslessly encodes
raw BF16 bit patterns as:

1. one sign-and-mantissa byte per value;
2. three exponent bitmaps (a 3-bit code for a seven-exponent window);
3. one offset per 4096-value CUDA block;
4. five absolute outlier offsets per block, delimiting four 1024-value warp
   segments;
5. one byte for every exponent outside the selected window.

The caller owns all CUDA allocations. `zipccl.cuh` exposes asynchronous
compression and decompression functions which run on a caller-provided CUDA
stream. `GetLayout` documents the buffer layout and `MaxCompressedBytes`
returns the required worst-case allocation.

## Fused decode-reduce-encode

`DecodeReduceEncodeAsync` accepts one compressed BF16 array and one raw BF16
array. A single CUDA kernel decodes the first array, performs native BF16 sum
with the raw array (`__hadd2`), and directly encodes the result. No intermediate
uncompressed result is written to global memory.

For Gaussian data, `ComputeExponentBase(std)` implements the paper formula:

```text
base = round(log2(std) + 127 + 0.5 * log2(14 * ln(2) / 16383))
```

For independent inputs with standard deviations `std_a` and `std_b`, pass an
output base computed from `sqrt(std_a^2 + std_b^2)`. For identically
distributed inputs this is `sqrt(2) * std`.

`dre_benchmark` warms GPU clocks with a separate random input, regenerates and
compresses the timed input afterwards, and scans a buffer twice the L2 size
before every measured launch. The cache-scrub kernel is outside the timing
events, so the reported rate is cold-cache DRE throughput rather than repeated
L2-resident throughput.

The fused kernel stages the fixed compressed stream, raw BF16 input, bitmaps,
and each warp's contiguous outlier segment with coalesced 16-byte asynchronous
copies in one CUDA pipeline stage. Warp-level outlier offsets remove the input
group prefix barrier and avoid irregular exponent loads during decode. Paired
BF16 arithmetic and encoding use conflict-free shared-memory layouts.

`DecodeReduceCopyAsync` performs the same decode and native BF16 reduction but
writes a raw BF16 result. It avoids output bitmap and outlier construction.
The complete 128 KiB to 64 MiB four-operator benchmark is recorded in
[`FOUR_OPS_RESULTS.md`](FOUR_OPS_RESULTS.md).

Build and run:

```sh
make
./zipccl_benchmark                  # 16M BF16 values, 100 iterations, std=0.02
./dre_benchmark 131072 1000 0.02 2000 4
# elements, iterations, std, warmup_ms, max_ctas
./four_ops_benchmark 0.02           # four operators, 128 KiB through 64 MiB
./zipccl_benchmark 33554432 200 0.02 # elements, iterations, std
./zipccl_benchmark 16777216 100 0.02 4 # cap the grid at four blocks
make bench-limited                    # scan 1,2,4,8,16,32,unlimited
```

The default target is `sm_80` and also runs on newer GPUs. For Hopper-specific
code generation use `make CUDA_ARCH=sm_90a`. The benchmark generates normally
distributed BF16 data, estimates the exponent base from its standard deviation,
checks bit-exact round-trip correctness, and reports logical input throughput
and compression ratio.

With pure `N(0, 0.02)` BF16 input and the estimated `[116, 122]` window on an
RTX 4070, standalone codec throughput was:

| BF16 input | Outliers | Compressed/input | Compress | Decompress |
|-----------:|---------:|-----------------:|---------:|-----------:|
| 1 MiB      | 2.108%   | 0.699x           | 69.21 GB/s | 92.27 GB/s |
| 32 MiB     | 2.126%   | 0.699x           | 238-245 GB/s | 229-239 GB/s |

The 32 MiB result uses 500 timed iterations; the 1 MiB result uses 5000.
The corresponding 32 MiB payload rates (raw + compressed bytes) are
389-417 GB/s. Compression and decompression stage their main streams through
shared memory with coalesced 16-byte transfers; decompression overlaps its
sign/mantissa load with bitmap prefix computation. The compression kernel uses
32 registers, zero stack bytes, and zero spills after eliminating its local
exponent array.

The optional fourth argument limits the number of physical CUDA blocks. A
limited grid uses persistent blocks and grid-stride iteration over all logical
4096-element chunks, so the input size and compressed format remain unchanged.
This provides a simple way to measure throughput under reduced compute
resources. `0` (the default) launches one block per logical chunk.

On an RTX 4070 with 16,777,216 BF16 values (32 MiB), pure `N(0, 0.02)` input,
and the estimated `[116, 122]` window, the optimized resource-limited results
were:

| Maximum blocks | Compress GB/s | Decompress GB/s |
|---------------:|--------------:|----------------:|
| 1              | 3.28          | 3.01            |
| 2              | 6.60          | 5.94            |
| 4              | 13.17         | 11.66           |
| 8              | 25.92         | 22.85           |
| 16             | 51.88         | 44.61           |
| 32             | 99.15         | 80.02           |
| 46             | 140.32        | 110.70          |
| unlimited      | 253.15        | 245.70          |

Every configuration passed bit-exact round-trip verification. Throughput is
logical uncompressed BF16 bytes divided by kernel time; host/device transfers
and input generation are excluded.

On an RTX 4070 with 16,777,216 elements per input and `std=0.02`, the fused
operator took 0.271 ms. This corresponds to 123.75 GB/s using one BF16 tensor
as the byte count, 247.50 GB/s using both logical inputs, or 297.84 GB/s using
the estimated physical payload (compressed A + raw B + compressed output).
With the optimized standalone codec, separate decode, BF16 add, and encode
kernels took 0.512 ms, so fusion provided a 1.89x speedup.
The output matched native BF16 reduction bit-for-bit. These rate definitions
are printed separately by `dre_benchmark`; the two-input rate is not DRAM
bandwidth and should not be compared directly with the standalone codec's
one-tensor rate.

The compressed byte count is `CompressedBytes(elements, outlier_count)`. The
device-side counter is available only after the compression stream reaches the
corresponding copy/synchronization point.
