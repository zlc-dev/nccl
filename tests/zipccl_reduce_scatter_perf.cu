#include <nccl.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using BF = __nv_bfloat16;

void check(cudaError_t e, const char* call) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(e));
}
void check(ncclResult_t e, const char* call) {
  if (e != ncclSuccess) throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(e));
}
#define CHECK(call) check((call), #call)

struct Options {
  size_t begin = 1 << 20, end = 64 << 20;
  int gpus = 4, factor = 2, warmup = 5, iterations = 20, validate = 1;
  unsigned long long seed = 1234;
  float sigma = 0.02f;
  bool native = false;
  bool supplySigma = false;
};

void usage() {
  puts("zipccl_reduce_scatter_perf [-g GPUs] [-b minBytes] [-e maxBytes] [-f factor]\n"
       "  [-w warmups] [-n iterations] [-c 0|1] [--sigma stddev] [--seed integer] [--native]\n"
       "Without --sigma, generate inputs at sigma=0.02 and let ZipCCL estimate it.\n"
       "Sizes are total uncompressed send bytes per GPU. K/M/G use powers of 1024.\n"
       "Each GPU receives size / GPUs bytes. BF16 Sum, out-of-place, one process.");
}

unsigned long long number(const std::string& s, bool sizes = false) {
  if (s.empty() || s[0] < '0' || s[0] > '9') throw std::invalid_argument("invalid number: " + s);
  char* end;
  errno = 0;
  auto value = std::strtoull(s.c_str(), &end, 10);
  unsigned shift = 0;
  if (sizes && *end && !end[1]) {
    switch (*end) {
      case 'k': case 'K': shift = 10; break;
      case 'm': case 'M': shift = 20; break;
      case 'g': case 'G': shift = 30; break;
    }
  }
  if ((*end && !shift) || errno || value > (ULLONG_MAX >> shift))
    throw std::invalid_argument("invalid number: " + s);
  return value << shift;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "-h" || key == "--help") { usage(); std::exit(0); }
    if (key == "--native") { o.native = true; continue; }
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    std::string val = argv[i];
    if (key == "-b") o.begin = number(val, true);
    else if (key == "-e") o.end = number(val, true);
    else if (key == "--seed") o.seed = number(val);
    else if (key == "--sigma") {
      size_t used;
      o.sigma = std::stof(val, &used);
      o.supplySigma = true;
      if (used != val.size() || !std::isfinite(o.sigma) || o.sigma <= 0)
        throw std::invalid_argument("sigma must be finite and positive");
    } else {
      auto v = number(val);
      if (v > INT_MAX) throw std::invalid_argument("integer too large: " + val);
      if (key == "-g") o.gpus = int(v);
      else if (key == "-f") o.factor = int(v);
      else if (key == "-w") o.warmup = int(v);
      else if (key == "-n") o.iterations = int(v);
      else if (key == "-c") o.validate = int(v);
      else throw std::invalid_argument("unknown option: " + key);
    }
  }
  if (o.gpus < 1 || o.begin < sizeof(BF)*o.gpus || o.end < o.begin ||
      o.factor < 2 || !o.iterations || o.validate > 1)
    throw std::invalid_argument("invalid size range, GPU count, factor, iteration count or check mode");
  return o;
}

// SplitMix64 supplies counter-based uniforms. Box-Muller yields four normal
// variates from two independent pairs, reproducibly at any rank and index.
__device__ uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
__device__ float2 normalPair(uint64_t bits) {
  constexpr float scale = 1.0f / 8388608.0f;
  float u = (float(bits >> 41) + 0.5f) * scale;
  float v = (float(unsigned(bits) >> 9) + 0.5f) * scale;
  float sine, cosine;
  sincosf(6.283185307179586f * v, &sine, &cosine);
  float radius = sqrtf(-2.0f * logf(u));
  return make_float2(radius * cosine, radius * sine);
}
__device__ float4 sample(unsigned long long seed, int rank, size_t packet) {
  uint64_t key = mix(seed + rank);
  float2 a = normalPair(mix(key + 2*packet));
  float2 b = normalPair(mix(key + 2*packet + 1));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ float component(float4 v, int lane) {
  return lane < 2 ? (lane ? v.y : v.x) : (lane == 2 ? v.z : v.w);
}
__global__ void generate(BF* data, size_t count, unsigned long long seed, int rank, float sigma) {
  for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x)*4; i < count;
       i += size_t(gridDim.x)*blockDim.x*4) {
    float4 v = sample(seed, rank, i/4);
    for (int j = 0; j < 4 && i+j < count; ++j) data[i+j] = __float2bfloat16_rn(sigma * component(v,j));
  }
}
__device__ float addBF(float a, float b) { return __bfloat162float(__float2bfloat16_rn(a+b)); }

__global__ void verify(const BF* output, size_t count, int dest, int ranks, float sigma,
                       unsigned long long seed, unsigned long long* errors) {
  unsigned long long localErrors = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(blockDim.x)*gridDim.x) {
    size_t index = size_t(dest)*count + i;
    float others[3] = {}, last = 0;
    int n = 0;
    double sum = 0, magnitude = 0;
    for (int r = 0; r < ranks; ++r) {
      float x = __bfloat162float(__float2bfloat16_rn(
          sigma * component(sample(seed, r, index/4), int(index%4))));
      sum += x;
      magnitude += fabs(double(x));
      if (r == dest) last = x;
      else if (ranks <= 4) others[n++] = x;
    }
    float got = __bfloat162float(output[i]);
    // Every directed Ring finishes with the destination rank's own summand.
    bool ok;
    if (ranks == 1) ok = got == last;
    else if (ranks == 2) ok = got == addBF(others[0], last);
    else if (ranks == 3) ok = got == addBF(addBF(others[0], others[1]), last);
    else if (ranks == 4) {
      ok = false;
      for (int k = 0; k < 3; ++k)
        ok |= got == addBF(addBF(addBF(others[k], others[(k+1)%3]), others[(k+2)%3]), last);
    } else {
      double bound = expm1((ranks-1) * log1p(1.0/256)) * magnitude;
      ok = isfinite(got) && fabs(double(got)-sum) <= bound;
    }
    localErrors += !ok;
  }
  if (localErrors) atomicAdd(errors, localErrors);
}

struct Rank {
  ncclComm_t comm;
  cudaStream_t stream;
  BF *send, *recv;
  unsigned long long* errors;
};

void collective(std::vector<Rank>& ranks, size_t count, const Options& o) {
  CHECK(ncclGroupStart());
  for (int r = 0; r < int(ranks.size()); ++r) {
    CHECK(cudaSetDevice(r));
    auto& p = ranks[r];
    if (o.native)
      CHECK(ncclReduceScatter(p.send, p.recv, count, ncclBfloat16, ncclSum, p.comm, p.stream));
    else
      CHECK(ncclZipCCLReduceScatterWithSigma(p.send, p.recv, count, ncclBfloat16, ncclSum, p.comm, p.stream, o.supplySigma ? &o.sigma : nullptr));
  }
  CHECK(ncclGroupEnd());
}

void sync(const std::vector<Rank>& ranks) {
  for (int r = 0; r < int(ranks.size()); ++r) {
    CHECK(cudaSetDevice(r));
    CHECK(cudaStreamSynchronize(ranks[r].stream));
  }
}

int run(const Options& o) {
  int available;
  CHECK(cudaGetDeviceCount(&available));
  if (o.gpus > available) throw std::runtime_error("requested more GPUs than are visible");
  size_t maxCount = o.end / sizeof(BF) / o.gpus;
  std::vector<ncclComm_t> comms(o.gpus);
  CHECK(ncclCommInitAll(comms.data(), o.gpus, nullptr));
  std::vector<Rank> ranks(o.gpus);
  for (int r = 0; r < o.gpus; ++r) {
    CHECK(cudaSetDevice(r));
    auto& p = ranks[r];
    p.comm = comms[r];
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, r));
    printf("# rank %d  device %d  %s\n", r, r, prop.name);
    CHECK(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
    CHECK(cudaMalloc(&p.send, maxCount*o.gpus*sizeof(BF)));
    CHECK(cudaMalloc(&p.recv, maxCount*sizeof(BF)));
    CHECK(cudaMalloc(&p.errors, sizeof(*p.errors)));
    generate<<<256,256,0,p.stream>>>(p.send,maxCount*o.gpus,o.seed,r,o.sigma);
    CHECK(cudaGetLastError());
  }
  sync(ranks);
  printf("# %s BF16 Sum  out-of-place  sigma=%g seed=%llu warmup=%d iterations=%d\n",
         o.native ? "ncclReduceScatter" : "ZipCCLReduceScatter",o.sigma,o.seed,o.warmup,o.iterations);
  if (!o.native) puts(o.supplySigma ? "# sigma supplied" : "# sigma estimated by StatEncode");
  puts("# send_bytes/GPU  recv_count/GPU  time_us  algbw_GB/s  busbw_GB/s  wrong");
  bool passed = true;
  for (size_t requested = o.begin;;) {
    size_t count = requested / sizeof(BF) / o.gpus;
    size_t bytes = count * sizeof(BF) * o.gpus;
    for (int w = 0; w < o.warmup; ++w) collective(ranks, count, o);
    sync(ranks);
    auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < o.iterations; ++n) collective(ranks, count, o);
    sync(ranks);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/o.iterations;
    double alg = double(bytes)/seconds/1e9;
    printf("%16zu %15zu %9.2f %11.3f %11.3f ", bytes, count, seconds*1e6, alg, alg*(o.gpus-1)/o.gpus);
    if (o.validate) {
      unsigned long long wrong = 0;
      for (int r = 0; r < o.gpus; ++r) {
        auto& p = ranks[r];
        CHECK(cudaSetDevice(r));
        CHECK(cudaMemsetAsync(p.errors, 0, sizeof(*p.errors), p.stream));
        verify<<<256,256,0,p.stream>>>(p.recv,count,r,o.gpus,o.sigma,o.seed,p.errors);
        CHECK(cudaGetLastError());
      }
      sync(ranks);
      for (int r = 0; r < o.gpus; ++r) {
        unsigned long long value;
        CHECK(cudaSetDevice(r));
        CHECK(cudaMemcpy(&value,ranks[r].errors,sizeof(value),cudaMemcpyDeviceToHost));
        wrong += value;
      }
      printf("%llu\n",wrong);
      passed &= wrong == 0;
    } else puts("off");
    fflush(stdout);
    if (requested > o.end / o.factor) break;
    requested *= o.factor;
  }
  for (int r = 0; r < o.gpus; ++r) {
    CHECK(cudaSetDevice(r));
    auto& p = ranks[r];
    CHECK(ncclCommDestroy(p.comm));
    CHECK(cudaFree(p.errors)); CHECK(cudaFree(p.recv)); CHECK(cudaFree(p.send));
    CHECK(cudaStreamDestroy(p.stream));
  }
  return passed ? 0 : 2;
}
} // namespace

int main(int argc, char** argv) {
  try { return run(parse(argc,argv)); }
  catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
