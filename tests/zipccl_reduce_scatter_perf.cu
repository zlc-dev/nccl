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
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using BF = __nv_bfloat16;

void check(cudaError_t e, const char* call) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(e));
}
void check(ncclResult_t e, const char* call) {
  if (e != ncclSuccess) throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(e));
}
#define CHECK(call) check((call), #call)

enum class Mix { None, Separate, Grouped };

struct Options {
  size_t begin = 1 << 20, end = 64 << 20;
  int gpus = 4, factor = 2, warmup = 5, iterations = 20, validate = 1;
  unsigned long long seed = 1234;
  float sigma = 0.02f;
  bool native = false;
  bool supplySigma = false;
  bool allOutliers = false;
  Mix mix = Mix::None;
};

void usage() {
  puts("zipccl_reduce_scatter_perf [-g GPUs] [-b minBytes] [-e maxBytes] [-f factor]\n"
       "  [-w warmups] [-n iterations] [-c 0|1] [--sigma stddev] [--seed integer] [--native]\n"
       "  [--mix separate|grouped] alternates AllGather and ReduceScatter on the same stream.\n"
       "Without --sigma, generate inputs at sigma=0.02 and let ZipCCL estimate it.\n"
       "--all-outliers uses positive BF16 inputs and supplies device sigma=0 to ZipCCL.\n"
       "Sizes are total uncompressed send bytes per GPU. K/M/G use powers of 1024.\n"
       "Each GPU receives size / GPUs bytes. BF16 Sum, out-of-place.\n"
       "Launches one process per GPU on the local node.");
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
    if (key == "--all-outliers") { o.allOutliers = true; continue; }
    // Accept the old flag in existing benchmark commands.
    if (key == "--multi-process") continue;
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    std::string val = argv[i];
    if (key == "-b") o.begin = number(val, true);
    else if (key == "-e") o.end = number(val, true);
    else if (key == "--seed") o.seed = number(val);
    else if (key == "--mix") {
      if (val == "separate") o.mix = Mix::Separate;
      else if (val == "grouped") o.mix = Mix::Grouped;
      else throw std::invalid_argument("--mix must be separate or grouped");
    } else if (key == "--sigma") {
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
  if (o.allOutliers && o.supplySigma)
    throw std::invalid_argument("--all-outliers sets the device sigma and cannot be combined with --sigma");
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
__device__ float4 sample(unsigned long long seed, int rank, size_t packet, bool allOutliers) {
  if (allOutliers) {
    if ((packet ^ unsigned(rank)) & 1) return make_float4(2, 1, 2, 1);
    return make_float4(1, 2, 1, 2);
  }
  uint64_t key = mix(seed + rank);
  float2 a = normalPair(mix(key + 2*packet));
  float2 b = normalPair(mix(key + 2*packet + 1));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ float component(float4 v, int lane) {
  return lane < 2 ? (lane ? v.y : v.x) : (lane == 2 ? v.z : v.w);
}
__global__ void generate(BF* data, size_t count, unsigned long long seed, int rank, float scale,
                         bool allOutliers, float* deviceSigma) {
  if (deviceSigma && blockIdx.x == 0 && threadIdx.x == 0) *deviceSigma = allOutliers ? 0.0f : scale;
  for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x)*4; i < count;
       i += size_t(gridDim.x)*blockDim.x*4) {
    float4 v = sample(seed, rank, i/4, allOutliers);
    for (int j = 0; j < 4 && i+j < count; ++j) data[i+j] = __float2bfloat16_rn(scale * component(v,j));
  }
}
__device__ float addBF(float a, float b) { return __bfloat162float(__float2bfloat16_rn(a+b)); }

__global__ void verifyGather(const BF* output, size_t count, int ranks, float scale,
                             unsigned long long seed, bool allOutliers, unsigned long long* errors) {
  unsigned long long localErrors = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count * ranks; i += size_t(blockDim.x)*gridDim.x) {
    size_t index = i % count;
    BF expected = __float2bfloat16_rn(scale * component(sample(seed, i / count, index / 4, allOutliers), int(index % 4)));
    localErrors += __bfloat16_as_ushort(output[i]) != __bfloat16_as_ushort(expected);
  }
  if (localErrors) atomicAdd(errors, localErrors);
}

__global__ void verify(const BF* output, size_t count, int dest, int ranks, float scale,
                       unsigned long long seed, bool allOutliers, unsigned long long* errors) {
  unsigned long long localErrors = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(blockDim.x)*gridDim.x) {
    size_t index = size_t(dest)*count + i;
    float others[3] = {}, last = 0;
    int n = 0;
    double sum = 0, magnitude = 0;
    for (int r = 0; r < ranks; ++r) {
      float x = __bfloat162float(__float2bfloat16_rn(
          scale * component(sample(seed, r, index/4, allOutliers), int(index%4))));
      sum += x;
      magnitude += fabs(double(x));
      if (r == dest) last = x;
      else if (ranks <= 4) others[n++] = x;
    }
    float got = __bfloat162float(output[i]);
    // Every directed Ring finishes with the destination rank's own summand.
    bool ok;
    if (allOutliers && ranks <= 128) ok = got == sum; // All positive integer partial sums fit exactly in BF16.
    else if (ranks == 1) ok = got == last;
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
  BF* gathered;
  float* sigma;
  unsigned long long* errors;
};

struct RankResult {
  double seconds;
  unsigned long long wrong;
};

struct SharedRun {
  pthread_barrier_t barrier;
  ncclUniqueId id;
  RankResult* results;

  void wait() {
    int result = pthread_barrier_wait(&barrier);
    if (result != 0 && result != PTHREAD_BARRIER_SERIAL_THREAD)
      throw std::runtime_error(std::string("pthread_barrier_wait: ") + std::strerror(result));
  }
};

void collective(const Rank& p, size_t count, const Options& o) {
  if (o.mix == Mix::Grouped) CHECK(ncclGroupStart());
  if (o.mix != Mix::None)
    CHECK(ncclAllGather(p.send, p.gathered, count, ncclBfloat16, p.comm, p.stream));
  if (o.native)
    CHECK(ncclReduceScatter(p.send, p.recv, count, ncclBfloat16, ncclSum, p.comm, p.stream));
  else
    CHECK(ncclZipCCLReduceScatterWithSigma(p.send, p.recv, count, ncclBfloat16, ncclSum, p.comm, p.stream, p.sigma));
  if (o.mix == Mix::Grouped) CHECK(ncclGroupEnd());
}

int runRank(const Options& o, int rank, SharedRun& shared) {
  const float scale = o.allOutliers ? 1.0f : o.sigma;
  int available;
  CHECK(cudaGetDeviceCount(&available));
  if (o.gpus > available) throw std::runtime_error("requested more GPUs than are visible");

  CHECK(cudaSetDevice(rank));
  if (rank == 0) CHECK(ncclGetUniqueId(&shared.id));
  shared.wait();
  Rank p{};
  CHECK(ncclCommInitRank(&p.comm, o.gpus, shared.id, rank));

  size_t maxCount = o.end / sizeof(BF) / o.gpus;
  CHECK(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
  CHECK(cudaMalloc(&p.send, maxCount * o.gpus * sizeof(BF)));
  CHECK(cudaMalloc(&p.recv, maxCount * sizeof(BF)));
  CHECK(cudaMalloc(&p.errors, sizeof(*p.errors)));
  if (o.mix != Mix::None) CHECK(cudaMalloc(&p.gathered, maxCount * o.gpus * sizeof(BF)));
  if (!o.native && (o.supplySigma || o.allOutliers)) CHECK(cudaMalloc(&p.sigma, sizeof(*p.sigma)));
  generate<<<256,256,0,p.stream>>>(p.send, maxCount * o.gpus, o.seed, rank, scale, o.allOutliers, p.sigma);
  CHECK(cudaGetLastError());
  CHECK(cudaStreamSynchronize(p.stream));

  if (rank == 0) {
    printf("# %s BF16 Sum, %d ranks, warmup=%d iterations=%d\n",
           o.native ? "ncclReduceScatter" : "ZipCCLReduceScatter", o.gpus, o.warmup, o.iterations);
    if (o.mix != Mix::None)
      printf("# AllGather + ReduceScatter (%s); bandwidth uses ReduceScatter bytes and pair time\n",
             o.mix == Mix::Grouped ? "grouped" : "separate");
    puts("# send_bytes/GPU  recv_count/GPU  time_us  algbw_GB/s  busbw_GB/s  wrong");
    fflush(stdout);
  }

  bool passed = true;
  for (size_t requested = o.begin;;) {
    size_t count = requested / sizeof(BF) / o.gpus;
    size_t bytes = count * sizeof(BF) * o.gpus;
    for (int w = 0; w < o.warmup; ++w) collective(p, count, o);
    CHECK(cudaStreamSynchronize(p.stream));
    shared.wait();

    auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < o.iterations; ++n) collective(p, count, o);
    CHECK(cudaStreamSynchronize(p.stream));
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / o.iterations;

    unsigned long long wrong = 0;
    if (o.validate) {
      CHECK(cudaMemsetAsync(p.errors, 0, sizeof(*p.errors), p.stream));
      verify<<<256,256,0,p.stream>>>(p.recv, count, rank, o.gpus, scale, o.seed, o.allOutliers, p.errors);
      CHECK(cudaGetLastError());
      if (p.gathered) {
        verifyGather<<<256,256,0,p.stream>>>(p.gathered, count, o.gpus, scale, o.seed, o.allOutliers, p.errors);
        CHECK(cudaGetLastError());
      }
      CHECK(cudaStreamSynchronize(p.stream));
      CHECK(cudaMemcpy(&wrong, p.errors, sizeof(wrong), cudaMemcpyDeviceToHost));
    }

    shared.results[rank] = {seconds, wrong};
    shared.wait();
    if (rank == 0) {
      wrong = 0;
      for (int r = 0; r < o.gpus; ++r) {
        seconds = std::max(seconds, shared.results[r].seconds);
        wrong += shared.results[r].wrong;
      }
      double alg = double(bytes) / seconds / 1e9;
      printf("%16zu %15zu %9.2f %11.3f %11.3f ", bytes, count, seconds * 1e6, alg, alg * (o.gpus - 1) / o.gpus);
      if (o.validate) printf("%llu\n", wrong);
      else puts("off");
      passed &= wrong == 0;
      fflush(stdout);
    }
    if (requested > o.end / o.factor) break;
    requested *= o.factor;
  }

  CHECK(ncclCommDestroy(p.comm));
  if (p.gathered) CHECK(cudaFree(p.gathered));
  if (p.sigma) CHECK(cudaFree(p.sigma));
  CHECK(cudaFree(p.errors));
  CHECK(cudaFree(p.recv));
  CHECK(cudaFree(p.send));
  CHECK(cudaStreamDestroy(p.stream));
  return passed ? 0 : 2;
}

int runProcesses(const Options& o) {
  std::vector<pid_t> children;
  children.reserve(o.gpus);
  size_t bytes = sizeof(SharedRun) + size_t(o.gpus) * sizeof(RankResult);
  void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) throw std::runtime_error(std::string("mmap: ") + std::strerror(errno));
  auto* shared = new (mapping) SharedRun{};
  shared->results = new (shared + 1) RankResult[o.gpus]{};
  pthread_barrierattr_t attr;
  int error = pthread_barrierattr_init(&attr);
  if (!error) {
    error = pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    if (!error) error = pthread_barrier_init(&shared->barrier, &attr, o.gpus);
    pthread_barrierattr_destroy(&attr);
  }
  if (error) {
    munmap(mapping, bytes);
    throw std::runtime_error(std::string("initializing process barrier: ") + std::strerror(error));
  }

  // The parent only launches and reaps children. CUDA is initialized inside each rank.
  const pid_t parent = getpid();
  int result = 0;
  for (int rank = 0; rank < o.gpus; ++rank) {
    pid_t child = fork();
    if (child == 0) {
      if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(1);
      try {
        _exit(runRank(o, rank, *shared));
      } catch (const std::exception& e) {
        fprintf(stderr, "rank %d: %s\n", rank, e.what());
        _exit(1);
      }
    }
    if (child < 0) {
      fprintf(stderr, "fork: %s\n", std::strerror(errno));
      result = 1;
      break;
    }
    children.push_back(child);
  }

  while (result != 1 && !children.empty()) {
    int status;
    pid_t child = waitpid(-1, &status, 0);
    if (child < 0) {
      if (errno == EINTR) continue;
      fprintf(stderr, "waitpid: %s\n", std::strerror(errno));
      result = 1;
      break;
    }
    auto it = std::find(children.begin(), children.end(), child);
    if (it == children.end()) continue;
    children.erase(it);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 2) {
      result = 2;
    } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "benchmark worker %ld failed\n", static_cast<long>(child));
      result = 1;
    }
  }
  if (result == 1) {
    for (pid_t child : children) kill(child, SIGKILL);
    for (pid_t child : children) {
      while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
    }
  } else {
    pthread_barrier_destroy(&shared->barrier);
  }
  munmap(mapping, bytes);
  return result;
}
} // namespace

int main(int argc, char** argv) {
  try {
    return runProcesses(parse(argc, argv));
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
