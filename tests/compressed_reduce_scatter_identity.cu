/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <vector>

#define CUDACHECK(cmd) do {                                                    \
  cudaError_t error = (cmd);                                                   \
  if (error != cudaSuccess) {                                                  \
    std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,       \
                 cudaGetErrorString(error));                                   \
    std::exit(EXIT_FAILURE);                                                   \
  }                                                                            \
} while (0)

#define NCCLCHECK(cmd) do {                                                    \
  ncclResult_t result = (cmd);                                                 \
  if (result != ncclSuccess) {                                                 \
    std::fprintf(stderr, "%s:%d NCCL error: %s\n", __FILE__, __LINE__,       \
                 ncclGetErrorString(result));                                  \
    std::exit(EXIT_FAILURE);                                                   \
  }                                                                            \
} while (0)

namespace {

constexpr size_t MinBlockBytes = 512 * 1024;
constexpr size_t DefaultBlockBytes = 1024 * 1024;
constexpr int Threads = 256;

struct RankResult {
  float timeUs;
  int errors;
};

struct IdentityContext {
  size_t blockElements;
};

bool writeAll(int fd, const void* data, size_t bytes) {
  const char* p = static_cast<const char*>(data);
  while (bytes != 0) {
    ssize_t n = write(fd, p, bytes);
    if (n <= 0) return false;
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

bool readAll(int fd, void* data, size_t bytes) {
  char* p = static_cast<char*>(data);
  while (bytes != 0) {
    ssize_t n = read(fd, p, bytes);
    if (n <= 0) return false;
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

__global__ void fillInput(float* data, size_t recvcount, int nranks, int sourceRank) {
  size_t total = recvcount * static_cast<size_t>(nranks);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t destination = i / recvcount;
    size_t element = i - destination * recvcount;
    data[i] = static_cast<float>(sourceRank * 1000 + destination * 10 + element % 7);
  }
}

__global__ void encodeIdentity(const ncclCompressionTask_t* tasks,
                               size_t taskCount, int blocksPerTask) {
  extern __shared__ float tile[];
  volatile float* sharedTile = tile;
  size_t workCount = taskCount * static_cast<size_t>(blocksPerTask);
  for (size_t work = blockIdx.x; work < workCount; work += gridDim.x) {
    const ncclCompressionTask_t* task = tasks + work / blocksPerTask;
    int taskBlock = static_cast<int>(work % blocksPerTask);
    const float* input = static_cast<const float*>(task->localInput);
    float* output = static_cast<float*>(task->compressedOutput);
    for (size_t base = static_cast<size_t>(taskBlock) * blockDim.x;
         base < task->elementCount;
         base += static_cast<size_t>(blocksPerTask) * blockDim.x) {
      size_t i = base + threadIdx.x;
      if (i < task->elementCount) sharedTile[threadIdx.x] = input[i];
      __syncthreads();
      if (i < task->elementCount) output[i] = sharedTile[threadIdx.x];
      __syncthreads();
    }
    if (taskBlock == 0 && threadIdx.x == 0) {
      task->outputHeader->payloadBytes =
          static_cast<uint32_t>(task->elementCount * sizeof(float));
      task->outputHeader->flags = 0;
    }
    __syncthreads();
  }
}

__global__ void decodeReduceEncodeIdentity(const ncclCompressionTask_t* task) {
  extern __shared__ float tile[];
  volatile float* compressedTile = tile;
  volatile float* localTile = tile + blockDim.x;
  const float* input = static_cast<const float*>(task->compressedInput);
  const float* local = static_cast<const float*>(task->localInput);
  float* output = static_cast<float*>(task->compressedOutput);
  for (size_t base = blockIdx.x * blockDim.x; base < task->elementCount;
       base += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t i = base + threadIdx.x;
    if (i < task->elementCount) {
      compressedTile[threadIdx.x] = input[i];
      localTile[threadIdx.x] = local[i];
    }
    __syncthreads();
    if (i < task->elementCount) output[i] = compressedTile[threadIdx.x] + localTile[threadIdx.x];
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    uint32_t expected = static_cast<uint32_t>(task->elementCount * sizeof(float));
    task->outputHeader->payloadBytes = expected;
    task->outputHeader->flags = task->inputHeader->flags |
        (task->inputHeader->payloadBytes == expected ? 0u : NCCL_COMPRESSED_FRAME_FLAG_SIZE_ERROR);
  }
}

__global__ void decodeReduceCopyIdentity(const ncclCompressionTask_t* task) {
  extern __shared__ float tile[];
  volatile float* compressedTile = tile;
  volatile float* localTile = tile + blockDim.x;
  const float* input = static_cast<const float*>(task->compressedInput);
  const float* local = static_cast<const float*>(task->localInput);
  float* output = static_cast<float*>(task->output);
  for (size_t base = blockIdx.x * blockDim.x; base < task->elementCount;
       base += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t i = base + threadIdx.x;
    if (i < task->elementCount) {
      compressedTile[threadIdx.x] = input[i];
      localTile[threadIdx.x] = local[i];
    }
    __syncthreads();
    if (i < task->elementCount) output[i] = compressedTile[threadIdx.x] + localTile[threadIdx.x];
    __syncthreads();
  }
}

int blocksFor(size_t elements) {
  return static_cast<int>(std::min<size_t>(1024, (elements + Threads - 1) / Threads));
}

ncclResult_t maxIdentityBytes(size_t elementCount, ncclDataType_t datatype,
                              void*, size_t* bytes) {
  if (datatype != ncclFloat || bytes == nullptr || elementCount > UINT32_MAX / sizeof(float))
    return ncclInvalidArgument;
  *bytes = elementCount * sizeof(float);
  return ncclSuccess;
}

ncclResult_t launchEncode(const ncclCompressionTask_t* tasks, size_t taskCount,
                          ncclDataType_t datatype,
                          void* context, cudaStream_t stream) {
  if (datatype != ncclFloat || context == nullptr || taskCount == 0)
    return ncclInvalidArgument;
  const IdentityContext* identity = static_cast<const IdentityContext*>(context);
  int blocksPerTask = blocksFor(identity->blockElements);
  if (taskCount > SIZE_MAX / static_cast<size_t>(blocksPerTask))
    return ncclInvalidArgument;
  size_t workCount = taskCount * static_cast<size_t>(blocksPerTask);
  int blocks = static_cast<int>(std::min<size_t>(65535, workCount));
  encodeIdentity<<<blocks, Threads, Threads * sizeof(float), stream>>>(
      tasks, taskCount, blocksPerTask);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t launchDecodeReduceEncode(const ncclCompressionTask_t* task,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      void* context, cudaStream_t stream) {
  if (datatype != ncclFloat || op != ncclSum || context == nullptr) return ncclInvalidArgument;
  const IdentityContext* identity = static_cast<const IdentityContext*>(context);
  decodeReduceEncodeIdentity<<<blocksFor(identity->blockElements), Threads, 2 * Threads * sizeof(float), stream>>>(task);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t launchDecodeReduceCopy(const ncclCompressionTask_t* task,
                                    ncclDataType_t datatype, ncclRedOp_t op,
                                    void* context, cudaStream_t stream) {
  if (datatype != ncclFloat || op != ncclSum || context == nullptr) return ncclInvalidArgument;
  const IdentityContext* identity = static_cast<const IdentityContext*>(context);
  decodeReduceCopyIdentity<<<blocksFor(identity->blockElements), Threads, 2 * Threads * sizeof(float), stream>>>(task);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

float expectedValue(int nranks, int destination, size_t element) {
  int sourceSum = nranks * (nranks - 1) / 2;
  return static_cast<float>(1000 * sourceSum + nranks * (destination * 10 + element % 7));
}

} // namespace

size_t parseBytes(const char* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 0);
  if (end && *end) {
    if (*end == 'K' || *end == 'k') value *= 1024ull;
    else if (*end == 'M' || *end == 'm') value *= 1024ull * 1024;
    else if (*end == 'G' || *end == 'g') value *= 1024ull * 1024 * 1024;
  }
  return static_cast<size_t>(value);
}

int runRank(int rank, int nranks, const ncclUniqueId& id, size_t minBytes,
            size_t maxBytes, int stepFactor, int warmupIters, int iters,
            size_t blockElements, int resultFd, int releaseFd) {
  CUDACHECK(cudaSetDevice(rank));
  ncclComm_t comm;
  cudaStream_t stream;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const size_t maxRecvcount = maxBytes / sizeof(float);
  const size_t sendcount = maxRecvcount * static_cast<size_t>(nranks);
  float* sendbuff = nullptr;
  float* recvbuff = nullptr;
  int* interopBarrier = nullptr;
  CUDACHECK(cudaMalloc(&sendbuff, sendcount * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, maxRecvcount * sizeof(float)));
  CUDACHECK(cudaMalloc(&interopBarrier, sizeof(*interopBarrier)));

  IdentityContext identity = {blockElements};
  ncclCompressionPlan_t plan = NCCL_COMPRESSION_PLAN_INITIALIZER;
  plan.blockElements = blockElements;
  plan.context = &identity;
  plan.maxCompressedBytes = maxIdentityBytes;
  plan.encode = launchEncode;
  plan.decodeReduceEncode = launchDecodeReduceEncode;
  plan.decodeReduceCopy = launchDecodeReduceCopy;

  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  bool passed = true;
  for (size_t bytes = minBytes; bytes <= maxBytes;) {
    size_t recvcount = bytes / sizeof(float);
    size_t total = recvcount * static_cast<size_t>(nranks);
    fillInput<<<blocksFor(total), Threads, 0, stream>>>(sendbuff, recvcount, nranks, rank);
    CUDACHECK(cudaGetLastError());
    for (int i = 0; i < warmupIters; i++) {
      NCCLCHECK(ncclReduceScatterCompressed(sendbuff, recvbuff, recvcount, ncclFloat,
                                             ncclSum, comm, &plan, stream));
    }
    CUDACHECK(cudaStreamSynchronize(stream));

    double totalMs = 0.0;
    for (int i = 0; i < iters; i++) {
      CUDACHECK(cudaEventRecord(start, stream));
      NCCLCHECK(ncclReduceScatterCompressed(sendbuff, recvbuff, recvcount, ncclFloat,
                                             ncclSum, comm, &plan, stream));
      CUDACHECK(cudaEventRecord(stop, stream));
      CUDACHECK(cudaEventSynchronize(stop));
      float ms = 0.0f;
      CUDACHECK(cudaEventElapsedTime(&ms, start, stop));
      totalMs += ms;
    }

    int errors = 0;
    std::vector<float> result(recvcount);
    CUDACHECK(cudaMemcpy(result.data(), recvbuff, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < recvcount; i++) {
      if (result[i] != expectedValue(nranks, rank, i)) errors++;
    }

    // Exercise a normal NCCL collective immediately after the compressed
    // operation. This is outside the timed region and guards its completion
    // boundary and communicator-resource lifetime.
    CUDACHECK(cudaMemsetAsync(interopBarrier, 0, sizeof(*interopBarrier), stream));
    NCCLCHECK(ncclAllReduce(interopBarrier, interopBarrier, 1, ncclInt, ncclSum,
                            comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    RankResult rankResult = {static_cast<float>(totalMs * 1000.0 / iters), errors};
    if (!writeAll(resultFd, &rankResult, sizeof(rankResult))) return EXIT_FAILURE;
    char release = 0;
    if (!readAll(releaseFd, &release, sizeof(release))) return EXIT_FAILURE;
    passed = passed && errors == 0;
    if (bytes > maxBytes / static_cast<size_t>(stepFactor)) break;
    bytes *= static_cast<size_t>(stepFactor);
  }

  // The compressed communicator owns tail events recorded on the user stream.
  // Tear it down while that stream and all buffers referenced by submitted
  // work are still alive.
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  CUDACHECK(cudaFree(interopBarrier));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv) {
  if (argc == 13 && std::string(argv[1]) == "--worker") {
    int rank = std::atoi(argv[2]);
    int nranks = std::atoi(argv[3]);
    ncclUniqueId id;
    const char* hex = argv[4];
    if (std::strlen(hex) != sizeof(id.internal) * 2) return EXIT_FAILURE;
    for (size_t i = 0; i < sizeof(id.internal); i++) {
      unsigned int byte = 0;
      if (std::sscanf(hex + 2 * i, "%2x", &byte) != 1) return EXIT_FAILURE;
      id.internal[i] = static_cast<char>(byte);
    }
    return runRank(rank, nranks, id, parseBytes(argv[5]), parseBytes(argv[6]),
                   std::atoi(argv[7]), std::atoi(argv[8]), std::atoi(argv[9]),
                   static_cast<size_t>(std::strtoull(argv[10], nullptr, 0)),
                   std::atoi(argv[11]), std::atoi(argv[12]));
  }
  size_t minBytes = DefaultBlockBytes;
  size_t maxBytes = 64 * 1024 * 1024;
  size_t blockBytes = DefaultBlockBytes;
  int stepFactor = 2;
  int warmupIters = 5;
  int iters = 20;
  int requestedRanks = 0;
  for (int i = 1; i < argc; i++) {
    if (i + 1 >= argc) return EXIT_FAILURE;
    if (std::string(argv[i]) == "-b") minBytes = parseBytes(argv[++i]);
    else if (std::string(argv[i]) == "-e") maxBytes = parseBytes(argv[++i]);
    else if (std::string(argv[i]) == "-f") stepFactor = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "-w") warmupIters = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "-n") iters = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "-g") requestedRanks = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "-c") blockBytes = parseBytes(argv[++i]);
    else return EXIT_FAILURE;
  }
  int visibleDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&visibleDevices));
  int nranks = requestedRanks ? requestedRanks : std::min(visibleDevices, 4);
  if (nranks < 2 || nranks > visibleDevices) {
    std::fprintf(stderr, "Need 2..%d GPUs, requested %d\n", visibleDevices, nranks);
    return 77;
  }
  minBytes = std::max(minBytes, MinBlockBytes);
  minBytes -= minBytes % sizeof(float);
  maxBytes -= maxBytes % sizeof(float);
  blockBytes -= blockBytes % sizeof(float);
  if (maxBytes < minBytes || blockBytes < MinBlockBytes ||
      blockBytes > UINT32_MAX - UINT32_MAX % sizeof(float) ||
      stepFactor < 2 || warmupIters < 0 || iters < 1) return EXIT_FAILURE;
  size_t blockElements = blockBytes / sizeof(float);

  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));
  static const char digits[] = "0123456789abcdef";
  std::string idHex(sizeof(id.internal) * 2, '0');
  for (size_t i = 0; i < sizeof(id.internal); i++) {
    unsigned char byte = static_cast<unsigned char>(id.internal[i]);
    idHex[2 * i] = digits[byte >> 4];
    idHex[2 * i + 1] = digits[byte & 15];
  }
  std::vector<pid_t> children;
  std::vector<int> resultFds;
  std::vector<int> releaseFds;
  for (int rank = 0; rank < nranks; rank++) {
    int resultPipe[2];
    int releasePipe[2];
    if (pipe(resultPipe) != 0 || pipe(releasePipe) != 0) {
      std::perror("pipe");
      return EXIT_FAILURE;
    }
    pid_t pid = fork();
    if (pid == 0) {
      close(resultPipe[0]);
      close(releasePipe[1]);
      std::string rankText = std::to_string(rank);
      std::string ranksText = std::to_string(nranks);
      std::string minText = std::to_string(minBytes);
      std::string maxText = std::to_string(maxBytes);
      std::string factorText = std::to_string(stepFactor);
      std::string warmupText = std::to_string(warmupIters);
      std::string itersText = std::to_string(iters);
      std::string blockElementsText = std::to_string(blockElements);
      std::string resultFdText = std::to_string(resultPipe[1]);
      std::string releaseFdText = std::to_string(releasePipe[0]);
      execl(argv[0], argv[0], "--worker", rankText.c_str(), ranksText.c_str(), idHex.c_str(),
            minText.c_str(), maxText.c_str(), factorText.c_str(), warmupText.c_str(),
            itersText.c_str(), blockElementsText.c_str(), resultFdText.c_str(),
            releaseFdText.c_str(), nullptr);
      std::perror("exec");
      _exit(EXIT_FAILURE);
    }
    if (pid < 0) {
      std::perror("fork");
      return EXIT_FAILURE;
    }
    close(resultPipe[1]);
    close(releasePipe[0]);
    children.push_back(pid);
    resultFds.push_back(resultPipe[0]);
    releaseFds.push_back(releasePipe[1]);
  }

  std::printf("# nThread 1 nGpus %d minBytes %zu maxBytes %zu blockBytes %zu stepFactor %d warmup %d iters %d\n",
              nranks, minBytes, maxBytes, blockBytes, stepFactor, warmupIters, iters);
  std::printf("# %-12s %-12s %-8s %-8s %-12s %-12s %-12s %-8s\n",
              "size", "count", "type", "op", "time(us)", "algbw(GB/s)",
              "busbw(GB/s)", "errors");
  double busBwSum = 0.0;
  int busBwCount = 0;
  bool passed = true;
  for (size_t bytes = minBytes; bytes <= maxBytes;) {
    float maxUs = 0.0f;
    int totalErrors = 0;
    for (int rank = 0; rank < nranks; rank++) {
      RankResult result = {};
      if (!readAll(resultFds[rank], &result, sizeof(result))) {
        std::fprintf(stderr, "rank %d exited before reporting size %zu\n", rank, bytes);
        passed = false;
        break;
      }
      maxUs = std::max(maxUs, result.timeUs);
      totalErrors += result.errors;
    }
    if (!passed) break;
    double sec = maxUs / 1.0e6;
    double algbw = static_cast<double>(bytes) * nranks / 1.0e9 / sec;
    double busbw = algbw * static_cast<double>(nranks - 1) / nranks;
    busBwSum += busbw;
    busBwCount++;
    std::printf("  %-12zu %-12zu %-8s %-8s %-12.1f %-12.2f %-12.2f %-8d\n",
                bytes, bytes / sizeof(float), "float", "sum", maxUs, algbw, busbw,
                totalErrors);
    std::fflush(stdout);
    passed = passed && totalErrors == 0;
    char release = 0;
    for (int rank = 0; rank < nranks; rank++)
      if (!writeAll(releaseFds[rank], &release, sizeof(release))) passed = false;
    if (bytes > maxBytes / static_cast<size_t>(stepFactor)) break;
    bytes *= static_cast<size_t>(stepFactor);
  }
  if (busBwCount != 0)
    std::printf("# Avg bus bandwidth: %.2f GB/s\n", busBwSum / busBwCount);

  for (size_t rank = 0; rank < children.size(); rank++) {
    int status = 0;
    pid_t child = children[rank];
    if (waitpid(child, &status, 0) != child) {
      std::fprintf(stderr, "failed to wait for rank %zu worker\n", rank);
      passed = false;
    } else if (WIFSIGNALED(status)) {
      std::fprintf(stderr, "rank %zu worker terminated by signal %d\n", rank, WTERMSIG(status));
      passed = false;
    } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::fprintf(stderr, "rank %zu worker exited with status %d\n", rank,
                   WIFEXITED(status) ? WEXITSTATUS(status) : -1);
      passed = false;
    }
  }
  for (int fd : resultFds) close(fd);
  for (int fd : releaseFds) close(fd);
  if (!passed) return EXIT_FAILURE;
  std::printf("# compressed ReduceScatter benchmark passed on %d GPUs\n", nranks);
  return EXIT_SUCCESS;
}
