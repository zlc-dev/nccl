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

constexpr size_t BlockElements = 131072;
constexpr int Threads = 256;

__global__ void fillInput(float* data, size_t recvcount, int nranks, int sourceRank) {
  size_t total = recvcount * static_cast<size_t>(nranks);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t destination = i / recvcount;
    size_t element = i - destination * recvcount;
    data[i] = static_cast<float>(sourceRank * 1000 + destination * 10 + element % 7);
  }
}

__global__ void encodeIdentity(const ncclCompressionTask_t* task) {
  extern __shared__ float tile[];
  volatile float* sharedTile = tile;
  const float* input = static_cast<const float*>(task->localInput);
  float* output = static_cast<float*>(task->compressedOutput);
  for (size_t base = blockIdx.x * blockDim.x; base < task->elementCount;
       base += static_cast<size_t>(gridDim.x) * blockDim.x) {
    size_t i = base + threadIdx.x;
    if (i < task->elementCount) sharedTile[threadIdx.x] = input[i];
    __syncthreads();
    if (i < task->elementCount) output[i] = sharedTile[threadIdx.x];
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    task->outputHeader->payloadBytes = static_cast<uint32_t>(task->elementCount * sizeof(float));
    task->outputHeader->flags = 0;
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

ncclResult_t launchEncode(const ncclCompressionTask_t* task, ncclDataType_t datatype,
                          void*, cudaStream_t stream) {
  if (datatype != ncclFloat) return ncclInvalidArgument;
  encodeIdentity<<<blocksFor(BlockElements), Threads, Threads * sizeof(float), stream>>>(task);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t launchDecodeReduceEncode(const ncclCompressionTask_t* task,
                                      ncclDataType_t datatype, ncclRedOp_t op,
                                      void*, cudaStream_t stream) {
  if (datatype != ncclFloat || op != ncclSum) return ncclInvalidArgument;
  decodeReduceEncodeIdentity<<<blocksFor(BlockElements), Threads, 2 * Threads * sizeof(float), stream>>>(task);
  return cudaGetLastError() == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

ncclResult_t launchDecodeReduceCopy(const ncclCompressionTask_t* task,
                                    ncclDataType_t datatype, ncclRedOp_t op,
                                    void*, cudaStream_t stream) {
  if (datatype != ncclFloat || op != ncclSum) return ncclInvalidArgument;
  decodeReduceCopyIdentity<<<blocksFor(BlockElements), Threads, 2 * Threads * sizeof(float), stream>>>(task);
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
            size_t maxBytes, int stepFactor, int warmupIters, int iters) {
  CUDACHECK(cudaSetDevice(rank));
  ncclComm_t comm;
  cudaStream_t stream;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const size_t maxRecvcount = maxBytes / sizeof(float);
  const size_t sendcount = maxRecvcount * static_cast<size_t>(nranks);
  float* sendbuff = nullptr;
  float* recvbuff = nullptr;
  CUDACHECK(cudaMalloc(&sendbuff, sendcount * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, maxRecvcount * sizeof(float)));

  ncclCompressionPlan_t plan = NCCL_COMPRESSION_PLAN_INITIALIZER;
  plan.blockElements = BlockElements;
  plan.maxCompressedBytes = maxIdentityBytes;
  plan.encode = launchEncode;
  plan.decodeReduceEncode = launchDecodeReduceEncode;
  plan.decodeReduceCopy = launchDecodeReduceCopy;

  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  float* deviceTime = nullptr;
  float* deviceMaxTime = nullptr;
  int* deviceErrors = nullptr;
  int* deviceTotalErrors = nullptr;
  CUDACHECK(cudaMalloc(&deviceTime, sizeof(float)));
  CUDACHECK(cudaMalloc(&deviceMaxTime, sizeof(float)));
  CUDACHECK(cudaMalloc(&deviceErrors, sizeof(int)));
  CUDACHECK(cudaMalloc(&deviceTotalErrors, sizeof(int)));

  if (rank == 0) {
    std::printf("# nThread 1 nGpus %d minBytes %zu maxBytes %zu stepFactor %d warmup %d iters %d\n",
                nranks, minBytes, maxBytes, stepFactor, warmupIters, iters);
    std::printf("# %-12s %-12s %-8s %-8s %-12s %-12s %-12s %-8s\n",
                "size", "count", "type", "op", "time(us)", "algbw(GB/s)",
                "busbw(GB/s)", "errors");
  }

  bool passed = true;
  double busBwSum = 0.0;
  int busBwCount = 0;
  for (size_t bytes = minBytes; bytes <= maxBytes;) {
    size_t recvcount = bytes / sizeof(float);
    size_t total = recvcount * static_cast<size_t>(nranks);
    fillInput<<<blocksFor(total), Threads, 0, stream>>>(sendbuff, recvcount, nranks, rank);
    CUDACHECK(cudaGetLastError());

    for (int i = 0; i < warmupIters; i++) {
      NCCLCHECK(ncclReduceScatterCompressed(sendbuff, recvbuff, recvcount, ncclFloat,
                                             ncclSum, comm, &plan, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
    }

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

    float localUs = static_cast<float>(totalMs * 1000.0 / iters);
    CUDACHECK(cudaMemcpyAsync(deviceTime, &localUs, sizeof(localUs), cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaMemcpyAsync(deviceErrors, &errors, sizeof(errors), cudaMemcpyHostToDevice, stream));
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce(deviceTime, deviceMaxTime, 1, ncclFloat, ncclMax, comm, stream));
    NCCLCHECK(ncclAllReduce(deviceErrors, deviceTotalErrors, 1, ncclInt, ncclSum, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    float maxUs = 0.0f;
    int totalErrors = 0;
    CUDACHECK(cudaMemcpyAsync(&maxUs, deviceMaxTime, sizeof(maxUs), cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaMemcpyAsync(&totalErrors, deviceTotalErrors, sizeof(totalErrors), cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    passed = passed && totalErrors == 0;

    if (rank == 0) {
      double sec = maxUs / 1.0e6;
      double algbw = static_cast<double>(bytes) * nranks / 1.0e9 / sec;
      double busbw = algbw * static_cast<double>(nranks - 1) / nranks;
      busBwSum += busbw;
      busBwCount++;
      std::printf("  %-12zu %-12zu %-8s %-8s %-12.1f %-12.2f %-12.2f %-8d\n",
                  bytes, recvcount, "float", "sum", maxUs, algbw, busbw, totalErrors);
      std::fflush(stdout);
    }
    if (bytes > maxBytes / static_cast<size_t>(stepFactor)) break;
    bytes *= static_cast<size_t>(stepFactor);
  }
  if (rank == 0 && busBwCount != 0)
    std::printf("# Avg bus bandwidth: %.2f GB/s\n", busBwSum / busBwCount);

  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  CUDACHECK(cudaFree(deviceTime));
  CUDACHECK(cudaFree(deviceMaxTime));
  CUDACHECK(cudaFree(deviceErrors));
  CUDACHECK(cudaFree(deviceTotalErrors));
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv) {
  if (argc == 10 && std::string(argv[1]) == "--worker") {
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
                   std::atoi(argv[7]), std::atoi(argv[8]), std::atoi(argv[9]));
  }
  size_t minBytes = 512 * 1024;
  size_t maxBytes = 64 * 1024 * 1024;
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
    else return EXIT_FAILURE;
  }
  int visibleDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&visibleDevices));
  int nranks = requestedRanks ? requestedRanks : std::min(visibleDevices, 4);
  if (nranks < 2 || nranks > visibleDevices) {
    std::fprintf(stderr, "Need 2..%d GPUs, requested %d\n", visibleDevices, nranks);
    return 77;
  }
  minBytes = std::max(minBytes, BlockElements * sizeof(float));
  minBytes -= minBytes % sizeof(float);
  maxBytes -= maxBytes % sizeof(float);
  if (maxBytes < minBytes || stepFactor < 2 || warmupIters < 0 || iters < 1) return EXIT_FAILURE;

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
  for (int rank = 0; rank < nranks; rank++) {
    pid_t pid = fork();
    if (pid == 0) {
      std::string rankText = std::to_string(rank);
      std::string ranksText = std::to_string(nranks);
      std::string minText = std::to_string(minBytes);
      std::string maxText = std::to_string(maxBytes);
      std::string factorText = std::to_string(stepFactor);
      std::string warmupText = std::to_string(warmupIters);
      std::string itersText = std::to_string(iters);
      execl(argv[0], argv[0], "--worker", rankText.c_str(), ranksText.c_str(), idHex.c_str(),
            minText.c_str(), maxText.c_str(), factorText.c_str(), warmupText.c_str(),
            itersText.c_str(), nullptr);
      std::perror("exec");
      _exit(EXIT_FAILURE);
    }
    if (pid < 0) {
      std::perror("fork");
      return EXIT_FAILURE;
    }
    children.push_back(pid);
  }
  bool passed = true;
  for (pid_t child : children) {
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
      passed = false;
  }
  if (!passed) return EXIT_FAILURE;
  std::printf("# compressed ReduceScatter benchmark passed on %d GPUs\n", nranks);
  return EXIT_SUCCESS;
}
