/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#include "argcheck.h"
#include "bootstrap.h"
#include "collectives.h"
#include "comm.h"
#include "compressed_reduce_scatter.h"
#include "core.h"
#include "group.h"
#include "ipcsocket.h"
#include "strongstream.h"
#include "transport.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

struct alignas(64) ncclCompressedRsEventRecordSync {
  uint64_t recordIssued;
  char padding[64 - sizeof(uint64_t)];
};

struct ncclCompressedRsEventWindowSync {
  ncclCompressedRsEventRecordSync generation[NCCL_COMPRESSED_EVENT_GENERATIONS];
  alignas(64) uint64_t waitIssuedTail;
  char padding[64 - sizeof(uint64_t)];
};

struct ncclCompressedRsRxSync {
  ncclCompressedRsEventWindowSync dataReady;
  ncclCompressedRsEventWindowSync canWrite;
};

struct ncclCompressedRsHostSyncPage {
  ncclCompressedRsRxSync rx[MAXCHANNELS][NCCL_COMPRESSED_RX_DEPTH];
  uint64_t importsClosed[MAXCHANNELS];
};

struct ncclCompressedRsRing {
  int channel = -1;
  int next = -1;
  int prev = -1;
  cudaMemPool_t nextPool = nullptr;
  bool ownsNextPool = false;
  ncclCompressedRsHostSyncPage* nextSync = nullptr;
  ncclShmHandleInternal* nextSyncHandle = nullptr;
  cudaEvent_t localDataReady[NCCL_COMPRESSED_RX_DEPTH][NCCL_COMPRESSED_EVENT_GENERATIONS] = {};
  cudaEvent_t localCanWrite[NCCL_COMPRESSED_RX_DEPTH][NCCL_COMPRESSED_EVENT_GENERATIONS] = {};
  cudaEvent_t nextDataReady[NCCL_COMPRESSED_RX_DEPTH][NCCL_COMPRESSED_EVENT_GENERATIONS] = {};
  cudaEvent_t nextCanWrite[NCCL_COMPRESSED_RX_DEPTH][NCCL_COMPRESSED_EVENT_GENERATIONS] = {};
  uint64_t localDataReadyPhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t localCanWritePhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t nextDataReadyPhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t nextCanWritePhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t localDataReadyWaitPublished[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t nextCanWriteWaitPublished[NCCL_COMPRESSED_RX_DEPTH] = {};
  bool localCreditInitialized[NCCL_COMPRESSED_RX_DEPTH] = {};
  cudaStream_t sendStream = nullptr;
  void* cachedRx = nullptr;
  void* cachedRemoteRx = nullptr;
};

struct ncclCompressedRsManager {
  std::mutex mutex;
  cudaMemPool_t localPool = nullptr;
  ncclCompressedRsHostSyncPage* localSync = nullptr;
  ncclShmHandleInternal* localSyncHandle = nullptr;
  ncclCompressedRsRing rings[MAXCHANNELS];
  int ringCount = 0;
  cudaEvent_t tail = nullptr;
  uint64_t sequence = 0;
  size_t cachedStride = 0;
  int cachedRingCount = 0;
  void* cachedPreencoded = nullptr;
  ncclCompressionTask_t* cachedPreencodeTasks = nullptr;
  int* cachedPreencodeDestinations = nullptr;
  size_t cachedPreencodeBlocks = 0;
  bool tailRecorded = false;
  bool initialized = false;
};

namespace {
constexpr size_t MinBlockBytes = 512 * 1024;
constexpr size_t TargetSliceBytes = 256 * 1024;
constexpr int MaxSliceCtas = 4;
constexpr int DefaultActiveRings = 1;
constexpr size_t SlotAlignment = 256;
constexpr size_t PayloadOffset = (sizeof(ncclCompressedP2pSlot) + 15) & ~size_t(15);
constexpr uint64_t IpcSocketHashSalt = 0x435253504f4f4cULL;

struct ManagerInitInfo {
  char syncPath[PATH_MAX];
  cudaIpcEventHandle_t dataReady[MAXCHANNELS][NCCL_COMPRESSED_RX_DEPTH]
                                    [NCCL_COMPRESSED_EVENT_GENERATIONS];
  cudaIpcEventHandle_t canWrite[MAXCHANNELS][NCCL_COMPRESSED_RX_DEPTH]
                                   [NCCL_COMPRESSED_EVENT_GENERATIONS];
};

struct OperationRing {
  void* rx = nullptr;
  void* remoteRx = nullptr;
};

struct Operation {
  ncclComm* comm = nullptr;
  ncclCompressedRsManager* manager = nullptr;
  OperationRing rings[MAXCHANNELS];
  void* preencoded = nullptr;
  ncclCompressionTask_t* preencodeTasks = nullptr;
  int* preencodeDestinations = nullptr;
  size_t stride = 0;
  uint64_t sequence = 0;
};

struct Job {
  ncclAsyncJob base;
  const void* sendbuff;
  void* recvbuff;
  size_t recvcount;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  ncclComm* comm;
  ncclCompressionPlan_t plan;
  cudaStream_t stream;
};

struct RingLaunchState {
  cudaEvent_t finish = nullptr;
  bool localCreditPending[NCCL_COMPRESSED_RX_DEPTH] = {};
  size_t blocks[NCCL_COMPRESSED_RX_DEPTH - 1] = {};
  int baseLanes[NCCL_COMPRESSED_RX_DEPTH - 1] = {};
  bool initialPending[NCCL_COMPRESSED_RX_DEPTH - 1] = {};
  size_t nextBlocks[NCCL_COMPRESSED_RX_DEPTH - 1] = {};
  int nextBaseLanes[NCCL_COMPRESSED_RX_DEPTH - 1] = {};
  bool nextInitialPending[NCCL_COMPRESSED_RX_DEPTH - 1] = {};
  int batchCount = 0;
  int nextBatchCount = 0;
  int token = -1;
  uint32_t round = 0;
  size_t nextBlockOrdinal = 0;
  size_t initialSendsSubmitted = 0;
  size_t completedBlocks = 0;
  size_t blockCount = 0;
  bool cohortComplete = false;
};

static size_t alignUp(size_t n, size_t a) { return (n + a - 1) / a * a; }
static char* slot(void* arena, size_t stride, int i) {
  return static_cast<char*>(arena) + stride * static_cast<size_t>(i);
}
static char* payload(void* frame) {
  return static_cast<char*>(frame) + PayloadOffset;
}
static ncclCompressedFrameHeader_t* header(void* frame) {
  return &static_cast<ncclCompressedP2pSlot*>(frame)->frame;
}
static ncclResult_t record(cudaEvent_t e, cudaStream_t s) {
  CUDACHECK(cudaEventRecord(e, s));
  return ncclSuccess;
}

static bool traceEnabled() {
  static bool enabled = std::getenv("NCCL_COMPRESSED_RS_TRACE") != nullptr;
  return enabled;
}

static int activeRingLimit() {
  static int limit = []() {
    const char* value = std::getenv("NCCL_COMPRESSED_RS_RINGS");
    if (value == nullptr || *value == '\0') return DefaultActiveRings;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (*end != '\0' || parsed < 1) return DefaultActiveRings;
    return static_cast<int>(std::min<long>(parsed, MAXCHANNELS));
  }();
  return limit;
}

static void trace(const Operation* op, const char* phase) {
  if (traceEnabled()) {
    std::fprintf(stderr, "compressed-rs rank=%d sequence=%llu phase=%s\n",
                 op->comm ? op->comm->rank : -1,
                 static_cast<unsigned long long>(op->sequence), phase);
    std::fflush(stderr);
  }
}
static void traceDestroy(const ncclComm* comm, const char* phase) {
  if (traceEnabled()) {
    std::fprintf(stderr, "compressed-rs rank=%d phase=destroy-%s\n",
                 comm ? comm->rank : -1, phase);
    std::fflush(stderr);
  }
}
static ncclResult_t wait(cudaStream_t s, cudaEvent_t e) {
  CUDACHECK(cudaStreamWaitEvent(s, e, 0));
  return ncclSuccess;
}

static void progressPause(unsigned* spins) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  asm volatile("" ::: "memory");
#endif
  // IPC phase changes normally arrive in a few microseconds. Yielding on every
  // miss turns that latency into an OS scheduling quantum and leaves visible
  // holes between communication kernels.
  if (++*spins == 16384) {
    *spins = 0;
    std::this_thread::yield();
  }
}

static char traceEventState(cudaEvent_t event) {
  cudaError_t status = CUDACLEARERROR(cudaEventQuery(event));
  return status == cudaSuccess ? 'C' : status == cudaErrorNotReady ? 'P' : 'E';
}

static ncclResult_t waitHostPhase(ncclComm* comm, uint64_t* address, uint64_t phase) {
  unsigned spins = 0;
  while (__atomic_load_n(address, __ATOMIC_ACQUIRE) < phase) {
    if (comm->abortFlag != nullptr && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0)
      return ncclSystemError;
    if ((++spins & 0xfff) == 0) std::this_thread::yield();
  }
  return ncclSuccess;
}

static int eventGeneration(uint64_t phase) {
  return static_cast<int>((phase - 1) % NCCL_COMPRESSED_EVENT_GENERATIONS);
}

static uint64_t eventGenerationPhase(uint64_t phase) {
  return (phase - 1) / NCCL_COMPRESSED_EVENT_GENERATIONS + 1;
}

static bool canRecordIpcWindow(
    const ncclCompressedRsEventWindowSync* sync, uint64_t phase) {
  uint64_t priorPhase = phase > NCCL_COMPRESSED_EVENT_GENERATIONS
      ? phase - NCCL_COMPRESSED_EVENT_GENERATIONS : 0;
  return __atomic_load_n(&sync->waitIssuedTail, __ATOMIC_ACQUIRE) >= priorPhase;
}

static bool canWaitIpcWindow(
    const ncclCompressedRsEventWindowSync* sync, uint64_t phase) {
  int generation = eventGeneration(phase);
  return __atomic_load_n(&sync->generation[generation].recordIssued,
                         __ATOMIC_ACQUIRE) >= eventGenerationPhase(phase);
}

static ncclResult_t tryRecordIpcWindow(
    cudaEvent_t* events, ncclCompressedRsEventWindowSync* sync,
    uint64_t* currentPhase, cudaStream_t stream, bool* submitted) {
  uint64_t phase = *currentPhase + 1;
  int generation = eventGeneration(phase);
  uint64_t generationPhase = eventGenerationPhase(phase);
  *submitted = false;
  if (!canRecordIpcWindow(sync, phase)) return ncclSuccess;
  CUDACHECK(cudaEventRecord(events[generation], stream));
  __atomic_store_n(&sync->generation[generation].recordIssued,
                   generationPhase, __ATOMIC_RELEASE);
  *currentPhase = phase;
  *submitted = true;
  return ncclSuccess;
}

static ncclResult_t tryWaitIpcWindow(
    cudaEvent_t* events, ncclCompressedRsEventWindowSync* sync,
    uint64_t* currentPhase, cudaStream_t stream, bool* submitted) {
  uint64_t phase = *currentPhase + 1;
  int generation = eventGeneration(phase);
  uint64_t generationPhase = eventGenerationPhase(phase);
  *submitted = false;
  if (__atomic_load_n(&sync->generation[generation].recordIssued,
                      __ATOMIC_ACQUIRE) < generationPhase)
    return ncclSuccess;
  CUDACHECK(cudaStreamWaitEvent(stream, events[generation], 0));
  *currentPhase = phase;
  *submitted = true;
  return ncclSuccess;
}

static void publishWindowWait(
    ncclCompressedRsEventWindowSync* sync, uint64_t phase,
    uint64_t* publishedPhase, bool force) {
  constexpr uint64_t PublishBatch = NCCL_COMPRESSED_EVENT_GENERATIONS;
  if (phase == *publishedPhase) return;
  if (!force && phase - *publishedPhase < PublishBatch) return;
  __atomic_store_n(&sync->waitIssuedTail, phase, __ATOMIC_RELEASE);
  *publishedPhase = phase;
}

static ncclResult_t validatePlan(const ncclCompressionPlan_t* p) {
  if (p == nullptr || p->size < sizeof(*p) || p->magic != NCCL_COMPRESSION_PLAN_MAGIC ||
      p->blockElements == 0 || p->maxCompressedBytes == nullptr || p->encode == nullptr ||
      p->decodeReduceEncode == nullptr || p->decodeReduceCopy == nullptr)
    return ncclInvalidArgument;
  return ncclSuccess;
}

static ncclResult_t claim(ncclComm* comm, Operation* op) {
  static std::mutex createMutex;
  {
    std::lock_guard<std::mutex> lock(createMutex);
    if (comm->compressedRsManager == nullptr) {
      comm->compressedRsManager = new (std::nothrow) ncclCompressedRsManager;
      if (comm->compressedRsManager == nullptr) return ncclSystemError;
    }
  }
  ncclCompressedRsManager* m = comm->compressedRsManager;
  op->comm = comm;
  op->manager = m;
  return ncclSuccess;
}

static ncclResult_t destroyManagerResources(ncclComm* comm, ncclCompressedRsManager* m) {
  ncclResult_t ret = ncclSuccess;
  int savedDev = -1;
  traceDestroy(comm, "begin");
  CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, done);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, done);
  if (m->tailRecorded) {
    traceDestroy(comm, "tail-begin");
    CUDACHECKGOTO(cudaEventSynchronize(m->tail), ret, done);
    traceDestroy(comm, "tail-end");
  }
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    if (ring.sendStream) CUDACHECKIGNORE(cudaStreamDestroy(ring.sendStream));
    ring.sendStream = nullptr;
  }
  // Cached remote pointers must be dropped before their imported pools. The
  // allocations they reference remain owned by the next rank until every
  // importer publishes importsClosed below.
  for (int c = 0; c < m->cachedRingCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    if (ring.cachedRemoteRx) CUDACHECKIGNORE(cudaFree(ring.cachedRemoteRx));
    ring.cachedRemoteRx = nullptr;
  }
  // Drop every imported CUDA object first. The next rank owns these objects,
  // so publish the Host-side acknowledgement before waiting for the previous
  // rank to drop the objects exported by this rank.
  traceDestroy(comm, "imported-events-close-begin");
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    for (int s = 0; s < NCCL_COMPRESSED_RX_DEPTH; s++) {
      for (int g = 0; g < NCCL_COMPRESSED_EVENT_GENERATIONS; g++) {
        if (ring.nextDataReady[s][g])
          CUDACHECKIGNORE(cudaEventDestroy(ring.nextDataReady[s][g]));
        if (ring.nextCanWrite[s][g])
          CUDACHECKIGNORE(cudaEventDestroy(ring.nextCanWrite[s][g]));
        ring.nextDataReady[s][g] = nullptr;
        ring.nextCanWrite[s][g] = nullptr;
      }
    }
  }
  traceDestroy(comm, "imported-events-close-end");
  traceDestroy(comm, "imported-pool-close-begin");
  for (int c = 0; c < m->ringCount; c++) {
    if (m->rings[c].nextPool && m->rings[c].ownsNextPool)
      CUDACHECKIGNORE(cudaMemPoolDestroy(m->rings[c].nextPool));
    m->rings[c].nextPool = nullptr;
    m->rings[c].ownsNextPool = false;
  }
  traceDestroy(comm, "imported-pool-close-end");
  if (m->initialized) {
    for (int c = 0; c < m->ringCount; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      if (ring.nextSync != nullptr)
        __atomic_store_n(&ring.nextSync->importsClosed[c], 1, __ATOMIC_RELEASE);
    }
  }
  traceDestroy(comm, "imports-closed");
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    if (ring.nextSyncHandle) (void)ncclOsShmClose(ring.nextSyncHandle);
    ring.nextSyncHandle = nullptr;
    ring.nextSync = nullptr;
  }
  if (m->initialized && m->localSync != nullptr) {
    traceDestroy(comm, "peer-imports-wait-begin");
    for (int c = 0; c < m->ringCount; c++)
      NCCLCHECKGOTO(waitHostPhase(comm, &m->localSync->importsClosed[c], 1), ret, done);
    traceDestroy(comm, "peer-imports-wait-end");
  }
  for (int c = 0; c < m->cachedRingCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    if (ring.cachedRx) CUDACHECKIGNORE(cudaFree(ring.cachedRx));
    ring.cachedRx = nullptr;
  }
  if (m->cachedPreencoded) CUDACHECKIGNORE(cudaFree(m->cachedPreencoded));
  if (m->cachedPreencodeTasks) CUDACHECKIGNORE(cudaFree(m->cachedPreencodeTasks));
  if (m->cachedPreencodeDestinations)
    CUDACHECKIGNORE(cudaFree(m->cachedPreencodeDestinations));
  m->cachedPreencoded = nullptr;
  m->cachedPreencodeTasks = nullptr;
  m->cachedPreencodeDestinations = nullptr;
  m->cachedPreencodeBlocks = 0;
  m->cachedStride = 0;
  m->cachedRingCount = 0;
done:
  traceDestroy(comm, "local-events-close-begin");
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    for (int s = 0; s < NCCL_COMPRESSED_RX_DEPTH; s++) {
      for (int g = 0; g < NCCL_COMPRESSED_EVENT_GENERATIONS; g++) {
        if (ring.nextDataReady[s][g])
          CUDACHECKIGNORE(cudaEventDestroy(ring.nextDataReady[s][g]));
        if (ring.nextCanWrite[s][g])
          CUDACHECKIGNORE(cudaEventDestroy(ring.nextCanWrite[s][g]));
        if (ring.localDataReady[s][g])
          CUDACHECKIGNORE(cudaEventDestroy(ring.localDataReady[s][g]));
        if (ring.localCanWrite[s][g])
          CUDACHECKIGNORE(cudaEventDestroy(ring.localCanWrite[s][g]));
        ring.nextDataReady[s][g] = nullptr;
        ring.nextCanWrite[s][g] = nullptr;
        ring.localDataReady[s][g] = nullptr;
        ring.localCanWrite[s][g] = nullptr;
      }
      ring.localDataReadyPhase[s] = 0;
      ring.localCanWritePhase[s] = 0;
      ring.nextDataReadyPhase[s] = 0;
      ring.nextCanWritePhase[s] = 0;
      ring.localDataReadyWaitPublished[s] = 0;
      ring.nextCanWriteWaitPublished[s] = 0;
    }
    if (ring.nextSyncHandle) (void)ncclOsShmClose(ring.nextSyncHandle);
    ring.nextSyncHandle = nullptr;
    ring.nextSync = nullptr;
    ring.nextPool = nullptr;
    ring.ownsNextPool = false;
  }
  traceDestroy(comm, "local-events-close-end");
  if (m->localSyncHandle) (void)ncclOsShmClose(m->localSyncHandle);
  m->localSyncHandle = nullptr;
  m->localSync = nullptr;
  traceDestroy(comm, "local-pool-close-begin");
  if (m->localPool) CUDACHECKIGNORE(cudaMemPoolDestroy(m->localPool));
  traceDestroy(comm, "local-pool-close-end");
  m->localPool = nullptr;
  m->ringCount = 0;
  m->cachedStride = 0;
  m->cachedRingCount = 0;
  m->cachedPreencodeBlocks = 0;
  m->tailRecorded = false;
  m->initialized = false;
  if (savedDev >= 0) CUDACHECKIGNORE(cudaSetDevice(savedDev));
  traceDestroy(comm, "end");
  return ret;
}

static ncclResult_t initializeManager(Operation* op) {
  ncclResult_t ret = ncclSuccess;
  ncclComm* comm = op->comm;
  ncclCompressedRsManager* m = op->manager;
  if (m->initialized) return ncclSuccess;
  int supported = 0;
  int ipcEventSupported = 0;
  ncclIpcSocket ipcSocket = {};
  ncclIpcFd exportedFd = NCCL_INVALID_IPC_FD;
  ncclIpcFd importedFd = NCCL_INVALID_IPC_FD;
  ManagerInitInfo localInfo = {};
  std::vector<ManagerInitInfo> allInfo(comm->nRanks);
  m->ringCount = comm->nChannels;
  if (m->ringCount <= 0 || m->ringCount > MAXCHANNELS) return ncclInternalError;

  CUDACHECKGOTO(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, comm->cudaDev), ret, fail);
  if (!supported) {
    WARN("ncclReduceScatterCompressed: CUDA memory pools are unsupported on device %d", comm->cudaDev);
    ret = ncclInvalidUsage;
    goto fail;
  }
  CUDACHECKGOTO(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolSupportedHandleTypes,
                                       comm->cudaDev), ret, fail);
  if ((supported & cudaMemHandleTypePosixFileDescriptor) == 0) {
    WARN("ncclReduceScatterCompressed: POSIX FD memory-pool IPC is unsupported on device %d", comm->cudaDev);
    ret = ncclInvalidUsage;
    goto fail;
  }
  CUDACHECKGOTO(cudaDeviceGetAttribute(&ipcEventSupported, cudaDevAttrIpcEventSupport,
                                       comm->cudaDev), ret, fail);
  if (!ipcEventSupported) {
    WARN("ncclReduceScatterCompressed: CUDA IPC events are unsupported on device %d", comm->cudaDev);
    ret = ncclInvalidUsage;
    goto fail;
  }
  {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = comm->cudaDev;
    CUDACHECKGOTO(cudaMemPoolCreate(&m->localPool, &props), ret, fail);
  }
  {
    std::vector<cudaMemAccessDesc> access;
    for (int c = 0; c < m->ringCount; c++) {
      int prev = comm->channels[c].ring.prev;
      m->rings[c].channel = c;
      m->rings[c].prev = prev;
      m->rings[c].next = comm->channels[c].ring.next;
      int dev = comm->peerInfo[prev].cudaDev;
      auto found = std::find_if(access.begin(), access.end(), [dev](const cudaMemAccessDesc& desc) {
        return desc.location.id == dev;
      });
      if (found == access.end()) {
        cudaMemAccessDesc desc = {};
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id = dev;
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        access.push_back(desc);
      }
    }
    CUDACHECKGOTO(cudaMemPoolSetAccess(m->localPool, access.data(), access.size()), ret, fail);
  }
  CUDACHECKGOTO(cudaEventCreateWithFlags(&m->tail, cudaEventDisableTiming), ret, fail);
  NCCLCHECKGOTO(ncclOsShmOpen(localInfo.syncPath, sizeof(localInfo.syncPath),
      sizeof(*m->localSync), reinterpret_cast<void**>(&m->localSync), nullptr,
      m->ringCount + 1, &m->localSyncHandle), ret, fail);
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    __atomic_store_n(&m->localSync->importsClosed[c], 0, __ATOMIC_RELAXED);
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&ring.sendStream, cudaStreamNonBlocking), ret, fail);
    for (int s = 0; s < NCCL_COMPRESSED_RX_DEPTH; s++) {
      for (int g = 0; g < NCCL_COMPRESSED_EVENT_GENERATIONS; g++) {
        CUDACHECKGOTO(cudaEventCreateWithFlags(&ring.localDataReady[s][g],
            cudaEventDisableTiming | cudaEventInterprocess), ret, fail);
        CUDACHECKGOTO(cudaEventCreateWithFlags(&ring.localCanWrite[s][g],
            cudaEventDisableTiming | cudaEventInterprocess), ret, fail);
        CUDACHECKGOTO(cudaIpcGetEventHandle(&localInfo.dataReady[c][s][g],
                                            ring.localDataReady[s][g]), ret, fail);
        CUDACHECKGOTO(cudaIpcGetEventHandle(&localInfo.canWrite[c][s][g],
                                            ring.localCanWrite[s][g]), ret, fail);
      }
    }
  }
  CUDACHECKGOTO(cudaMemPoolExportToShareableHandle(&exportedFd, m->localPool,
                                                   cudaMemHandleTypePosixFileDescriptor, 0), ret, fail);
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    uint64_t socketHash = (comm->commHash ^ IpcSocketHashSalt) + static_cast<uint64_t>(c);
    NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSocket, comm->rank, socketHash, comm->abortFlag), ret, fail);
    NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks,
                                   0x43525000 + c), ret, fail);
    NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSocket, exportedFd, ring.prev, socketHash), ret, fail);
    NCCLCHECKGOTO(ncclIpcSocketRecvFd(&ipcSocket, &importedFd), ret, fail);
    for (int p = 0; p < c; p++)
      if (m->rings[p].next == ring.next) ring.nextPool = m->rings[p].nextPool;
    if (ring.nextPool == nullptr) {
      CUDACHECKGOTO(cudaMemPoolImportFromShareableHandle(&ring.nextPool,
          reinterpret_cast<void*>(static_cast<intptr_t>(importedFd)),
          cudaMemHandleTypePosixFileDescriptor, 0), ret, fail);
      ring.ownsNextPool = true;
      cudaMemAccessDesc access = {};
      access.location.type = cudaMemLocationTypeDevice;
      access.location.id = comm->cudaDev;
      access.flags = cudaMemAccessFlagsProtReadWrite;
      CUDACHECKGOTO(cudaMemPoolSetAccess(ring.nextPool, &access, 1), ret, fail);
    }
    NCCLCHECKGOTO(ncclIpcSocketClose(&ipcSocket), ret, fail);
    ipcSocket.fd = NCCL_INVALID_SOCKET;
    NCCLCHECKGOTO(ncclIpcFdClose(importedFd) == 0 ? ncclSuccess : ncclSystemError, ret, fail);
    importedFd = NCCL_INVALID_IPC_FD;
  }
  NCCLCHECKGOTO(ncclIpcFdClose(exportedFd) == 0 ? ncclSuccess : ncclSystemError, ret, fail);
  exportedFd = NCCL_INVALID_IPC_FD;

  allInfo[comm->rank] = localInfo;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allInfo.data(), sizeof(localInfo)), ret, fail);
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    NCCLCHECKGOTO(ncclOsShmOpen(allInfo[ring.next].syncPath,
        sizeof(allInfo[ring.next].syncPath), sizeof(*ring.nextSync),
        reinterpret_cast<void**>(&ring.nextSync), nullptr, 0,
        &ring.nextSyncHandle), ret, fail);
    for (int s = 0; s < NCCL_COMPRESSED_RX_DEPTH; s++) {
      for (int g = 0; g < NCCL_COMPRESSED_EVENT_GENERATIONS; g++) {
        CUDACHECKGOTO(cudaIpcOpenEventHandle(&ring.nextDataReady[s][g],
            allInfo[ring.next].dataReady[c][s][g]), ret, fail);
        CUDACHECKGOTO(cudaIpcOpenEventHandle(&ring.nextCanWrite[s][g],
            allInfo[ring.next].canWrite[c][s][g]), ret, fail);
      }
    }
  }
  m->initialized = true;
  trace(op, "connection-ready");
fail:
  if (ipcSocket.fd != NCCL_INVALID_SOCKET) (void)ncclIpcSocketClose(&ipcSocket);
  if (exportedFd != NCCL_INVALID_IPC_FD) (void)ncclIpcFdClose(exportedFd);
  if (importedFd != NCCL_INVALID_IPC_FD) (void)ncclIpcFdClose(importedFd);
  if (ret != ncclSuccess) (void)destroyManagerResources(comm, m);
  return ret;
}

static ncclResult_t ensureCachedResources(
    Operation* op, int activeRings, size_t requiredStride, size_t requiredBlocks,
    cudaStream_t stream, bool* rebuilt) {
  ncclResult_t ret = ncclSuccess;
  ncclComm* comm = op->comm;
  ncclCompressedRsManager* m = op->manager;
  bool rebuildRx = m->cachedRingCount < activeRings ||
      m->cachedStride < requiredStride;
  size_t oldStride = m->cachedStride;
  *rebuilt = rebuildRx;

  if (rebuildRx) {
    int newRingCount = std::max(m->cachedRingCount, activeRings);
    size_t newStride = std::max(m->cachedStride, requiredStride);
    if (newStride > SIZE_MAX / NCCL_COMPRESSED_RX_DEPTH) return ncclInvalidArgument;
    size_t arenaBytes = newStride * NCCL_COMPRESSED_RX_DEPTH;

    if (m->cachedRingCount != 0) {
      // Pointer replacement is rare (capacity/ring growth). Make it a
      // collective generation transition: all ranks finish the old DAG,
      // release imports, then owners release and replace their allocations.
      if (m->tailRecorded) CUDACHECKGOTO(cudaEventSynchronize(m->tail), ret, fail);
      NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks,
          0x43525210), ret, fail);
      for (int c = 0; c < m->cachedRingCount; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        if (ring.cachedRemoteRx) CUDACHECKGOTO(cudaFree(ring.cachedRemoteRx), ret, fail);
        ring.cachedRemoteRx = nullptr;
      }
      NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks,
          0x43525211), ret, fail);
      for (int c = 0; c < m->cachedRingCount; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        if (ring.cachedRx) CUDACHECKGOTO(cudaFree(ring.cachedRx), ret, fail);
        ring.cachedRx = nullptr;
      }
    }

    std::vector<cudaMemPoolPtrExportData> allRxData(
        static_cast<size_t>(comm->nRanks) * newRingCount);
    for (int c = 0; c < newRingCount; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      CUDACHECKGOTO(cudaMallocFromPoolAsync(
          &ring.cachedRx, arenaBytes, m->localPool, stream), ret, fail);
      CUDACHECKGOTO(cudaMemsetAsync(ring.cachedRx, 0, arenaBytes, stream), ret, fail);
    }
    // Exporting a pool allocation and publishing it to another process must
    // happen after the allocation command has completed.
    CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail);
    for (int c = 0; c < newRingCount; c++) {
      CUDACHECKGOTO(cudaMemPoolExportPointer(
          &allRxData[static_cast<size_t>(comm->rank) * newRingCount + c],
          m->rings[c].cachedRx), ret, fail);
    }
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allRxData.data(),
        sizeof(cudaMemPoolPtrExportData) * newRingCount), ret, fail);
    for (int c = 0; c < newRingCount; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      CUDACHECKGOTO(cudaMemPoolImportPointer(&ring.cachedRemoteRx, ring.nextPool,
          &allRxData[static_cast<size_t>(ring.next) * newRingCount + c]), ret, fail);
    }
    m->cachedStride = newStride;
    m->cachedRingCount = newRingCount;
  }

  if (m->cachedPreencodeBlocks < requiredBlocks || oldStride != m->cachedStride) {
    *rebuilt = true;
    if (m->cachedStride > SIZE_MAX / requiredBlocks ||
        sizeof(*m->cachedPreencodeTasks) > SIZE_MAX / requiredBlocks)
      return ncclInvalidArgument;
    if (m->tailRecorded) NCCLCHECKGOTO(wait(stream, m->tail), ret, fail);
    if (m->cachedPreencoded)
      CUDACHECKGOTO(cudaFreeAsync(m->cachedPreencoded, stream), ret, fail);
    if (m->cachedPreencodeTasks)
      CUDACHECKGOTO(cudaFreeAsync(m->cachedPreencodeTasks, stream), ret, fail);
    if (m->cachedPreencodeDestinations)
      CUDACHECKGOTO(cudaFreeAsync(m->cachedPreencodeDestinations, stream), ret, fail);
    CUDACHECKGOTO(cudaMallocFromPoolAsync(&m->cachedPreencoded,
        m->cachedStride * requiredBlocks, m->localPool, stream), ret, fail);
    CUDACHECKGOTO(cudaMallocFromPoolAsync(
        reinterpret_cast<void**>(&m->cachedPreencodeTasks),
        sizeof(*m->cachedPreencodeTasks) * requiredBlocks,
        m->localPool, stream), ret, fail);
    CUDACHECKGOTO(cudaMallocFromPoolAsync(
        reinterpret_cast<void**>(&m->cachedPreencodeDestinations),
        sizeof(*m->cachedPreencodeDestinations) * requiredBlocks,
        m->localPool, stream), ret, fail);
    m->cachedPreencodeBlocks = requiredBlocks;
  }

  op->stride = m->cachedStride;
  op->preencoded = m->cachedPreencoded;
  op->preencodeTasks = m->cachedPreencodeTasks;
  op->preencodeDestinations = m->cachedPreencodeDestinations;
  for (int c = 0; c < activeRings; c++) {
    op->rings[c].rx = m->rings[c].cachedRx;
    op->rings[c].remoteRx = m->rings[c].cachedRemoteRx;
  }
fail:
  return ret;
}

static ncclResult_t launch(Job* j, Operation* op, size_t capacity, int typeSize) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t cs = nullptr;
  cudaEvent_t prepareDone = nullptr;
  cudaEvent_t preencodeDone = nullptr;
  RingLaunchState ringState[MAXCHANNELS];
  bool acquired = false;
  bool resourcesRebuilt = false;
  ncclComm* comm = j->comm;
  ncclCompressedRsManager* m = op->manager;
  int ctasPerSegment = std::max(1, std::min<int>(MaxSliceCtas,
      static_cast<int>((capacity + TargetSliceBytes - 1) / TargetSliceBytes)));
  size_t requiredStride = op->stride;
  if (requiredStride > SIZE_MAX / NCCL_COMPRESSED_RX_DEPTH)
    return ncclInvalidArgument;
  size_t nblocks = 1 + (j->recvcount - 1) / j->plan.blockElements;
  int activeRings = static_cast<int>(std::min<size_t>(nblocks,
      static_cast<size_t>(std::min(m->ringCount, activeRingLimit()))));
  std::vector<int> destinations(nblocks);
  if (traceEnabled()) {
    std::fprintf(stderr,
        "compressed-rs rank=%d sequence=%llu phase=enqueue-config "
        "rings=%d blocks=%zu eventWindow=%d earlySend=1 resourceCache=1\n",
        comm->rank, static_cast<unsigned long long>(op->sequence), activeRings,
        nblocks, NCCL_COMPRESSED_EVENT_GENERATIONS);
    std::fflush(stderr);
  }
  trace(op, "enqueue-begin");
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream, false, &cs), ret, fail);
  acquired = true;
  CUDACHECKGOTO(cudaEventCreateWithFlags(&prepareDone, cudaEventDisableTiming), ret, fail);
  CUDACHECKGOTO(cudaEventCreateWithFlags(&preencodeDone, cudaEventDisableTiming), ret, fail);
  for (int c = 0; c < activeRings; c++) {
    CUDACHECKGOTO(cudaEventCreateWithFlags(&ringState[c].finish, cudaEventDisableTiming), ret, fail);
  }
  {
    size_t preencodeBlocks = nblocks;
    if (requiredStride > SIZE_MAX / preencodeBlocks ||
        sizeof(*op->preencodeTasks) > SIZE_MAX / preencodeBlocks) {
      ret = ncclInvalidArgument;
      goto fail;
    }
    if (m->tailRecorded) NCCLCHECKGOTO(wait(cs, m->tail), ret, fail);
    NCCLCHECKGOTO(ensureCachedResources(op, activeRings, requiredStride,
        preencodeBlocks, cs, &resourcesRebuilt), ret, fail);
    if (traceEnabled()) {
      std::fprintf(stderr,
          "compressed-rs rank=%d sequence=%llu phase=resource-cache %s "
          "rings=%d stride=%zu blocks=%zu\n",
          comm->rank, static_cast<unsigned long long>(op->sequence),
          resourcesRebuilt ? "rebuilt" : "reused", m->cachedRingCount,
          m->cachedStride, m->cachedPreencodeBlocks);
      std::fflush(stderr);
    }
    for (size_t b = 0; b < nblocks; b++) {
      int c = static_cast<int>(b % activeRings);
      destinations[b] = comm->channels[c].ring.userRanks[comm->nRanks-1];
    }
    CUDACHECKGOTO(cudaMemcpyAsync(op->preencodeDestinations, destinations.data(),
        sizeof(*op->preencodeDestinations) * nblocks, cudaMemcpyHostToDevice, cs), ret, fail);
    for (int c = 0; c < activeRings; c++) {
      RingLaunchState& state = ringState[c];
      ncclCompressedRsRing& ring = m->rings[c];
      state.blockCount = (nblocks + activeRings - 1 - c) / activeRings;
      for (int lane = 0; lane < NCCL_COMPRESSED_RX_DEPTH; lane++)
        state.localCreditPending[lane] = !ring.localCreditInitialized[lane];
      state.batchCount = static_cast<int>(std::min<size_t>(
          NCCL_COMPRESSED_RX_DEPTH - 1, state.blockCount));
      for (int token = 0; token < state.batchCount; token++) {
        state.blocks[token] = c + static_cast<size_t>(token) * activeRings;
        state.baseLanes[token] = token;
        state.initialPending[token] = true;
      }
      state.nextBlockOrdinal = state.batchCount;
      state.token = state.batchCount - 1;
    }
    NCCLCHECKGOTO(ncclLaunchCompressedPrepareEncodeAll(op->preencodeTasks,
        op->preencoded, op->stride, j->sendbuff, j->recvcount,
        j->plan.blockElements, nblocks, op->preencodeDestinations,
        typeSize, capacity, cs),
        ret, fail);
    NCCLCHECKGOTO(record(prepareDone, cs), ret, fail);
    NCCLCHECKGOTO(wait(j->stream, prepareDone), ret, fail);
    NCCLCHECKGOTO(j->plan.encode(op->preencodeTasks, preencodeBlocks,
        j->datatype, j->plan.context, j->stream), ret, fail);
    NCCLCHECKGOTO(record(preencodeDone, j->stream), ret, fail);
    for (int c = 0; c < activeRings; c++) {
      NCCLCHECKGOTO(wait(m->rings[c].sendStream, preencodeDone), ret, fail);
    }

    // Build a bounded asynchronous DAG. Host counters only advertise that an
    // Event record/wait has been submitted; CUDA Events carry the actual GPU
    // dependency. No stage waits for GPU completion on the Host.
    auto nextStallTrace = std::chrono::steady_clock::now() +
        std::chrono::seconds(2);
    unsigned pipelineSpins = 0;
    for (;;) {
      bool done = true;
      bool progressed = false;
      for (int c = 0; c < activeRings; c++) {
        RingLaunchState& state = ringState[c];
        if (state.completedBlocks != state.blockCount ||
            state.initialSendsSubmitted != state.blockCount)
          done = false;
      }
      if (done) break;

      // Seed every local RX slot with one credit. The internal orchestration
      // stream is used here only to avoid delaying credits behind pre-encode.
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        for (int lane = 0; lane < NCCL_COMPRESSED_RX_DEPTH; lane++) {
          if (!state.localCreditPending[lane]) continue;
          bool submitted = false;
          NCCLCHECKGOTO(tryRecordIpcWindow(ring.localCanWrite[lane],
              &m->localSync->rx[c][lane].canWrite,
              &ring.localCanWritePhase[lane], cs, &submitted), ret, fail);
          if (submitted) {
            state.localCreditPending[lane] = false;
            ring.localCreditInitialized[lane] = true;
            progressed = true;
          }
        }
      }

      // Only the initial, locally encoded frame uses the send stream. Every
      // reduced intermediate frame is written directly into the next rank's
      // RX lane by decodeReduceEncode on the compute stream.
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        OperationRing& ringOp = op->rings[c];
        for (int queue = 0; queue < 2; queue++) {
          int count = queue == 0 ? state.batchCount : state.nextBatchCount;
          size_t* blocks = queue == 0 ? state.blocks : state.nextBlocks;
          int* baseLanes = queue == 0 ? state.baseLanes : state.nextBaseLanes;
          bool* initialPending = queue == 0
              ? state.initialPending : state.nextInitialPending;
          for (int token = count - 1; token >= 0; token--) {
            if (!initialPending[token]) continue;
            int lane = baseLanes[token];
            ncclCompressedRsEventWindowSync* credit =
                &ring.nextSync->rx[c][lane].canWrite;
            ncclCompressedRsEventWindowSync* ready =
                &ring.nextSync->rx[c][lane].dataReady;
            if (!canWaitIpcWindow(credit, ring.nextCanWritePhase[lane] + 1) ||
                !canRecordIpcWindow(ready, ring.nextDataReadyPhase[lane] + 1))
              continue;
            bool submitted = false;
            NCCLCHECKGOTO(tryWaitIpcWindow(ring.nextCanWrite[lane], credit,
                &ring.nextCanWritePhase[lane], ring.sendStream, &submitted), ret, fail);
            if (!submitted) { ret = ncclInternalError; goto fail; }
            publishWindowWait(credit, ring.nextCanWritePhase[lane],
                &ring.nextCanWriteWaitPublished[lane], false);
            void* source = slot(op->preencoded, op->stride, blocks[token]);
            NCCLCHECKGOTO(ncclLaunchCompressedSend(source,
                slot(ringOp.remoteRx, op->stride, lane), op->stride, capacity,
                1, ctasPerSegment, ring.sendStream), ret, fail);
            NCCLCHECKGOTO(tryRecordIpcWindow(ring.nextDataReady[lane], ready,
                &ring.nextDataReadyPhase[lane], ring.sendStream, &submitted), ret, fail);
            if (!submitted) { ret = ncclInternalError; goto fail; }
            initialPending[token] = false;
            state.initialSendsSubmitted++;
            // With two ranks the initial send is immediately followed by the
            // final reduction. Once this send is submitted, the lane's next
            // credit can only be produced by that final-copy, so the next
            // cohort may already enqueue its wait/send DAG.
            if (queue == 0 && comm->nRanks == 2 &&
                state.nextBlockOrdinal < state.blockCount &&
                state.nextBatchCount < NCCL_COMPRESSED_RX_DEPTH - 1) {
              int nextToken = state.nextBatchCount++;
              state.nextBlocks[nextToken] =
                  c + state.nextBlockOrdinal * activeRings;
              state.nextBaseLanes[nextToken] = lane;
              state.nextInitialPending[nextToken] = true;
              state.nextBlockOrdinal++;
            }
            progressed = true;
          }
        }
      }

      // Receiving and reducing a cohort does not imply that this rank has
      // submitted every initial send belonging to it. Early-send makes those
      // two progress paths independent. Keep the physical cohort intact until
      // all of its initial sends have entered the send stream; otherwise a
      // later cohort can reuse the token/lane mapping while an older send is
      // still pending and different ranks can associate the same lane phase
      // with different blocks.
      for (int c = 0; c < activeRings; c++) {
        RingLaunchState& state = ringState[c];
        if (!state.cohortComplete) continue;
        bool sendsSubmitted = true;
        for (int token = 0; token < state.batchCount; token++)
          sendsSubmitted = sendsSubmitted && !state.initialPending[token];
        if (!sendsSubmitted) continue;
        if (state.completedBlocks == state.blockCount) {
          state.cohortComplete = false;
          continue;
        }
        if (state.nextBatchCount == 0) continue;
        state.batchCount = state.nextBatchCount;
        // Tokens are consumed from batchCount-1 down to zero. Reverse the
        // rolling cohort so consumption follows the order in which the prior
        // cohort released its lanes.
        for (int token = 0; token < state.batchCount; token++) {
          int nextToken = state.batchCount - 1 - token;
          state.blocks[token] = state.nextBlocks[nextToken];
          state.baseLanes[token] = state.nextBaseLanes[nextToken];
          state.initialPending[token] =
              state.nextInitialPending[nextToken];
          state.nextInitialPending[nextToken] = false;
        }
        state.nextBatchCount = 0;
        state.round = 0;
        state.token = state.batchCount - 1;
        state.cohortComplete = false;
        progressed = true;
      }

      // A single compute stream scans all ready RX lanes. For intermediate
      // rounds DRE waits for downstream credit, then writes the frame directly
      // into the peer RX lane and publishes dataReady from the same stream.
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        OperationRing& ringOp = op->rings[c];
        {
          if (state.token < 0) continue;
          int token = state.token;
          int lane = (state.baseLanes[token] + state.round) %
              NCCL_COMPRESSED_RX_DEPTH;
          ncclCompressedRsEventWindowSync* ready = &m->localSync->rx[c][lane].dataReady;
          ncclCompressedRsEventWindowSync* credit = &m->localSync->rx[c][lane].canWrite;
          if (!canWaitIpcWindow(ready, ring.localDataReadyPhase[lane] + 1) ||
              !canRecordIpcWindow(credit, ring.localCanWritePhase[lane] + 1))
            continue;
          size_t block = state.blocks[token];
          uint32_t round = state.round;
          bool final = round + 1 == static_cast<uint32_t>(comm->nRanks - 1);
          int remoteLane = (state.baseLanes[token] + round + 1) %
              NCCL_COMPRESSED_RX_DEPTH;
          if (!final) {
            ncclCompressedRsRxSync* next = &ring.nextSync->rx[c][remoteLane];
            if (!canWaitIpcWindow(&next->canWrite,
                                 ring.nextCanWritePhase[remoteLane] + 1) ||
                !canRecordIpcWindow(&next->dataReady,
                                   ring.nextDataReadyPhase[remoteLane] + 1))
              continue;
          }
          bool submitted = false;
          NCCLCHECKGOTO(tryWaitIpcWindow(ring.localDataReady[lane], ready,
              &ring.localDataReadyPhase[lane], j->stream, &submitted), ret, fail);
          if (!submitted) { ret = ncclInternalError; goto fail; }
          publishWindowWait(ready, ring.localDataReadyPhase[lane],
              &ring.localDataReadyWaitPublished[lane], false);
          if (!final) {
            ncclCompressedRsRxSync* next = &ring.nextSync->rx[c][remoteLane];
            NCCLCHECKGOTO(tryWaitIpcWindow(ring.nextCanWrite[remoteLane],
                &next->canWrite, &ring.nextCanWritePhase[remoteLane],
                j->stream, &submitted), ret, fail);
            if (!submitted) { ret = ncclInternalError; goto fail; }
            publishWindowWait(&next->canWrite,
                ring.nextCanWritePhase[remoteLane],
                &ring.nextCanWriteWaitPublished[remoteLane], false);
          }
          int dst = comm->channels[c].ring.userRanks[comm->nRanks - 2 - round];
          size_t off = block * j->plan.blockElements;
          size_t count = std::min(j->plan.blockElements, j->recvcount - off);
          void* rxSlot = slot(ringOp.rx, op->stride, lane);
          void* remoteRxSlot = final ? nullptr
              : slot(ringOp.remoteRx, op->stride, remoteLane);
          size_t inputElement = static_cast<size_t>(dst) * j->recvcount + off;
          ncclCompressionTask_t task = {};
          task.compressedInput = payload(rxSlot);
          task.inputHeader = header(rxSlot);
          task.compressedOutput = remoteRxSlot == nullptr ? nullptr : payload(remoteRxSlot);
          task.outputHeader = remoteRxSlot == nullptr ? nullptr : header(remoteRxSlot);
          task.localInput = static_cast<const char*>(j->sendbuff) + inputElement * typeSize;
          task.output = final ? static_cast<char*>(j->recvbuff) + off * typeSize : nullptr;
          task.elementCount = count;
          task.compressedCapacity = capacity;
          if (final)
            NCCLCHECKGOTO(j->plan.decodeReduceCopy(task,
                j->datatype, j->op, j->plan.context, j->stream), ret, fail);
          else
            NCCLCHECKGOTO(j->plan.decodeReduceEncode(task,
                j->datatype, j->op, j->plan.context, j->stream), ret, fail);
          NCCLCHECKGOTO(tryRecordIpcWindow(ring.localCanWrite[lane], credit,
              &ring.localCanWritePhase[lane], j->stream, &submitted), ret, fail);
          if (!submitted) { ret = ncclInternalError; goto fail; }
          if (!final) {
            NCCLCHECKGOTO(tryRecordIpcWindow(ring.nextDataReady[remoteLane],
                &ring.nextSync->rx[c][remoteLane].dataReady,
                &ring.nextDataReadyPhase[remoteLane], j->stream, &submitted), ret, fail);
            if (!submitted) { ret = ncclInternalError; goto fail; }
            // The last DRE writes the exact lane consumed by final-copy. Its
            // next credit therefore belongs to final-copy, not to another DRE.
            // Publish the following initial send now; the send stream waits on
            // that credit and can start as soon as the peer releases the lane,
            // while this rank is still running the rest of its final copies.
            if (round + 2 == static_cast<uint32_t>(comm->nRanks - 1) &&
                state.nextBlockOrdinal < state.blockCount &&
                state.nextBatchCount < NCCL_COMPRESSED_RX_DEPTH - 1) {
              int nextToken = state.nextBatchCount++;
              state.nextBlocks[nextToken] =
                  c + state.nextBlockOrdinal * activeRings;
              state.nextBaseLanes[nextToken] = remoteLane;
              state.nextInitialPending[nextToken] = true;
              state.nextBlockOrdinal++;
            }
            if (state.token > 0) {
              state.token--;
            } else {
              state.round++;
              state.token = state.batchCount - 1;
            }
          } else {
            state.completedBlocks++;
            if (state.token > 0) {
              state.token--;
            } else {
              state.token = -1;
              state.cohortComplete = true;
            }
          }
          progressed = true;
        }
      }
      if (progressed) pipelineSpins = 0;
      if (!progressed) {
        if (traceEnabled() && std::chrono::steady_clock::now() >= nextStallTrace) {
          std::fprintf(stderr,
              "compressed-rs rank=%d sequence=%llu phase=pipeline-stall "
              "prepare=%c preencode=%c\n",
              comm->rank, static_cast<unsigned long long>(op->sequence),
              traceEventState(prepareDone), traceEventState(preencodeDone));
          for (int c = 0; c < activeRings; c++) {
            RingLaunchState& state = ringState[c];
            ncclCompressedRsRing& ring = m->rings[c];
            int pending = 0;
            for (int token = 0; token < state.batchCount; token++)
              pending += state.initialPending[token] ? 1 : 0;
            for (int token = 0; token < state.nextBatchCount; token++)
              pending += state.nextInitialPending[token] ? 1 : 0;
            std::fprintf(stderr,
                "  ring=%d completed=%zu/%zu batch=%d nextBatch=%d "
                "token=%d round=%u pending=%d\n",
                c, state.completedBlocks, state.blockCount, state.batchCount,
                state.nextBatchCount, state.token, state.round, pending);
            for (int lane = 0; lane < NCCL_COMPRESSED_RX_DEPTH; lane++) {
              ncclCompressedRsRxSync* local = &m->localSync->rx[c][lane];
              ncclCompressedRsRxSync* next = &ring.nextSync->rx[c][lane];
              std::fprintf(stderr,
                  "    lane=%d localReady=%llu/%c localCredit=%llu/%c(%c) "
                  "nextReady=%llu/%c nextCredit=%llu/%c(%c)\n",
                  lane,
                  static_cast<unsigned long long>(ring.localDataReadyPhase[lane]),
                  canWaitIpcWindow(&local->dataReady,
                      ring.localDataReadyPhase[lane] + 1) ? 'R' : 'P',
                  static_cast<unsigned long long>(ring.localCanWritePhase[lane]),
                  canRecordIpcWindow(&local->canWrite,
                      ring.localCanWritePhase[lane] + 1) ? 'R' : 'P',
                  traceEventState(ring.localCanWrite[lane][eventGeneration(
                      ring.localCanWritePhase[lane] + 1)]),
                  static_cast<unsigned long long>(ring.nextDataReadyPhase[lane]),
                  canRecordIpcWindow(&next->dataReady,
                      ring.nextDataReadyPhase[lane] + 1) ? 'R' : 'P',
                  static_cast<unsigned long long>(ring.nextCanWritePhase[lane]),
                  canWaitIpcWindow(&next->canWrite,
                      ring.nextCanWritePhase[lane] + 1) ? 'R' : 'P',
                  traceEventState(ring.nextCanWrite[lane][eventGeneration(
                      ring.nextCanWritePhase[lane] + 1)]));
            }
          }
          std::fflush(stderr);
          nextStallTrace = std::chrono::steady_clock::now() +
              std::chrono::seconds(2);
        }
        if (comm->abortFlag != nullptr &&
            __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) {
          ret = ncclSystemError;
          goto fail;
        }
        progressPause(&pipelineSpins);
      }
    }

    // Publish any partial acknowledgement batch at the operation boundary.
    // Ready/credit records were already published immediately;
    // these cumulative tails only release Event generations for later reuse.
    for (int c = 0; c < activeRings; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      for (int lane = 0; lane < NCCL_COMPRESSED_RX_DEPTH; lane++) {
        publishWindowWait(&m->localSync->rx[c][lane].dataReady,
            ring.localDataReadyPhase[lane],
            &ring.localDataReadyWaitPublished[lane], true);
        publishWindowWait(&ring.nextSync->rx[c][lane].canWrite,
            ring.nextCanWritePhase[lane],
            &ring.nextCanWriteWaitPublished[lane], true);
      }
    }

    // Cached mappings remain valid across operations. The operation boundary
    // only has to join the send streams into the user stream; final credits
    // remain outstanding and become the initial credits of the next call.
    for (int c = 0; c < activeRings; c++) {
      NCCLCHECKGOTO(record(ringState[c].finish, m->rings[c].sendStream), ret, fail);
      NCCLCHECKGOTO(wait(j->stream, ringState[c].finish), ret, fail);
    }
    NCCLCHECKGOTO(record(m->tail, j->stream), ret, fail);
    m->tailRecorded = true;
  }
fail:
  if (acquired) {
    ncclResult_t x = ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream, false);
    if (ret == ncclSuccess) ret = x;
  }
  if (prepareDone) CUDACHECKIGNORE(cudaEventDestroy(prepareDone));
  if (preencodeDone) CUDACHECKIGNORE(cudaEventDestroy(preencodeDone));
  for (int c = 0; c < activeRings; c++) {
    if (ringState[c].finish) CUDACHECKIGNORE(cudaEventDestroy(ringState[c].finish));
  }
  trace(op, ret == ncclSuccess ? "enqueue-end" : "enqueue-error");
  return ret;
}

static ncclResult_t jobMain(ncclAsyncJob* base) {
  Job* j=reinterpret_cast<Job*>(base);
  ncclResult_t ret=ncclSuccess;
  Operation* op=new(std::nothrow) Operation;
  size_t capacity=0;
  if(!op) return ncclSystemError;
  NCCLCHECKGOTO(claim(j->comm,op),ret,fail);
  {
    std::unique_lock<std::mutex> lock(op->manager->mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
      WARN("ncclReduceScatterCompressed: concurrent submissions from multiple host threads are unsupported");
      ret = ncclInvalidUsage;
      goto fail;
    }
    op->sequence = ++op->manager->sequence;
    NCCLCHECKGOTO(j->plan.maxCompressedBytes(j->plan.blockElements,j->datatype,
      j->plan.context,&capacity),ret,fail);
    if(capacity==0 || capacity>UINT32_MAX || capacity>SIZE_MAX-PayloadOffset-SlotAlignment) {
      ret=ncclInvalidArgument; goto fail;
    }
    op->stride = alignUp(PayloadOffset + capacity, SlotAlignment);
    if (op->stride > SIZE_MAX / NCCL_COMPRESSED_RX_DEPTH) {
      ret=ncclInvalidArgument; goto fail;
    }
    NCCLCHECKGOTO(initializeManager(op),ret,fail);
    NCCLCHECKGOTO(launch(j,op,capacity,ncclTypeSize(j->datatype)),ret,fail);
  }
  delete op;
  return ncclSuccess;
fail:
  delete op;
  return ret;
}
static void jobFree(void* p) { delete reinterpret_cast<Job*>(p); }
} // namespace

ncclResult_t ncclCompressedRsDestroy(ncclComm* comm) {
  ncclCompressedRsManager* m=comm->compressedRsManager;
  if(!m) return ncclSuccess;
  ncclResult_t ret;
  {
    std::lock_guard<std::mutex> lock(m->mutex);
    ret=destroyManagerResources(comm,m);
    if(m->tail) CUDACHECKIGNORE(cudaEventDestroy(m->tail));
  }
  delete m;
  comm->compressedRsManager=nullptr;
  return ret;
}

NCCL_API(ncclResult_t,ncclReduceScatterCompressed,const void*,void*,size_t,
         ncclDataType_t,ncclRedOp_t,ncclComm_t,const ncclCompressionPlan_t*,cudaStream_t);
ncclResult_t ncclReduceScatterCompressed(const void* sendbuff,void* recvbuff,size_t recvcount,
    ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,
    const ncclCompressionPlan_t* plan,cudaStream_t stream) {
  ncclResult_t ret=ncclSuccess;
  ncclCudaGraph graph={};
  Job* job=nullptr;
  int ts=0;
  NCCLCHECKGOTO(CommCheck(comm,"ncclReduceScatterCompressed","comm"),ret,fail);
  NCCLCHECKGOTO(validatePlan(plan),ret,fail);
  ts=ncclTypeSize(datatype);
  if(!sendbuff || !recvbuff || !recvcount || op!=ncclSum || ts<=0) return ncclInvalidArgument;
  if(comm->nNodes!=1 || comm->localRanks!=comm->nRanks) return ncclInvalidUsage;
  if(recvcount>SIZE_MAX/static_cast<size_t>(ts)/comm->nRanks ||
     plan->blockElements>SIZE_MAX/static_cast<size_t>(ts) ||
     plan->blockElements*static_cast<size_t>(ts)<MinBlockBytes ||
     recvcount*static_cast<size_t>(ts)<MinBlockBytes) return ncclInvalidArgument;
  NCCLCHECKGOTO(ncclCudaGetCapturingGraph(&graph,stream,comm->config.graphUsageMode),ret,fail);
  if(ncclCudaGraphValid(graph)) return ncclInvalidUsage;
  if(comm->nRanks==1) {
    CUDACHECK(cudaMemcpyAsync(recvbuff,sendbuff,recvcount*ts,cudaMemcpyDefault,stream));
    return ncclSuccess;
  }
  job=new(std::nothrow) Job;
  if(!job) return ncclSystemError;
  job->sendbuff=sendbuff; job->recvbuff=recvbuff; job->recvcount=recvcount;
  job->datatype=datatype; job->op=op; job->comm=comm; job->plan=*plan; job->stream=stream;
  {
    Job* submitted=job;
    job=nullptr;
    NCCLCHECKGOTO(ncclAsyncLaunch(&submitted->base,jobMain,nullptr,jobFree,comm),ret,fail);
  }
  return ncclSuccess;
fail:
  if(job) delete job;
  return ret;
}
