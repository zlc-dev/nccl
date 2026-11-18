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
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

struct alignas(64) ncclCompressedRsEventSync {
  uint64_t recordIssued;
  uint64_t waitIssued;
  char padding[64 - 2 * sizeof(uint64_t)];
};
static_assert(sizeof(ncclCompressedRsEventSync) == 64, "event sync must occupy one cache line");

struct ncclCompressedRsRxSync {
  ncclCompressedRsEventSync dataReady;
  ncclCompressedRsEventSync canWrite;
};

struct ncclCompressedRsHostSyncPage {
  ncclCompressedRsRxSync rx[MAXCHANNELS][NCCL_COMPRESSED_RX_DEPTH];
  ncclCompressedRsEventSync pointerClosed[MAXCHANNELS];
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
  cudaEvent_t localDataReady[NCCL_COMPRESSED_RX_DEPTH] = {};
  cudaEvent_t localCanWrite[NCCL_COMPRESSED_RX_DEPTH] = {};
  cudaEvent_t nextDataReady[NCCL_COMPRESSED_RX_DEPTH] = {};
  cudaEvent_t nextCanWrite[NCCL_COMPRESSED_RX_DEPTH] = {};
  cudaEvent_t localPointerClosed = nullptr;
  cudaEvent_t nextPointerClosed = nullptr;
  uint64_t localDataReadyPhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t localCanWritePhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t nextDataReadyPhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t nextCanWritePhase[NCCL_COMPRESSED_RX_DEPTH] = {};
  uint64_t localPointerClosedPhase = 0;
  uint64_t nextPointerClosedPhase = 0;
  cudaStream_t sendStream = nullptr;
  cudaStream_t recvStream = nullptr;
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
  bool tailRecorded = false;
  bool initialized = false;
};

namespace {
constexpr size_t MinBlockBytes = 512 * 1024;
constexpr size_t TargetSliceBytes = 256 * 1024;
constexpr int MaxSliceCtas = 4;
constexpr int DefaultActiveRings = 2;
constexpr size_t SlotAlignment = 256;
constexpr size_t PayloadOffset = (sizeof(ncclCompressedP2pSlot) + 15) & ~size_t(15);
constexpr uint64_t IpcSocketHashSalt = 0x435253504f4f4cULL;

struct ManagerInitInfo {
  char syncPath[PATH_MAX];
  cudaIpcEventHandle_t dataReady[MAXCHANNELS][NCCL_COMPRESSED_RX_DEPTH];
  cudaIpcEventHandle_t canWrite[MAXCHANNELS][NCCL_COMPRESSED_RX_DEPTH];
  cudaIpcEventHandle_t pointerClosed[MAXCHANNELS];
};

struct OperationRing {
  void* tx = nullptr;
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

struct RxLaneState {
  size_t block = 0;
  int round = 0;
  bool active = false;
  bool receivePending = false;
  bool releasePending = false;
  bool finalCreditWaited = false;
};

struct TxSlotState {
  cudaEvent_t computeDone = nullptr;
  cudaEvent_t sendDone = nullptr;
  bool available = true;
  bool reuseWait = false;
};

struct OutboundItem {
  int tx = -1;
  size_t block = 0;
};

struct RingLaunchState {
  cudaEvent_t finish = nullptr;
  RxLaneState rx[NCCL_COMPRESSED_RX_DEPTH] = {};
  TxSlotState tx[NCCL_COMPRESSED_TX_DEPTH] = {};
  std::deque<OutboundItem> outbound[NCCL_COMPRESSED_RX_DEPTH];
  int activeRx = 0;
  size_t completedBlocks = 0;
  size_t blockCount = 0;
  bool remotePointerClosed = false;
  bool localPointerClosed = false;
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

static ncclResult_t waitHostPhase(ncclComm* comm, uint64_t* address, uint64_t phase) {
  unsigned spins = 0;
  while (__atomic_load_n(address, __ATOMIC_ACQUIRE) < phase) {
    if (comm->abortFlag != nullptr && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0)
      return ncclSystemError;
    if ((++spins & 0xfff) == 0) std::this_thread::yield();
  }
  return ncclSuccess;
}

static bool canRecordIpcPhase(const ncclCompressedRsEventSync* sync, uint64_t phase) {
  return phase != 0 &&
      __atomic_load_n(&sync->waitIssued, __ATOMIC_ACQUIRE) >= phase - 1;
}

static bool canWaitIpcPhase(const ncclCompressedRsEventSync* sync, uint64_t phase) {
  return __atomic_load_n(&sync->recordIssued, __ATOMIC_ACQUIRE) >= phase;
}

static ncclResult_t tryRecordIpcPhase(
    cudaEvent_t event, ncclCompressedRsEventSync* sync, uint64_t* currentPhase,
    cudaStream_t stream, bool* submitted) {
  uint64_t phase = *currentPhase + 1;
  *submitted = false;
  if (!canRecordIpcPhase(sync, phase)) return ncclSuccess;
  CUDACHECK(cudaEventRecord(event, stream));
  __atomic_store_n(&sync->recordIssued, phase, __ATOMIC_RELEASE);
  *currentPhase = phase;
  *submitted = true;
  return ncclSuccess;
}

static ncclResult_t tryWaitIpcPhase(
    cudaEvent_t event, ncclCompressedRsEventSync* sync, uint64_t* currentPhase,
    cudaStream_t stream, bool* submitted) {
  uint64_t phase = *currentPhase + 1;
  *submitted = false;
  // recordIssued means cudaEventRecord has been submitted, not completed.
  if (!canWaitIpcPhase(sync, phase)) return ncclSuccess;
  CUDACHECK(cudaStreamWaitEvent(stream, event, 0));
  __atomic_store_n(&sync->waitIssued, phase, __ATOMIC_RELEASE);
  *currentPhase = phase;
  *submitted = true;
  return ncclSuccess;
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
    if (ring.recvStream) CUDACHECKIGNORE(cudaStreamDestroy(ring.recvStream));
    ring.sendStream = nullptr;
    ring.recvStream = nullptr;
  }
  // Drop every imported CUDA object first. The next rank owns these objects,
  // so publish the Host-side acknowledgement before waiting for the previous
  // rank to drop the objects exported by this rank.
  traceDestroy(comm, "imported-events-close-begin");
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    for (int s = 0; s < NCCL_COMPRESSED_RX_DEPTH; s++) {
      if (ring.nextDataReady[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.nextDataReady[s]));
      if (ring.nextCanWrite[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.nextCanWrite[s]));
      ring.nextDataReady[s] = nullptr;
      ring.nextCanWrite[s] = nullptr;
    }
    if (ring.nextPointerClosed) CUDACHECKIGNORE(cudaEventDestroy(ring.nextPointerClosed));
    ring.nextPointerClosed = nullptr;
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
done:
  traceDestroy(comm, "local-events-close-begin");
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    for (int s = 0; s < NCCL_COMPRESSED_RX_DEPTH; s++) {
      if (ring.nextDataReady[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.nextDataReady[s]));
      if (ring.nextCanWrite[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.nextCanWrite[s]));
      if (ring.localDataReady[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.localDataReady[s]));
      if (ring.localCanWrite[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.localCanWrite[s]));
      ring.nextDataReady[s] = nullptr;
      ring.nextCanWrite[s] = nullptr;
      ring.localDataReady[s] = nullptr;
      ring.localCanWrite[s] = nullptr;
      ring.localDataReadyPhase[s] = 0;
      ring.localCanWritePhase[s] = 0;
      ring.nextDataReadyPhase[s] = 0;
      ring.nextCanWritePhase[s] = 0;
    }
    if (ring.nextPointerClosed) CUDACHECKIGNORE(cudaEventDestroy(ring.nextPointerClosed));
    if (ring.localPointerClosed) CUDACHECKIGNORE(cudaEventDestroy(ring.localPointerClosed));
    ring.nextPointerClosed = nullptr;
    ring.localPointerClosed = nullptr;
    ring.localPointerClosedPhase = 0;
    ring.nextPointerClosedPhase = 0;
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
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&ring.recvStream, cudaStreamNonBlocking), ret, fail);
    for (int s = 0; s < NCCL_COMPRESSED_RX_DEPTH; s++) {
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ring.localDataReady[s],
          cudaEventDisableTiming | cudaEventInterprocess), ret, fail);
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ring.localCanWrite[s],
          cudaEventDisableTiming | cudaEventInterprocess), ret, fail);
      CUDACHECKGOTO(cudaIpcGetEventHandle(&localInfo.dataReady[c][s],
                                          ring.localDataReady[s]), ret, fail);
      CUDACHECKGOTO(cudaIpcGetEventHandle(&localInfo.canWrite[c][s],
                                          ring.localCanWrite[s]), ret, fail);
    }
    CUDACHECKGOTO(cudaEventCreateWithFlags(&ring.localPointerClosed,
        cudaEventDisableTiming | cudaEventInterprocess), ret, fail);
    CUDACHECKGOTO(cudaIpcGetEventHandle(&localInfo.pointerClosed[c],
                                        ring.localPointerClosed), ret, fail);
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
      CUDACHECKGOTO(cudaIpcOpenEventHandle(&ring.nextDataReady[s],
          allInfo[ring.next].dataReady[c][s]), ret, fail);
      CUDACHECKGOTO(cudaIpcOpenEventHandle(&ring.nextCanWrite[s],
          allInfo[ring.next].canWrite[c][s]), ret, fail);
    }
    CUDACHECKGOTO(cudaIpcOpenEventHandle(&ring.nextPointerClosed,
        allInfo[ring.next].pointerClosed[c]), ret, fail);
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

static ncclResult_t launch(Job* j, Operation* op, size_t capacity, int typeSize) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t cs = nullptr;
  cudaEvent_t setupDone = nullptr;
  cudaEvent_t prepareDone = nullptr;
  cudaEvent_t preencodeDone = nullptr;
  cudaEvent_t cleanupDone = nullptr;
  RingLaunchState ringState[MAXCHANNELS];
  bool acquired = false;
  ncclComm* comm = j->comm;
  ncclCompressedRsManager* m = op->manager;
  int ctasPerSegment = std::max(1, std::min<int>(MaxSliceCtas,
      static_cast<int>((capacity + TargetSliceBytes - 1) / TargetSliceBytes)));
  if (op->stride > SIZE_MAX / NCCL_COMPRESSED_RX_DEPTH ||
      op->stride > SIZE_MAX / NCCL_COMPRESSED_TX_DEPTH)
    return ncclInvalidArgument;
  size_t rxArenaBytes = op->stride * NCCL_COMPRESSED_RX_DEPTH;
  size_t txArenaBytes = op->stride * NCCL_COMPRESSED_TX_DEPTH;
  size_t nblocks = 1 + (j->recvcount - 1) / j->plan.blockElements;
  int activeRings = static_cast<int>(std::min<size_t>(nblocks,
      static_cast<size_t>(std::min(m->ringCount, activeRingLimit()))));
  std::vector<cudaMemPoolPtrExportData> allRxData(
      static_cast<size_t>(comm->nRanks) * activeRings);
  std::vector<int> destinations(nblocks);
  if (traceEnabled()) {
    std::fprintf(stderr,
        "compressed-rs rank=%d sequence=%llu phase=enqueue-config rings=%d blocks=%zu\n",
        comm->rank, static_cast<unsigned long long>(op->sequence), activeRings, nblocks);
    std::fflush(stderr);
  }
  trace(op, "enqueue-begin");
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream, false, &cs), ret, fail);
  acquired = true;
  CUDACHECKGOTO(cudaEventCreateWithFlags(&setupDone, cudaEventDisableTiming), ret, fail);
  CUDACHECKGOTO(cudaEventCreateWithFlags(&prepareDone, cudaEventDisableTiming), ret, fail);
  CUDACHECKGOTO(cudaEventCreateWithFlags(&preencodeDone, cudaEventDisableTiming), ret, fail);
  CUDACHECKGOTO(cudaEventCreateWithFlags(&cleanupDone, cudaEventDisableTiming), ret, fail);
  for (int c = 0; c < activeRings; c++) {
    CUDACHECKGOTO(cudaEventCreateWithFlags(&ringState[c].finish, cudaEventDisableTiming), ret, fail);
    for (int s = 0; s < NCCL_COMPRESSED_TX_DEPTH; s++) {
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ringState[c].tx[s].computeDone,
          cudaEventDisableTiming), ret, fail);
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ringState[c].tx[s].sendDone,
          cudaEventDisableTiming), ret, fail);
    }
  }
  {
    size_t preencodeBlocks = nblocks;
    if (op->stride > SIZE_MAX / preencodeBlocks ||
        sizeof(*op->preencodeTasks) > SIZE_MAX / preencodeBlocks) {
      ret = ncclInvalidArgument;
      goto fail;
    }
    size_t preencodeBytes = op->stride * preencodeBlocks;
    if (m->tailRecorded) NCCLCHECKGOTO(wait(cs, m->tail), ret, fail);
    for (int c = 0; c < activeRings; c++) {
      OperationRing& ringOp = op->rings[c];
      CUDACHECKGOTO(cudaMallocFromPoolAsync(&ringOp.tx, txArenaBytes, m->localPool, cs), ret, fail);
      CUDACHECKGOTO(cudaMallocFromPoolAsync(&ringOp.rx, rxArenaBytes, m->localPool, cs), ret, fail);
      CUDACHECKGOTO(cudaMemsetAsync(ringOp.rx, 0, rxArenaBytes, cs), ret, fail);
      CUDACHECKGOTO(cudaMemPoolExportPointer(
          &allRxData[static_cast<size_t>(comm->rank) * activeRings + c], ringOp.rx), ret, fail);
    }
    CUDACHECKGOTO(cudaMallocFromPoolAsync(&op->preencoded, preencodeBytes,
        m->localPool, cs), ret, fail);
    CUDACHECKGOTO(cudaMallocFromPoolAsync(
        reinterpret_cast<void**>(&op->preencodeTasks),
        sizeof(*op->preencodeTasks) * preencodeBlocks, m->localPool, cs), ret, fail);
    CUDACHECKGOTO(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&op->preencodeDestinations),
        sizeof(*op->preencodeDestinations) * nblocks, m->localPool, cs), ret, fail);
    for (size_t b = 0; b < nblocks; b++) {
      int c = static_cast<int>(b % activeRings);
      destinations[b] = comm->channels[c].ring.userRanks[comm->nRanks-1];
    }
    CUDACHECKGOTO(cudaMemcpyAsync(op->preencodeDestinations, destinations.data(),
        sizeof(*op->preencodeDestinations) * nblocks, cudaMemcpyHostToDevice, cs), ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allRxData.data(),
        sizeof(cudaMemPoolPtrExportData) * activeRings), ret, fail);
    for (int c = 0; c < activeRings; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      OperationRing& ringOp = op->rings[c];
      CUDACHECKGOTO(cudaMemPoolImportPointer(&ringOp.remoteRx, ring.nextPool,
          &allRxData[static_cast<size_t>(ring.next) * activeRings + c]), ret, fail);
    }
    NCCLCHECKGOTO(record(setupDone, cs), ret, fail);
    for (int c = 0; c < activeRings; c++) {
      NCCLCHECKGOTO(wait(m->rings[c].sendStream, setupDone), ret, fail);
      NCCLCHECKGOTO(wait(m->rings[c].recvStream, setupDone), ret, fail);
    }
    for (int c = 0; c < activeRings; c++) {
      RingLaunchState& state = ringState[c];
      state.blockCount = (nblocks + activeRings - 1 - c) / activeRings;
      state.activeRx = std::min<size_t>(NCCL_COMPRESSED_RX_DEPTH, state.blockCount);
      for (int lane = 0; lane < state.activeRx; lane++) {
        RxLaneState& rx = state.rx[lane];
        rx.block = c + static_cast<size_t>(lane) * activeRings;
        rx.active = true;
        rx.receivePending = true;
        rx.releasePending = true;
        state.outbound[lane].push_back({-1, rx.block});
      }
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
    for (;;) {
      bool done = true;
      bool progressed = false;
      for (int c = 0; c < activeRings; c++) {
        RingLaunchState& state = ringState[c];
        if (state.completedBlocks != state.blockCount) done = false;
        for (int lane = 0; lane < state.activeRx; lane++)
          if (!state.outbound[lane].empty()) done = false;
      }
      if (done) break;

      // Return RX credits independently. This is the receive FIFO's
      // backpressure edge; each lane has a fixed upstream producer.
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        for (int lane = 0; lane < state.activeRx; lane++) {
          RxLaneState& rx = state.rx[lane];
          if (!rx.releasePending) continue;
          bool submitted = false;
          NCCLCHECKGOTO(tryRecordIpcPhase(ring.localCanWrite[lane],
              &m->localSync->rx[c][lane].canWrite,
              &ring.localCanWritePhase[lane], ring.recvStream, &submitted), ret, fail);
          if (submitted) {
            rx.releasePending = false;
            progressed = true;
          }
        }
      }

      // Drain each lane's outbound queue without coupling it to a TX index.
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        OperationRing& ringOp = op->rings[c];
        for (int lane = 0; lane < state.activeRx; lane++) {
          if (state.outbound[lane].empty()) continue;
          ncclCompressedRsEventSync* credit = &ring.nextSync->rx[c][lane].canWrite;
          ncclCompressedRsEventSync* ready = &ring.nextSync->rx[c][lane].dataReady;
          if (!canWaitIpcPhase(credit, ring.nextCanWritePhase[lane] + 1) ||
              !canRecordIpcPhase(ready, ring.nextDataReadyPhase[lane] + 1))
            continue;
          bool submitted = false;
          NCCLCHECKGOTO(tryWaitIpcPhase(ring.nextCanWrite[lane], credit,
              &ring.nextCanWritePhase[lane], ring.sendStream, &submitted), ret, fail);
          if (!submitted) continue;
          OutboundItem item = state.outbound[lane].front();
          void* source = item.tx < 0
              ? slot(op->preencoded, op->stride, item.block)
              : slot(ringOp.tx, op->stride, item.tx);
          if (item.tx >= 0)
            NCCLCHECKGOTO(wait(ring.sendStream, state.tx[item.tx].computeDone), ret, fail);
          NCCLCHECKGOTO(ncclLaunchCompressedSend(source,
              slot(ringOp.remoteRx, op->stride, lane), op->stride, capacity,
              1, ctasPerSegment, ring.sendStream), ret, fail);
          NCCLCHECKGOTO(tryRecordIpcPhase(ring.nextDataReady[lane], ready,
              &ring.nextDataReadyPhase[lane], ring.sendStream, &submitted), ret, fail);
          if (!submitted) { ret = ncclInternalError; goto fail; }
          if (item.tx >= 0) {
            NCCLCHECKGOTO(record(state.tx[item.tx].sendDone, ring.sendStream), ret, fail);
            state.tx[item.tx].available = true;
            state.tx[item.tx].reuseWait = true;
          }
          state.outbound[lane].pop_front();
          progressed = true;
        }
      }

      // A single compute stream scans all ready RX lanes. TX allocation is
      // independent, so a busy send does not pin the corresponding RX lane.
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        OperationRing& ringOp = op->rings[c];
        for (int lane = 0; lane < state.activeRx; lane++) {
          RxLaneState& rx = state.rx[lane];
          if (!rx.active || !rx.receivePending || rx.releasePending) continue;
          ncclCompressedRsEventSync* ready = &m->localSync->rx[c][lane].dataReady;
          ncclCompressedRsEventSync* credit = &m->localSync->rx[c][lane].canWrite;
          if (!canWaitIpcPhase(ready, ring.localDataReadyPhase[lane] + 1) ||
              !canRecordIpcPhase(credit, ring.localCanWritePhase[lane] + 1))
            continue;
          bool final = rx.round + 1 == comm->nRanks - 1;
          int tx = -1;
          if (!final) {
            for (int candidate = 0; candidate < NCCL_COMPRESSED_TX_DEPTH; candidate++)
              if (state.tx[candidate].available) { tx = candidate; break; }
            if (tx < 0) continue;
            state.tx[tx].available = false;
          }
          bool submitted = false;
          NCCLCHECKGOTO(tryWaitIpcPhase(ring.localDataReady[lane], ready,
              &ring.localDataReadyPhase[lane], j->stream, &submitted), ret, fail);
          if (!submitted) { ret = ncclInternalError; goto fail; }
          if (tx >= 0 && state.tx[tx].reuseWait) {
            NCCLCHECKGOTO(wait(j->stream, state.tx[tx].sendDone), ret, fail);
            state.tx[tx].reuseWait = false;
          }
          int dst = comm->channels[c].ring.userRanks[comm->nRanks - 2 - rx.round];
          size_t off = rx.block * j->plan.blockElements;
          size_t count = std::min(j->plan.blockElements, j->recvcount - off);
          void* rxSlot = slot(ringOp.rx, op->stride, lane);
          void* txSlot = tx >= 0 ? slot(ringOp.tx, op->stride, tx) : nullptr;
          size_t inputElement = static_cast<size_t>(dst) * j->recvcount + off;
          ncclCompressionTask_t task = {};
          task.compressedInput = payload(rxSlot);
          task.inputHeader = header(rxSlot);
          task.compressedOutput = txSlot == nullptr ? nullptr : payload(txSlot);
          task.outputHeader = txSlot == nullptr ? nullptr : header(txSlot);
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
          NCCLCHECKGOTO(tryRecordIpcPhase(ring.localCanWrite[lane], credit,
              &ring.localCanWritePhase[lane], j->stream, &submitted), ret, fail);
          if (!submitted) { ret = ncclInternalError; goto fail; }
          rx.receivePending = false;
          if (!final) {
            NCCLCHECKGOTO(record(state.tx[tx].computeDone, j->stream), ret, fail);
            state.outbound[lane].push_back({tx, rx.block});
            rx.round++;
            rx.receivePending = true;
          } else {
            state.completedBlocks++;
            size_t nextBlock = rx.block +
                static_cast<size_t>(activeRings) * state.activeRx;
            if (nextBlock < nblocks) {
              rx.block = nextBlock;
              rx.round = 0;
              rx.receivePending = true;
              state.outbound[lane].push_back({-1, nextBlock});
            } else {
              rx.active = false;
            }
          }
          progressed = true;
        }
      }
      if (!progressed) {
        if (comm->abortFlag != nullptr &&
            __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) {
          ret = ncclSystemError;
          goto fail;
        }
        std::this_thread::yield();
      }
    }

    // Consume the final credit for every remote RX lane before dropping the
    // imported pointer. A separate IPC Event then closes pointer ownership.
    for (;;) {
      bool closed = true;
      bool progressed = false;
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        bool creditsDone = true;
        for (int lane = 0; lane < state.activeRx; lane++) {
          if (state.rx[lane].finalCreditWaited) continue;
          creditsDone = false;
          bool submitted = false;
          NCCLCHECKGOTO(tryWaitIpcPhase(ring.nextCanWrite[lane],
              &ring.nextSync->rx[c][lane].canWrite,
              &ring.nextCanWritePhase[lane], ring.sendStream, &submitted), ret, fail);
          if (submitted) {
            state.rx[lane].finalCreditWaited = true;
            progressed = true;
          }
        }
        if (!creditsDone) closed = false;
        if (creditsDone && !state.remotePointerClosed) {
          closed = false;
          ncclCompressedRsEventSync* sync = &ring.nextSync->pointerClosed[c];
          if (canRecordIpcPhase(sync, ring.nextPointerClosedPhase + 1)) {
            CUDACHECKGOTO(cudaFreeAsync(op->rings[c].remoteRx, ring.sendStream), ret, fail);
            op->rings[c].remoteRx = nullptr;
            bool submitted = false;
            NCCLCHECKGOTO(tryRecordIpcPhase(ring.nextPointerClosed, sync,
                &ring.nextPointerClosedPhase, ring.sendStream, &submitted), ret, fail);
            if (!submitted) { ret = ncclInternalError; goto fail; }
            NCCLCHECKGOTO(record(state.finish, ring.sendStream), ret, fail);
            state.remotePointerClosed = true;
            progressed = true;
          }
        }
        if (!state.localPointerClosed) {
          closed = false;
          bool submitted = false;
          NCCLCHECKGOTO(tryWaitIpcPhase(ring.localPointerClosed,
              &m->localSync->pointerClosed[c], &ring.localPointerClosedPhase,
              cs, &submitted), ret, fail);
          if (submitted) {
            state.localPointerClosed = true;
            progressed = true;
          }
        }
      }
      if (closed) break;
      if (!progressed) {
        if (comm->abortFlag != nullptr &&
            __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) {
          ret = ncclSystemError;
          goto fail;
        }
        std::this_thread::yield();
      }
    }
    for (int c = 0; c < activeRings; c++) {
      NCCLCHECKGOTO(wait(cs, ringState[c].finish), ret, fail);
      CUDACHECKGOTO(cudaFreeAsync(op->rings[c].rx, cs), ret, fail);
      CUDACHECKGOTO(cudaFreeAsync(op->rings[c].tx, cs), ret, fail);
      op->rings[c].rx = nullptr;
      op->rings[c].tx = nullptr;
    }
    CUDACHECKGOTO(cudaFreeAsync(op->preencoded, cs), ret, fail);
    CUDACHECKGOTO(cudaFreeAsync(op->preencodeTasks, cs), ret, fail);
    CUDACHECKGOTO(cudaFreeAsync(op->preencodeDestinations, cs), ret, fail);
    op->preencoded = nullptr;
    op->preencodeTasks = nullptr;
    op->preencodeDestinations = nullptr;
    NCCLCHECKGOTO(record(cleanupDone, cs), ret, fail);
    NCCLCHECKGOTO(wait(j->stream, cleanupDone), ret, fail);
    NCCLCHECKGOTO(record(m->tail, j->stream), ret, fail);
    m->tailRecorded = true;
  }
fail:
  if (acquired) {
    ncclResult_t x = ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream, false);
    if (ret == ncclSuccess) ret = x;
  }
  if (setupDone) CUDACHECKIGNORE(cudaEventDestroy(setupDone));
  if (prepareDone) CUDACHECKIGNORE(cudaEventDestroy(prepareDone));
  if (preencodeDone) CUDACHECKIGNORE(cudaEventDestroy(preencodeDone));
  if (cleanupDone) CUDACHECKIGNORE(cudaEventDestroy(cleanupDone));
  for (int c = 0; c < activeRings; c++) {
    if (ringState[c].finish) CUDACHECKIGNORE(cudaEventDestroy(ringState[c].finish));
    for (int s = 0; s < NCCL_COMPRESSED_TX_DEPTH; s++) {
      if (ringState[c].tx[s].computeDone)
        CUDACHECKIGNORE(cudaEventDestroy(ringState[c].tx[s].computeDone));
      if (ringState[c].tx[s].sendDone)
        CUDACHECKIGNORE(cudaEventDestroy(ringState[c].tx[s].sendDone));
    }
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
    if (op->stride > SIZE_MAX / std::max(NCCL_COMPRESSED_RX_DEPTH,
                                         NCCL_COMPRESSED_TX_DEPTH)) {
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
