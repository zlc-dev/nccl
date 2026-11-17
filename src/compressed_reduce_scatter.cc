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
#include <mutex>
#include <new>
#include <thread>
#include <vector>

struct alignas(64) ncclCompressedRsHostSync {
  uint64_t recordIssued;
  uint64_t waitIssued;
  char padding[64 - 2 * sizeof(uint64_t)];
};
static_assert(sizeof(ncclCompressedRsHostSync) == 64, "slot sync must occupy one cache line");

struct ncclCompressedRsHostSyncPage {
  ncclCompressedRsHostSync slots[MAXCHANNELS][NCCL_COMPRESSED_PIPELINE_DEPTH];
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
  cudaEvent_t localSlotEvents[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaEvent_t nextSlotEvents[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  uint64_t localSlotPhase[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  uint64_t nextSlotPhase[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaStream_t sendStream = nullptr;
  cudaStream_t recvStream = nullptr;
  bool used = false;
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
constexpr size_t SlotAlignment = 256;
constexpr size_t PayloadOffset = (sizeof(ncclCompressedP2pSlot) + 15) & ~size_t(15);
constexpr uint64_t IpcSocketHashSalt = 0x435253504f4f4cULL;

struct ManagerInitInfo {
  char syncPath[PATH_MAX];
  cudaIpcEventHandle_t slotEvents[MAXCHANNELS][NCCL_COMPRESSED_PIPELINE_DEPTH];
};

struct OperationRing {
  void* tx = nullptr;
  void* rx = nullptr;
  void* remoteRx = nullptr;
  ncclCompressionTask_t* tasks = nullptr;
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

struct SlotState {
  size_t block = 0;
  int segments = 0;
  int round = 0;
  bool active = false;
};

struct RingLaunchState {
  cudaEvent_t ready[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaEvent_t done[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaEvent_t sendDone[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaEvent_t finish = nullptr;
  SlotState slots[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  size_t nextBlock = 0;
  int activeSlots = 0;
  int active = 0;
};

static size_t alignUp(size_t n, size_t a) { return (n + a - 1) / a * a; }
static char* slot(void* arena, size_t stride, int i) {
  return static_cast<char*>(arena) + stride * static_cast<size_t>(i);
}
static ncclResult_t record(cudaEvent_t e, cudaStream_t s) {
  CUDACHECK(cudaEventRecord(e, s));
  return ncclSuccess;
}

static bool traceEnabled() {
  static bool enabled = std::getenv("NCCL_COMPRESSED_RS_TRACE") != nullptr;
  return enabled;
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

static ncclResult_t recordIpcPhase(
    ncclComm* comm, cudaEvent_t event, ncclCompressedRsHostSync* sync,
    uint64_t phase, cudaStream_t stream) {
  if (phase == 0) return ncclInternalError;
  // Re-record is legal only after the peer has submitted the wait which
  // captured the preceding record.
  NCCLCHECK(waitHostPhase(comm, &sync->waitIssued, phase - 1));
  CUDACHECK(cudaEventRecord(event, stream));
  __atomic_store_n(&sync->recordIssued, phase, __ATOMIC_RELEASE);
  return ncclSuccess;
}

static ncclResult_t waitIpcPhase(
    ncclComm* comm, cudaEvent_t event, ncclCompressedRsHostSync* sync,
    uint64_t phase, cudaStream_t stream) {
  // recordIssued means cudaEventRecord has been submitted, not completed.
  NCCLCHECK(waitHostPhase(comm, &sync->recordIssued, phase));
  CUDACHECK(cudaStreamWaitEvent(stream, event, 0));
  __atomic_store_n(&sync->waitIssued, phase, __ATOMIC_RELEASE);
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
    for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
      if (ring.nextSlotEvents[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.nextSlotEvents[s]));
      ring.nextSlotEvents[s] = nullptr;
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
done:
  traceDestroy(comm, "local-events-close-begin");
  for (int c = 0; c < m->ringCount; c++) {
    ncclCompressedRsRing& ring = m->rings[c];
    for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
      if (ring.nextSlotEvents[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.nextSlotEvents[s]));
      if (ring.localSlotEvents[s]) CUDACHECKIGNORE(cudaEventDestroy(ring.localSlotEvents[s]));
      ring.nextSlotEvents[s] = nullptr;
      ring.localSlotEvents[s] = nullptr;
      ring.localSlotPhase[s] = 0;
      ring.nextSlotPhase[s] = 0;
    }
    if (ring.nextSyncHandle) (void)ncclOsShmClose(ring.nextSyncHandle);
    ring.nextSyncHandle = nullptr;
    ring.nextSync = nullptr;
    ring.nextPool = nullptr;
    ring.ownsNextPool = false;
    ring.used = false;
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
    for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ring.localSlotEvents[s],
          cudaEventDisableTiming | cudaEventInterprocess), ret, fail);
      CUDACHECKGOTO(cudaIpcGetEventHandle(&localInfo.slotEvents[c][s],
                                          ring.localSlotEvents[s]), ret, fail);
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
    for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
      CUDACHECKGOTO(cudaIpcOpenEventHandle(&ring.nextSlotEvents[s],
          allInfo[ring.next].slotEvents[c][s]), ret, fail);
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
  size_t arenaSlots = NCCL_COMPRESSED_PIPELINE_DEPTH;
  if (op->stride > SIZE_MAX / arenaSlots) return ncclInvalidArgument;
  size_t arenaBytes = op->stride * arenaSlots;
  size_t nblocks = 1 + (j->recvcount - 1) / j->plan.blockElements;
  int activeRings = std::min<size_t>(m->ringCount, nblocks);
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
    for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ringState[c].ready[s], cudaEventDisableTiming), ret, fail);
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ringState[c].done[s], cudaEventDisableTiming), ret, fail);
      CUDACHECKGOTO(cudaEventCreateWithFlags(&ringState[c].sendDone[s], cudaEventDisableTiming), ret, fail);
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
    if (m->tailRecorded)
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        if (!ring.used) continue;
        NCCLCHECKGOTO(waitIpcPhase(comm, ring.nextSlotEvents[0],
            &ring.nextSync->slots[c][0], ++ring.nextSlotPhase[0],
            ring.sendStream), ret, fail);
      }
    for (int c = 0; c < activeRings; c++) {
      OperationRing& ringOp = op->rings[c];
      CUDACHECKGOTO(cudaMallocFromPoolAsync(&ringOp.tx, arenaBytes, m->localPool, cs), ret, fail);
      CUDACHECKGOTO(cudaMallocFromPoolAsync(&ringOp.rx, arenaBytes, m->localPool, cs), ret, fail);
      CUDACHECKGOTO(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&ringOp.tasks),
          sizeof(*ringOp.tasks) * arenaSlots, m->localPool, cs), ret, fail);
      CUDACHECKGOTO(cudaMemsetAsync(ringOp.rx, 0, arenaBytes, cs), ret, fail);
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
    // Publish every Ring's local allocation before waiting on any remote Ring.
    for (int c = 0; c < activeRings; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      size_t ringBlocks = (nblocks + activeRings - 1 - c) / activeRings;
      ringState[c].activeSlots = std::min<size_t>(NCCL_COMPRESSED_PIPELINE_DEPTH, ringBlocks);
      for (int s = 0; s < ringState[c].activeSlots; s++)
        NCCLCHECKGOTO(recordIpcPhase(comm, ring.localSlotEvents[s],
            &m->localSync->slots[c][s], ++ring.localSlotPhase[s],
            ring.sendStream), ret, fail);
    }
    for (int c = 0; c < activeRings; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      for (int s = 0; s < ringState[c].activeSlots; s++)
        NCCLCHECKGOTO(waitIpcPhase(comm, ring.nextSlotEvents[s],
            &ring.nextSync->slots[c][s], ++ring.nextSlotPhase[s],
            ring.sendStream), ret, fail);
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
    int totalActive = 0;
    for (int c = 0; c < activeRings; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      RingLaunchState& state = ringState[c];
      NCCLCHECKGOTO(wait(ring.sendStream, preencodeDone), ret, fail);
      state.nextBlock = c;
      state.active = state.activeSlots;
      totalActive += state.active;
      for (int s = 0; s < state.activeSlots; s++) {
        SlotState& ss = state.slots[s];
        ss.block = state.nextBlock;
        state.nextBlock += activeRings;
        ss.segments = 1;
        ss.active = true;
        NCCLCHECKGOTO(record(state.done[s], j->stream), ret, fail);
        NCCLCHECKGOTO(wait(ring.sendStream, state.done[s]), ret, fail);
        NCCLCHECKGOTO(ncclLaunchCompressedSend(slot(op->preencoded, op->stride, ss.block),
            slot(op->rings[c].remoteRx, op->stride, s), op->stride, capacity,
            1, ctasPerSegment, ring.sendStream), ret, fail);
        NCCLCHECKGOTO(recordIpcPhase(comm, ring.nextSlotEvents[s],
            &ring.nextSync->slots[c][s], ++ring.nextSlotPhase[s],
            ring.sendStream), ret, fail);
        NCCLCHECKGOTO(record(state.sendDone[s], ring.sendStream), ret, fail);
      }
    }
    while (totalActive != 0) {
      for (int c = 0; c < activeRings; c++) {
        ncclCompressedRsRing& ring = m->rings[c];
        RingLaunchState& state = ringState[c];
        OperationRing& ringOp = op->rings[c];
        for (int s = 0; s < state.activeSlots; s++) {
          SlotState& ss = state.slots[s];
          if (!ss.active) continue;
          int r = ss.round;
          bool final = r+1 == comm->nRanks-1;
          int dst = comm->channels[c].ring.userRanks[comm->nRanks-2-r];
          NCCLCHECKGOTO(wait(ring.recvStream, state.done[s]), ret, fail);
          NCCLCHECKGOTO(waitIpcPhase(comm, ring.localSlotEvents[s],
              &m->localSync->slots[c][s], ++ring.localSlotPhase[s],
              ring.recvStream), ret, fail);
          if (!final) NCCLCHECKGOTO(wait(ring.recvStream, state.sendDone[s]), ret, fail);
          size_t off = ss.block * j->plan.blockElements;
          size_t count = std::min(j->plan.blockElements, j->recvcount-off);
          NCCLCHECKGOTO(ncclLaunchCompressedPrepareRound(ringOp.tasks+s,
              slot(ringOp.rx, op->stride, s), slot(ringOp.tx, op->stride, s),
              !final, j->sendbuff, final ? j->recvbuff : nullptr, j->recvcount,
              off, count, dst, typeSize, capacity, ring.recvStream), ret, fail);
          NCCLCHECKGOTO(record(state.ready[s], ring.recvStream), ret, fail);
          NCCLCHECKGOTO(wait(j->stream, state.ready[s]), ret, fail);
          if (final) NCCLCHECKGOTO(j->plan.decodeReduceCopy(ringOp.tasks+s,
              j->datatype, j->op, j->plan.context, j->stream), ret, fail);
          else NCCLCHECKGOTO(j->plan.decodeReduceEncode(ringOp.tasks+s,
              j->datatype, j->op, j->plan.context, j->stream), ret, fail);
          NCCLCHECKGOTO(recordIpcPhase(comm, ring.localSlotEvents[s],
              &m->localSync->slots[c][s], ++ring.localSlotPhase[s],
              j->stream), ret, fail);
          NCCLCHECKGOTO(record(state.done[s], j->stream), ret, fail);
          if (!final) {
            NCCLCHECKGOTO(wait(ring.sendStream, state.done[s]), ret, fail);
            NCCLCHECKGOTO(waitIpcPhase(comm, ring.nextSlotEvents[s],
                &ring.nextSync->slots[c][s], ++ring.nextSlotPhase[s],
                ring.sendStream), ret, fail);
            NCCLCHECKGOTO(ncclLaunchCompressedSend(slot(ringOp.tx, op->stride, s),
                slot(ringOp.remoteRx, op->stride, s), op->stride, capacity,
                1, ctasPerSegment, ring.sendStream), ret, fail);
            NCCLCHECKGOTO(recordIpcPhase(comm, ring.nextSlotEvents[s],
                &ring.nextSync->slots[c][s], ++ring.nextSlotPhase[s],
                ring.sendStream), ret, fail);
            NCCLCHECKGOTO(record(state.sendDone[s], ring.sendStream), ret, fail);
            ss.round++;
          } else if (state.nextBlock < nblocks) {
            ss.block = state.nextBlock;
            state.nextBlock += activeRings;
            ss.round = 0;
            NCCLCHECKGOTO(wait(ring.sendStream, state.done[s]), ret, fail);
            NCCLCHECKGOTO(waitIpcPhase(comm, ring.nextSlotEvents[s],
                &ring.nextSync->slots[c][s], ++ring.nextSlotPhase[s],
                ring.sendStream), ret, fail);
            NCCLCHECKGOTO(ncclLaunchCompressedSend(
                slot(op->preencoded, op->stride, ss.block),
                slot(ringOp.remoteRx, op->stride, s), op->stride, capacity,
                1, ctasPerSegment, ring.sendStream), ret, fail);
            NCCLCHECKGOTO(recordIpcPhase(comm, ring.nextSlotEvents[s],
                &ring.nextSync->slots[c][s], ++ring.nextSlotPhase[s],
                ring.sendStream), ret, fail);
            NCCLCHECKGOTO(record(state.sendDone[s], ring.sendStream), ret, fail);
          } else {
            ss.active = false;
            state.active--;
            totalActive--;
          }
        }
      }
    }
    for (int c = 0; c < activeRings; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      RingLaunchState& state = ringState[c];
      OperationRing& ringOp = op->rings[c];
      for (int s = 0; s < state.activeSlots; s++)
        NCCLCHECKGOTO(waitIpcPhase(comm, ring.nextSlotEvents[s],
            &ring.nextSync->slots[c][s], ++ring.nextSlotPhase[s],
            ring.sendStream), ret, fail);
      CUDACHECKGOTO(cudaFreeAsync(ringOp.remoteRx, ring.sendStream), ret, fail);
      ringOp.remoteRx = nullptr;
      NCCLCHECKGOTO(recordIpcPhase(comm, ring.nextSlotEvents[0],
          &ring.nextSync->slots[c][0], ++ring.nextSlotPhase[0],
          ring.sendStream), ret, fail);
      NCCLCHECKGOTO(waitIpcPhase(comm, ring.localSlotEvents[0],
          &m->localSync->slots[c][0], ++ring.localSlotPhase[0],
          ring.sendStream), ret, fail);
      NCCLCHECKGOTO(record(state.finish, ring.sendStream), ret, fail);
      NCCLCHECKGOTO(wait(cs, state.finish), ret, fail);
    }
    for (int c = 0; c < activeRings; c++) {
      CUDACHECKGOTO(cudaFreeAsync(op->rings[c].rx, cs), ret, fail);
      CUDACHECKGOTO(cudaFreeAsync(op->rings[c].tx, cs), ret, fail);
      CUDACHECKGOTO(cudaFreeAsync(op->rings[c].tasks, cs), ret, fail);
      op->rings[c].rx = nullptr;
      op->rings[c].tx = nullptr;
      op->rings[c].tasks = nullptr;
    }
    CUDACHECKGOTO(cudaFreeAsync(op->preencoded, cs), ret, fail);
    CUDACHECKGOTO(cudaFreeAsync(op->preencodeTasks, cs), ret, fail);
    CUDACHECKGOTO(cudaFreeAsync(op->preencodeDestinations, cs), ret, fail);
    op->preencoded = nullptr;
    op->preencodeTasks = nullptr;
    op->preencodeDestinations = nullptr;
    for (int c = 0; c < activeRings; c++) {
      ncclCompressedRsRing& ring = m->rings[c];
      NCCLCHECKGOTO(recordIpcPhase(comm, ring.localSlotEvents[0],
          &m->localSync->slots[c][0], ++ring.localSlotPhase[0], cs), ret, fail);
      ring.used = true;
    }
    NCCLCHECKGOTO(record(cleanupDone, cs), ret, fail);
    NCCLCHECKGOTO(wait(j->stream, cleanupDone), ret, fail);
    NCCLCHECKGOTO(record(m->tail,j->stream),ret,fail);
    m->tailRecorded = true;
  }
fail:
  if (acquired) {
    ncclResult_t x=ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream,false);
    if (ret==ncclSuccess) ret=x;
  }
  if(setupDone) CUDACHECKIGNORE(cudaEventDestroy(setupDone));
  if(prepareDone) CUDACHECKIGNORE(cudaEventDestroy(prepareDone));
  if(preencodeDone) CUDACHECKIGNORE(cudaEventDestroy(preencodeDone));
  if(cleanupDone) CUDACHECKIGNORE(cudaEventDestroy(cleanupDone));
  for (int c = 0; c < activeRings; c++) {
    if (ringState[c].finish) CUDACHECKIGNORE(cudaEventDestroy(ringState[c].finish));
    for(int s=0;s<NCCL_COMPRESSED_PIPELINE_DEPTH;s++) {
      if(ringState[c].ready[s]) CUDACHECKIGNORE(cudaEventDestroy(ringState[c].ready[s]));
      if(ringState[c].done[s]) CUDACHECKIGNORE(cudaEventDestroy(ringState[c].done[s]));
      if(ringState[c].sendDone[s]) CUDACHECKIGNORE(cudaEventDestroy(ringState[c].sendDone[s]));
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
    if (op->stride > SIZE_MAX / NCCL_COMPRESSED_PIPELINE_DEPTH) {
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
