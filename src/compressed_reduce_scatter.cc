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
  ncclCompressedRsHostSync slots[NCCL_COMPRESSED_PIPELINE_DEPTH];
  uint64_t importsClosed;
};

struct ncclCompressedRsManager {
  std::mutex mutex;
  cudaMemPool_t localPool = nullptr;
  cudaMemPool_t nextPool = nullptr;
  ncclCompressedRsHostSyncPage* localSync = nullptr;
  ncclCompressedRsHostSyncPage* nextSync = nullptr;
  ncclShmHandleInternal* localSyncHandle = nullptr;
  ncclShmHandleInternal* nextSyncHandle = nullptr;
  cudaEvent_t localSlotEvents[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaEvent_t nextSlotEvents[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  // Last phase submitted for the locally owned and next rank's imported slot.
  // Every record and wait advances the corresponding phase exactly once.
  uint64_t localSlotPhase[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  uint64_t nextSlotPhase[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaStream_t recvStream = nullptr;
  cudaEvent_t tail = nullptr;
  uint64_t sequence = 0;
  bool tailRecorded = false;
  bool initialized = false;
};

namespace {
constexpr size_t MinBlockBytes = 512 * 1024;
constexpr size_t TargetSliceBytes = 512 * 1024;
constexpr int MaxSliceCtas = 4;
constexpr size_t SlotAlignment = 256;
constexpr size_t PayloadOffset = (sizeof(ncclCompressedP2pSlot) + 15) & ~size_t(15);
constexpr uint64_t IpcSocketHashSalt = 0x435253504f4f4cULL;

struct ManagerInitInfo {
  char syncPath[PATH_MAX];
  cudaIpcEventHandle_t slotEvents[NCCL_COMPRESSED_PIPELINE_DEPTH];
};

struct Operation {
  ncclComm* comm = nullptr;
  ncclCompressedRsManager* manager = nullptr;
  void* tx = nullptr;
  void* rx = nullptr;
  void* remoteRx = nullptr;
  ncclCompressionTask_t* tasks = nullptr;
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
  int round = 0;
  bool active = false;
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
  if (m->recvStream) {
    CUDACHECKGOTO(cudaStreamDestroy(m->recvStream), ret, done);
    m->recvStream = nullptr;
  }
  // Drop every imported CUDA object first. The next rank owns these objects,
  // so publish the Host-side acknowledgement before waiting for the previous
  // rank to drop the objects exported by this rank.
  for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
    if (m->nextSlotEvents[s]) CUDACHECKIGNORE(cudaEventDestroy(m->nextSlotEvents[s]));
    m->nextSlotEvents[s] = nullptr;
  }
  if (m->nextPool) CUDACHECKIGNORE(cudaMemPoolDestroy(m->nextPool));
  m->nextPool = nullptr;
  if (m->initialized && m->nextSync != nullptr)
    __atomic_store_n(&m->nextSync->importsClosed, 1, __ATOMIC_RELEASE);
  traceDestroy(comm, "imports-closed");
  if (m->nextSyncHandle) (void)ncclOsShmClose(m->nextSyncHandle);
  m->nextSyncHandle = nullptr;
  m->nextSync = nullptr;
  if (m->initialized && m->localSync != nullptr) {
    traceDestroy(comm, "peer-imports-wait-begin");
    NCCLCHECKGOTO(waitHostPhase(comm, &m->localSync->importsClosed, 1), ret, done);
    traceDestroy(comm, "peer-imports-wait-end");
  }
done:
  for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
    if (m->nextSlotEvents[s]) CUDACHECKIGNORE(cudaEventDestroy(m->nextSlotEvents[s]));
    if (m->localSlotEvents[s]) CUDACHECKIGNORE(cudaEventDestroy(m->localSlotEvents[s]));
    m->nextSlotEvents[s] = nullptr;
    m->localSlotEvents[s] = nullptr;
    m->localSlotPhase[s] = 0;
    m->nextSlotPhase[s] = 0;
  }
  if (m->nextSyncHandle) (void)ncclOsShmClose(m->nextSyncHandle);
  if (m->localSyncHandle) (void)ncclOsShmClose(m->localSyncHandle);
  m->nextSyncHandle = nullptr;
  m->localSyncHandle = nullptr;
  m->nextSync = nullptr;
  m->localSync = nullptr;
  if (m->localPool) CUDACHECKIGNORE(cudaMemPoolDestroy(m->localPool));
  m->nextPool = nullptr;
  m->localPool = nullptr;
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
  int next = comm->channels[0].ring.next;
  int prev = comm->channels[0].ring.prev;
  int supported = 0;
  int ipcEventSupported = 0;
  ncclIpcSocket ipcSocket = {};
  ncclIpcFd exportedFd = NCCL_INVALID_IPC_FD;
  ncclIpcFd importedFd = NCCL_INVALID_IPC_FD;
  ManagerInitInfo localInfo = {};
  std::vector<ManagerInitInfo> allInfo(comm->nRanks);
  uint64_t socketHash = comm->commHash ^ IpcSocketHashSalt;

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
    cudaMemAccessDesc access = {};
    access.location.type = cudaMemLocationTypeDevice;
    access.location.id = comm->peerInfo[prev].cudaDev;
    access.flags = cudaMemAccessFlagsProtReadWrite;
    CUDACHECKGOTO(cudaMemPoolSetAccess(m->localPool, &access, 1), ret, fail);
  }
  CUDACHECKGOTO(cudaEventCreateWithFlags(&m->tail, cudaEventDisableTiming), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&m->recvStream, cudaStreamNonBlocking), ret, fail);
  NCCLCHECKGOTO(ncclOsShmOpen(localInfo.syncPath, sizeof(localInfo.syncPath),
      sizeof(*m->localSync), reinterpret_cast<void**>(&m->localSync), nullptr,
      2, &m->localSyncHandle), ret, fail);
  __atomic_store_n(&m->localSync->importsClosed, 0, __ATOMIC_RELAXED);
  for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
    CUDACHECKGOTO(cudaEventCreateWithFlags(&m->localSlotEvents[s],
        cudaEventDisableTiming | cudaEventInterprocess), ret, fail);
    CUDACHECKGOTO(cudaIpcGetEventHandle(&localInfo.slotEvents[s],
                                        m->localSlotEvents[s]), ret, fail);
  }
  NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSocket, comm->rank, socketHash, comm->abortFlag), ret, fail);
  CUDACHECKGOTO(cudaMemPoolExportToShareableHandle(&exportedFd, m->localPool,
                                                   cudaMemHandleTypePosixFileDescriptor, 0), ret, fail);
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0x43525000), ret, fail);
  NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSocket, exportedFd, prev, socketHash), ret, fail);
  NCCLCHECKGOTO(ncclIpcSocketRecvFd(&ipcSocket, &importedFd), ret, fail);
  CUDACHECKGOTO(cudaMemPoolImportFromShareableHandle(&m->nextPool,
      reinterpret_cast<void*>(static_cast<intptr_t>(importedFd)),
      cudaMemHandleTypePosixFileDescriptor, 0), ret, fail);
  {
    // Imported pools do not inherit the exporter's access map. Grant this
    // rank's GPU access before importing or dereferencing any allocation.
    cudaMemAccessDesc access = {};
    access.location.type = cudaMemLocationTypeDevice;
    access.location.id = comm->cudaDev;
    access.flags = cudaMemAccessFlagsProtReadWrite;
    CUDACHECKGOTO(cudaMemPoolSetAccess(m->nextPool, &access, 1), ret, fail);
  }
  NCCLCHECKGOTO(ncclIpcSocketClose(&ipcSocket), ret, fail);
  NCCLCHECKGOTO(ncclIpcFdClose(exportedFd) == 0 ? ncclSuccess : ncclSystemError, ret, fail);
  exportedFd = NCCL_INVALID_IPC_FD;
  NCCLCHECKGOTO(ncclIpcFdClose(importedFd) == 0 ? ncclSuccess : ncclSystemError, ret, fail);
  importedFd = NCCL_INVALID_IPC_FD;

  allInfo[comm->rank] = localInfo;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allInfo.data(), sizeof(localInfo)), ret, fail);
  NCCLCHECKGOTO(ncclOsShmOpen(allInfo[next].syncPath, sizeof(allInfo[next].syncPath),
      sizeof(*m->nextSync), reinterpret_cast<void**>(&m->nextSync), nullptr,
      0, &m->nextSyncHandle), ret, fail);
  for (int s = 0; s < NCCL_COMPRESSED_PIPELINE_DEPTH; s++) {
    CUDACHECKGOTO(cudaIpcOpenEventHandle(&m->nextSlotEvents[s],
                                         allInfo[next].slotEvents[s]), ret, fail);
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
  cudaEvent_t ready[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaEvent_t done[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaEvent_t sendDone[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
  cudaMemPoolPtrExportData rxData = {};
  bool acquired = false;
  ncclComm* comm = j->comm;
  ncclCompressedRsManager* m = op->manager;
  int next = comm->channels[0].ring.next;
  int sliceCtas = std::max(1, std::min<int>(MaxSliceCtas,
      static_cast<int>((capacity + TargetSliceBytes - 1) / TargetSliceBytes)));
  size_t arenaBytes = op->stride * NCCL_COMPRESSED_PIPELINE_DEPTH;
  std::vector<cudaMemPoolPtrExportData> allRxData(comm->nRanks);
  trace(op, "enqueue-begin");
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream, false, &cs), ret, fail);
  acquired = true;
  CUDACHECKGOTO(cudaEventCreateWithFlags(&setupDone, cudaEventDisableTiming), ret, fail);
  for (int i = 0; i < NCCL_COMPRESSED_PIPELINE_DEPTH; i++) {
    CUDACHECKGOTO(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming), ret, fail);
    CUDACHECKGOTO(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming), ret, fail);
    CUDACHECKGOTO(cudaEventCreateWithFlags(&sendDone[i], cudaEventDisableTiming), ret, fail);
  }
  {
    size_t nblocks = 1 + (j->recvcount - 1) / j->plan.blockElements;
    int activeSlots = static_cast<int>(std::min<size_t>(
        NCCL_COMPRESSED_PIPELINE_DEPTH, nblocks));
    if (m->tailRecorded) NCCLCHECKGOTO(wait(cs, m->tail), ret, fail);
    // Slot 0 carries the previous operation's allocation-free ack. Consume it
    // before this operation publishes a new allocation-ready phase.
    if (m->tailRecorded) {
      NCCLCHECKGOTO(waitIpcPhase(comm, m->nextSlotEvents[0],
          &m->nextSync->slots[0], ++m->nextSlotPhase[0], cs), ret, fail);
    }
    CUDACHECKGOTO(cudaMallocFromPoolAsync(&op->tx, arenaBytes, m->localPool, cs), ret, fail);
    CUDACHECKGOTO(cudaMallocFromPoolAsync(&op->rx, arenaBytes, m->localPool, cs), ret, fail);
    CUDACHECKGOTO(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&op->tasks),
        sizeof(*op->tasks) * NCCL_COMPRESSED_PIPELINE_DEPTH, m->localPool, cs), ret, fail);
    CUDACHECKGOTO(cudaMemsetAsync(op->rx, 0, arenaBytes, cs), ret, fail);
    CUDACHECKGOTO(cudaMemPoolExportPointer(&rxData, op->rx), ret, fail);
    allRxData[comm->rank] = rxData;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allRxData.data(), sizeof(rxData)), ret, fail);
    CUDACHECKGOTO(cudaMemPoolImportPointer(&op->remoteRx, m->nextPool,
                                           &allRxData[next]), ret, fail);
    NCCLCHECKGOTO(record(setupDone,cs),ret,fail);
    NCCLCHECKGOTO(wait(m->recvStream,setupDone),ret,fail);
    // Imported memory must not be accessed until the allocation has completed
    // in the exporting process. Publish all local slots first so the ring has
    // no Host-side record/wait cycle, then consume the next rank's phases.
    for (int s = 0; s < activeSlots; s++) {
      NCCLCHECKGOTO(recordIpcPhase(comm, m->localSlotEvents[s],
          &m->localSync->slots[s], ++m->localSlotPhase[s], cs), ret, fail);
    }
    for (int s = 0; s < activeSlots; s++) {
      NCCLCHECKGOTO(waitIpcPhase(comm, m->nextSlotEvents[s],
          &m->nextSync->slots[s], ++m->nextSlotPhase[s], cs), ret, fail);
    }
    NCCLCHECKGOTO(record(done[0], j->stream), ret, fail);
    NCCLCHECKGOTO(wait(cs, done[0]), ret, fail);
    SlotState states[NCCL_COMPRESSED_PIPELINE_DEPTH] = {};
    size_t nextBlock = static_cast<size_t>(activeSlots);
    int active = activeSlots;
    for (int s = 0; s < activeSlots; s++) {
      states[s].block = static_cast<size_t>(s);
      states[s].active = true;
      size_t off = states[s].block*j->plan.blockElements;
      size_t count = std::min(j->plan.blockElements, j->recvcount-off);
      int dst = comm->channels[0].ring.userRanks[comm->nRanks-1];
      NCCLCHECKGOTO(ncclLaunchCompressedPrepareEncode(op->tasks+s, slot(op->tx,op->stride,s),
        j->sendbuff,j->recvcount,off,count,dst,typeSize,capacity,cs), ret, fail);
      NCCLCHECKGOTO(record(ready[s],cs),ret,fail); NCCLCHECKGOTO(wait(j->stream,ready[s]),ret,fail);
      NCCLCHECKGOTO(j->plan.encode(op->tasks+s,j->datatype,j->plan.context,j->stream),ret,fail);
      NCCLCHECKGOTO(record(done[s],j->stream),ret,fail);
    }
    for (int s=0;s<activeSlots;s++) {
      NCCLCHECKGOTO(wait(cs,done[s]),ret,fail);
      NCCLCHECKGOTO(ncclLaunchCompressedSend(slot(op->tx,op->stride,s),
        slot(op->remoteRx,op->stride,s),capacity,sliceCtas,cs),ret,fail);
      NCCLCHECKGOTO(recordIpcPhase(comm, m->nextSlotEvents[s],
          &m->nextSync->slots[s], ++m->nextSlotPhase[s], cs),ret,fail);
      NCCLCHECKGOTO(record(sendDone[s],cs),ret,fail);
    }
    while (active != 0) {
      for (int s=0;s<activeSlots;s++) {
          if (!states[s].active) continue;
          int r = states[s].round;
          bool final = r+1 == comm->nRanks-1;
          int dst = comm->channels[0].ring.userRanks[comm->nRanks-2-r];
          size_t off=states[s].block*j->plan.blockElements;
          size_t count=std::min(j->plan.blockElements,j->recvcount-off);
          NCCLCHECKGOTO(wait(m->recvStream,done[s]),ret,fail);
          NCCLCHECKGOTO(waitIpcPhase(comm, m->localSlotEvents[s],
              &m->localSync->slots[s], ++m->localSlotPhase[s],
              m->recvStream),ret,fail);
          // prepareRound resets the TX frame header for non-final rounds.
          // Protect the entire TX slot, not only the codec payload write,
          // from racing the preceding send kernel.
          if (!final) NCCLCHECKGOTO(wait(m->recvStream,sendDone[s]),ret,fail);
          NCCLCHECKGOTO(ncclLaunchCompressedPrepareRound(op->tasks+s,slot(op->rx,op->stride,s),
            slot(op->tx,op->stride,s),!final,j->sendbuff,final?j->recvbuff:nullptr,j->recvcount,
            off,count,dst,typeSize,capacity,m->recvStream),ret,fail);
          NCCLCHECKGOTO(record(ready[s],m->recvStream),ret,fail);
          NCCLCHECKGOTO(wait(j->stream,ready[s]),ret,fail);
          if (final) NCCLCHECKGOTO(j->plan.decodeReduceCopy(op->tasks+s,j->datatype,j->op,
              j->plan.context,j->stream),ret,fail);
          else NCCLCHECKGOTO(j->plan.decodeReduceEncode(op->tasks+s,j->datatype,j->op,
              j->plan.context,j->stream),ret,fail);
          NCCLCHECKGOTO(recordIpcPhase(comm, m->localSlotEvents[s],
              &m->localSync->slots[s], ++m->localSlotPhase[s],
              j->stream),ret,fail);
          NCCLCHECKGOTO(record(done[s],j->stream),ret,fail);
          if (!final) {
            NCCLCHECKGOTO(wait(cs,done[s]),ret,fail);
            NCCLCHECKGOTO(waitIpcPhase(comm, m->nextSlotEvents[s],
                &m->nextSync->slots[s], ++m->nextSlotPhase[s], cs),ret,fail);
            NCCLCHECKGOTO(ncclLaunchCompressedSend(slot(op->tx,op->stride,s),
              slot(op->remoteRx,op->stride,s),capacity,sliceCtas,cs),ret,fail);
            NCCLCHECKGOTO(recordIpcPhase(comm, m->nextSlotEvents[s],
                &m->nextSync->slots[s], ++m->nextSlotPhase[s], cs),ret,fail);
            NCCLCHECKGOTO(record(sendDone[s],cs),ret,fail);
            states[s].round++;
          } else if (nextBlock < nblocks) {
            // This physical slot is now free. Refill and send it immediately
            // instead of draining the other slot and ending a two-block wave.
            states[s].block = nextBlock++;
            states[s].round = 0;
            off = states[s].block*j->plan.blockElements;
            count = std::min(j->plan.blockElements, j->recvcount-off);
            dst = comm->channels[0].ring.userRanks[comm->nRanks-1];
            NCCLCHECKGOTO(wait(cs,done[s]),ret,fail);
            NCCLCHECKGOTO(ncclLaunchCompressedPrepareEncode(op->tasks+s,
              slot(op->tx,op->stride,s),j->sendbuff,j->recvcount,off,count,dst,
              typeSize,capacity,cs),ret,fail);
            NCCLCHECKGOTO(record(ready[s],cs),ret,fail);
            NCCLCHECKGOTO(wait(j->stream,ready[s]),ret,fail);
            NCCLCHECKGOTO(j->plan.encode(op->tasks+s,j->datatype,
              j->plan.context,j->stream),ret,fail);
            NCCLCHECKGOTO(record(done[s],j->stream),ret,fail);
            NCCLCHECKGOTO(wait(cs,done[s]),ret,fail);
            NCCLCHECKGOTO(waitIpcPhase(comm, m->nextSlotEvents[s],
              &m->nextSync->slots[s], ++m->nextSlotPhase[s], cs),ret,fail);
            NCCLCHECKGOTO(ncclLaunchCompressedSend(slot(op->tx,op->stride,s),
              slot(op->remoteRx,op->stride,s),capacity,sliceCtas,cs),ret,fail);
            NCCLCHECKGOTO(recordIpcPhase(comm, m->nextSlotEvents[s],
              &m->nextSync->slots[s], ++m->nextSlotPhase[s], cs),ret,fail);
            NCCLCHECKGOTO(record(sendDone[s],cs),ret,fail);
          } else {
            states[s].active = false;
            active--;
          }
      }
    }
    for (int s=0;s<activeSlots;s++)
      NCCLCHECKGOTO(waitIpcPhase(comm, m->nextSlotEvents[s],
          &m->nextSync->slots[s], ++m->nextSlotPhase[s], cs),ret,fail);
    CUDACHECKGOTO(cudaFreeAsync(op->remoteRx, cs), ret, fail);
    op->remoteRx = nullptr;
    NCCLCHECKGOTO(recordIpcPhase(comm, m->nextSlotEvents[0],
        &m->nextSync->slots[0], ++m->nextSlotPhase[0], cs),ret,fail);
    NCCLCHECKGOTO(waitIpcPhase(comm, m->localSlotEvents[0],
        &m->localSync->slots[0], ++m->localSlotPhase[0], cs),ret,fail);
    CUDACHECKGOTO(cudaFreeAsync(op->rx, cs), ret, fail);
    CUDACHECKGOTO(cudaFreeAsync(op->tx, cs), ret, fail);
    CUDACHECKGOTO(cudaFreeAsync(op->tasks, cs), ret, fail);
    op->rx = nullptr;
    op->tx = nullptr;
    op->tasks = nullptr;
    NCCLCHECKGOTO(recordIpcPhase(comm, m->localSlotEvents[0],
        &m->localSync->slots[0], ++m->localSlotPhase[0], cs),ret,fail);
    NCCLCHECKGOTO(record(ready[0],cs),ret,fail);
    NCCLCHECKGOTO(wait(j->stream,ready[0]),ret,fail);
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
  for(int i=0;i<NCCL_COMPRESSED_PIPELINE_DEPTH;i++) {
    if(ready[i]) CUDACHECKIGNORE(cudaEventDestroy(ready[i]));
    if(done[i]) CUDACHECKIGNORE(cudaEventDestroy(done[i]));
    if(sendDone[i]) CUDACHECKIGNORE(cudaEventDestroy(sendDone[i]));
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
