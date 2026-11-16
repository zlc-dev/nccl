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
#include "strongstream.h"
#include "transport.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

struct ncclCompressedRsManager {
  std::mutex mutex;
  std::thread cleanup;
  std::atomic<bool> active{false};
  uint64_t sequence = 0;
  ncclResult_t cleanupResult = ncclSuccess;
};

namespace {
constexpr size_t MinBlockBytes = 512 * 1024;
constexpr size_t SlotAlignment = 256;
constexpr size_t PayloadOffset = (sizeof(ncclCompressedP2pSlot) + 15) & ~size_t(15);
constexpr int CleanupTagBase = 0x43520000;

struct Operation {
  ncclComm* comm = nullptr;
  ncclCompressedRsManager* manager = nullptr;
  void* tx = nullptr;
  void* rx = nullptr;
  void* remoteRx = nullptr;
  ncclCompressionTask_t* tasks = nullptr;
  cudaEvent_t complete = nullptr;
  size_t stride = 0;
  uint64_t sequence = 0;
  bool exchanged = false;
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

static size_t alignUp(size_t n, size_t a) { return (n + a - 1) / a * a; }
static char* slot(void* arena, size_t stride, int i) {
  return static_cast<char*>(arena) + stride * static_cast<size_t>(i);
}
static ncclResult_t record(cudaEvent_t e, cudaStream_t s) {
  CUDACHECK(cudaEventRecord(e, s));
  return ncclSuccess;
}
static ncclResult_t wait(cudaStream_t s, cudaEvent_t e) {
  CUDACHECK(cudaStreamWaitEvent(s, e, 0));
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
  std::thread finished;
  {
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->cleanup.joinable()) finished = std::move(m->cleanup);
  }
  if (finished.joinable()) finished.join();
  std::lock_guard<std::mutex> lock(m->mutex);
  if (m->active.load(std::memory_order_acquire)) return ncclInvalidUsage;
  if (m->cleanupResult != ncclSuccess) return m->cleanupResult;
  m->active.store(true, std::memory_order_release);
  op->comm = comm;
  op->manager = m;
  op->sequence = m->sequence++;
  return ncclSuccess;
}

static ncclResult_t release(Operation* op, bool waitGpu) {
  ncclResult_t ret = ncclSuccess;
  int savedDev = -1;
  CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, done);
  CUDACHECKGOTO(cudaSetDevice(op->comm->cudaDev), ret, done);
  if (waitGpu && op->complete) CUDACHECKGOTO(cudaEventSynchronize(op->complete), ret, done);
  if (op->remoteRx) {
    CUDACHECKGOTO(cudaIpcCloseMemHandle(op->remoteRx), ret, done);
    op->remoteRx = nullptr;
  }
  if (op->exchanged && op->comm->nRanks > 1) {
    uint64_t ack = op->sequence;
    int tag = CleanupTagBase | static_cast<int>(op->sequence & 0xffff);
    NCCLCHECKGOTO(bootstrapSend(op->comm->bootstrap, op->comm->channels[0].ring.next,
                                tag, &ack, sizeof(ack)), ret, done);
    NCCLCHECKGOTO(bootstrapRecv(op->comm->bootstrap, op->comm->channels[0].ring.prev,
                                tag, &ack, sizeof(ack)), ret, done);
  }
done:
  if (op->complete) CUDACHECKIGNORE(cudaEventDestroy(op->complete));
  if (op->tasks) CUDACHECKIGNORE(cudaFree(op->tasks));
  if (op->tx) CUDACHECKIGNORE(cudaFree(op->tx));
  if (op->rx) CUDACHECKIGNORE(cudaFree(op->rx));
  if (savedDev >= 0) CUDACHECKIGNORE(cudaSetDevice(savedDev));
  return ret;
}

static void cleanup(Operation* op) {
  ncclResult_t ret = release(op, true);
  ncclCompressedRsManager* m = op->manager;
  ncclComm* comm = op->comm;
  delete op;
  if (ret != ncclSuccess) (void)ncclCommSetAsyncError(comm, ret);
  std::lock_guard<std::mutex> lock(m->mutex);
  m->cleanupResult = ret;
  m->active.store(false, std::memory_order_release);
}

static ncclResult_t allocateExchange(Operation* op, size_t capacity) {
  ncclResult_t ret = ncclSuccess;
  ncclComm* comm = op->comm;
  int next = comm->channels[0].ring.next;
  int access = 0;
  op->stride = alignUp(PayloadOffset + capacity, SlotAlignment);
  if (op->stride > SIZE_MAX / NCCL_COMPRESSED_PIPELINE_DEPTH) return ncclInvalidArgument;
  size_t bytes = op->stride * NCCL_COMPRESSED_PIPELINE_DEPTH;
  std::vector<cudaIpcMemHandle_t> handles(comm->nRanks);

  CUDACHECKGOTO(cudaDeviceCanAccessPeer(&access, comm->cudaDev, comm->peerInfo[next].cudaDev), ret, fail);
  if (!access) {
    WARN("ncclReduceScatterCompressed: no direct P2P access from rank %d to %d", comm->rank, next);
    return ncclInvalidUsage;
  }
  CUDACHECKGOTO(cudaMalloc(&op->tx, bytes), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&op->rx, bytes), ret, fail);
  CUDACHECKGOTO(cudaMalloc(&op->tasks, sizeof(*op->tasks) * NCCL_COMPRESSED_PIPELINE_DEPTH), ret, fail);
  CUDACHECKGOTO(cudaMemset(op->rx, 0, bytes), ret, fail);
  CUDACHECKGOTO(cudaIpcGetMemHandle(&handles[comm->rank], op->rx), ret, fail);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, handles.data(), sizeof(cudaIpcMemHandle_t)), ret, fail);
  CUDACHECKGOTO(cudaIpcOpenMemHandle(&op->remoteRx, handles[next],
                                     cudaIpcMemLazyEnablePeerAccess), ret, fail);
  op->exchanged = true;
fail:
  return ret;
}

static ncclResult_t launch(Job* j, Operation* op, size_t capacity, int typeSize) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t cs = nullptr;
  cudaEvent_t ready[2] = {}, done[2] = {};
  bool acquired = false;
  ncclComm* comm = j->comm;
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream, false, &cs), ret, fail);
  acquired = true;
  for (int i = 0; i < 2; i++) {
    CUDACHECKGOTO(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming), ret, fail);
    CUDACHECKGOTO(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming), ret, fail);
  }
  CUDACHECKGOTO(cudaEventCreateWithFlags(&op->complete, cudaEventDisableTiming), ret, fail);
  {
    size_t nblocks = 1 + (j->recvcount - 1) / j->plan.blockElements;
    NCCLCHECKGOTO(record(done[0], j->stream), ret, fail);
    NCCLCHECKGOTO(wait(cs, done[0]), ret, fail);
    for (size_t w = 0; w < nblocks; w += 2) {
      int slots = static_cast<int>(std::min<size_t>(2, nblocks - w));
      for (int s = 0; s < slots; s++) {
        size_t off = (w+s)*j->plan.blockElements;
        size_t count = std::min(j->plan.blockElements, j->recvcount-off);
        int dst = comm->channels[0].ring.userRanks[comm->nRanks-1];
        NCCLCHECKGOTO(ncclLaunchCompressedPrepareEncode(op->tasks+s, slot(op->tx,op->stride,s),
          j->sendbuff,j->recvcount,off,count,dst,typeSize,capacity,cs), ret, fail);
        NCCLCHECKGOTO(record(ready[s],cs),ret,fail); NCCLCHECKGOTO(wait(j->stream,ready[s]),ret,fail);
        NCCLCHECKGOTO(j->plan.encode(op->tasks+s,j->datatype,j->plan.context,j->stream),ret,fail);
        NCCLCHECKGOTO(record(done[s],j->stream),ret,fail);
      }
      for (int s=0;s<slots;s++) {
        NCCLCHECKGOTO(wait(cs,done[s]),ret,fail);
        NCCLCHECKGOTO(ncclLaunchCompressedSend(slot(op->tx,op->stride,s),
          slot(op->remoteRx,op->stride,s),capacity,comm->abortFlagDev,cs),ret,fail);
      }
      for (int r=0;r<comm->nRanks-1;r++) {
        bool final = r+1 == comm->nRanks-1;
        int dst = comm->channels[0].ring.userRanks[comm->nRanks-2-r];
        for (int s=0;s<slots;s++) {
          size_t off=(w+s)*j->plan.blockElements;
          size_t count=std::min(j->plan.blockElements,j->recvcount-off);
          NCCLCHECKGOTO(ncclLaunchCompressedPrepareRound(op->tasks+s,slot(op->rx,op->stride,s),
            slot(op->tx,op->stride,s),!final,j->sendbuff,final?j->recvbuff:nullptr,j->recvcount,
            off,count,dst,typeSize,capacity,comm->abortFlagDev,cs),ret,fail);
          NCCLCHECKGOTO(record(ready[s],cs),ret,fail);
        }
        for (int s=0;s<slots;s++) {
          NCCLCHECKGOTO(wait(j->stream,ready[s]),ret,fail);
          if (final) NCCLCHECKGOTO(j->plan.decodeReduceCopy(op->tasks+s,j->datatype,j->op,
              j->plan.context,j->stream),ret,fail);
          else NCCLCHECKGOTO(j->plan.decodeReduceEncode(op->tasks+s,j->datatype,j->op,
              j->plan.context,j->stream),ret,fail);
          NCCLCHECKGOTO(record(done[s],j->stream),ret,fail);
        }
        for (int s=0;s<slots;s++) {
          NCCLCHECKGOTO(wait(cs,done[s]),ret,fail);
          NCCLCHECKGOTO(ncclLaunchCompressedRelease(slot(op->rx,op->stride,s),comm->abortFlagDev,cs),ret,fail);
          if (!final) NCCLCHECKGOTO(ncclLaunchCompressedSend(slot(op->tx,op->stride,s),
            slot(op->remoteRx,op->stride,s),capacity,comm->abortFlagDev,cs),ret,fail);
        }
      }
    }
    NCCLCHECKGOTO(record(ready[0],cs),ret,fail);
    NCCLCHECKGOTO(wait(j->stream,ready[0]),ret,fail);
    NCCLCHECKGOTO(record(op->complete,j->stream),ret,fail);
  }
fail:
  if (acquired) {
    ncclResult_t x=ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
      &comm->sharedRes->deviceStream,false);
    if (ret==ncclSuccess) ret=x;
  }
  for(int i=0;i<2;i++) {
    if(ready[i]) CUDACHECKIGNORE(cudaEventDestroy(ready[i]));
    if(done[i]) CUDACHECKIGNORE(cudaEventDestroy(done[i]));
  }
  return ret;
}

static ncclResult_t jobMain(ncclAsyncJob* base) {
  Job* j=reinterpret_cast<Job*>(base);
  ncclResult_t ret=ncclSuccess;
  Operation* op=new(std::nothrow) Operation;
  size_t capacity=0;
  if(!op) return ncclSystemError;
  NCCLCHECKGOTO(claim(j->comm,op),ret,fail);
  NCCLCHECKGOTO(j->plan.maxCompressedBytes(j->plan.blockElements,j->datatype,
    j->plan.context,&capacity),ret,fail);
  if(capacity==0 || capacity>UINT32_MAX || capacity>SIZE_MAX-PayloadOffset-SlotAlignment) {
    ret=ncclInvalidArgument; goto fail;
  }
  NCCLCHECKGOTO(allocateExchange(op,capacity),ret,fail);
  NCCLCHECKGOTO(launch(j,op,capacity,ncclTypeSize(j->datatype)),ret,fail);
  {
    std::lock_guard<std::mutex> lock(op->manager->mutex);
    op->manager->cleanup=std::thread(cleanup,op);
  }
  return ncclSuccess;
fail:
  if(op->manager) {
    (void)release(op,false);
    op->manager->active.store(false,std::memory_order_release);
  }
  delete op;
  return ret;
}
static void jobFree(void* p) { delete reinterpret_cast<Job*>(p); }
} // namespace

ncclResult_t ncclCompressedRsDestroy(ncclComm* comm) {
  ncclCompressedRsManager* m=comm->compressedRsManager;
  if(!m) return ncclSuccess;
  if(m->cleanup.joinable()) m->cleanup.join();
  ncclResult_t ret=m->cleanupResult;
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
