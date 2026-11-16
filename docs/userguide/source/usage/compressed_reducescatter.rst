Compressed ReduceScatter
========================

``ncclReduceScatterCompressed`` is an experimental, Host-orchestrated Ring
ReduceScatter for multiple processes controlling GPUs on one node. It does
not enqueue NCCL's regular ReduceScatter kernel and does not use the SIMPLE
protocol FIFO for compressed payloads.

Codec interface
---------------

Initialize ``ncclCompressionPlan_t`` with
``NCCL_COMPRESSION_PLAN_INITIALIZER``. ``blockElements`` is an element count,
not a byte count. One callback invocation always processes exactly one
contiguous block (or the final short tail block).

.. code-block:: c

   ncclCompressionPlan_t plan = NCCL_COMPRESSION_PLAN_INITIALIZER;
   plan.blockElements = block_elements;
   plan.context = codec_context;
   plan.maxCompressedBytes = query_worst_case_bytes;
   plan.encode = launch_encode;
   plan.decodeReduceEncode = launch_decode_reduce_encode;
   plan.decodeReduceCopy = launch_decode_reduce_copy;

``maxCompressedBytes`` is a Host callback. Given ``blockElements`` and the
datatype, it returns the codec's worst-case compressed byte count. The bound
must also cover every shorter tail block. NCCL uses it only to size two TX and
two RX slots; the wire transfer uses the actual
``task.outputHeader->payloadBytes`` written by the codec.

Each codec callback receives one device-resident ``ncclCompressionTask_t`` and
enqueues its kernel on the supplied stream. It must return without synchronizing
the GPU and must not retain the task pointer. ``compressedInput`` and
``compressedOutput`` point directly into the RX and TX P2P slots, so no payload
workspace or payload copy is inserted around a codec invocation.

If communicator abort is observed while waiting for a frame, NCCL presents an
empty task with ``elementCount == 0``. Codec callbacks must skip such a task.

Partitioning
------------

The input has the normal rank-major layout of ``nranks * recvcount`` elements.
Each destination rank owns one contiguous ``recvcount``-element interval. That
interval is divided only along the element dimension:

``blockOffset = blockIndex * blockElements``

``elementCount = min(blockElements, recvcount - blockOffset)``

There is no channel subdivision and no aggregation of several small tasks into
one codec launch. For example, with FP32 and ``blockElements=262144``, every full
codec invocation consumes 1 MiB of uncompressed input. The implementation
rejects calls whose configured full block or complete rank interval is below
512 KiB; small-message behavior is intentionally out of scope.

Pipeline and transport
----------------------

Each GPU has two TX slots and two RX slots. A slot's payload capacity is the
codec's worst-case size for one full block. For each two-block window:

#. ``encode`` writes local source data directly into TX slot 0, then TX slot 1.
#. A lightweight communication CTA copies only each slot's actual header and
   ``payloadBytes`` into the next rank's RX slot using direct CUDA P2P access.
#. ``decodeReduceEncode`` reads an RX slot and local original input, then writes
   the next compressed result directly into the corresponding TX slot.
#. The final Ring round uses ``decodeReduceCopy`` to write the local result
   directly to ``recvbuff``.

GPU-visible slot states provide credit and readiness. The sender waits for a
free remote RX slot, copies the variable-length frame, performs a system fence,
and marks it ready. The receiving stream waits for ready, launches the codec,
then marks the slot free. Compressed lengths never return to the CPU. CUDA
events order the communication stream and codec stream, allowing communication
on one slot to overlap codec work on the other.

The communicator owns an IPC-capable CUDA memory pool connection and a small,
capacity-independent control page. Every invocation allocates its own TX/RX
slots and task descriptors from that pool and exchanges only the exported RX
allocation metadata. Codec identity, capacity, layout, and other metadata are
assumed to be identical on every rank and are not negotiated or validated.

The API returns after work has been enqueued and never synchronizes the user
stream. Cleanup is part of the enqueued operation: after the final RX-slot
release, the communication stream frees the imported RX pointer, signals its
owner through the control page, waits until the previous rank has freed this
rank's exported pointer, and then frees the local RX, TX, and task allocations.
The user-stream completion point follows these stream-ordered frees. There is
no background cleanup thread, and a capacity change does not rebuild the
connection. An internal tail event orders consecutive invocations, including
invocations submitted on different CUDA streams. No resource handle or
explicit context is exposed to the caller.

Restrictions
------------

* All ranks must be on one node. Separate rank processes are supported.
* Direct CUDA peer access must exist along channel 0's Ring edges.
* Only ``ncclSum`` is supported.
* The codec's worst-case compressed size must fit in ``uint32_t``.
* CUDA memory-pool IPC with POSIX file-descriptor handles is required.
* Codec configuration and capacity may change between non-overlapping calls;
  every rank is required to use matching values without runtime negotiation.
* CUDA graph capture is unsupported. ``ncclGroupStart``/``ncclGroupEnd`` defer
  setup and launch until group completion like other NCCL operations.
* Calls on one rank are internally serialized by the connection tail event.
  Concurrent submission from multiple Host threads is unsupported. Every rank
  must enter calls in the same order.
