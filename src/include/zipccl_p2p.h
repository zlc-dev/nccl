#pragma once

// ncclConnInfo::flags, set only on a directly mapped ZipCCL send FIFO.
constexpr int ncclZipFixedSlots = 0x08;
// Set on a ZipCCL SHM receive FIFO when the receiving CTA stages frames in VRAM.
constexpr int ncclZipStageRecv = 0x10;
