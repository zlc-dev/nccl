#
# SPDX-License-Identifier: Apache-2.0
#

NCCLDIR := ..
include $(NCCLDIR)/makefiles/common.mk

BUILDDIR ?= $(abspath ../build)
BINDIR := $(BUILDDIR)/bin
TARGETS := $(BINDIR)/compressed_reduce_scatter_perf \
	$(BINDIR)/reduce_scatter_perf

.PHONY: all build run clean

all: build
build: $(TARGETS)

$(BINDIR)/compressed_reduce_scatter_perf: compressed_reduce_scatter_identity.cu $(BUILDDIR)/include/nccl.h $(BUILDDIR)/lib/libnccl.so
	@mkdir -p $(BINDIR)
	$(NVCC) $(CXXSTD) -O2 -I$(BUILDDIR)/include -I$(CUDA_INC) \
		$< -L$(BUILDDIR)/lib -L$(CUDA_LIB) -lnccl -lcudart \
		-Xlinker -rpath -Xlinker $(BUILDDIR)/lib -o $@

$(BINDIR)/reduce_scatter_perf: reduce_scatter.cu $(BUILDDIR)/include/nccl.h $(BUILDDIR)/lib/libnccl.so
	@mkdir -p $(BINDIR)
	$(NVCC) $(CXXSTD) -O2 -I$(BUILDDIR)/include -I$(CUDA_INC) \
		$< -L$(BUILDDIR)/lib -L$(CUDA_LIB) -lnccl -lcudart \
		-Xlinker -rpath -Xlinker $(BUILDDIR)/lib -o $@

run: $(TARGETS)
	$(BINDIR)/compressed_reduce_scatter_perf
	$(BINDIR)/reduce_scatter_perf

clean:
	rm -f $(TARGETS)
