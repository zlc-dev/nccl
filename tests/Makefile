#
# SPDX-License-Identifier: Apache-2.0
#

NCCLDIR := ..
include $(NCCLDIR)/makefiles/common.mk

BUILDDIR ?= $(abspath ../build)
BINDIR := $(BUILDDIR)/bin
TARGET := $(BINDIR)/compressed_reduce_scatter_perf
SOURCE := compressed_reduce_scatter_identity.cu

.PHONY: all build run clean

all: build
build: $(TARGET)

$(TARGET): $(SOURCE) $(BUILDDIR)/include/nccl.h $(BUILDDIR)/lib/libnccl.so
	@mkdir -p $(BINDIR)
	$(NVCC) $(CXXSTD) -O2 -I$(BUILDDIR)/include -I$(CUDA_INC) \
		$(SOURCE) -L$(BUILDDIR)/lib -L$(CUDA_LIB) -lnccl -lcudart \
		-Xlinker -rpath -Xlinker $(BUILDDIR)/lib -o $@

run: $(TARGET)
	$(TARGET)

clean:
	rm -f $(TARGET)
