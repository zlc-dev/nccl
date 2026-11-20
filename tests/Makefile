NVCC ?= nvcc
CUDA_ARCH ?= sm_86
NCCL_HOME ?= $(abspath ../build)
TARGET := build/zipccl_reduce_scatter_perf

.PHONY: all zipccl_reduce_scatter_perf
all: $(TARGET)
zipccl_reduce_scatter_perf: $(TARGET)

$(TARGET): zipccl_reduce_scatter_perf.cu $(NCCL_HOME)/include/nccl.h $(NCCL_HOME)/lib/libnccl.so
	mkdir -p $(@D)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -I$(NCCL_HOME)/include $< \
	  -L$(NCCL_HOME)/lib -Xlinker -rpath=$(NCCL_HOME)/lib -lnccl -o $@
