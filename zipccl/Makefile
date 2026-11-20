NVCC ?= nvcc
CUDA_ARCH ?= sm_89
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo --extra-device-vectorization

TARGETS := dre_benchmark reduce_benchmark
HEADERS := zipccl.cuh frame.cuh single_cta.cuh
BENCH_ELEMENTS ?= 131072
BENCH_ITERATIONS ?= 5000

.PHONY: all run run-reduce clean

all: $(TARGETS)

dre_benchmark: dre_benchmark.cu zipccl.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -I. dre_benchmark.cu zipccl.cu -o build/$@

reduce_benchmark: reduce_benchmark.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) reduce_benchmark.cu -o build/$@

run: dre_benchmark
	./build/dre_benchmark $(BENCH_ELEMENTS) $(BENCH_ITERATIONS) 0.02

run-reduce: reduce_benchmark
	./build/reduce_benchmark $(BENCH_ELEMENTS) $(BENCH_ITERATIONS) 0.02

clean:
	rm -f $(addprefix build/,$(TARGETS))
