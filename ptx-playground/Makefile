# Makefile for PTX Tensor Core GEMM examples
# Target: Ampere+ GPUs (sm_80+), Hopper (sm_90+) for TMA

NVCC = nvcc
NVCC_FLAGS_SM80 = -arch=sm_80 -O3 --ptxas-options=-v
NVCC_FLAGS_SM90 = -arch=sm_90 -O3 --ptxas-options=-v

# Targets
TARGETS = simple large large_tma large_tma_hopper

all: $(TARGETS)

simple: simple.cu
	$(NVCC) $(NVCC_FLAGS_SM80) simple.cu -o simple

large: large.cu
	$(NVCC) $(NVCC_FLAGS_SM80) large.cu -o large

# TMA version with cp.async fallback (runs on sm_80+)
large_tma: large_tma.cu
	$(NVCC) $(NVCC_FLAGS_SM80) -DUSE_TMA=0 large_tma.cu -o large_tma

# TMA version for Hopper (requires sm_90+)
large_tma_hopper: large_tma.cu
	$(NVCC) $(NVCC_FLAGS_SM90) -DUSE_TMA=1 large_tma.cu -o large_tma_hopper

# TMA with Tensor Descriptors (requires CUDA 12.0+, Hopper)
tma_descriptor: tma_descriptor_example.cu
	$(NVCC) $(NVCC_FLAGS_SM90) -lcuda tma_descriptor_example.cu -o tma_descriptor

clean:
	rm -f $(TARGETS)

run_simple: simple
	./simple

run_large: large
	./large

run_tma: large_tma
	@echo "=== Running with cp.async (Ampere) ==="
	./large_tma 0
	@echo ""
	@echo "=== Attempting TMA (will fallback if not Hopper) ==="
	./large_tma 1

run_tma_hopper: large_tma_hopper
	@echo "=== Running with TMA on Hopper ==="
	./large_tma_hopper 1

.PHONY: all clean run_simple run_large run_tma run_tma_hopper
