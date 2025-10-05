# Simple PTX Tensor Core Kernel

This example demonstrates the low-level PTX instructions for tensor core operations:
- `ldmatrix.sync` - Load matrices from shared memory into fragment registers
- `mma.sync` - Matrix multiply-accumulate using tensor cores

## Overview

The kernel implements a simple 16x8x16 matrix multiplication (C = A × B) using:
- **A**: 16×16 matrix (FP16)
- **B**: 16×8 matrix (FP16)
- **C**: 16×8 result matrix (FP16)

## Key PTX Instructions

### ldmatrix.sync
```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];
```
- Loads a matrix tile from shared memory into registers
- `m8n8` - matrix shape per instruction
- `x4` - loads 4 registers (for larger tiles)
- `shared.b16` - from shared memory, 16-bit elements

### mma.sync
```ptx
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {D}, {A}, {B}, {C};
```
- Performs D = A × B + C using tensor cores
- `m16n8k16` - output is 16×8, with K=16 inner dimension
- `row.col` - A is row-major, B is column-major layout
- `f16` - all operands are FP16

## Matrix Fragments

For the `m16n8k16` instruction with FP16:
- **A fragment**: 8×16-bit = 4×32-bit registers
- **B fragment**: 4×16-bit = 2×32-bit registers
- **C/D fragment**: 4×16-bit = 2×32-bit registers

These fragments are distributed across the 32 threads of a warp.

## Building

```bash
# Compile (adjust -arch based on your GPU)
make

# For RTX 40xx (Ada):
nvcc -arch=sm_89 -o simple_mma simple.cu

# For RTX 30xx (Ampere):
nvcc -arch=sm_86 -o simple_mma simple.cu

# For A100 (Ampere):
nvcc -arch=sm_80 -o simple_mma simple.cu
```

## Running

```bash
make run
# or
./simple_mma
```

## Expected Output

The example uses an identity matrix for A, so the result should equal matrix B.

## Requirements

- CUDA-capable GPU with Compute Capability 7.0+ (Tensor Cores)
- CUDA Toolkit 11.0+
- For full functionality: Ampere (sm_80+) or newer architecture

## Further Reading

- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C++ Programming Guide - Tensor Cores](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions)

