# PTX Tensor Core GEMM Examples

Direct PTX implementation of matrix multiplication using NVIDIA Tensor Cores.

## Files

| File | Description | Target |
|------|-------------|--------|
| `simple.cu` | Basic 16x8 GEMM (fixed size) | sm_80+ (Ampere) |
| `large.cu` | Scalable GEMM (arbitrary size) | sm_80+ (Ampere) |
| `large_tma.cu` | TMA vs cp.async (conditional) | sm_80+ / sm_90+ |

## Features

### simple.cu
- ✅ Fixed-size kernel (16×16 × 16×8 → 16×8)
- ✅ Single warp, single block
- ✅ Demonstrates: `cp.async`, `ldmatrix`, `mma.m16n8k16`
- ✅ Extensive inline comments

### large.cu
- ✅ Scalable to arbitrary M×N×K dimensions
- ✅ Multi-block grid for large matrices
- ✅ K-dimension loop for accumulation
- ✅ Supports M=16n, N=8n, K=16n

### large_tma.cu
- ✅ **Conditional TMA vs cp.async**
- ✅ Runtime GPU architecture detection
- ✅ Template-based dispatch
- ✅ Command-line mode selection
- ✅ Automatic fallback for compatibility

## Quick Start

```bash
# Compile all examples (Ampere/Ada)
make all

# Run basic example
./simple

# Run large matrix example
./large

# Run TMA example (auto-detects GPU)
./large_tma 0    # Force cp.async
./large_tma 1    # Try TMA (falls back on Ampere)

# Or use test script
./test_tma.sh
```

## Memory Copy Methods

### cp.async (Ampere sm_80+)
```cuda
// All threads participate in copying
for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "l"(smem_dst), "l"(gmem_src)
    );
}
```

**Pros**: Wide compatibility (A100, RTX 30xx/40xx)
**Cons**: Moderate bandwidth, all threads involved

### TMA (Hopper sm_90+)
```cuda
// Single thread issues command, hardware broadcasts
if (tid == 0) {
    asm volatile(
        "cp.async.bulk.shared.global [%0], [%1], %2;\n"
        :: "l"(smem_dst), "l"(gmem_src), "n"(bytes)
    );
}
```

**Pros**: Higher bandwidth (~1.5x), less register pressure
**Cons**: Hopper only (H100, H200)

## Build Options

### Basic Build (Ampere)
```bash
make simple      # Fixed-size kernel
make large       # Scalable kernel
make large_tma   # TMA-capable (cp.async fallback)
```

### Hopper Build (TMA Enabled)
```bash
make large_tma_hopper   # Compile with sm_90 + TMA
```

### Custom Build
```bash
# Ampere (A100, RTX 30xx/40xx)
nvcc -arch=sm_80 -O3 large.cu -o large

# Hopper (H100) with TMA
nvcc -arch=sm_90 -O3 -DUSE_TMA=1 large_tma.cu -o large_tma_hopper
```

## Architecture Support

| GPU | Arch | sm_xx | simple | large | TMA |
|-----|------|-------|--------|-------|-----|
| A100 | Ampere | sm_80 | ✅ | ✅ | ❌ (uses cp.async) |
| RTX 4090 | Ada | sm_89 | ✅ | ✅ | ❌ (uses cp.async) |
| H100 | Hopper | sm_90 | ✅ | ✅ | ✅ |
| H200 | Hopper | sm_90 | ✅ | ✅ | ✅ |

## Performance Characteristics

### Tensor Core Throughput
- **FP16**: 312 TFLOPS (A100), 989 TFLOPS (H100)
- **Memory**: ~1.5 TB/s (A100), ~3.0 TB/s (H100)

### Expected Performance
```
Matrix: 64×64 × 64×32 (FP16)
- Ampere (cp.async): ~0.5 TFLOPS
- Hopper (TMA):      ~0.8 TFLOPS (1.6x)
```

## Kernel Architecture

```
┌─────────────────────────────────────────────┐
│  Grid: (M/16, N/8) blocks                  │
│  Each block: 32 threads (1 warp)           │
└─────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │   K Loop (step=16)      │
        │  ┌──────────────────┐   │
        │  │ 1. GMEM → SMEM   │   │  cp.async or TMA
        │  │    (cp.async/TMA)│   │
        │  ├──────────────────┤   │
        │  │ 2. SMEM → REG    │   │  ldmatrix
        │  │    (ldmatrix)    │   │
        │  ├──────────────────┤   │
        │  │ 3. MMA           │   │  mma.m16n8k16
        │  │    (accumulate)  │   │
        │  └──────────────────┘   │
        └─────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │  Store to GMEM          │
        └─────────────────────────┘
```

## Documentation

- **[SCALING_GUIDE.md](SCALING_GUIDE.md)** - How to scale from fixed to arbitrary sizes
- **[TMA_GUIDE.md](TMA_GUIDE.md)** - TMA vs cp.async detailed comparison

## Verification

All kernels validate against CPU reference implementation:
- Tolerance: 1e-2 (FP16 precision)
- Test pattern: Identity matrix × pattern matrix

## Limitations

1. **Size constraints**:
   - M: multiple of 16
   - N: multiple of 8
   - K: multiple of 16

2. **Memory layout**: Row-major only

3. **Data type**: FP16 only (input/output/accumulator)

## Next Steps

1. **FP32 accumulation**: `mma.*.f32.f16.f16.f32`
2. **Double buffering**: Pipeline memory and compute
3. **Warp specialization**: Split load/compute threads
4. **Tensor Descriptors**: Full TMA with 2D addressing
5. **Cluster groups**: Multi-block cooperation (Hopper)

## References

- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide - Tensor Cores](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [cutlass](https://github.com/NVIDIA/cutlass) - Production GEMM library

## License

MIT License - Feel free to use and modify!

