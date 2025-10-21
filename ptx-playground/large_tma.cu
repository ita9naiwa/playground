// large_tma.cu - PTX Tensor Core GEMM with TMA (Hopper) or cp.async (Ampere)
// Demonstrates: Conditional TMA vs cp.async based on architecture
// Target: Ampere+ (sm_80+), TMA requires Hopper+ (sm_90+)

#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Compile-time or runtime selection
#define USE_TMA 1
#ifndef USE_TMA
#define USE_TMA 0  // Default: use cp.async (compatible with sm_80+)
#endif

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void cpu_matmul_reference(const half* A, const half* B, half* C,
                                 int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
            }
            C[m * N + n] = __float2half(sum);
        }
    }
}

static void print_mat_f16(const char* name, const half* h, int M, int N, int max_print = 16) {
    std::printf("%s (%dx%d):\n", name, M, N);
    int print_m = (M < max_print) ? M : max_print;
    int print_n = (N < max_print) ? N : max_print;

    for (int i = 0; i < print_m; ++i) {
        for (int j = 0; j < print_n; ++j)
            std::printf("%.1f ", __half2float(h[i*N + j]));
        if (print_n < N) std::printf("...");
        std::printf("\n");
    }
    if (print_m < M) std::printf("...\n");
    std::printf("\n");
}

// ============================================================================
// Memory Copy Functions
// ============================================================================

// cp.async: Cooperative copy using all threads (Ampere sm_80+)
__device__ __forceinline__ void load_tile_async(
    half* smem_dst,
    const half* gmem_src,
    int tile_rows,
    int tile_cols,
    int gmem_stride,
    int tid)
{
    unsigned long long smem_ptr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_ptr = (unsigned long long)gmem_src;

    // Calculate total chunks (16-byte aligned)
    int total_bytes = tile_rows * tile_cols * sizeof(half);
    int num_chunks = total_bytes / 16;

    // Each thread copies multiple 16-byte chunks cooperatively
    for (int chunk = tid; chunk < num_chunks; chunk += blockDim.x) {
        int element_idx = chunk * 8;  // 8 halves per 16-byte chunk
        int row = element_idx / tile_cols;
        int col = element_idx % tile_cols;

        if (row < tile_rows) {
            unsigned long long src = gmem_ptr + (row * gmem_stride + col) * sizeof(half);
            unsigned long long dst = smem_ptr + element_idx * sizeof(half);

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "l"(dst), "l"(src)
            );
        }
    }
}

#if USE_TMA
// TMA: Bulk copy using single thread (Hopper sm_90+)
__device__ __forceinline__ void load_tile_tma(
    half* smem_dst,
    const half* gmem_src,
    int tile_rows,
    int tile_cols,
    int gmem_stride,
    int tid)
{
    // Only thread 0 issues TMA command (hardware handles distribution)
    if (tid == 0) {
        unsigned long long smem_ptr = __cvta_generic_to_shared(smem_dst);
        unsigned long long gmem_ptr = (unsigned long long)gmem_src;
        int bytes = tile_rows * tile_cols * sizeof(half);

        // TMA bulk copy with L2 cache hint
        // NOTE: This is a SIMPLIFIED version using raw memory addresses
        //
        // Production code should use Tensor Descriptors (cuTensorMapEncodeTiled):
        //   - Enables 2D tile coordinates instead of byte addresses
        //   - Hardware-optimized stride handling
        //   - Better cache control and swizzling
        //   - See tma_descriptor_example.cu for full implementation
        //
        // Current: cp.async.bulk with raw pointers (works but suboptimal)
        // Optimal: cp.async.bulk.tensor.2d with CUtensorMap
        asm volatile(
            "cp.async.bulk.shared.global.L2::cache_hint [%0], [%1], %2;\n"
            :: "l"(smem_ptr), "l"(gmem_ptr), "n"(bytes)
        );
    }
}
#endif

// ============================================================================
// Unified Kernel: TMA (Hopper sm_90+) or cp.async (Ampere sm_80+)
// ============================================================================
// Template parameter UseTMA selects memory copy method at compile time
template<bool UseTMA>
__global__ void mma_m16n8k16_unified(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if (tid >= 32) return;

    const int tile_m = bx * 16;
    const int tile_n = by * 8;

    if (tile_m >= M || tile_n >= N) return;

    __shared__ __align__(128) half As[16 * 16];
    __shared__ __align__(128) half Bs[16 * 8];

    unsigned c_reg[2] = {0u, 0u};

    // ========================================================================
    // K Loop with conditional TMA/cp.async
    // ========================================================================
    for (int k_base = 0; k_base < K; k_base += 16) {
        // ====================================================================
        // Stage 1: Load tiles - TMA or cp.async
        // ====================================================================
        const half* A_tile = A + tile_m * K + k_base;
        const half* B_tile = B + k_base * N + tile_n;

        if constexpr (UseTMA) {
#if USE_TMA
            // ================================================================
            // TMA Path (Hopper sm_90+)
            // ================================================================
            load_tile_tma(As, A_tile, 16, 16, K, tid);
            load_tile_tma(Bs, B_tile, 16, 8, N, tid);

            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
            asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
#else
            static_assert(!UseTMA || __CUDA_ARCH__ >= 900,
                          "TMA requires sm_90+. Compile with -arch=sm_90 -DUSE_TMA=1");
#endif
        } else {
            // ================================================================
            // cp.async Path (Ampere sm_80+)
            // ================================================================
            load_tile_async(As, A_tile, 16, 16, K, tid);
            load_tile_async(Bs, B_tile, 16, 8, N, tid);

            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }

        __syncthreads();

        // ====================================================================
        // Stage 2: ldmatrix (same for both paths)
        // ====================================================================
        unsigned long long a_ptr = __cvta_generic_to_shared(As);
        unsigned long long b_ptr = __cvta_generic_to_shared(Bs);

        unsigned a_reg[4];
        int a_quad = lane >> 3;
        int a_row  = lane & 7;
        int a_col_block = (a_quad & 1) * 8;
        int a_row_block = (a_quad >> 1) * 8;

        unsigned long long a_addr =
            a_ptr + (unsigned long long)((a_row_block + a_row) * 16 + a_col_block) * sizeof(half);

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
            "{%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
            : "l"(a_addr)
        );

        unsigned b_reg[2];
        int b_quad = lane >> 3;
        int b_row  = lane & 7;
        int b_k_block = (b_quad & 1) * 8;

        unsigned long long b_addr =
            b_ptr + (unsigned long long)((b_k_block + b_row) * 8) * sizeof(half);

        if (b_quad > 1) {
            int lower = lane & 15;
            int lg = lower >> 3;
            int lr = lower & 7;
            int lrBlk = (lg & 1) * 8;
            b_addr = b_ptr + (unsigned long long)((lrBlk + lr) * 8) * sizeof(half);
        }

        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
            "{%0,%1}, [%2];\n"
            : "=r"(b_reg[0]), "=r"(b_reg[1])
            : "l"(b_addr)
        );

        // ====================================================================
        // Stage 3: MMA (same for both paths)
        // ====================================================================
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
            : "+r"(c_reg[0]), "+r"(c_reg[1])
            : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
              "r"(b_reg[0]), "r"(b_reg[1])
        );

        __syncthreads();
    }

    // ========================================================================
    // Stage 4: Store (same for both paths)
    // ========================================================================
    half2* c_as_h2 = reinterpret_cast<half2*>(c_reg);
    auto a = __low2half(c_as_h2[0]), b = __high2half(c_as_h2[0]),
         c = __low2half(c_as_h2[1]), d = __high2half(c_as_h2[1]);

    int quad = lane / 4;
    int col_in_quad = lane % 4;
    int r0 = quad;
    int c0 = col_in_quad * 2;

    int out_row0 = tile_m + r0;
    int out_row1 = tile_m + r0 + 8;
    int out_col  = tile_n + c0;

    if (out_row0 < M && out_col < N) {
        C[out_row0 * N + out_col] = a;
        if (out_col + 1 < N) {
            C[out_row0 * N + out_col + 1] = b;
        }
    }

    if (out_row1 < M && out_col < N) {
        C[out_row1 * N + out_col] = c;
        if (out_col + 1 < N) {
            C[out_row1 * N + out_col + 1] = d;
        }
    }
}

// ============================================================================
// Runtime Dispatch: Select TMA or cp.async based on user choice
// ============================================================================
void launch_gemm(const half* dA, const half* dB, half* dC,
                 int M, int N, int K, bool use_tma)
{
    dim3 grid(M/16, N/8);
    dim3 block(32);

    if (use_tma) {
#if USE_TMA
        std::printf("Launching kernel with TMA (Hopper sm_90+)\n");
        mma_m16n8k16_unified<true><<<grid, block>>>(dA, dB, dC, M, N, K);
#else
        std::printf("ERROR: TMA not enabled at compile time!\n");
        std::printf("       Rebuild with: nvcc -arch=sm_90 -DUSE_TMA=1 ...\n");
#endif
    } else {
        std::printf("Launching kernel with cp.async (Ampere sm_80+)\n");
        mma_m16n8k16_unified<false><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
}

// ============================================================================
// Main: Benchmark TMA vs cp.async
// ============================================================================
int main(int argc, char** argv) {
    // Command line: ./large_tma [0=cp.async | 1=TMA]
    bool use_tma = false;
    if (argc > 1) {
        use_tma = (std::atoi(argv[1]) != 0);
    }

    // Detect GPU architecture and validate TMA support
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int sm_version = prop.major * 10 + prop.minor;

    std::printf("========== GPU Information ==========\n");
    std::printf("Device: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d (sm_%d)\n",
                prop.major, prop.minor, sm_version);

    if (use_tma && sm_version < 90) {
        std::printf("WARNING: TMA requires Hopper (sm_90+)\n");
        std::printf("         Falling back to cp.async\n");
        use_tma = false;
    }
    std::printf("\n");

    const int M = 64;
    const int K = 64;
    const int N = 32;

    std::printf("========== Configuration ==========\n");
    std::printf("Matrix dimensions: M=%d, K=%d, N=%d\n", M, K, N);
    std::printf("Memory path: %s\n", use_tma ? "TMA" : "cp.async");
    std::printf("Grid: (%d, %d) blocks of 32 threads\n\n", M/16, N/8);

    half *hA = new half[M*K];
    half *hB = new half[K*N];
    half *hC = new half[M*N];
    half *hC_ref = new half[M*N];

    // Initialize: A = Identity, B = (i+j)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            hA[i*K + j] = __float2half(i==j ? 1.f : 0.f);

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            hB[i*N + j] = __float2half(float(i + j));

    half *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(half));

    cudaMemcpy(dA, hA, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*sizeof(half));

    // Launch with selected method
    launch_gemm(dA, dB, dC, M, N, K, use_tma);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC, dC, M*N*sizeof(half), cudaMemcpyDeviceToHost);

    std::printf("========== PTX Result (partial) ==========\n");
    print_mat_f16("C (GPU)", hC, M, N);

    cpu_matmul_reference(hA, hB, hC_ref, M, N, K);

    // Verification
    std::printf("========== Verification ==========\n");
    bool match = true;
    float max_abs_diff = 0.0f;
    const float tol = 1e-2f;
    int mismatch_count = 0;

    for (int i = 0; i < M*N; ++i) {
        float gpu_val = __half2float(hC[i]);
        float ref_val = __half2float(hC_ref[i]);
        float diff = fabsf(gpu_val - ref_val);

        if (diff > max_abs_diff) max_abs_diff = diff;

        if (diff > tol) {
            match = false;
            mismatch_count++;
            if (mismatch_count <= 10) {
                std::printf("Mismatch at [%d,%d]: CPU=%.2f  GPU=%.2f  (diff=%.3f)\n",
                            i/N, i%N, ref_val, gpu_val, diff);
            }
        }
    }

    if (match) {
        std::printf("[OK] All %d elements match! (max abs diff: %.6f)\n",
                    M*N, max_abs_diff);
    } else {
        std::printf("[FAIL] %d/%d elements mismatch. (max abs diff: %.6f)\n",
                    mismatch_count, M*N, max_abs_diff);
    }

    delete[] hA; delete[] hB; delete[] hC; delete[] hC_ref;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return match ? 0 : 1;
}

