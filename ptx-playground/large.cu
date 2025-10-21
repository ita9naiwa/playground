// large.cu - Scalable PTX Tensor Core GEMM using mma.m16n8k16
// Demonstrates: Tiling strategy for arbitrary M, N, K dimensions
// Target: Ampere+ GPUs (sm_80+)

#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void cpu_matmul_reference(const half* A, // [M x K] row-major
                                 const half* B, // [K x N] row-major
                                 half* C, // [M x N] row-major
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
// Scalable PTX Kernel: mma.m16n8k16
// ============================================================================
// Each block computes a 16x8 tile of output
// - Grid size: (M/16, N/8)
// - Block size: 32 threads (1 warp)
// - Loops over K dimension in steps of 16
//
// Supports arbitrary M, N, K (must be multiples of 16, 8, 16 respectively)
__global__ void mma_m16n8k16_large(const half* __restrict__ A,   // [M x K], row-major
                                   const half* __restrict__ B,   // [K x N], row-major
                                   half* __restrict__ C,         // [M x N], row-major
                                   int M, int N, int K)
{
    // ========================================================================
    // Block and Thread Indexing
    // ========================================================================
    const int bx = blockIdx.x;  // Block index in M dimension
    const int by = blockIdx.y;  // Block index in N dimension

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    // Calculate this block's output tile position
    const int tile_m = bx * 16;  // Starting row
    const int tile_n = by * 8;   // Starting column

    // Boundary check
    if (tile_m >= M || tile_n >= N) return;

    // ========================================================================
    // Shared Memory Allocation
    // ========================================================================
    __shared__ __align__(128) half As[16 * 16]; // A tile: 16x16
    __shared__ __align__(128) half Bs[16 * 8];  // B tile: 16x8

    // ========================================================================
    // Accumulator Initialization
    // ========================================================================
    unsigned c_reg[2] = {0u, 0u};  // Accumulator for this tile

    // ========================================================================
    // Loop over K dimension
    // ========================================================================
    for (int k_base = 0; k_base < K; k_base += 16) {
        // ====================================================================
        // Stage 1: Load A and B tiles from GMEM to SMEM using cp.async
        // ====================================================================
        unsigned long long a_smem_ptr = __cvta_generic_to_shared(As);
        unsigned long long b_smem_ptr = __cvta_generic_to_shared(Bs);

        // A tile: [tile_m:tile_m+16, k_base:k_base+16]
        // Global address of A tile start
        const half* A_tile = A + tile_m * K + k_base;
        unsigned long long a_gmem_ptr = (unsigned long long)A_tile;

        // Copy A: 16 rows x 16 cols = 512 bytes = 32 chunks of 16 bytes
        for (int chunk = tid; chunk < 32; chunk += blockDim.x) {
            int row = chunk / 2;        // Which row (0-15)
            int col_chunk = chunk % 2;  // Which half of row (0 or 1)

            // Each row has 16 halves = 32 bytes, split into 2 chunks of 16 bytes
            unsigned long long src = a_gmem_ptr + row * K * sizeof(half) + col_chunk * 16;
            unsigned long long dst = a_smem_ptr + chunk * 16;

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "l"(dst), "l"(src)
            );
        }

        // B tile: [k_base:k_base+16, tile_n:tile_n+8]
        // Global address of B tile start
        const half* B_tile = B + k_base * N + tile_n;
        unsigned long long b_gmem_ptr = (unsigned long long)B_tile;

        // Copy B: 16 rows x 8 cols = 256 bytes = 16 chunks of 16 bytes
        for (int row = tid; row < 16; row += blockDim.x) {
            unsigned long long src = b_gmem_ptr + row * N * sizeof(half);
            unsigned long long dst = b_smem_ptr + row * 16; // 8 halves * 2 bytes = 16 bytes per row

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                :: "l"(dst), "l"(src)
            );
        }

        // Wait for async copies to complete
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // ====================================================================
        // Stage 2: Load fragments from SMEM using ldmatrix
        // ====================================================================
        unsigned long long a_ptr = a_smem_ptr;
        unsigned long long b_ptr = b_smem_ptr;

        // Load A fragment (16x16 -> 4 registers per thread)
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

        // Load B fragment (16x8 -> 2 registers per thread)
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
        // Stage 3: MMA - Accumulate results
        // ====================================================================
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
            : "+r"(c_reg[0]), "+r"(c_reg[1])
            : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
              "r"(b_reg[0]), "r"(b_reg[1])
        );

        __syncthreads();  // Sync before next K iteration
    }

    // ========================================================================
    // Stage 4: Store Results to Global Memory
    // ========================================================================
    half2* c_as_h2 = reinterpret_cast<half2*>(c_reg);
    auto a = __low2half(c_as_h2[0]), b = __high2half(c_as_h2[0]),
         c = __low2half(c_as_h2[1]), d = __high2half(c_as_h2[1]);

    int quad = lane / 4;
    int col_in_quad = lane % 4;

    int r0 = quad;
    int c0 = col_in_quad * 2;

    // Global output position
    int out_row0 = tile_m + r0;
    int out_row1 = tile_m + r0 + 8;
    int out_col  = tile_n + c0;

    // Boundary checks and write
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
// Main - Test with larger matrices
// ============================================================================
int main() {
    // Test with larger matrices (must be multiples of 16, 8, 16)
    const int M = 64;   // Multiple of 16
    const int K = 64;   // Multiple of 16
    const int N = 32;   // Multiple of 8

    std::printf("========== Configuration ==========\n");
    std::printf("Matrix dimensions: M=%d, K=%d, N=%d\n", M, K, N);
    std::printf("Output size: %dx%d\n", M, N);
    std::printf("Grid: (%d, %d) blocks of 32 threads\n\n", M/16, N/8);

    half *hA = new half[M*K];
    half *hB = new half[K*N];
    half *hC = new half[M*N];
    half *hC_ref = new half[M*N];

    // Initialize matrices
    // A = Identity-like (diagonal blocks)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            hA[i*K + j] = __float2half(i==j ? 1.f : 0.f);

    // B = (i+j) for pattern verification
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            hB[i*N + j] = __float2half(float(i + j));

    std::printf("========== Input Matrices (partial) ==========\n");
    print_mat_f16("Matrix A (Identity)", hA, M, K);
    print_mat_f16("Matrix B", hB, K, N);

    // Allocate device memory
    half *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(half));

    cudaMemcpy(dA, hA, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*sizeof(half));

    // Launch kernel
    dim3 grid(M/16, N/8);
    dim3 block(32);

    std::printf("========== Launching Kernel ==========\n");
    std::printf("Grid: (%d, %d, 1), Block: (%d, 1, 1)\n", grid.x, grid.y, block.x);

    mma_m16n8k16_large<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC, dC, M*N*sizeof(half), cudaMemcpyDeviceToHost);

    std::printf("========== PTX Result (partial) ==========\n");
    print_mat_f16("C (PTX)", hC, M, N);

    // CPU reference
    cpu_matmul_reference(hA, hB, hC_ref, M, N, K);

    std::printf("========== CPU Reference (partial) ==========\n");
    print_mat_f16("C (CPU Reference)", hC_ref, M, N);

    // Verification
    std::printf("========== Verification ==========\n");
    bool match = true;
    float max_abs_diff = 0.0f;
    const float tol = 1e-2f;
    int mismatch_count = 0;

    for (int i = 0; i < M*N; ++i) {
        float ptx_val = __half2float(hC[i]);
        float ref_val = __half2float(hC_ref[i]);
        float diff = fabsf(ptx_val - ref_val);

        if (diff > max_abs_diff) max_abs_diff = diff;

        if (diff > tol) {
            match = false;
            mismatch_count++;
            if (mismatch_count <= 10) {  // Print first 10 mismatches
                std::printf("Mismatch at [%d,%d] (idx=%d): CPU=%.2f  PTX=%.2f  (diff=%.3f)\n",
                            i/N, i%N, i, ref_val, ptx_val, diff);
            }
        }
    }

    if (match) {
        std::printf("[OK] All %d elements match! (max abs diff: %.6f)\n", M*N, max_abs_diff);
    } else {
        std::printf("[FAIL] %d/%d elements do NOT match. (max abs diff: %.6f)\n",
                    mismatch_count, M*N, max_abs_diff);
    }

    delete[] hA; delete[] hB; delete[] hC; delete[] hC_ref;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return match ? 0 : 1;
}

