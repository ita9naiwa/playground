// simple.cu - Direct PTX Tensor Core GEMM using mma.m16n8k16
// Demonstrates: cp.async, ldmatrix, mma instruction, 8-row interleaved output
// Target: Ampere+ GPUs (sm_80+)

#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void cpu_matmul_reference(const half* A, // [M x K] row-major
                                 const half* B, // [K x 16] row-major
                                 half* C, // [M x N] row-major
                                 int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[m * K + k]) * __half2float(B[k * 16 + n]);
            }
            C[m * N + n] = __float2half(sum);
        }
    }
}

static void print_mat_f16(const char* name, const half* h, int M, int N) {
    std::printf("%s (%dx%d):\n", name, M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            std::printf("%.1f ", __half2float(h[i*N + j]));
        std::printf("\n");
    }
    std::printf("\n");
}

// ============================================================================
// PTX Kernel: mma.m16n8k16.row.col.f16
// ============================================================================
// Direct PTX implementation using Tensor Core mma instruction
// Computes: C[16x8] = A[16x16] * B[16x8]
//
// Memory Layout:
// - A: row-major in both GMEM and SMEM
// - B: row-major in both GMEM and SMEM
// - C: row-major in GMEM
//
// MMA Instruction Format:
// - .row: A operand is in row-major layout in registers
// - .col: B operand is in column-major layout in registers
// - We use ldmatrix.trans to transpose data when loading into registers
__global__ void mma_m16n8k16_ptx(const half* __restrict__ A,   // [16x16], row-major
                                 const half* __restrict__ B,   // [16x16], row-major (we'll use left 8 cols)
                                 half* __restrict__ C)         // [16x8],  row-major
{
    // ========================================================================
    // Shared Memory Allocation
    // ========================================================================
    __shared__ __align__(128) half As[16 * 16]; // 16x16 row-major, 512 bytes
    __shared__ __align__(128) half Bs[16 * 8];  // 16x8 row-major, 256 bytes
    // Note: 128-byte alignment required for ldmatrix instruction

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    if (tid >= 32) return;

    // ========================================================================
    // Stage 1: Global Memory -> Shared Memory using cp.async
    // ========================================================================
    // cp.async provides hardware-accelerated asynchronous copy
    // Benefits: Non-blocking, bypasses L1 cache, higher bandwidth

    // Convert generic pointers to address space-specific pointers
    // Required by PTX instructions that need explicit address space
    unsigned long long a_smem_ptr = __cvta_generic_to_shared(As);
    unsigned long long b_smem_ptr = __cvta_generic_to_shared(Bs);
    unsigned long long a_gmem_ptr = (unsigned long long)A;
    unsigned long long b_gmem_ptr = (unsigned long long)B;

    // Copy A: 16x16 halves = 512 bytes = 32 chunks of 16 bytes
    // Each thread copies 16-byte chunks (8 halves)
    // cp.async.cg: commit group, ensures 16-byte aligned transfers
    for (int chunk = tid; chunk < 32; chunk += blockDim.x) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "l"(a_smem_ptr + chunk * 16),   // Destination: SMEM address
               "l"(a_gmem_ptr + chunk * 16)    // Source: GMEM address
        );
    }

    // Copy B: 16x8 halves = 256 bytes, non-contiguous in GMEM
    // Need to copy row by row since B has stride 16 but we only use 8 columns
    // Each row: 8 halves = 16 bytes
    for (int row = tid; row < 16; row += blockDim.x) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "l"(b_smem_ptr + row * 16),     // Destination: SMEM (stride 16 bytes)
               "l"(b_gmem_ptr + row * 32)      // Source: GMEM (32 = 16 halves * 2 bytes)
        );
    }

    // Synchronization for cp.async
    // commit_group: Commits all preceding cp.async operations
    // wait_group 0: Waits for all committed groups to complete
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();  // Ensure all threads see the copied data

    // Reuse the shared pointers for ldmatrix
    unsigned long long a_ptr = a_smem_ptr;
    unsigned long long b_ptr = b_smem_ptr;

    // ========================================================================
    // Stage 2: Load A, B Fragments using ldmatrix
    // ========================================================================
    // ldmatrix: Loads data from SMEM into registers in the layout expected by mma
    // .m8n8.x4: Loads 4 8x8 matrices (total 16x16)
    // .trans: Transposes during load (row-major SMEM -> column-major-like registers)
    //
    // Thread-to-data mapping:
    // - Warp is divided into 4 groups of 8 threads
    // - Each group loads one 8x8 tile
    // - Group 0,1: top-left and top-right (rows 0-7)
    // - Group 2,3: bottom-left and bottom-right (rows 8-15)

    unsigned a_reg[4];              // A fragment: 4x 32-bit registers (8 halves)
    int a_quad = lane >> 3;                  // Thread group ID: 0..3 (8 threads per group)
    int a_row  = lane & 7;                   // Row within 8x8 tile: 0..7
    int a_col_block = (a_quad & 1) * 8;      // Column block: 0 or 8
    int a_row_block = (a_quad >> 1) * 8;     // Row block: 0 or 8

    // Calculate SMEM address for this thread's starting position
    // Row-major address: &As[(row_block + row) * 16 + col_block]
    unsigned long long a_addr =
        a_ptr + (unsigned long long)((a_row_block + a_row) * 16 + a_col_block) * sizeof(half);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])  // Output: 4 registers
        : "l"(a_addr)                                                     // Input: SMEM address
    );

    // B is stored as row-major [16 rows x 8 cols] in SMEM
    // .m8n8.x2: Loads 2 8x8 matrices (total 16x8)
    // .trans: Transposes during load to get column-major format for mma
    //
    // Thread-to-data mapping:
    // - Two 8x8 tiles along K dimension (rows 0-7 and 8-15)
    // - Upper 16 lanes (16-31) reuse addresses of lower 16 lanes for safety
    unsigned b_reg[2];              // B fragment: 2x 32-bit registers (4 halves)
    int b_quad = lane >> 3;                  // Thread group ID: 0..3 (8 threads per group)
    int b_row  = lane & 7;                   // Row within 8x8 tile: 0..7
    int b_k_block = (b_quad & 1) * 8;        // K block: 0 or 8 (along rows)

    // Calculate SMEM address: &Bs[(k_block + row) * 8 + 0]
    unsigned long long b_addr =
        b_ptr + (unsigned long long)((b_k_block + b_row) * 8) * sizeof(half);

    // Upper 16 lanes reuse lower addresses for .x2 safety
    // This is a common pattern to avoid addressing issues
    if (b_quad > 1) {
        int lower = lane & 15;               // Map to lower 16 lanes: 0..15
        int lg = lower >> 3;                 // Group in lower half: 0..1
        int lr = lower & 7;                  // Row in lower half: 0..7
        int lrBlk = (lg & 1) * 8;           // K block for lower half
        b_addr = b_ptr + (unsigned long long)((lrBlk + lr) * 8) * sizeof(half);
    }

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0,%1}, [%2];\n"
        : "=r"(b_reg[0]), "=r"(b_reg[1])    // Output: 2 registers
        : "l"(b_addr)                        // Input: SMEM address
    );

    // ========================================================================
    // Stage 3: Matrix Multiply-Accumulate (MMA)
    // ========================================================================
    // mma.m16n8k16.row.col.f16.f16.f16.f16
    // - m16n8k16: Output 16x8, inner dimension K=16
    // - row: A operand in row-major layout
    // - col: B operand in column-major layout
    // - f16 (4x): All operands and accumulator are fp16
    //
    // Operation: D = A * B + C
    // - A: 4 registers (a_reg[0..3])
    // - B: 2 registers (b_reg[0..1])
    // - C/D: 2 registers (c_reg[0..1]), C is input accumulator, D is output
    unsigned c_reg[2] = {0u, 0u};   // C accumulator: 2x 32-bit registers (4 halves), init to 0
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c_reg[0]), "+r"(c_reg[1])                                // Output: C/D registers (in-place)
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),   // Input: A registers
          "r"(b_reg[0]), "r"(b_reg[1])                                  // Input: B registers
    );

    // ========================================================================
    // Stage 4: Store Results to Global Memory (Scatter)
    // ========================================================================
    // Each thread's c_reg contains 4 fp16 values (2 uint32 registers)
    // These map to a 2x2 tile in the output matrix
    //
    // CRITICAL: mma.m16n8k16 uses 8-row interleaved output pattern!
    // - Lanes 0-3: columns 0,2,4,6 of rows 0 and 8
    // - Lanes 4-7: columns 0,2,4,6 of rows 1 and 9
    // - Lanes 8-11: columns 0,2,4,6 of rows 2 and 10
    // - ... and so on
    //
    // Each lane writes to two rows that are 8 rows apart

    // Reinterpret c_reg as half2 for easier access to packed halves
    half2* c_as_h2 = reinterpret_cast<half2*>(c_reg);
    auto a = __low2half(c_as_h2[0]), b = __high2half(c_as_h2[0]),
         c = __low2half(c_as_h2[1]), d = __high2half(c_as_h2[1]);

    // Calculate output position for this lane
    int quad = lane / 4;           // Quad ID: 0..7 (4 threads per quad)
    int col_in_quad = lane % 4;    // Column position within quad: 0..3

    int r0 = quad;                 // Base row: 0..7
    int c0 = col_in_quad * 2;      // Base column: 0,2,4,6 (each thread handles 2 columns)

    // Calculate pointers to two rows (r0 and r0+8)
    half* Cbase0 = C + r0 * 8 + c0;       // Row r0
    half* Cbase1 = C + (r0 + 8) * 8 + c0; // Row r0+8 (8 rows later)

    // Write 2x2 tile: v0 goes to row r0, v1 goes to row r0+8
    Cbase0[0] = a;     // [r0, c0]
    Cbase0[1] = b;    // [r0, c0+1]
    Cbase1[0] = c;     // [r0+8, c0]
    Cbase1[1] = d;    // [r0+8, c0+1]
}

int main() {
    const int M = 16, K = 16, N = 8;

    half *hA = new half[M*K];
    half *hB = new half[K*16];      // B has 16 columns; kernel uses leftmost 8
    half *hC = new half[M*N];       // PTX result
    half *hC_ref = new half[M*N];   // CPU reference result

    // A = Identity, B = (i+j) for easy verification
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            hA[i*K + j] = __float2half(i==j ? 1.f : 0.f);

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < 16; ++j)
            hB[i*16 + j] = __float2half(float(i + j));

    std::printf("========== Input Matrices ==========\n");
    print_mat_f16("Matrix A (Identity)", hA, M, K);

    std::printf("Matrix B (full 16 cols - debug):\n");
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < 16; ++j)
            std::printf("%.1f ", __half2float(hB[i*16 + j]));
        std::printf("\n");
    }
    std::printf("\n");

    std::printf("Expected Result (left 8 cols of B, since A=I):\n");
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j)
            std::printf("%.1f ", __half2float(hB[i*16 + j]));
        std::printf("\n");
    }
    std::printf("\n");

    half *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*16*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(half));
    cudaMemcpy(dA, hA, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*16*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*sizeof(half));

    // 1 block, 32 threads (1 warp)
    mma_m16n8k16_ptx<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, M*N*sizeof(half), cudaMemcpyDeviceToHost);

    std::printf("========== PTX Result ==========\n");
    print_mat_f16("C (PTX)", hC, M, N);

    cpu_matmul_reference(hA, hB, hC_ref, M, N, K);

    std::printf("========== CPU Reference Result ==========\n");
    print_mat_f16("C (CPU Reference)", hC_ref, M, N);

    std::printf("========== Verification (PTX vs CPU) ==========\n");
    bool match = true;
    float max_abs_diff = 0.0f;
    const float tol = 1e-2f;

    for (int i = 0; i < M*N; ++i) {
        float ptx_val = __half2float(hC[i]);
        float ref_val = __half2float(hC_ref[i]);
        float diff = fabsf(ptx_val - ref_val);

        if (diff > max_abs_diff) max_abs_diff = diff;

        if (diff > tol) {
            match = false;
            std::printf("Mismatch at [%d,%d] (idx=%d): CPU=%.2f  PTX=%.2f  (diff=%.3f)\n",
                        i/N, i%N, i, ref_val, ptx_val, diff);
        }
    }

    if (match) {
        std::printf("[OK] Results match! (max abs diff: %.6f)\n", max_abs_diff);
    } else {
        std::printf("[FAIL] Results do NOT match. (max abs diff: %.6f)\n", max_abs_diff);
    }

    delete[] hA; delete[] hB; delete[] hC; delete[] hC_ref;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return 0;
}