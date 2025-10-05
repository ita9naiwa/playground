// simple.cu  —  m16n8k16 PTX + Host reference GEMM compare (no WMMA)

#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

// -------------------- CPU Reference: C++ 행렬 곱셈 구현 --------------------
// C = A * B (간단한 3중 루프)
static void cpu_matmul_reference(const half* A, const half* B, half* C,
                                 int M, int N, int K) {
    // A: [M x K] row-major
    // B: [K x 16] row-major (실제로는 16열이지만 N=8열만 사용)
    // C: [M x N] row-major

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;  // float로 누적 (정밀도)

            // 행렬 곱셈: C[m,n] = Σ_k A[m,k] * B[k,n]
            for (int k = 0; k < K; ++k) {
                float a_val = __half2float(A[m * K + k]);
                float b_val = __half2float(B[k * 16 + n]);  // B stride = 16
                sum += a_val * b_val;
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

// -------------------- PTX kernel: mma.m16n8k16.row.col.f16 --------------------
__global__ void mma_m16n8k16_ptx(const half* __restrict__ A,   // [16x16], row-major
                                 const half* __restrict__ B,   // [16x16], row-major
                                 half* __restrict__ C)         // [16x8],  row-major
{
    __shared__ __align__(128) half As[16 * 16]; // row-major 16x16
    __shared__ __align__(128) half Bs[16 * 8];  // row-major 16x8 (좌측 8열만 사용)

    int tid  = threadIdx.x;
    int lane = tid & 31;
    if (tid >= 32) return; // 단일 워프 데모

    // ---- GMEM -> SMEM ----
    // A 복사
    for (int i = tid; i < 16*16; i += blockDim.x) {
        As[i] = A[i];
    }
    // B의 좌측 8열만 row-major 16x8로 SMEM에 저장
    for (int i = tid; i < 16*8; i += blockDim.x) {
        int k = i / 8;  // 0..15
        int n = i % 8;  // 0..7
        Bs[k*8 + n] = B[k*16 + n];
    }
    __syncthreads();

    // ---- 64-bit shared pointers ----
    unsigned long long a_ptr = __cvta_generic_to_shared(As);
    unsigned long long b_ptr = __cvta_generic_to_shared(Bs);

    // ---- fragment registers ----
    unsigned a_reg[4];
    unsigned b_reg[2];
    unsigned c_reg[2] = {0u, 0u}; // accumulator

    // ===== A: ldmatrix.m8n8.x4.trans (row-major -> A fragment) =====
    int groupA = lane >> 3;          // 0..3 (8-thread group)
    int rowA   = lane & 7;           // 0..7 (row within 8x8)
    int rBlkA  = (groupA & 1) * 8;   // 0 or 8
    int cBlkA  = (groupA >> 1) * 8;  // 0 or 8
    const int LDA = 16;
    unsigned long long a_addr =
        a_ptr + (unsigned long long)((rBlkA + rowA) * LDA + cBlkA) * sizeof(half);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
        : "l"(a_addr)
    );

    // ===== B: ldmatrix.m8n8.x2.trans (SMEM에 row-major 16x8) =====
    // 두 개의 8x8 블록: rBlkB in {0,8}, cBlkB=0
    int groupB = lane >> 3;          // 0..3
    int rowB   = lane & 7;           // 0..7
    int rBlkB  = (groupB & 1) * 8;   // 0 or 8
    const int LDB = 8;               // row-major stride for Bs
    unsigned long long b_addr =
        b_ptr + (unsigned long long)((rBlkB + rowB) * LDB) * sizeof(half);

    // 상위 16 lanes는 하위 0..15의 주소 복제 (.x2에서 안전)
    if (groupB > 1) {
        int lower = lane & 15;
        int lg = lower >> 3;
        int lr = lower & 7;
        int lrBlk = (lg & 1) * 8;
        b_addr = b_ptr + (unsigned long long)((lrBlk + lr) * LDB) * sizeof(half);
    }

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0,%1}, [%2];\n"
        : "=r"(b_reg[0]), "=r"(b_reg[1])
        : "l"(b_addr)
    );

    // ===== MMA: C = A*B + C (f16) =====
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c_reg[0]), "+r"(c_reg[1])
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
          "r"(b_reg[0]), "r"(b_reg[1])
    );

    // ===== Scatter: lane별 2x2 타일을 C[16x8]에 저장 =====
    // c_reg(2 x u32)에는 half 4개가 packed됨.
    half2* c_as_h2 = reinterpret_cast<half2*>(c_reg);
    half2 v0 = c_as_h2[0]; // 2개 half
    half2 v1 = c_as_h2[1]; // 2개 half

    // ---- m16n8k16 output layout (패턴 1) ----
    // 출력: 16x8, 각 thread가 4개 element 소유
    // 일반적인 패턴: 각 thread group이 특정 위치 담당

    int group_id = lane >> 2;  // 0..7 (4-thread groups)
    int thread_in_group = lane & 0x3;  // 0..3

    int r0 = (group_id >> 1) * 4 + (thread_in_group >> 1) * 2;  // row base
    int c0 = ((group_id & 1) << 2) + (thread_in_group & 1) * 2;  // col base

    // C는 row-major [16x8]
    half* Cbase = C + r0 * 8 + c0;
    Cbase[0] = __low2half(v0);
    Cbase[1] = __high2half(v0);
    Cbase[8] = __low2half(v1);
    Cbase[9] = __high2half(v1);
}

// ---------------------------------- main ----------------------------------
int main() {
    const int M = 16, K = 16, N = 8;

    // Host alloc & init
    half *hA = new half[M*K];
    half *hB = new half[K*16];   // B는 16열 준비(좌측 8열만 커널에서 사용)
    half *hC = new half[M*N];
    half *hC_ref = new half[M*N];  // CPU reference 결과

    // A = I
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            hA[i*K + j] = __float2half(i==j ? 1.f : 0.f);

    // B = (i+j)
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < 16; ++j)
            hB[i*16 + j] = __float2half(float(i + j));

    std::printf("========== Input Matrices ==========\n");
    print_mat_f16("Matrix A (Identity)", hA, M, K);

    // B 전체 출력 (16열)
    std::printf("Matrix B (full 16 cols - 디버그용):\n");
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < 16; ++j)
            std::printf("%.1f ", __half2float(hB[i*16 + j]));
        std::printf("\n");
    }
    std::printf("\n");

    // Expected result: A가 identity이므로 C = B의 첫 8열
    std::printf("Expected Result (B의 첫 8열, A=I이므로):\n");
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j)
            std::printf("%.1f ", __half2float(hB[i*16 + j]));
        std::printf("\n");
    }
    std::printf("\n");

    // Device alloc/copy
    half *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*16*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(half));
    cudaMemcpy(dA, hA, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*16*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*sizeof(half));

    // Launch (1 warp)
    mma_m16n8k16_ptx<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, M*N*sizeof(half), cudaMemcpyDeviceToHost);

    std::printf("========== PTX Result ==========\n");
    print_mat_f16("C (PTX)", hC, M, N);

    // CPU reference: 순수 C++ 행렬 곱셈
    cpu_matmul_reference(hA, hB, hC_ref, M, N, K);

    std::printf("========== CPU Reference Result ==========\n");
    print_mat_f16("C (CPU Reference)", hC_ref, M, N);

    // Compare PTX vs CPU reference
    std::printf("========== Verification (PTX vs CPU) ==========\n");
    bool match = true;
    float max_abs_diff = 0.0f;
    const float tol = 1e-2f; // half 연산 오차 여유
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
        std::printf("✓ Results match! (max abs diff: %.6f)\n", max_abs_diff);
        std::printf("  -> PTX ldmatrix + mma.sync 구현이 정확합니다!\n");
    } else {
        std::printf("✗ Results do NOT match. (max abs diff: %.6f)\n", max_abs_diff);
        std::printf("  -> NOTE: scatter 매핑(r0,c0)을 조정해야 할 수 있습니다.\n");
    }

    // Cleanup
    delete[] hA; delete[] hB; delete[] hC; delete[] hC_ref;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}