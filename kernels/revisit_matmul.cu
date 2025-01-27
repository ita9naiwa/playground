#include <cassert>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "ATen/Dispatch.h"
#include "ATen/ops/transpose.h"
#include "revisit_matmul.h"

__global__ void matmul_naive(int *A, int *B, int *C,
                             int m, int k, int n,
                             bool B_transposed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        int sum = 0;
        for (int i = 0; i < k; i++) {
            if (!B_transposed)
                sum += A[row * k + i] * B[i * n + col];
            else
                sum += A[row * k + i] * B[col * k + i];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matmul_grid_stride(int *A, int *B, int *C,
                                   int m, int k, int n,
                                   bool B_transposed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int row_stride = gridDim.x * blockDim.x;
    int col_stride = gridDim.y * blockDim.y;

    for (int i = row; i < m; i += row_stride) {
        for (int j = col; j < n; j += col_stride) {
            int sum = 0;
            for (int l = 0; l < k; ++l) {
                if (!B_transposed)
                    sum += A[i * k + l] * B[l * n + j];
                else
                    sum += A[i * k + l] * B[j * k + l];
            }
            if (i < m && j < n)
                C[i * n + j] = sum;
        }
    }
}

__global__ void matmul_smem(int *A, int *B, int *C, int M, int K, int N) {
    __shared__ int tile_A[16][16];
    __shared__ int tile_B[16][16];

    int i = blockIdx.x * blockDim.x; // Grid parallelizing i (rows)
    int j = blockIdx.y * blockDim.y; // Grid parallelizing j (columns)
    int i0 = threadIdx.x;
    int j0 = threadIdx.y;
    int sum = 0;

    for (int k = 0; k < K; k += 16) { // Tiling over K dimension
        // Load A's tile into shared memory
        if (i + i0 < M && k + j0 < K)
            tile_A[i0][j0] = A[(i + i0) * K + (k + j0)];
        else
            tile_A[i0][j0] = 0;

        // Load B's tile into shared memory
        if (j + j0 < N && k + i0 < K)
            tile_B[i0][j0] = B[(k + i0) * N + (j + j0)];
        else
            tile_B[i0][j0] = 0;
        __syncthreads();

        // Perform computation within the tile
        for (int k0 = 0; k0 < 16; k0++) { // Iterate over tile's K dimension
            sum += tile_A[i0][k0] * tile_B[k0][j0];
        }
        __syncthreads();
    }

    // Write the result to C
    if (i + i0 < M && j + j0 < N) {
        C[(i + i0) * N + (j + j0)] = sum;
    }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B, int version, bool B_transposed) {
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    assert(A.size(1) == B.size(0));

    auto C = torch::zeros({m, n}, A.options());

    auto _B = B;
    if (B_transposed)
        _B = torch::transpose(B, 1, 0);

    dim3 block_size(16, 16);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    switch (version) {
    case 0:
        AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "matmul", [&] {
            matmul_naive<<<grid_size, block_size>>>(
                A.data_ptr<int>(),
                _B.data_ptr<int>(),
                C.data_ptr<int>(),
                m, k, n, B_transposed);
        });
        break;
    case 1:
        AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "matmul", [&] {
            matmul_grid_stride<<<grid_size, block_size>>>(
                A.data_ptr<int>(),
                _B.data_ptr<int>(),
                C.data_ptr<int>(),
                m, k, n, B_transposed);
        });
        break;
    case 2:
        AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "matmul", [&] {
            matmul_smem<<<grid_size, block_size>>>(
                A.data_ptr<int>(),
                B.data_ptr<int>(),
                C.data_ptr<int>(),
                m, k, n);
        });
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    return C;
}