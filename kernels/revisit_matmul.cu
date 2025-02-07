#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cassert>
#include <iostream>

#include "ATen/Dispatch.h"
#include "ATen/ops/transpose.h"
#include "c10/core/ScalarType.h"
#include "revisit_matmul.h"
#include "wmma_matmul.h"

template <typename scalar_t>
__global__ void matmul_naive(scalar_t *A, scalar_t *B, scalar_t *C, int M, int K, int N, bool B_transposed) {
  int init_i = blockIdx.x * blockDim.x + threadIdx.x;
  int init_j = blockIdx.y * blockDim.y + threadIdx.y;

  int i_stride = gridDim.x * blockDim.x;
  int j_stride = gridDim.y * blockDim.y;

  for (int i = init_i; i < M; i += i_stride) {
    for (int j = init_j; j < N; j += j_stride) {
      scalar_t sum = 0;
      for (int k = 0; k < K; ++k) {
        if (!B_transposed)
          sum += A[i * K + k] * B[k * N + j];
        else
          sum += A[i * K + k] * B[j * K + k];
      }
      if (i < M && j < N) C[i * N + j] = sum;
    }
  }
}

template <typename scalar_t>
__global__ void matmul_smem(scalar_t *A, scalar_t *B, scalar_t *C, int M, int K, int N, bool B_transposed) {
  __shared__ scalar_t tile_A[16][16];
  __shared__ scalar_t tile_B[16][16];

  int init_i = blockIdx.x * blockDim.x;
  int init_j = blockIdx.y * blockDim.y;

  int i_stride = gridDim.x * blockDim.x;
  int j_stride = gridDim.y * blockDim.y;

  int i0 = threadIdx.x;
  int j0 = threadIdx.y;
  for (int i = init_i; i < M; i += i_stride) {
    for (int j = init_j; j < N; j += j_stride) {
      int sum = 0;
      for (int k = 0; k < K; k += 16) {  // Tiling over K dimension
        // Load A's tile into shared memory
        if (i + i0 < M && k + j0 < K)
          tile_A[i0][j0] = A[(i + i0) * K + (k + j0)];
        else
          tile_A[i0][j0] = 0;

        // Load B's tile into shared memory
        if (j + j0 < N && k + i0 < K)
          if (!B_transposed)
            tile_B[i0][j0] = B[(k + i0) * N + (j + j0)];
          else
            tile_B[i0][j0] = B[(j + j0) * N + (k + i0)];
        else
          tile_B[i0][j0] = 0;
        __syncthreads();

        // Perform computation within the tile
        for (int k0 = 0; k0 < 16; k0++) {  // Iterate over tile's K dimension
          sum += tile_A[i0][k0] * tile_B[k0][j0];
        }
        __syncthreads();
      }

      // Write the result to C
      if (i + i0 < M && j + j0 < N) {
        C[(i + i0) * N + (j + j0)] = sum;
      }
    }
  }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B, std::optional<torch::Tensor> C, int version, bool B_transposed) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  assert(A.size(1) == B.size(0));

  torch::Tensor _C;
  if (!C.has_value())
    _C = torch::zeros({M, N}, A.options());
  else
    _C = C.value();

  if (version == 2) {
    return wmma_matmul(A, B, _C);
  }

  auto _B = B;
  if (B_transposed) _B = torch::transpose(B, 1, 0);

  dim3 block_size(16, 16);
  dim3 grid_size(64, 64);

  switch (version) {
    case 0:
      AT_DISPATCH_REDUCED_FLOATING_TYPES(A.scalar_type(), "matmul", [&] {
        matmul_naive<<<grid_size, block_size>>>(A.data_ptr<scalar_t>(), _B.data_ptr<scalar_t>(),
                                                _C.data_ptr<scalar_t>(), M, K, N, B_transposed);
      });
      break;
    case 1:
      AT_DISPATCH_REDUCED_FLOATING_TYPES(A.scalar_type(), "matmul", [&] {
        matmul_smem<<<grid_size, block_size>>>(A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), _C.data_ptr<scalar_t>(),
                                               M, K, N, B_transposed);
      });
      break;
    default:
      throw std::runtime_error("Invalid version");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  return _C;
}
