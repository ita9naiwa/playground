#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include "c10/core/ScalarType.h"
#include "flash_attn.h"
#include "torch/types.h"

const int BLOCK_SIZE = 32;
const int DIM_SPACE = 64;

template <typename scalar_t>
__global__ void flash_attention_v1_kernel(
    const scalar_t *Q, const scalar_t *K, const scalar_t *V,
    scalar_t *O, scalar_t *l, scalar_t *m,
    int B, int N, int D) {
  __shared__ scalar_t Q_shared[BLOCK_SIZE][DIM_SPACE];
  __shared__ scalar_t K_shared[BLOCK_SIZE][DIM_SPACE];
  __shared__ scalar_t V_shared[BLOCK_SIZE][DIM_SPACE];
  __shared__ scalar_t shared_vals[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x / BLOCK_SIZE;
  int ty = threadIdx.x % BLOCK_SIZE;

  for (int j = 0; j < (N / BLOCK_SIZE); j++) {
    for (int t = ty; t < D; t += BLOCK_SIZE) {
      int idx = blockIdx.x * N * D + j * BLOCK_SIZE * D + tx * D + t;
      K_shared[tx][t] = K[idx];
      V_shared[tx][t] = V[idx];
    }
    __syncthreads();

    for (int i = 0; i < (N / BLOCK_SIZE); i++) {
      auto _m = m + blockIdx.x * N + i * BLOCK_SIZE;
      auto _l = l + blockIdx.x * N + i * BLOCK_SIZE;

      for (int t = ty; t < D; t += BLOCK_SIZE) {
        int idx = blockIdx.x * N * D + i * BLOCK_SIZE * D + tx * D + t;
        Q_shared[tx][t] = Q[idx];
      }
      __syncthreads();

      float S_ij_orig = 0.0f;
      for (int t = 0; t < D; t++) {
        S_ij_orig += Q_shared[tx][t] * K_shared[ty][t];
      }

      // Use xor_sync for max reduction
      float S_ij = S_ij_orig;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        float val = __shfl_xor_sync(0xffffffff, S_ij, offset, 32);
        S_ij = fmaxf(S_ij, val);
      }
      float max_ij = S_ij;
      __syncthreads();

      float P_ij = __expf(S_ij_orig - max_ij);

      // Use xor_sync for sum reduction
      float row_sum = P_ij;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        float val = __shfl_xor_sync(0xffffffff, row_sum, offset, 32);
        row_sum += val;
      }
      float l_ij = row_sum;
      __syncthreads();

      float old_mi = _m[tx];
      float old_l = _l[tx];
      float new_mi = fmaxf(old_mi, max_ij);
      float alpha = __expf(old_mi - new_mi);
      float beta = __expf(max_ij - new_mi);
      float new_l = alpha * old_l + beta * l_ij;

      shared_vals[tx][ty] = P_ij;
      __syncthreads();

      for (int t = ty; t < D; t += BLOCK_SIZE) {
        int O_index = blockIdx.x * N * D + i * BLOCK_SIZE * D + tx * D + t;
        float O_temp = (old_l / new_l) * alpha * O[O_index];
        #pragma unroll
        for (int l0 = 0; l0 < BLOCK_SIZE; l0++) {
          O_temp += (beta / new_l) * shared_vals[tx][l0] * V_shared[l0][t];
        }
        O[O_index] = O_temp;
      }
      __syncthreads();


      _m[tx] = new_mi;
      _l[tx] = new_l;
    }
  }
}

torch::Tensor flash_attention_v1(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  auto dtype = Q.scalar_type();
  if (!(dtype == torch::kFloat32 || dtype == torch::kHalf)) {
    throw std::runtime_error("Unsupported data type");
  }

  auto B = Q.size(0);
  auto H = Q.size(1);
  auto N = Q.size(2);
  auto D = Q.size(3);

  auto options = Q.options();
  auto O = torch::zeros({B, H, N, D}, options);
  auto l = torch::zeros({B, H, N}, options);
  auto m = torch::full({B, H, N}, -10000.0, options);

  auto batch_size = B * H;
  dim3 block_size(BLOCK_SIZE * BLOCK_SIZE);
  dim3 grid_size(batch_size);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
    Q.scalar_type(), "flash_attention_v1", [&] {
      flash_attention_v1_kernel<<<grid_size, block_size>>>(
        Q.data_ptr<scalar_t>(),
        K.data_ptr<scalar_t>(),
        V.data_ptr<scalar_t>(),
        O.data_ptr<scalar_t>(),
        l.data_ptr<scalar_t>(),
        m.data_ptr<scalar_t>(),
        batch_size, N, D
      );
    }
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  }
  return O;
}
