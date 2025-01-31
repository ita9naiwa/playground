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

const int BLOCK_SIZE = 16;
const int DIM_SPACE = 64;

__device__ inline float warpReduceMax(float val) {
  // Suppose each warp is 32 threads, and we want separate reduce for 0..15, 16..31.
  // laneId = 0..31
  unsigned laneId = threadIdx.x & 31;
  // halfId = 0 for lanes 0..15, 1 for lanes 16..31
  unsigned halfId = laneId >> 4;

  // Masks for the two half-warps:
  unsigned maskLo = 0x0000ffff;  // active lanes 0..15
  unsigned maskHi = 0xffff0000;  // active lanes 16..31
  unsigned mask   = (halfId == 0) ? maskLo : maskHi;

  // Each half-warp does a separate reduction on its 16 threads:

  // XOR-based shuffle ignoring the other half
  #pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(mask, val, offset, 32);
    // e.g. sum or max:
    // val += other;
    val = fmaxf(val, other);
  }
  return val;

}
template <typename scalar_t>
__global__ void flash_attention_v1_kernel(
    const scalar_t *Q, const scalar_t *K, const scalar_t *V,
    scalar_t *O, scalar_t *l, scalar_t *m,
    int B, int N, int D) {

  __shared__ scalar_t Q_shared[BLOCK_SIZE][DIM_SPACE];
  __shared__ scalar_t K_shared[BLOCK_SIZE][DIM_SPACE];
  __shared__ scalar_t V_shared[BLOCK_SIZE][DIM_SPACE];
  __shared__ scalar_t shared_vals[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x / 16;
  int ty = threadIdx.x % 16;

  for (int j = 0; j < (N / BLOCK_SIZE); j++) {
    for (int t = ty; t < D; t += 16) {
      int idx = blockIdx.x * N * D + j * BLOCK_SIZE * D + tx * D + t;
      K_shared[tx][t] = K[idx];
      V_shared[tx][t] = V[idx];
    }
    __syncthreads();

    for (int i = 0; i < (N / BLOCK_SIZE); i++) {
      auto _m = m + blockIdx.x * N + i * BLOCK_SIZE;
      auto _l = l + blockIdx.x * N + i * BLOCK_SIZE;

      for (int t = ty; t < D; t += 16) {
        int idx = blockIdx.x * N * D + i * BLOCK_SIZE * D + tx * D + t;
        Q_shared[tx][t] = Q[idx];
      }
      __syncthreads();

      float S_ij_orig = 0.0f;
      for (int t = 0; t < D; t++) {
        S_ij_orig += Q_shared[tx][t] * K_shared[ty][t];
      }

      float S_ij = S_ij_orig;
      for (int offset = 8; offset > 0; offset >>= 1) {
        float val = __shfl_down_sync(0xffff, S_ij, offset, 16);
        S_ij = fmaxf(S_ij, val);
      }

      if (ty == 0) {
        shared_vals[tx][0] = S_ij;
      }
      __syncthreads();

      float max_ij = shared_vals[tx][0];
      float P_ij = expf(S_ij_orig - max_ij);

      shared_vals[tx][ty] = P_ij;
      __syncthreads();

      float row_sum = P_ij;
      for (int offset = 8; offset > 0; offset >>= 1) {
        float val = __shfl_down_sync(0xffff, row_sum, offset, 16);
        row_sum += val;
      }

      if (ty == 0) {
        shared_vals[tx][0] = row_sum;
      }
      __syncthreads();

      float l_ij = shared_vals[tx][0];
      float old_mi = _m[tx];
      float old_l = _l[tx];
      float new_mi = fmaxf(old_mi, max_ij);
      float alpha = expf(old_mi - new_mi);
      float beta = expf(max_ij - new_mi);
      float new_l = alpha * old_l + beta * l_ij;
      __syncthreads();

      shared_vals[tx][ty] = P_ij;
      __syncthreads();

      for (int t = ty; t < D; t += 16) {
        int O_index = blockIdx.x * N * D + i * BLOCK_SIZE * D + tx * D + t;
        float O_temp = (old_l / new_l) * alpha * O[O_index];
        for (int l0 = 0; l0 < BLOCK_SIZE; l0++) {
          O_temp += (beta / new_l) * shared_vals[tx][l0] * V_shared[l0][t];
        }
        O[O_index] = O_temp;
      }
      __syncthreads();

      if (ty == 0) {
        _m[tx] = new_mi;
        _l[tx] = new_l;
      }
      __syncthreads();
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
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
