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

int64_t getDeviceBlockSharedMemSize() {
  int device_id = at::cuda::current_device();
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  return static_cast<int64_t>(prop.sharedMemPerBlock);
}

template <typename T>
T ceil(T x, T y) {
  return (x + y - 1) / y;
}

inline __device__ void kceil(int x, int y, int *ret) {
  *ret = ((x + y - 1) / y);
}

const int BLOCK_SIZE = 16;
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

  // Q: (B, N, D)
  // K: (B, N, D)
  // V: (B, N, D)
  for(int j = 0; j < (N / BLOCK_SIZE); j++) {
    // Load K_j, V_j into shared memory

    for (int t = threadIdx.y; t < D; t += blockDim.y) {
      int K_index = blockIdx.x * N * D + j * BLOCK_SIZE * D + threadIdx.x * D + t;
      K_shared[threadIdx.x][t] = K[K_index];
      V_shared[threadIdx.x][t] = V[K_index];
    }
    __syncthreads();
    for (int i = 0; i < (N / BLOCK_SIZE); i++) {
      // fix current _m, _l;
      auto _m = m + blockIdx.x * N + i * BLOCK_SIZE;
      auto _l = l + blockIdx.x * N + i * BLOCK_SIZE;
      // // Load Q_i into shared memory
      for (int t = threadIdx.y; t < D; t += blockDim.y) {
        int Q_index = blockIdx.x * N * D + i * BLOCK_SIZE * D + threadIdx.x * D + t;
        Q_shared[threadIdx.x][t] = Q[Q_index];
      }
      __syncthreads();

      float S_ij = 0;
      for (int t = 0 ; t < D; ++t)
        S_ij += Q_shared[threadIdx.x][t] * K_shared[threadIdx.y][t];

      // S 16 x 16 matrix, is stored across thread block (16 x 16)
      float max_ij;
      shared_vals[threadIdx.x][threadIdx.y] = S_ij;
      __syncthreads();
      if (threadIdx.y == 0) {
        for (int t = 1; t < BLOCK_SIZE; ++t)
          shared_vals[threadIdx.x][0] = max(shared_vals[threadIdx.x][0], shared_vals[threadIdx.x][t]);
      }
      __syncthreads();
      max_ij = shared_vals[threadIdx.x][0];

      float P_ij = exp(S_ij - max_ij);

      shared_vals[threadIdx.x][threadIdx.y] = P_ij;
      __syncthreads();
      if (threadIdx.y == 0) {
        for (int t = 1; t < BLOCK_SIZE; ++t)
          shared_vals[threadIdx.x][0] += shared_vals[threadIdx.x][t];
      }
      __syncthreads();
      float l_ij = shared_vals[threadIdx.x][0];
      float old_mi = _m[threadIdx.x];
      float old_l = _l[threadIdx.x];
      float new_mi = max(old_mi, max_ij);
      float exp_left = exp(old_mi - new_mi);
      float exp_right = exp(max_ij - new_mi);
      float new_l = old_l * exp(old_mi - new_mi) + l_ij;

      shared_vals[threadIdx.x][threadIdx.y] = P_ij;
      __syncthreads();
      float divd = (1.0f / (1e-6f + new_l));
      for (int t = threadIdx.y; t < D; t += blockDim.y) {
        auto O_index = blockIdx.x * N * D + i * BLOCK_SIZE * D + threadIdx.x * D + t;
        float O_temp = divd * exp_left * old_l * O[O_index];
        if (O_temp != 0.0f)
          printf("O_temp: %f\n", O_temp);
        for (int l = 0; l < BLOCK_SIZE; ++l) {
          O_temp += divd * exp_right * shared_vals[threadIdx.x][l] * V_shared[l][t];
        }
        O[O_index] = O_temp;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
        _m[threadIdx.x] = new_mi;
        _l[threadIdx.x] = new_l;
      }
      __syncthreads();
    }
  }
}

torch::Tensor flash_attention_v1(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // Q: (B, H, N, D)
  // K: (B, H, N, D)
  // V: (B, H, N, D)
  // B: batch size, H: number of heads, N: sequence length, D: hidden size
  auto dtype = Q.scalar_type();

  if (!(dtype == torch::kFloat32 || dtype == torch::kHalf))
    throw std::runtime_error("Unsupported data type");

  auto B = Q.size(0);
  auto H = Q.size(1);
  auto N = Q.size(2);
  auto D = Q.size(3);

  auto options = Q.options();
  auto O = torch::zeros({B, H, N, D}, options);
  auto l = torch::zeros({B, H, N}, options);
  // if fill_value = -infty then -infty + infty = nan error,
  // so fill appropriate min value
  auto m = torch::full({B, H, N}, -10000.0, options);

  // We just view the input tensors as 2D tensors of shape (B*H, N, D)
  auto batch_size = B * H;
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
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
        batch_size, N, D);
    }
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  return O;
}

