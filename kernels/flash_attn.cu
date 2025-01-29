#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
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

__global__ void ceil(int x, int y, int *ret) {
  *ret = ((x + y - 1) / y);
}

template <typename scalar_t>
__global__ void flash_attention_v1_kernel(
  const scalar_t *Q, const scalar_t *K, const scalar_t *V,
  scalar_t *O, scalar_t *l, scalar_t *m,
  int B, int N, int D) {
  __shared__ scalar_t Q_shared[16 * 16];
  __shared__ scalar_t K_shared[16 * 16];
  __shared__ scalar_t V_shared[16 * 16];
  __shared__ scalar_t shared_vals[16 * 16];

  // Q: (B, N, D)
  // K: (B, N, D)
  // V: (B, N, D)

  int block_c = blockDim.x;
  int block_r = blockDim.y;

  auto curr_Q = Q + gridDim.x * N * D + block_r * D;
  auto curr_K = K + gridDim.x * N * D + block_c * D;
  auto curr_V = V + gridDim.x * N * D + block_c * D;
  auto curr_O = O + gridDim.x * N * D + block_r * D;
  auto curr_l = l + gridDim.x * N;
  auto curr_m = m + gridDim.x * N;

  for(int j = 0; j < 16; j++) {
    // Load K_j, V_j into shared memory
    for (int t = 0; t < 16; ++t) {
      K_shared[t * 16 + j] = curr_K[t];
      V_shared[t * 16 + j] = curr_V[t];
    }
    __syncthreads();
    printf("222\n");
    for (int i = 0; i < 16; i++) {
      auto m = curr_m + i * 16;
      auto l = curr_l + i * 16;
      // Load Q_i into shared memory
      for (int t = 0; t < 16; ++t)
        Q_shared[t * tile_r + i] = curr_Q[t];

      __syncthreads();
      scalar_t S_ij = 0;
      printf("333\n");
      for (int t = 0 ; t < 16; ++t)
        S_ij += Q_shared[threadIdx.x * 16 + t] * K_shared[threadIdx.y * 16 + t];
      printf("%0.4f\n", float(S_ij));
      float max_ij;
      shared_vals[threadIdx.x * 16 + threadIdx.y] = S_ij;
      __syncthreads();
      if (threadIdx.y == 0) {

        for (int t = 1; t < 16; ++t)
          shared_vals[threadIdx.x * 16 + 0] = max(shared_vals[threadIdx.x * 16 + 0], shared_vals[threadIdx.x * 16 + t]);
      }
      __syncthreads();
      max_ij = shared_vals[threadIdx.x * 16 + 0];


      scalar_t P_ij = exp(S_ij - max_ij);

      float l_ij;
      shared_vals[threadIdx.x * 16 + threadIdx.y] = P_ij;
      __syncthreads();
      if (threadIdx.y == 0) {
        for (int t = 1; t < 16; ++t)
          shared_vals[threadIdx.x * 16 + 0] += shared_vals[threadIdx.x * 16 + t];
      }
      __syncthreads();
      l_ij = shared_vals[threadIdx.x * 16 + 0];

      // auto mi_old = m[i];
      auto old_mi = m[threadIdx.x];
      auto old_l = l[threadIdx.x];
      auto exp_left = exp(old_mi - m[i]);
      auto exp_right = exp(max_ij - m[i]);
      auto new_mi = max(old_mi, max_ij);
      auto new_l = old_l * exp(old_mi - new_mi) + l_ij;

      // store Pij into shared memory
      shared_vals[threadIdx.x * 16 + threadIdx.y] = exp_right * P_ij;
      __syncthreads();

      scalar_t O_ij = 0.0;
      for (int t = 0; t < 16; ++t)
        O_ij += (1.0 / l_ij) * shared_vals[threadIdx.x * 16 + t] * V_shared[t * 16 + threadIdx.y];
      O_ij += (1.0 / l_ij) * l[threadIdx.x] * curr_O[threadIdx.x * 16 + threadIdx.y];
      curr_O[threadIdx.x * 16 + threadIdx.y] = O_ij;
      if (threadIdx.x == 0) {
        m[i] = new_mi;
        l[i] = new_l;
      }
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
  auto m = torch::full({B, H, N}, -std::numeric_limits<float>::infinity(), options);

  // We just view the input tensors as 2D tensors of shape (B*H, N, D)
  auto batch_size = B * H;
  auto M = getDeviceBlockSharedMemSize();

  if (dtype == torch::kFloat32)
    M /= 4;
  else if (dtype == torch::kHalf)
    M /= 2;


  dim3 block_size(16, 16);
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

