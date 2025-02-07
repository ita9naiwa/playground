#include <cuda_fp16.h>
#include <mma.h>

#include <iostream>

using namespace std;
using namespace nvcuda;
const int TILE_SIZE = 16;

using A_FRAGMENT = wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major>;
using B_FRAGMENT = wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major>;
using ACCM_FRAGMENT = wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float>;

__global__ void wmmaKernel(half *a, half *b, float *c, int M, int N, int K) {
  A_FRAGMENT a_frag;
  B_FRAGMENT b_frag;
  ACCM_FRAGMENT c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  half *a_tile = a + blockIdx.x * TILE_SIZE * K;
  half *b_tile = b + blockIdx.y * TILE_SIZE;
  for (int iter = 0; iter < (K / TILE_SIZE); ++iter) {
    wmma::load_matrix_sync(a_frag, a_tile + iter * TILE_SIZE, K);
    wmma::load_matrix_sync(b_frag, b_tile + iter * TILE_SIZE * N, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  float *c_pos = c + blockIdx.x * TILE_SIZE * N + blockIdx.y * TILE_SIZE;
  wmma::store_matrix_sync(c_pos, c_frag, N, wmma::mem_row_major);
}

int main() {
  const int M = 32;
  const int N = 64;
  const int K = 48;

  half *h_a = new half[M * K];
  half *h_b = new half[K * N];
  float *h_c = new float[M * N];

  for (int i = 0; i < M * K; i++) {
    h_a[i] = __float2half(1.0f);
  }

  for (int i = 0; i < K * N; i++) {
    h_b[i] = __float2half(1.0f);
  }

  half *d_a, *d_b;
  float *d_c;

  cudaMalloc(&d_a, M * K * sizeof(half));
  cudaMalloc(&d_b, K * N * sizeof(half));
  cudaMalloc(&d_c, M * N * sizeof(float));

  cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice);

  dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
  dim3 block(32);

  wmmaKernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

  cudaDeviceSynchronize();

  cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      cout << h_c[i * N + j] << " ";
    }
    cout << endl;
  }
}