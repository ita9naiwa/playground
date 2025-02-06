#include <iostream>

#include "ATen/ops/pad.h"
#include "c10/util/Half.h"
#include "wmma_matmul.h"

const int TILE_SIZE = 16;
using A_FRAGMENT = wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major>;
using B_FRAGMENT = wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major>;
using ACCM_FRAGMENT = wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, half>;

__global__ void wmmaKernel(half *a, half *b, half *c, int M, int N, int K) {
  A_FRAGMENT a_frag;
  B_FRAGMENT b_frag;
  ACCM_FRAGMENT c_frag;

  wmma::fill_fragment(c_frag, __float2half(0.0f));

  half *a_tile = a + blockIdx.x * TILE_SIZE * K;
  half *b_tile = b + blockIdx.y * TILE_SIZE;
  for (int iter = 0; iter < (K / TILE_SIZE); ++iter) {
    wmma::load_matrix_sync(a_frag, a_tile + iter * TILE_SIZE, K);
    wmma::load_matrix_sync(b_frag, b_tile + iter * TILE_SIZE * N, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  half *c_pos = c + blockIdx.x * TILE_SIZE * N + blockIdx.y * TILE_SIZE;
  wmma::store_matrix_sync(c_pos, c_frag, N, wmma::mem_row_major);
}

int next_multiple_of_16(int n) {
  int un = static_cast<unsigned int>(n);
  unsigned int ret = (un + 15U) & ~15U;
  return static_cast<int>(ret);
}

int ceil(int a, int b=TILE_SIZE) {
  return (a + b - 1) / b;
}

torch::Tensor wmma_matmul(torch::Tensor A,
                   torch::Tensor B,
                   torch::Tensor C) {
  if (A.scalar_type() != torch::kHalf || B.scalar_type() != torch::kHalf || C.scalar_type() != torch::kHalf) {
    throw std::runtime_error("Input tensors must be of type torch::kHalf");
  }

  const int M = A.size(0);
  const int N = B.size(1);
  const int K = A.size(1);

  if ((M % 16) || (N % 16) || (K % 16)) {
    throw std::runtime_error("Input dimensions must be multiples of 16");
  }

  dim3 grid(M / TILE_SIZE, N / TILE_SIZE);
  dim3 block(32);
  wmmaKernel<<<grid, block>>>(
    (half*)A.data_ptr<at::Half>(),
    (half*)B.data_ptr<at::Half>(),
    (half*)C.data_ptr<at::Half>(),
    M, N, K);
  return C;
}