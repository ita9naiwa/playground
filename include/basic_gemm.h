#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cuda_runtime.h"

#include "cutlass/gemm/device/gemm.h"

cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc);

cudaError_t CutlassHGemmRelu(
  int M,
  int N,
  int K,
  float alpha,
  void *A,
  int lda,
  void *B,
  int ldb,
  void *C,
  int ldc,
  void* bias);


torch::Tensor simple_cutlass_gemm(
    torch::Tensor &A,
    torch::Tensor &B,
    float alpha,
    float beta);

torch::Tensor cutlass_half_gemm_relu(
  const torch::Tensor &A,
  const torch::Tensor &B,
  const torch::Tensor &bias,
  float alpha);