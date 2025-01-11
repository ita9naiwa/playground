#pragma once

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
