#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

torch::Tensor matmul(torch::Tensor A,
  torch::Tensor B, std::optional<torch::Tensor> C,
  int version, bool B_transposed);