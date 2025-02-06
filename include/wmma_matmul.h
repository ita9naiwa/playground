#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;


torch::Tensor wmma_matmul(torch::Tensor A,
                   torch::Tensor B,
                   torch::Tensor C);
