#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

torch::Tensor flash_attention_v1(torch::Tensor Q, torch::Tensor k, torch::Tensor V);