#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

torch::Tensor conv1d(torch::Tensor input,
                     torch::Tensor weight,
                     std::optional<torch::Tensor> bias,
                     int stride,
                     int padding);