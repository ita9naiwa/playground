#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "util.cuh"


__global__ void conv1d_naive(
  const float* __restrict__ input,
  const float* __restrict__ weight,
  const float* __restrict__ bias,
  float* __restrict__ output,
  int B, int C_in, int L_in, int C_out, int L_out, int K, int stride, int padding)
{
  __shared__ float smem[1024];

  int b     = blockIdx.x;
  int c_out = blockIdx.y;
  int l_out = blockIdx.z;
  int t     = threadIdx.x;
  int c_in  = t / K;
  int k     = t % K;
  int l_in = l_out * stride - padding + k;


  // weight: [C_out, C_in, K]
  // input: [B, C_in, L]
  // output: [B, C_out, L_out]
  // bias: [C_out]

  float ret = 0.0f;
  if (l_in >= 0 && l_in < L_in)
    ret += weight[c_out * C_in * K + c_in * K + k] * input[b * C_in * L_in + c_in * L_in + l_in];

  smem[threadIdx.x] = ret;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (t < offset) {
      smem[t] += smem[t + offset];
    }
    __syncthreads();
  }

  if (t == 0) {
    output[b * C_out * L_out + c_out * L_out + l_out] = smem[0] + bias[c_out];
  }
}

torch::Tensor conv1d(torch::Tensor input,
  torch::Tensor weight, std::optional<torch::Tensor> bias,
  int stride, int padding) {
  assert(input.dim() == 3); // [B, C_in, L_in]
  assert(weight.dim() == 3); // [C_out, C_in, K]

  assert (input.size(1) == weight.size(1));
  torch::Tensor _bias;
  if (bias.has_value()) {
    _bias = bias.value();
  } else {
    _bias = torch::zeros({weight.size(0)}, input.options());
  }

  int B = input.size(0);
  int C_in = input.size(1);
  int L_in = input.size(2);
  int C_out = weight.size(0);
  int K = weight.size(2);
  int L_out = (L_in + 2 * padding - K) / stride + 1;
  auto out = torch::empty({B, C_out, L_out}, input.options());

  dim3 grid_size(B, C_out, L_out);
  dim3 block_size(C_in * K);
  if (C_in * K > 1024) {
    throw std::runtime_error("C_in * K > 1024");
  }

  conv1d_naive<<<grid_size, block_size>>>(
    input.data_ptr<float>(), weight.data_ptr<float>(),
    _bias.data_ptr<float>(), out.data_ptr<float>(),
    B, C_in, L_in, C_out, L_out, K, stride, padding);
  return out;
}
