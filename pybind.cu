#include <torch/extension.h>

#include <vector>
#include <iostream>

using std::cerr, std::endl;


#include "cuda_runtime.h"
#include "basic_gemm.h"

std::vector<torch::Tensor> virtually_anything(
  torch::Tensor &A,
  torch::Tensor &B
);

torch::Tensor simple_cutlass_gemm(
    torch::Tensor &A,
    torch::Tensor &B,
    float alpha,
    float beta) {
  int64_t M = A.size(0);
  int64_t N = B.size(1);
  int64_t K = A.size(1);
  auto options = A.options();
  torch::Tensor C = torch::zeros({M, N}, options);

  const float* A_ptr = A.data_ptr<float>();
  const float* B_ptr = B.data_ptr<float>();
  float* C_ptr = C.data_ptr<float>();

  int lda = int(M);
  int ldb = int(K);
  int ldc = int(M);

  cudaError_t err = CutlassSgemmNN(
    (int)M, (int)N, (int)K,
    alpha,
    A_ptr, lda,
    B_ptr, ldb,
    beta,
    C_ptr, ldc
  );
  if (cudaSuccess != err) {
    cerr << "Error in CutlassSgemmNN: " << cudaGetErrorString(err) << endl;
  }
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_cutlass_gemm", &simple_cutlass_gemm, "gemm",
  py::arg("A"),
  py::arg("B"),
  py::arg("alpha") = 1.0F,
  py::arg("beta") = 0.0F);
};