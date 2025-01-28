#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <iostream>

#include "basic_gemm.h"
using std::cerr, std::endl;

cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb,
                           float beta, float *C, int ldc) {
  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,      // Data-type of A matrix
                                                  RowMajor,   // Layout of A matrix
                                                  float,      // Data-type of B matrix
                                                  RowMajor,   // Layout of B matrix
                                                  float,      // Data-type of C matrix
                                                  RowMajor>;  // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args(
      {M, N, K},       // Gemm Problem dimensions
      {A, lda},        // Tensor-ref for source matrix A
      {B, ldb},        // Tensor-ref for source matrix B
      {C, ldc},        // Tensor-ref for source matrix C
      {C, ldc},        // Tensor-ref for destination matrix D (may be different memory than source C matrix)
      {alpha, beta});  // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //

  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //
  if (status != cutlass::Status::kSuccess) {
    cerr << cutlassGetStatusString(status) << endl;
  }
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

torch::Tensor simple_cutlass_gemm(torch::Tensor &A, torch::Tensor &B, float alpha, float beta) {
  int64_t M = A.size(0);
  int64_t N = B.size(1);
  int64_t K = A.size(1);
  auto options = A.options();
  torch::Tensor C = torch::zeros({M, N}, options);

  const float *A_ptr = A.data_ptr<float>();
  const float *B_ptr = B.data_ptr<float>();
  float *C_ptr = C.data_ptr<float>();

  int lda = int(K);
  int ldb = int(N);
  int ldc = int(N);

  cudaError_t err = CutlassSgemmNN((int)M, (int)N, (int)K, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);
  if (cudaSuccess != err) {
    cerr << "Error in CutlassSgemmNN: " << cudaGetErrorString(err) << endl;
  }
  return C;
}
