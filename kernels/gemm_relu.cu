#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

using std::cerr, std::endl;

#include "basic_gemm.h"
#include "helper.h"

cudaError_t CutlassHGemmRelu(int M, int N, int K, float alpha, void *A, int lda, void *B, int ldb, void *C, int ldc,
                             void *bias) {
  // RowMajor로 할거야 -_-

  using ElemType = cutlass::half_t;
  using RowMajor = cutlass::layout::RowMajor;
  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using ReluEpilogue = cutlass::epilogue::thread::LinearCombinationRelu<ElemType, 16, ElemType, ElemType>;

  using Gemm = cutlass::gemm::device::Gemm<ElemType, RowMajor, ElemType, RowMajor, ElemType, RowMajor, ElemType,
                                           cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75, ShapeMMAThreadBlock,
                                           ShapeMMAWarp, ShapeMMAOp, ReluEpilogue, SwizzleThreadBlock,
                                           2  // <- 2 stages in pipeline
                                           >;

  cutlass::TensorRef<ElemType, cutlass::layout::RowMajor> A_tensor_ref((ElemType *)A, RowMajor(K));
  cutlass::TensorRef<ElemType, cutlass::layout::RowMajor> B_tensor_ref((ElemType *)B, RowMajor(N));
  cutlass::TensorRef<ElemType, cutlass::layout::RowMajor> C_tensor_ref((ElemType *)C, RowMajor(N));

  cutlass::TensorRef<ElemType, cutlass::layout::RowMajor> bias_tensor_ref((ElemType *)bias, RowMajor(1));

  auto A_tensor_view = cutlass::TensorView<ElemType, RowMajor>(A_tensor_ref, {M, K});
  auto B_tensor_view = cutlass::TensorView<ElemType, RowMajor>(B_tensor_ref, {K, N});
  auto C_tensor_view = cutlass::TensorView<ElemType, RowMajor>(C_tensor_ref, {M, N});
  auto bias_tensor_view = cutlass::TensorView<ElemType, RowMajor>(bias_tensor_ref, {1, M});

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  Gemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  typename Gemm::Arguments args{problem_size,  A_tensor_ref,
                                B_tensor_ref,  {bias_tensor_ref.data(), 0},
                                C_tensor_ref,  {static_cast<ElemType>(alpha), static_cast<ElemType>(0)},
                                split_k_slices};
  cutlass::Status status = gemm_op.can_implement(args);
  CUTLASS_CHECK(status);

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  status = gemm_op.initialize(args, workspace.get());
  CUTLASS_CHECK(status);

  status = gemm_op();  // Launch GEMM on device
  if (status == cutlass::Status::kSuccess) {
    return cudaSuccess;
  } else {
    return cudaErrorUnknown;
  }
}

torch::Tensor cutlass_half_gemm_relu(const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &bias,
                                     float alpha) {
  assert(A.scalar_type() == at::kHalf);
  assert(B.scalar_type() == at::kHalf);
  assert(bias.scalar_type() == at::kHalf);

  int M = A.size(0);
  int N = B.size(1);
  int K = A.size(1);
  auto options = A.options();
  torch::Tensor C = torch::zeros({M, N}, options);
  cudaError_t err =
      CutlassHGemmRelu(M, N, K, alpha, static_cast<void *>(A.data_ptr()), K, static_cast<void *>(B.data_ptr()), N,
                       static_cast<void *>(C.data_ptr()), N, static_cast<void *>(bias.data_ptr()));

  if (cudaSuccess != err) {
    cerr << "Error in CutlassSgemmNN: " << cudaGetErrorString(err) << endl;
  }

  return C;
}
