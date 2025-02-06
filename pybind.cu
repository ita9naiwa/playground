#include <driver_types.h>
#include <torch/extension.h>

#include "basic_gemm.h"
#include "flash_attn.h"
#include "revisit_matmul.h"
#include "wmma_matmul.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_cutlass_gemm", &simple_cutlass_gemm, "gemm", py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0F,
        py::arg("beta") = 0.0F);

  m.def("cutlass_half_gemm_relu", &cutlass_half_gemm_relu, "gemm", py::arg("A"), py::arg("B"), py::arg("bias"),
        py::arg("alpha") = 1.0F);

  m.def("matmul", &matmul, "matmul", py::arg("A"), py::arg("B"), py::arg("C") = py::none(), py::arg("version") = 0,
        py::arg("B_transposed") = false);

  m.def("flash_attention_v1", &flash_attention_v1, "flash_attention_v1", py::arg("Q"), py::arg("K"), py::arg("V"));
};
