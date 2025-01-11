import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# cuda_root = "/usr/local/cuda-12.6/include"
cutlass_root = os.path.expanduser("~/src/cutlass/include/")
cutlass_util = os.path.expanduser("~/src/cutlass/tools/util/include")
build_ext = BuildExtension.with_options(
    use_ninja=True,
    cmake_args=["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]  # Ask CMake to emit compile_commands.json
)

setup(
    name='cuda_playground',
    ext_modules=[
        CUDAExtension(
            name='cuda_playground',
            sources=[
                'kernels/basic_gemm.cu',
                'kernels/gemm_relu.cu',
                'pybind.cu'
            ],
            include_dirs=[
                # cuda_root,
                cutlass_root,
                cutlass_util,
                os.path.abspath("./include"),
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-gencode=arch=compute_89,code=sm_89',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': build_ext},
)