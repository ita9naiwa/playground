import torch
import os
import sys


from cuda_playground import simple_cutlass_gemm

M, N, K = 1024, 32, 512
A = torch.FloatTensor(M, K).cuda().random_().normal_().contiguous()
B = torch.FloatTensor(K, N).cuda().random_().normal_().contiguous()


C = simple_cutlass_gemm(A, B, 1, 0)
C_ref = torch.mm(A, B)
if torch.allclose(C, C_ref, rtol=1e-4, atol=1e-4):
    print("Test passed")
