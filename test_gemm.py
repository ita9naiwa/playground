import torch
import os
import sys


from cuda_playground import simple_cutlass_gemm

M, N, K = 2048, 512, 1024
A = torch.FloatTensor(M, K).cuda().random_().normal_().contiguous()
B = torch.FloatTensor(K, N).cuda().random_().normal_().contiguous()


C = simple_cutlass_gemm(A, B, 1, 0)
C_ref = torch.mm(A, B)

print(C - C_ref)