import torch
import os
import sys


from cuda_playground import cutlass_half_gemm_relu

M, N, K = 4, 2, 4
A = torch.HalfTensor(M, K).cuda().random_().normal_().contiguous()
B = torch.HalfTensor(K, N).cuda().random_().normal_().contiguous()
bias = torch.HalfTensor(N).cuda().random_().normal_().contiguous()


C = cutlass_half_gemm_relu(A, B, bias)
C_ref = torch.relu(torch.mm(A, B) + bias)
