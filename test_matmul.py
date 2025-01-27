import time
import torch

import cuda_playground as cp

M, N, K = 512, 512, 32
A = torch.IntTensor(M, N).cuda().random_(0, 10).contiguous()
B = torch.IntTensor(N, K).cuda().random_(0, 10).contiguous()


C_ref = torch.matmul(A.cpu(), B.cpu()).cuda()

for version in [0, 1, 2]:
    C = cp.matmul(A, B, version=version)
    if not torch.equal(C, C_ref):
        print(f"Error in version {version}")
        print(C)
        print(C_ref)
