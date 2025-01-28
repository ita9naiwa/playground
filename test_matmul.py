import time
import torch

import cuda_playground as cp


def verify():
    M, N, K = 17, 28, 31
    A = torch.IntTensor(M, N).cuda().random_(0, 10).contiguous()
    B = torch.IntTensor(N, K).cuda().random_(0, 10).contiguous()
    C_ref = torch.matmul(A.cpu(), B.cpu()).cuda()
    for version in [0, 1]:
        C = cp.matmul(A, B, version=version)
        if not torch.equal(C, C_ref):
            print(f"Error in version {version}")
            print(C)
            print(C_ref)


verify()

torch.cuda.empty_cache()

M, N, K = 16384, 8192, 16384
A = torch.IntTensor(M, N).cuda().random_(0, 10).contiguous()
B = torch.IntTensor(N, K).cuda().random_(0, 10).contiguous()
C_holder = torch.IntTensor(M, K).cuda().zero_().contiguous()
for version in [0, 1]:
    prev = time.time()
    for _ in range(10):
        C = cp.matmul(A, B, C=C_holder, version=version, B_transposed=False)
    now = time.time()
    print(f"Version {version}:  {1000.0 * (now - prev):.8f} seconds")
    torch.cuda.empty_cache()

