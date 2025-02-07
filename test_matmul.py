import time
import torch

import cuda_playground as cp


def verify():
    M, N, K = 17, 28, 31
    A = torch.HalfTensor(M, N).cuda().random_(0, 10).contiguous()
    B = torch.HalfTensor(N, K).cuda().random_(0, 10).contiguous()
    C_ref = torch.matmul(A.cpu(), B.cpu()).cuda()
    for version in [0, 1, 2]:
        C = cp.matmul(A, B, version=version)
        if not torch.equal(C, C_ref):
            print(f"Error in version {version}")
            print(C)
            print(C_ref)
        else:
            print("Version", version, "passed")


verify()

torch.cuda.empty_cache()

M, N, K = 1024, 16384, 8192

A = torch.HalfTensor(M, N).cuda().random_(0, 1).contiguous()
B = torch.HalfTensor(N, K).cuda().random_(0, 1).contiguous()
C_holder = torch.HalfTensor(M, K).cuda().zero_().contiguous()
for version in [2, 1, 0]:
    prev = time.time()
    for _ in range(100):
        cp.matmul(A, B, C=C_holder, version=version, B_transposed=False)
    now = time.time()
    torch.cuda.empty_cache()

    print(f"Version {version}:  {1000.0 * (now - prev):.8f} miliseconds")

