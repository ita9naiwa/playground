import torch

from cuda_playground import flash_attention_v1


def test_validation(B=1, H=1, N=32, D=64):
    Q = torch.randn(B, H, N, D).to(torch.float16).cuda()
    K = torch.randn(B, H, N, D).to(torch.float16).cuda()
    V = torch.randn(B, H, N, D).to(torch.float16).cuda()

    att_ref = torch._scaled_dot_product_attention_math(Q, K, V, scale=1.0)[0].cpu()
    att_my = flash_attention_v1(Q, K, V).cpu()

    if (torch.allclose(att_ref, att_my, atol=1e-2, rtol=0.1)):
        print("Test passed when B,H,N,D = ", B, H, N, D)
    else:
        print("Test failed when B,H,N,D = ", B, H, N, D)
        print(att_ref / att_my)


B, H, N, D = 1, 1, 64, 64

test_validation(B, H, N, D)


def test_perf():
    B, H, N, D = 128, 64, 64, 64
    Q = torch.randn(B, H, N, D).to(torch.float16).cuda()
    K = torch.randn(B, H, N, D).to(torch.float16).cuda()
    V = torch.randn(B, H, N, D).to(torch.float16).cuda()

    import time
    start = time.time()
    for i in range(100):
        torch._scaled_dot_product_attention_math(Q, K, V, scale=1.0)[0]
    print("Torch: Time taken for 100 iterations: ", time.time() - start)

    start = time.time()
    for i in range(100):
        flash_attention_v1(Q, K, V)
    print("Mine: Time taken for 100 iterations: ", time.time() - start)


test_perf()