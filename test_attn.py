import torch

from cuda_playground import flash_attention_v1

B, H, N, D = 1, 1, 32, 64


def test(B=1, H=1, N=32, D=64):
    Q = torch.randn(B, H, N, D).to(torch.float16).cuda()
    K = torch.randn(B, H, N, D).to(torch.float16).cuda()
    V = torch.randn(B, H, N, D).to(torch.float16).cuda()

    att_ref = torch._scaled_dot_product_attention_math(Q, K, V, scale=1.0)[0].cpu()
    att_my = flash_attention_v1(Q, K, V).cpu()

    if (torch.allclose(att_ref, att_my, atol=1e-2, rtol=0.1)):
        print("Test passed when B,H,N,D = ", B, H, N, D)
    else:
        print("Test failed when B,H,N,D = ", B, H, N, D)
        print((att_ref / att_my).numpy())


test(1, 1, 16, 32)
test(1, 1, 32, 32)