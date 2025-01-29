import torch

from cuda_playground import flash_attention_v1

B, H, N, D = 1, 2, 16, 16

Q = torch.randn(B, H, N, D).to(torch.float16)
K = torch.randn(B, H, N, D).to(torch.float16)
V = torch.randn(B, H, N, D).to(torch.float16)

att_ref = torch._scaled_dot_product_attention_math(Q, K, V, scale=1.0)[0]
att_my = flash_attention_v1(Q, K, V)

# print(att_ref, att_my)