import torch
import torch_npu
from quant.opt_npu import optimized_mxfp8_e5m2_matmul
A = torch.load("grad_output.pt").npu()
print(f"A_shape:{A.shape},grad_max:{torch.max(A)},grad_min:{torch.min(A)}")
B = torch.load("total_input.pt").npu()
print(f"B_shape:{B.shape},input_max:{torch.max(B)},input_min:{torch.min(B)}")
O = optimized_mxfp8_e5m2_matmul(A.t(),B).to(torch.bfloat16)
print(f"O_shape:{O.shape},output_max:{torch.max(O)},output_min:{torch.min(O)}")

print(torch.isnan(O).any())
