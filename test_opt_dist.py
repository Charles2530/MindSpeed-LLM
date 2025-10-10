import torch
import torch_npu
import torch.distributed as dist
from quant.opt_npu import optimized_mxfp8_e5m2_matmul

def init_distributed():
    dist.init_process_group(backend='hccl')
    local_rank = dist.get_rank()
    torch.npu.set_device(local_rank)

def main():
    init_distributed()

    A = torch.load("grad_output.pt").npu()
    B = torch.load("total_input.pt").npu()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    A_local = A.chunk(world_size, dim=1)[rank]  # 按第一个维度分割A
    B_local = B.chunk(world_size, dim=1)[rank]  # 按第二个维度分割B

    print(f"Rank {rank}: A_shape:{A_local.shape}, grad_max:{torch.max(A_local)}, grad_min:{torch.min(A_local)}")
    # print(f"Rank {rank}: B_shape:{B_local.shape}, input_max:{torch.max(B_local)}, input_min:{torch.min(B_local)}")
    print(f"B_shape:{B.shape}")

    # O_local = optimized_mxfp8_e5m2_matmul(A_local.t(), B_local).to(torch.bfloat16)
    O_local = optimized_mxfp8_e5m2_matmul(A_local.t(), B).to(torch.bfloat16)
    print(f"Rank {rank}: Has NaN values: {torch.isnan(O_local).any()}")
    if torch.isnan(O_local).any():
        torch.save(A_local,"grad_output.pt")
        torch.save(B,"total_input.pt")

    # if rank == 0:
    #     O_list = [torch.empty_like(O_local) for _ in range(world_size)]
    #     dist.all_gather(O_list, O_local)
    #     O = torch.cat(O_list, dim=1)  # 按第一个维度拼接
    #     print(f"Final output shape: {O.shape}")
    # else:
    #     dist.all_gather([O_local], O_local)

if __name__ == "__main__":
    main()
