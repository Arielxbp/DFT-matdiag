# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import torch
import ttl
import ttnn

def from_torch(tensor: torch.Tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

TILE = 32

@ttl.operation(grid=(1, 1))
def qr(A : ttnn.Tensor, Q : ttnn.Tensor, R : ttnn.Tensor, O : ttnn.Tensor) -> None:

    rows = A.shape[0] // TILE
    cols = A.shape[1] // TILE

    A_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    Q_dfb = ttl.make_dataflow_buffer_like(Q, shape=(1, 1), block_count=2)
    R_dfb = ttl.make_dataflow_buffer_like(R, shape=(1, 1), block_count=2)
    O_dfb = ttl.make_dataflow_buffer_like(O, shape=(1, 1), block_count=2)
    U_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def reader():
        for row in range(rows):
            for col in range(cols): 
                with (
                    A_dfb.reserve() as A_blk,
                    Q_dfb.reserve() as Q_blk,
                    R_dfb.reserve() as R_blk,
                    U_dfb.reserve() as U_blk
                ):
                    A_blk = ttl.copy(A[row, col], A_blk)
                    Q_blk = ttl.copy(Q[row, col], Q_blk)
                    R_blk = ttl.copy(R[row, col], R_blk)
                    U_blk = ttl.copy(A[row, col], U_blk)
                    A_blk.wait()
                    Q_blk.wait()
                    R_blk.wait()
                    U_blk.wait()

    @ttl.compute()
    def compute():

        for row in range(rows):
            for col in range(cols):
                with (
                    A_dfb.wait() as A_blk,
                    Q_dfb.wait() as Q_blk,
                    R_dfb.wait() as R_blk,
                    U_dfb.wait() as U_blk,
                    O_dfb.reserve() as O_blk
                ):
                    # Factorize A into Q and R
                    U_blk -= (A_blk @ Q_blk) * Q_blk
                    with (A_dfb.wait() as A_blk, U_dfb.wait() as U_blk):
                        norm = ttl.sqrt(U_blk)
                        Q_blk = U_blk / norm
                        R_blk = A_blk @ Q_blk
                        O_blk.store(R_blk)

                    
        

    @ttl.datamovement()
    def writer():
        for row in range(rows):
            for col in range(cols):
                with (
                    O_dfb.wait() as out_blk
                ):
                    tf = ttl.copy(out_blk, O[row*TILE:(row+1)*TILE, col*TILE:(col+1)*TILE])
                    tf.wait()


def main():
    torch.manual_seed(0)
    
    tensor_size = 1024

    
    device = ttnn.open_device(device_id=0)
    try:

        A = torch.rand((tensor_size, tensor_size), dtype=torch.float32)
        Q = torch.zeros_like(A)
        R = torch.zeros_like(A)
        O = torch.zeros_like(A)
        
        A = from_torch(A, device)
        Q = from_torch(Q, device)
        R = from_torch(R, device)
        O = from_torch(O, device)

        qr(A, Q, R, O)
        O = O.to_torch()
        print("QR eigenvalues:", O)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()