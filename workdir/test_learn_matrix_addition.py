import numpy as np
import ttl
import ttnn
import torch

TILE_SIZE = 4

# Function to convert a PyTorch tensor to a ttnn tensor with specified dtype, layout, device, and memory configuration
def from_torch(tensor : torch.Tensor, device):
    return ttnn.from_torch(
        tensor, 
        dtype=ttnn.bfloat16, 
        layout=ttnn.TILE_LAYOUT, 
        device=device, 
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

@ttl.operation(grid=(1, 1))
def operation(a : ttnn.Tensor, b : ttnn.Tensor, y : ttnn.Tensor) -> None:

    rows = a.shape[0] // TILE_SIZE
    cols = a.shape[1] // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, y_dfb.reserve() as y_blk:
                    print(y_blk, thread="pack")
                    y_blk.store(a_blk + b_blk)
                    

    @ttl.datamovement()
    def read():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    tx_a = ttl.copy(a[r, c], a_blk)
                    tx_b = ttl.copy(b[r, c], b_blk)
                    tx_a.wait()
                    tx_b.wait()

    @ttl.datamovement()
    def write():
        for r in range(rows):
            for c in range(cols):
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(y_blk, y[r, c])
                    tx.wait()


def main() -> None:

    device = ttnn.open_device(device_id=0)

    try:
        shape = (4, 4)
        a = from_torch(torch.rand(shape, dtype=torch.bfloat16), device)
        b = from_torch(torch.rand(shape, dtype=torch.bfloat16), device)
        y = from_torch(torch.zeros(shape, dtype=torch.bfloat16), device)

        

        operation(a, b, y)

        print("#"*40 + " Output " + "#"*40 + "\n")
        print("Input tensor a:")
        print(ttnn.to_torch(a).float())
        print("Input tensor b:")
        print(ttnn.to_torch(b).float())
        print("Output tensor y:")
        print(ttnn.to_torch(y).float())

        print("\n" +"#"*40 + " Output " + "#"*40)
    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    main()