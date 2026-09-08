
import ttl
import ttnn
import torch


def from_torch(tensor: torch.Tensor):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

TILE_SIZE = 32

@ttl.operation(grid=(1, 1))
def op(
    a: ttnn.Tensor, b: ttnn.Tensor
):

    # Compute iteration counts in tile coordinates.

    rows = a.shape[0] // TILE_SIZE
    cols = a.shape[1] // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def read():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.reserve() as b_blk,
                ):
                    temp = ttl.raw_element_read(a_blk, 1, 1)
                    ttl.raw_element_write(b_blk, 1, 1, temp)
                    ttl.raw_element_write(b_blk, 1, 2, temp)



    # The second DM kernel writes computed output tiles from L1 back to DRAM.

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with b_dfb.wait() as b_blk:
                    # Copy the computed tile from L1 to the output DRAM tensor.
                    tx = ttl.copy(
                        b_blk,
                        b[row, col],
                    )
                    tx.wait()


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

try:
    shape = (32, 32)

    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.zeros(shape, dtype=torch.bfloat16)
    print(a)

    a = from_torch(a)
    b = from_torch(b)

    op(a, b)

    b = ttnn.to_torch(b)
    print(b)


finally:
    ttnn.close_device(device)
