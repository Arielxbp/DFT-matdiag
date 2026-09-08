import numpy as np
import ttl
import ttnn
import torch

TILE_SIZE = 32
GRANULARITY = 4

def from_torch(tensor : torch.Tensor, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

@ttl.operation(grid=(8, 8))
def operation(a : ttnn.Tensor, b : ttnn.Tensor, y : ttnn.Tensor) -> None: # type: ignore

    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY


    grid_cols, grid_rows = ttl.grid_size(dims=2)

    rows = a.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = a.shape[1] // TILE_SIZE // col_tiles_per_block

    # Divide the total blocks per row/col to iterate on
    # with the number of cores in the grid
    rows_per_node = rows // grid_rows
    cols_per_node = cols // grid_cols

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(row_tiles_per_block, col_tiles_per_block), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(row_tiles_per_block, col_tiles_per_block), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(row_tiles_per_block, col_tiles_per_block), block_count=2)

    @ttl.compute()
    def compute():
        for r in range(rows_per_node):
            for c in range(cols_per_node):
                # wait() --> blocks the compute kernel until it can read the filled input blocks
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    y_dfb.reserve() as y_blk
                ):
                    result = a_blk + b_blk
                    y_blk.store(result)


    @ttl.datamovement()
    def read():

        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):


            row = node_row * rows_per_node + local_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for local_col in range(cols_per_node):

                # same as above for columns
                col = node_col * cols_per_node + local_col
                start_col_tile = col * col_tiles_per_block
                end_col_tile = (col + 1) * col_tiles_per_block

                temp = 0
                with (
                    a_dfb.reserve() as a_blk,
                    b_dfb.reserve() as b_blk
                ):
                    temp = ttl.raw_element_read(a_blk, 1, 1)

                    ttl.raw_element_write(b_blk, 1, 1, temp)
                    ttl.raw_element_write(b_blk, 1, 2, temp)
                    
                    transfer_handler_a = ttl.copy(a[start_row_tile:end_row_tile, start_col_tile:end_col_tile], a_blk)
                    transfer_handler_b = ttl.copy(b[start_row_tile:end_row_tile, start_col_tile:end_col_tile], b_blk)
                    transfer_handler_a.wait()
                    transfer_handler_b.wait()
                    

    @ttl.datamovement()
    def write():

        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):

            row = node_row * rows_per_node + local_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for local_col in range(cols_per_node):

                col = node_col * cols_per_node + local_col
                start_col_tile = col * col_tiles_per_block
                end_col_tile = (col + 1) * col_tiles_per_block

                
                with (
                    y_dfb.wait() as y_blk
                ):
                    
                    
                    transfer_handler_y = ttl.copy(y_blk, y[start_row_tile:end_row_tile, start_col_tile:end_col_tile])
                    transfer_handler_y.wait()



def main() -> None:

    device = ttnn.open_device(device_id=0)

    try:
        shape = (2048, 2048)
        a = from_torch(torch.rand(shape, dtype=torch.bfloat16), device)
        b = from_torch(torch.rand(shape, dtype=torch.bfloat16), device)
        y = from_torch(torch.zeros(shape, dtype=torch.bfloat16), device)

        operation(a, b, y)

        print("#"*40 + " Output " + "#"*40 + "\n")

        # Convert the ttnn tensors back to PyTorch tensors for validation
        pt_a = ttnn.to_torch(a)
        pt_b = ttnn.to_torch(b)
        pt_out = ttnn.to_torch(y)

        # Force PyTorch to use bfloat16 just like the Tenstorrent hardware does
        expected_out = (pt_a.to(torch.bfloat16) + pt_b.to(torch.bfloat16)).to(pt_out.dtype)

        # Validate the output against the expected result with abs tolerance and relative tolerance
        is_correct = torch.allclose(pt_out, expected_out, atol=1e-4, rtol=1e-2)

        # Print the validation result
        print("=====================================")
        print(f"MATH VALIDATION PASSED: {is_correct}")
        print("=====================================")

        # If the math fails, format and print the specific differences
        if not is_correct:
            difference = torch.abs(pt_out - expected_out)
            max_error = difference.max().item()

            # Specify the formatting for error inspection
            torch.set_printoptions(precision=4, sci_mode=False, edgeitems=3)

            print(f"Max Error Found: {max_error}")

        print("\n" +"#"*40 + " Output " + "#"*40)
    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    main()
