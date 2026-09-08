# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import torch
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 4


@ttl.operation(grid=(1, 1))
def qr_iteration(input_matrix: ttnn.Tensor, v_in: ttnn.Tensor,
                  a_out: ttnn.Tensor, v_out: ttnn.Tensor) -> None:
    
    # Granularity indicates how many tiles will a block contain in each dimension
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    # obtain the grid size, i.e., the number of Tensix in each dimension
    grid_cols, grid_rows = ttl.grid_size(dims=2)

    # Define the block size based on the granularity
    # e.g., granularity=4 -> each block will contain 4x4 tiles
    block_dim = (row_tiles_per_block, col_tiles_per_block)

    # Calculate the number of iterations needed to cover the input matrix
    # In this case it will iterate over the number of blocks
    # e.g., 2048x2048 matrix
    # -> 2048 / 32 = 64 tiles in each dimension
    # -> 64 / 4 = 16 blocks in each dimension
    rows = input_matrix.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = input_matrix.shape[1] // TILE_SIZE // col_tiles_per_block

    # Divide the total blocks per row/col to iterate on 
    # with the number of cores in the grid 
    rows_per_node = rows // grid_rows
    cols_per_node = cols // grid_cols

    # Create dfbs where each block will contain a 4x4 tile of the input matrix
    input_dfb = ttl.make_dataflow_buffer_like(input_matrix, shape=block_dim, block_count=2)
    q_dfb = ttl.make_dataflow_buffer_like(input_matrix, shape=block_dim, block_count=2)
    r_dfb = ttl.make_dataflow_buffer_like(input_matrix, shape=block_dim, block_count=2)
    temp_dfb = ttl.make_dataflow_buffer_like(input_matrix, shape=block_dim, block_count=2)

    @ttl.datamovement()
    def read():
        
        # Get the node's coordinates in the grid
        node_col, node_row = ttl.node(dims=2)

        # Iterate over the number of blocks assigned to this node
        for local_row in range(rows_per_node):

            # Calculate the index of tiles to copy in the matrix
            row = node_row * rows_per_node + local_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for local_col in range(cols_per_node):

                col = node_col * cols_per_node + local_col
                start_col_tile = col * col_tiles_per_block
                end_col_tile = (col + 1) * col_tiles_per_block

                with (
                    input_dfb.reserve() as input_blk,
                ):
                    th_input = ttl.copy(input_matrix[start_row_tile:end_row_tile, start_col_tile:end_col_tile], input_blk)
                    th_input.wait()


    @ttl.compute()
    def compute():
        
        for r in range(rows_per_node):
            for c in range(cols_per_node):
                with (
                    input_dfb.wait() as input_blk,
                    q_dfb.wait() as q_blk,
                    r_dfb.wait() as r_blk,
                    temp_dfb.reserve() as temp_blk
                ):
                    for col in range(input_blk.shape[1]):
                        v_j = input_blk[:, col].clone()
                        for i in range(col):
                            r_blk[i, col] = q_blk[:, i] @ v_j
                            v_j -= r_blk[i, col] * q_blk[:, i]
                        r_blk[col, col] = torch.norm(v_j)
                        q_blk[:, col] = v_j / r_blk[col, col]
        
        # input = R @ Q
        for r in range(rows_per_node):
            for c in range(cols_per_node):
                with (
                    r_dfb.wait() as r_blk,
                    q_dfb.wait() as q_blk,
                    temp_dfb.reserve() as temp_blk
                ):
                    temp_blk.copy_(r_blk @ q_blk)
        


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
                    input_dfb.wait() as input_blk,
                ):
                    th_input = ttl.copy(input_blk, input_matrix[start_row_tile:end_row_tile, start_col_tile:end_col_tile])
                    th_input.wait()


def main() -> None:
    torch.manual_seed(0)
    
    # Exact same 3x3 symmetric integer matrix from the Python example,
    # parsed directly as floats for the tensor arithmetic logic.
    A = torch.tensor([
        [4.0, 1.0, 2.0],
        [1.0, 5.0, 0.0],
        [2.0, 0.0, 3.0]
    ])

    device = ttnn.open_device(device_id=0)
    try:

        # Initialize Q and R matrices with zeros
        Q = torch.zeros_like(A)
        R = torch.zeros_like(A)

        print("QR eigenvalues:    ")
        print("Golden eigenvalues:")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()