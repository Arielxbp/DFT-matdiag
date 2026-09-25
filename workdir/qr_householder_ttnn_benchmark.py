import ttnn
import torch

import cProfile
import pstats
import time
import sys

def to_tt_tile(torch_tensor):
   return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

def ttnn_norm(x):

    square = ttnn.square(x)

    sum = ttnn.sum(square)

    sqrt = ttnn.sqrt(sum)

    return sqrt

def get_identity_matrix(n, device):
    buffer = [0] * (n * n)
    for i in range(n):
        buffer[i * n + i] = 1

    identity_matrix = ttnn.from_buffer(buffer=buffer, shape=[n, n], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return identity_matrix

def get_householder_vector(x_ttnn, i, device):
    x = ttnn.to_torch(x_ttnn).float()

    norm_x = torch.linalg.norm(x)

    if norm_x == 0:
        return x_ttnn

    x_i = x[i, 0]

    sgn = torch.sign(x_i) if x_i != 0 else torch.tensor(1.0)

    x[i, 0] = x_i + (sgn * norm_x)

    v_norm = torch.linalg.norm(x)

    v = x / v_norm

    return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

def zero_out_above_index(x_ttnn, i, device):

    m = x_ttnn.shape[0]

    # Create the mask directly in PyTorch
    mask_vec = torch.zeros((m, 1), dtype=torch.bfloat16)
    mask_vec[i:, 0] = 1
    
    # Push to device
    mask_ttnn = ttnn.from_torch(mask_vec, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.multiply(x_ttnn, mask_ttnn)

def ttnn_qr_householder(A, device):

    start = time.perf_counter()

    m, n = A.shape

    R = ttnn.clone(A)    
    Q = get_identity_matrix(m, device)

    end = time.perf_counter()
    print(f"Time taken for initialization: {end - start:.6f} seconds")

    tot_start = time.perf_counter()
    for i in range(n - 1):

        start = time.perf_counter()

        current_col = ttnn.reshape(R[:, i], [m, 1])
        current_col = zero_out_above_index(current_col, i, device)
        
        v = get_householder_vector(current_col, i, device)

        vT = ttnn.transpose(v, 0, 1)

        vT_R = ttnn.matmul(vT, R)

        update_R = ttnn.multiply(ttnn.matmul(v, vT_R), 2)

        R = ttnn.subtract(R, update_R)

        mask = torch.ones((m, n), dtype=torch.bfloat16)
        if i + 1 < m:
            mask[i + 1:, i] = 0

        mask_ttnn = ttnn.from_torch(mask, layout=ttnn.TILE_LAYOUT, device=device)

        R = ttnn.multiply(R, mask_ttnn)

        Q_v  = ttnn.matmul(Q, v)
        update_Q = ttnn.multiply(ttnn.matmul(Q_v, vT), 2)
        Q = ttnn.subtract(Q, update_Q)
        end = time.perf_counter()
        print(f"Time taken for iteration {i}: {end - start:.6f} seconds")
    tot_end = time.perf_counter()
    print(f"Total time taken for QR decomposition: {tot_end - tot_start:.6f} seconds")
    return Q, R


if __name__ == "__main__":


    device = ttnn.open_device(device_id=0)

    # Set the shape from the passed argument
    if len(sys.argv) > 1:
        shape_arg = sys.argv[1]
        shape = tuple(map(int, shape_arg.split(','))) 
    else:
        shape = (32, 32)

    # Set the dtype to use from the passed argument
    if len(sys.argv) > 2:
        dtype_arg = sys.argv[2]
        if dtype_arg == "float32":
            matrix_dtype = torch.float32
        elif dtype_arg == "bfloat16":
            matrix_dtype = torch.bfloat16
        elif dtype_arg == "int32":
            matrix_dtype = torch.int32
        else:
            raise ValueError("Unsupported dtype.")

    if len(sys.argv) > 3:
        num_iterations = int(sys.argv[3])
    else:
        num_iterations = 1

    # example: func.py 1024,1024 int32 5

    torch.manual_seed(0)  # For reproducibility

    if matrix_dtype == torch.int32:
        torch_A = torch.randint(0, 100, shape)
    else:
        torch_A = torch.rand(shape, dtype=matrix_dtype) * 100

    A = to_tt_tile(torch_A)

    times = []
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        with cProfile.Profile() as pr:

            Q, R = ttnn_qr_householder(A, device)

        end_time = time.perf_counter()
        times.append(end_time - start_time)


    print(A)
    print(Q)
    print(R)


    ttnn.close_device(device)

    for i in range(num_iterations):
        print(f"Time for run {i+1}: {times[i]:.6f} seconds")
    print(f"Average time: {sum(times) / len(times):.6f} seconds")

    print("Last iteration profiling stats:")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)