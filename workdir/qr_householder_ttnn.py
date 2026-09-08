# Updated version of qr_ttnn.py using householder reflections instead of classical Gram-Schmidt

import ttnn
import torch

import cProfile
import pstats
import time

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

    buffer = [0] * m * m
    
    for row in range(i, m):
        buffer[row * m + row] = 1

    mask = ttnn.from_buffer(buffer=buffer, shape=[m, m], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    new_col = ttnn.matmul(mask, x_ttnn)
    return new_col

def ttnn_qr_householder(A, device):
    m, n = A.shape

    R = ttnn.clone(A)    
    Q = get_identity_matrix(m, device)

    for i in range(n - 1):

        current_col = ttnn.reshape(R[:, i], [m, 1])
        current_col = zero_out_above_index(current_col, i, device)
        
        v = get_householder_vector(current_col, i, device)

        vT = ttnn.transpose(v, 0, 1)

        vT_R = ttnn.matmul(vT, R)

        update_R = ttnn.multiply(ttnn.matmul(v, vT_R), 2)

        R = ttnn.subtract(R, update_R)

        R_torch = ttnn.to_torch(R)
        if i +1 < m:
            R_torch[i + 1:, i] = 0
        R = ttnn.from_torch(R_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        Q_v  = ttnn.matmul(Q, v)
        update_Q = ttnn.multiply(ttnn.matmul(Q_v, vT), 2)
        Q = ttnn.subtract(Q, update_Q)

    return Q, R


if __name__ == "__main__":

    device = ttnn.open_device(device_id=0)

    shape = (2048, 2048)

    torch_A = torch.randint(0, 32, (2048, 2048))

    A = torch_A.clone()

    A = to_tt_tile(A)

    start_time = time.perf_counter()
    with cProfile.Profile() as pr:

        Q, R = ttnn_qr_householder(A, device)

    end_time = time.perf_counter()

    print(A)
    print(Q)
    print(R)

    ttnn.close_device(device)

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)