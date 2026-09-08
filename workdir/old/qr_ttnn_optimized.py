# Updated version of qr_ttnn.py using householder reflections instead of classical Gram-Schmidt

import ttnn
import torch

import cProfile
import pstats
import time

def to_tt_tile(torch_tensor):
   return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Returns a scalar
def ttnn_norm(x):
    # 2-norm of a vector x
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

# def get_householder_vector(x, index, device):
#     norm = ttnn_norm(x)
#     if norm == 0:
#         return x
#     x_i = x[index, 0]

#     if x_i >= 0:
#         sign = 1
#     else:
#         sign = -1

#     x[index, 0] += sign * norm

#     v_norm = ttnn_norm(x)
#     v = ttnn.divide(x, v_norm)
#     return v

def get_householder_vector(x_ttnn, i, device):
    x = ttnn.to_torch(x_ttnn).float()
    print(f"Current column vector x:\n{x}")
    
    # 2. Compute the L2 norm. Zeros above index 'i' do not affect the norm.
    norm_x = torch.linalg.norm(x)
    print(f"L2 norm of x: {norm_x}")
    
    # If the column is already all zeros, no reflection is needed
    if norm_x == 0:
        return x_ttnn
        
    # 3. Extract the diagonal element at index i
    x_i = x[i, 0]
    print(f"Diagonal element x[{i}, 0]: {x_i}")
    
    # 4. Extract the sign (default to 1.0 if exactly 0 to avoid collapsing)
    sgn = torch.sign(x_i) if x_i != 0 else torch.tensor(1.0)
    print(f"Sign of x[{i}, 0]: {sgn}")
    
    # 5. Add the signed norm to the diagonal element to construct the reflection vector
    # This prevents catastrophic cancellation (loss of precision)
    print(f"Adding {sgn * norm_x} to x[{i}, 0]")
    x[i, 0] = x_i + (sgn * norm_x)
    print(f"Updated vector x after adding signed norm:\n{x}")
    
    # 6. Normalize the new vector so that v^T v = 1
    v_norm = torch.linalg.norm(x)
    print(f"Norm of the updated vector x: {v_norm}")
    v = x / v_norm
    print(f"Normalized Householder vector v:\n{v}")
    
    
    # 7. Convert back to bfloat16 and send to the TTNN device
    return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

def zero_out_above_index(x_ttnn, i, device):

    m = x_ttnn.shape[0]

    # Create a mask that has 1s on the diagonal from index i
    buffer = [0] * m * m
    
    for row in range(i, m):
        buffer[row * m + row] = 1

    mask = ttnn.from_buffer(buffer=buffer, shape=[m, m], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    new_col = ttnn.matmul(mask, x_ttnn)
    return new_col

def zero_out_under_diagonal(A, i, device):
    
    pass

def ttnn_qr_householder(A, device):
    m, n = A.shape

    R = ttnn.clone(A)    
    Q = get_identity_matrix(m, device)

    for i in range(n - 1):

        current_col = ttnn.reshape(R[:, i], [m, 1])
        current_col = zero_out_above_index(current_col, i, device)
        
        v = get_householder_vector(current_col, i, device)

        vT = ttnn.transpose(v, 0, 1)
        print(f"vT is {vT}")

        vT_R = ttnn.matmul(vT, R)
        print(f"vT_R is {vT_R}")

        update_R = ttnn.multiply(ttnn.matmul(v, vT_R), 2)
        print(f"update_R is {update_R}")
        R = ttnn.subtract(R, update_R)
        print(f"Updated R after iteration {i}:\n{ttnn.to_torch(R)}")

        R_torch = ttnn.to_torch(R)
        if i +1 < m:
            R_torch[i + 1:, i] = 0
        R = ttnn.from_torch(R_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        print(f"R after zeroing out below diagonal:\n{ttnn.to_torch(R)}")

        Q_v  = ttnn.matmul(Q, v)
        update_Q = ttnn.multiply(ttnn.matmul(Q_v, vT), 2)
        Q = ttnn.subtract(Q, update_Q)

    return Q, R


if __name__ == "__main__":

    device = ttnn.open_device(device_id=0)

    shape = (4, 4)
    num_iterations = 30

    torch.manual_seed(0)  # For reproducibility
    torch_A = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]], dtype=torch.bfloat16) 
    print(f"Original matrix A:\n{torch_A}")

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
    print(f"Total time for {num_iterations} iterations: {elapsed_time:.6f} seconds")

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)