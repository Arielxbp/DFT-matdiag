# Updated version of qr_ttnn.py using householder reflections instead of classical Gram-Schmidt

import ttnn
import torch

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

        mask = torch.ones((m, n), dtype=torch.bfloat16)
        if i + 1 < m:
            mask[i + 1:, i] = 0

        mask_ttnn = ttnn.from_torch(mask, layout=ttnn.TILE_LAYOUT, device=device)

        R = ttnn.multiply(R, mask_ttnn)

        Q_v  = ttnn.matmul(Q, v)
        update_Q = ttnn.multiply(ttnn.matmul(Q_v, vT), 2)
        Q = ttnn.subtract(Q, update_Q)

    return Q, R


if __name__ == "__main__":

    device = ttnn.open_device(device_id=0)

    shape = (2048, 2048)

    torch.manual_seed(0) # For reproducibility

    torch_A = torch.randint(0, 100, (4, 4))

    A = torch_A.clone()

    A = to_tt_tile(A)

    Q, R = ttnn_qr_householder(A, device)

    ttnn.close_device(device)
