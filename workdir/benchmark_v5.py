import ttnn
import torch
import math

import cProfile
import pstats
import time
import sys

def to_tt_tile(torch_tensor, device):
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


def factor_panel(R_torch, p, b, m):
    R64 = R_torch.double()
    V = torch.zeros((m, b), dtype=torch.float64)

    for j in range(b):
        i = p + j

        x = R64[:, i:i + 1].clone()
        x[:i, 0] = 0

        norm_x = torch.linalg.norm(x)
        if norm_x == 0:
            continue

        x_i = x[i, 0]
        sgn = torch.sign(x_i) if x_i != 0 else torch.tensor(1.0, dtype=torch.float64)
        x[i, 0] = x_i + sgn * norm_x

        v = x / torch.linalg.norm(x)
        V[:, j:j + 1] = v

        R64[:, i:p + b] -= 2 * (v @ (v.T @ R64[:, i:p + b]))

        if i + 1 < m:
            R64[i + 1:, i] = 0

    return V.float(), R64.float()

def build_block_T(V_torch):

    V64 = V_torch.double()
    m, b = V64.shape
    T = torch.zeros((b, b), dtype=torch.float64)
    tau = 2.0

    T[0, 0] = tau
    for j in range(1, b):
        z = -tau * (V64[:, :j].T @ V64[:, j])
        T[:j, j] = T[:j, :j] @ z
        T[j, j] = tau

    return T.float()

def normalize_diagonal_signs(Q, R, device):

    R_torch = ttnn.to_torch(R).float()
    signs = torch.where(torch.diagonal(R_torch) < 0, -1.0, 1.0)

    D_torch = torch.diag(signs).to(torch.bfloat16)
    D = ttnn.from_torch(D_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    R = ttnn.matmul(D, R)
    Q = ttnn.matmul(Q, D)

    return Q, R

def ttnn_qr_householder_blocked(A, device, block_size=32):

    m, n = A.shape

    R = ttnn.clone(A)

    reflectors = []

    p = 0
    while p < n - 1:
        b = min(block_size, (n - 1) - p)

        R_torch = ttnn.to_torch(R).float()

        V_torch, R_torch = factor_panel(R_torch, p, b, m)
        T_torch = build_block_T(V_torch)

        reflectors.append((V_torch, T_torch, p, b))

        R = ttnn.from_torch(R_torch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        V = ttnn.from_torch(V_torch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        T = ttnn.from_torch(T_torch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        Vt = ttnn.transpose(V, 0, 1)
        Tt = ttnn.transpose(T, 0, 1)
        VtR = ttnn.matmul(Vt, R)
        TtVtR = ttnn.matmul(Tt, VtR)
        update_R = ttnn.matmul(V, TtVtR)

        trail_mask = torch.zeros((m, n), dtype=torch.bfloat16)
        trail_mask[:, p + b:] = 1
        trail_mask_ttnn = ttnn.from_torch(trail_mask, layout=ttnn.TILE_LAYOUT, device=device)

        update_R = ttnn.multiply(update_R, trail_mask_ttnn)
        R = ttnn.subtract(R, update_R)

        p += b

    Q = get_identity_matrix(m, device)

    for V_torch, T_torch, p, b in reversed(reflectors):
        V = ttnn.from_torch(V_torch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        T = ttnn.from_torch(T_torch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        Vt = ttnn.transpose(V, 0, 1)

        QV = ttnn.matmul(Q, V)
        QVT = ttnn.matmul(QV, T)
        update_Q = ttnn.matmul(QVT, Vt)
        Q = ttnn.subtract(Q, update_Q)

    Q, R = normalize_diagonal_signs(Q, R, device)

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

    A = to_tt_tile(torch_A, device)

    times = []

    for _ in range(num_iterations):
        start_time = time.perf_counter()
        with cProfile.Profile() as pr:

            Q, R = ttnn_qr_householder_blocked(A, device, block_size=32)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # torch.set_printoptions(profile="full")

    print(ttnn.to_torch(A))

    print("TTNN QR decomposition:")
    print(ttnn.to_torch(Q))
    print(ttnn.to_torch(R))

    # compare with PyTorch's QR decomposition
    print("PyTorch QR decomposition:")
    Q_torch, R_torch = torch.linalg.qr(torch_A.float())
    print(Q_torch.to(torch.float32))
    print(R_torch.to(torch.float32))
    
    torch.set_printoptions(profile="default")

    recon_error = (ttnn.to_torch(Q) @ ttnn.to_torch(R) - torch_A).abs().max()
    orthogonality_error = (ttnn.to_torch(Q).T @ ttnn.to_torch(Q) - torch.eye(shape[0])).abs().max()

    print(f"Reconstruction error: {recon_error}")
    print(f"Orthogonality error: {orthogonality_error}")

    recon_error_torch = (Q_torch @ R_torch - torch_A.float()).abs().max()
    orthogonality_error_torch = (Q_torch.T @ Q_torch - torch.eye(shape[0])).abs().max()

    print(f"PyTorch Reconstruction error: {recon_error_torch}")
    print(f"PyTorch Orthogonality error: {orthogonality_error_torch}")

    eps = 0.0078
    scale = torch_A.abs().max().item()
    n = A.shape[0]

    # reconstruction: error should be roughly within a small multiple of eps * scale,
    # growing slowly (~sqrt(n) to n) due to accumulated rounding across matmuls
    tol_recon = 5 * eps * scale * math.sqrt(n)

    # orthogonality: entries of Q^T Q are O(1), so tolerance is just a few eps
    tol_ortho = 5 * eps

    print(f"Estimated tolerance for reconstruction error: {tol_recon}")
    print(f"Estimated tolerance for orthogonality error: {tol_ortho}")

    ttnn.close_device(device)

    for i in range(num_iterations):
        print(f"Time for run {i+1}: {times[i]:.6f} seconds")
    print(f"Average time: {sum(times) / len(times):.6f} seconds")

    print("Last iteration profiling stats:")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)
