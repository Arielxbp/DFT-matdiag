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


def factor_panel(R_torch, p, b, m):

    V = torch.zeros((m, b), dtype=torch.float32)

    for j in range(b):
        i = p + j

        x = R_torch[:, i:i + 1].clone()
        x[:i, 0] = 0

        norm_x = torch.linalg.norm(x)
        if norm_x == 0:
            continue

        x_i = x[i, 0]
        sgn = torch.sign(x_i) if x_i != 0 else torch.tensor(1.0)
        x[i, 0] = x_i + sgn * norm_x

        v = x / torch.linalg.norm(x)
        V[:, j:j + 1] = v

        # Apply this reflector to the remaining columns of the panel only.
        R_torch[:, i:p + b] -= 2 * (v @ (v.T @ R_torch[:, i:p + b]))

        if i + 1 < m:
            R_torch[i + 1:, i] = 0

    return V, R_torch


def build_block_T(V_torch):

    m, b = V_torch.shape
    T = torch.zeros((b, b), dtype=torch.float32)
    tau = 2.0

    T[0, 0] = tau
    for j in range(1, b):
        z = -tau * (V_torch[:, :j].T @ V_torch[:, j])
        T[:j, j] = T[:j, :j] @ z
        T[j, j] = tau

    return T


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
    Q = get_identity_matrix(m, device)

    p = 0
    while p < n - 1:
        b = min(block_size, (n - 1) - p)

        R_torch = ttnn.to_torch(R).float()

        V_torch, R_torch = factor_panel(R_torch, p, b, m)
        T_torch = build_block_T(V_torch)

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

        QV = ttnn.matmul(Q, V)
        QVT = ttnn.matmul(QV, T)
        update_Q = ttnn.matmul(QVT, Vt)
        Q = ttnn.subtract(Q, update_Q)

        p += b

    Q, R = normalize_diagonal_signs(Q, R, device)

    return Q, R


if __name__ == "__main__":

    device = ttnn.open_device(device_id=0)

    shape = (2048, 2048)

    torch.manual_seed(0)  # For reproducibility

    torch_A = torch.randint(0, 100, (4, 4))

    A = torch_A.clone()

    A = to_tt_tile(A)

    Q, R = ttnn_qr_householder_blocked(A, device, block_size=32)

    print(A)
    print(Q)
    print(R)

    ttnn.close_device(device)