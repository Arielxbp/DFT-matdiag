import ttnn
import math
import torch

def to_scalar(t):
    # host round-trip for a reduction result; robust regardless of ttnn's
    # current broadcast rules
    return ttnn.to_torch(t).flatten()[0].item()

def norm_scalar(v):
    # v: (m, 1) column tensor -> Python float
    sq = ttnn.multiply(v, v)
    s = ttnn.sum(sq)
    return math.sqrt(to_scalar(s))

def gram_schmidt(A):
    m, n = A.shape
    cols = []

    q0 = A[:, 0:1]
    inv0 = 1.0 / norm_scalar(q0)
    cols.append(ttnn.multiply(q0, inv0))

    for i in range(1, n):
        qi = A[:, i:i+1]
        for j in range(i):
            qj = cols[j]
            inner = ttnn.matmul(ttnn.transpose(qj, -2, -1), qi) # (1,1) on device
            c = to_scalar(inner) # pull coefficient to host
            proj = ttnn.multiply(qj, c) # scale qj by scalar
            qi = ttnn.subtract(qi, proj)
        inv = 1.0 / norm_scalar(qi)
        cols.append(ttnn.multiply(qi, inv))

    return ttnn.concat(cols, dim=1)

def qr_gs(A, device):
    m, n = A.shape
    Q = gram_schmidt(A)

    R_host = ttnn.to_torch(A).new_zeros((n, n))  # torch tensor, matches A's dtype/device(host)
    Q_torch = ttnn.to_torch(Q)
    A_torch = ttnn.to_torch(A)
    for i in range(n):
        for j in range(i + 1):
            R_host[j, i] = Q_torch[:, j].dot(A_torch[:, i])

    R = ttnn.from_torch(
        R_host,
        dtype=A.dtype,
        layout=A.layout,
        device=device,
    )
    return Q, R

device = ttnn.open_device(device_id=0)
torch_A = torch.randn(32, 32, dtype=torch.float32) * 10  # well-conditioned, O(1)-O(10) magnitude
A = ttnn.from_torch(torch_A.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
Q, R = qr_gs(A, device)
print("Q:", ttnn.to_torch(Q))
print("R:", ttnn.to_torch(R))
ttnn.close_device(device)

