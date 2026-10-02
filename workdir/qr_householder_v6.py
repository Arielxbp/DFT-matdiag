import math
import sys
import time
import cProfile
import pstats

import ttnn
import torch

COMPUTE_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

def to_dev(t, device):
    return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

def matmul(a, b):
    return ttnn.matmul(a, b, compute_kernel_config=COMPUTE_CFG)

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

def apply_sign_normalization(Q, R):

    Q, R = Q.clone(), R.clone()
    s = torch.where(torch.diagonal(R) < 0, -1.0, 1.0).to(R.dtype)
    r = s.numel()
    Q[:, :r] *= s
    R[:r, :] *= s[:, None]
    return Q, R

def normalize_diagonal_signs(Q, R, device):
    Q_t, R_t = apply_sign_normalization(ttnn.to_torch(Q).float(), ttnn.to_torch(R).float())
    return to_dev(Q_t, device), to_dev(R_t, device)

def ttnn_qr_householder_blocked(A, device, block_size=32):
    m, n = A.shape

    R = ttnn.clone(A)
    Q = to_dev(torch.eye(m), device)

    k = min(m - 1, n)
    p = 0
    while p < k:
        b = min(block_size, k - p)

        R_torch = ttnn.to_torch(R).float()
        V_torch, R_torch = factor_panel(R_torch, p, b, m)
        T_torch = build_block_T(V_torch)

        R = to_dev(R_torch, device)
        V = to_dev(V_torch, device)
        T = to_dev(T_torch, device)

        Vt = ttnn.transpose(V, 0, 1)
        Tt = ttnn.transpose(T, 0, 1)
        update_R = matmul(V, matmul(Tt, matmul(Vt, R)))

        trail_mask = torch.zeros((m, n))
        trail_mask[:, p + b:] = 1
        update_R = ttnn.multiply(update_R, to_dev(trail_mask, device))
        R = ttnn.subtract(R, update_R)

        update_Q = matmul(matmul(matmul(Q, V), T), Vt)
        Q = ttnn.subtract(Q, update_Q)

        p += b

    return normalize_diagonal_signs(Q, R, device)

def measure_device_eps(device):

    X = 1.0 + torch.rand((32, 32), generator=torch.Generator().manual_seed(1))
    Y = ttnn.to_torch(matmul(to_dev(X, device), to_dev(torch.eye(32), device)))
    X, Y = X.double(), Y.double()
    rel = ((Y - X).abs() / X).max().item()
    return max(rel, torch.finfo(torch.float32).eps)

def qr_metrics(A, Q, R):

    A, Q, R = A.double(), Q.double(), R.double()
    m, n = A.shape
    fro = lambda X: torch.linalg.matrix_norm(X, ord="fro")
    I = torch.eye(m, dtype=torch.float64)
    return {
        "recon": (fro(Q @ R - A) / fro(A)).item(),             # ||QR - A||_F / ||A||_F
        "ortho": torch.linalg.matrix_norm(Q.T @ Q - I, ord=2).item(),  # ||QtQ - I||_2
        "tri": (fro(R.tril(-1)) / fro(R)).item(),               # ||tril(R,-1)||_F / ||R||_F
    }


def qr_tolerances(m, n, eps, c=10.0):

    k = max(min(m - 1, n), 1)
    return {"recon": c * eps * math.sqrt(k), "ortho": c * eps * math.sqrt(m), "tri": c * eps * math.sqrt(k)}


def check_qr(A, Q, R, eps, c=10.0):
    A_h = ttnn.to_torch(A).double()
    Q_h = ttnn.to_torch(Q).double()
    R_h = ttnn.to_torch(R).double()
    m, n = A_h.shape

    got = qr_metrics(A_h, Q_h, R_h)

    Q_ref, R_ref = torch.linalg.qr(A_h, mode="complete")
    Q_ref, R_ref = apply_sign_normalization(Q_ref, R_ref)
    ref = qr_metrics(A_h, Q_ref, R_ref)
    r_fwd = (torch.linalg.matrix_norm(R_h - R_ref, ord="fro") / torch.linalg.matrix_norm(R_ref, ord="fro")).item()

    tol = qr_tolerances(m, n, eps, c)
    ok = True
    print(f"shape {m}x{n}, eps = {eps:.3e}, c = {c}")
    print(f"{'metric':8s} {'device':>11s} {'fp64 ref':>11s} {'tol':>11s} {'normalized':>11s}")
    for key in ("recon", "ortho", "tri"):
        norm = got[key] / (eps * math.sqrt(m if key == "ortho" else max(min(m - 1, n), 1)))
        passed = got[key] <= tol[key]
        ok &= passed
        print(f"{key:8s} {got[key]:11.3e} {ref[key]:11.3e} {tol[key]:11.3e} {norm:11.2f} {'' if passed else '  FAIL'}")
    print(f"R forward error vs fp64 QR (depends on cond(A), no tolerance): {r_fwd:.3e}")
    return ok

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    shape = tuple(map(int, sys.argv[1].split(","))) if len(sys.argv) > 1 else (32, 32)
    dtype_arg = sys.argv[2] if len(sys.argv) > 2 else "float32"
    num_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    # example: func.py 1024,512 float32 5   (m,n need not be equal)

    torch.manual_seed(0)
    if dtype_arg == "float32":
        torch_A = torch.rand(shape, dtype=torch.float32)
    elif dtype_arg == "int32":
        torch_A = torch.randint(0, 100, shape).float()
    else:
        raise ValueError("Unsupported dtype (float32 or int32).")

    A = to_dev(torch_A, device)

    times = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        with cProfile.Profile() as pr:
            Q, R = ttnn_qr_householder_blocked(A, device, block_size=32)
        times.append(time.perf_counter() - start_time)

    if max(shape) <= 16:
        print("Q:\n", ttnn.to_torch(Q))
        print("R:\n", ttnn.to_torch(R))
        print("R (torch):\n", torch.linalg.qr(torch_A)[1])

    eps = measure_device_eps(device)
    print(f"Measured device matmul eps: {eps:.3e}  (true fp32 eps: {torch.finfo(torch.float32).eps:.3e})")
    passed = check_qr(A, Q, R, eps)
    print("PASS" if passed else "FAIL")

    A_h = ttnn.to_torch(A).double()
    Q_h = ttnn.to_torch(Q).double()
    R_h = ttnn.to_torch(R).double()
    m, n = A_h.shape
    Q_torch, R_torch = torch.linalg.qr(A_h, mode="complete")
    I_m = torch.eye(m, dtype=torch.float64)
 
    recon_error = (Q_h @ R_h - A_h).abs().max().item()
    orthogonality_error = (Q_h.T @ Q_h - I_m).abs().max().item()
    print(f"Reconstruction error: {recon_error}")
    print(f"Orthogonality error: {orthogonality_error}")
 
    recon_error_torch = (Q_torch @ R_torch - A_h).abs().max().item()
    orthogonality_error_torch = (Q_torch.T @ Q_torch - I_m).abs().max().item()
    print(f"PyTorch Reconstruction error: {recon_error_torch}")
    print(f"PyTorch Orthogonality error: {orthogonality_error_torch}")

    scale = A_h.abs().max().item()
    tol_recon = 5 * eps * scale * math.sqrt(m)
    tol_ortho = 5 * eps
    print(f"Estimated tolerance for reconstruction error: {tol_recon}")
    print(f"Estimated tolerance for orthogonality error: {tol_ortho}")

    ttnn.close_device(device)

    for i, t in enumerate(times):
        print(f"Time for run {i + 1}: {t:.6f} seconds")
    print(f"Average time: {sum(times) / len(times):.6f} seconds")

    print("Last iteration profiling stats:")
    pstats.Stats(pr).sort_stats(pstats.SortKey.TIME).print_stats(10)