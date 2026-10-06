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

def _hi(x):
    return ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)

def matmul_x3(a, b):
    a_hi = _hi(a); a_lo = ttnn.subtract(a, a_hi)
    b_hi = _hi(b); b_lo = ttnn.subtract(b, b_hi)
    small = ttnn.add(matmul(a_hi, b_lo), matmul(a_lo, b_hi))
    return ttnn.add(matmul(a_hi, b_hi), small)

def sum_x2(t, dim, keepdim=True):
    hi = _hi(t)
    lo = ttnn.subtract(t, hi)
    return ttnn.add(ttnn.sum(hi, dim=dim, keepdim=keepdim),
                    ttnn.sum(lo, dim=dim, keepdim=keepdim))


# convert torch tensors to ttnn tensors
def to_dev(t, device):
    return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

# ttnn.matmul wrapper with the custom fp32 config
def matmul(a, b):
    return ttnn.matmul(a, b, compute_kernel_config=COMPUTE_CFG)


def _sum_col(t):

    return sum_x2(t, dim=0, keepdim=True)


def _one_minus(t):
    return ttnn.add(ttnn.multiply(t, -1.0), 1.0)


def _sign_flip(neg_flag):

    return _one_minus(ttnn.multiply(neg_flag, 2.0))


def factor_panel_dev(R, p, b, m, device, rows, cols):

    b_idx = to_dev(torch.arange(b, dtype=torch.float32).reshape(1, b), device)
    V = to_dev(torch.zeros((m, b)), device)
    panel_hi = ttnn.lt(cols, float(p + b))                      # (1,n): column < p+b

    for j in range(b):
        i = p + j
        col = sum_x2(ttnn.multiply(R, ttnn.eq(cols, float(i))), dim=1, keepdim=True)  # (m,1)
        x = ttnn.multiply(col, ttnn.ge(rows, float(i)))         # zero rows above i
        e_i = ttnn.eq(rows, float(i))                           # (m,1) one-hot

        norm = ttnn.sqrt(_sum_col(ttnn.multiply(x, x)))         # (1,1)
        x_i = _sum_col(ttnn.multiply(x, e_i))                   # (1,1)
        sgn = _sign_flip(ttnn.ltz(x_i))                         # sign(x_i), +1 for 0

        vu = ttnn.add(x, ttnn.multiply(e_i, ttnn.multiply(sgn, norm)))
        vsq = _sum_col(ttnn.multiply(vu, vu))
        vsq = ttnn.add(vsq, ttnn.eqz(vsq))                      # avoid 0/0: zero column -> v = 0
        v = ttnn.divide(vu, ttnn.sqrt(vsq))                     # (m,1)

        V = ttnn.add(V, ttnn.multiply(v, ttnn.eq(b_idx, float(j))))

        w = sum_x2(ttnn.multiply(v, R), dim=0, keepdim=True)  # (1,n) = v^T R
        w = ttnn.multiply(w, ttnn.multiply(ttnn.ge(cols, float(i)), panel_hi))
        R = ttnn.subtract(R, ttnn.multiply(ttnn.multiply(v, w), 2.0))

        below = ttnn.multiply(ttnn.gt(rows, float(i)), ttnn.eq(cols, float(i)))  # (m,n)
        R = ttnn.subtract(R, ttnn.multiply(R, below))           # exact zeros below the diagonal

    return V, R


def build_block_T_dev(V, b, device):

    Vt = ttnn.transpose(V, 0, 1)
    S = matmul_x3(Vt, V)                                           # (b,b)
    upper = to_dev(torch.triu(torch.ones(b, b), diagonal=1), device)
    eye = to_dev(torch.eye(b), device)

    N = ttnn.multiply(ttnn.multiply(S, upper), 2.0)
    P = ttnn.subtract(eye, N)
    Npow = N
    s = 1
    while (1 << s) < b:
        Npow = matmul_x3(Npow, Npow)                               # N^(2^s)
        P = matmul_x3(P, ttnn.add(eye, Npow))
        s += 1
    return ttnn.multiply(P, 2.0)


def normalize_diagonal_signs(Q, R, rows, cols):
    """Make diag(R) >= 0 on the device. Q (m,m), R (m,n), any shape; QR is unchanged."""
    diag = sum_x2(ttnn.multiply(R, ttnn.eq(rows, cols)), dim=1, keepdim=True)   # (m,1), 0 beyond n
    s_col = _sign_flip(ttnn.ltz(diag))                          # (m,1)
    R = ttnn.multiply(R, s_col)                                 # scale rows of R
    Q = ttnn.multiply(Q, ttnn.transpose(s_col, 0, 1))           # scale columns of Q
    return Q, R


def ttnn_qr_householder_blocked(A, device, block_size=32):
    """Full QR of an m x n matrix (any shape): Q is m x m, R is m x n upper trapezoidal."""
    m, n = A.shape

    R = ttnn.clone(A)
    Q = to_dev(torch.eye(m), device)
    rows = to_dev(torch.arange(m, dtype=torch.float32).reshape(m, 1), device)
    cols = to_dev(torch.arange(n, dtype=torch.float32).reshape(1, n), device)

    k = min(m - 1, n)  # number of reflectors
    p = 0
    while p < k:
        b = min(block_size, k - p)

        V, R = factor_panel_dev(R, p, b, m, device, rows, cols)
        T = build_block_T_dev(V, b, device)

        Vt = ttnn.transpose(V, 0, 1)
        Tt = ttnn.transpose(T, 0, 1)

        if p + b < n:  # trailing columns exist
            update_R = matmul_x3(V, matmul_x3(Tt, matmul_x3(Vt, R)))
            R = ttnn.subtract(R, ttnn.multiply(update_R, ttnn.ge(cols, float(p + b))))

        update_Q = matmul_x3(matmul_x3(matmul_x3(Q, V), T), Vt)
        Q = ttnn.subtract(Q, update_Q)

        p += b

    return normalize_diagonal_signs(Q, R, rows, cols)


def apply_sign_normalization(Q, R):
    Q, R = Q.clone(), R.clone()
    s = torch.where(torch.diagonal(R) < 0, -1.0, 1.0).to(R.dtype)
    r = s.numel()
    Q[:, :r] *= s
    R[:r, :] *= s[:, None]
    return Q, R


def measure_device_eps(device):
    g = torch.Generator().manual_seed(1)
    A = torch.rand(32, 32, generator=g); B = torch.rand(32, 32, generator=g)
    Y = ttnn.to_torch(matmul(to_dev(A, device), to_dev(B, device))).double()
    ref = A.double() @ B.double()
    return max(((Y - ref).abs().max() / ref.abs().max()).item(), torch.finfo(torch.float32).eps)


def qr_metrics(A, Q, R):
    """All in float64 on the host. A must be the matrix the device actually received."""
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
    """Heuristic bounds: random-walk accumulation over the k reflectors, not worst case.

    recon / tri ~ c * eps * sqrt(k),  ortho ~ c * eps * sqrt(m).  The worst case is
    ~ k * eps, so c needs calibrating on your hardware (see 'normalized' in the report).
    """
    k = max(min(m - 1, n), 1)
    return {"recon": c * eps * math.sqrt(k), "ortho": c * eps * math.sqrt(m), "tri": c * eps * math.sqrt(k)}


def check_qr(A, Q, R, eps, c=10.0):
    A_h = ttnn.to_torch(A).double()
    Q_h = ttnn.to_torch(Q).double()
    R_h = ttnn.to_torch(R).double()
    m, n = A_h.shape

    got = qr_metrics(A_h, Q_h, R_h)

    # float64 reference, full mode so shapes match for non-square input, same sign convention
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


# --------------------------------------------------------------------------
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
        print("V7")
        print("A:\n", ttnn.to_torch(A))
        print("Q:\n", ttnn.to_torch(Q))
        print("R:\n", ttnn.to_torch(R))
        print("R (torch):\n", torch.linalg.qr(torch_A)[1])

    eps = measure_device_eps(device)
    print(f"Measured device matmul eps: {eps:.3e}  (true fp32 eps: {torch.finfo(torch.float32).eps:.3e})")
    passed = check_qr(A, Q, R, eps)
    print("PASS" if passed else "FAIL")

    # --- Legacy absolute, entry-wise report (float64, on the device's copy of A) ---
    A_h = ttnn.to_torch(A).double()
    Q_h = ttnn.to_torch(Q).double()
    R_h = ttnn.to_torch(R).double()
    m, n = A_h.shape
    Q_torch, R_torch = torch.linalg.qr(A_h, mode="complete")  # "complete" so non-square shapes match
    I_m = torch.eye(m, dtype=torch.float64)

    recon_error = (Q_h @ R_h - A_h).abs().max().item()
    orthogonality_error = (Q_h.T @ Q_h - I_m).abs().max().item()
    print(f"Reconstruction error: {recon_error}")
    print(f"Orthogonality error: {orthogonality_error}")

    recon_error_torch = (Q_torch @ R_torch - A_h).abs().max().item()
    orthogonality_error_torch = (Q_torch.T @ Q_torch - I_m).abs().max().item()
    print(f"PyTorch Reconstruction error: {recon_error_torch}")
    print(f"PyTorch Orthogonality error: {orthogonality_error_torch}")

    # Same formulas as before, but with the measured device eps instead of the bf16 constant.
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