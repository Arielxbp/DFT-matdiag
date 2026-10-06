import math
import sys
import time
import cProfile
import pstats

import ttnn
import torch

TILE = 32

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


def _hi(x):  # bf16 is a subset of what the matmul keeps, so it survives exactly
    return ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)


def matmul_x3(a, b):
    a_hi = _hi(a); a_lo = ttnn.subtract(a, a_hi)
    b_hi = _hi(b); b_lo = ttnn.subtract(b, b_hi)
    small = ttnn.add(matmul(a_hi, b_lo), matmul(a_lo, b_hi))
    return ttnn.add(matmul(a_hi, b_hi), small)


def T_(x):
    return ttnn.transpose(x, 0, 1)


# ---------------------------------------------------------------------------
# Device-resident constants (uploaded once, never rebuilt per panel / column)
# ---------------------------------------------------------------------------
class DeviceConsts:
    """rows: (m,1) column holding 0..m-1 (used to build row masks on device).
    units(size): one-hot column (size,1) and row (1,size) selectors, used to
    extract / place a single column with matmuls instead of slicing."""

    def __init__(self, device, m):
        self.device = device
        self.rows = to_dev(torch.arange(m, dtype=torch.float32).unsqueeze(1), device)
        self._units = {}

    def units(self, size):
        if size not in self._units:
            eye = torch.eye(size)
            cols = [to_dev(eye[:, j:j + 1].contiguous(), self.device) for j in range(size)]
            rws = [to_dev(eye[j:j + 1, :].contiguous(), self.device) for j in range(size)]
            self._units[size] = (cols, rws)
        return self._units[size]


def sign_pm1(t):
    """+1 where t >= 0 else -1 (same convention as torch.where(x < 0, -1, 1))."""
    return ttnn.subtract(ttnn.multiply(ttnn.ge(t, 0.0), 2.0), 1.0)


# ---------------------------------------------------------------------------
# Panel factorization entirely on device
# ---------------------------------------------------------------------------
def factor_panel_device(P, p, b, m, w, C, s_col):
    """Householder-factor the first b columns of the (m x w) column block P,
    whose first column sits at global column/row index p.

    Every reflector is applied to the whole block (all w columns), so columns
    b..w-1 (only present in the last block when k < n) are updated correctly.

    Returns the updated block, V (m x b), the compact-WY factor T (b x b), and
    s_col (m x 1) with the sign of each R[i,i] recorded (used for the final
    sign normalization, so no diagonal extraction is needed for these rows).
    """
    dev = C.device
    sel_col_w, sel_row_w = C.units(w)
    sel_col_b, sel_row_b = C.units(b)

    V = ttnn.zeros((m, b), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    T = ttnn.zeros((b, b), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    for j in range(b):
        i = p + j

        # row masks built on device: rows >= i, rows > i, and one-hot of row i
        m_ge = ttnn.ge(C.rows, float(i))
        m_gt = ttnn.ge(C.rows, float(i + 1))
        e_i = ttnn.subtract(m_ge, m_gt)

        # x = P[:, j] with rows < i zeroed
        x = ttnn.multiply(matmul_x3(P, sel_col_w[j]), m_ge)

        # alpha = sign(x_i) * ||x||   (all (1,1) tensors)
        nrm = ttnn.sqrt(matmul_x3(T_(x), x))
        x_i = matmul_x3(T_(e_i), x)
        alpha = ttnn.multiply(nrm, sign_pm1(x_i))

        # sign of R[i,i] = -alpha  ->  record into s_col at row i
        s_i = sign_pm1(ttnn.neg(alpha))
        s_col = ttnn.add(s_col, matmul_x3(e_i, ttnn.subtract(s_i, 1.0)))

        # v = (x + alpha*e_i) / ||.||    (1e-30 guards the all-zero column)
        x = ttnn.add(x, matmul_x3(e_i, alpha))
        inv = ttnn.rsqrt(ttnn.add(matmul_x3(T_(x), x), 1e-30))
        v = matmul_x3(x, inv)          # (m,1)
        vt = matmul_x3(inv, T_(x))     # (1,m)

        # compact-WY T column:  T[:j, j] = T[:j,:j] @ (-2 V[:, :j]^T v),  T[j,j] = 2
        z = T_(matmul_x3(vt, V))                                   # (b,1), zero for cols >= j
        tcol = ttnn.add(ttnn.multiply(matmul_x3(T, z), -2.0),
                        ttnn.multiply(sel_col_b[j], 2.0))
        T = ttnn.add(T, matmul_x3(tcol, sel_row_b[j]))
        V = ttnn.add(V, matmul_x3(v, sel_row_b[j]))

        # reflect the whole block:  P -= 2 v (v^T P)
        P = ttnn.subtract(P, ttnn.multiply(matmul_x3(v, matmul_x3(vt, P)), 2.0))

        # force exact zeros below the diagonal in column j
        below = ttnn.multiply(matmul_x3(P, sel_col_w[j]), m_gt)
        P = ttnn.subtract(P, matmul_x3(below, sel_row_w[j]))

    return P, V, T, s_col


# ---------------------------------------------------------------------------
# Blocked Householder QR, device resident
# ---------------------------------------------------------------------------
def ttnn_qr_householder_blocked(A, device, block_size=TILE):
    if block_size != TILE:
        raise ValueError(f"block_size must equal the tile width ({TILE}): R and Q are kept as tile-wide column blocks.")

    m, n = A.shape
    C = DeviceConsts(device, m)

    # R and Q are kept as lists of tile-wide column blocks on the device.
    # This avoids full-matrix masks, column slicing of R, and any host round trip.
    R_blocks = [ttnn.slice(A, [0, c0], [m, min(c0 + TILE, n)]) for c0 in range(0, n, TILE)]
    eye = torch.eye(m)
    Q_blocks = [to_dev(eye[:, c0:min(c0 + TILE, m)].contiguous(), device) for c0 in range(0, m, TILE)]

    s_col = ttnn.ones((m, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    k = min(m - 1, n)
    p, t = 0, 0
    while p < k:
        b = min(TILE, k - p)
        w = min(TILE, n - p)

        # --- panel factorization (device) ---
        P, V, T, s_col = factor_panel_device(R_blocks[t], p, b, m, w, C, s_col)
        R_blocks[t] = P

        Vt = T_(V)
        Tt = T_(T)

        # --- trailing update of R (device): R_u -= V (T^T (V^T R_u)) ---
        B = matmul_x3(Tt, Vt)  # (b, m)
        for u in range(t + 1, len(R_blocks)):
            R_blocks[u] = ttnn.subtract(R_blocks[u], matmul_x3(V, matmul_x3(B, R_blocks[u])))

        # --- update of Q (device): Q -= ((Q V) T) V^T ---
        # V is zero above row p, so only column blocks c >= t of Q are involved.
        Vc = []
        for c in range(t, len(Q_blocks)):
            r0 = c * TILE
            Vc.append(ttnn.slice(V, [r0, 0], [min(r0 + TILE, m), b]))
        QV = matmul_x3(Q_blocks[t], Vc[0])
        for idx in range(1, len(Vc)):
            QV = ttnn.add(QV, matmul_x3(Q_blocks[t + idx], Vc[idx]))
        Y = matmul_x3(QV, T)  # (m, b)
        for idx in range(len(Vc)):
            Q_blocks[t + idx] = ttnn.subtract(Q_blocks[t + idx], matmul_x3(Y, T_(Vc[idx])))

        p += TILE
        t += 1

    # --- sign normalization (device) ---
    # Rows 0..k-1 were recorded during factorization. When m <= n the last
    # diagonal entry R[m-1, m-1] has no reflector, so read it with selectors.
    if m <= n:
        cb = (m - 1) // TILE
        w_last = min(TILE, n - cb * TILE)
        sel_col_w, _ = C.units(w_last)
        e_last = ttnn.ge(C.rows, float(m - 1))
        d = matmul_x3(T_(e_last), matmul_x3(R_blocks[cb], sel_col_w[(m - 1) % TILE]))
        s_col = ttnn.add(s_col, matmul_x3(e_last, ttnn.subtract(sign_pm1(d), 1.0)))

    R_blocks = [ttnn.multiply(Rb, s_col) for Rb in R_blocks]          # scale rows of R
    for c in range(len(Q_blocks)):                                     # scale columns of Q
        r0 = c * TILE
        s_row = T_(ttnn.slice(s_col, [r0, 0], [min(r0 + TILE, m), 1]))
        Q_blocks[c] = ttnn.multiply(Q_blocks[c], s_row)

    Q = Q_blocks[0] if len(Q_blocks) == 1 else ttnn.concat(Q_blocks, dim=1)
    R = R_blocks[0] if len(R_blocks) == 1 else ttnn.concat(R_blocks, dim=1)
    return Q, R


# ---------------------------------------------------------------------------
# Verification (host, not part of the algorithm)
# ---------------------------------------------------------------------------
def apply_sign_normalization(Q, R):
    Q, R = Q.clone(), R.clone()
    s = torch.where(torch.diagonal(R) < 0, -1.0, 1.0).to(R.dtype)
    r = s.numel()
    Q[:, :r] *= s
    R[:r, :] *= s[:, None]
    return Q, R


def measure_device_eps(device):
    X = 1.0 + torch.rand((32, 32), generator=torch.Generator().manual_seed(1))
    Y = ttnn.to_torch(matmul_x3(to_dev(X, device), to_dev(torch.eye(32), device)))
    X, Y = X.double(), Y.double()
    rel = ((Y - X).abs() / X).max().item()
    return max(rel, torch.finfo(torch.float32).eps)


def qr_metrics(A, Q, R):
    A, Q, R = A.double(), Q.double(), R.double()
    m, n = A.shape
    fro = lambda X: torch.linalg.matrix_norm(X, ord="fro")
    I = torch.eye(m, dtype=torch.float64)
    return {
        "recon": (fro(Q @ R - A) / fro(A)).item(),
        "ortho": torch.linalg.matrix_norm(Q.T @ Q - I, ord=2).item(),
        "tri": (fro(R.tril(-1)) / fro(R)).item(),
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
    # example: V7.py 1024,512 float32 5   (m,n need not be equal)

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
        ttnn.synchronize_device(device)
        start_time = time.perf_counter()
        with cProfile.Profile() as pr:
            Q, R = ttnn_qr_householder_blocked(A, device)
            ttnn.synchronize_device(device)  # include device execution in the timing
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

    A_h = ttnn.to_torch(A).double()
    Q_h = ttnn.to_torch(Q).double()
    R_h = ttnn.to_torch(R).double()
    m, n = A_h.shape
    Q_torch, R_torch = torch.linalg.qr(A_h, mode="complete")
    I_m = torch.eye(m, dtype=torch.float64)

    print(f"Reconstruction error: {(Q_h @ R_h - A_h).abs().max().item()}")
    print(f"Orthogonality error: {(Q_h.T @ Q_h - I_m).abs().max().item()}")
    print(f"PyTorch Reconstruction error: {(Q_torch @ R_torch - A_h).abs().max().item()}")
    print(f"PyTorch Orthogonality error: {(Q_torch.T @ Q_torch - I_m).abs().max().item()}")

    scale = A_h.abs().max().item()
    print(f"Estimated tolerance for reconstruction error: {5 * eps * scale * math.sqrt(m)}")
    print(f"Estimated tolerance for orthogonality error: {5 * eps}")

    ttnn.close_device(device)

    for i, t in enumerate(times):
        print(f"Time for run {i + 1}: {t:.6f} seconds")
    print(f"Average time: {sum(times) / len(times):.6f} seconds")

    print("Last iteration profiling stats:")
    pstats.Stats(pr).sort_stats(pstats.SortKey.TIME).print_stats(10)