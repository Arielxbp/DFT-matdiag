
import torch
import ttnn


def to_dev(t, device):
    return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

device = ttnn.open_device(device_id=0)

COMPUTE_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

def matmul(a, b):
    return ttnn.matmul(a, b, compute_kernel_config=COMPUTE_CFG)

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

a = torch.rand(32, 32); b = torch.rand(32, 32)
at, bt = to_dev(a, device), to_dev(b, device)
ad, bd = a.double(), b.double()
def err(out, ref): return ((ttnn.to_torch(out).double() - ref).abs().max() / ref.abs().max()).item()

a_hi = ttnn.typecast(ttnn.typecast(at, ttnn.bfloat16), ttnn.float32)
b_hi = ttnn.typecast(ttnn.typecast(bt, ttnn.bfloat16), ttnn.float32)
ah, bh = ttnn.to_torch(a_hi).double(), ttnn.to_torch(b_hi).double()
ref = ah @ bh
for c in (1, 2, 4, 8, 32):
    acc = None
    for k0 in range(0, 32, c):
        mask = torch.zeros(1, 32); mask[:, k0:k0 + c] = 1
        part = matmul(ttnn.multiply(a_hi, to_dev(mask, device)), b_hi)
        acc = part if acc is None else ttnn.add(acc, part)
    print(f"chunk {c:2d}  ", err(acc, ref))

ttnn.close_device(device)
