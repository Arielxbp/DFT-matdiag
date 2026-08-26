import numpy as np
import ttl
import ttnn

TILE = 32

@ttl.operation(grid="auto")
def eltwise_add(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    rows = a.shape[0] // TILE
    cols = a.shape[1] // TILE
    a_dfb = ttl.make_dataflow_buffer_like(a,   shape=(1,1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b,   shape=(1,1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.wait() as ab, b_dfb.wait() as bb, o_dfb.reserve() as ob:
                    ob.store(ab + bb)

    @ttl.datamovement()
    def read():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                    ttl.copy(a[r:r+1, c:c+1], ab).wait()
                    ttl.copy(b[r:r+1, c:c+1], bb).wait()

    @ttl.datamovement()
    def write():
        for r in range(rows):
            for c in range(cols):
                with o_dfb.wait() as ob:
                    ttl.copy(ob, out[r:r+1, c:c+1]).wait()

device = ttnn.open_device(device_id=0)
dim = 64
a_np = np.random.rand(dim, dim).astype(np.float32)
b_np = np.random.rand(dim, dim).astype(np.float32)
a   = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b   = ttnn.from_torch(b_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.from_torch(np.zeros((dim,dim), dtype=np.float32),
                      dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
eltwise_add(a, b, out)
result = ttnn.to_torch(out)
print(f"eltwise_add: max_err={np.max(np.abs(result.float().numpy() - (a_np+b_np))):.6f}")
ttnn.close_device(device)