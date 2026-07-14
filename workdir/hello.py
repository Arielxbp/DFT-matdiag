# Minimal ttnn tensor round-trip — no kernels, no hardware needed.
import numpy as np
import ttnn

device = ttnn.open_device(device_id=0)

a_np = np.arange(16, dtype=np.float32).reshape(4, 4)
t = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

print("Tensor shape:", t.shape)
print("Tensor dtype:", t.dtype)

recovered = ttnn.to_torch(t)
#print("Roundtrip match:", np.allclose(a_np, recovered, atol=1e-5))
print(recovered)

ttnn.close_device(device)
