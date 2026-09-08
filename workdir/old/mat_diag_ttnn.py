import numpy as np
import torch
import ttnn

def qr_diagonalization(A_tt, device, num_iterations=50):
    A_k = A_tt

    for i in range(num_iterations):

        A_torch = ttnn.to_torch(A_k).to(torch.float32)

        Q_torch, R_torch = torch.linalg.qr(A_torch)

        Q_tt = ttnn.from_torch(
            Q_torch, 
            dtype=ttnn.float32, 
            layout=ttnn.TILE_LAYOUT, 
            device=device
        )
        R_tt = ttnn.from_torch(
            R_torch, 
            dtype=ttnn.float32, 
            layout=ttnn.TILE_LAYOUT, 
            device=device
        )

        A_k = ttnn.matmul(R_tt, Q_tt)

        ttnn.deallocate(Q_tt)
        ttnn.deallocate(R_tt)

    return A_k


if __name__ == "__main__":

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    try:

        A_symmetric = np.array(
            [
                [4, 1, 2],
                [1, 5, 0],
                [2, 0, 3],
            ],
            dtype=np.float32,
        )

        A_tt_initial = ttnn.from_torch(
            A_symmetric, 
            dtype=ttnn.float32, 
            layout=ttnn.TILE_LAYOUT, 
            device=device
        )

        A_diagonalized_tt = qr_diagonalization(A_tt_initial, device, num_iterations=50)
        
        A_diagonalized_torch = ttnn.to_torch(A_diagonalized_tt)

        eigenvalues = torch.diag(A_diagonalized_torch)
        print("Eigenvalues:\n", eigenvalues)
        
    finally:
        ttnn.close_device(device)