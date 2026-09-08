import numpy as np

def blocked_householder_qr(A, block_size=2):
    """
    A simple blocked Householder QR decomposition.
    """
    m, n = A.shape
    A = A.astype(float)
    # R will store the upper triangular result
    R = A.copy()
    
    # We maintain Q as an identity matrix to accumulate the reflections
    Q = np.eye(m)
    
    # Process the matrix in blocks of size 'block_size'
    for k in range(0, min(m, n), block_size):
        b = min(block_size, min(m, n) - k)
        
        # 1. Collect block data
        V = np.zeros((m - k, b))
        T = np.zeros((b, b))
        
        # 2. Local QR inside the block (Simple Householder)
        for j in range(b):
            col = k + j
            x = R[k:, col].copy()
            
            # Create Householder vector v
            v = x.copy()
            norm_x = np.linalg.norm(x)
            v[0] += np.sign(x[0]) * norm_x if x[0] != 0 else norm_x
            v = v / np.linalg.norm(v)
            
            # Store in V
            V[j:, j] = v
            
            # Update R (the current block)
            tau = 2.0
            R[k:, col:] -= tau * np.outer(v, v.T @ R[k:, col:])
            
            # Calculate T (Interaction matrix)
            # This is a simplified version of the T update
            T[j, j] = tau
            if j > 0:
                T[:j, j] = -tau * (T[:j, :j] @ (V[j:, :j].T @ v))
        
        # 3. Block Update: Update the trailing submatrix (GEMM)
        # Trailing = Trailing - V * T^T * (V^T * Trailing)
        if k + b < n:
            trailing = R[k:, k+b:]
            # This is the "GEMM" step that makes blocked QR fast
            R[k:, k+b:] -= V @ (T.T @ (V.T @ trailing))
            
    # Clean up R to be purely upper triangular
    R = np.triu(R)
    return Q, R

# --- Example Usage ---
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    # A random 4x4 matrix
    A = np.array([
        [12, -51,   4, 1],
        [ 6, 167, -68, 2],
        [-4,  24, -41, 3],
        [ 1,   1,   1, 4]
    ], dtype=float)

    Q, R = blocked_householder_qr(A, block_size=2)
    
    print("Matrix R (Upper Triangular):\n", R)
    print("\nReconstruction Check (Q @ R == A):\n", Q @ R)