import numpy as np

def qr_iteration_numpy(matrix, iterations=20):
    # Convert input to a numpy array for efficient linear algebra
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    
    for _ in range(iterations):
        Q = np.zeros((n, n))
        R = np.zeros((n, n))
        
        # Modified Gram-Schmidt
        # We process the matrix column by column
        for j in range(n):
            v_j = A[:, j].copy()  # Extract j-th column
            
            for i in range(j):
                # Using the @ operator for the dot product
                # This replaces: sum(q[i][k] * v_j[k] for k in range(n))
                R[i, j] = Q[:, i] @ v_j #
                
                # Subtract the projection
                v_j -= R[i, j] * Q[:, i]
            
            R[j, j] = np.linalg.norm(v_j)
            Q[:, j] = v_j / R[j, j]
            
        # Recompose: A_next = R @ Q
        A = R @ Q
        
    return A

# Example Usage
A_start = np.array([
    [4, 1, 2],
    [1, 5, 0],
    [2, 0, 3]
])

A_final = qr_iteration_numpy(A_start)

print("Final Matrix:\n", np.round(A_final, 4))
print("Eigenvalues (Diagonal):", np.round(np.diag(A_final), 4))