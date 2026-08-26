import ttnn
import torch


def to_tt_tile(torch_tensor):
   return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Returns a scalar
def ttnn_norm(x):
    # 2-norm of a vector x
    square = ttnn.square(x)

    sum = ttnn.sum(square)

    sqrt = ttnn.sqrt(sum)

    return sqrt

def ttnn_update_single_element(matrix, row, col, value, device):

    i = row
    j = col

    m, n = matrix.shape

    buffer = [0]*m # e_i = torch.zeros((m, 1))
    buffer[i] = 1 # e_i[i, 0] = 1
    e_i = ttnn.from_buffer(buffer=buffer, shape=[m, 1], dtype=matrix.dtype, layout=matrix.layout, device=device)
    # print(f"e_i is {e_i}")

    buffer = [0]*n # e_j = torch.zeros((n, 1))
    buffer[j] = 1 # e_j[j, 0] = 1
    e_j = ttnn.from_buffer(buffer=buffer, shape=[n, 1], dtype=matrix.dtype, layout=matrix.layout, device=device)
    # print(f"e_j is {e_j}")

    current_value = ttnn.matmul(ttnn.transpose(e_i, 0, 1), ttnn.matmul(matrix, e_j))
    current_value = ttnn.squeeze(current_value)
    # print(f"current_value is {current_value}")

    mask = ttnn.matmul(e_i, ttnn.transpose(e_j, 0 ,1))
    # print(f"mask is {mask}")

    # maybe subtract using ttnn.subtract
    updated_matrix = ttnn.add(matrix, ttnn.multiply(value - current_value, mask))
    # print(f"updated_matrix is {updated_matrix}")

    return updated_matrix

def ttnn_update_single_column(matrix, col, c, device):

    j = col

    m, n = matrix.shape

    buffer = [0]*n
    buffer[j] = 1
    e_j = ttnn.from_buffer(buffer=buffer, shape=[n, 1], dtype=matrix.dtype, layout=matrix.layout, device=device)
    # print(f"single column e_j is {e_j}")

    current_column = ttnn.matmul(matrix, e_j)
    # print(f"current_column is {current_column}")

    diff_vector = ttnn.subtract(c, current_column)
    # print(f"diff_vector is {diff_vector}")

    update_matrix = ttnn.matmul(diff_vector, ttnn.transpose(e_j, 0, 1))

    result = ttnn.add(matrix, update_matrix)

    return result

def ttnn_qr(A, device):
    m, n = A.shape
    Q = ttnn.zeros(shape=[m, n], dtype=A.dtype, layout=A.layout, device=device) # Same shape as A
    R = ttnn.zeros(shape=[n, n], dtype=A.dtype, layout=A.layout, device=device) # Square matrix

    # Loop over each column of A
    for i in range(n):

        a_i = A[:, i] # column vector of A 
        # a_i = ttnn.reshape(a_i, [m, 1]) # Ensure a_i is a column vector
        # print(f"a_i is {a_i}")
        v = ttnn.clone(a_i)

        for j in range(i):

            q_j = Q[:, j]
            # q_j = ttnn.reshape(q_j, [m, 1]) # Ensure q_j is a column vector

            r_ji = ttnn.matmul(q_j, a_i)


            R = ttnn_update_single_element(R, j, i, r_ji, device) # R[j, i] = r_ji

            v -= r_ji * q_j # Update v

        r_ii = ttnn_norm(v)
        # print(f"r_ii is {r_ii}")

        R = ttnn_update_single_element(R, i, i, r_ii, device) #R[i, i] = r_ii

        if r_ii != 0:

            # print(f"v is {v}")
            # print(f"r_ii is {r_ii}")
            column_vector = v / r_ii
            # print(f"column_vector is {column_vector}")
            column_vector = ttnn.reshape(column_vector, [m, 1]) # Ensure column_vector is a column vector
            # print(f"column_vector reshaped is {column_vector}")

            Q = ttnn_update_single_column(Q, i, column_vector, device) # Q[:, i] = v / r_ii

    return Q, R

if __name__ == "__main__":

    # useful apis reshape, squeeze, clone, add, multiply, subtract, matmul, transpose, sqrt, square, sum

    device = ttnn.open_device(device_id=0)

    shape = (4, 4)
    torch_A = torch.rand(shape, dtype=torch.float32) * 20 - 10 # [-10, 10] range
    print(f"Original matrix A:\n{torch_A}")
    A = to_tt_tile(torch_A)
    Q, R = ttnn_qr(A, device)

    print(f"Q matrix:\n{ttnn.to_torch(Q)}")
    print(f"R matrix:\n{ttnn.to_torch(R)}")

    Q = ttnn.to_torch(Q).float()
    R = ttnn.to_torch(R).float()

    m, n = torch_A.shape

    print("- Check -")

    # Check if Q * R = A
    reconstructed_A = Q @ R
    is_reconstructed = torch.allclose(reconstructed_A, torch_A, atol=1e-2)
    print(f"Q * R reconstruct A: {is_reconstructed}")

    # Check if Q is orthogonal (Q.T * Q = identity matrix)
    identity_matrix = torch.eye(n)
    is_orthogonal = torch.allclose(Q.T @ Q, identity_matrix, atol=1e-2)
    print(f"Q orthogonal: {is_orthogonal}")

    # Check if R is upper triangular
    lower_triangle_of_R = torch.tril(R, diagonal=-1)
    is_triangular = torch.allclose(lower_triangle_of_R, torch.zeros_like(lower_triangle_of_R), atol=1e-2)
    print(f"R upper triangular: {is_triangular}")

    ttnn.close_device(device)
