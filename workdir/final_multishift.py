import math
import sys
import torch
import numpy as np
import ttnn
import time

TILE_SIZE = 32
PANEL_WIDTH = 32 # Householder reflectors per WY block
WINDOW_SIZE = 128 # side of the diagonal window the bulges are chased in
SMALL_BLOCK_SIZE = WINDOW_SIZE - TILE_SIZE # active blocks this small are finished in one window

MAX_BULGES = 15 # bulges chased together in one sweep each with its own pair of shifts
BULGE_SPACING = 4 # rows between two bulges: the reflectors of one step then touch different rows and different columns
EXCEPTIONAL_SHIFT_PERIOD = 6 # sweeps without deflation before exceptional shifts are used (LAPACK's value)

AED_WINDOW_SIZE = 32 # size of the aggressive-early-deflation window
AED_NIBBLE = 14 # percent of the aed window that must deflate to look for deflations again before sweeping (LAPACK's value)
AED_AUTO_MIN_SIZE = 256 # threshold for automatic aed (measured: from 256 on it is as fast or up to 28% faster, and fewer sweeps mean less device drift)

F32 = torch.float32
F64 = torch.float64
MACHINE_EPS = torch.finfo(F32).eps # deflation tolerance

DEVICE_MATMUL_ROUNDOFF = 2.0 ** -10 # measured unit roundoff of the fp32 device matmul

COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def to_device(host_tensor, device):
    return ttnn.from_torch(host_tensor.to(F32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

def to_host(device_tensor):
    return ttnn.to_torch(device_tensor).to(F64)

def matmul(left, right):
    return ttnn.matmul(left, right, compute_kernel_config=COMPUTE_CONFIG)

def device_columns(matrix, col_start, col_end):
    return ttnn.slice(matrix, [0, col_start], [matrix.shape[0], col_end])

def device_matvec(matrix_device, vector, device):
    return to_host(matmul(matrix_device, to_device(vector[:, None], device)))[:, 0]

def round_down_to_tile(index):
    return (index // TILE_SIZE) * TILE_SIZE


def householder_reflector(vector):

    # take the first element
    alpha = float(vector[0])

    # check if every other element is zero
    tail = vector[1:]
    if tail.numel() > 0:
        tail_max = tail.abs().max().item()
    else:
        # if no elements then max is zero
        tail_max = 0.0

    if tail_max == 0.0:
        return None, 0.0, alpha

    # prevent overflow/underflow when squaring
    scale = max(tail_max, abs(alpha))
    scaled = vector / scale

    # compute the length of the vector
    norm = scale * math.sqrt(float(torch.dot(scaled, scaled)))

    # compute the reflector
    beta = -math.copysign(norm, alpha)
    v = vector / (alpha - beta)
    v[0] = 1.0
    tau = (beta - alpha) / beta

    return v, tau, beta


def householder_reflector_small(values):

    # take the first element
    alpha = values[0]

    # check if every other element is zero
    if not any(values[1:]):
        return None, 0.0, alpha

    # compute the length of the vector
    norm = math.hypot(*values)

    # compute the reflector
    beta = -math.copysign(norm, alpha)
    factor = 1.0 / (alpha - beta)
    tau = (beta - alpha) / beta
    v = [1.0] + [value * factor for value in values[1:]]
    return v, tau, beta


def householder_batched(targets):

    # the reflectors of householder_reflector_small for many 3-element target columns at once (one per row)
    alpha = targets[:, 0]

    # compute the lengths (hypot prevents overflow/underflow when squaring)
    norm = np.hypot(np.hypot(alpha, targets[:, 1]), targets[:, 2])

    # an all-zero column needs no reflector: a length of 1 avoids dividing by zero, then tau = 0 below
    zero = norm == 0.0
    norm[zero] = 1.0

    # compute the reflectors (a column that is already (alpha, 0, 0) gets beta = -alpha, a sign flip)
    beta = -np.copysign(norm, alpha)
    v = targets / (alpha - beta)[:, None]
    v[:, 0] = 1.0
    tau = (beta - alpha) / beta
    tau[zero] = 0.0
    beta[zero] = 0.0
    return v, tau, beta


def apply_reflector_two_sided(stacked, v, tau, first_index):

    # stacked: the window block on top of the transform (the product of every reflector applied to this window so far)
    # first_index: the index of the first row/column touched by P

    v = np.array(v)
    tau_v = tau * v
    last_index = first_index + v.size

    # apply to left side (B <- P B), updating rows of the block (a[:, None] * b is the outer product of a and b)
    rows = stacked[first_index:last_index, first_index:]
    rows -= tau_v[:, None] * (v @ rows)

    # apply to right side of the block and of the transform with one operation (B <- B P, T <- T P), updating columns
    # the block rows below the bulge are exact zeros in these columns, so including them changes nothing
    cols = stacked[:, first_index:last_index]
    cols -= (cols @ v)[:, None] * tau_v


def wy_T_from_gram(gram, taus):

    # compute matrix T needed for the compact WY form, so I - VTV^T = reflectors for that panel

    # get the number of reflectors 
    width = len(taus)

    # remove everything on and under the diagonal
    t_inverse = torch.triu(gram[:width, :width].to(F64), diagonal=1)

    # add on the diagonal the 1/tau elements or 1 if tau is zero
    diagonal = torch.tensor([1.0 / tau if tau != 0.0 else 1.0 for tau in taus], dtype=F64)
    t_inverse += torch.diag(diagonal)

    # invert the upper triangular matrix by row wise back substitution
    T_matrix = torch.zeros_like(t_inverse)
    for row in range(width - 1, -1, -1):
        new_row = -(t_inverse[row, row + 1:] @ T_matrix[row + 1:, :])
        new_row[row] += 1.0
        T_matrix[row, :] = new_row / t_inverse[row, row]

    # zero the rows and columns that have tau to zero
    for index, tau in enumerate(taus):
        if tau == 0.0:
            T_matrix[index, :] = 0.0
            T_matrix[:, index] = 0.0
    return T_matrix


def reduce_to_hessenberg(matrix_device, size, padded_size, device):

    # every device operation works on the full padded matrix, so the operand shapes are the same for every panel
    # and ttnn builds each program once; rows and columns outside the trailing block are multiplied by exact zeros

    # Q accumulates the reflectors of every panel on the device (A = Q H Q^T), for the eigenvectors
    q_device = to_device(torch.eye(padded_size, dtype=F64), device)

    # for every panel of columns (last 2 columns are already in Hessenberg form)
    for panel_start in range(0, size - 2, PANEL_WIDTH):

        # get the number of columns for this panel
        width = min(PANEL_WIDTH, size - 2 - panel_start)

        # get the panel of 'width' columns
        panel = to_host(device_columns(matrix_device, panel_start, panel_start + PANEL_WIDTH))

        # init the reflectors, the update Y=AVT matrices and the WY matrix T
        reflectors = torch.zeros(padded_size, PANEL_WIDTH, dtype=F64)
        y_block = torch.zeros(padded_size, PANEL_WIDTH, dtype=F64)
        T = torch.zeros(PANEL_WIDTH, PANEL_WIDTH, dtype=F64)
        taus = [0.0] * PANEL_WIDTH

        # for every column of the panel, build the reflector and the update vector
        for index in range(width):
            column = panel_start + index

            # get the column
            current = panel[:, index].clone()

            # update the column with earlier reflectors
            if index != 0:
                # from the right side A - YV^T
                current -= y_block[:, :index] @ reflectors[column, :index]

                # from the left side (I - VTV^T)^T = I - VT^TV^T
                current -= reflectors[:, :index] @ (T[:index, :index].T @ (reflectors[:, :index].T @ current))

            # compute the Householder reflector for the current column
            v, tau, _ = householder_reflector(current[column + 1:])
            if v is None:
                continue
            taus[index] = tau
            reflectors[column + 1:, index] = v

            # inner products of the new reflector with the earlier ones (a column of V^TV)
            inner = reflectors[:, :index].T @ reflectors[:, index]

            # add the new column to T
            T[:index, index] = -tau * (T[:index, :index] @ inner)
            T[index, index] = tau

            # compute Y's column (the reflector is zero above column + 1, so the full matrix gives the trailing product)
            a_times_v = device_matvec(matrix_device, reflectors[:, index], device)
            y_block[:, index] = tau * (a_times_v - y_block[:, :index] @ inner)

        # if panel was already in Hessenberg form skip useless computes
        if not any(taus):
            continue

        reflectors_device = to_device(reflectors, device)
        reflectors_t_device = to_device(reflectors.T, device)

        # build the T of the trailing update from V^TV computed on the device
        # (the T built column by column in the loop is still what updates the panel's own columns)
        T = wy_T_from_gram(to_host(matmul(reflectors_t_device, reflectors_device)), taus)

        # right update with reflectors (columns up to panel_start get "minus 0")
        matrix_device = ttnn.subtract(matrix_device, matmul(to_device(y_block, device), reflectors_t_device))

        # left update with reflectors (rows up to panel_start get "minus 0")
        temp = matmul(to_device(T.T, device), matmul(reflectors_t_device, matrix_device))
        matrix_device = ttnn.subtract(matrix_device, matmul(reflectors_device, temp))

        # accumulate the panel's reflectors Q <- Q (I - VTV^T) = Q - ((QV)T)V^T
        q_device = ttnn.subtract(q_device, matmul(matmul(matmul(q_device, reflectors_device), to_device(T, device)), reflectors_t_device))

    # explicitly zero out the entries below the subdiagonal
    return ttnn.triu(matrix_device, diagonal=-1), q_device


def reduce_to_tridiagonal(matrix_device, size, padded_size, device):

    # every device operation works on the full padded matrix, so the operand shapes are the same for every panel
    # and ttnn builds each program once; only the trailing block (rows and columns from panel_start) is up to date

    # init tridiagonal result
    diagonal = [0.0] * size
    off_diagonal = [0.0] * max(size - 1, 0)

    # Q accumulates the reflectors of every panel on the device (A = Q T Q^T), for the eigenvectors
    q_device = to_device(torch.eye(padded_size, dtype=F64), device)

    # for every panel, build its reflector and update vector
    for panel_start in range(0, size, PANEL_WIDTH):

        # get the current panel
        panel = to_host(device_columns(matrix_device, panel_start, panel_start + PANEL_WIDTH))

        # init the reflectors and the update W matrices
        reflectors = torch.zeros(padded_size, PANEL_WIDTH, dtype=F64)
        w_block = torch.zeros(padded_size, PANEL_WIDTH, dtype=F64)
        taus = [0.0] * PANEL_WIDTH

        # for every column of the panel, build the reflector and the update vector
        for index in range(min(PANEL_WIDTH, size - panel_start)):
            column = panel_start + index

            # get the column starting from the diagonal
            current = panel[column:, index].clone()

            # update the column with earlier reflectors
            if index != 0:
                current -= reflectors[column:, :index] @ w_block[column, :index] + w_block[column:, :index] @ reflectors[column, :index]

            # set the current diagonal element
            diagonal[column] = float(current[0])

            # skip last column since last element is on the diagonal
            if column == size - 1:
                break

            # compute the Householder reflector for the current column
            v, tau, beta = householder_reflector(current[1:])

            # set the current off diagonal element
            off_diagonal[column] = beta

            if v is None:
                continue
            taus[index] = tau
            reflectors[column + 1:, index] = v

            # Compute W's column (the reflector is zero above column + 1, so the full matrix gives the trailing product)
            a_times_v = device_matvec(matrix_device, reflectors[:, index], device)[column + 1:]
            v_prev = reflectors[column + 1:, :index]
            w_prev = w_block[column + 1:, :index]
            w = tau * (a_times_v - w_prev @ (v_prev.T @ v) - v_prev @ (w_prev.T @ v))
            w -= 0.5 * tau * torch.dot(w, v) * v
            w_block[column + 1:, index] = w

        # accumulate the panel's reflectors Q <- Q (I - VTV^T), with T built from V^TV computed on the device
        if any(taus):
            reflectors_device = to_device(reflectors, device)
            reflectors_t_device = to_device(reflectors.T, device)
            T = wy_T_from_gram(to_host(matmul(reflectors_t_device, reflectors_device)), taus)
            q_device = ttnn.subtract(q_device, matmul(matmul(matmul(q_device, reflectors_device), to_device(T, device)), reflectors_t_device))

        if panel_start + PANEL_WIDTH >= size:
            break

        if not any(taus):
            continue

        # update the next trailing block A - VW^T - WV^T (rows and columns before it get "minus 0")
        next_start = panel_start + PANEL_WIDTH
        v_and_w = torch.zeros(padded_size, 2 * PANEL_WIDTH, dtype=F64)
        w_and_v = torch.zeros(padded_size, 2 * PANEL_WIDTH, dtype=F64)
        v_and_w[next_start:] = torch.cat([reflectors[next_start:], w_block[next_start:]], dim=1)
        w_and_v[next_start:] = torch.cat([w_block[next_start:], reflectors[next_start:]], dim=1)
        matrix_device = ttnn.subtract(matrix_device, matmul(to_device(v_and_w, device), to_device(w_and_v.T, device)))

    return diagonal, off_diagonal, q_device



class HostHessenberg:

    def __init__(self, matrix, device):
        self.matrix = matrix
        self.device = device

        # Schur vectors Z: the product of every window transform (identity at the start, padded like the matrix)
        self.vectors = torch.eye(matrix.shape[0], dtype=F64)

        # numpy view of the same memory, for the small window operations (numpy has a much lower cost per call than torch)
        self.array = matrix.numpy()

    def block(self, start, end):
        return self.array[start:end, start:end].copy()

    def update(self, start, end, new_block, transform):

        # update the Hessenberg matrix with the new block and the accumulated transform on all the rows above and all the
        # columns to the right (the eigenvectors need the whole Schur form, also the parts coupling different blocks),
        # and the Schur vectors with Z <- Z U
        # the two products on the matrix run on the device on full columns / full rows (same shapes every window, so ttnn
        # builds each program once); only the part outside the window is copied back, the block itself comes exact from the host
        transform = torch.from_numpy(transform)
        transform_device = to_device(transform, self.device)

        # rows above
        if start > 0:
            columns = to_host(matmul(to_device(self.matrix[:, start:end], self.device), transform_device))
            self.matrix[:start, start:end] = columns[:start]

        # columns to the right
        if end < self.matrix.shape[0]:
            rows = to_host(matmul(to_device(transform.T, self.device), to_device(self.matrix[start:end, :], self.device)))
            self.matrix[start:end, end:] = rows[:, end:]

        # Schur vectors, on the host in fp64: every window multiplies the same columns of Z again, and the device's
        # error on each product adds up in the eigenvectors (measured: residual 0.12 at n=1024 on the device, 0.03 here)
        self.vectors[:, start:end] = self.vectors[:, start:end] @ transform

        self.array[start:end, start:end] = new_block

    def full(self):
        return self.matrix


def wilkinson_shift(previous_diagonal, last_off_diagonal, last_diagonal):

    # compute the eigenvalue closest to the last diagonal element
    if last_off_diagonal == 0.0:
        return last_diagonal
    delta = 0.5 * (previous_diagonal - last_diagonal)
    temp = math.hypot(delta, last_off_diagonal)
    return last_diagonal - last_off_diagonal ** 2 / (delta + math.copysign(temp, delta))


def tridiagonal_sweep(diagonal, off_diagonal, active_low, active_high, shift):

    # subtract the shift from the first diagonal element
    x = diagonal[active_low] - shift
    z = off_diagonal[active_low]

    # the rotations of the sweep, for the eigenvectors
    cosines, sines = [], []

    # chase the bulge from top to bottom one rotation per iteration
    for step in range(active_low, active_high):

        # compute the Givens rotation
        radius = math.hypot(x, z)
        cos, sin = (1.0, 0.0) if radius == 0.0 else (x / radius, z / radius)
        cosines.append(cos)
        sines.append(sin)

        # fold the bulge into the off diagonal
        if step > active_low:
            off_diagonal[step - 1] = radius

        # apply the rotation to the 2x2 block on the diagonal
        a, b, d = diagonal[step], off_diagonal[step], diagonal[step + 1]
        cos2 = cos * cos
        sin2 = sin * sin
        diagonal[step] = cos2 * a + 2.0 * cos * sin * b + sin2 * d
        diagonal[step + 1] = sin2 * a - 2.0 * cos * sin * b + cos2 * d
        off_diagonal[step] = cos * sin * (d - a) + (cos2 - sin2) * b

        # move the implicit bulge caused by rotation
        if step < active_high - 1:
            z = sin * off_diagonal[step + 1]
            off_diagonal[step + 1] = cos * off_diagonal[step + 1]
            x = off_diagonal[step]

    return cosines, sines


def rotation_block(cosines, sines):

    # product G_1 ... G_m of the m consecutive rotations of a sweep as one (m + 1) x (m + 1) matrix
    # (G_r acts on columns r - 1 and r: col_r-1 <- c col_r-1 + s col_r, col_r <- -s col_r-1 + c col_r);
    # in closed form the part above the diagonal is a cumulative product of the -sin values, so no loop over the rotations
    count = len(cosines)
    cos = torch.tensor(cosines, dtype=F64)
    sin = torch.tensor(sines, dtype=F64)
    one = torch.ones(1, dtype=F64)
    cos_shifted = torch.cat([one, cos])
    minus_sin = torch.cat([one, -sin])
    index = torch.arange(count + 1)
    factors = torch.where(index[None, :] > index[:, None], minus_sin[None, :].expand(count + 1, count + 1), torch.ones((), dtype=F64))
    partial = torch.triu(cos_shifted[:, None] * torch.cumprod(factors, dim=1))
    product = torch.zeros(count + 1, count + 1, dtype=F64)
    product[:, :count] = partial[:, :count] * cos[None, :]
    product[index[1:], index[:count]] += sin
    product[:, count] = partial[:, count]
    return product


def apply_rotations_to_vectors(vectors, cosines, sines, active_low):

    # Z <- Z G_lo ... G_hi-1, grouped into tile aligned windows with one product per window, on the host:
    # the symmetric iteration does one sweep per shift (about 2n sweeps), and on the device the same columns of Z
    # would collect thousands of lossy products
    total, done = len(cosines), 0
    while done < total:
        step = active_low + done
        start = round_down_to_tile(step)
        end = min(start + WINDOW_SIZE, vectors.shape[0])

        # rotations on planes step .. step + count stay inside the window
        count = min(total - done, end - 1 - step)
        transform = torch.eye(end - start, dtype=F64)
        local = step - start
        transform[local:local + count + 1, local:local + count + 1] = rotation_block(cosines[done:done + count], sines[done:done + count])
        vectors[:, start:end] = vectors[:, start:end] @ transform
        done += count


def zeroable(diagonal, off_diagonal, index):

    return abs(off_diagonal[index]) <= MACHINE_EPS * (abs(diagonal[index]) + abs(diagonal[index + 1]))


def tridiagonal_qr_iteration(diagonal, off_diagonal, size, vectors):

    # vectors: the Schur vectors Z (padded identity at the start), multiplied by the rotations of every sweep

    active_high = size - 1
    max_sweeps = 30 * max(size, 10)
    sweeps_on_block = 0
    last_block = None

    # start from bottom
    while active_high > 0:
        active_low = active_high

        # find the first zeroable element on the sub diagonal
        while active_low > 0 and not zeroable(diagonal, off_diagonal, active_low - 1):
            active_low -= 1

        # zero the subdiagonal element if zeroable
        if active_low > 0:
            off_diagonal[active_low - 1] = 0.0

        # eigenvalue is the element if 1x1 block
        if active_low == active_high:
            active_high -= 1
            continue

        # a block that fits in one window is solved completely, collecting its rotations in one transform,
        # so Z is multiplied only once for the whole block
        if active_high - active_low + 1 <= SMALL_BLOCK_SIZE:
            window_start = round_down_to_tile(active_low)
            window_end = min(window_start + WINDOW_SIZE, vectors.shape[0])
            transform = torch.eye(window_end - window_start, dtype=F64)
            bottom, sweeps_on_block = active_high, 0
            while bottom > active_low:
                low = bottom
                while low > active_low and not zeroable(diagonal, off_diagonal, low - 1):
                    low -= 1
                if low > active_low:
                    off_diagonal[low - 1] = 0.0
                if low == bottom:
                    bottom, sweeps_on_block = bottom - 1, 0
                    continue
                sweeps_on_block += 1
                if sweeps_on_block > max_sweeps:
                    raise RuntimeError(f"Symmetric QR: no convergence in block [{low}, {bottom}]")
                mu = wilkinson_shift(diagonal[bottom - 1], off_diagonal[bottom - 1], diagonal[bottom])
                cosines, sines = tridiagonal_sweep(diagonal, off_diagonal, low, bottom, mu)
                first, last = low - window_start, bottom - window_start + 1
                transform[:, first:last] = transform[:, first:last] @ rotation_block(cosines, sines)
            vectors[:, window_start:window_end] = vectors[:, window_start:window_end] @ transform
            active_high = active_low - 1
            continue

        # reset the sweep counter if new block
        if (active_low, active_high) != last_block:
            sweeps_on_block = 0
            last_block = (active_low, active_high)

        sweeps_on_block += 1
        if sweeps_on_block > max_sweeps:
            raise RuntimeError(f"Symmetric QR: no convergence in block [{active_low}, {active_high}]")

        # do one sweep with the computed shift
        mu = wilkinson_shift(diagonal[active_high - 1], off_diagonal[active_high - 1], diagonal[active_high])
        cosines, sines = tridiagonal_sweep(diagonal, off_diagonal, active_low, active_high, mu)
        apply_rotations_to_vectors(vectors, cosines, sines, active_low)


def francis_shifts(a, b, c, d, exceptional, element_above):

    # if normal shifts are not converging use LAPACK shifts
    if exceptional:
        scale = abs(c) + abs(element_above)
        a = 0.75 * scale + d
        b, c, d = -0.4375 * scale, scale, a

    trace = a + d
    determinant = a * d - b * c
    return trace, determinant


def compute_first_column(block, active_low, active_high, shift_sum, shift_product):

    # get the 2x2 block at the top
    h00, h01 = block[active_low, active_low].item(), block[active_low, active_low + 1].item()
    h10, h11 = block[active_low + 1, active_low].item(), block[active_low + 1, active_low + 1].item()

    scale = max(abs(h00), abs(h01), abs(h10), abs(h11))
    if scale == 0.0:
        scale = 1.0
    h00, h01, h10, h11 = h00 / scale, h01 / scale, h10 / scale, h11 / scale
    shift_sum, shift_product = shift_sum / scale, shift_product / (scale * scale)

    # compute the column
    column = [h00 * h00 + h01 * h10 - shift_sum * h00 + shift_product, h10 * (h00 + h11 - shift_sum)]
    if active_high - active_low >= 2:
        column.append(h10 * block[active_low + 2, active_low + 1].item() / scale)

    return column


def bulge_step(stacked, step, active_low, active_high, shift_sum, shift_product):

    # stacked: the window block on top of the transform, so the block entries are the top rows

    # work on 3 elements (2 at the end)
    length = min(3, active_high - step + 1)

    # create the bulge
    if step == active_low:
        target = compute_first_column(stacked, active_low, active_high, shift_sum, shift_product)
    else:
        # take the elements from the current column of the bulge
        target = stacked[step:step + length, step - 1].tolist()

    # build the reflector to zero the last two elements of the target column
    v, tau, beta = householder_reflector_small(target)

    # direct apply the known reflector
    if step > active_low:
        stacked[step, step - 1] = beta
        stacked[step + 1:step + length, step - 1] = 0.0

    # if target was already in the correct form, skip the reflector
    if v is not None:
        apply_reflector_two_sided(stacked, v, tau, step)


def chase_bulge(stacked, active_low, active_high, shift_sum, shift_product, first_step, last_step):

    # for each reflector
    for step in range(first_step, last_step + 1):
        bulge_step(stacked, step, active_low, active_high, shift_sum, shift_product)


def chase_bulges_together(stacked, top, count, active_low, active_high, entering_shifts):

    # one step of 'count' bulges BULGE_SPACING rows apart, the highest at row top: their reflectors touch different rows
    # and different columns, and the only entries two of them share get one from the left and the other from the right,
    # which commute, so every reflector is built from the current window and all are applied with a few array operations
    positions = top + BULGE_SPACING * np.arange(count)

    # take the current column of every bulge (a bulge entering at the top of the block is created from its shifts)
    target_rows = positions[:, None] + np.arange(3)
    target_columns = np.maximum(positions - 1, 0)[:, None]
    targets = stacked[target_rows, target_columns]
    entering = bool(top == active_low)
    if entering:
        targets[0] = compute_first_column(stacked, active_low, active_high, *entering_shifts)

    # build the reflectors to zero the last two elements of every target column
    v, tau, beta = householder_batched(targets)
    tau_v = tau[:, None] * v

    # apply to left side (B <- P B): the 3 rows of every bulge are the first 3 of each group of BULGE_SPACING rows
    # (all the columns to the end of the window: the eigenvectors need the whole Schur form)
    rows = stacked[top:top + BULGE_SPACING * count, max(top - 1, 0):].reshape(count, BULGE_SPACING, -1)[:, :3]
    rows -= tau_v[:, :, None] * np.matmul(v[:, None, :], rows)

    # direct apply the known reflectors: the target columns become (beta, 0, 0)
    exact = np.zeros_like(targets)
    exact[:, 0] = beta
    stacked[target_rows[entering:], target_columns[entering:]] = exact[entering:]

    # apply to right side of the block and of the transform (B <- B P, T <- T P): the 3 columns of every bulge, copied
    # into a buffer where each column is contiguous in memory (the same update on the strided columns is 2x slower)
    buffer = stacked[:, top:top + BULGE_SPACING * count].T.copy()
    cols = buffer.reshape(count, BULGE_SPACING, -1)[:, :3]
    cols -= tau_v[:, :, None] * np.matmul(v[:, None, :], cols)
    stacked[:, top:top + BULGE_SPACING * count] = buffer.T


def chase_bulge_chain(stacked, window_start, active_low, active_high, shifts, step, last_step):

    # macro step 'step' moves every bulge down one row: bulge j enters the block at step BULGE_SPACING * j,
    # so it is at row active_low + step - BULGE_SPACING * j (rows and columns here are local to the window)
    window_size = stacked.shape[1]
    low, high = active_low - window_start, active_high - window_start
    bulges = len(shifts)

    while step <= last_step:

        # the bulges inside the block go from the newest (top row) to the oldest (bottom row)
        newest = min(bulges - 1, step // BULGE_SPACING)
        oldest = max(0, -((high - 1 - low - step) // BULGE_SPACING))
        top, bottom = low + step - BULGE_SPACING * newest, low + step - BULGE_SPACING * oldest

        # stop when the oldest bulge would leave the window
        if min(bottom + 3, high) >= window_size:
            break

        # the oldest bulge moves alone when it reaches the bottom of the block (2-element reflector) or the window edge
        alone = bottom > high - 2 or bottom + BULGE_SPACING > window_size
        together = newest - oldest + 1 - alone
        if together:
            chase_bulges_together(stacked, top, together, low, high, shifts[newest])
        if alone:
            bulge_step(stacked, bottom, low, high, *shifts[oldest])

        step += 1

    return step


def subdiagonal_zeroable(subdiagonal_element, diagonal_above, diagonal_below):
    return abs(subdiagonal_element) <= MACHINE_EPS * (abs(diagonal_above) + abs(diagonal_below))


def lowest_split(block, active_low, active_high):

    # find the lowest split point in the block where the sub diagonal is zeroable
    if active_high <= active_low:
        return active_low
    subdiagonal = np.abs(np.diagonal(block, -1)[active_low:active_high])
    diagonal = np.abs(np.diagonal(block)[active_low:active_high + 1])
    hits = np.nonzero(subdiagonal <= MACHINE_EPS * (diagonal[:-1] + diagonal[1:]))[0]
    if hits.size == 0:
        return active_low
    return active_low + int(hits[-1]) + 1


def schur_on_host_block(stacked, active_low, active_high):

    # the window block is the top square of stacked, the transform is under it
    block = stacked[:stacked.shape[1]]

    converged = []
    bottom, sweeps_on_block = active_high, 0
    max_sweeps = 30 * max(10, active_high - active_low + 1)

    # if not every element has converged
    while bottom >= active_low:

        split = lowest_split(block, active_low, bottom)

        # zero the found sub diagonal element
        if split > active_low:
            block[split, split - 1] = 0.0

        # save the converged 1x1 or 2x2 blocks
        if split == bottom:
            converged.append((bottom, 1))
            bottom, sweeps_on_block = bottom - 1, 0
            continue
        if split == bottom - 1:
            converged.append((bottom - 1, 2))
            bottom, sweeps_on_block = bottom - 2, 0
            continue

        # else do one sweep to try congerge elements
        sweeps_on_block += 1
        if sweeps_on_block > max_sweeps:
            raise RuntimeError(f"Francis QR: no convergence in block [{split}, {bottom}]")

        # get the element above for special shift
        if bottom - 2 >= split:
            element_above = block[bottom - 1, bottom - 2].item()
        else:
            element_above = 0.0
        a, b, c, d = block[bottom - 1, bottom - 1].item(), block[bottom - 1, bottom].item(), block[bottom, bottom - 1].item(), block[bottom, bottom].item()
        shift_sum, shift_product = francis_shifts(a, b, c, d, sweeps_on_block in (10, 20), element_above)
        chase_bulge(stacked, split, bottom, shift_sum, shift_product, split, bottom - 1)

    return converged


def shift_pairs(block, schur_blocks):

    # turn converged Schur blocks into pairs of shifts given as (sum, product), like francis_shifts:
    # a 2x2 block gives its trace and determinant, two 1x1 blocks are paired together
    pairs, single = [], None
    for start, length in schur_blocks:
        if length == 2:
            a, b ,c, d = block[start, start].item(), block[start, start + 1].item(), block[start + 1, start].item(), block[start + 1, start + 1].item()
            trace = a + d
            determinant = a * d - b * c
            pairs.append((trace, determinant))
        elif single is None:
            single = block[start, start].item()
        else:
            value = block[start, start].item()
            pairs.append((single + value, single * value))
            single = None
    return pairs


def trailing_shifts(state, active_high, count):

    # shifts from the eigenvalues of the trailing count x count block, computed on a copy (the matrix does not change)
    top = active_high - count + 1
    window_start = round_down_to_tile(top)
    window_end = min(window_start + WINDOW_SIZE, state.padded_size)
    stacked = state.window(window_start, window_end)
    try:
        blocks = schur_on_host_block(stacked, top - window_start, active_high - window_start)
    except RuntimeError:
        return []

    # bottom blocks first
    return shift_pairs(stacked, sorted(blocks, reverse=True))


class HessenbergIterationState:

    def __init__(self, hessenberg, size, padded_size):

        self.hessenberg = hessenberg
        self.size = size
        self.padded_size = padded_size

        hessenberg_host = hessenberg.full()
        self.diagonal = torch.diagonal(hessenberg_host).tolist()[:size]
        self.subdiagonal = torch.diagonal(hessenberg_host, -1).tolist()[:max(size - 1, 0)]
        self.deflated = set()
        self.converged_blocks = []

    def window(self, start, end):

        block = self.hessenberg.block(start, end)

        # zero the deflated elements in the block
        for index in self.deflated:
            if start <= index and index + 1 < end:
                block[index + 1 - start, index - start] = 0.0

        # put the transform (identity for now) under the block, so a reflector applied from the right updates both
        return np.vstack([block, np.eye(end - start)])

    def update(self, start, end, stacked):

        # split stacked back into the block and the accumulated transform
        new_block, transform = stacked[:end - start], stacked[end - start:]

        self.hessenberg.update(start, end, new_block, transform)
        stop = min(end, self.size)
        self.diagonal[start:stop] = np.diagonal(new_block)[:stop - start].tolist()
        self.subdiagonal[start:stop - 1] = np.diagonal(new_block, -1)[:stop - 1 - start].tolist()

    def zeroable(self, index):
        return index in self.deflated or subdiagonal_zeroable(self.subdiagonal[index], self.diagonal[index], self.diagonal[index + 1])


def multishift_sweep(state, active_low, active_high, shifts):

    # chase one bulge per pair of shifts down the active block, every bulge moving one row per macro step
    bulges = len(shifts)
    last_step = active_high - 1 - active_low + BULGE_SPACING * (bulges - 1)
    step = 0

    while step <= last_step:

        # the window starts above the newest bulge (at the top of the block while bulges are still entering)
        newest_row = max(active_low, active_low + step - BULGE_SPACING * (bulges - 1))
        window_start = round_down_to_tile(newest_row - 1 if newest_row > active_low else newest_row)
        window_end = min(window_start + WINDOW_SIZE, state.padded_size)

        # get the current block with deflated elements zeroed and the transform accumulating the reflectors under it
        stacked = state.window(window_start, window_end)

        # chase until the oldest bulge reaches the bottom of the window
        step = chase_bulge_chain(stacked, window_start, active_low, active_high, shifts, step, last_step)

        # update the Hessenberg matrix (the products with the rest of the matrix run on the device)
        state.update(window_start, window_end, stacked)


def aggressive_early_deflation(state, active_high):

    window_top = active_high - AED_WINDOW_SIZE + 1
    block_start = round_down_to_tile(window_top - 1)
    block_end = min(block_start + WINDOW_SIZE, state.padded_size)

    stacked = state.window(block_start, block_end)
    block, transform = stacked[:block_end - block_start], stacked[block_end - block_start:]

    top = window_top - block_start
    bottom = active_high - block_start
    coupling = block[top, top - 1].item()

    # if it fails then no deflation and no shifts
    try:
        window_blocks = sorted(schur_on_host_block(stacked, top, bottom), reverse=True)
    except RuntimeError:
        return None, []

    spike = coupling * transform[top, top:bottom + 1]
    last_kept, deflatable = bottom, []

    for start, length in window_blocks:
        scale = sum(abs(block[start + q, start + q].item()) for q in range(length))
        if float(np.abs(spike[start - top:start - top + length]).max()) <= MACHINE_EPS * scale:
            deflatable.append((start, length))
            last_kept = start - 1
        else:
            break

    # the eigenvalues of the blocks that did not deflate are the shifts for the next sweep (bottom first)
    shifts = shift_pairs(block, window_blocks[len(deflatable):])

    if not deflatable:
        return None, shifts

    block[top:bottom + 1, top - 1] = spike
    block[last_kept + 1:bottom + 1, top - 1] = 0.0

    if last_kept >= top:
        for column in range(top - 1, last_kept - 1):
            v, tau, beta = householder_reflector(torch.from_numpy(block[column + 1:last_kept + 1, column].copy()))
            block[column + 1, column] = beta
            block[column + 2:last_kept + 1, column] = 0.0

            if v is not None:
                apply_reflector_two_sided(stacked, v.numpy(), tau, column + 1)

    state.update(block_start, block_end, stacked)
    for start, length in deflatable:
        state.converged_blocks.append((start + block_start, length))
    new_active_high = last_kept + block_start
    state.deflated.add(new_active_high)
    return new_active_high, shifts


def francis_qr_iteration(state, use_aed):

    # bottom row index of the unsolved block
    active_high = state.size - 1

    # convergence threshold per block
    max_sweeps = 30 * max(state.size, 10)

    # sweeps done on the current block
    sweeps_on_block = 0

    # last block's coordinates
    last_block = None

    while active_high >= 0:

        # find from the bottom up the first zeroable element on the subdiagonal
        active_low = active_high
        while active_low > 0 and not state.zeroable(active_low - 1):
            active_low -= 1

        # record the zeroed element if found
        if active_low > 0:
            state.deflated.add(active_low - 1)

        # compute the current block on host if small enough
        if active_high - active_low + 1 <= SMALL_BLOCK_SIZE:

            # get the size of the window that contains the current block
            window_start = round_down_to_tile(active_low)
            window_end = min(window_start + WINDOW_SIZE, state.padded_size)

            # get the current block with deflated elements zeroed and the transform under it
            stacked = state.window(window_start, window_end)

            # run Francis on the current block until it is in Schur form
            for start, length in schur_on_host_block(stacked, active_low - window_start, active_high - window_start):
                # record each converged block's coordinates
                state.converged_blocks.append((start + window_start, length))

            # update the Hessenberg matrix
            state.update(window_start, window_end, stacked)

            # update the index of the unsolved part of the matrix
            active_high = active_low - 1

            continue

        # reset the sweep counter if new big block
        if (active_low, active_high) != last_block:
            sweeps_on_block, last_block = 0, (active_low, active_high)

        # aggressive early deflation, which also gives the shifts for the next sweep
        shifts = []
        if use_aed:
            new_active_high, shifts = aggressive_early_deflation(state, active_high)
            if new_active_high is not None:
                deflated = active_high - new_active_high
                active_high = new_active_high

                # if enough eigenvalues deflated, look for more deflations before sweeping
                if 100 * deflated > AED_NIBBLE * AED_WINDOW_SIZE or active_high - active_low + 1 <= SMALL_BLOCK_SIZE:
                    continue

        # do one sweep and exit if not converging
        sweeps_on_block += 1
        if sweeps_on_block > max_sweeps:
            raise RuntimeError(f"Francis QR: no convergence in block [{active_low}, {active_high}]")

        # number of bulges (pairs of shifts) for this block
        bulges = min(MAX_BULGES, (active_high - active_low + 1) // 16)

        # get the bottom right 2x2 block for the exceptional and fallback shifts
        corner = state.hessenberg.block(active_high - 1, active_high + 1)
        a, b, c, d = corner[0, 0].item(), corner[0, 1].item(), corner[1, 0].item(), corner[1, 1].item()

        # use exceptional shifts if the block is not converging, else the eigenvalues from aed or from the trailing block
        if sweeps_on_block % EXCEPTIONAL_SHIFT_PERIOD == 0:
            shifts = [francis_shifts(a, b, c, d, True, state.subdiagonal[active_high - 2])] * bulges
        elif not shifts:
            shifts = trailing_shifts(state, active_high, 2 * bulges) or [francis_shifts(a, b, c, d, False, 0.0)]

        # do the sweep to create the bulges and then chase them together
        multishift_sweep(state, active_low, active_high, shifts[:bulges])


def eigenvalues_2x2(a, b, c, d):

    half_difference = 0.5 * (a - d)
    bc = b * c
    discriminant = half_difference ** 2 + bc

    # for real eigenvalues
    if discriminant >= 0.0:
        z = half_difference + math.copysign(math.sqrt(discriminant), half_difference)
        if z == 0.0:
            return complex(d), complex(d)
        return complex(d + z), complex(d - bc / z)

    # for complex conjugate pairs of eigenvalues
    real, imag = 0.5 * (a + d), math.sqrt(-discriminant)
    return complex(real, imag), complex(real, -imag)


def schur_eigenvectors(schur, blocks):

    # eigenvalues and eigenvectors X of the quasi-triangular Schur form T (T X = X diag(eigenvalues)):
    # every 2x2 block is made triangular by a 2x2 unitary W, then all the eigenvectors of the complex
    # upper triangular W^H T W come from one back substitution (here the parts coupling different blocks matter)
    size = schur.shape[0]
    triangular = schur.to(torch.complex128)
    unitary = torch.eye(size, dtype=torch.complex128)
    values = torch.zeros(size, dtype=torch.complex128)

    for start, length in blocks:
        if length == 1:
            values[start] = schur[start, start]
            continue

        # eigenvalues of the 2x2 block and an eigenvector of the first one (the larger of two equivalent formulas)
        a, b, c, d = schur[start, start].item(), schur[start, start + 1].item(), schur[start + 1, start].item(), schur[start + 1, start + 1].item()
        first, second = eigenvalues_2x2(a, b, c, d)
        u = (complex(b), first - a)
        other = (first - d, complex(c))
        if abs(other[0]) + abs(other[1]) > abs(u[0]) + abs(u[1]):
            u = other
        norm = math.hypot(abs(u[0]), abs(u[1]))
        if norm == 0.0:
            u, norm = (1.0 + 0j, 0j), 1.0
        u0, u1 = u[0] / norm, u[1] / norm

        # 2x2 unitary whose first column is that eigenvector: it makes the block upper triangular
        rotation = torch.tensor([[u0, -u1.conjugate()], [u1, u0.conjugate()]], dtype=torch.complex128)
        triangular[start:start + 2, :] = rotation.conj().T @ triangular[start:start + 2, :]
        triangular[:, start:start + 2] = triangular[:, start:start + 2] @ rotation
        triangular[start + 1, start] = 0.0
        triangular[start, start], triangular[start + 1, start + 1] = first, second
        unitary[start:start + 2, start:start + 2] = rotation
        values[start], values[start + 1] = first, second

    # back substitution for all the eigenvectors at once, row by row from the bottom
    # (a denominator smaller than eps * max|T| is replaced by that value, as for nearly equal eigenvalues)
    smallest = MACHINE_EPS * (float(schur.abs().max()) or 1.0)
    vectors = torch.eye(size, dtype=torch.complex128)
    for row in range(size - 2, -1, -1):
        numerator = triangular[row, row + 1:] @ vectors[row + 1:, row + 1:]
        denominator = triangular[row, row] - values[row + 1:]
        denominator = torch.where(denominator.abs() < smallest, torch.full_like(denominator, smallest), denominator)
        vectors[row, row + 1:] = -numerator / denominator

    # back to the basis of T
    vectors = unitary @ vectors
    return values, vectors / vectors.abs().amax(dim=0, keepdim=True)


class EigenvalueResult:

    def __init__(self, eigenvalues, eigenvectors, reduced, reduction_vectors, schur_vectors, schur_form, symmetric):

        # computed eigenvalues
        self.eigenvalues = eigenvalues

        # computed eigenvectors, column i belongs to eigenvalues[i] (unit 2-norm columns)
        self.eigenvectors = eigenvectors

        # reduced matrix (Hessenberg/tridiagonal) and its reflectors Q (A = Q reduced Q^T), for verification
        self.reduced = reduced
        self.reduction_vectors = reduction_vectors

        # Schur vectors S and Schur form T (A = S T S^T), for verification
        self.schur_vectors = schur_vectors
        self.schur_form = schur_form

        # symmetric boolean
        self.symmetric = symmetric

def eigenvalues(matrix, device, symmetric=None, use_aed="auto"):

    matrix = matrix.to(F64)
    size = matrix.shape[0]

    # check that the input matrix is square
    assert matrix.dim() == 2 and matrix.shape[1] == size, "input matrix must be square"

    # check if the input matrix size is large enough to use aed
    if use_aed == "auto":
        use_aed = size >= AED_AUTO_MIN_SIZE

    # check if the input matrix is symmetric (A == A^T)
    if symmetric is None:
        symmetric = bool(torch.allclose(matrix, matrix.T))

    # add padding to the input matrix to make it a multiple of TILE_SIZE
    padded_size = max(TILE_SIZE, math.ceil(size / TILE_SIZE) * TILE_SIZE)
    padded = torch.zeros(padded_size, padded_size, dtype=F64)
    padded[:size, :size] = matrix
    matrix_device = to_device(padded, device)

    if symmetric:
        # compute directly the diagonal and off diagonal elements of the tridiagonal matrix, and its reflectors Q
        diagonal, off_diagonal, q_device = reduce_to_tridiagonal(matrix_device, size, padded_size, device)

        # assemble the reduced tridiagonal matrix for verification
        reduced = torch.diag(torch.tensor(diagonal, dtype=F64))
        if size > 1:
            off = torch.tensor(off_diagonal, dtype=F64)
            reduced += torch.diag(off, 1) + torch.diag(off, -1)

        # start the QR factorization process, collecting its rotations in the Schur vectors Z
        vectors = torch.eye(padded_size, dtype=F64)
        tridiagonal_qr_iteration(diagonal, off_diagonal, size, vectors)

        # eigenvectors S = Q Z with one device product, in ascending order of the eigenvalues
        schur_vectors = to_host(matmul(q_device, to_device(vectors, device)))[:size, :size]
        values = torch.tensor(diagonal, dtype=F64)
        order = torch.argsort(values)
        eigenvectors = schur_vectors[:, order]
        eigenvectors = eigenvectors / torch.linalg.vector_norm(eigenvectors, dim=0, keepdim=True)

        return EigenvalueResult(eigenvalues=values[order], eigenvectors=eigenvectors, reduced=reduced,
                                reduction_vectors=to_host(q_device)[:size, :size], schur_vectors=schur_vectors,
                                schur_form=torch.diag(values), symmetric=True)

    if not symmetric:
        # compute the Hessenberg form of the matrix and its reflectors Q on the device
        hessenberg_device, q_device = reduce_to_hessenberg(matrix_device, size, padded_size, device)
        hessenberg_host = to_host(hessenberg_device)

        # copy the reduced Hessenberg matrix for verification
        reduced = hessenberg_host[:size, :size].clone()

        # initialize needed instances
        hessenberg = HostHessenberg(hessenberg_host, device)
        state = HessenbergIterationState(hessenberg, size, padded_size)

        # start the QR factorization process
        francis_qr_iteration(state, use_aed)

        # sort the "coordinates" of the converged Schur blocks
        blocks = sorted(state.converged_blocks)

        # check that they cover the matrix's original size diagonally
        assert sum(length for _, length in blocks) == size, "Schur blocks do not cover the matrix"

        # get the Schur form T: the iterated matrix with exact zeros below the diagonal outside the 2x2 blocks
        schur_form = torch.triu(hessenberg.full()[:size, :size], diagonal=-1)
        two_by_two = {start for start, length in blocks if length == 2}
        for index in range(size - 1):
            if index not in two_by_two:
                schur_form[index + 1, index] = 0.0

        # Schur vectors S = Q Z with one device product (A = S T S^T)
        schur_vectors_device = matmul(q_device, to_device(hessenberg.vectors, device))

        # eigenvalues and eigenvectors X of the Schur form on the host
        values, schur_form_vectors = schur_eigenvectors(schur_form, blocks)

        # eigenvectors V = S X on the device (real and imaginary parts separately), unit 2-norm columns
        real = torch.zeros(padded_size, padded_size, dtype=F64)
        imag = torch.zeros(padded_size, padded_size, dtype=F64)
        real[:size, :size], imag[:size, :size] = schur_form_vectors.real, schur_form_vectors.imag
        eigenvectors = torch.complex(to_host(matmul(schur_vectors_device, to_device(real, device)))[:size, :size],
                                     to_host(matmul(schur_vectors_device, to_device(imag, device)))[:size, :size])
        eigenvectors = eigenvectors / torch.linalg.vector_norm(eigenvectors, dim=0, keepdim=True)

        return EigenvalueResult(eigenvalues=values, eigenvectors=eigenvectors, reduced=reduced,
                                reduction_vectors=to_host(q_device)[:size, :size],
                                schur_vectors=to_host(schur_vectors_device)[:size, :size],
                                schur_form=schur_form, symmetric=False)


def make_matrix(size, symmetric, generator):

    matrix = torch.randn(size, size, dtype=F64, generator=generator)
    if symmetric:
        matrix = matrix + matrix.T
    return matrix.to(F32).to(F64)


def max_matched_error(computed, reference):

    remaining = list(reference)
    worst = 0.0
    for value in computed:
        distances = [abs(candidate - value) for candidate in remaining]
        smallest = min(distances)
        nearest = distances.index(smallest)
        remaining.pop(nearest)
        worst = max(worst, smallest)
    return worst


def report(matrix, result):

    size = matrix.shape[0]
    passed = True

    # spectral norm of the input matrix
    norm = torch.linalg.matrix_norm(matrix, 2).item()
    if norm == 0.0:
        for value in result.eigenvalues.tolist():
            if value != 0:
                print("FAIL: non-zero eigenvalue for zero matrix")
        print("PASS: zero matrix eigenvalues are all zero")
        return

    # tolerance used for comparing the eigenvalues
    tolerance = 10.0 * DEVICE_MATMUL_ROUNDOFF * math.sqrt(size)
    print("Tolerance: ", tolerance)

    if result.symmetric:
        # check error difference from eigenvalues computed from the input matrix
        error = (result.eigenvalues - torch.linalg.eigvalsh(matrix)).abs().max().item() / norm
        print("Error difference (computed eigenvalues / input matrix): ", error)
        if error > tolerance:
            passed = False

        # check error difference from eigenvalues computed from the reduced tridiagonal matrix
        error = (result.eigenvalues - torch.linalg.eigvalsh(result.reduced)).abs().max().item() / norm
        print("Error difference (computed eigenvalues / reduced tridiagonal matrix): ", error)
        if error > tolerance:
            passed = False
    else:
        # check error difference from eigenvalues computed from the input matrix (not counted because of conditioning)
        error = max_matched_error(result.eigenvalues.tolist(), torch.linalg.eigvals(matrix).tolist()) / norm
        print("Error difference (computed eigenvalues / input matrix): ", error)

        # check error difference from eigenvalues computed from the reduced Hessenberg matrix
        error = max_matched_error(result.eigenvalues.tolist(), torch.linalg.eigvals(result.reduced).tolist()) / norm
        print("Error difference (computed eigenvalues / reduced Hessenberg matrix): ", error)
        if error > tolerance:
            passed = False

    identity = torch.eye(size, dtype=F64)
    frobenius = torch.linalg.matrix_norm(matrix).item()

    # check the reduction A = Q R Q^T and the orthogonality of Q
    q = result.reduction_vectors
    checks = [("Reduction residual ||Q R Q^T - A|| / ||A||: ", torch.linalg.matrix_norm(q @ result.reduced @ q.T - matrix).item() / frobenius),
              ("Reduction orthogonality ||Q^T Q - I||_2: ", torch.linalg.matrix_norm(q.T @ q - identity, 2).item())]

    # check the Schur form A = S T S^T
    if not result.symmetric:
        s = result.schur_vectors
        checks.append(("Schur residual ||S T S^T - A|| / ||A||: ", torch.linalg.matrix_norm(s @ result.schur_form @ s.T - matrix).item() / frobenius))

    # check the eigenvectors: A v_i = lambda_i v_i (and orthonormal for a symmetric matrix)
    vectors = result.eigenvectors
    residuals = torch.linalg.vector_norm(matrix.to(vectors.dtype) @ vectors - vectors * result.eigenvalues[None, :], dim=0)
    checks.append(("Eigenvector residual max ||A v - lambda v|| / ||A||_2: ", residuals.max().item() / norm))
    if result.symmetric:
        checks.append(("Eigenvector orthogonality ||V^T V - I||_2: ", torch.linalg.matrix_norm(vectors.T @ vectors - identity, 2).item()))

    for label, error in checks:
        print(label, error)
        if error > tolerance:
            passed = False

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":

    DEBUG_MAX_PRINT_SIZE = 8
    DEBUG_PRECISION = 6

    positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    flags = [arg for arg in sys.argv[1:] if arg.startswith("--")]

    # default input matrix parameters
    size = 64
    symmetric = False # generic or symmetric

    if len(positional) > 0:
        size = int(positional[0])
    if len(positional) > 1:
        symmetric = positional[1] == "symmetric"

    # additional flags (seed, aed)
    seed = 0
    use_aed = "auto"

    for flag in flags:
        if flag.startswith("--seed="):
            seed = int(flag.split("=")[1])
        elif flag == "--aed" and use_aed == "auto":
            use_aed = True
        elif flag == "--no-aed":
            use_aed = False

    # generate the requested input matrix from args
    matrix = make_matrix(size, symmetric, torch.Generator().manual_seed(seed))

    device = ttnn.open_device(device_id=0)
    try:
        try:
            start_time = time.perf_counter()
            result = eigenvalues(matrix, device, symmetric=symmetric, use_aed=use_aed)
            end_time = time.perf_counter() - start_time
            print(f"Execution time: {end_time:.6f} seconds")
        except RuntimeError as error:
            print(f"Failed to compute eigenvalues: {error}")
            sys.exit(1)

        # verification phase
        report(matrix, result)

        # debug prints
        if size <= DEBUG_MAX_PRINT_SIZE:
            torch.set_printoptions(precision=DEBUG_PRECISION, sci_mode=False)
            print("Input matrix A:\n", matrix)
            print("Reduced matrix:\n", result.reduced)
            print("Computed eigenvalues:\n", result.eigenvalues)
            print("Computed eigenvectors (column i belongs to eigenvalue i):\n", result.eigenvectors)

            torch_eigenvalues = torch.linalg.eigvals(matrix)
            print("Torch eigenvalues:\n", torch_eigenvalues)
    finally:
        ttnn.close_device(device)
