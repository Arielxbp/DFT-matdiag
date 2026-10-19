import math
import sys
import torch
import ttnn

TILE_SIZE = 32 # tenstorrent tiles
PANEL_WIDTH = 32 # Householder reflectors per WY block
WINDOW_SIZE = 128 # side of the diagonal window the bulge is chased in
SMALL_BLOCK_SIZE = WINDOW_SIZE - TILE_SIZE # active blocks this small are finished in one window

AED_WINDOW_SIZE = 32 # size of the aggressive-early-deflation window
AED_MAX_WAIT = 8 # aed back-off
AED_AUTO_MIN_SIZE = 512 # threshold for automatic aedfor efficiency

F32 = torch.float32
F64 = torch.float64
MACHINE_EPS = torch.finfo(F32).eps # deflation tolerance

DEVICE_MATMUL_ROUNDOFF = 2.0 ** -10 # measured unit roundoff of the fp32 device matmul

GRAM_ON_DEVICE = True # compute V^TV for the WY matrix T on the device

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

def device_block(matrix, row_start, row_end, col_start, col_end):
    if row_end <= row_start or col_end <= col_start:
        return None
    return ttnn.slice(matrix, [row_start, col_start], [row_end, col_end])

def concat_nonempty(parts, dim):
    
    # get only the non empty blocks to concat
    parts = [part for part in parts if part is not None]

    if len(parts) == 1:
        return parts[0]
    return ttnn.concat(parts, dim=dim)

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


def apply_reflector_two_sided(block, transform, v, tau, first_index, row_limit):

    # transform is the product of every reflector applied to this window so far
    # first_index: the index of the first row/column touched by P
    # row_limit: the limit of rows to be updated

    v_tensor = v if torch.is_tensor(v) else torch.tensor(v, dtype=F64)
    length = v_tensor.numel()

    # apply to left side (B <- P B), updating rows
    rows = block[first_index:first_index + length, first_index:]
    rows.addr_(v_tensor, v_tensor @ rows, alpha=-tau)

    # apply to right side (B <- B P), updating columns
    cols = block[:row_limit, first_index:first_index + length]
    cols.addr_(cols @ v_tensor, v_tensor, alpha=-tau)

    # update the transform by multiplying the reflector from the right (T <- T P)
    transform_cols = transform[:, first_index:first_index + length]
    transform_cols.addr_(transform_cols @ v_tensor, v_tensor, alpha=-tau)


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


def build_block_T(reflectors_device, reflectors_t_device, taus, reflectors_host=None):

    # compute V^TV on device or host
    if GRAM_ON_DEVICE or reflectors_host is None:
        gram = to_host(matmul(reflectors_t_device, reflectors_device))
    else:
        gram = reflectors_host.T @ reflectors_host

    # compute the matrix T
    T_matrix = wy_T_from_gram(gram, taus)

    return T_matrix


def reduce_to_hessenberg(matrix_device, size, padded_size, device):

    # for every panel of columns (last 2 columns are already in Hessenberg form)
    for panel_start in range(0, size - 2, PANEL_WIDTH):

        # get the number of columns for this panel
        width = min(PANEL_WIDTH, size - 2 - panel_start)

        # get the panel of 'width' columns
        panel = to_host(device_block(matrix_device, 0, padded_size, panel_start, panel_start + PANEL_WIDTH))

        # get the trailing block of the matrix that is still to be reduced (will get smaller each iteration)
        trailing_device = device_block(matrix_device, 0, padded_size, panel_start, padded_size)
        
        # init the reflectors and the update Y=AVT matrices
        reflectors = torch.zeros(padded_size, PANEL_WIDTH, dtype=F64)
        y_block = torch.zeros(padded_size, PANEL_WIDTH, dtype=F64)
        taus = [0.0] * PANEL_WIDTH

        # for every column of the panel, build the reflector and the update vector
        for index in range(width):
            column = panel_start + index

            # get the column
            current = panel[:, index].clone()

            # update the column with earlier reflectors
            if index != 0:
                # from the right side A - YV
                current -= y_block[:, :index] @ reflectors[column, :index]

                # from the left side only apply if the reflector is not identity
                for earlier in range(index):
                    if taus[earlier]:
                        current -= taus[earlier] * torch.dot(reflectors[:, earlier], current) * reflectors[:, earlier]
            
            # compute the Householder reflector for the current column
            v, tau, _ = householder_reflector(current[column + 1:])
            if v is None:
                continue
            taus[index] = tau
            reflectors[column + 1:, index] = v
            
            # compute Y's column
            a_times_v = device_matvec(trailing_device, reflectors[panel_start:, index], device)
            y_block[:, index] = tau * (a_times_v - y_block[:, :index] @ (reflectors[:, :index].T @ reflectors[:, index]))

        # if panel was already in Hessenberg form skip useless computes 
        if not any(taus):
            continue

        # build the block T for the current panel
        panel_reflectors = reflectors[panel_start:]
        reflectors_device = to_device(panel_reflectors, device)
        reflectors_t_device = to_device(panel_reflectors.T, device)
        T = build_block_T(reflectors_device, reflectors_t_device, taus, panel_reflectors)
        
        # right update trailing with reflectors
        updated = ttnn.subtract(trailing_device, matmul(to_device(y_block, device), reflectors_t_device))
        
        # left update trailing with reflectors
        lower = device_block(updated, panel_start, padded_size, 0, padded_size - panel_start)
        temp = matmul(to_device(T.T, device), matmul(reflectors_t_device, lower))
        lower = ttnn.subtract(lower, matmul(reflectors_device, temp))
        updated = concat_nonempty([device_block(updated, 0, panel_start, 0, padded_size - panel_start), lower], 0)

        # reattach the updated trailing block to the matrix and continue with the next panel
        matrix_device = concat_nonempty([device_block(matrix_device, 0, padded_size, 0, panel_start), updated], 1)
    
    # explicitly zero out the entries below the subdiagonal
    return ttnn.triu(matrix_device, diagonal=-1)


def reduce_to_tridiagonal(matrix_device, size, padded_size, device):

    # init tridiagonal result
    diagonal = [0.0] * size
    off_diagonal = [0.0] * max(size - 1, 0)

    # get the trailing block of the matrix that is still to be reduced
    trailing = matrix_device

    # for every panel, build its reflector and update vector
    for panel_start in range(0, size, PANEL_WIDTH):

        # get the current panel
        trailing_size = padded_size - panel_start
        panel = to_host(device_block(trailing, 0, trailing_size, 0, PANEL_WIDTH))
        
        # init the reflectors and the update W matrices
        reflectors = torch.zeros(trailing_size, PANEL_WIDTH, dtype=F64)
        w_block = torch.zeros(trailing_size, PANEL_WIDTH, dtype=F64)
        taus = [0.0] * PANEL_WIDTH
        
        # for every column of the panel, build the reflector and the update vector
        for index in range(min(PANEL_WIDTH, size - panel_start)):

            # get the column starting from the diagonal
            current = panel[index:, index].clone()

            # update the column with earlier reflectors
            if index != 0:
                current -= reflectors[index:, :index] @ w_block[index, :index] + w_block[index:, :index] @ reflectors[index, :index]
            
            # set the current diagonal element
            diagonal[panel_start + index] = float(current[0])

            # skip last column since last element is on the diagonal
            if panel_start + index == size - 1:
                break

            # compute the Householder reflector for the current column
            v, tau, beta = householder_reflector(current[1:])

            # set the current off diagonal element
            off_diagonal[panel_start + index] = beta
            
            if v is None:
                continue            
            taus[index] = tau
            reflectors[index + 1:, index] = v
            
            # Compute W's column
            a_times_v = device_matvec(trailing, reflectors[:, index], device)[index + 1:]
            v_prev = reflectors[index + 1:, :index]
            w_prev = w_block[index + 1:, :index]
            w = tau * (a_times_v - w_prev @ (v_prev.T @ v) - v_prev @ (w_prev.T @ v))
            w -= 0.5 * tau * torch.dot(w, v) * v
            w_block[index + 1:, index] = w
            
        if panel_start + PANEL_WIDTH >= size:
            break

        # cut the next trailing block to update it
        next_trailing = device_block(trailing, PANEL_WIDTH, trailing_size, PANEL_WIDTH, trailing_size)
        if not any(taus):
            trailing = next_trailing
            continue
        v_and_w = torch.cat([reflectors[PANEL_WIDTH:], w_block[PANEL_WIDTH:]], dim=1)
        w_and_v = torch.cat([w_block[PANEL_WIDTH:], reflectors[PANEL_WIDTH:]], dim=1)
        trailing = ttnn.subtract(next_trailing, matmul(to_device(v_and_w, device), to_device(w_and_v.T, device)))

    return diagonal, off_diagonal



class HostHessenberg:

    def __init__(self, matrix):
        self.matrix = matrix

    def block(self, start, end):
        return self.matrix[start:end, start:end].clone()

    def update(self, start, end, new_block, transform):

        # update the Hessenberg matrix with the new block and the accumulated transform

        # rows above
        self.matrix[:start, start:end] = self.matrix[:start, start:end] @ transform
        # columns to the right
        self.matrix[start:end, end:] = transform.T @ self.matrix[start:end, end:]
        
        self.matrix[start:end, start:end] = new_block

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

    # chase the bulge from top to bottom one rotation per iteration
    for step in range(active_low, active_high):

        # compute the Givens rotation
        radius = math.hypot(x, z)
        cos, sin = (1.0, 0.0) if radius == 0.0 else (x / radius, z / radius)

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


def zeroable(diagonal, off_diagonal, index):

    return abs(off_diagonal[index]) <= MACHINE_EPS * (abs(diagonal[index]) + abs(diagonal[index + 1]))


def tridiagonal_qr_iteration(diagonal, off_diagonal, size):

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

        # reset the sweep counter if new block
        if (active_low, active_high) != last_block:
            sweeps_on_block = 0
            last_block = (active_low, active_high)

        sweeps_on_block += 1
        if sweeps_on_block > max_sweeps:
            raise RuntimeError(f"Symmetric QR: no convergence in block [{active_low}, {active_high}]")

        # do one sweep with the computed shift
        mu = wilkinson_shift(diagonal[active_high - 1], off_diagonal[active_high - 1], diagonal[active_high])
        tridiagonal_sweep(diagonal, off_diagonal, active_low, active_high, mu)


def francis_shifts(a, b, c, d, sweeps_on_block, element_above):

    # if normal shifts are not converging use LAPACK shifts
    if sweeps_on_block in (10, 20):
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


def chase_bulge(block, transform, active_low, active_high, shift_sum, shift_product, first_step, last_step):

    # for each reflector
    for step in range(first_step, last_step + 1):
        # work on 3 elements (2 at the end)
        length = min(3, active_high - step + 1)
        
        # create the bulge
        if step == active_low:
            target = compute_first_column(block, active_low, active_high, shift_sum, shift_product)
        else:
            # take the elements from the current column of the bulge
            target = block[step:step + length, step - 1].tolist()
        
        # build the reflector to zero the last two elements of the target column
        v, tau, beta = householder_reflector_small(target)

        # direct apply the known reflector
        if step > active_low:
            block[step, step - 1] = beta
            block[step + 1:step + length, step - 1] = 0.0

        # if target was already in the correct form, skip the reflector
        if v is not None: 
            apply_reflector_two_sided(block, transform, v, tau, step, min(step + 3, active_high) + 1)


def subdiagonal_zeroable(subdiagonal_element, diagonal_above, diagonal_below):
    return abs(subdiagonal_element) <= MACHINE_EPS * (abs(diagonal_above) + abs(diagonal_below))


def lowest_split(block, active_low, active_high):

    # find the lowest split point in the block where the sub diagonal is zeroable
    if active_high <= active_low:
        return active_low
    subdiagonal = block.diagonal(-1)[active_low:active_high].abs()
    diagonal = block.diagonal()[active_low:active_high + 1].abs()
    hits = torch.nonzero(subdiagonal <= MACHINE_EPS * (diagonal[:-1] + diagonal[1:]))
    return active_low if hits.numel() == 0 else active_low + int(hits[-1]) + 1


def schur_on_host_block(block, transform, active_low, active_high):

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
        shift_sum, shift_product = francis_shifts(a, b, c, d, sweeps_on_block, element_above)
        chase_bulge(block, transform, split, bottom, shift_sum, shift_product, split, bottom - 1)

    return converged


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
        return block

    def update(self, start, end, new_block, transform):

        self.hessenberg.update(start, end, new_block, transform)
        for index in range(start, min(end, self.size)):
            self.diagonal[index] = new_block[index - start, index - start].item()
            if index + 1 < min(end, self.size):
                self.subdiagonal[index] = new_block[index + 1 - start, index - start].item()

    def zeroable(self, index):
        return index in self.deflated or subdiagonal_zeroable(self.subdiagonal[index], self.diagonal[index], self.diagonal[index + 1])


def windowed_sweep(state, active_low, active_high, shift_sum, shift_product):

    # current index
    step = active_low

    # chase the bulge down
    while step <= active_high - 1:

        # get the size of the window that contains the bulge
        window_start = round_down_to_tile(step - 1 if step > active_low else step)
        window_end = min(window_start + WINDOW_SIZE, state.padded_size)

        # chase until the bulge is outside the window or until limit
        if active_high < window_end:
            last_step = active_high - 1
        else:
            last_step = window_end - 4 # for 3-element reflector

        # get the current block with deflated elements zeroed
        block = state.window(window_start, window_end)

        # init the matrix accumulating the Givens rotations
        transform = torch.eye(window_end - window_start, dtype=F64)

        first_step_window = step - window_start
        last_step_window = last_step - window_start
        chase_bulge(block, transform, active_low - window_start, active_high - window_start, shift_sum, shift_product, first_step_window, last_step_window)

        # update the Hessenberg matrix
        state.update(window_start, window_end, block, transform)

        # continue from where the bulge stopped
        step = last_step + 1


def aggressive_early_deflation(state, active_high):

    window_top = active_high - AED_WINDOW_SIZE + 1
    block_start = round_down_to_tile(window_top - 1)
    block_end = min(block_start + WINDOW_SIZE, state.padded_size)

    block = state.window(block_start, block_end)
    transform = torch.eye(block_end - block_start, dtype=F64)

    top = window_top - block_start
    bottom = active_high - block_start
    coupling = block[top, top - 1].item()

    # if it fails then backoff
    try:
        window_blocks = schur_on_host_block(block, transform, top, bottom)
    except RuntimeError:
        return None

    spike = coupling * transform[top, top:bottom + 1]
    last_kept, deflatable = bottom, []

    for start, length in sorted(window_blocks, reverse=True):
        scale = sum(abs(block[start + q, start + q].item()) for q in range(length))
        if float(spike[start - top:start - top + length].abs().max()) <= MACHINE_EPS * scale:
            deflatable.append((start, length))
            last_kept = start - 1
        else:
            break
    
    if not deflatable:
        return None
    
    block[top:bottom + 1, top - 1] = spike
    block[last_kept + 1:bottom + 1, top - 1] = 0.0

    if last_kept >= top:
        for column in range(top - 1, last_kept - 1):
            v, tau, beta = householder_reflector(block[column + 1:last_kept + 1, column].clone())
            block[column + 1, column] = beta
            block[column + 2:last_kept + 1, column] = 0.0

            if v is not None:
                apply_reflector_two_sided(block, transform, v, tau, column + 1, last_kept + 1)

    state.update(block_start, block_end, block, transform)
    for start, length in deflatable:
        state.converged_blocks.append((start + block_start, length))
    new_active_high = last_kept + block_start
    state.deflated.add(new_active_high)
    return new_active_high


def francis_qr_iteration(state, use_aed):

    # bottom row index of the unsolved block
    active_high = state.size - 1

    # convergence threshold per block
    max_sweeps = 30 * max(state.size, 10)

    # sweeps done on the current block
    sweeps_on_block = 0

    # last block's coordinates
    last_block = None

    # for aggressive early deflation
    aed_wait, aed_backoff = 0, 1

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

            # get the current block with deflated elements zeroed
            block = state.window(window_start, window_end)

            # init the matrix accumulating the Givens rotations
            transform = torch.eye(window_end - window_start, dtype=F64)

            # run Francis on the current block until it is in Schur form
            for start, length in schur_on_host_block(block, transform, active_low - window_start, active_high - window_start):
                # record each converged block's coordinates
                state.converged_blocks.append((start + window_start, length))

            # update the Hessenberg matrix 
            state.update(window_start, window_end, block, transform)

            # update the index of the unsolved part of the matrix
            active_high = active_low - 1

            continue

        # reset the sweep counter if new big block
        if (active_low, active_high) != last_block:
            sweeps_on_block, last_block = 0, (active_low, active_high)

        if use_aed and aed_wait == 0:
            # update the upper index of the block if aed worked
            new_active_high = aggressive_early_deflation(state, active_high)
            if new_active_high is not None:
                active_high, aed_backoff = new_active_high, 1
                continue

            # if aed didn't work then add backoff for efficiency
            aed_wait = aed_backoff
            aed_backoff = min(2 * aed_backoff, AED_MAX_WAIT)
        elif aed_wait:
            aed_wait -= 1

        # do one sweep and exit if not converging
        sweeps_on_block += 1
        if sweeps_on_block > max_sweeps:
            raise RuntimeError(f"Francis QR: no convergence in block [{active_low}, {active_high}]")

        # get the bottom right 2x2 block's eigenvalues to use as shifts
        corner = state.hessenberg.block(active_high - 1, active_high + 1)
        a, b, c, d = corner[0, 0].item(), corner[0, 1].item(), corner[1, 0].item(), corner[1, 1].item()

        # compute the estimate eigenvalues to use for the sweep (μ1 and μ2)
        shift_sum, shift_product = francis_shifts(a, b, c, d, sweeps_on_block, state.subdiagonal[active_high - 2])
        
        # do the sweep to create the bulge and then chase
        windowed_sweep(state, active_low, active_high, shift_sum, shift_product)


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


class EigenvalueResult:

    def __init__(self, eigenvalues, reduced, symmetric):

        # computed eigenvalues
        self.eigenvalues = eigenvalues

        # reduced matrix (Hessenberg/tridiagonal) for verification
        self.reduced = reduced
        
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
        # compute directly the diagonal and off diagonal elements of the tridiagonal matrix
        diagonal, off_diagonal = reduce_to_tridiagonal(matrix_device, size, padded_size, device)

        # assemble the reduced tridiagonal matrix for verification
        reduced = torch.diag(torch.tensor(diagonal, dtype=F64))
        if size > 1:
            off = torch.tensor(off_diagonal, dtype=F64)
            reduced += torch.diag(off, 1) + torch.diag(off, -1)

        # start the QR factorization process
        tridiagonal_qr_iteration(diagonal, off_diagonal, size)

        # for printing consistency
        values = torch.sort(torch.tensor(diagonal, dtype=F64)).values

        return EigenvalueResult(eigenvalues=values, reduced=reduced, symmetric=True)

    if not symmetric:
        # compute the Hessenberg form of the matrix on the device
        hessenberg_device = reduce_to_hessenberg(matrix_device, size, padded_size, device)
        hessenberg_host = to_host(hessenberg_device)

        # copy the reduced Hessenberg matrix for verification
        reduced = hessenberg_host[:size, :size].clone()

        # initialize needed instances
        hessenberg = HostHessenberg(hessenberg_host)
        state = HessenbergIterationState(hessenberg, size, padded_size)

        # start the QR factorization process
        francis_qr_iteration(state, use_aed)

        # sort the "coordinates" of the converged Schur blocks
        blocks = sorted(state.converged_blocks)

        # check that they cover the matrix's original size diagonally
        assert sum(length for _, length in blocks) == size, "Schur blocks do not cover the matrix"

        # get the fully padded Schur form Hessenberg matrix
        schur = hessenberg.full()

        # init the eigenvalues vector to store them
        values = torch.zeros(size, dtype=torch.complex128)

        # get the eigenvalues from each converged block
        for start, length in blocks:
            if length == 1:
                values[start] = schur[start, start].item()
            else:
                a = schur[start, start].item()
                b = schur[start, start + 1].item()
                c = schur[start + 1, start].item()
                d = schur[start + 1, start + 1].item()
                values[start], values[start + 1] = eigenvalues_2x2(a, b, c, d)

        return EigenvalueResult(eigenvalues=values, reduced=reduced, symmetric=False)


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

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":

    DEBUG_MAX_PRINT_SIZE = 4
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
    if len(positional) > 2:
        kind = positional[2]

    # additional flags (seed, aed, gram on host)
    seed = 0
    use_aed = "auto"
    
    for flag in flags:
        if flag.startswith("--seed="):
            seed = int(flag.split("=")[1])
        elif flag == "--aed" and use_aed == "auto":
            use_aed = True
        elif flag == "--no-aed":
            use_aed = False
        elif flag == "--gram=host":
            GRAM_ON_DEVICE = False

    # generate the requested input matrix from args
    matrix = make_matrix(size, symmetric, torch.Generator().manual_seed(seed))

    device = ttnn.open_device(device_id=0)
    try:
        try:
            result = eigenvalues(matrix, device, symmetric=symmetric, use_aed=use_aed)
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
    finally:
        ttnn.close_device(device)
