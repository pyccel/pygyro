import numpy as np
import scipy.sparse as sparse

from .matrix_assembly_pyccel import lists_order2_dirichlet, lists_order2_periodic, lists_order4_dirichlet, lists_order4_periodic


def assemble_bracket_arakawa(bc, order, phi, grid_theta, grid_r):
    """
    Assemble the Arakawa bracket J: f -> {phi, f} as a sparse matrix

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'; which boundary conditions to use in r-direction

        phi : np.ndarray
            array of length N_theta*N_r; point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        res : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*N_r;
            Stencil matrix for discrete Poisson bracket: J * f = {phi, f}
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r)
    N_nodes = N_theta * N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor = -1 / (24 * dr * dtheta)

    if bc == 'periodic':

        if order == 2:
            size = N_r * N_theta * 9

            row = np.zeros(size, dtype=int)
            col = np.zeros(size, dtype=int)
            data = np.zeros(size, dtype=float)

            lists_order2_periodic(N_theta, N_r, phi, row, col, data)

        elif order == 4:
            size = N_r * N_theta * 17

            row = np.zeros(size, dtype=int)
            col = np.zeros(size, dtype=int)
            data = np.zeros(size, dtype=float)

            lists_order4_periodic(N_theta, N_r, phi, row, col, data)

        else:
            raise NotImplementedError(
                f'A scheme of order {order} is not implemented!')

    elif bc == 'dirichlet':
        if order == 2:
            size = N_theta * ((N_r - 2) * 9 + 6 + 6)

            row = np.zeros(size, dtype=int)
            col = np.zeros(size, dtype=int)
            data = np.zeros(size, dtype=float)

            lists_order2_dirichlet(N_theta, N_r, phi, row, col, data)

        elif order == 4:
            size = N_theta * ((N_r - 2) * 9 + 6 + 6) \
                    + N_theta * ((N_r - 4) * 8 + 12 + 12)

            row = np.zeros(size, dtype=int)
            col = np.zeros(size, dtype=int)
            data = np.zeros(size, dtype=float)

            lists_order4_dirichlet(N_theta, N_r, phi, row, col, data)

        else:
            raise NotImplementedError(
                f'A scheme of order {order} is not implemented!')

    else:
        raise NotImplementedError(
            f'{bc} is an unknown option for boundary conditions')

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)

    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()

    return J
