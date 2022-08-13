import numpy as np
import scipy.sparse as sparse
from .utilities import neighbour_index, ind_to_tp_ind


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

    if bc == 'periodic':
        J = assemble_awk_bracket_periodic(phi, grid_theta, grid_r)
        if order == 2:
            res = J
        elif order == 4:
            J2 = assemble_awk_bracket_4th_order_periodic(
                phi, grid_theta, grid_r)
            res = 2 * J - J2

    elif bc == 'dirichlet':
        J = assemble_awk_bracket_dirichlet(phi, grid_theta, grid_r)

        if order == 2:
            res = J
        elif order == 4:
            J2 = assemble_awk_bracket_4th_order_dirichlet(
                phi, grid_theta, grid_r)
            res = 2 * J - J2

    elif bc == 'extrapolation':
        if order == 2:
            res = assemble_awk_bracket_dirichlet_extrapolation(
                phi, grid_theta, grid_r)
        elif order == 4:
            res = assemble_awk_bracket_4th_order_dirichlet_extrapolation(
                phi, grid_theta, grid_r)

    else:
        raise NotImplementedError(
            f'{bc} is an unknown option for boundary conditions')

    return res

#######################################
# Arakawas paper implementation of BC #
#######################################


def assemble_awk_bracket_periodic(phi, grid_theta, grid_r):
    """
    Assemble a discrete bracket J: f -> {phi, f} based on the Arakawa scheme as
    a sparse matrix with periodic boundary conditions in r-direction

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'

        phi : np.ndarray
            array of length N_theta*N_r; point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*N_r;
            Stencil matrix for discrete Poisson bracket: J * f = {phi, f}
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r)
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor = -1/(12 * dr * dtheta)

    row = list()
    col = list()
    data = list()

    for ir in range(N_r):
        for it in range(N_theta):

            br1 = phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]

            br2 = phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]

            br3 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)]

            br4 = phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br5 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]

            br6 = phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br7 = phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br8 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]

            ii = ind_to_tp_ind(ir, it, N_r)

            #f_ir, it
            coef = -br1 + br2 - br3 + br4 - br5 + br6 - br7 + br8
            row.append(ii)
            col.append(neighbour_index(0, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir+1, it
            coef = br1
            row.append(ii)
            col.append(neighbour_index(1, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir-1, it
            coef = -br2
            row.append(ii)
            col.append(neighbour_index(-1, 0, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir,it+1
            coef = br3
            row.append(ii)
            col.append(neighbour_index(0, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it+1
            coef = br5
            row.append(ii)
            col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it+1
            coef = br7
            row.append(ii)
            col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it-1
            coef = -br6
            row.append(ii)
            col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir,it-1
            coef = -br4
            row.append(ii)
            col.append(neighbour_index(0, -1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it-1
            coef = -br8
            row.append(ii)
            col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
            data.append(coef)

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)
    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
    return J


def assemble_awk_bracket_dirichlet(phi, grid_theta, grid_r):
    """
    Assemble a discrete bracket J: f -> {phi, f} based on the Arakawa scheme as
    a sparse matrix with Dirichlet boundary conditions in r-direction

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'

        phi : np.ndarray
            array of length N_theta*N_r; point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*N_r;
            Stencil matrix for discrete Poisson bracket: J * f = {phi, f}
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r)
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor = -1/(12 * dr * dtheta)

    row = list()
    col = list()
    data = list()

    for ir in range(N_r)[1:-1]:
        for it in range(N_theta):
            br1 = phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]

            br2 = phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]

            br3 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)]

            br4 = phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br5 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]

            br6 = phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br7 = phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br8 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]

            ii = ind_to_tp_ind(ir, it, N_r)

            #f_ir, it
            coef = -br1 + br2 - br3 + br4 - br5 + br6 - br7 + br8
            row.append(ii)
            col.append(neighbour_index(0, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir+1, it
            coef = br1
            row.append(ii)
            col.append(neighbour_index(1, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir-1, it
            coef = -br2
            row.append(ii)
            col.append(neighbour_index(-1, 0, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir,it+1
            coef = br3
            row.append(ii)
            col.append(neighbour_index(0, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it+1
            coef = br5
            row.append(ii)
            col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it+1
            coef = br7
            row.append(ii)
            col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it-1
            coef = -br6
            row.append(ii)
            col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir,it-1
            coef = -br4
            row.append(ii)
            col.append(neighbour_index(0, -1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it-1
            coef = -br8
            row.append(ii)
            col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
            data.append(coef)

    # Treatment of the left boundary
    # -Coefficient in the following in order to keep the order of variables
    # in sync with the paper calculations (note j.T = -j)
    ir = 0
    for it in range(N_theta):
        ii = ind_to_tp_ind(ir, it, N_r)
        br1 = phi[neighbour_index(0, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]

        br2 = -phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 0, ir, it, N_r, N_theta)]

        br3 = phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]

        br4 = phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 0, ir, it, N_r, N_theta)]

        br5 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]

        # f_i,0
        coef = br1 + br2 + br3 + br4 + br5
        row.append(ii)
        col.append(neighbour_index(0, 0, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i+1,0
        coef = br1
        row.append(ii)
        col.append(neighbour_index(0, 1, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i-1,0
        coef = br2
        row.append(ii)
        col.append(neighbour_index(0, -1, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i,1
        coef = br3
        row.append(ii)
        col.append(neighbour_index(1, 0, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i+1,1
        coef = br4
        row.append(ii)
        col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i-1,1
        coef = br5
        row.append(ii)
        col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
        data.append(-coef)

    # Treatment of the right boundary
    ir = N_r-1
    for it in range(N_theta):
        ii = ind_to_tp_ind(ir, it, N_r)
        br1 = phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 0, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]

        br2 = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, 0, ir, it, N_r, N_theta)]

        br3 = -phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]

        br4 = -phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]

        br5 = -phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

        # f_i,N-1
        coef = br1 + br2 + br3 + br4 + br5
        row.append(ii)
        col.append(neighbour_index(0, 0, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i+1,N-1
        coef = br1
        row.append(ii)
        col.append(neighbour_index(0, 1, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i-1,N-1
        coef = br2
        row.append(ii)
        col.append(neighbour_index(0, -1, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i,N-2
        coef = br3
        row.append(ii)
        col.append(neighbour_index(-1, 0, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i-1,N-2
        coef = br4
        row.append(ii)
        col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
        data.append(-coef)

        # f_i+1,N-2
        coef = br5
        row.append(ii)
        col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)
    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
    return J


def assemble_awk_bracket_4th_order_periodic(phi, grid_theta, grid_r):
    """
    Assemble the extra terms needed for fourth order Arakawa scheme for the discrete
    bracket J: f -> {phi, f} a sparse matrix with periodic boundary conditions in r-direction

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'

        phi : np.ndarray
            array of length N_theta*N_r; point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*N_r;
            Stencil matrix for discrete Poisson bracket: J * f = {phi, f}
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r)
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor = 1/(24 * dr * dtheta)

    row = list()
    col = list()
    data = list()

    for ir in range(N_r):
        for it in range(N_theta):

            ii = ind_to_tp_ind(ir, it, N_r)

            #f_ir-2, it
            coef = phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(-2, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir, it-2
            coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(0, -2, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir-1, it-1
            coef = -phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir, it+2
            coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(0, 2, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it+1
            coef = phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+2,it
            coef = -phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(2, 0, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it+1
            coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it-1
            coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
            data.append(coef)

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)
    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
    return J


def assemble_awk_bracket_4th_order_dirichlet(phi, grid_theta, grid_r):
    """
    Assemble the extra terms needed for fourth order Arakawa scheme for the discrete
    bracket J: f -> {phi, f} a sparse matrix with Dirichlet boundary conditions in r-direction

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'

        phi : np.ndarray
            array of length N_theta*N_r; point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*N_r;
            Stencil matrix for discrete Poisson bracket: J * f = {phi, f}
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r)
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor = 1/(24 * dr * dtheta)

    row = list()
    col = list()
    data = list()

    for ir in range(N_r)[2:-2]:
        for it in range(N_theta):

            ii = ind_to_tp_ind(ir, it, N_r)

            #f_ir-2, it
            coef = phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(-2, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir, it-2
            coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(0, -2, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir-1, it-1
            coef = -phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir, it+2
            coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(0, 2, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it+1
            coef = phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+2,it
            coef = -phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(2, 0, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it+1
            coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it-1
            coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
            row.append(ii)
            col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
            data.append(coef)

    ir = 0
    for it in range(N_theta):
        ii = ind_to_tp_ind(ir, it, N_r)

        #f_ir, it-2
        coef = -phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, -2, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir, it+2
        coef = phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, 2, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+2,it
        coef = -phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(2, 0, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+1,it+1
        coef = phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+1,it-1
        coef = -phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
        data.append(coef)

    ir = 1
    for it in range(N_theta):
        ii = ind_to_tp_ind(ir, it, N_r)

        #f_ir, it-2
        coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, -2, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir-1, it-1
        coef = -phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir, it+2
        coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, 2, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir-1,it+1
        coef = phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+2,it
        coef = -phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(2, 0, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+1,it+1
        coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+1,it-1
        coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(2, 0, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
        data.append(coef)

    ir = N_r-2
    for it in range(N_theta):
        ii = ind_to_tp_ind(ir, it, N_r)

        #f_ir-2, it
        coef = phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-2, 0, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir, it-2
        coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, -2, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir-1, it-1
        coef = -phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir, it+2
        coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, 2, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir-1,it+1
        coef = phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+1,it+1
        coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(1, 0, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir+1,it-1
        coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(1, 0, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
        data.append(coef)

    ir = N_r - 1
    for it in range(N_theta):
        ii = ind_to_tp_ind(ir, it, N_r)

        #f_ir-2, it
        coef = phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-2, 0, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir, it-2
        coef = -phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, -2, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir-1, it-1
        coef = -phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -2, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
        data.append(coef)

        #f_ir, it+2
        coef = phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(0, 2, ir, it, N_r, N_theta))
        data.append(coef)

        # f_ir-1,it+1
        coef = phi[neighbour_index(-2, 0, ir, it, N_r, N_theta)] \
            + phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 2, ir, it, N_r, N_theta)] \
            - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]
        row.append(ii)
        col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
        data.append(coef)

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)
    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
    return J

################################
# Extrapolation methods for BC #
################################


def assemble_awk_bracket_dirichlet_extrapolation(phi, grid_theta, grid_r):
    """
    Assemble a discrete bracket J: f -> {phi, f} based on the Arakawa scheme as
    a sparse matrix with extrapolation in r-direction

    Parameters
    ----------

        phi : np.ndarray
            array of length N_theta*(N_r+2); point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*(N_r+2);
            Stencil matrix for discrete Poisson bracket: J * f = {phi, f}
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r) + 2
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor = -1/(12 * dr * dtheta)

    row = list()
    col = list()
    data = list()

    # Arakawa in the interior
    # And allow columns to get outside of the domain
    for ir in range(N_r)[1:-1]:
        for it in range(N_theta):
            br1 = phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(1, 1, ir, it, N_r, N_theta)]

            br2 = phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]

            br3 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 1, ir, it, N_r, N_theta)]

            br4 = phi[neighbour_index(1, -1, ir, it, N_r, N_theta)] \
                + phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br5 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, 1, ir, it, N_r, N_theta)]

            br6 = phi[neighbour_index(0, -1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br7 = phi[neighbour_index(0, 1, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(-1, 0, ir, it, N_r, N_theta)]

            br8 = phi[neighbour_index(1, 0, ir, it, N_r, N_theta)] \
                - phi[neighbour_index(0, -1, ir, it, N_r, N_theta)]

            ii = ind_to_tp_ind(ir, it, N_r)

            #f_ir, it
            coef = -br1 + br2 - br3 + br4 - br5 + br6 - br7 + br8
            row.append(ii)
            col.append(neighbour_index(0, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir+1, it
            coef = br1
            row.append(ii)
            col.append(neighbour_index(1, 0, ir, it, N_r, N_theta))
            data.append(coef)

            #f_ir-1, it
            coef = -br2
            row.append(ii)
            col.append(neighbour_index(-1, 0, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir,it+1
            coef = br3
            row.append(ii)
            col.append(neighbour_index(0, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it+1
            coef = br5
            row.append(ii)
            col.append(neighbour_index(1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it+1
            coef = br7
            row.append(ii)
            col.append(neighbour_index(-1, 1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir-1,it-1
            coef = -br6
            row.append(ii)
            col.append(neighbour_index(-1, -1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir,it-1
            coef = -br4
            row.append(ii)
            col.append(neighbour_index(0, -1, ir, it, N_r, N_theta))
            data.append(coef)

            # f_ir+1,it-1
            coef = -br8
            row.append(ii)
            col.append(neighbour_index(1, -1, ir, it, N_r, N_theta))
            data.append(coef)

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)
    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
    return J


def assemble_awk_bracket_4th_order_dirichlet_extrapolation(phi, grid_theta, grid_r):
    """
    Assemble a discrete bracket J: f -> {phi, f} based on the Arakawa scheme as
    a sparse matrix with extrapolation in r-direction

    Parameters
    ----------

        phi : np.ndarray
            array of length N_theta*(N_r+4); point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*(N_r+4);
            Stencil matrix for discrete Poisson bracket: J * f = {phi, f}
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r) + 4
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor1 = 1/(6 * dr * dtheta)
    factor2 = 1/4 * factor1
    data = list()

    row = list()
    col = list()

    # Arakawa in the interior
    # allow for columns to get outside of the domain
    for ir in range(N_r)[2:-2]:
        for it in range(N_theta):
            izm = neighbour_index(0, -1, ir, it, N_r, N_theta)
            izp = neighbour_index(0, 1, ir, it, N_r, N_theta)
            imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
            imz = neighbour_index(-1, 0, ir, it, N_r, N_theta)
            imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
            ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
            ipz = neighbour_index(1, 0, ir, it, N_r, N_theta)
            ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
            i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)
            iz2p = neighbour_index(0, 2, ir, it, N_r, N_theta)
            i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)
            iz2m = neighbour_index(0, -2, ir, it, N_r, N_theta)

            ii = ind_to_tp_ind(ir, it, N_r)

            # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
            coef = -(phi[imm] - phi[imp]) * factor2
            row.append(ii)
            col.append(i2mz)
            data.append(coef)

            # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
            coef = (phi[imm] - phi[imp] + phi[izm] - phi[izp]) * factor1
            row.append(ii)
            col.append(imz)
            data.append(coef)

            # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
            coef = -(phi[imz] * factor1)+(phi[i2mz]-phi[iz2m]) * \
                factor2+phi[izm] * factor1+(phi[imp]-phi[ipm]) * factor2
            row.append(ii)
            col.append(imm)
            data.append(coef)

            # -(F[0,-2] (-phi[imm]+phi[ipm]))* factor2
            coef = -(-phi[imm]+phi[ipm]) * factor2
            row.append(ii)
            col.append(iz2m)
            data.append(coef)

            # +(F[0,-1] (-phi[imm]-phi[imz]+phi[ipm]+phi[ipz]))* factor1
            coef = (-phi[imm]-phi[imz]+phi[ipm]+phi[ipz]) * factor1
            row.append(ii)
            col.append(izm)
            data.append(coef)

            # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
            coef = phi[imz] * factor1-phi[izp] * factor1 - \
                (phi[i2mz]-phi[iz2p]) * factor2-(phi[imm]-phi[ipp]) * factor2
            row.append(ii)
            col.append(imp)
            data.append(coef)

            # -(F[0,2] (phi[imp]-phi[ipp]))* factor2
            coef = -(phi[imp]-phi[ipp]) * factor2
            row.append(ii)
            col.append(iz2p)
            data.append(coef)

            # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
            coef = -(-phi[ipm]+phi[ipp]) * factor2
            row.append(ii)
            col.append(i2pz)
            data.append(coef)

            # +F[0,1] (phi[imz]* factor1+phi[imp]* factor1-phi[ipz]* factor1-phi[ipp]* factor1)
            coef = (phi[imz] * factor1+phi[imp] * factor1 -
                    phi[ipz] * factor1-phi[ipp] * factor1)
            row.append(ii)
            col.append(izp)
            data.append(coef)

            # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
            coef = (-(phi[izm] * factor1)+phi[izp] * factor1 -
                    phi[ipm] * factor1+phi[ipp] * factor1)
            row.append(ii)
            col.append(ipz)
            data.append(coef)

            # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
            coef = (phi[izp] * factor1-(phi[imp]-phi[ipm]) * factor2 -
                    phi[ipz] * factor1-(phi[iz2p]-phi[i2pz]) * factor2)
            row.append(ii)
            col.append(ipp)
            data.append(coef)

            # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
            coef = (-(phi[izm] * factor1)+phi[ipz] * factor1+(phi[imm] -
                    phi[ipp]) * factor2-(-phi[iz2m]+phi[i2pz]) * factor2)
            row.append(ii)
            col.append(ipm)
            data.append(coef)

    ir = 0
    for it in range(N_theta):
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        coef = -(-phi[ipm]+phi[ipp]) * factor2
        row.append(ii)
        col.append(i2pz)
        data.append(coef)

    ir = 1
    for it in range(N_theta):
        izm = neighbour_index(0, -1, ir, it, N_r, N_theta)
        izp = neighbour_index(0, 1, ir, it, N_r, N_theta)
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipz = neighbour_index(1, 0, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)
        iz2p = neighbour_index(0, 2, ir, it, N_r, N_theta)
        iz2m = neighbour_index(0, -2, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        coef = -(-phi[ipm]+phi[ipp]) * factor2
        row.append(ii)
        col.append(i2pz)
        data.append(coef)

        # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
        coef = (-(phi[izm] * factor1)+phi[izp] * factor1 -
                phi[ipm] * factor1+phi[ipp] * factor1)
        row.append(ii)
        col.append(ipz)
        data.append(coef)

        # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
        coef = (phi[izp] * factor1-(phi[imp]-phi[ipm]) * factor2 -
                phi[ipz] * factor1-(phi[iz2p]-phi[i2pz]) * factor2)
        row.append(ii)
        col.append(ipp)
        data.append(coef)

        # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
        coef = (-(phi[izm] * factor1)+phi[ipz] * factor1+(phi[imm] -
                phi[ipp]) * factor2-(-phi[iz2m]+phi[i2pz]) * factor2)
        row.append(ii)
        col.append(ipm)
        data.append(coef)

    ir = N_r-2
    for it in range(N_theta):
        izm = neighbour_index(0, -1, ir, it, N_r, N_theta)
        izp = neighbour_index(0, 1, ir, it, N_r, N_theta)
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imz = neighbour_index(-1, 0, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        iz2p = neighbour_index(0, 2, ir, it, N_r, N_theta)
        i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)
        iz2m = neighbour_index(0, -2, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        coef = -(phi[imm] - phi[imp]) * factor2
        row.append(ii)
        col.append(i2mz)
        data.append(coef)

        # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
        coef = (phi[imm] - phi[imp] + phi[izm] - phi[izp]) * factor1
        row.append(ii)
        col.append(imz)
        data.append(coef)

        # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
        coef = -(phi[imz] * factor1)+(phi[i2mz]-phi[iz2m]) * \
            factor2+phi[izm] * factor1+(phi[imp]-phi[ipm]) * factor2
        row.append(ii)
        col.append(imm)
        data.append(coef)

        # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
        coef = phi[imz] * factor1-phi[izp] * factor1 - \
            (phi[i2mz]-phi[iz2p]) * factor2-(phi[imm]-phi[ipp]) * factor2
        row.append(ii)
        col.append(imp)
        data.append(coef)

    ir = N_r-1
    for it in range(N_theta):
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        coef = -(phi[imm] - phi[imp]) * factor2
        row.append(ii)
        col.append(i2mz)
        data.append(coef)

    data = np.array(data)
    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
    return J

# The following functions implement the 4th order extrapolation with precomputation of indices and just updating the matrix


def assemble_row_columns_akw_bracket_4th_order_extrapolation(grid_theta, grid_r):
    """
    Precompute the used rows and columns for the indices of values in the bracket. Additionally pre-allocate the matrix.

    Parameters
    ----------

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        rowcols : tuple of two np.darray
            rows and columns of the entries of J

        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*(N_r+4);
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r) + 4
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    # does scipy sparse delete zeros?
    row = list()
    col = list()

    # Arakawa in the interior
    # allow for columns to get outside of the domain
    for ir in range(N_r)[2:-2]:
        for it in range(N_theta):
            izm = neighbour_index(0, -1, ir, it, N_r, N_theta)
            izp = neighbour_index(0, 1, ir, it, N_r, N_theta)
            imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
            imz = neighbour_index(-1, 0, ir, it, N_r, N_theta)
            imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
            ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
            ipz = neighbour_index(1, 0, ir, it, N_r, N_theta)
            ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
            i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)
            iz2p = neighbour_index(0, 2, ir, it, N_r, N_theta)
            i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)
            iz2m = neighbour_index(0, -2, ir, it, N_r, N_theta)

            ii = ind_to_tp_ind(ir, it, N_r)

            # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
            row.append(ii)
            col.append(i2mz)

            # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
            row.append(ii)
            col.append(imz)

            # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
            row.append(ii)
            col.append(imm)

            # -(F[0,-2] (-phi[imm]+phi[ipm]))* factor2
            row.append(ii)
            col.append(iz2m)

            # +(F[0,-1] (-phi[imm]-phi[imz]+phi[ipm]+phi[ipz]))* factor1
            row.append(ii)
            col.append(izm)

            # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
            row.append(ii)
            col.append(imp)

            # -(F[0,2] (phi[imp]-phi[ipp]))* factor2
            row.append(ii)
            col.append(iz2p)

            # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
            row.append(ii)
            col.append(i2pz)

            # +F[0,1] (phi[imz]* factor1+phi[imp]* factor1-phi[ipz]* factor1-phi[ipp]* factor1)
            row.append(ii)
            col.append(izp)

            # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
            row.append(ii)
            col.append(ipz)

            # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
            row.append(ii)
            col.append(ipp)

            # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
            row.append(ii)
            col.append(ipm)

    ir = 0
    for it in range(N_theta):
        i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        row.append(ii)
        col.append(i2pz)

    ir = 1
    for it in range(N_theta):
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipz = neighbour_index(1, 0, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        row.append(ii)
        col.append(i2pz)

        # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
        row.append(ii)
        col.append(ipz)

        # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
        row.append(ii)
        col.append(ipp)

        # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
        row.append(ii)
        col.append(ipm)

    ir = N_r-2
    for it in range(N_theta):
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imz = neighbour_index(-1, 0, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        row.append(ii)
        col.append(i2mz)

        # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
        row.append(ii)
        col.append(imz)

        # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
        row.append(ii)
        col.append(imm)

        # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
        row.append(ii)
        col.append(imp)

    ir = N_r-1
    for it in range(N_theta):
        i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        row.append(ii)
        col.append(i2mz)

    data = np.zeros(len(row))
    J = (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()

    return (row, col), J

# Needs to get pyccelized as its used in every step


def assemble_data_4th_order_extrapolation(phi, grid_theta, grid_r):
    """
    Assemble only the values of the discrete bracket J: f -> {phi, f} based on the Arakawa scheme as
    a sparse matrix with extrapolation in r-direction

    Parameters
    ----------

        phi : np.ndarray
            array of length N_theta*(N_r+2); point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    Returns
    -------
        data : np.ndarray
            Contains the entries of J 
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r) + 4
    N_nodes = N_theta*N_r

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor1 = 1/(6 * dr * dtheta)
    factor2 = 1/4 * factor1

    data = list()
    # Arakawa in the interior
    # allow for columns to get outside of the domain
    for ir in range(N_r)[2:-2]:
        for it in range(N_theta):
            izm = neighbour_index(0, -1, ir, it, N_r, N_theta)
            izp = neighbour_index(0, 1, ir, it, N_r, N_theta)
            imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
            imz = neighbour_index(-1, 0, ir, it, N_r, N_theta)
            imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
            ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
            ipz = neighbour_index(1, 0, ir, it, N_r, N_theta)
            ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
            i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)
            iz2p = neighbour_index(0, 2, ir, it, N_r, N_theta)
            i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)
            iz2m = neighbour_index(0, -2, ir, it, N_r, N_theta)

            ii = ind_to_tp_ind(ir, it, N_r)

            # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
            coef = -(phi[imm] - phi[imp]) * factor2
            data.append(coef)

            # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
            coef = (phi[imm] - phi[imp] + phi[izm] - phi[izp]) * factor1
            data.append(coef)

            # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
            coef = -(phi[imz] * factor1)+(phi[i2mz]-phi[iz2m]) * \
                factor2+phi[izm] * factor1+(phi[imp]-phi[ipm]) * factor2
            data.append(coef)

            # -(F[0,-2] (-phi[imm]+phi[ipm]))* factor2
            coef = -(-phi[imm]+phi[ipm]) * factor2
            data.append(coef)

            # +(F[0,-1] (-phi[imm]-phi[imz]+phi[ipm]+phi[ipz]))* factor1
            coef = (-phi[imm]-phi[imz]+phi[ipm]+phi[ipz]) * factor1
            data.append(coef)

            # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
            coef = phi[imz] * factor1-phi[izp] * factor1 - \
                (phi[i2mz]-phi[iz2p]) * factor2-(phi[imm]-phi[ipp]) * factor2
            data.append(coef)

            # -(F[0,2] (phi[imp]-phi[ipp]))* factor2
            coef = -(phi[imp]-phi[ipp]) * factor2
            data.append(coef)

            # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
            coef = -(-phi[ipm]+phi[ipp]) * factor2
            data.append(coef)

            # +F[0,1] (phi[imz]* factor1+phi[imp]* factor1-phi[ipz]* factor1-phi[ipp]* factor1)
            coef = (phi[imz] * factor1+phi[imp] * factor1 -
                    phi[ipz] * factor1-phi[ipp] * factor1)
            data.append(coef)

            # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
            coef = (-(phi[izm] * factor1)+phi[izp] * factor1 -
                    phi[ipm] * factor1+phi[ipp] * factor1)
            data.append(coef)

            # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
            coef = (phi[izp] * factor1-(phi[imp]-phi[ipm]) * factor2 -
                    phi[ipz] * factor1-(phi[iz2p]-phi[i2pz]) * factor2)
            data.append(coef)

            # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
            coef = (-(phi[izm] * factor1)+phi[ipz] * factor1+(phi[imm] -
                    phi[ipp]) * factor2-(-phi[iz2m]+phi[i2pz]) * factor2)
            data.append(coef)

    ir = 0
    for it in range(N_theta):
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        coef = -(-phi[ipm]+phi[ipp]) * factor2
        data.append(coef)

    ir = 1
    for it in range(N_theta):
        izm = neighbour_index(0, -1, ir, it, N_r, N_theta)
        izp = neighbour_index(0, 1, ir, it, N_r, N_theta)
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipz = neighbour_index(1, 0, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)
        iz2p = neighbour_index(0, 2, ir, it, N_r, N_theta)
        iz2m = neighbour_index(0, -2, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        coef = -(-phi[ipm]+phi[ipp]) * factor2
        data.append(coef)

        # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
        coef = (-(phi[izm] * factor1)+phi[izp] * factor1 -
                phi[ipm] * factor1+phi[ipp] * factor1)
        data.append(coef)

        # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
        coef = (phi[izp] * factor1-(phi[imp]-phi[ipm]) * factor2 -
                phi[ipz] * factor1-(phi[iz2p]-phi[i2pz]) * factor2)
        data.append(coef)

        # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
        coef = (-(phi[izm] * factor1)+phi[ipz] * factor1+(phi[imm] -
                phi[ipp]) * factor2-(-phi[iz2m]+phi[i2pz]) * factor2)
        data.append(coef)

    ir = N_r-2
    for it in range(N_theta):
        izm = neighbour_index(0, -1, ir, it, N_r, N_theta)
        izp = neighbour_index(0, 1, ir, it, N_r, N_theta)
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imz = neighbour_index(-1, 0, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        iz2p = neighbour_index(0, 2, ir, it, N_r, N_theta)
        i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)
        iz2m = neighbour_index(0, -2, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        coef = -(phi[imm] - phi[imp]) * factor2
        data.append(coef)

        # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
        coef = (phi[imm] - phi[imp] + phi[izm] - phi[izp]) * factor1
        data.append(coef)

        # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
        coef = -(phi[imz] * factor1)+(phi[i2mz]-phi[iz2m]) * \
            factor2+phi[izm] * factor1+(phi[imp]-phi[ipm]) * factor2
        data.append(coef)

        # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
        coef = phi[imz] * factor1-phi[izp] * factor1 - \
            (phi[i2mz]-phi[iz2p]) * factor2-(phi[imm]-phi[ipp]) * factor2
        data.append(coef)

    ir = N_r-1
    for it in range(N_theta):
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)

        ii = ind_to_tp_ind(ir, it, N_r)

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        coef = -(phi[imm] - phi[imp]) * factor2
        data.append(coef)

    data = np.array(data)

    return data


def update_bracket_4th_order_dirichlet_extrapolation(J, rowcols, phi, grid_theta, grid_r):
    """
    Compute the entries of J, use the precomputed row and column information and update J in place. 

    Parameters
    ----------
        J : scipy.sparse.coo_matrix
            sparse matrix of shape [nPoints, nPoints] where nPoints = N_theta*(N_r+4);

        rowcols : tuple of two np.darray
            rows and columns of the entries of J

        phi : np.ndarray
            array of length N_theta*(N_r+2); point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

    """

    data = assemble_data_4th_order_extrapolation(phi, grid_theta, grid_r)
    J[rowcols] = data
