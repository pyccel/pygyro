import numpy as np
import scipy.sparse as sparse


def ind_to_tp_ind(ir, it, N_r):
    """
    Convert one-dimensional indices to tensorproduct index

    Parameters
    ----------
        ir : int
            index in r-direction

        it : int
            index in theta-direction

        N_r : int
            number of nodes in r-direction

    Returns
    -------
        tp_ind : int
            tensorproduct index
    """
    tp_ind = ir + it * N_r

    return tp_ind


def neighbour_index(posr, post, ir, it, N_r, N_theta):
    """
    Calculate the tensorproduct index given two one-dimensional indices
    and a positional offset with periodic continuation for both variables

    Parameters
    ----------
        posr : int
            position of the neighbour in r-direction relative to index ir

        post : int
            position of the neighbour in theta-direction relative to index it

        ir : int
            index in r-direction of the node of which we want to compute the neighbour index

        it : int
            index in theta-direction of the node of which we want to compute the neighbour index

        N_r : int
            number of nodes in r-direction

        N_theta : int
            number of nodes in theta-direction

    Returns
    -------
        neigh_ind : int
            index of the neighbour
    """
    neigh_ind = (ir + posr) % N_r + ((it + post) % N_theta) * N_r

    return neigh_ind


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

    else:
        raise NotImplementedError(
            f'{bc} is an unknown option for boundary conditions')

    return res


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
