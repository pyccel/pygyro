import numpy as np
import scipy.sparse as sparse


def neighbor_index(pos0, pos1, i0, i1, nPoints_theta, nPoints_r):
    """
    TODO

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """
    index = (i0 + pos0) % nPoints_theta + \
        ((i1 + pos1) % nPoints_r) * nPoints_theta
    return index


def assemble_bracket(scheme, bc, phi_hh, nPoints_theta, nPoints_r, r_grid):
    """
    TODO

    Parameters
    ----------
        scheme : str
            which scheme for the stencil matrix is to be used ('pp', 'px', 'xp', 'akw')

        bc : str
            determines the boundary condition for r

        TODO

    Returns
    -------
        TODO
    """
    
    if bc == 'periodic':
        if scheme == 'akw':
            Jpp = assemble_Jpp(phi_hh, nPoints_theta, nPoints_r, r_grid)
            Jpx = -assemble_Jpx(phi_hh, nPoints_theta, nPoints_r, r_grid)
            Jxp = -assemble_Jxp(phi_hh, nPoints_theta, nPoints_r, r_grid)
            J_phi = 1/3 * (Jpp + Jpx + Jxp)
        elif scheme == 'pp':
            J_phi = assemble_Jpp(phi_hh, nPoints_theta, nPoints_r, r_grid)
        elif scheme == 'px':
            J_phi = -assemble_Jpx(phi_hh, nPoints_theta, nPoints_r, r_grid)
        elif scheme == 'xp':
            J_phi = -assemble_Jxp(phi_hh, nPoints_theta, nPoints_r, r_grid)
        else:
            raise NotImplementedError(
                f'Unknown option for the scheme : {scheme}')

    elif bc == 'dirichlet':
        if scheme == 'akw':
            Jpp = assemble_Jpp_dirichlet(
                phi_hh, nPoints_theta, nPoints_r, r_grid)
            Jpx = -assemble_Jpx_dirichlet(phi_hh,
                                          nPoints_theta, nPoints_r, r_grid)
            Jxp = -assemble_Jxp_dirichlet(phi_hh,
                                          nPoints_theta, nPoints_r, r_grid)
            J_phi = 1/3 * (Jpp + Jpx + Jxp)
        elif scheme == 'pp':
            J_phi = assemble_Jpp_dirichlet(
                phi_hh, nPoints_theta, nPoints_r, r_grid)
        elif scheme == 'px':
            J_phi = - \
                assemble_Jpx_dirichlet(
                    phi_hh, nPoints_theta, nPoints_r, r_grid)
        elif scheme == 'xp':
            J_phi = - \
                assemble_Jxp_dirichlet(
                    phi_hh, nPoints_theta, nPoints_r, r_grid)
        else:
            raise NotImplementedError(
                f'Unknown option for the scheme : {scheme}')

    else:
        raise NotImplementedError(
            f'Unknown option for boundary conditions : {bc}')

    return J_phi


def assemble_Jpp(phi_hh, nPoints_theta, nPoints_r, r_grid):
    """
    assemble J_++(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """

    N_nodes = nPoints_theta * nPoints_r

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % nPoints_theta
        i1 = ii // nPoints_theta

        # .. terms (phi_0+ - phi_0-)/4h^2 * (f_+0 - f_-0)
        coef = phi_hh[neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r)]
        coef -= phi_hh[neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

        # .. terms -(phi_+0 - phi_-0)/4h^2 * (f_0+ - f_0-)
        coef = - \
            phi_hh[neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r)]
        coef += phi_hh[neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    res = (sparse.coo_matrix((data, (row, col)),
           shape=(N_nodes, N_nodes))).tocsr()

    return res


def assemble_Jpx(phi_hh, nPoints_theta, nPoints_r, r_grid):
    """
    assemble J_+x(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """

    N_nodes = nPoints_theta*nPoints_r

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % nPoints_theta
        i1 = ii // nPoints_theta

        # .. terms phi_++/4h^2 * (f_0+ - f_+0)
        coef = phi_hh[neighbor_index(+1, +1, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

        # .. terms -phi_--/4h^2 * (f_-0 - f_0-)
        coef = - \
            phi_hh[neighbor_index(-1, -1, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

        # .. terms -phi_-+/4h^2 * (f_0+ - f_-0)
        coef = - \
            phi_hh[neighbor_index(-1, +1, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

        # .. terms phi_+-/4h^2 * (f_+0 - f_0-)
        coef = phi_hh[neighbor_index(+1, -1, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    res = (sparse.coo_matrix((data, (row, col)),
           shape=(N_nodes, N_nodes))).tocsr()

    return res


def assemble_Jxp(phi_hh, nPoints_theta, nPoints_r, r_grid):
    """
    assemble J_x+(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """

    N_nodes = nPoints_theta*nPoints_r

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % nPoints_theta
        i1 = ii // nPoints_theta

        # .. terms phi_+0/4h^2 * (f_++ - f_+-)
        coef = phi_hh[neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

        # .. terms -phi_-0/4h^2 * (f_-+ - f_--)
        coef = - \
            phi_hh[neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

        # .. terms -phi_0+/4h^2 * (f_++ - f_-+)
        coef = -phi_hh[neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

        # .. terms phi_0-/4h^2 * (f_+- - f_--)
        coef = phi_hh[neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, nPoints_theta, nPoints_r))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, nPoints_theta, nPoints_r))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    res = (sparse.coo_matrix((data, (row, col)),
           shape=(N_nodes, N_nodes))).tocsr()

    return res


def assemble_Jpp_dirichlet(phi_hh, nPoints_theta, nPoints_r, r_grid):
    """
    assemble J_++(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """

    N_nodes = nPoints_theta * nPoints_r

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % nPoints_theta
        i1 = ii // nPoints_theta

        if i1 != 0 and i1 != nPoints_r-1:
            # .. terms (phi_0+ - phi_0-)/4h^2 * (f_+0 - f_-0)
            coef = phi_hh[neighbor_index(
                0, +1, i0, i1, nPoints_theta, nPoints_r)]
            coef -= phi_hh[neighbor_index(0, -1,
                                          i0, i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r))
            data.append(-coef)

            # .. terms -(phi_+0 - phi_-0)/4h^2 * (f_0+ - f_0-)
            coef = - \
                phi_hh[neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r)]
            coef += phi_hh[neighbor_index(-1, 0,
                                          i0, i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    res = (sparse.coo_matrix((data, (row, col)),
           shape=(N_nodes, N_nodes))).tocsr()

    return res


def assemble_Jpx_dirichlet(phi_hh, nPoints_theta, nPoints_r, r_grid):
    """
    assemble J_+x(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """

    N_nodes = nPoints_theta*nPoints_r

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % nPoints_theta
        i1 = ii//nPoints_theta

        if i1 != 0 and i1 != nPoints_r-1:
            # .. terms phi_++/4h^2 * (f_0+ - f_+0)
            coef = phi_hh[neighbor_index(+1, +1,
                                         i0, i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r))
            data.append(-coef)

            # .. terms -phi_--/4h^2 * (f_-0 - f_0-)
            coef = - \
                phi_hh[neighbor_index(-1, -1, i0, i1,
                                      nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r))
            data.append(-coef)

            # .. terms -phi_-+/4h^2 * (f_0+ - f_-0)
            coef = - \
                phi_hh[neighbor_index(-1, +1, i0, i1,
                                      nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r))
            data.append(-coef)

            # .. terms phi_+-/4h^2 * (f_+0 - f_0-)
            coef = phi_hh[neighbor_index(+1, -1,
                                         i0, i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, 0, i0, i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, nPoints_theta, nPoints_r))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jxp_dirichlet(phi_hh, nPoints_theta, nPoints_r, r_grid):
    """
    assemble J_x+(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2

    Parameters
    ----------
        TODO

    Returns
    -------
        TODO
    """

    N_nodes = nPoints_theta*nPoints_r

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % nPoints_theta
        i1 = ii//nPoints_theta

        if i1 != 0 and i1 != nPoints_r-1:
            # .. terms phi_+0/4h^2 * (f_++ - f_+-)
            coef = phi_hh[neighbor_index(+1, 0, i0,
                                         i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, +1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, -1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(-coef)

            # .. terms -phi_-0/4h^2 * (f_-+ - f_--)
            coef = - \
                phi_hh[neighbor_index(-1, 0, i0, i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(-1, +1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, -1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(-coef)

            # .. terms -phi_0+/4h^2 * (f_++ - f_-+)
            coef = - \
                phi_hh[neighbor_index(0, +1, i0, i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, +1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, +1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(-coef)

            # .. terms phi_0-/4h^2 * (f_+- - f_--)
            coef = phi_hh[neighbor_index(
                0, -1, i0, i1, nPoints_theta, nPoints_r)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, -1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, -1, i0,
                       i1, nPoints_theta, nPoints_r))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
