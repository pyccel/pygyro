import numpy as np
import scipy.sparse as sparse


def neighbor_index(pos0, pos1, i0, i1, N0_nodes0, N0_nodes1):
    return (i0+pos0) % N0_nodes0 + ((i1+pos1) % N0_nodes1)*N0_nodes0


def assemble_bracket(scheme, bc, phi_hh, N0_nodes0, N0_nodes1, r_grid):
    if bc == 'periodic':
        if scheme == 'akw4th':
            Jpp1 = assemble_Jpp1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jpx1 = -assemble_Jpx1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxp1 = -assemble_Jxp1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxx2 = assemble_Jxx2(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxp2 = -assemble_Jpx2(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jpx2 = -assemble_Jxp2(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            J1 = 1/3 * (Jpp1 + Jpx1 + Jxp1)
            J2 = 1/3 * (Jxx2 + Jxp2 + Jpx2)
            return 2 * J1 - J2
        elif scheme == 'akw2nd':
            Jpp1 = assemble_Jpp1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jpx1 = -assemble_Jpx1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxp1 = -assemble_Jxp1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            return 1/3 * (Jpp1 + Jpx1 + Jxp1)
        elif scheme == 'pp':
            return assemble_Jpp1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        elif scheme == 'px':
            return -assemble_Jpx1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        elif scheme == 'xp':
            return -assemble_Jxp1(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        elif scheme == 'xx2':
            return assemble_Jxx2(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        elif scheme == 'xp2':
            return -assemble_Jxp2(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        elif scheme == 'px2':
            return -assemble_Jpx2(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        else:
            print("scheme not found")

    elif bc == 'dirichlet':
        if scheme == 'akw2nd':
            Jpp1 = assemble_Jpp1_dirichlet(
                phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jpx1 = - \
                assemble_Jpx1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxp1 = - \
                assemble_Jxp1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            return 1/3 * (Jpp1 + Jpx1 + Jxp1)
        elif scheme == 'akw4th':
            Jpp1 = assemble_Jpp1_dirichlet(
                phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jpx1 = - \
                assemble_Jpx1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxp1 = - \
                assemble_Jxp1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxx2 = assemble_Jxx2_dirichlet(
                phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jxp2 = - \
                assemble_Jpx2_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            Jpx2 = - \
                assemble_Jxp2_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
            J1 = 1/3 * (Jpp1 + Jpx1 + Jxp1)
            J2 = 1/3 * (Jxx2 + Jxp2 + Jpx2)
            return 2 * J1 - J2
        elif scheme == 'pp':
            return assemble_Jpp1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        elif scheme == 'px':
            return -assemble_Jpx1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        elif scheme == 'xp':
            return -assemble_Jxp1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid)
        else:
            print("scheme not found")


def assemble_Jpp1(phi_hh, N0_nodes0, N0_nodes1, r_grid):
    """
    assemble J_++(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2
    """

    N_nodes = N0_nodes0 * N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii // N0_nodes0

        # .. terms (phi_0+ - phi_0-)/4h^2 * (f_+0 - f_-0)
        coef = phi_hh[neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef -= phi_hh[neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -(phi_+0 - phi_-0)/4h^2 * (f_0+ - f_0-)
        coef = -phi_hh[neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1)]
        coef += phi_hh[neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jpx1(phi_hh, N0_nodes0, N0_nodes1, r_grid):
    """
    assemble J_+x(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2
    """

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        # .. terms phi_++/4h^2 * (f_0+ - f_+0)
        coef = phi_hh[neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_--/4h^2 * (f_-0 - f_0-)
        coef = -phi_hh[neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_-+/4h^2 * (f_0+ - f_-0)
        coef = -phi_hh[neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms phi_+-/4h^2 * (f_+0 - f_0-)
        coef = phi_hh[neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jxp1(phi_hh, N0_nodes0, N0_nodes1, r_grid):
    """
    assemble J_x+(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2
    """

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        # .. terms phi_+0/4h^2 * (f_++ - f_+-)
        coef = phi_hh[neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_-0/4h^2 * (f_-+ - f_--)
        coef = -phi_hh[neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_0+/4h^2 * (f_++ - f_-+)
        coef = -phi_hh[neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms phi_0-/4h^2 * (f_+- - f_--)
        coef = phi_hh[neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/r_grid[i1]

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()

#############################################################################################


def assemble_Jxx2(phi_hh, N0_nodes0, N0_nodes1, r_grid):
    N_nodes = N0_nodes0 * N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        # .. terms (phi_-+ - phi_+-)/8h^2 * (f_++ - f_--)
        coef = phi_hh[neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef -= phi_hh[neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -(phi_++ - phi_--)/8h^2 * (f_-+ - f_+-)
        coef = -phi_hh[neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef += phi_hh[neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jpx2(phi_hh, N0_nodes0, N0_nodes1, r_grid):

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        # .. terms phi_+,+/8h^2 * (f_0,++ - f_++,0)
        coef = phi_hh[neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(0, +2, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+2, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_-,-/8h^2 * (f_--,0 - f_0,--)
        coef = -phi_hh[neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(-2, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -2, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_-,+/8h^2 * (f_--,0 - f_0,++)
        coef = phi_hh[neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(-2, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, +2, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms phi_+,-/8h^2 * (f_++,0 - f_0,--)
        coef = phi_hh[neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(+2, 0, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(0, -2, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jxp2(phi_hh, N0_nodes0, N0_nodes1, r_grid):

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        # .. terms phi_0,++/8h^2 * (f_-+ - f_++)
        coef = phi_hh[neighbor_index(0, +2, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_++,0/8h^2 * (f_+- - f_++)
        coef = -phi_hh[neighbor_index(+2, 0, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms -phi_0,--/8h^2 * (f_-- - f_+-)
        coef = -phi_hh[neighbor_index(0, -2, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

        # .. terms phi_--,0/8h^2 * (f_-- - f_-+)
        coef = phi_hh[neighbor_index(-2, 0, i0, i1, N0_nodes0, N0_nodes1)]
        coef *= 1/(2*r_grid[i1])

        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(coef)

        row.append(ii)
        col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
        data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()

######################################################################################################


def assemble_Jpp1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid):
    """
    assemble J_++(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2
    """

    N_nodes = N0_nodes0 * N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii // N0_nodes0

        if i1 != 0 and i1 != N0_nodes1-1:
            # .. terms (phi_0+ - phi_0-)/4h^2 * (f_+0 - f_-0)
            coef = phi_hh[neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef -= phi_hh[neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -(phi_+0 - phi_-0)/4h^2 * (f_0+ - f_0-)
            coef = -phi_hh[neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1)]
            coef += phi_hh[neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jpx1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid):
    """
    assemble J_+x(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2
    """

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        if i1 != 0 and i1 != N0_nodes1-1:
            # .. terms phi_++/4h^2 * (f_0+ - f_+0)
            coef = phi_hh[neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_--/4h^2 * (f_-0 - f_0-)
            coef = - \
                phi_hh[neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_-+/4h^2 * (f_0+ - f_-0)
            coef = - \
                phi_hh[neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms phi_+-/4h^2 * (f_+0 - f_0-)
            coef = phi_hh[neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jxp1_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid):
    """
    assemble J_x+(phi, . ) as sparse matrix 

    phi_hh: phi/(2h)**2
    """

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        if i1 != 0 and i1 != N0_nodes1-1:
            # .. terms phi_+0/4h^2 * (f_++ - f_+-)
            coef = phi_hh[neighbor_index(+1, 0, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_-0/4h^2 * (f_-+ - f_--)
            coef = -phi_hh[neighbor_index(-1, 0, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_0+/4h^2 * (f_++ - f_-+)
            coef = -phi_hh[neighbor_index(0, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms phi_0-/4h^2 * (f_+- - f_--)
            coef = phi_hh[neighbor_index(0, -1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/r_grid[i1]

            row.append(ii)
            col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()

#########################################################################################

# treatment of exterior points???


def assemble_Jxx2_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid):

    N_nodes = N0_nodes0 * N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii // N0_nodes0

        if i1 != 0 and i1 != N0_nodes1-1:
            # .. terms (phi_-+ - phi_+-)/8h^2 * (f_++ - f_--)
            coef = phi_hh[neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef -= phi_hh[neighbor_index(+1, -1,
                                          i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -(phi_++ - phi_--)/8h^2 * (f_-+ - f_+-)
            coef = - \
                phi_hh[neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef += phi_hh[neighbor_index(-1, -1,
                                          i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jpx2_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid):

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        if i1 != 0 and i1 != N0_nodes1-1:
            # .. terms phi_+,+/8h^2 * (f_0,++ - f_++,0)
            coef = phi_hh[neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(0, +2, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+2, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_-,-/8h^2 * (f_--,0 - f_0,--)
            coef = - \
                phi_hh[neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(-2, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -2, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_-,+/8h^2 * (f_--,0 - f_0,++)
            coef = phi_hh[neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(-2, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, +2, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms phi_+,-/8h^2 * (f_++,0 - f_0,--)
            coef = phi_hh[neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(+2, 0, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(0, -2, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()


def assemble_Jxp2_dirichlet(phi_hh, N0_nodes0, N0_nodes1, r_grid):

    N_nodes = N0_nodes0*N0_nodes1

    row = list()
    col = list()
    data = list()

    for ii in range(N_nodes):

        i0 = ii % N0_nodes0
        i1 = ii//N0_nodes0

        if i1 != 0 and i1 != N0_nodes1-1:

            # .. terms phi_0,++/8h^2 * (f_-+ - f_++)
            coef = phi_hh[neighbor_index(0, +2, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_++,0/8h^2 * (f_+- - f_++)
            coef = -phi_hh[neighbor_index(+2, 0, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms -phi_0,--/8h^2 * (f_-- - f_+-)
            coef = -phi_hh[neighbor_index(0, -2, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(+1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

            # .. terms phi_--,0/8h^2 * (f_-- - f_-+)
            coef = phi_hh[neighbor_index(-2, 0, i0, i1, N0_nodes0, N0_nodes1)]
            coef *= 1/(2*r_grid[i1])

            row.append(ii)
            col.append(neighbor_index(-1, -1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(coef)

            row.append(ii)
            col.append(neighbor_index(-1, +1, i0, i1, N0_nodes0, N0_nodes1))
            data.append(-coef)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()
