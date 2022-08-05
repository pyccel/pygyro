import numpy as np
import scipy.sparse as sparse

def ind_to_tp_ind(i0, i1, N_r):
    """
    Convert one-dimensional indices to tensorproduct index
    """
    return i0 + i1*N_r

def neighbor_index(pos0, pos1, i0, i1, N_theta, N_r):
    """
    Calculate the tensorproduct index 
    given two one-dimensional indices and a positional offset
    with periodic continuation
    """
    return (i0+pos0)%N_r + (i1+pos1)%N_theta*N_r

def assemble_bracket_arakawa(bc, phi, grid_theta, grid_r):
    """
    Assemble the Arakawa bracket J: f -> {phi, f} as a sparse matrix

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'
        
        phi : np.array of length N_theta*N_r
            point values of the potential on the grid

        grid_theta: np.array of length N_theta
            grid of theta

        grid_r : np.array of length N_r
            grid of r

    Returns
    -------
        J : sparse array
    """

    if bc == 'periodic':
        return assemble_awk_bracket_periodic(phi, grid_theta, grid_r)
    elif bc == 'dirichlet':
        return assemble_awk_bracket_dirichlet(phi, grid_theta, grid_r)

def assemble_awk_bracket_periodic(phi, grid_theta, grid_r):
    """
    Assemble the periodic Arakawa bracket J: f -> {phi, f} as a sparse matrix

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'
        
        phi : np.array of length N_theta*N_r
            point values of the potential on the grid

        grid_theta: np.array of length N_theta
            grid of theta

        grid_r : np.array of length N_r
            grid of r

    Returns
    -------
        J : sparse array
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

    for i0 in range(N_r):
        for i1 in range(N_theta):

            br1 = phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(1, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(1, 1, i0, i1, N_theta, N_r)]

            br2 = phi[neighbor_index(-1, -1, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)]

            br3 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(1, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 1, i0, i1, N_theta, N_r)]

            br4 = phi[neighbor_index(1, -1, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)]

            br5 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)]

            br6 = phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] 

            br7 = phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] 

            br8 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] 

            ii = ind_to_tp_ind(i0, i1, N_r)

            #f_i0, i1
            coef = -br1 + br2 -br3 +br4 -br5 +br6 -br7 +br8
            row.append(ii)
            col.append(neighbor_index(0, 0, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0+1, i1
            coef = br1
            row.append(ii)
            col.append(neighbor_index(1, 0, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0-1, i1
            coef = -br2
            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0,i1+1
            coef = br3
            row.append(ii)
            col.append(neighbor_index(0, 1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0+1,i1+1
            coef = br5
            row.append(ii)
            col.append(neighbor_index(1, 1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0-1,i1+1
            coef = br7
            row.append(ii)
            col.append(neighbor_index(-1, 1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0-1,i1-1
            coef = -br6
            row.append(ii)
            col.append(neighbor_index(-1, -1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0,i1-1
            coef = -br4
            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0+1,i1-1
            coef = -br8
            row.append(ii)
            col.append(neighbor_index(1, -1, i0, i1, N_theta, N_r))
            data.append(coef)

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()

def assemble_awk_bracket_dirichlet(phi, grid_theta, grid_r):
    """
    Assemble the periodic Arakawa bracket J: f -> {phi, f} as a sparse matrix

    Parameters
    ----------
        bc : str
            'periodic' or 'dirichlet'
        
        phi : np.array of length N_theta*N_r
            point values of the potential on the grid

        grid_theta: np.array of length N_theta
            grid of theta

        grid_r : np.array of length N_r
            grid of r

    Returns
    -------
        J : sparse array
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

    for i0 in range(N_r)[1:-1]:
        for i1 in range(N_theta):
            br1 = phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(1, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(1, 1, i0, i1, N_theta, N_r)]

            br2 = phi[neighbor_index(-1, -1, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)]

            br3 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(1, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 1, i0, i1, N_theta, N_r)]

            br4 = phi[neighbor_index(1, -1, i0, i1, N_theta, N_r)] \
                + phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)]

            br5 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)]

            br6 = phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] 

            br7 = phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] 

            br8 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
                - phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] 

            ii = ind_to_tp_ind(i0, i1, N_r)

            #f_i0, i1
            coef = -br1 + br2 -br3 +br4 -br5 +br6 -br7 +br8
            row.append(ii)
            col.append(neighbor_index(0, 0, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0+1, i1
            coef = br1
            row.append(ii)
            col.append(neighbor_index(1, 0, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0-1, i1
            coef = -br2
            row.append(ii)
            col.append(neighbor_index(-1, 0, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0,i1+1
            coef = br3
            row.append(ii)
            col.append(neighbor_index(0, 1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0+1,i1+1
            coef = br5
            row.append(ii)
            col.append(neighbor_index(1, 1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0-1,i1+1
            coef = br7
            row.append(ii)
            col.append(neighbor_index(-1, 1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0-1,i1-1
            coef = -br6
            row.append(ii)
            col.append(neighbor_index(-1, -1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0,i1-1
            coef = -br4
            row.append(ii)
            col.append(neighbor_index(0, -1, i0, i1, N_theta, N_r))
            data.append(coef)

            #f_i0+1,i1-1
            coef = -br8
            row.append(ii)
            col.append(neighbor_index(1, -1, i0, i1, N_theta, N_r))
            data.append(coef)

    # Treatment of the left boundary
    i1 = 0
    for i1 in range(N_theta):
        ii = ind_to_tp_ind(i0, i1, N_r)
        br1 = phi[neighbor_index(0, 0, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(1, 1, i0, i1, N_theta, N_r)]

        br2 = -phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(0, 0, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(-1, 1, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)]

        br3 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(1, 1, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(-1, 1, i0, i1, N_theta, N_r)]

        br4 = phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)]

        br5 = phi[neighbor_index(0, 1, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] 
        
        #f_i,0
        coef = br1 +br2 +br3 +br4 +br5
        row.append(ii)
        col.append(neighbor_index(0, 0, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i+1,0
        coef = br1
        row.append(ii)
        col.append(neighbor_index(1, 0, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i-1,0
        coef = br2
        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i,1
        coef = br3
        row.append(ii)
        col.append(neighbor_index(0, 1, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i+1,1
        coef = br4
        row.append(ii)
        col.append(neighbor_index(1, 1, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i-1,1
        coef = br5
        row.append(ii)
        col.append(neighbor_index(-1, 1, i0, i1, N_theta, N_r))
        data.append(coef)

    # Treatment of the right boundary
    i1 = N_r-1
    for i1 in range(N_theta):
        ii = ind_to_tp_ind(i0, i1, N_r)
        br1 = phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(1, -1, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(0, 0, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)]

        br2 = -phi[neighbor_index(-1, -1, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(0, 0, i0, i1, N_theta, N_r)]

        br3 = -phi[neighbor_index(1, -1, i0, i1, N_theta, N_r)] \
            - phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(-1, -1, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)]

        br4 = -phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(-1, 0, i0, i1, N_theta, N_r)]

        br5 = -phi[neighbor_index(1, 0, i0, i1, N_theta, N_r)] \
            + phi[neighbor_index(0, -1, i0, i1, N_theta, N_r)] 

        #f_i,N-1
        coef = br1 + br2 + br3 + br4 + br5
        row.append(ii)
        col.append(neighbor_index(0, 0, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i+1,N-1
        coef = br1
        row.append(ii)
        col.append(neighbor_index(1, 0, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i-1,N-1
        coef = br2
        row.append(ii)
        col.append(neighbor_index(-1, 0, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i,N-2
        coef = br3
        row.append(ii)
        col.append(neighbor_index(0, -1, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i-1,N-2
        coef = br4
        row.append(ii)
        col.append(neighbor_index(-1, -1, i0, i1, N_theta, N_r))
        data.append(coef)

        #f_i+1,N-2
        coef = br5
        row.append(ii)
        col.append(neighbor_index(1, -1, i0, i1, N_theta, N_r))
        data.append(coef)

    row = np.array(row)
    col = np.array(col)
    data = factor * np.array(data)
    return (sparse.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))).tocsr()