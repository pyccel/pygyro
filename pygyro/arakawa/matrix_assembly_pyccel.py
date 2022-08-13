from pyccel.decorators import pure

from .utilities_pyccel import ind_to_tp_ind, neighbour_index


@pure
def row_col_data_creator(N_theta: int, N_r: int,
                         phi: 'float[:]',
                         row: 'int[:]', col: 'int[:]', data: 'float[:]'):
    """
    Fill the row, column, and data arrays which are needed for the
    creation of scipy sparse matrices in the assembly of the discrete
    Arakawa bracket.

    Parameters:
    -----------
        N_theta : int
            Number of points in angular direction

        N_r : int
            Number of points in radial direction

        phi : array[float]
            Array of length (N_theta*N_r); flattened array with the values for phi

        row : array[int]
            array of length ((N_r-2)*N_theta*9 + N_theta*(6+6)), will be filled with
            the row indices of the bracket matrix

        col : array[int]
            array of length ((N_r-2)*N_theta*9 + N_theta*(6+6)), will be filled with
            the column indices of the bracket matrix

        data : array[int]
            array of length ((N_r-2)*N_theta*9 + N_theta*(6+6)), will be filled with
            the matrix entries of the bracket matrix
    """
    k = 0

    for ir in range(1, N_r - 1):
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
            row[k] = ii
            col[k] = neighbour_index(0, 0, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            #f_ir+1, it
            coef = br1
            row[k] = ii
            col[k] = neighbour_index(1, 0, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            #f_ir-1, it
            coef = -br2
            row[k] = ii
            col[k] = neighbour_index(-1, 0, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            # f_ir,it+1
            coef = br3
            row[k] = ii
            col[k] = neighbour_index(0, 1, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            # f_ir+1,it+1
            coef = br5
            row[k] = ii
            col[k] = neighbour_index(1, 1, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            # f_ir-1,it+1
            coef = br7
            row[k] = ii
            col[k] = neighbour_index(-1, 1, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            # f_ir-1,it-1
            coef = -br6
            row[k] = ii
            col[k] = neighbour_index(-1, -1, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            # f_ir,it-1
            coef = -br4
            row[k] = ii
            col[k] = neighbour_index(0, -1, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

            # f_ir+1,it-1
            coef = -br8
            row[k] = ii
            col[k] = neighbour_index(1, -1, ir, it, N_r, N_theta)
            data[k] = coef
            k += 1

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
        row[k]  = ii
        col[k]  = neighbour_index(0, 0, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i+1,0
        coef = br1
        row[k]  = ii
        col[k]  = neighbour_index(0, 1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i-1,0
        coef = br2
        row[k]  = ii
        col[k]  = neighbour_index(0, -1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i,1
        coef = br3
        row[k]  = ii
        col[k]  = neighbour_index(1, 0, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i+1,1
        coef = br4
        row[k]  = ii
        col[k]  = neighbour_index(1, 1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i-1,1
        coef = br5
        row[k]  = ii
        col[k]  = neighbour_index(1, -1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

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
        row[k]  = ii
        col[k]  = neighbour_index(0, 0, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i+1,N-1
        coef = br1
        row[k]  = ii
        col[k]  = neighbour_index(0, 1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i-1,N-1
        coef = br2
        row[k]  = ii
        col[k]  = neighbour_index(0, -1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i,N-2
        coef = br3
        row[k]  = ii
        col[k]  = neighbour_index(-1, 0, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i-1,N-2
        coef = br4
        row[k]  = ii
        col[k]  = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

        # f_i+1,N-2
        coef = br5
        row[k]  = ii
        col[k]  = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        data[k]  = -coef
        k += 1

