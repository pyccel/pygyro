from pyccel.decorators import pure

from .utilities_pyccel import neighbour_index


@pure
def assemble_data_4th_order_extrapolation(phi: 'float[:]', grid_theta: 'float[:]', grid_r: 'float[:]', data: 'float[:]'):
    """
    Assemble only the values of the discrete bracket J: f -> {phi, f} based on the Arakawa scheme as
    a sparse matrix with extrapolation in r-direction.

    Parameters
    ----------
        phi : np.ndarray
            array of length N_theta*(N_r + 2); point values of the potential on the grid

        grid_theta: np.ndarray
            array of length N_theta; grid of theta

        grid_r : np.ndarray
            array of length N_r; grid of r

        data : np.ndarray
            empty array of length N_theta * (N_r * 12 + 10); entries of J will be
            written into here.
    """

    N_theta = len(grid_theta)
    N_r = len(grid_r) + 4

    dtheta = (grid_theta[-1] - grid_theta[0]) / (len(grid_theta) - 1)
    dr = (grid_r[-1] - grid_r[0]) / (len(grid_r) - 1)

    factor1 = 1/(6 * dr * dtheta)
    factor2 = 1/4 * factor1

    k = 0
    # Arakawa in the interior
    # allow for columns to get outside of the domain
    for ir in range(2, N_r - 2):
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

            # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
            coef = -(phi[imm] - phi[imp]) * factor2
            data[k] = coef
            k += 1

            # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
            coef = (phi[imm] - phi[imp] + phi[izm] - phi[izp]) * factor1
            data[k] = coef
            k += 1

            # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
            coef = -(phi[imz] * factor1) + (phi[i2mz] - phi[iz2m]) * \
                factor2 + phi[izm] * factor1 + (phi[imp] - phi[ipm]) * factor2
            data[k] = coef
            k += 1

            # -(F[0,-2] (-phi[imm]+phi[ipm]))* factor2
            coef = -(-phi[imm] + phi[ipm]) * factor2
            data[k] = coef
            k += 1

            # +(F[0,-1] (-phi[imm]-phi[imz]+phi[ipm]+phi[ipz]))* factor1
            coef = (-phi[imm] - phi[imz] + phi[ipm] + phi[ipz]) * factor1
            data[k] = coef
            k += 1

            # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
            coef = phi[imz] * factor1 - phi[izp] * factor1 - \
                (phi[i2mz] - phi[iz2p]) * factor2 - \
                (phi[imm] - phi[ipp]) * factor2
            data[k] = coef
            k += 1

            # -(F[0,2] (phi[imp]-phi[ipp]))* factor2
            coef = -(phi[imp] - phi[ipp]) * factor2
            data[k] = coef
            k += 1

            # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
            coef = -(-phi[ipm] + phi[ipp]) * factor2
            data[k] = coef
            k += 1

            # +F[0,1] (phi[imz]* factor1+phi[imp]* factor1-phi[ipz]* factor1-phi[ipp]* factor1)
            coef = (phi[imz] * factor1 + phi[imp] * factor1 -
                    phi[ipz] * factor1 - phi[ipp] * factor1)
            data[k] = coef
            k += 1

            # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
            coef = (-(phi[izm] * factor1) + phi[izp] * factor1 -
                    phi[ipm] * factor1 + phi[ipp] * factor1)
            data[k] = coef
            k += 1

            # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
            coef = (phi[izp] * factor1 - (phi[imp] - phi[ipm]) * factor2 -
                    phi[ipz] * factor1 - (phi[iz2p] - phi[i2pz]) * factor2)
            data[k] = coef
            k += 1

            # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
            coef = (-(phi[izm] * factor1) + phi[ipz] * factor1 + (phi[imm] -
                    phi[ipp]) * factor2 - (-phi[iz2m] + phi[i2pz]) * factor2)
            data[k] = coef
            k += 1

    ir = 0
    for it in range(N_theta):
        ipm = neighbour_index(1, -1, ir, it, N_r, N_theta)
        ipp = neighbour_index(1, 1, ir, it, N_r, N_theta)
        i2pz = neighbour_index(2, 0, ir, it, N_r, N_theta)

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        coef = -(-phi[ipm] + phi[ipp]) * factor2
        data[k] = coef
        k += 1

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

        # -(F[2,0] (-phi[ipm]+phi[ipp]))* factor2
        coef = -(-phi[ipm] + phi[ipp]) * factor2
        data[k] = coef
        k += 1

        # +F[1,0] (-(phi[izm]* factor1)+phi[izp]* factor1-phi[ipm]* factor1+phi[ipp]* factor1)
        coef = (-(phi[izm] * factor1) + phi[izp] * factor1 -
                phi[ipm] * factor1 + phi[ipp] * factor1)
        data[k] = coef
        k += 1

        # +F[1,1] (phi[izp]* factor1-(phi[imp]-phi[ipm])* factor2-phi[ipz]* factor1-(phi[iz2p]-phi[i2pz])* factor2)
        coef = (phi[izp] * factor1 - (phi[imp] - phi[ipm]) * factor2 -
                phi[ipz] * factor1 - (phi[iz2p] - phi[i2pz]) * factor2)
        data[k] = coef
        k += 1

        # +F[1,-1] (-(phi[izm]* factor1)+phi[ipz]* factor1+(phi[imm]-phi[ipp])* factor2-(-phi[iz2m]+phi[i2pz])* factor2)
        coef = (-(phi[izm] * factor1) + phi[ipz] * factor1 + (phi[imm] -
                phi[ipp]) * factor2 - (-phi[iz2m] + phi[i2pz]) * factor2)
        data[k] = coef
        k += 1

    ir = N_r - 2
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

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        coef = -(phi[imm] - phi[imp]) * factor2
        data[k] = coef
        k += 1

        # +(F[-1,0] (phi[imm]-phi[imp]+phi[izm]-phi[izp]))* factor1
        coef = (phi[imm] - phi[imp] + phi[izm] - phi[izp]) * factor1
        data[k] = coef
        k += 1

        # +F[-1,-1] (-(phi[imz]* factor1)+(phi[i2mz]-phi[iz2m])* factor2+phi[izm]* factor1+(phi[imp]-phi[ipm])* factor2)
        coef = -(phi[imz] * factor1) + (phi[i2mz] - phi[iz2m]) * \
            factor2+phi[izm] * factor1 + (phi[imp] - phi[ipm]) * factor2
        data[k] = coef
        k += 1

        # +F[-1,1] (phi[imz]* factor1-phi[izp]* factor1-(phi[i2mz]-phi[iz2p])* factor2-(phi[imm]-phi[ipp])* factor2)
        coef = phi[imz] * factor1-phi[izp] * factor1 - \
            (phi[i2mz]-phi[iz2p]) * factor2-(phi[imm]-phi[ipp]) * factor2
        data[k] = coef
        k += 1

    ir = N_r - 1
    for it in range(N_theta):
        imm = neighbour_index(-1, -1, ir, it, N_r, N_theta)
        imp = neighbour_index(-1, 1, ir, it, N_r, N_theta)
        i2mz = neighbour_index(-2, 0, ir, it, N_r, N_theta)

        # -((F[-2,0] (phi[imm]-phi[imp]))* factor2)
        coef = -(phi[imm] - phi[imp]) * factor2
        data[k] = coef
        k += 1
