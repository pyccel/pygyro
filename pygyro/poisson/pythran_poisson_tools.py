# pythran export get_perturbed_rho(complex128[:,:,:]order(C), float64[:,:]order(C), float64[:,:,:,:]order(C), float64[:])
# pythran export get_perturbed_rho(float64[:,:,:]order(C), float64[:,:]order(C), float64[:,:,:,:]order(C), float64[:])

def get_perturbed_rho(rho, feq, grid, quad_coeffs):
    """
    Calculate:
    rho = \int f - f_eq dv

    Parameters
    ----------
    rho  : 3d array
           Array where the result will be stored
    f_eq : 2d array
           Pre-calculated values of the equilibrium distribution in the (r,v) plane
    grid : 4d array
           The section of the distribution function available in memory
    quad_coeffs : 1d array
           The quadrature coefficients
    """

    n, m, p = rho.shape

    nc, = quad_coeffs.shape

    for i in range(n):
        for j in range(m):
            for k in range(p):
                rho[i, j, k] = 0.0
                for l in range(nc):
                    rho[i, j, k] += quad_coeffs[l] * \
                        (grid[i, j, k, l] - feq[i, l])


# pythran export get_rho(complex128[:,:,:]order(C), float64[:,:,:,:]order(C), float64[:])
# pythran export get_rho(float64[:,:,:]order(C), float64[:,:,:,:]order(C), float64[:])

def get_rho(rho, grid, quad_coeffs):
    """
    Calculate:
    rho = \int f dv

    Parameters
    ----------
    rho  : 3d array
           Array where the result will be stored
    grid : 4d array
           The section of the distribution function available in memory
    quad_coeffs : 1d array
           The quadrature coefficients
    """

    n, m, p = rho.shape

    nc, = quad_coeffs.shape

    for i in range(n):
        for j in range(m):
            for k in range(p):
                rho[i, j, k] = 0.0
                for l in range(nc):
                    rho[i, j, k] += quad_coeffs[l] * grid[i, j, k, l]
