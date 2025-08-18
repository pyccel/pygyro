from typing import TypeVar

T = TypeVar('T', 'complex128[:,:,:]', 'float[:,:,:]')


def get_perturbed_rho(rho: T, feq: 'float[:,:]', grid: 'float[:,:,:,:]',
                      quad_coeffs: 'float[:]'):
    """
    Calculate:
    rho = \\int f - f_eq dv

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


def get_rho(rho: T, grid: 'float[:,:,:,:]', quad_coeffs: 'float[:]'):
    """
    Calculate:
    rho = \\int f dv

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
