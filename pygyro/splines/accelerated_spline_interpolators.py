from typing import TypeVar, Final
import numpy as np
from pyccel.stdlib.internal.lapack import dgbtrs, zgbtrs
from .splines import Spline1D, Spline2D, BSplines
from .sll_m_spline_matrix_periodic_banded import PeriodicBandedMatrix

T = TypeVar('T', float, complex)

def solve_system_periodic(ug : 'Final[float[:]]', spl : Spline1D, offset : int, splu : Final[PeriodicBandedMatrix]):
    """
    Compute the coefficients c of the spline which interpolates the points ug
    for a periodic spline
    """

    basis = spl.basis

    n = basis.nbasis
    p = basis.degree

    c = spl.coeffs

    c[offset:n+offset] = ug
    splu.solve_inplace(c[offset:n+offset])
    c[:offset] = c[n:n+offset]
    c[n+offset:] = c[offset:p]

# ...
def solve_system_nonperiodic(ug : 'Final[T[:]]', c : 'T[:]', bmat : 'Final[T[:,:](order=F)]', l : np.int32, u : np.int32, ipiv : 'Final[int32[:]]'):
    """
    Compute the coefficients c of the spline which interpolates the points ug
    for a non-periodic spline
    """

    assert ug.shape[0] == bmat.shape[1]
    assert c.shape[0] == ug.shape[0]

    sinfo : np.int32

    c[:] = ug
    if isinstance(c[0], np.float64):
        dgbtrs('N', np.int32(bmat.shape[1]), l, u, np.int32(1), bmat, np.int32(bmat.shape[0]), ipiv, c, np.int32(c.shape[0]), sinfo)
    else:
        zgbtrs('N', np.int32(bmat.shape[1]), l, u, np.int32(1), bmat, np.int32(bmat.shape[0]), ipiv, c, np.int32(c.shape[0]), sinfo)

    return sinfo

def solve_2d_system(ug : 'float[:,:]', spl : Spline2D, wt : 'float[:,:]',
                    r_bmat : 'float[:,:](order=F)', r_l : np.int32, r_u : np.int32, r_ipiv : 'int32[:]',
                    theta_offset : int, theta_splu : PeriodicBandedMatrix):
    basis1 = spl.basis1
    basis2 = spl.basis2
    n1, n2 = basis1.nbasis, basis2.nbasis
    p1, p2 = basis1.degree, basis2.degree
    assert ug.shape[0] == n1
    assert ug.shape[1] == n2

    spline1 = Spline1D(basis1)

    w = spl.coeffs

    # Cycle over x1 position and interpolate f along x2 direction.
    # Work on spl.coeffs
    #$ omp parallel for
    for i1 in range(n1):
        solve_system_nonperiodic(ug[i1, :], w[i1, :], r_bmat, r_l, r_u, r_ipiv)

    s1, s2 = w.shape

    # Transpose coefficients to self._bwork
    #$ omp parallel for collapse(2)
    for i1 in range(s1):
        for i2 in range(s2):
            wt[i2, i1] = w[i1, i2]

    # Cycle over x2 position and interpolate w along x1 direction.
    # Work on self._bwork
    #$ omp parallel for firstprivate(spline1) private(c)
    for i2 in range(n2):
        solve_system_periodic(wt[i2, :n1], spline1, theta_offset, theta_splu)
        #self._interp1.compute_interpolant(wt[i2, :n1], self._spline1)
        c = spline1.coeffs
        wt[i2, :] = c

    # Transpose coefficients to spl.coeffs
    #$ omp parallel for collapse(2)
    for i1 in range(s1):
        for i2 in range(s2):
            w[i1, i2] = wt[i2, i1]

    # x1-periodic only: "wrap around" coefficients onto extended array
    for i1 in range(p1):
        for i2 in range(s2):
            w[n1 + i1, i2] = w[i1, i2]
