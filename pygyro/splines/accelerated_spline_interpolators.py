from typing import TypeVar
import numpy as np
from pyccel.stdlib.internal.lapack import dgbtrs, zgbtrs
from .splines import Spline1D, Spline2D, BSplines
from .sll_m_spline_matrix_periodic_banded import PeriodicBandedMatrix

T = TypeVar('T', float, complex)

def solve_system_periodic(ug : 'float[:]', c : 'float[:]', basis : BSplines, offset : int, splu : PeriodicBandedMatrix):
    """
    Compute the coefficients c of the spline which interpolates the points ug
    for a periodic spline
    """

    n = basis.nbasis
    p = basis.degree

    c[offset:n+offset] = ug
    splu.solve_inplace(c[offset:n+offset])
    c[:offset] = c[n:n+offset]
    c[n+offset:] = c[offset:p]

# ...
def solve_system_nonperiodic(ug : 'T[:]', c : 'T[:]', bmat : 'T[:,:](order=F)', l : np.int32, u : np.int32, ipiv : 'int32[:]'):
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
    #assert basis1 is self._basis1
    #assert basis2 is self._basis2

    n1, n2 = basis1.nbasis, basis2.nbasis
    p1, p2 = basis1.degree, basis2.degree

    w = spl.coeffs
    spline1 = Spline1D(basis1)

    # Cycle over x1 position and interpolate f along x2 direction.
    # Work on spl.coeffs
    for i1 in range(n1):
        solve_system_periodic(ug[i1, :], w[i1, :], basis2, theta_offset, theta_splu)
        #self._interp2.compute_interpolant(ug[i1, :], self._spline2)
        #w[i1, :] = self._spline2.coeffs

    # Transpose coefficients to self._bwork
    wt[:, :] = w.transpose()

    # Cycle over x2 position and interpolate w along x1 direction.
    # Work on self._bwork
    for i2 in range(n2):
        solve_system_nonperiodic(wt[i2, :n1], spline1.coeffs, r_bmat, r_l, r_u, r_ipiv)
        #self._interp1.compute_interpolant(wt[i2, :n1], self._spline1)
        wt[i2, :] = spline1.coeffs

    # x2-periodic only: "wrap around" coefficients onto extended array
    if basis2.periodic:
        wt[n2:n2 + p2, :] = wt[:p2, :]

    # Transpose coefficients to spl.coeffs
    w[:, :] = wt.transpose()

    # x1-periodic only: "wrap around" coefficients onto extended array
    if basis1.periodic:
        w[n1:n1 + p1, :] = w[:p1, :]
