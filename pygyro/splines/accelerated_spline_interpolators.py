from typing import TypeVar, Final
import numpy as np
from pyccel.stdlib.internal.lapack import dgbtrs, zgbtrs
from .splines import Spline1D, Spline2D, BSplines
from .sll_m_spline_matrix_periodic_banded import PeriodicBandedMatrix

T = TypeVar('T', float, complex)


def solve_system_periodic(ug: 'Final[float[:]]', spl: Spline1D, offset: int, splu: Final[PeriodicBandedMatrix]):
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


def solve_system_nonperiodic(ug: 'Final[T[:]]', c: 'T[:]', bmat: 'Final[T[:,:](order=F)]', l: np.int32, u: np.int32, ipiv: 'Final[int32[:]]'):
    """
    Compute the coefficients c of the spline which interpolates the points ug
    for a non-periodic spline
    """

    assert ug.shape[0] == bmat.shape[1]
    assert c.shape[0] == ug.shape[0]

    sinfo: np.int32

    c[:] = ug
    if isinstance(c[0], np.float64):
        dgbtrs('N', np.int32(bmat.shape[1]), l, u, np.int32(1), bmat, np.int32(
            bmat.shape[0]), ipiv, c, np.int32(c.shape[0]), sinfo)
    else:
        zgbtrs('N', np.int32(bmat.shape[1]), l, u, np.int32(1), bmat, np.int32(
            bmat.shape[0]), ipiv, c, np.int32(c.shape[0]), sinfo)
    assert sinfo == 0

    return sinfo


def solve_2d_system(ug: 'float[:,:]', spl: Spline2D, wt: 'float[:,:]',
                    r_bmat: 'float[:,:](order=F)', r_l: np.int32, r_u: np.int32, r_ipiv: 'int32[:]',
                    theta_offset: int, theta_splu: Final[PeriodicBandedMatrix]):
    basis1 = spl.basis1
    basis2 = spl.basis2
    n1, n2 = basis1.nbasis, basis2.nbasis
    p1, _ = basis1.degree, basis2.degree
    assert ug.shape[0] == n1
    assert ug.shape[1] == n2

    spline1 = Spline1D(basis1)

    w = spl.coeffs

    s1, s2 = w.shape

    # Cycle over x1 position and interpolate f along x2 direction.
    # Work on spl.coeffs
    sinfo: np.int32
    dgbtrs('N', np.int32(r_bmat.shape[1]), r_l, r_u, np.int32(
        n1), r_bmat, np.int32(r_bmat.shape[0]), r_ipiv, ug.T, np.int32(s2), sinfo)
    assert sinfo == 0

    # Transpose coefficients to self._bwork
    #$omp parallel for collapse(2)
    for i1 in range(n1):
        for i2 in range(n2):
            wt[i2, i1] = ug[i1, i2]

    # Cycle over x2 position and interpolate w along x1 direction.
    # Work on self._bwork
    #$ omp parallel for default(none) shared(wt, n1, n2, theta_offset) firstprivate(spline1, theta_splu) schedule(static)
    for i2 in range(n2):
        solve_system_periodic(wt[i2, :n1], spline1, theta_offset, theta_splu)
        wt[i2, :] = spline1.coeffs

    # Transpose coefficients to spl.coeffs
    #$omp parallel for collapse(2)
    for i1 in range(s1):
        for i2 in range(s2):
            w[i1, i2] = wt[i2, i1]

    # x1-periodic only: "wrap around" coefficients onto extended array
    #$omp parallel for collapse(2)
    for i1 in range(p1):
        for i2 in range(s2):
            w[n1 + i1, i2] = w[i1, i2]
