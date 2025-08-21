from typing import TypeVar
import numpy as np
from pyccel.stdlib.internal.lapack import dgbtrs, zgbtrs
from .splines import Spline1D, BSplines
from .sll_m_spline_matrix_periodic_banded import PeriodicBandedMatrix

T = TypeVar('T', float, complex)

def solve_system_periodic(ug : 'float[:]', spl : Spline1D, offset : int, splu : PeriodicBandedMatrix):
    """
    Compute the coefficients c of the spline which interpolates the points ug
    for a periodic spline
    """

    n = spl.basis.nbasis
    p = spl.basis.degree

    c = spl.coeffs

    c[offset:n+offset] = ug
    splu.solve_inplace(c[offset:n+offset])
    c[:offset] = c[n:n+offset]
    c[n+offset:] = c[offset:p]

# ...
def solve_system_nonperiodic(ug : 'T[:]', c : 'T[:,:](order=F)', bmat : 'T[:,:](order=F)', l : np.int32, u : np.int32, ipiv : 'int32[:]'):
    """
    Compute the coefficients c of the spline which interpolates the points ug
    for a non-periodic spline
    """

    assert ug.shape[0] == bmat.shape[1]
    assert c.shape[0] == 1
    assert c.shape[1] == ug.shape[0]

    sinfo : np.int32

    c[0,:] = ug
    if isinstance(c[0,0], np.float64):
        dgbtrs('N', np.int32(bmat.shape[0]), l, u, np.int32(1), bmat, np.int32(bmat.shape[0]), ipiv, c, np.int32(c.shape[1]), sinfo)
    else:
        zgbtrs('N', np.int32(bmat.shape[0]), l, u, np.int32(1), bmat, np.int32(bmat.shape[0]), ipiv, c, np.int32(c.shape[1]), sinfo)

    return sinfo
