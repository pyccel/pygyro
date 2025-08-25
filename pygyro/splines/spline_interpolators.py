# coding: utf-8
# Copyright 2018 Yaman Güçlü

import numpy as np
from scipy.linalg.lapack import zgbtrf, zgbtrs, dgbtrf, dgbtrs
from scipy.sparse import csr_matrix, csc_matrix, dia_matrix
from scipy.sparse.linalg import splu

from . import sll_m_spline_matrix_periodic_banded as SLL
from .splines import BSplines, Spline2D, Spline1D, Spline1DComplex
from .spline_eval_funcs import nu_find_span, nu_basis_funs
from .cubic_uniform_spline_eval_funcs import cu_find_span, cu_basis_funs
from .accelerated_spline_interpolators import solve_system_periodic, solve_system_nonperiodic, solve_2d_system

__all__ = ["SplineInterpolator1D", "SplineInterpolator2D"]

# ===============================================================================


class SplineInterpolator1D():
    """
    TODO
    """

    def __init__(self, basis, dtype=float):
        assert isinstance(basis, BSplines)
        self._basis = basis
        self._imat = self.collocation_matrix(
            basis.nbasis, basis.knots, basis.degree, basis.greville, basis.periodic, basis.cubic_uniform)
        if basis.periodic:
            self._offset = self._basis.degree // 2
            max_ku = basis.degree
            n = np.int32(basis.nbasis)
            top_size = n-max_ku
            if (max_ku * (n + top_size) + (3 * max_ku + 1) * top_size >= n * n):
                self._splu = None
            else:
                dmat = dia_matrix(self._imat[max_ku:n-max_ku,max_ku:n-max_ku])
                l = abs(dmat.offsets.min()-self._offset)
                u = dmat.offsets.max()-self._offset
                ku = np.int32(max(l,u))

                self._splu = SLL.PeriodicBandedMatrix(n, ku, ku)
                for i in range(basis.nbasis):
                    for jmin in range(basis.nbasis):
                        j = (jmin - self._offset) % basis.nbasis
                        if self._imat[i,jmin] != 0:
                            self._splu.set_element(np.int32(i+1), np.int32(j+1), self._imat[i,jmin])
                self._splu.factorize()

        else:
            dmat = dia_matrix(self._imat)
            self._l = abs(dmat.offsets.min())
            self._u = dmat.offsets.max()
            cmat = csr_matrix(dmat)
            bmat = np.zeros((1 + self._u + 2 * self._l, cmat.shape[1]), order='F')
            for i, j in zip(*cmat.nonzero()):
                bmat[self._u + self._l+i-j, j] = cmat[i, j]
            if (dtype == complex):
                self._bmat, self._ipiv, self._finfo = zgbtrf(
                    bmat, self._l, self._u)
                self._solveFunc = zgbtrs
            else:
                self._bmat, self._ipiv, self._finfo = dgbtrf(
                    bmat, self._l, self._u)
                self._solveFunc = dgbtrs
            # Move to Fortran indexing
            self._ipiv += 1
            self._sinfo = None

    # ...
    @property
    def basis(self):
        return self._basis

    # ...
    def compute_interpolant(self, ug, spl):
        """
        Compute the coefficients of the spline which interpolates the points ug

        Parameters
        ----------
        ug  : float[:]
               Array of values at the interpolation points
        spl : Spline1D
              The spline in which the coefficients will be saved
        """

        assert isinstance(spl, (Spline1D, Spline1DComplex))
        #assert spl.basis is self._basis
        assert len(ug) == self._basis.nbasis

        if self._basis.periodic:
            if self._splu:
                solve_system_periodic(ug, spl, self._offset, self._splu)
                #self._solve_system_periodic(ug, spl.coeffs)
            else:
                n = spl.basis.nbasis
                p = spl.basis.degree
                c = spl.coeffs
                c[0:n] = np.linalg.solve(self._imat, ug)
                c[n:n+p] = c[0:p]
        else:
            sinfo = solve_system_nonperiodic(ug, spl.coeffs, self._bmat, self._l, self._u, self._ipiv)
            assert sinfo == 0
            #self._solve_system_nonperiodic(ug, spl.coeffs)

    ## ...
    #def _solve_system_periodic(self, ug, c):
    #    """
    #    Compute the coefficients c of the spline which interpolates the points ug
    #    for a periodic spline
    #    """

    #    n = self._basis.nbasis
    #    p = self._basis.degree

    #    if self._splu:
    #        c[self._offset:n+self._offset] = ug
    #        self._splu.solve_inplace(c[self._offset:n+self._offset])
    #        c[:self._offset] = c[n:n+self._offset]
    #        c[n+self._offset:] = c[self._offset:p]
    #    else:
    #        c[0:n] = np.linalg.solve(self._imat, ug)
    #        c[n:n+p] = c[0:p]

    ## ...
    #def _solve_system_nonperiodic(self, ug, c):
    #    """
    #    Compute the coefficients c of the spline which interpolates the points ug
    #    for a non-periodic spline
    #    """

    #    assert ug.shape[0] == self._bmat.shape[1]

    #    assert c.shape == ug.shape
    #    c[:], self._sinfo = self._solveFunc(
    #        self._bmat, self._l, self._u, ug, self._ipiv)

    # ...
    def get_quadrature_coefficients(self):
        """
        Compute the quadrature coefficients equivalent to integrating a spline
        """

        n = self._basis.nbasis
        p = self._basis.degree

        if self._basis.periodic:
            inv_deg = 1 / (p + 1)
            knots = self._basis.knots
            basis_quads = self._basis.integrals[:n].copy()
            basis_quads[:p] += self._basis.integrals[n:]
            return np.linalg.solve(self._imat.T, basis_quads)
        else:
            c, self._sinfo = self._solveFunc(
                self._bmat, self._l, self._u, self._basis.integrals, self._ipiv-1, trans=True)
            return c

    @staticmethod
    def collocation_matrix(nb, knots, degree, xgrid, periodic, cubic_uniform_splines):
        """
        Compute the collocation matrix $C_ij = B_j(x_i)$, which contains the
        values of each B-spline basis function $B_j$ at all locations $x_i$.

        Parameters
        ----------
        knots : 1D array_like
            Knots sequence.

        degree : int
            Polynomial degree of B-splines.

        xgrid : 1D array_like
            Evaluation points.

        periodic : bool
            True if domain is periodic, False otherwise.

        Returns
        -------
        mat : 2D numpy.ndarray
            Collocation matrix: values of all basis functions on each point in xgrid.

        """
        # Number of evaluation points
        nx = len(xgrid)

        # Collocation matrix as 2D Numpy array (dense storage)
        mat = np.zeros((nx, nb))

        # Indexing of basis functions (periodic or not) for a given span
        if periodic:
            def js(span): return [(span-degree+s) %
                                  nb for s in range(degree+1)]
        else:
            def js(span): return slice(span-degree, span+1)

        basis = np.empty(degree+1)

        if cubic_uniform_splines:
            xmin, xmax, dx, f_ncells = knots
            ncells = int(f_ncells)
            # Fill in non-zero matrix values
            for i, x in enumerate(xgrid):
                span, offset = cu_find_span(xmin, xmax, dx, x, ncells)
                cu_basis_funs(span, offset, basis)
                mat[i, js(span)] = basis
        else:
            # Fill in non-zero matrix values
            for i, x in enumerate(xgrid):
                span = nu_find_span(knots, degree, x)
                nu_basis_funs(knots, degree, x, span, basis)
                mat[i, js(span)] = basis

        return mat

# ===============================================================================


class SplineInterpolator2D():
    """
    TODO
    """

    def __init__(self, basis1, basis2, dtype=float):

        assert isinstance(basis1, BSplines)
        assert isinstance(basis2, BSplines)

        self._basis1 = basis1
        self._basis2 = basis2

        if dtype is float:
            self._spline1 = Spline1D(basis1)
            self._spline2 = Spline1D(basis2)
        else:
            assert dtype is np.complex128
            self._spline1 = Spline1DComplex(basis1)
            self._spline2 = Spline1DComplex(basis2)

        self._interp1 = SplineInterpolator1D(basis1, dtype)
        self._interp2 = SplineInterpolator1D(basis2, dtype)

        n1, n2 = basis1.ncells, basis2.ncells
        p1, p2 = basis1.degree, basis2.degree
        self._bwork = np.zeros((n2 + p2, n1 + p1))

    def compute_interpolant(self, ug, spl):
        """
        Compute the coefficients of the spline which interpolates the points ug

        Parameters
        ----------
        ug  : float[:,:]
               Array of values at the interpolation points
        spl : Spline2D
              The spline in which the coefficients will be saved
        """

        assert isinstance(spl, Spline2D)
        basis1 = spl.basis1
        basis2 = spl.basis2
        #assert basis1 is self._basis1
        #assert basis2 is self._basis2

        if basis1.periodic and not basis2.periodic and self._interp1._splu:
            solve_2d_system(ug, spl, self._bwork, self._interp2._bmat, self._interp2._l,
                            self._interp2._u, self._interp2._ipiv, self._interp1._offset,
                            self._interp1._splu)
            return

        n1, n2 = basis1.nbasis, basis2.nbasis
        p1, p2 = basis1.degree, basis2.degree
        assert ug.shape == (n1, n2)

        w = spl.coeffs
        wt = self._bwork

        # Cycle over x1 position and interpolate f along x2 direction.
        # Work on spl.coeffs
        for i1 in range(n1):
            self._interp2.compute_interpolant(ug[i1, :], self._spline2)
            w[i1, :] = self._spline2.coeffs

        # Transpose coefficients to self._bwork
        wt[:, :] = w.transpose()

        # Cycle over x2 position and interpolate w along x1 direction.
        # Work on self._bwork
        for i2 in range(n2):
            self._interp1.compute_interpolant(wt[i2, :n1], self._spline1)
            wt[i2, :] = self._spline1.coeffs

        # x2-periodic only: "wrap around" coefficients onto extended array
        if (self._basis2.periodic):
            wt[n2:n2 + p2, :] = wt[:p2, :]

        # Transpose coefficients to spl.coeffs
        w[:, :] = wt.transpose()

        # x1-periodic only: "wrap around" coefficients onto extended array
        if (self._basis1.periodic):
            w[n1:n1 + p1, :] = w[:p1, :]
