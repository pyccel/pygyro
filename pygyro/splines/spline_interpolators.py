# coding: utf-8
# Copyright 2018 Yaman Güçlü

import numpy as np
from scipy.linalg        import solve_banded
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.sparse        import csr_matrix, csc_matrix, dia_matrix
from scipy.sparse.linalg import splu

from .splines           import BSplines, Spline1D, Spline2D
from .                  import mod_context_1

if ('mod_pygyro_splines_mod_context_1' in dir(mod_context_1)):
    mod_context_1 = mod_context_1.mod_pygyro_splines_mod_context_1

__all__ = ["SplineInterpolator1D", "SplineInterpolator2D"]

#===============================================================================
class SplineInterpolator1D():

    def __init__( self, basis ):
        assert isinstance( basis, BSplines )
        self._basis = basis
        self._imat = self.collocation_matrix(basis.knots,basis.degree,basis.greville,basis.periodic)
        if basis.periodic:
            self._splu = splu( csc_matrix(self._imat) )
            self._offset = self._basis.degree // 2
        else:
            dmat = dia_matrix( self._imat )
            self._l = abs( dmat.offsets.min() )
            self._u =      dmat.offsets.max()
            cmat = csr_matrix( dmat )
            bmat = np.zeros( (1+self._u+2*self._l, cmat.shape[1]) )
            for i,j in zip( *cmat.nonzero() ):
                bmat[self._u+self._l+i-j,j] = cmat[i,j]
            self._bmat, self._ipiv, self._finfo = dgbtrf(bmat, self._l, self._u)
            self._sinfo = None

    # ...
    @property
    def basis( self ):
        return self._basis

    # ...
    def compute_interpolant( self, ug, spl ):

        assert isinstance( spl, Spline1D )
        assert spl.basis is self._basis
        assert len(ug) == self._basis.nbasis

        if self._basis.periodic:
            self._solve_system_periodic   ( ug, spl.coeffs )
        else:
            self._solve_system_nonperiodic( ug, spl.coeffs )

    # ...
    def _solve_system_periodic( self, ug, c ):

        n = self._basis.nbasis
        p = self._basis.degree

        c[0:n  ] = self._splu.solve( ug )
        c[n:n+p] = c[0:p]

    # ...
    def _solve_system_nonperiodic( self, ug, c ):
        
        assert ug.shape[0] == self._bmat.shape[1]

        assert c.shape == ug.shape
        c[:], self._sinfo = dgbtrs(self._bmat, self._l, self._u, ug, self._ipiv)
    
    @staticmethod
    def collocation_matrix( knots, degree, xgrid, periodic ):
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
        # Number of basis functions (in periodic case remove degree repeated elements)
        nb = len(knots)-degree-1
        if periodic:
            nb -= degree

        # Number of evaluation points
        nx = len(xgrid)

        # Collocation matrix as 2D Numpy array (dense storage)
        mat = np.zeros( (nx,nb) )

        # Indexing of basis functions (periodic or not) for a given span
        if periodic:
            js = lambda span: [(span-degree+s) % nb for s in range( degree+1 )]
        else:
            js = lambda span: slice( span-degree, span+1 )

        basis = np.empty(degree+1)
        # Fill in non-zero matrix values
        for i,x in enumerate( xgrid ):
            span  =  mod_context_1.find_span( knots, degree, x )
            mod_context_1.basis_funs( knots, degree, x, span, basis )
            mat[i,js(span)] = basis

        return mat

#===============================================================================
class SplineInterpolator2D():

    def __init__( self, basis1, basis2 ):

        assert isinstance( basis1, BSplines )
        assert isinstance( basis2, BSplines )

        self._basis1  = basis1
        self._basis2  = basis2
        self._spline1 = Spline1D( basis1 )
        self._spline2 = Spline1D( basis2 )
        self._interp1 = SplineInterpolator1D( basis1 )
        self._interp2 = SplineInterpolator1D( basis2 )

        n1,n2 = basis1.ncells, basis2.ncells
        p1,p2 = basis1.degree, basis2.degree
        self._bwork = np.zeros( (n2+p2,n1+p1) )

    def compute_interpolant( self, ug, spl ):

        assert isinstance( spl, Spline2D )
        basis1, basis2 = spl.basis
        assert basis1 is self._basis1
        assert basis2 is self._basis2

        n1,n2 = basis1.nbasis, basis2.nbasis
        p1,p2 = basis1.degree, basis2.degree
        assert ug.shape == (n1,n2)

        w  = spl.coeffs
        wt = self._bwork

        # Cycle over x1 position and interpolate f along x2 direction.
        # Work on spl.coeffs
        for i1 in range(n1):
            self._interp2.compute_interpolant( ug[i1,:], self._spline2 )
            w[i1,:] = self._spline2.coeffs

        # Transpose coefficients to self._bwork
        wt[:,:] = w.transpose()

        # Cycle over x2 position and interpolate w along x1 direction.
        # Work on self._bwork
        for i2 in range(n2):
            self._interp1.compute_interpolant( wt[i2,:n1], self._spline1 )
            wt[i2,:] = self._spline1.coeffs

        # x2-periodic only: "wrap around" coefficients onto extended array
        if (self._basis2.periodic):
            wt[n2:n2+p2,:] = wt[:p2,:]

        # Transpose coefficients to spl.coeffs
        w[:,:] = wt.transpose()

        # x1-periodic only: "wrap around" coefficients onto extended array
        if (self._basis1.periodic):
            w[n1:n1+p1,:] = w[:p1,:]
