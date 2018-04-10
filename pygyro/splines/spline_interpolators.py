# coding: utf-8
# Copyright 2018 Yaman Güçlü

import numpy as np
from scipy.linalg        import solve_banded
from scipy.sparse        import lil_matrix, csr_matrix
from scipy.sparse.linalg import splu

from .splines import BSplines, Spline1D

__all__ = ["SplineInterpolator1D", "SplineInterpolator2D"]

#===============================================================================
class SplineInterpolator1D():

    def __init__( self, basis ):
        assert isinstance( basis, BSplines )
        self._basis = basis
        if basis.periodic:
            self._build_system_periodic()
        else:
            self._build_system_nonperiodic()

    # ...
    @property
    def basis( self ):
        return self._basis

    # ...
    def interpolate( self, ug, spl ):

        assert isinstance( spl, Spline1D )
        assert spl.basis is self._basis
        assert len(ug) == self._basis.nbasis

        if self._basis.periodic:
            self._solve_system_periodic   ( ug, spl.coeffs )
        else:
            self._solve_system_nonperiodic( ug, spl.coeffs )

    # ...
    def _build_system_periodic( self ):

        self._offset = self._basis.degree // 2

        n  = self._basis.nbasis
        p  = self._basis.degree
        u  = self._basis.degree-1
        xg = self._basis.greville
        off= self._offset

        imat = lil_matrix( (n,n) )

        for i in range(n):
            xi = xg[i]
            jmin = self._basis.find_cell( xi )
            for s in range(p+1):
                j = (jmin-off+s) % n
                imat[i,j] = self._basis[jmin+s].eval( xi )

        self._splu = splu( imat.tocsc() )

    # ...
    def _solve_system_periodic( self, ug, c ):

        n = self._basis.nbasis
        p = self._basis.degree
        o = self._offset

        c[o:n+o]   = self._splu.solve( ug )
        c[:o]      = c[n:n+o]
        c[n+o:n+p] = c[o:p]

    # ...
    def _build_system_nonperiodic( self ):

        # TODO: compute number of diagonals in a more accurate way
        #       (see Selalib!)
        u  = self._basis.degree-1  # number of super-diagonals
        n  = self._basis.nbasis
        xg = self._basis.greville

        imat = np.zeros( (2*u+1,n) ) # interpolation matrix

        # TODO: clean the following two cycles
        for i in range(n-u):
            bspl_i = self._basis[i]
            for j in range(2*u+1):
                imat[j,i] = bspl_i.eval( xg[i+j-u] )

        iend = 0
        for i in range(n-u,n):
            iend  += 1
            bspl_i = self._basis[i]
            for j in range(2*u+1-iend):
                imat[j,i] = bspl_i.eval( xg[i+j-u] )

        self._imat = imat

    # ...
    def _solve_system_nonperiodic( self, ug, c ):
        u = self._basis.degree-1
        l = u
        c[:] = solve_banded( (l,u), self._imat, ug )

#===============================================================================
class SplineInterpolator2D():
    pass
