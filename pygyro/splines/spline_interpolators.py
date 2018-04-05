# coding: utf-8
# Copyright 2018 Yaman Güçlü

from scipy.linalg import solve_banded
from .splines     import BSplines, Spline1D

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
        raise NotImplementedError( "Cannot interpolate periodic splines yet" )

    # ...
    def _solve_system_periodic( self, ug, c ):
        raise NotImplementedError( "Cannot interpolate periodic splines yet" )

    # ...
    def _build_system_nonperiodic( self ):

        # TODO: compute number of diagonals in a more accurate way
        #       (see Selalib!)
        u  = self._basis.degree-1  # number of super-diagonals
        n  = self._basis.nbasis
        xg = self._basis.greville

        imat = np.zeros( (2*u+1,n) ) # interpolation matrix

        # TODO: clean the following two cycles
        for i in range(ns-u):
            bspl_i = self._basis[i]
            for j in range(2*u+1):
                imat[j,i] = bspl_i( xg[i+j-u] )

        iend = 0
        for i in range(ns-u,ns):
            iend  += 1
            bspl_i = self._basis[i]
            for j in range(2*u+1-iend):
                imat[j,i] = bspl_i( xg[i+j-u] )

        self.imat = imat

    # ...
    def _solve_system_nonperiodic( self, ug, c ):
        u = self._basis.degree-1
        l = u
        c[:] = solve_banded( (l,u), self.imat, ug )

#===============================================================================
class SplineInterpolator2D():
    pass
