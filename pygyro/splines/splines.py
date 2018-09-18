# coding: utf-8
# Copyright 2018 Yaman Güçlü

import numpy as np
from scipy.interpolate  import splev, bisplev

from .      import spline_eval_funcs as SEF

__all__ = ['make_knots', 'BSplines', 'Spline1D', 'Spline2D']

if ('mod_pygyro_splines_spline_eval_funcs' in dir(SEF)):
    SEF = SEF.mod_pygyro_splines_spline_eval_funcs
    modFunc = np.transpose
else:
    modFunc = lambda c: c

#===============================================================================
def make_knots( breaks, degree, periodic ):
    """
    Create spline knots from breakpoints, with appropriate boundary conditions.
    Let p be spline degree. If domain is periodic, knot sequence is extended
    by periodicity so that first p basis functions are identical to last p.
    Otherwise, knot sequence is clamped (i.e. endpoints are repeated p times).

    Parameters
    ----------
    breaks : array_like
        Coordinates of breakpoints (= cell edges); given in increasing order and
        with no duplicates.

    degree : int
        Spline degree (= polynomial degree within each interval).

    periodic : bool
        True if domain is periodic, False otherwise.

    Result
    ------
    T : numpy.ndarray (1D)
        Coordinates of spline knots.

    """
    # Type checking
    assert isinstance( degree  , int  )
    assert isinstance( periodic, bool )

    # Consistency checks
    assert len(breaks) > 1
    assert all( np.diff(breaks) > 0 )
    assert degree > 0
    if periodic:
        assert len(breaks) > degree

    p = degree
    T = np.zeros( len(breaks)+2*p )
    T[p:-p] = breaks

    if periodic:
        period = breaks[-1]-breaks[0]
        T[0:p] = [xi-period for xi in breaks[-p-1:-1 ]]
        T[-p:] = [xi+period for xi in breaks[   1:p+1]]
    else:
        T[0:p] = breaks[ 0]
        T[-p:] = breaks[-1]

    return T

#===============================================================================
class BSplines():
    """
    B-splines: basis functions of 1D spline space.

    Parameters
    ----------
    knots : array_like
        Coordinates of knots (clamped or extended by periodicity).

    degree : int
        Polynomial degree.

    periodic : bool
        True if domain is periodic, False otherwise.

    Notes
    -----
    We assume that internal knots are not duplicated. This might change in the
    future.

    """
    def __init__( self, knots, degree, periodic ):

        self._knots    = knots
        self._degree   = degree
        self._periodic = periodic
        self._ncells   = len(knots)-2*degree-1
        self._nbasis   = self._ncells if periodic else self._ncells+degree

    @property
    def degree( self ):
        """ Degree of B-splines.
        """
        return self._degree

    @property
    def ncells( self ):
        """ Number of cells in domain.
        """
        return self._ncells

    @property
    def nbasis( self ):
        """ Number of basis functions, taking into account periodicity.
        """
        return self._nbasis

    @property
    def periodic( self ):
        """ True if domain is periodic, False otherwise.
        """
        return self._periodic

    @property
    def knots( self ):
        """ Knot sequence.
        """
        return self._knots

    @property
    def breaks( self ):
        """ List of breakpoints.
        """
        p = self._degree
        return self._knots[p:-p]

    @property
    def domain( self ):
        """ Domain boundaries [a,b].
        """
        breaks = self.breaks
        return breaks[0], breaks[-1]
        
    @property
    def greville( self ):
        """ Coordinates of all Greville points.
        """
        p = self._degree
        n = self._nbasis
        T = self._knots
        s = 1+p//2 if self._periodic else 1
        x = np.array( [np.sum(T[i:i+p])/p for i in range(s,s+n)] )

        if self._periodic:
            a,b = self.domain
            x = np.around( x, decimals=15 )
            x = (x-a) % (b-a) + a

        return np.around( x, decimals=15 )

    # ...
    def __getitem__( self, i ):
        """
        Get the i-th basis function as a 1D spline.

        Parameters
        ----------
        i : int
            Basis function index: 0 <= i < nbasis.

        Result
        ------
        spl : Spline1D
            Basis function.

        """
        assert isinstance( i, int )
        spl = Spline1D( self )
        spl.coeffs[i] = 1.0
        return spl

    # ...
    def find_cell( self, x ):
        """ Index i of cell $C_{i} := [x_{i},x_{i+1})$ that contains point x.
            Last cell includes right endpoint.
        """
        a, b = self.domain
        assert( a <= x <= b )
        return int( np.searchsorted( self.breaks, x, side='right' ) - 1 )

#===============================================================================

class Spline1D():

    def __init__( self, basis, dtype = float ):
        assert isinstance( basis, BSplines )
        self._basis  = basis
        self._coeffs = np.zeros( basis.ncells + basis.degree, dtype=dtype )

    @property
    def basis( self ):
        return self._basis

    @property
    def coeffs( self ):
        return self._coeffs

    def eval( self, x, der=0 ):
        if (hasattr(x,'__len__')):
            result = np.empty_like(x)
            SEF.eval_spline_1d_vector(x,self._basis.knots,self._basis.degree,self._coeffs,result,der)
        else:
            result = SEF.eval_spline_1d_scalar(x,self._basis.knots,self._basis.degree,self._coeffs,der)
        return result
        
        """
        tck = (self._basis.knots, self._coeffs, self._basis.degree)
        return splev( x, tck, der )
        """

#===============================================================================
class Spline2D():

    def __init__( self, basis1, basis2 ):
        assert isinstance( basis1, BSplines )
        assert isinstance( basis2, BSplines )
        shape = (basis1.ncells + basis1.degree, basis2.ncells + basis2.degree)
        self._basis1 = basis1
        self._basis2 = basis2
        self._coeffs = np.zeros( shape )

        if basis1.degree > 5:
            raise NotImplementedError( "scipy.interpolate.bisplev needs p1 <= 5" )

        if basis2.degree > 5:
            raise NotImplementedError( "scipy.interpolate.bisplev needs p2 <= 5" )

    @property
    def basis( self ):
        return self._basis1, self._basis2

    @property
    def coeffs( self ):
        return self._coeffs

    def eval( self, x1, x2, der1=0, der2=0 ):
        if (hasattr(x1,'__len__')):
            result = np.empty((len(x1),len(x2)))
            SEF.eval_spline_2d_cross(x1,x2,self._basis1.knots,self._basis1.degree,
                                        self._basis2.knots,self._basis2.degree,
                                        modFunc(self._coeffs),modFunc(result),der1,der2)
        else:
            result = SEF.eval_spline_2d_scalar(x1,x2,self._basis1.knots,self._basis1.degree,
                                                 self._basis2.knots,self._basis2.degree,
                                                 modFunc(self._coeffs),der1,der2)
        return result

        """
        t1  = self._basis1.knots
        t2  = self._basis2.knots
        c   = self._coeffs.flat
        k1  = self._basis1.degree
        k2  = self._basis2.degree

        tck = (t1, t2, c, k1, k2)
        return bisplev( x1, x2, tck, der1, der2 )
        """
