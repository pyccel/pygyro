# coding: utf-8
# Copyright 2018 Yaman Güçlü

import pytest
import numpy as np
from  .utilities            import horner, random_grid
from ..splines              import make_knots, BSplines, Spline1D
from ..spline_interpolators import SplineInterpolator1D

#===============================================================================
def random_grid( domain, ncells, random_fraction ):
    """ Create random grid over 1D domain with given number of cells.
    """
    # Create uniform grid on [0,1]
    x = np.linspace( 0.0, 1.0, ncells+1 )

    # Apply random displacement to all points, then sort grid
    x += (np.random.random_sample( ncells+1 )-0.5) * (random_fraction/ncells)
    x.sort()

    # Apply linear transformation y=m*x+q to match domain limits
    xa, xb = x[0], x[-1]
    ya, yb = domain
    m = (   yb-ya   )/(xb-xa)
    q = (xb*ya-xa*yb)/(xb-xa)
    y = m*x + q

    # Avoid possible round-off
    y[0], y[-1] = domain

    return y

#===============================================================================
def horner( x, *poly_coeffs ):
    """ Use Horner's Scheme to evaluate a polynomial
        of coefficients *poly_coeffs at location x.
    """
    r = 0
    for a in poly_coeffs[::-1]:
        r = r*x + a
    return r

#===============================================================================
@pytest.mark.parametrize( "ncells", [1,5,10,23] )
@pytest.mark.parametrize( "degree", range(1,11) )

def test_SplineInterpolator1D_exact( ncells, degree ):

    domain   = [-1.0, 1.0]
    periodic = False

    poly_coeffs = np.random.random_sample( degree+1 ) # 0 <= c < 1
    poly_coeffs = 1.0 - poly_coeffs                   # 0 < c <= 1
    f = lambda x : horner( x, *poly_coeffs )

    breaks = random_grid( domain, ncells, 0.5 )
    knots  = make_knots( breaks, degree, periodic )
    basis  = BSplines( knots, degree, periodic )
    spline = Spline1D( basis )
    interp = SplineInterpolator1D( basis )

    xg = basis.greville
    ug = f( xg )

    interp.interpolate( ug, spline )

    xt  = np.linspace( *domain, num=100 )
    err = spline.eval( xt ) - f( xt )

    max_norm_err = np.max( abs( err ) )
    assert max_norm_err < 1.0e-14
