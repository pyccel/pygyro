# coding: utf-8
# Copyright 2018 Yaman Güçlü

import pytest
import numpy as np
from  .utilities            import horner, random_grid
from ..splines              import make_knots, BSplines, Spline1D
from ..spline_interpolators import SplineInterpolator1D

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
