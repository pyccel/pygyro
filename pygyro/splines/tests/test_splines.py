# coding: utf-8
# Copyright 2018 Yaman Güçlü

import pytest
import numpy as np
from itertools import product
from ..splines import make_knots, BSplines, Spline1D, Spline2D

#===============================================================================
def args_make_knots_periodic():
    for ncells in [1,5,10,23]:
        for degree in range( 1, ncells+1 ):
            yield (ncells, degree)

@pytest.mark.serial
@pytest.mark.parametrize( "ncells,degree", args_make_knots_periodic() )
def test_make_knots_periodic( ncells, degree ):
    breaks = np.arange( ncells+1, dtype=float )
    knots  = make_knots( breaks, degree, periodic=True )
    assert all( np.diff(knots)==1 )

#===============================================================================
@pytest.mark.serial
@pytest.mark.parametrize( "ncells", [1,5,10,23] )
def test_make_knots_periodic_should_fail( ncells ):
    breaks = np.arange( ncells+1, dtype=float )
    with pytest.raises( AssertionError ):
        _ = make_knots( breaks, degree=ncells+1, periodic=True )

#===============================================================================
@pytest.mark.serial
@pytest.mark.parametrize( "ncells", [1,5,10,23] )
@pytest.mark.parametrize( "degree", range(1,11) )
def test_make_knots_clamped( ncells, degree ):
    breaks = np.arange( ncells+1, dtype=float )
    knots  = make_knots( breaks, degree, periodic=False )
    p = degree
    assert all( knots[0:p]==knots[p] )
    assert all( knots[-p:]==knots[-p-1] )
    assert all( np.diff(knots[p:-p])==1 )

#===============================================================================
def args_BSplines():
    for ncells in [1,5,10,23]:
        for periodic in [True, False]:
            pmax = min(ncells,11) if periodic else 11
            for degree in range(1,pmax+1):
                yield (ncells, degree, periodic)

@pytest.mark.serial
@pytest.mark.parametrize( "ncells,degree,periodic", args_BSplines() )
def test_BSplines( ncells, degree, periodic, npts=50, tol=1e-15 ):

    breaks = np.arange( ncells+1, dtype=float )
    knots  = make_knots( breaks, degree, periodic )
    basis  = BSplines( knots, degree, periodic )

    x = np.linspace( breaks[0], breaks[-1], npts ) # Test points
    f = np.zeros( npts ) # Accumulated values of all basis functions

    for i in range(ncells+degree):
        fi = basis[i].eval( x )  # Evaluate basis function at all test points
        f += fi                  # Sum contributions from all basis functions
        assert all( fi >= 0.0 )  # Check positivity of each basis function
    assert all( abs(1.0-f)<tol ) # Check partition of unity

# ...
@pytest.mark.serial
@pytest.mark.parametrize( "ncells,degree,periodic", args_BSplines() )
def test_Spline1D_unit( ncells, degree, periodic, npts=50, tol=1e-15 ):

    breaks = np.arange( ncells+1, dtype=float )
    knots  = make_knots( breaks, degree, periodic )
    basis  = BSplines( knots, degree, periodic )
    spline = Spline1D( basis )
    spline.coeffs.fill(1.0)

    x = np.linspace( breaks[0], breaks[-1], npts ) # Test points
    f = spline.eval( x )

    assert all( abs(1.0-f)<tol )

#===============================================================================
def args_Spline2D_split():
    for ncells in [1,5,10,23]:
        for periodic in [True, False]:
            pmax = min(ncells,5) if periodic else 5
            for degree in range(1,pmax+1):
                yield (ncells, degree, periodic)

def args_Spline2D():
    args1_values = args_Spline2D_split()
    args2_values = args_Spline2D_split()
    for args1, args2 in product( args1_values, args2_values ):
        yield tuple( zip( args1, args2 ) )

@pytest.mark.serial
@pytest.mark.parametrize( "ncells,degree,periodic", args_Spline2D() )
def test_Spline2D_unit( ncells, degree, periodic, npts=10, tol=1e-15 ):

    n1, n2 = ncells
    d1, d2 = degree
    P1, P2 = periodic

    breaks1 = np.arange( n1+1, dtype=float )
    knots1  = make_knots( breaks1, d1, P1 )
    basis1  = BSplines( knots1, d1, P1 )

    breaks2 = np.arange( n2+1, dtype=float )
    knots2  = make_knots( breaks2, d2, P2 )
    basis2  = BSplines( knots2, d2, P2 )

    spline = Spline2D( basis1, basis2 )
    spline.coeffs.fill(1.0)

    x1 = np.linspace( breaks1[0], breaks1[-1], npts ) # Test points
    x2 = np.linspace( breaks2[0], breaks2[-1], npts ) # Test points
    f  = spline.eval( x1, x2 )

    assert np.all( abs(1.0-f)<tol )
