# coding: utf-8
# Copyright 2018 Yaman Güçlü

import pytest
import numpy as np

from .utilities import random_grid
from .splines_error_bounds import spline_1d_error_bound, spline_2d_error_bound
from .analytical_profiles_1d import AnalyticalProfile1D_Poly
from .analytical_profiles_1d import AnalyticalProfile1D_Cos
from .analytical_profiles_2d import AnalyticalProfile2D_CosCos
from ..splines import make_knots, BSplines, Spline1D, Spline2D
from ..spline_interpolators import SplineInterpolator1D, SplineInterpolator2D

# ===============================================================================


@pytest.mark.serial
@pytest.mark.parametrize("ncells", [1, 5, 10, 23])
@pytest.mark.parametrize("degree", range(1, 11))
def test_SplineInterpolator1D_exact(ncells, degree):
    """
    TODO
    """

    domain = [-1.0, 1.0]
    periodic = False

    poly = AnalyticalProfile1D_Poly(degree)

    breaks = random_grid(domain, ncells, 0.5)
    knots = make_knots(breaks, degree, periodic)
    basis = BSplines(knots, degree, periodic, False)
    spline = Spline1D(basis)
    interp = SplineInterpolator1D(basis)

    xg = basis.greville
    ug = poly.eval(xg)

    interp.compute_interpolant(ug, spline)

    xt = np.linspace(*domain, num=100)
    err = spline.eval(xt) - poly.eval(xt)
    derr = spline.eval(xt, der=1) - poly.eval(xt, diff=1)

    max_norm_err = np.max(abs(err))
    max_norm_derr = np.max(abs(derr))
    assert max_norm_err < 2.0e-14
    assert max_norm_derr < 2.0e-12


@pytest.mark.serial
@pytest.mark.parametrize("ncells", [1, 5, 10, 23])
@pytest.mark.parametrize("degree", range(1, 11))
def test_SplineInterpolator1D_exact_uniform(ncells, degree):
    """
    TODO
    """

    domain = [-1.0, 1.0]
    periodic = False

    poly = AnalyticalProfile1D_Poly(degree)

    breaks = np.linspace(*domain, ncells+1)
    knots = make_knots(breaks, degree, periodic)
    basis = BSplines(knots, degree, periodic, True)
    spline = Spline1D(basis)
    interp = SplineInterpolator1D(basis)

    xg = basis.greville
    ug = poly.eval(xg)

    interp.compute_interpolant(ug, spline)

    xt = np.linspace(*domain, num=100)
    err = spline.eval(xt) - poly.eval(xt)
    derr = spline.eval(xt, der=1) - poly.eval(xt, diff=1)

    max_norm_err = np.max(abs(err))
    max_norm_derr = np.max(abs(derr))
    assert max_norm_err < 2.0e-14
    assert max_norm_derr < 2.0e-12

# ===============================================================================


def args_SplineInterpolator1D_cosine():
    """
    TODO
    """
    for ncells in [5, 10, 23]:
        for periodic in [True, False]:
            pmax = min(ncells, 9) if periodic else 9
            for degree in range(1, pmax):
                yield (ncells, degree, periodic)


@pytest.mark.serial
@pytest.mark.parametrize("ncells,degree,periodic",
                         args_SplineInterpolator1D_cosine())
def test_SplineInterpolator1D_cosine(ncells, degree, periodic):
    """
    TODO
    """

    f = AnalyticalProfile1D_Cos()

    breaks = random_grid(f.domain, ncells, 0.5)
    knots = make_knots(breaks, degree, periodic)
    basis = BSplines(knots, degree, periodic, False)
    spline = Spline1D(basis)
    interp = SplineInterpolator1D(basis)

    xg = basis.greville
    ug = f.eval(xg)

    interp.compute_interpolant(ug, spline)

    xt = np.linspace(*f.domain, num=100)
    err = spline.eval(xt) - f.eval(xt)

    max_norm_err = np.max(abs(err))
    err_bound = spline_1d_error_bound(f, np.diff(breaks).max(), degree)

    assert max_norm_err < err_bound

# ===============================================================================


@pytest.mark.parametrize("nc1", [1, 5, 10, 23])
@pytest.mark.parametrize("nc2", [1, 5, 10, 23])
@pytest.mark.parametrize("deg1", range(1, 5))
@pytest.mark.parametrize("deg2", range(1, 5))
def test_SplineInterpolator2D_exact(nc1, nc2, deg1, deg2):
    """
    TODO
    """

    domain1 = [-1.0, 0.8]
    periodic1 = False

    domain2 = [-0.9, 1.0]
    periodic2 = False

    degree = min(deg1, deg2)

    poly = AnalyticalProfile1D_Poly(degree)
    def f(x1, x2): return poly.eval(x1-0.5*x2)

    # Along x1
    breaks1 = random_grid(domain1, nc1, 0.0)
    knots1 = make_knots(breaks1, deg1, periodic1)
    basis1 = BSplines(knots1, deg1, periodic1, False)

    # Along x2
    breaks2 = random_grid(domain2, nc2, 0.0)
    knots2 = make_knots(breaks2, deg2, periodic2)
    basis2 = BSplines(knots2, deg2, periodic2, False)

    # 2D spline and interpolator on tensor-product space
    spline = Spline2D(basis1, basis2)
    interp = SplineInterpolator2D(basis1, basis2)

    x1g = basis1.greville
    x2g = basis2.greville
    ug = f(*np.meshgrid(x1g, x2g, indexing='ij'))

    interp.compute_interpolant(ug, spline)

    x1t = np.linspace(*domain1, num=100)
    x2t = np.linspace(*domain2, num=100)
    err = spline.eval(x1t, x2t) - f(*np.meshgrid(x1t, x2t, indexing='ij'))

    max_norm_err = np.max(abs(err))
    assert max_norm_err < 2.0e-14

# ===============================================================================


@pytest.mark.parametrize("ncells", [10, 20, 40, 80, 160])
@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("periodic1", [True, False])
@pytest.mark.parametrize("periodic2", [True, False])
def test_SplineInterpolator2D_cosine(ncells, degree, periodic1, periodic2):
    """
    TODO
    """

    nc1 = nc2 = ncells
    deg1 = deg2 = degree

    f = AnalyticalProfile2D_CosCos(n1=3, n2=3, c1=0.3, c2=0.7)
    domain1, domain2 = f.domain
    periodic1, periodic2 = (True, True)

    # Along x1
    breaks1 = random_grid(domain1, nc1, 0.0)
    knots1 = make_knots(breaks1, deg1, periodic1)
    basis1 = BSplines(knots1, deg1, periodic1, False)

    # Along x2
    breaks2 = random_grid(domain2, nc2, 0.0)
    knots2 = make_knots(breaks2, deg2, periodic2)
    basis2 = BSplines(knots2, deg2, periodic2, False)

    # 2D spline and interpolator on tensor-product space
    spline = Spline2D(basis1, basis2)
    interp = SplineInterpolator2D(basis1, basis2)

    x1g = basis1.greville
    x2g = basis2.greville
    ug = f.eval(np.meshgrid(x1g, x2g, indexing='ij'))

    interp.compute_interpolant(ug, spline)

    x1t = np.linspace(*domain1, num=20)
    x2t = np.linspace(*domain2, num=20)
    err = spline.eval(x1t, x2t) - f.eval(np.meshgrid(x1t, x2t, indexing='ij'))

    max_norm_err = np.max(abs(err))
    err_bound = spline_2d_error_bound(f, np.diff(
        breaks1).max(), np.diff(breaks2).max(), deg1, deg2)

    assert max_norm_err < err_bound
