# coding: utf-8
# Copyright 2022 Emily Bourne

import pytest
import numpy as np

from .utilities import random_grid
from .splines_error_bounds import spline_1d_error_bound, spline_2d_error_bound, spline_1d_error_bound_on_integ
from .analytical_profiles_1d import AnalyticalProfile1D_Cos
from .analytical_profiles_2d import AnalyticalProfile2D_CosCos
from ..splines import make_knots, BSplines, Spline1D, Spline2D
from ..spline_interpolators import SplineInterpolator1D, SplineInterpolator2D

# ===============================================================================


@pytest.mark.serial
@pytest.mark.parametrize("ncells", [1, 5, 10, 23])
@pytest.mark.parametrize("degree", range(1, 11))
def test_SplineInterpolator1D_quadrature_exact(ncells, degree):
    """
    Ensure that the spline quadrature method is accurate to the degree of the
    spline used in its construction
    """

    domain = [-1.0, 1.0]
    periodic = False

    poly_coeffs = np.random.random_sample(degree+1)  # 0 <= c < 1
    poly_coeffs = 1.0 - poly_coeffs                   # 0 < c <= 1
    poly = np.poly1d(poly_coeffs)

    breaks = random_grid(domain, ncells, 0.5)
    knots = make_knots(breaks, degree, periodic)
    basis = BSplines(knots, degree, periodic)
    spline = Spline1D(basis)
    interp = SplineInterpolator1D(basis)

    coeffs = interp.get_quadrature_coefficients()

    xg = basis.greville
    ug = poly(xg)

    quad_value = coeffs @ ug
    poly_integ = poly.integ()
    integral_value = poly_integ(1.0) - poly_integ(-1.0)

    err = abs(integral_value - quad_value)
    assert err < 2.0e-14


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
    Ensure that the spline quadrature method converges to the right result
    when examining the integral of a cosine function
    """

    f = AnalyticalProfile1D_Cos()

    #breaks = random_grid(f.domain, ncells, 0.5)
    breaks = np.linspace(*f.domain, ncells+1)
    knots = make_knots(breaks, degree, periodic)
    basis = BSplines(knots, degree, periodic)
    spline = Spline1D(basis)
    interp = SplineInterpolator1D(basis)

    coeffs = interp.get_quadrature_coefficients()

    xg = basis.greville
    ug = f.eval(xg)

    quad_value = coeffs @ ug
    integral_value = f.eval(
        f.domain[1], diff=-1) - f.eval(f.domain[0], diff=-1)

    err = abs(integral_value - quad_value)

    err_bound = spline_1d_error_bound_on_integ(
        f, np.diff(breaks).max(), degree)

    assert err < err_bound
