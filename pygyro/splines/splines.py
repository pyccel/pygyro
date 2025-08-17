# coding: utf-8
# Copyright 2018 Yaman Güçlü
from typing import TypeVar
import numpy as np
# from scipy.interpolate  import splev, bisplev
from .spline_eval_funcs import nu_eval_spline_1d_scalar, nu_eval_spline_1d_vector
from .spline_eval_funcs import nu_eval_spline_2d_cross, nu_eval_spline_2d_scalar
from .spline_eval_funcs import nu_find_span, nu_basis_funs
from .cubic_uniform_spline_eval_funcs import cu_eval_spline_1d_scalar, cu_eval_spline_1d_vector
from .cubic_uniform_spline_eval_funcs import cu_eval_spline_2d_cross, cu_eval_spline_2d_scalar

__all__ = ['make_knots', 'BSplines', 'Spline1D', 'Spline2D']

float_or_arr = TypeVar('float_or_arr', float, 'float[:]')

# ===============================================================================

@allow_negative_index('T', 'breaks')
def make_knots(breaks : 'float[:]', degree : int, periodic : bool):
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
    assert isinstance(degree, int)
    assert isinstance(periodic, bool)

    # Consistency checks
    assert len(breaks) > 1
    #assert all(np.diff(breaks) > 0)
    assert degree > 0
    if periodic:
        assert len(breaks) > degree

    p = degree
    T = np.zeros(len(breaks)+2*p)
    T[p:-p] = breaks

    if periodic:
        period = breaks[-1]-breaks[0]
        iter_var = breaks[-p-1:-1]
        tmp_start = [xi-period for xi in iter_var]
        T[0:p] = tmp_start
        iter_var = breaks[1:p+1]
        tmp_end = [xi+period for xi in iter_var]
        T[-p:] = tmp_end
    else:
        T[0:p] = breaks[0]
        T[-p:] = breaks[-1]

    return T

# ===============================================================================


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

    uniform : bool
        True if knots are equidistant, False otherwise.

    Notes
    -----
    We assume that internal knots are not duplicated. This might change in the
    future.

    """

    @allow_negative_index('knots')
    def __init__(self, knots : 'float[:]', degree : int, periodic : bool, uniform : bool):
        xmin = knots[degree]
        xmax = knots[-degree-1]
        dx = knots[degree+1]-knots[degree]

        self._cubic_uniform_splines = (degree == 3) and uniform

        self._degree = degree
        self._periodic = periodic
        self._ncells = len(knots)-2*degree-1
        self._nbasis = self._ncells if periodic else self._ncells+degree
        self._offset = degree//2 if periodic else 0
        self._integrals : 'float[:]'

        if self._cubic_uniform_splines:
            self._knots = np.array([xmin, xmax, dx, float(self._ncells)])
            assert (int(self._knots[3]) == self._ncells)
        else:
            self._knots = np.array(knots)

        self._build_integrals()

        if self._cubic_uniform_splines:
            if periodic:
                self._interp_pts = np.linspace(
                    xmin, xmax, self._ncells, endpoint=False)
            else:
                self._interp_pts = np.empty(self._nbasis+2)
                self._interp_pts[0] = xmin
                self._interp_pts[1] = xmin+dx/3
                self._interp_pts[2:-2] = np.linspace(xmin+dx, xmax-dx, self._nbasis-4)
                self._interp_pts[-2] = xmax-dx/3
                self._interp_pts[-1] = xmax
                #self._interp_pts = np.array([xmin,
                #                            xmin+dx/3,
                #                            *np.linspace(xmin+dx, xmax-dx, self._nbasis-4),
                #                            xmax-dx/3,
                #                            xmax])

    @property
    def degree(self):
        """ Degree of B-splines.
        """
        return self._degree

    @property
    def ncells(self):
        """ Number of cells in domain.
        """
        return self._ncells

    @property
    def nbasis(self):
        """ Number of basis functions, taking into account periodicity.
        """
        return self._nbasis

    @property
    def periodic(self):
        """ True if domain is periodic, False otherwise.
        """
        return self._periodic

    @property
    def knots(self):
        """ Knot sequence.
        """
        return self._knots

    @property
    def breaks(self):
        """ List of breakpoints.
        """
        if self.cubic_uniform:
            xmin, xmax, _, _ = self._knots
            return np.linspace(xmin, xmax, self._ncells+1)
        else:
            p = self._degree
            n = len(self._knots)
            return np.array(self._knots[p:n-p])

    @property
    def domain(self):
        """ Domain boundaries [a,b].
        """
        #breaks = self.breaks
        return self.breaks[0], self.breaks[-1]

    @property
    def cubic_uniform(self):
        return self._cubic_uniform_splines

    @property
    def greville(self):
        """ Coordinates of all Greville points.
        """
        if self._cubic_uniform_splines:
            x = np.array(self._interp_pts)
            return x
        else:
            p = self._degree
            n = self._nbasis
            T = self._knots
            s = 1+p//2 if self._periodic else 1
            x = np.array([np.sum(T[i:i+p])/p for i in range(s, s+n)])

            if self._periodic:
                a, b = self.domain
                #x = np.around(x, decimals=15)
                x[:] = (x-a) % (b-a) + a

            return x#np.around(x, decimals=15)

    @property
    def integrals(self):
        return self._integrals

    # ...
    #def __getitem__(self, i : int):
    #    """
    #    Get the i-th basis function as a 1D spline.

    #    Parameters
    #    ----------
    #    i : int
    #        Basis function index: 0 <= i < nbasis.

    #    Result
    #    ------
    #    spl : Spline1D
    #        Basis function.

    #    """
    #    #assert isinstance(i, int)
    #    spl = Spline1D(self)
    #    spl.coeffs[i] = 1.0
    #    if spl.basis.periodic:
    #        n = spl.basis.ncells
    #        p = spl.basis.degree
    #        spl.coeffs[n:n+p] = spl.coeffs[0:p]
    #    return spl

    # ...
    #def find_cell(self, x : float):
    #    """ Index i of cell $C_{i} := [x_{i},x_{i+1})$ that contains point x.
    #        Last cell includes right endpoint.
    #    """
    #    a, b = self.domain
    #    assert a <= x <= b
    #    #return int(np.searchsorted(self.breaks, x, side='right') - 1)
    #    return 0

    @allow_negative_index('integrals')
    def _build_integrals(self):
        n = self.nbasis
        d = self.degree

        self._integrals = np.empty(self.ncells + d)
        integrals = self._integrals
        inv_deg = 1 / (d + 1)

        if self.cubic_uniform:
            #knots = self._knots
            #xmin = knots[0]
            #dx = knots[2]
            #xmin, _, dx, _ = self._knots
            xmin = self._knots[0]
            dx = self._knots[2]
            if self.periodic:
                self._integrals[:] = dx
                self._integrals[n:] = 0
            else:
                self._integrals[d:-d] = dx
                values = np.empty(d+2)
                knots = np.linspace(xmin, xmin+dx*11, 12)
                test_pt = xmin + 4*dx
                span = nu_find_span(knots, 4, test_pt)
                nu_basis_funs(knots, 4, test_pt, span, values)

                for i in range(3):
                    step = dx*(1 - np.sum(values[:3-i]))
                    self._integrals[i] = step
                    integrals[-i-1] = step
        else:
            knots = np.empty(self.knots.shape[0]+2)
            knots[0] = self._knots[0]
            knots[1:-1] = self._knots
            knots[-1] = self._knots[-1]
            #knots = np.array([self.knots[0], *self.knots, self.knots[-1]])
            values = np.empty(d+2)

            for i in range(n):
                integ_deg = d+1
                lbound = max(self.breaks[0], knots[i+1])
                ubound = min(self.breaks[-1], knots[d+2+i])
                span_l = nu_find_span(knots, integ_deg, lbound)
                span_u = nu_find_span(knots, integ_deg, ubound)

                nu_basis_funs(knots, integ_deg, lbound, span_l, values)
                first_available = span_l - integ_deg
                first_wanted = i+1
                min_idx = first_wanted-first_available
                l = np.sum(values[min_idx:])

                nu_basis_funs(knots, integ_deg, ubound, span_u, values)
                first_available = span_u - integ_deg
                first_wanted = i+1
                min_idx = first_wanted-first_available
                u = np.sum(values[min_idx:])

                self._integrals[i] = (
                    knots[d+2+i] - knots[i+1])*inv_deg*(u - l)

            if self.periodic:
                for i in range(d):
                    self._integrals[n+i] = self._integrals[d-i-1]

# ===============================================================================


class Spline1D():
    """
    TODO
    """

    def __init__(self, basis : BSplines):
        #assert isinstance(basis, BSplines)
        self._basis = basis
        self._coeffs = np.zeros(basis.ncells + basis.degree, dtype=float)

    @property
    def basis(self):
        """
        TODO
        """
        return self._basis

    @property
    def coeffs(self):
        """
        TODO
        """
        return self._coeffs

    def eval(self, x : 'float|float[:]', der : int = 0):
        """
        TODO
        """
        #if self._basis.cubic_uniform:
        #    result = cu_eval_spline_1d_scalar(
        #        x, self._basis.knots, self._basis.degree, self._coeffs, der)
        #else:
        #    result = nu_eval_spline_1d_scalar(
        #        x, self._basis.knots, self._basis.degree, self._coeffs, der)
        if isinstance(x, float):
            if self._basis.cubic_uniform:
                result = cu_eval_spline_1d_scalar(
                    x, self._basis.knots, self._basis.degree, self._coeffs, der)
            else:
                result = nu_eval_spline_1d_scalar(
                    x, self._basis.knots, self._basis.degree, self._coeffs, der)
        else:
            result = np.empty_like(x)
            if self._basis.cubic_uniform:
                cu_eval_spline_1d_vector(x, self._basis.knots,
                                         self._basis.degree, self._coeffs, result, der)
            else:
                nu_eval_spline_1d_vector(x, self._basis.knots,
                                         self._basis.degree, self._coeffs, result, der)
        return result

        """
        tck = (self._basis.knots, self._coeffs, self._basis.degree)
        return splev( x, tck, der )
        """

    def eval_vector(self, x : 'float[:]', y : 'float[:]', der : int =0):
        """
        TODO
        """
        if self._basis.cubic_uniform:
            cu_eval_spline_1d_vector(x, self._basis.knots,
                                     self._basis.degree, self._coeffs, y, der)
        else:
            nu_eval_spline_1d_vector(x, self._basis.knots,
                                     self._basis.degree, self._coeffs, y, der)

# ===============================================================================


class Spline2D():
    """
    TODO
    """

    def __init__(self, basis1 : BSplines, basis2 : BSplines):
        #assert isinstance(basis1, BSplines)
        #assert isinstance(basis2, BSplines)
        self._basis1 = basis1
        self._basis2 = basis2
        self._coeffs = np.zeros((basis1.ncells + basis1.degree, basis2.ncells + basis2.degree))

        assert basis1.degree <= 5, "scipy.interpolate.bisplev needs p1 <= 5"

        assert basis2.degree <= 5, "scipy.interpolate.bisplev needs p2 <= 5"

        assert basis1.cubic_uniform == basis2.cubic_uniform

    @property
    def basis(self):
        """
        TODO
        """
        return self._basis1, self._basis2

    @property
    def coeffs(self):
        """
        TODO
        """
        return self._coeffs

    def eval(self, x1 : float_or_arr, x2 : float_or_arr, der1 : int = 0, der2 : int = 0):
        """
        TODO
        """
        if isinstance(x1, float):
            if self._basis1.cubic_uniform:
                result = cu_eval_spline_2d_scalar(x1, x2, self._basis1.knots, self._basis1.degree,
                                                  self._basis2.knots, self._basis2.degree,
                                                  self._coeffs, der1, der2)
            else:
                result = nu_eval_spline_2d_scalar(x1, x2, self._basis1.knots, self._basis1.degree,
                                                  self._basis2.knots, self._basis2.degree,
                                                  self._coeffs, der1, der2)
        else:
            result = np.empty((len(x1), len(x2)))
            if self._basis1.cubic_uniform:
                cu_eval_spline_2d_cross(x1, x2, self._basis1.knots, self._basis1.degree,
                                        self._basis2.knots, self._basis2.degree,
                                        self._coeffs, result, der1, der2)
            else:
                nu_eval_spline_2d_cross(x1, x2, self._basis1.knots, self._basis1.degree,
                                        self._basis2.knots, self._basis2.degree,
                                        self._coeffs, result, der1, der2)
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

    def eval_vector(self, x1 : 'float[:]', x2 : 'float[:]', y : 'float[:,:]', der1 : int = 0, der2 : int = 0):
        """
        TODO
        """
        if self._basis1.cubic_uniform:
            cu_eval_spline_2d_cross(x1, x2, self._basis1.knots, self._basis1.degree,
                                    self._basis2.knots, self._basis2.degree,
                                    self._coeffs, y, der1, der2)
        else:
            nu_eval_spline_2d_cross(x1, x2, self._basis1.knots, self._basis1.degree,
                                    self._basis2.knots, self._basis2.degree,
                                    self._coeffs, y, der1, der2)
