# coding: utf-8
# Copyright 2018 Yaman Güçlü

import math
import numpy as np

from .analytical_profiles_base import AnalyticalProfile
from .utilities import horner, falling_factorial

__all__ = ['AnalyticalProfile1D_Cos', 'AnalyticalProfile1D_Poly']

# ===============================================================================


class AnalyticalProfile1D_Cos(AnalyticalProfile):
    """
    TODO
    """

    def __init__(self, n=1, c=0.0):
        twopi = 2.0*math.pi
        self._k = twopi * n
        self._phi = twopi * c

    @property
    def ndims(self):
        """
        TODO
        """
        return 1

    @property
    def domain(self):
        """
        TODO
        """
        return (0.0, 1.0)

    @property
    def poly_order(self):
        """
        TODO
        """
        return -1

    def eval(self, x, diff=0):
        """
        TODO
        """
        return self._k**diff * np.cos(0.5*math.pi*diff + self._k*x + self._phi)

    def max_norm(self, diff=0):
        """
        TODO
        """
        return self._k**diff

# ===============================================================================


class AnalyticalProfile1D_Poly(AnalyticalProfile):
    """
    TODO
    """

    def __init__(self, deg):

        coeffs = np.random.random_sample(1+deg)  # 0 <= c < 1
        coeffs = 1.0 - coeffs                      # 0 < c <= 1

        self._deg = deg
        self._coeffs = coeffs

    @property
    def ndims(self):
        """
        TODO
        """
        return 1

    @property
    def domain(self):
        """
        TODO
        """
        return (-1.0, 1.0)

    @property
    def poly_order(self):
        """
        TODO
        """
        return self._deg

    def eval(self, x, diff=0):
        """
        TODO
        """
        d = diff
        coeffs = [c * falling_factorial(i+d, d)
                  for i, c in enumerate(self._coeffs[d:])]
        return horner(x, *coeffs)

    def max_norm(self, diff=0):
        """
        TODO
        """
        xmin, xmax = self.domain

        if xmax < abs(xmin):
            raise NotImplementedError("General formula not implemented")

        # For xmax >= |xmin|:
        # max(|f^(d)(x)|) = f^(d)(xmax)
        return self.eval(xmax, diff)
