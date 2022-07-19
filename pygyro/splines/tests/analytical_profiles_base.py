# coding: utf-8
# Copyright 2018 Yaman Güçlü

from abc import ABCMeta, abstractmethod

__all__ = ['AnalyticalProfile']

# ===============================================================================


class AnalyticalProfile(metaclass=ABCMeta):
    """
    TODO
    """

    @property
    @abstractmethod
    def ndims(self):
        """ Number of dimensions. """

    @property
    @abstractmethod
    def domain(self):
        """ Domain limits in each dimension. """

    @property
    @abstractmethod
    def poly_order(self):
        """ If profile is polynomial, poly_order=degree;
            otherwise poly_order=-1.
        """

    @abstractmethod
    def eval(self, x, diff):
        """ Evaluate profile (or its derivative) at position x. """

    @abstractmethod
    def max_norm(self, diff):
        """ Compute max-norm of profile (or its derivative) over domain. """
