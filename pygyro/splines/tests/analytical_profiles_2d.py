# coding: utf-8
# Copyright 2018 Yaman Güçlü

import math
import numpy as np

from .analytical_profiles_base import AnalyticalProfile

__all__ = ['AnalyticalProfile2D_CosCos']

#===============================================================================
class AnalyticalProfile2D_CosCos( AnalyticalProfile ):

    def __init__( self, n1=1, n2=1, c1=0.0, c2=0.0 ):
        twopi      = 2.0*math.pi
        self._k1   = twopi * n1
        self._k2   = twopi * n2
        self._phi1 = twopi * c1
        self._phi2 = twopi * c2

    @property
    def ndims( self ):
        return 2

    @property
    def domain( self ):
        return (0.,1.), (0.,1.)

    @property
    def poly_order( self ):
        return (-1,-1)

    def eval( self, x, diff=[0,0] ):
        x1, x2 = x
        d1, d2 = diff
        halfpi = 0.5*math.pi
        return self._k1**d1 * np.cos( halfpi*d1 + self._k1*x1 + self._phi1 ) \
             * self._k2**d2 * np.cos( halfpi*d2 + self._k2*x2 + self._phi2 )

    def max_norm( self, diff=[0,0] ):
        d1, d2 = diff
        return self._k1**d1 * self._k2**d2
