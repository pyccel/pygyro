import numpy as np

from ..splines.splines import Spline1D, Spline2D
from ..splines.spline_interpolators import SplineInterpolator1D, SplineInterpolator2D

def fieldline(theta,z,idx,full_z):
    return theta+iota(r0)*(full_z[idx%len(full_z)]-z)

def fluxSurfaceAdv(f,splines):
    pass

class vParAdvection:
    def __init__(self,eta_grid,splines):
        self._points = eta_grid[3]
        self._nPoints = self._points.shape
        self._interpolator = SplineInterpolator1D(splines[3])
        self._spline = Spline1D(splines[3])
    
    def step(f,dt,c):
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        f[:]=self._spline.eval(self._points-c*dt)


def vParallelAdv(f,greville,spline,dt,c):
    
    interp = SplineInterpolator1D(spline)
    mySpline = Spline1D(spline)
    
    interp.compute_interpolant(f,mySpline)
    
    f[:]=mySpline.eval(greville-c*dt)

def poloidalAdv(f):
    pass
