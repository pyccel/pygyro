import numpy as np
from scipy.interpolate              import lagrange
from math                           import pi

from ..splines.splines              import Spline1D, Spline2D
from ..splines.spline_interpolators import SplineInterpolator1D, SplineInterpolator2D
from ..initialisation.initialiser   import fEq
from ..initialisation               import constants

"""
def iota(r):
    return 0
    #return 0.8

def fieldline(theta,z,idx,full_z):
    return theta+iota(r0)*(full_z[idx%len(full_z)]-z)
"""

class fluxSurfaceAdvection:
    def __init__( self, eta_grid, splines, iota = constants.iota ):
        self._points = eta_grid[1:3]
        self._nPoints = (self._points[0].size,self._points[1].size)
        self._interpolator = SplineInterpolator2D(splines[0],splines[1])
        self._spline = Spline2D(splines[0],splines[1])
        self._dz = eta_grid[2][1]-eta_grid[2][0]
        try:
            self._dtheta =  np.atleast_2d(self._dz * iota() / constants.R0)
        except:
            self._dtheta = np.atleast_2d(self._dz * iota(eta_grid[0]) / constants.R0).T
        
        self._bz = self._dz / np.sqrt(self._dz**2+self._dtheta**2)
        
        self._thetaStep = np.mod(self._points[0]+self._dtheta,2*pi)
        self._theta2Step = np.mod(self._points[0]+2*self._dtheta,2*pi)
        self._theta3Step = np.mod(self._points[0]+3*self._dtheta,2*pi)
        self._thetaBStep = np.mod(self._points[0]-self._dtheta,2*pi)
        self._theta2BStep = np.mod(self._points[0]-2*self._dtheta,2*pi)
        self._theta3BStep = np.mod(self._points[0]-3*self._dtheta,2*pi)
    
    def step( self, f, dt, c, rGIdx = 0 ):
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        for i,z in enumerate(self._points[1]):
            zEvalPts = np.take(self._points[1],[i-3,i-2,i-1,i,i+1,i+2,i+3],mode='wrap')
            zPts = [z-3*self._dz, z-2*self._dz, z-self._dz, z, z+self._dz, z+2*self._dz, z+3*self._dz]
            for j,theta in enumerate(self._points[0]):
                thetaPts = [self._theta3BStep[rGIdx][j],self._theta2BStep[rGIdx][j],
                            self._thetaBStep[rGIdx][j],self._points[0][j],
                            self._thetaStep[rGIdx][j],self._theta2Step[rGIdx][j],
                            self._theta3Step[rGIdx][j]]
                #lagrangePts = self._spline.eval(thetaPts,zEvalPts)
                #lagrangePts = self._spline.eval(thetaPts,zEvalPts[0])
                #lagrangePts = self._spline.eval(thetaPts[0],zEvalPts)
                lagrangePts = [self._spline.eval(thetaPts[0],zEvalPts[0]),
                               self._spline.eval(thetaPts[1],zEvalPts[1]),
                               self._spline.eval(thetaPts[2],zEvalPts[2]),
                               self._spline.eval(thetaPts[3],zEvalPts[3]),
                               self._spline.eval(thetaPts[4],zEvalPts[4]),
                               self._spline.eval(thetaPts[5],zEvalPts[5]),
                               self._spline.eval(thetaPts[6],zEvalPts[6])]
                poly = lagrange(zPts,lagrangePts)
                if (self._points[1][i]-c*self._bz[rGIdx]*dt>zPts[-1] or self._points[1][i]-c*self._bz[rGIdx]*dt<zPts[0]):
                    print("outside")
                f[j,i] = poly(self._points[1][i]-c*self._bz[rGIdx]*dt)

class vParallelAdvection:
    def __init__( self, eta_vals, splines ):
        self._points = eta_vals[3]
        self._nPoints = (self._points.size,)
        self._interpolator = SplineInterpolator1D(splines)
        self._spline = Spline1D(splines)
        
        self.evalFunc = np.vectorize(self.evaluate)
    
    def step( self, f, dt, c, r ):
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        #f[:]=self._spline.eval(self._points-c*dt)
        f[:]=self.evalFunc(self._points-c*dt, r)
    
    def evaluate( self, v, r ):
        if (v<self._points[0] or v>self._points[-1]):
            return fEq(r,v);
        else:
            return self._spline.eval(v)

class poloidalAdvection:
    def __init__( self, eta_vals, splines ):
        self._points = eta_vals
        self._nPoints = (self._points[0].size,self._points[1].size)
        self._interpolator = SplineInterpolator2D(splines[0],splines[1])
        self._spline = Spline2D(splines[0],splines[1])
    
    def step( self, f, dt, c ):
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        #f[:]=self._spline.eval(self._points-c*dt)
