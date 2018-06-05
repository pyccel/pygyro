import numpy as np
from scipy.interpolate              import lagrange
from math                           import pi, floor, ceil

from ..splines.splines              import Spline1D, Spline2D
from ..splines.spline_interpolators import SplineInterpolator1D, SplineInterpolator2D
from ..initialisation.initialiser   import fEq
from ..initialisation               import constants

def fieldline(theta,z,full_z,idx,iota):
    return theta+iota(constants.R0)*(full_z[idx]-z)/constants.R0

class parallelGradient:
    def __init__( self, spline, eta_grid, iota = constants.iota ):
        self._dz = eta_grid[2][1]-eta_grid[2][0]
        self._nz = eta_grid[2].size
        assert(self._nz>5)
        self._inv_dz = 1.0/self._dz
        try:
            dtheta =  np.atleast_2d(self._dz * iota() / constants.R0)
        except:
            dtheta = np.atleast_2d(self._dz * iota(eta_grid[0]) / constants.R0).T
        
        self._bz = self._dz / np.sqrt(self._dz**2+dtheta**2)
        
        self._interpolator = SplineInterpolator1D(spline)
        self._thetaSpline = Spline1D(spline)
        
        self._variesInR = self._bz.size!=1
        
        if (self._variesInR):
            self._thetaVals = np.empty([eta_grid[0].size, eta_grid[1].size, eta_grid[2].size, 6])
            for i,r in enumerate(eta_gird[0]):
                self._getThetaVals(r,self._thetaVals[i],eta_grid,iota)
        else:
            self._thetaVals = np.empty([eta_grid[1].size, eta_grid[2].size, 6])
            self._getThetaVals(eta_grid[0][0],self._thetaVals,eta_grid,iota)
    
    def _getThetaVals( self, r, thetaVals, eta_grid, iota ):
        n = eta_grid[2].size
        for k,z in enumerate(eta_grid[2][:3]):
            for i,l in enumerate([-3,-2,-1,1,2,3]):
                thetaVals[:,(k+l)%n,i]=fieldline(eta_grid[1],z,eta_grid[2],(k+l)%n,iota)
        for k,z in enumerate(eta_grid[2][3:-3]):
            for i,l in enumerate([-3,-2,-1,1,2,3]):
                thetaVals[:,(k+l),i]=fieldline(eta_grid[1],z,eta_grid[2],k+l,iota)
        for k,z in enumerate(eta_grid[2][-3:]):
            for i,l in enumerate([-3,-2,-1,1,2,3]):
                thetaVals[:,(k+l)%n,i]=fieldline(eta_grid[1],z,eta_grid[2],(k+l)%n,iota)
    
    def parallel_gradient( self, phi_r, i ):
        if (self._variesInR):
            bz=self._bz[i]
            thetaVals = self._thetaVals[i]
        else:
            bz=self._bz
            thetaVals = self._thetaVals
        der=np.full_like(phi_r,0)
        
        for i in range(3):
            self._interpolator.compute_interpolant(phi_r[:,i],self._thetaSpline)
            der[:,(i+3)%self._nz]-=self._thetaSpline.eval(thetaVals[:,i,0])
            der[:,(i+2)%self._nz]-=9*self._thetaSpline.eval(thetaVals[:,i,1])
            der[:,(i+1)%self._nz]-=45*self._thetaSpline.eval(thetaVals[:,i,2])
            der[:,(i-1)%self._nz]+=45*self._thetaSpline.eval(thetaVals[:,i,3])
            der[:,(i-2)%self._nz]+=9*self._thetaSpline.eval(thetaVals[:,i,4])
            der[:,(i-3)%self._nz]+=self._thetaSpline.eval(thetaVals[:,i,5])
        
        for i in range(3,self._nz-3):
            self._interpolator.compute_interpolant(phi_r[:,i],self._thetaSpline)
            der[:,(i+3)]-=self._thetaSpline.eval(thetaVals[:,i,0])
            der[:,(i+2)]-=9*self._thetaSpline.eval(thetaVals[:,i,1])
            der[:,(i+1)]-=45*self._thetaSpline.eval(thetaVals[:,i,2])
            der[:,(i-1)]+=45*self._thetaSpline.eval(thetaVals[:,i,3])
            der[:,(i-2)]+=9*self._thetaSpline.eval(thetaVals[:,i,4])
            der[:,(i-3)]+=self._thetaSpline.eval(thetaVals[:,i,5])
        
        for i in range(self._nz-3,self._nz):
            self._interpolator.compute_interpolant(phi_r[:,i],self._thetaSpline)
            der[:,(i+3)%self._nz]-=self._thetaSpline.eval(thetaVals[:,i,0])
            der[:,(i+2)%self._nz]-=9*self._thetaSpline.eval(thetaVals[:,i,1])
            der[:,(i+1)%self._nz]-=45*self._thetaSpline.eval(thetaVals[:,i,2])
            der[:,(i-1)%self._nz]+=45*self._thetaSpline.eval(thetaVals[:,i,3])
            der[:,(i-2)%self._nz]+=9*self._thetaSpline.eval(thetaVals[:,i,4])
            der[:,(i-3)%self._nz]+=self._thetaSpline.eval(thetaVals[:,i,5])
        
        der*= ( bz * self._inv_dz )/60
        
        return der

class fluxSurfaceAdvection:
    def __init__( self, eta_grid, splines, iota = constants.iota ):
        self._points = eta_grid[1:3]
        self._nPoints = (self._points[0].size,self._points[1].size)
        self._interpolator = SplineInterpolator1D(splines[0])
        self._thetaSpline = Spline1D(splines[0])
        self._dz = eta_grid[2][1]-eta_grid[2][0]
        try:
            self._dtheta =  np.atleast_2d(self._dz * iota() / constants.R0)
        except:
            self._dtheta = np.atleast_2d(self._dz * iota(eta_grid[0]) / constants.R0).T
        
        self._bz = self._dz / np.sqrt(self._dz**2+self._dtheta**2)
    
    def step( self, f, dt, c, rGIdx = 0 ):
        assert(f.shape==self._nPoints)
        
        zDist = -c*self._bz[rGIdx]*dt
        
        Shifts = floor( zDist ) + np.array([-2,-1,0,1,2,3])
        thetaShifts = self._dtheta[rGIdx]*Shifts
        
        LagrangeVals = np.ndarray([self._nPoints[1],self._nPoints[0], 6])
        
        for i in range(self._nPoints[1]):
            self._interpolator.compute_interpolant(f[:,i],self._thetaSpline)
            for j,s in enumerate(Shifts):
                LagrangeVals[(i-s)%self._nPoints[1],:,j]=self._thetaSpline.eval(self._points[0]+thetaShifts[j])
        
        for i,z in enumerate(self._points[1]):
            zPts = z+self._dz*Shifts
            for j in range(self._nPoints[0]):
                poly = lagrange(zPts,LagrangeVals[i,j,:])
                f[j,i] = poly(z+zDist)

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
        
        f[:]=self.evalFunc(self._points-c*dt, r)
    
    def evaluate( self, v, r ):
        if (v<self._points[0] or v>self._points[-1]):
            return fEq(r,v);
        else:
            return self._spline.eval(v)

class poloidalAdvection:
    def __init__( self, eta_vals, splines ):
        self._points = eta_vals[1::-1]
        self._shapedQ = np.atleast_2d(self._points[0]).T
        self._nPoints = (self._points[0].size,self._points[1].size)
        self._interpolator = SplineInterpolator2D(splines[0],splines[1])
        self._spline = Spline2D(splines[0],splines[1])
        
        self.evalFunc = np.vectorize(self.evaluate)
    
    def step( self, f, dt, phi: Spline2D, v ):
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        multFactor = dt/constants.B0/self._points[1]
        
        drPhi = phi.eval(*self._points,0,1)
        dthetaPhi = phi.eval(*self._points,1,0)
        
        endPts = ( self._shapedQ   -     drPhi*multFactor,
                   self._points[1] + dthetaPhi*multFactor )
        
        for i in range(self._nPoints[0]):
            for j in range(self._nPoints[1]):
                while (endPts[0][i][j]<0):
                    endPts[0][i][j]+=2*pi
                while (endPts[0][i][j]>2*pi):
                    endPts[0][i][j]-=2*pi
                # TODO: handle boundary conditions in r
                drPhi[i,j]     += phi.eval(endPts[0][i][j],endPts[1][i][j],0,1)
                dthetaPhi[i,j] += phi.eval(endPts[0][i][j],endPts[1][i][j],1,0)
        
        multFactor*=0.5
        
        endPts = ( self._shapedQ   -     drPhi*multFactor,
                   self._points[1] + dthetaPhi*multFactor )
        
        #f[:]=self.evalFunc(np.atleast_2d(self._points[0]).T-drPhi*dt/constants.B0/self.points[1],
        #                   self._points[1]+dthetaPhi*dt/constants.B0/self.points[1])
        
        for i,theta in enumerate(self._points[0]):
            for j,r in enumerate(self._points[1]):
                f[i,j]=self.evalFunc(endPts[0][i][j],endPts[1][i][j],v)
    
    def evaluate( self, theta, r, v ):
        if (r<self._points[1][0]):
            return fEq(self._points[1][0],v);
        elif (r>self._points[1][-1]):
            return fEq(r,v);
        else:
            while (theta>2*pi):
                theta-=2*pi
            while (theta<0):
                theta+=2*pi
            return self._spline.eval(theta,r)
