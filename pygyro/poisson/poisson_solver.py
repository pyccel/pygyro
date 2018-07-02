from scipy.integrate                import fixed_quad, quadrature   # fixed_quad = fixed order
                                                                    # quadrature = fixed tolerance
from scipy.fftpack                  import fft,ifft
from scipy.sparse                   import coo_matrix
from scipy.sparse.linalg            import spsolve
import numpy                        as np
from numpy.polynomial.legendre      import leggauss

from ..model.grid                   import Grid
from ..initialisation               import constants
from ..initialisation.initialiser   import fEq, Te, n0
from ..splines.splines              import BSplines, Spline1D
from ..splines.spline_interpolators import SplineInterpolator1D

class DensityFinder:
    def __init__ ( self, degree: int, grid: Grid ):
        n=degree//2+1
        points,weights = leggauss(n)
        bspline = grid.getSpline(3)
        breaks = bspline.breaks
        starts = (breaks[:-1]+breaks[1:])/2
        mult = (breaks[1]-breaks[0])/2
        self._points = np.repeat(starts,n) + np.tile(mult*points,len(starts))
        self._weights = np.tile(points,len(starts))
        self._interpolator = SplineInterpolator1D(grid.getSpline(3))
        self._spline = Spline1D(grid.getSpline(3))
    
    def getRho ( self, grid: Grid , rho: Grid ):
        assert(grid.currentLayout=="v_parallel")
        assert(rho.currentLayout=="v_parallel")
        
        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                rho_qv = rho.get1DSlice([i,j])
                for k,theta in grid.getCoords(2):
                    self._interpolator.compute_interpolant(grid.get1DSlice([i,j,k]),self._spline)
                    rho_qv[k] = np.sum(self._weights*(self._spline.eval(self._points)-fEq(r,self._points)))
                rho_qv/=n0(r)
    
class PoissonSolver:
    def __init__( self, eta_grid: list, spline_degree: int, gauss_degree: int, rspline: BSplines):
        n=gauss_degree//2+1
        points,self._weights = leggauss(n)
        r = eta_grid[0]
        starts = (r[:-1]+r[1:])/2
        mults = (r[1:]-r[:-1])/2
        self._points = [start + mult*points for start,mult in zip(starts,mults)]
        self._spline_degree = spline_degree
        
        self._nBasis = spline_degree+1
        
        self._coeff1 = np.empty((self._nBasis,self._nBasis))
        x = np.linspace(starts[0],starts[1],self._nBasis)
        dx = x-x[:,None]
        factor = np.prod(dx+np.eye(self._nBasis),axis=1)
        poly = [np.poly1d(list(x).pop(i),True)/factor[i] for i in range(self._nBasis)]
        
        evalPts = ((x[-1]-x[0])*points+x[0]+x[-1])*0.5
        self._coeff2 = np.empty((self._nBasis,self._nBasis))
        
        self._phi = np.empty((self._nBasis,len(evalPts)))
        
        for i in range(0,self._nBasis):
            self._phi[i,:] = poly[i](evalPts)
            for j in range(0,self._nBasis):
                psi = self._phi[i,:]
                dPsi = poly[i].deriv(1)(evalPts)
                phi = poly[j](evalPts)
                dPhi = poly[j].deriv(1)(evalPts)
                
                self._coeff1[i,j] = -np.sum((dPsi*dPhi + \
                                             ( 1/evalPts - constants.kN0 + \
                                              constants.kN0*np.tanh((evalPts-constants.rp)/constants.deltaRN0)**2 ) * \
                                             psi*dPhi - \
                                             psi*phi/Te(evalPts))*self._weights )
                
                self._coeff2[i,j] = np.sum( psi*phi/(evalPts**2)*self._weights )
        self._nCell = len(eta_grid[0])-1
        
        self._iCoords = [k//self._nBasis + l*spline_degree for l in range(self._nCell) \
                                                           for k in range(self._nBasis*self._nBasis)]
        self._jCoords = [k%self._nBasis + l*spline_degree for l in range(self._nCell) \
                                                          for k in range(self._nBasis*self._nBasis)]
        self._interpolator = SplineInterpolator1D(rspline)
        self._spline = Spline1D(rspline,np.complex128)
        self._nPts = self._nCell*(self._nBasis-1)+1
    
    def getModes( self, phi: Grid, rho: Grid ):
        assert(type(phi.get1DSlice([0,0])[0])==np.complex128)
        assert(type(rho.get1DSlice([0,0])[0])==np.complex128)
        for i,r in phi.getCoords(0):
            for j,z in phi.getCoords(1):
                vec=phi.get1DSlice([i,j])
                mode=fft(vec)
                vec[:]=mode
                vec=rho.get1DSlice([i,j])
                mode[:]=fft(vec)
                vec[:]=mode
    
    def solveEquation( self, m: int, phi: Grid, rho: Grid ):
        allVals = np.tile((self._coeff1+self._coeff2 * m*m).flatten(),self._nCell)
        
        A = coo_matrix((allVals,(self._iCoords,self._jCoords))).tocsr()
        b = np.empty(self._nPts)
        
        for i,q in rho.getCoords(0):
            for j,z in rho.getCoords(1):
                self._interpolator.compute_interpolant(rho.get1DSlice([i,j]),self._spline)
                for k in range(self._nCell):
                    for l in range(self._nBasis):
                        b[k] = np.sum(self._spline.eval(self._points[k])*self._phi[l,:]*self._weights)
                phi.get1DSlice([i,j])[:] = spsolve(A,b)[::self._spline_degree]
    
    def findPotential( self, phi: Grid ):
        assert(type(phi.get1DSlice([0,0])[0])==np.complex128)
        for i,r in phi.getCoords(0):
            for j,z in phi.getCoords(1):
                vec=phi.get1DSlice([i,j])
                mode=ifft(vec)
                vec[:]=mode
