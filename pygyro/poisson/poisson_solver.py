from scipy.integrate                import fixed_quad, quadrature   # fixed_quad = fixed order
                                                                    # quadrature = fixed tolerance
from scipy.fftpack                  import fft,ifft
import numpy                        as np
from numpy.polynomial.legendre      import leggauss

from ..model.grid                   import Grid
from ..initialisation.initialiser   import fEq, Te, n0
from ..splines.splines              import Spline1D
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
    
    def getRho ( self, grid: Grid , rho: Grid ):
        assert(grid.currentLayout=="v_parallel")
        assert(rho.currentLayout=="v_parallel")
        
        interpolator = SplineInterpolator1D(grid.get1DSpline())
        bspline = grid.get1DSpline()
        spline = Spline1D(bspline)
        
        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                rho_qv = rho.get1DSlice([i,j])
                for k,theta in grid.getCoords(2):
                    interpolator.compute_interpolant(grid.get1DSlice([i,j,k]),spline)
                    rho_qv[k] = np.sum(self._weights*(spline.eval(self._points)-fEq(r,self._points)))
                rho_qv/=n0(r)
    
class PoissonSolver:
    def __init__( self, eta_grid: list):
        pass
    
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
    
    def findPotential( self, phi: Grid ):
        assert(type(phi.get1DSlice([0,0])[0])==np.complex128)
        for i,r in phi.getCoords(0):
            for j,z in phi.getCoords(1):
                vec=phi.get1DSlice([i,j])
                mode=ifft(vec)
                vec[:]=mode
