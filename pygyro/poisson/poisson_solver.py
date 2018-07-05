from scipy.integrate                import fixed_quad, quadrature   # fixed_quad = fixed order
                                                                    # quadrature = fixed tolerance
from scipy.fftpack                  import fft,ifft
import scipy.sparse                 as sparse
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
        self._multFact = (breaks[1]-breaks[0])/2
        self._points = np.repeat(starts,n) + np.tile(self._multFact*points,len(starts))
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
                    rho_qv[k] = np.sum(self._multFact*self._weights*(self._spline.eval(self._points)-fEq(r,self._points)))
                rho_qv/=n0(r)
    
class PoissonSolver:
    def __init__( self, eta_grid: list, degree: int, rspline: BSplines,
                  lBoundary: str = 'dirichlet', rBoundary: str = 'dirichlet'):
        n=degree//2+1
        points,self._weights = leggauss(n)
        
        nDiags = rspline.degree + 1
        
        nPts = eta_grid[0].size
        
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        self._evalPts = startPoints[:,None]+points[None,:]*multFactor
        
        self._coeffs = np.empty([rspline.nbasis,1],np.complex128)
        if (lBoundary=='dirichlet'):
            start_range = 1
            self._coeffs[0] = 0
        else:
            start_range = 0
        if (rBoundary=='dirichlet'):
            end_range = rspline.nbasis-1
            self._coeffs[end_range] = 0
        else:
            end_range = rspline.nbasis
        self._coeff_range = slice(start_range,end_range)
        
        self._nUnknowns = end_range-start_range
        maxEnd = self._nUnknowns-1
        
        if (rspline.nbasis > 4*rspline.degree):
            massCoeffs = np.zeros((2*rspline.degree+1,))
            dPhidPsiCoeffs = np.zeros((2*rspline.degree+1,))
            dPhiPsiCoeffs = np.zeros((2*rspline.degree+1,))
            
            refSpline = rspline[2*rspline.degree]
            start = rspline.degree
            end = 2*rspline.degree+1
            
            vals = refSpline.eval(self._evalPts[start:end].flatten())
            derivs = refSpline.eval(self._evalPts[start:end].flatten(),1)
            nPts = n*(rspline.degree+1)
            
            massCoeffs[rspline.degree] = np.sum( np.tile(self._weights,rspline.degree+1) * \
                                                 multFactor * vals**2 )
            
            dPhidPsiCoeffs[rspline.degree] = np.sum( np.tile(self._weights,rspline.degree+1) * \
                                                 multFactor * derivs**2 )
            
            dPhiPsiCoeffs[rspline.degree] = np.sum( np.tile(self._weights,rspline.degree+1) * \
                                                 multFactor * derivs * vals )
            
            for i in range(1,rspline.degree+1):
                start_i = start + i
                end_i   = end + i
                
                diff=i*n
                
                massCoeffs[rspline.degree+i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        vals[:nPts-diff] * vals[diff:] )
                dPhidPsiCoeffs[rspline.degree+i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        derivs[:nPts-diff] * derivs[diff:] )
                dPhiPsiCoeffs[rspline.degree+i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        vals[diff:] * derivs[:nPts-diff] )
                dPhiPsiCoeffs[rspline.degree-i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        vals[:nPts-diff] * derivs[diff:] )
                
                massCoeffs[rspline.degree-i]=massCoeffs[rspline.degree+i]
                dPhidPsiCoeffs[rspline.degree-i]=dPhidPsiCoeffs[rspline.degree+i]
            
            self._massMatrix = sparse.diags(massCoeffs,range(-rspline.degree,rspline.degree+1),
                                         (self._nUnknowns,self._nUnknowns),'lil')
            self._dPhidPsi = sparse.diags(dPhidPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                         (self._nUnknowns,self._nUnknowns),'lil')
            self._dPhiPsi = sparse.diags(dPhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                         (self._nUnknowns,self._nUnknowns),'lil')
            
            start_range=1
            if (lBoundary!=rBoundary):
                if (lBoundary=='neumann'):
                    for i in range(start_range,rspline.degree+1):
                        spline = rspline[i]
                        
                        for j in range(i,i+rspline.degree):
                            self._massMatrix[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                        spline.eval(self._evalPts[:j+1].flatten()) )
                            self._dPhidPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                        spline.eval(self._evalPts[:j+1].flatten(),1) )
                            self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                        spline.eval(self._evalPts[:j+1].flatten()) )
                            self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                        spline.eval(self._evalPts[:j+1].flatten(),1) )
                            
                            self._massMatrix[j,i]=self._massMatrix[i,j]
                            self._dPhidPsi[j,i]=self._dPhidPsi[i,j]
                else:
                    for i in range(start_range,rspline.degree+1):
                        spline = rspline[i]
                        
                        for j in range(i,i+rspline.degree):
                            self._massMatrix[maxEnd-i,maxEnd-j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                        spline.eval(self._evalPts[:j+1].flatten()) )
                            self._dPhidPsi[maxEnd-i,maxEnd-j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                        spline.eval(self._evalPts[:j+1].flatten(),1) )
                            self._dPhiPsi[maxEnd-i,maxEnd-j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                        spline.eval(self._evalPts[:j+1].flatten()) )
                            self._dPhiPsi[maxEnd-i,maxEnd-j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                    rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                        spline.eval(self._evalPts[:j+1].flatten(),1) )
                            
                            self._massMatrix[maxEnd-j,maxEnd-i]=self._massMatrix[maxEnd-i,maxEnd-j]
                            self._dPhidPsi[maxEnd-j,maxEnd-i]=self._dPhidPsi[maxEnd-i,maxEnd-j]
            elif (lBoundary=='neumann'):
                start_range=0
            
            for i in range(start_range,rspline.degree+1):
                spline = rspline[i]
                
                for j in range(i,i+rspline.degree):
                    self._massMatrix[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                            rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                spline.eval(self._evalPts[:j+1].flatten()) )
                    self._dPhidPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                            rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                spline.eval(self._evalPts[:j+1].flatten(),1) )
                    self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                            rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                spline.eval(self._evalPts[:j+1].flatten()) )
                    self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                            rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                spline.eval(self._evalPts[:j+1].flatten(),1) )
                    
                    self._massMatrix[j,i]=self._massMatrix[i,j]
                    self._massMatrix[maxEnd-i,maxEnd-j]=self._massMatrix[i,j]
                    self._massMatrix[maxEnd-j,maxEnd-i]=self._massMatrix[i,j]
                    
                    self._dPhidPsi[j,i]=self._dPhidPsi[i,j]
                    self._dPhidPsi[maxEnd-i,maxEnd-j]=self._dPhidPsi[i,j]
                    self._dPhidPsi[maxEnd-j,maxEnd-i]=self._dPhidPsi[i,j]
                    
                    self._dPhiPsi[maxEnd-i,maxEnd-j]=self._dPhiPsi[i,j]
                    self._dPhiPsi[maxEnd-j,maxEnd-i]=self._dPhiPsi[i,j]
            self._massMatrix=self._massMatrix.tocsr()
            self._dPhidPsi=self._dPhidPsi.tocsr()
            self._dPhidPsi=self._dPhidPsi.tocsr()
        else:
            self._massMatrix = np.zeros((self._nUnknowns,self._nUnknowns))
            self._dPhidPsi = np.zeros((self._nUnknowns,self._nUnknowns))
            self._dPhiPsi = np.zeros((self._nUnknowns,self._nUnknowns))
            
            for i in range(rspline.nbasis):
                start_i = max(0,i-rspline.degree)
                end_i = min(maxEnd,i+1)
                spline = rspline[i]
                
                for j in range(i,rspline.nbasis):
                    start_j = max(0,j-rspline.degree)
                    if (start_j<end_i):
                        end_j = min(maxEnd,j+1)
                        start = max(start_i,start_j)
                        end = min(end_i,end_j)
                        self._massMatrix[i,j]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[j].eval(self._evalPts[start:end].flatten()) * \
                                    spline.eval(self._evalPts[start:end].flatten()) )
                        
                        self._dPhidPsi[i,j]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[j].eval(self._evalPts[start:end].flatten(),1) * \
                                    spline.eval(self._evalPts[start:end].flatten(),1) )
                        
                        self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[j].eval(self._evalPts[start:end].flatten(),1) * \
                                    spline.eval(self._evalPts[start:end].flatten()) )
                        self._dPhiPsi[j,i]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[j].eval(self._evalPts[start:end].flatten()) * \
                                    spline.eval(self._evalPts[start:end].flatten(),1) )
                        
                        self._massMatrix[j,i]=self._massMatrix[i,j]
                        self._dPhidPsi[j,i]=self._dPhidPsi[i,j]
            self._massMatrix = sparse.csr_matrix(self._massMatrix)
            self._dPhidPsi = sparse.csr_matrix(self._dPhidPsi)
            self._dPhiPsi = sparse.csr_matrix(self._dPhiPsi)
        
        r = eta_grid[0][self._coeff_range]
        
        self._stiffnessMatrix = sparse.csr_matrix(-self._dPhidPsi \
                                -np.diag( 1/r - constants.kN0* (1 + \
                                  np.tanh( (r - constants.rp ) / constants.deltaRN0 )**2) ) \
                                @ self._dPhiPsi \
                                + np.diag(1/Te(r)) @ self._massMatrix)
        self._stiffnessM = sparse.csr_matrix(np.diag(1/r**2) @ self._massMatrix)
        self._interpolator = SplineInterpolator1D(rspline)
        self._spline = Spline1D(rspline,np.complex128)
        self._real_spline = Spline1D(rspline)
    
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
        stiffnessMatrix = self._stiffnessMatrix + m*m*self._stiffnessM
        
        for i,q in rho.getCoords(0):
            for j,z in rho.getCoords(1):
                self._interpolator.compute_interpolant(rho.get1DSlice([i,j]),self._spline)
                self._coeffs[self._coeff_range,0] = spsolve(stiffnessMatrix,self._massMatrix @ self._spline.coeffs[self._coeff_range])
                self._real_spline.coeffs[:,None] = np.real(self._coeffs)
                reals = self._real_spline.eval(rho.getCoordVals(2))
                self._real_spline.coeffs[:,None] = np.imag(self._coeffs)
                imags = self._real_spline.eval(rho.getCoordVals(2))
                phi.get1DSlice([i,j])[:] = reals+1j*imags
    
    def findPotential( self, phi: Grid ):
        assert(type(phi.get1DSlice([0,0])[0])==np.complex128)
        for i,r in phi.getCoords(0):
            for j,z in phi.getCoords(1):
                vec=phi.get1DSlice([i,j])
                mode=ifft(vec)
                vec[:]=mode
