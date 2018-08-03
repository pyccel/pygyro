from scipy.integrate                import fixed_quad, quadrature   # fixed_quad = fixed order
                                                                    # quadrature = fixed tolerance
from scipy.fftpack                  import fft,ifft
import scipy.sparse                 as sparse
from scipy.sparse.linalg            import spsolve
import numpy                        as np
from numpy.polynomial.legendre      import leggauss
import warnings

from ..model.grid                   import Grid
from ..initialisation               import constants
from ..initialisation.initialiser   import fEq, Te, n0
from ..splines.splines              import BSplines, Spline1D
from ..splines.spline_interpolators import SplineInterpolator1D

class DensityFinder:
    """
    DensityFinder: Class used to integrate the particle distribution
    function along the v parallel direction in order to find the particle
    density

    Parameters
    ----------
    degree : int
        The degree of the highest degree polynomial function which will
        be exactly integrated

    spline : BSplines
        A spline along the v parallel direction

    """
    def __init__ ( self, degree: int, spline: BSplines ):
        # Calculate the number of points required for the Gauss-Legendre
        # quadrature
        n=degree//2+1
        
        # Calculate the points and weights required for the Gauss-Legendre
        # quadrature
        points,weights = leggauss(n)
        
        # Calculate the values used for the Gauss-Legendre quadrature
        # over the required domain
        breaks = spline.breaks
        starts = (breaks[:-1]+breaks[1:])/2
        self._multFact = (breaks[1]-breaks[0])/2
        self._points = np.repeat(starts,n) + np.tile(self._multFact*points,len(starts))
        self._weights = np.tile(points,len(starts))
        
        # Create the tools required for the interpolation
        self._interpolator = SplineInterpolator1D(spline)
        self._spline = Spline1D(spline)
    
    def getPerturbedRho ( self, grid: Grid , rho: Grid ):
        """
        Get the perturbed particle density

        Parameters
        ----------
        grid : Grid
            A grid containing the values of the particle distribution
            function
        
        rho : Grid
            The grid in which the values of the perturbed particle
            density will be stored
        
        """
        assert(grid.currentLayout=="v_parallel")
        assert(rho.currentLayout=="v_parallel")
        
        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                rho_qv = rho.get1DSlice([i,j])
                for k,theta in grid.getCoords(2):
                    self._interpolator.compute_interpolant(grid.get1DSlice([i,j,k]),self._spline)
                    rho_qv[k] = np.sum(self._multFact*self._weights*(self._spline.eval(self._points)-fEq(r,self._points)))
    
class DiffEqSolver:
    """
    DiffEqSolver: Class used to solve a differential equation of the form:
    
        A \partial_r^2 \phi + B \partial_r \phi + C \phi
        + D \partial_\theta^2 \phi = E \rho
    
    It contains functions to handle the discrete Fourier transforms and
    a function which uses the finite elements method to solve the equation
    in Fourier space

    Parameters
    ----------
    degree : int
        The degree of the highest degree polynomial function which will
        be exactly integrated

    rspline : BSplines
        A spline along the r direction
    
    nTheta : int
        The number of points in the periodic direction
    
    lNeumannIdx : list - optional
        The modes for which the boundary condition on the lower boundary
        should be Neumann rather than Dirichlet.
        Default is []
    
    uNeumannIdx : list - optional
        The modes for which the boundary condition on the upper boundary
        should be Neumann rather than Dirichlet.
        Default is []
    
    ddrFactor : function handle - optional
        The factor in front of the double derivative of the unknown with
        respect to r. (A in the expression above)
        Default is: lambda r: -1
    
    drFactor : function handle - optional
        The factor in front of the derivative of the unknown with respect
        to r. (B in the expression above)
        Default is: lambda r: 0
    
    rFactor : function handle - optional
        The factor in front of the unknown. (C in the expression above)
        Default is: lambda r: 0
    
    ddThetaFactor : function handle - optional
        The factor in front of the double derivative of the unknown with
        respect to theta. (D in the expression above)
        Default is: lambda r: -1
    
    rhoFactor : 

    """
    def __init__( self, degree: int, rspline: BSplines, nTheta: int,
                  lNeumannIdx: list = [], uNeumannIdx: list = [],*args,**kwargs):
        # Calculate the number of points required for the Gauss-Legendre
        # quadrature
        n=degree//2+1
        
        self._mVals = np.fft.fftfreq(nTheta,1/nTheta)
        
        # Calculate the points and weights required for the Gauss-Legendre
        # quadrature over the required domain
        points,self._weights = leggauss(n)
        multFactor = (rspline.breaks[1]-rspline.breaks[0])*0.5
        startPoints = (rspline.breaks[1:]+rspline.breaks[:-1])*0.5
        self._evalPts = startPoints[:,None]+points[None,:]*multFactor
        
        # Initialise the memory used for the calculated result
        self._coeffs = np.empty([rspline.nbasis],np.complex128)
        
        # If dirichlet boundary conditions are used then assign the
        # required value on the boundary
        if (lNeumannIdx==[]):
            start_range = 1
            lBoundary = 'dirichlet'
        else:
            start_range = 0
            lBoundary = 'neumann'
        if (uNeumannIdx==[]):
            end_range = rspline.nbasis-1
            excluded_end_pts = 1
            uBoundary = 'dirichlet'
        else:
            end_range = rspline.nbasis
            excluded_end_pts = 0
            uBoundary = 'neumann'
        
        # Calculate the size of the matrices required
        self._nUnknowns = end_range-start_range
        
        # Save the access pattern for the calculated result.
        # This ensures that the boundary conditions remain intact
        self._coeff_range = [slice(                  0 if i in lNeumannIdx else 1,
                                   rspline.nbasis - (0 if i in uNeumannIdx else 1))
                             for i in self._mVals]
        
        self._stiffness_range = [slice(0 if i in lNeumannIdx else (1-start_range),
                                   self._nUnknowns - (0 if i in uNeumannIdx else (1-excluded_end_pts)))
                             for i in self._mVals]
        
        self._mVals*=self._mVals
        
        # Collect the factors in front of each element of the equation
        ddrFactor = kwargs.pop('ddrFactor',lambda r: -1)
        drFactor = kwargs.pop('drFactor',lambda r: 0)
        rFactor = kwargs.pop('rFactor',lambda r: 0)
        ddqFactor = kwargs.pop('ddThetaFactor',lambda r: -1)
        rhoFactor = kwargs.pop('rhoFactor',lambda r: 1)
        
        # The matrices are diagonal so the storage can be reduced if
        # they are not stored in a full matrix.
        # Create the storage for the diagonal values
        massCoeffs = [np.zeros(self._nUnknowns-np.abs(i)) for i in range(-rspline.degree,1)]
        # By extending with references to the lower diagonals the symmetrical
        # nature of the matrix will be programatically ensured which
        # allows the matrix to be built while only referring to upper diagonals
        massCoeffs.extend(massCoeffs[-2::-1])
        k2PhiPsiCoeffs = [np.zeros(self._nUnknowns-np.abs(i)) for i in range(-rspline.degree,1)]
        k2PhiPsiCoeffs.extend(k2PhiPsiCoeffs[-2::-1])
        PhiPsiCoeffs = [np.zeros(self._nUnknowns-np.abs(i)) for i in range(-rspline.degree,1)]
        PhiPsiCoeffs.extend(PhiPsiCoeffs[-2::-1])
        dPhidPsiCoeffs = [np.zeros(self._nUnknowns-np.abs(i)) for i in range(-rspline.degree,1)]
        dPhidPsiCoeffs.extend(dPhidPsiCoeffs[-2::-1])
        dPhiPsiCoeffs = [np.zeros(self._nUnknowns-np.abs(i)) for i in range(-rspline.degree,rspline.degree+1)]
        
        for i,s_i in enumerate(range(start_range,end_range)):
            # For each spline, find the spline and its domain
            spline = rspline[s_i]
            start_i = max(0,s_i-rspline.degree)
            end_i = min(rspline.ncells,s_i+1)
            
            for j,s_j in enumerate(range(s_i,min(s_i+rspline.degree+1,end_range)),rspline.degree):
                # For overlapping splines find the domain of the overlap
                start_j = max(0,s_j-rspline.degree)
                end_j = min(rspline.ncells,s_j+1)
                start = max(start_i,start_j)
                end = min(end_i,end_j)
                
                evalPts=self._evalPts[start:end].flatten()
                
                # Find the integral of the multiplication of these splines
                # and their coefficients and save the value in the
                # appropriate place for the matrix
                massCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        rhoFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts) )
                k2PhiPsiCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        ddqFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts) )
                PhiPsiCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        rFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts) )
                dPhidPsiCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        -ddrFactor(evalPts) * rspline[s_j].eval(evalPts,1) * spline.eval(evalPts,1) )
                dPhiPsiCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        drFactor(evalPts) * rspline[s_j].eval(evalPts,1) * spline.eval(evalPts) )
                dPhiPsiCoeffs[rspline.degree*2-j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        drFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts,1) )
        
        # Create the diagonal matrices
        # Diagonal matrices contain many 0 valued points so sparse
        # matrices can be used to reduce storage
        # Csc format is used to allow slicing
        self._massMatrix = sparse.diags(massCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (self._nUnknowns,self._nUnknowns),'csc')
        self._k2PhiPsi = sparse.diags(k2PhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (self._nUnknowns,self._nUnknowns),'csc')
        self._PhiPsi = sparse.diags(PhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (self._nUnknowns,self._nUnknowns),'csc')
        self._dPhidPsi = sparse.diags(dPhidPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (self._nUnknowns,self._nUnknowns),'csc')
        self._dPhiPsi = sparse.diags(dPhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (self._nUnknowns,self._nUnknowns),'csc')
        
        for name,value in kwargs.items():
            warnings.warn("{0} is not a recognised parameter for PoissonSolver".format(name))
        
        # Construct the part of the stiffness matrix which has no theta
        # dependencies
        self._stiffnessMatrix = self._dPhidPsi + self._dPhiPsi + self._PhiPsi
        
        assert(np.linalg.cond(self._stiffnessMatrix.todense())<1e10)
        
        # Create the tools required for the interpolation
        self._interpolator = SplineInterpolator1D(rspline)
        self._spline = Spline1D(rspline,np.complex128)
        self._real_spline = Spline1D(rspline)
    
    def getModes( self, rho: Grid ):
        """
        Get the Fourier transform of the right hand side of the
        differential equation.

        Parameters
        ----------
        rho : Grid
            A grid containing the values of the right hand side of the
            differential equation.
        
        """
        assert(type(rho.get1DSlice([0,0])[0])==np.complex128)
        assert(rho.getLayout(rho.currentLayout).dims_order[-1]==1)
        for i,r in rho.getCoords(0):
            for j,z in rho.getCoords(1):
                vec=rho.get1DSlice([i,j])
                mode=fft(vec,overwrite_x=True)
                vec[:]=mode
    
    def solveEquation( self, phi: Grid, rho: Grid ):
        """
        Solve the differential equation.
        The equation is solved in Fourier space. The application of the
        Fourier transform and inverse Fourier transform is not handled 
        by this function
        
        Parameters
        ----------
        phi : Grid
            The grid in which the calculated values of the Fourier transform
            of the unknown will be stored
        
        rho : Grid
            A grid containing the values of the the right hand side of the
            differential equation.
        
        """
        
        assert(rho.getLayout(rho.currentLayout).dims_order[-1]==0)
        
        for i,I in enumerate(rho.getGlobalIdxVals(0)):
            #m = i + rho.getLayout(rho.currentLayout).starts[0]-self._nq2
            # For each mode on this process, create the necessary matrix
            stiffnessMatrix = (self._stiffnessMatrix - self._mVals[I]*self._k2PhiPsi) \
                                    [self._stiffness_range[I],self._stiffness_range[I]]
            
            # Set Dirichlet boundary conditions
            # In the case of Neumann boundary conditions these values
            # will be overwritten
            self._coeffs[0] = 0
            self._coeffs[-1] = 0
            
            for j,z in rho.getCoords(1):
                # Calculate the coefficients related to rho
                self._interpolator.compute_interpolant(rho.get1DSlice([i,j]),self._spline)
                
                # Save the solution to the preprepared buffer
                # The boundary values of this buffer are already set if
                # dirichlet boundary conditions are used
                self._coeffs[self._coeff_range[I]] = spsolve(stiffnessMatrix, \
                                                            self._massMatrix[self._stiffness_range[I],self._stiffness_range[I]] @ \
                                                            self._spline.coeffs[self._coeff_range[I]])
                
                # Find the values at the greville points by interpolating
                # the real and imaginary parts of the coefficients individually
                self._real_spline.coeffs[:] = np.real(self._coeffs)
                reals = self._real_spline.eval(rho.getCoordVals(2))
                
                self._real_spline.coeffs[:] = np.imag(self._coeffs)
                imags = self._real_spline.eval(rho.getCoordVals(2))
                
                phi.get1DSlice([i,j])[:] = reals+1j*imags
    
    def findPotential( self, phi: Grid ):
        """
        Get the inverse Fourier transform of the (solved) unknown
        
        Parameters
        ----------
        phi : Grid
            A grid containing the values of the Fourier transform of the
            (solved) unknown
        
        """
        
        assert(type(phi.get1DSlice([0,0])[0])==np.complex128)
        assert(phi.getLayout(phi.currentLayout).dims_order[-1]==1)
        
        for i,r in phi.getCoords(0):
            for j,z in phi.getCoords(1):
                vec=phi.get1DSlice([i,j])
                mode=ifft(vec,overwrite_x=True)
                vec[:]=mode

