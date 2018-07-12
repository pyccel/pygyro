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
    
    def getRho ( self, grid: Grid , rho: Grid ):
        """
        Get the particle density

        Parameters
        ----------
        grid : Grid
            A grid containing the values of the particle distribution
            function
        
        rho : Grid
            The grid in which the values of the particle density will be
            stored
        
        """
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
    """
    PoissonSolver: Class used to solve a poisson equation. It contains
    functions to handle the discrete Fourier transforms and a function
    which uses the finite elements method to solve the equation in Fourier
    space

    Parameters
    ----------
    eta_grid : list of array_like
        The coordinates of the grid points in each dimension
    
    degree : int
        The degree of the highest degree polynomial function which will
        be exactly integrated

    rspline : BSplines
        A spline along the r direction
    
    lNeumannIdx : list - optional
        The modes for which the boundary condition on the lower boundary
        should be Neumann rather than Dirichlet.
        Default is []
    
    uNeumannIdx : list - optional
        The modes for which the boundary condition on the upper boundary
        should be Neumann rather than Dirichlet.
        Default is []
    
    ddrFactor : float or array_like - optional
        The factor in front of the double derivative of the electric
        potential with respect to r.
        Default is -1
    
    drFactor : float or array_like - optional
        The factor in front of the derivative of the electric potential
        with respect to r.
        Default is kN0 * ( 1 + tanh( (r - rp)/deltaRN0 )**2 ) - 1/r
    
    rFactor : float or array_like - optional
        The factor in front of the electric potential
        Default is 1/Te(r)
    
    ddThetaFactor : float or array_like - optional
        The factor in front of the double derivative of the electric
        potential with respect to theta.
        Default is -1/r**2

    """
    def __init__( self, eta_grid: list, degree: int, rspline: BSplines,
                  lNeumannIdx: list = [], uNeumannIdx: list = [],*args,**kwargs):
        # Calculate the number of points required for the Gauss-Legendre
        # quadrature
        n=degree//2+1
        
        self._mVals = np.fft.fftfreq(eta_grid[1].size,1/eta_grid[1].size)
        
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
        
        if (rspline.nbasis > 4*rspline.degree):
            maxEnd = self._nUnknowns-1
            
            # If there is a spline basis function which only overlaps with
            # other spline basis functions of the same shape (i.e. no border
            # spline basis functions) then the construction can be optimised
            # by finding the values for that function then repeating those
            # values along the diagonals
            
            # Create the storage for the diagonal values
            massCoeffs = np.zeros((2*rspline.degree+1,))
            dPhidPsiCoeffs = np.zeros((2*rspline.degree+1,))
            dPhiPsiCoeffs = np.zeros((2*rspline.degree+1,))
            
            # Find the reference spline basis function
            refSpline = rspline[2*rspline.degree]
            
            # Find the domain of this spline
            start = rspline.degree
            end = 2*rspline.degree+1
            
            # Evaluate the spline and the derivative of the spline at
            # the points required for integration between each break point
            vals = refSpline.eval(self._evalPts[start:end].flatten())
            derivs = refSpline.eval(self._evalPts[start:end].flatten(),1)
            
            # Find the central diagonal
            massCoeffs[rspline.degree] = np.sum( np.tile(self._weights,rspline.degree+1) * \
                                                 multFactor * vals**2 )
            
            dPhidPsiCoeffs[rspline.degree] = np.sum( np.tile(self._weights,rspline.degree+1) * \
                                                 multFactor * derivs**2 )
            
            dPhiPsiCoeffs[rspline.degree] = np.sum( np.tile(self._weights,rspline.degree+1) * \
                                                 multFactor * derivs * vals )
            
            nPts = n*(rspline.degree+1)
            for i in range(1,rspline.degree+1):
                start_i = start + i
                
                diff=i*n
                
                # Find the i-th upper diagonal
                massCoeffs[rspline.degree+i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        vals[:nPts-diff] * vals[diff:] )
                dPhidPsiCoeffs[rspline.degree+i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        derivs[:nPts-diff] * derivs[diff:] )
                dPhiPsiCoeffs[rspline.degree+i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        vals[diff:] * derivs[:nPts-diff] )
                
                # Find the i-th lower diagonal
                dPhiPsiCoeffs[rspline.degree-i]=np.sum( np.tile(self._weights,end-start_i) * multFactor * \
                        vals[:nPts-diff] * derivs[diff:] )
                # Find the i-th lower diagonal (for the symmetric matrices)
                massCoeffs[rspline.degree-i]=massCoeffs[rspline.degree+i]
                dPhidPsiCoeffs[rspline.degree-i]=dPhidPsiCoeffs[rspline.degree+i]
            
            # Create the diagonal matrices
            # Diagonal matrices contain many 0 valued points so sparse
            # matrices can be used to reduce storage
            # Lil format is used to facilitate modifying the edge values
            self._massMatrix = sparse.diags(massCoeffs,range(-rspline.degree,rspline.degree+1),
                                         (self._nUnknowns,self._nUnknowns),'lil')
            self._dPhidPsi = sparse.diags(dPhidPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                         (self._nUnknowns,self._nUnknowns),'lil')
            self._dPhiPsi = sparse.diags(dPhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                         (self._nUnknowns,self._nUnknowns),'lil')
            
            # The values near the boundaries are not the same and must be handled separately
            
            # The first and last spline basis functions should only be
            # included if the associated boundary condition is a neumann
            # boundary condition
            start_range=1
            start_enum=0
            if (lBoundary!=uBoundary):
                # If only one boundary condition is neumann then the loop
                # for that spline is carried out separately
                if (lBoundary=='neumann'):
                    i=0
                    spline = rspline[i]
                    
                    for j in range(i,i+rspline.degree+1):
                        # Calculate the required values
                        self._massMatrix[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                    spline.eval(self._evalPts[:j+1].flatten()) )
                        self._dPhidPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                    spline.eval(self._evalPts[:j+1].flatten(),1) )
                        self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                rspline[j].eval(self._evalPts[:j+1].flatten(),1) * \
                                    spline.eval(self._evalPts[:j+1].flatten()) )
                        self._dPhiPsi[j,i]=np.sum( np.tile(self._weights,j+1) * multFactor * \
                                rspline[j].eval(self._evalPts[:j+1].flatten()) * \
                                    spline.eval(self._evalPts[:j+1].flatten(),1) )
                        
                        # Save the symmetric values
                        self._massMatrix[j,i]=self._massMatrix[i,j]
                        self._dPhidPsi[j,i]=self._dPhidPsi[i,j]
                    start_enum=1
                    maxEnd=maxEnd+1
                else:
                    idx_i=0
                    i=maxEnd
                    spline = rspline[idx_i]
                    
                    for idx_j in range(idx_i,idx_i+rspline.degree+1):
                        j=maxEnd-idx_j
                        # Calculate the required values
                        self._massMatrix[i,j]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                                rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten()) * \
                                    spline.eval(self._evalPts[:idx_j+1].flatten()) )
                        self._dPhidPsi[i,j]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                                rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten(),1) * \
                                    spline.eval(self._evalPts[:idx_j+1].flatten(),1) )
                        self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                                rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten(),1) * \
                                    spline.eval(self._evalPts[:idx_j+1].flatten()) )
                        self._dPhiPsi[j,i]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                                rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten()) * \
                                    spline.eval(self._evalPts[:idx_j+1].flatten(),1) )
                        
                        # Save the symmetric values
                        self._massMatrix[j,i]=self._massMatrix[i,j]
                        self._dPhidPsi[j,i]=self._dPhidPsi[i,j]
                    maxEnd=maxEnd-1
            elif (lBoundary=='neumann'):
                # If both boundary conditions are neumann then the first
                # and last basis spline can be handled in the same way
                # as the other splines
                start_range=0
            
            for i,idx_i in enumerate(range(start_range,rspline.degree+1+start_range),start_enum):
                spline = rspline[idx_i]
                
                for j,idx_j in enumerate(range(idx_i,idx_i+rspline.degree+1),i):
                    # Calculate the required values
                    self._massMatrix[i,j]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                            rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten()) * \
                                spline.eval(self._evalPts[:idx_j+1].flatten()) )
                    self._dPhidPsi[i,j]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                            rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten(),1) * \
                                spline.eval(self._evalPts[:idx_j+1].flatten(),1) )
                    self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                            rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten(),1) * \
                                spline.eval(self._evalPts[:idx_j+1].flatten()) )
                    self._dPhiPsi[j,i]=np.sum( np.tile(self._weights,idx_j+1) * multFactor * \
                            rspline[idx_j].eval(self._evalPts[:idx_j+1].flatten()) * \
                                spline.eval(self._evalPts[:idx_j+1].flatten(),1) )
                    
                    # Save the symmetric values
                    self._massMatrix[j,i]=self._massMatrix[i,j]
                    self._massMatrix[maxEnd-i,maxEnd-j]=self._massMatrix[i,j]
                    self._massMatrix[maxEnd-j,maxEnd-i]=self._massMatrix[i,j]
                    
                    self._dPhidPsi[j,i]=self._dPhidPsi[i,j]
                    self._dPhidPsi[maxEnd-i,maxEnd-j]=self._dPhidPsi[i,j]
                    self._dPhidPsi[maxEnd-j,maxEnd-i]=self._dPhidPsi[i,j]
                    
                    self._dPhiPsi[maxEnd-i,maxEnd-j]=self._dPhiPsi[j,i]
                    self._dPhiPsi[maxEnd-j,maxEnd-i]=self._dPhiPsi[i,j]
            
            # Convert the matrices to diagonal storage for better
            # addition/multiplication etc operations
            self._massMatrix=self._massMatrix.tocsc()
            self._dPhidPsi=self._dPhidPsi.tocsc()
            self._dPhidPsi=self._dPhidPsi.tocsc()
        else:
            # Create the storage for the values
            self._massMatrix = np.zeros((self._nUnknowns,self._nUnknowns))
            self._dPhidPsi = np.zeros((self._nUnknowns,self._nUnknowns))
            self._dPhiPsi = np.zeros((self._nUnknowns,self._nUnknowns))
            
            for i,s_i in enumerate(range(start_range,end_range)):
                # For each spline, find the spline and its domain
                spline = rspline[s_i]
                start_i = max(0,s_i-rspline.degree)
                end_i = min(rspline.ncells,s_i+1)
                
                for j,s_j in enumerate(range(s_i,end_range),i):
                    # Verify if it overlaps with any other splines
                    start_j = max(0,s_j-rspline.degree)
                    
                    if (start_j<end_i):
                        # For overlapping splines find the domain of the overlap
                        end_j = min(rspline.ncells,s_j+1)
                        start = max(start_i,start_j)
                        end = min(end_i,end_j)
                        
                        # Find the integral of the multiplication of these splines and
                        # save the value in the appropriate place in the matrix
                        self._massMatrix[i,j]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[s_j].eval(self._evalPts[start:end].flatten()) * \
                                    spline.eval(self._evalPts[start:end].flatten()) )
                        
                        self._dPhidPsi[i,j]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[s_j].eval(self._evalPts[start:end].flatten(),1) * \
                                    spline.eval(self._evalPts[start:end].flatten(),1) )
                        
                        self._dPhiPsi[i,j]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[s_j].eval(self._evalPts[start:end].flatten(),1) * \
                                    spline.eval(self._evalPts[start:end].flatten()) )
                        self._dPhiPsi[j,i]=np.sum( np.tile(self._weights,end-start) * multFactor * \
                                rspline[s_j].eval(self._evalPts[start:end].flatten()) * \
                                    spline.eval(self._evalPts[start:end].flatten(),1) )
                        
                        # Save the symmetric values
                        self._massMatrix[j,i]=self._massMatrix[i,j]
                        self._dPhidPsi[j,i]=self._dPhidPsi[i,j]
            
            # It is supposed that the matrices will usually be larger and
            # will therefore not be created by this command. The method
            # used elsewhere means that sparse diagonal matrices are
            # initialised here so the sparse commands don't throw errors
            self._massMatrix = sparse.csc_matrix(self._massMatrix)
            self._dPhidPsi = sparse.csc_matrix(self._dPhidPsi)
            self._dPhiPsi = sparse.csc_matrix(self._dPhiPsi)
        
        # Collect the factors in front of each element of the equation
        r = eta_grid[0]
        ddrFactor = kwargs.pop('ddrFactor',-1)
        drFactor = kwargs.pop('drFactor',-( 1/r - constants.kN0 * \
                                (1 - np.tanh( (r - constants.rp ) / \
                                              constants.deltaRN0 )**2 ) ))
        rFactor = kwargs.pop('rFactor',1/Te(r))
        ddqFactor = kwargs.pop('ddThetaFactor',-1/r**2)
        
        for name,value in kwargs.items():
            warnings.warn("{0} is not a recognised parameter for setupCylindricalGrid".format(name))
        
        # If the factors are vectors and the boundary is a dirichlet boundary
        # then the first value is not needed
        if (lNeumannIdx==[]):
            if (hasattr(ddrFactor,'__len__')):
                ddrFactor = ddrFactor[1:]
            if (hasattr(drFactor,'__len__')):
                drFactor = drFactor[1:]
            if (hasattr(rFactor,'__len__')):
                rFactor = rFactor[1:]
            if (hasattr(ddqFactor,'__len__')):
                ddqFactor = ddqFactor[1:]
        
        if (uNeumannIdx==[]):
            if (hasattr(ddrFactor,'__len__')):
                ddrFactor = ddrFactor[:-1]
            if (hasattr(drFactor,'__len__')):
                drFactor = drFactor[:-1]
            if (hasattr(rFactor,'__len__')):
                rFactor = rFactor[:-1]
            if (hasattr(ddqFactor,'__len__')):
                ddqFactor = ddqFactor[:-1]
        
        # Construct the part of the stiffness matrix which has no theta
        # dependencies
        self._stiffnessMatrix = sparse.csc_matrix(self._dPhidPsi.multiply(np.atleast_2d(-ddrFactor).T) \
                                + self._dPhiPsi.multiply(np.atleast_2d(drFactor).T) \
                                + self._massMatrix.multiply(np.atleast_2d(rFactor).T))
        
        assert(np.linalg.cond(self._stiffnessMatrix.todense())<1e10)
        
        # The part of the stiffness matrix relating to the double derivative
        # with respect to theta must be constructed separately as it will
        # be multiplied by the different modes
        self._stiffnessM = sparse.csc_matrix(self._massMatrix.multiply(np.atleast_2d(ddqFactor).T))
        
        # Create the tools required for the interpolation
        self._interpolator = SplineInterpolator1D(rspline)
        self._spline = Spline1D(rspline,np.complex128)
        self._real_spline = Spline1D(rspline)
    
    def getModes( self, rho: Grid ):
        """
        Get the Fourier transform of the particle density

        Parameters
        ----------
        rho : Grid
            A grid containing the values of the particle density
        
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
        Solve the Poisson equation where the electric potential is unknown.
        The equation is solved in Fourier space. The application of the
        Fourier transform and inverse Fourier transform is not handled 
        by this function
        
        Parameters
        ----------
        phi : Grid
            The grid in which the calculated values of the Fourier transform
            of the electric potential will be stored
        
        rho : Grid
            A grid containing the values of the the Fourier transform of
            the particle density
        
        """
        
        assert(rho.getLayout(rho.currentLayout).dims_order[-1]==0)
        
        for i,I in enumerate(rho.getGlobalIdxVals(0)):
            #m = i + rho.getLayout(rho.currentLayout).starts[0]-self._nq2
            # For each mode on this process, create the necessary matrix
            stiffnessMatrix = (self._stiffnessMatrix - self._mVals[I]*self._stiffnessM) \
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
        Get the inverse Fourier transform of the electric potential
        
        Parameters
        ----------
        phi : Grid
            A grid containing the values of the Fourier transform of the
            electric potential
        
        """
        
        assert(type(phi.get1DSlice([0,0])[0])==np.complex128)
        assert(phi.getLayout(phi.currentLayout).dims_order[-1]==1)
        
        for i,r in phi.getCoords(0):
            for j,z in phi.getCoords(1):
                vec=phi.get1DSlice([i,j])
                mode=ifft(vec,overwrite_x=True)
                vec[:]=mode
