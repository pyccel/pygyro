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
from ..initialisation               import initialiser
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
    def __init__ ( self, degree: int, spline: BSplines, eta_grid : list ):
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
        self._weights = np.tile(weights,len(starts))
        
        # Create the tools required for the interpolation
        self._interpolator = SplineInterpolator1D(spline)
        self._spline = Spline1D(spline)
        
        self._fEq = initialiser.fEq(eta_grid[0][:,None],self._points[None,:])
    
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
        assert(grid.getLayout(grid.currentLayout).dims_order==(0,2,1,3))
        assert(rho.getLayout(rho.currentLayout).dims_order==(0,2,1))
        
        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                rho_qv = rho.get1DSlice([i,j])
                for k,theta in grid.getCoords(2):
                    self._interpolator.compute_interpolant(grid.get1DSlice([i,j,k]),self._spline)
                    rho_qv[k] = np.sum(self._multFact*self._weights*(self._spline.eval(self._points)-self._fEq[i]))
    
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
    
    rhoFactor : function handle - optional
        The factor in front of the right hand side (E in the expression
        above)
        Default is: lambda r: 1

    """
    def __init__( self, degree: int, rspline: BSplines, nTheta: int,
                  lNeumannIdx: list = [], uNeumannIdx: list = [],
                  ddrFactor = lambda r: -1, drFactor = lambda r: 0,
                  rFactor = lambda r: 0, ddThetaFactor = lambda r: -1,
                  rhoFactor = lambda r: 1):
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
        
        # Ensure dirichlet boundaries are not used at both boundaries on
        # any mode
        poorlyDefined = [b for b in lNeumannIdx if b in uNeumannIdx]
        if (rFactor==(lambda r:0) and len(poorlyDefined)!=0):
            raise ValueError("Modes {0} are poorly defined as they use 0 Dirichlet boundary conditions".format(poorlyDefined))
        
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
        
        # The matrices are diagonal so the storage can be reduced if
        # they are not stored in a full matrix.
        # Create the storage for the diagonal values
        massCoeffs = [np.zeros(rspline.nbasis-np.abs(i)) for i in range(-rspline.degree,1)]
        # By extending with references to the lower diagonals the symmetrical
        # nature of the matrix will be programatically ensured which
        # allows the matrix to be built while only referring to upper diagonals
        massCoeffs.extend(massCoeffs[-2::-1])
        k2PhiPsiCoeffs = [np.zeros(rspline.nbasis-np.abs(i)) for i in range(-rspline.degree,1)]
        k2PhiPsiCoeffs.extend(k2PhiPsiCoeffs[-2::-1])
        PhiPsiCoeffs = [np.zeros(rspline.nbasis-np.abs(i)) for i in range(-rspline.degree,1)]
        PhiPsiCoeffs.extend(PhiPsiCoeffs[-2::-1])
        dPhidPsiCoeffs = [np.zeros(rspline.nbasis-np.abs(i)) for i in range(-rspline.degree,rspline.degree+1)]
        dPhiPsiCoeffs = [np.zeros(rspline.nbasis-np.abs(i)) for i in range(-rspline.degree,rspline.degree+1)]
        
        for i in range(rspline.nbasis):
            # For each spline, find the spline and its domain
            spline = rspline[i]
            start_i = max(0,i-rspline.degree)
            end_i = min(rspline.ncells,i+1)
            
            for j,s_j in enumerate(range(i,min(i+rspline.degree+1,rspline.nbasis)),rspline.degree):
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
                        rhoFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts) * evalPts )
                k2PhiPsiCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        ddThetaFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts) * evalPts )
                PhiPsiCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        rFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts) * evalPts )
                dPhidPsi = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        -ddrFactor(evalPts) * rspline[s_j].eval(evalPts,1) * spline.eval(evalPts,1) * evalPts )
                dPhidPsiCoeffs[j][i] = dPhidPsi + \
                        np.sum( np.tile(self._weights,end-start) * multFactor * \
                        -ddrFactor(evalPts) * rspline[s_j].eval(evalPts,1) * spline.eval(evalPts) )
                dPhidPsiCoeffs[rspline.degree*2-j][i] = dPhidPsi + \
                        np.sum( np.tile(self._weights,end-start) * multFactor * \
                        -ddrFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts,1) )
                dPhiPsiCoeffs[j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        drFactor(evalPts) * rspline[s_j].eval(evalPts,1) * spline.eval(evalPts) * evalPts )
                dPhiPsiCoeffs[rspline.degree*2-j][i] = np.sum( np.tile(self._weights,end-start) * multFactor * \
                        drFactor(evalPts) * rspline[s_j].eval(evalPts) * spline.eval(evalPts,1) * evalPts )
        
        # Create the diagonal matrices
        # Diagonal matrices contain many 0 valued points so sparse
        # matrices can be used to reduce storage
        # Csc format is used to allow slicing
        self._massMatrix = sparse.diags(massCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (rspline.nbasis,rspline.nbasis),'csc')[start_range:end_range,:]
        self._k2PhiPsi = sparse.diags(k2PhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (rspline.nbasis,rspline.nbasis),'csc')[start_range:end_range,start_range:end_range]
        self._PhiPsi = sparse.diags(PhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (rspline.nbasis,rspline.nbasis),'csc')[start_range:end_range,start_range:end_range]
        self._dPhidPsi = sparse.diags(dPhidPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (rspline.nbasis,rspline.nbasis),'csc')[start_range:end_range,start_range:end_range]
        self._dPhiPsi = sparse.diags(dPhiPsiCoeffs,range(-rspline.degree,rspline.degree+1),
                                     (rspline.nbasis,rspline.nbasis),'csc')[start_range:end_range,start_range:end_range]
        
        # Construct the part of the stiffness matrix which has no theta
        # dependencies
        self._stiffnessMatrix = self._dPhidPsi + self._dPhiPsi + self._PhiPsi
        
        # Create the tools required for the interpolation
        self._interpolator = SplineInterpolator1D(rspline,dtype=np.complex)
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
        assert(rho.getLayout(rho.currentLayout).dims_order==(0,2,1))
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
            # For each mode on this process, create the necessary matrix
            stiffnessMatrix = (self._stiffnessMatrix - self._mVals[I]*self._k2PhiPsi) \
                                    [self._stiffness_range[I],self._stiffness_range[I]]
            
            # Set Dirichlet boundary conditions
            # In the case of Neumann boundary conditions these values
            # will be overwritten
            self._coeffs[0] = 0
            self._coeffs[-1] = 0
            
            self._solveMode(phi,rho,stiffnessMatrix,i,I)
    
    def solveEquationForFunction( self, phi: Grid, rho ):
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
        
        rho : function handle
            A function returning the values on the the right hand side of the
            differential equation.
        
        """
        
        for i,I in enumerate(phi.getGlobalIdxVals(0)):
            # For each mode on this process, create the necessary matrix
            stiffnessMatrix = (self._stiffnessMatrix - self._mVals[I]*self._k2PhiPsi) \
                                    [self._stiffness_range[I],self._stiffness_range[I]]
            
            # Set Dirichlet boundary conditions
            # In the case of Neumann boundary conditions these values
            # will be overwritten
            self._coeffs[0] = 0
            self._coeffs[-1] = 0
            
            self._solveModeFunc(phi,rho,stiffnessMatrix,i,I)
    
    def _solveMode(self, phi: Grid, rho: Grid, stiffnessMatrix: sparse.csc.csc_matrix, i: int, I: int):
        massMat = self._massMatrix[self._stiffness_range[I],:]
        coeffs= self._coeffs[self._coeff_range[I]]
        for j,z in rho.getCoords(1):
            # Calculate the coefficients related to rho
            self._interpolator.compute_interpolant(rho.get1DSlice([i,j]),self._spline)
            
            # Save the solution to the preprepared buffer
            # The boundary values of this buffer are already set if
            # dirichlet boundary conditions are used
            coeffs[:] = spsolve(stiffnessMatrix, massMat.dot(self._spline.coeffs))
            
            # Find the values at the greville points by interpolating
            # the real and imaginary parts of the coefficients individually
            self._real_spline.coeffs[:] = np.real(self._coeffs)
            reals = self._real_spline.eval(rho.getCoordVals(2))
            
            self._real_spline.coeffs[:] = np.imag(self._coeffs)
            imags = self._real_spline.eval(rho.getCoordVals(2))
            
            phi.get1DSlice([i,j])[:] = reals+1j*imags
    
    def _solveModeFunc(self, phi: Grid, rho, stiffnessMatrix: sparse.csc.csc_matrix, i: int, I: int):
        coeffs  = self._coeffs[self._coeff_range[I]]
        
        rhoVec = np.zeros(self._rspline.greville.size)
        
        for j in range(self._rspline.nbasis):
            rhoVec[j]=np.sum(np.tile(self._weights,len(self._evalPts))*self._multFactor \
                            * self._rspline[j].eval(self._evalPts.flatten()) \
                            * rho(self._evalPts.flatten()))
        
        for j,z in phi.getCoords(1):
            # Save the solution to the preprepared buffer
            # The boundary values of this buffer are already set if
            # dirichlet boundary conditions are used
            coeffs[:] = spsolve(stiffnessMatrix, rhoVec[1:-1])
            
            # Find the values at the greville points by interpolating
            # the real and imaginary parts of the coefficients individually
            self._real_spline.coeffs[:] = np.real(self._coeffs)
            reals = self._real_spline.eval(phi.getCoordVals(2))
            
            self._real_spline.coeffs[:] = np.imag(self._coeffs)
            imags = self._real_spline.eval(phi.getCoordVals(2))
            
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
        assert(phi.getLayout(phi.currentLayout).dims_order==(0,2,1))
        
        for i,r in phi.getCoords(0):
            for j,z in phi.getCoords(1):
                vec=phi.get1DSlice([i,j])
                mode=ifft(vec,overwrite_x=True)
                vec[:]=mode

class QuasiNeutralitySolver(DiffEqSolver):
    """
    QuasiNeutralitySolver: Class used to solve the equation resulting
    from the quasi-neutrality equation. It contains
    functions to handle the discrete Fourier transforms and a function
    which uses the finite elements method to solve the equation in Fourier
    space
    
    -[d_r^2+( 1/r+g'(r)/g(r) )d_r+1/r^2 d_\theta^2]\phi 
    + 1/(g\lambda_D^2) [\phi-\chi \langle\phi\rangle_\theta]=rho/g
    
    Parameters
    ----------
    eta_grid : list of array_like
        The coordinates of the grid points in each dimension
    
    degree : int
        The degree of the highest degree polynomial function which will
        be exactly integrated

    rspline : BSplines
        A spline along the r direction
    
    adiabaticElectrons : bool - optional
        Indicates whether the electrons are considered to have an adiabatic
        response. If this is false then it is assumed 1/\lambda_D^2=0,
        the electrons are a kinetic species and their contribution will
        appear in the perturbed density
        Default is True
    
    chi: int
        The parameter indicated in the equation.
        Its value is either 0 or 1.
        This parameter is required if adiabaticElectrons = True otherwise
        it will be ignored.
    
    n0 : function handle - optional
        The ion/electron density at equilibrium
        Default is initialiser.n0
    
    B: float - optional
        The intensity of the equilibrium magnetic field
        Default is 1
    
    n0derivNormalised: function handle - optional
        n_0'(r)/n_0(r) where n_0(r) is the ion/electron density at equilibrium
        Default is initialiser.n0derivNormalised
    
    n0deriv: function handle - optional
        n_0'(r) where n_0(r) is the ion/electron density at equilibrium
        If n0derivNormalised is provided this will be ignored
        Default is initialiser.n0derivNormalised*n0
    
    Te: function handle - optional
        The temperature of the electrons
        This parameter will be ignored if adiabaticElectrons = False.
        Default is initialiser.Te

    """
    def __init__( self, eta_grid: list, degree: int, rspline: BSplines,
                    adiabaticElectrons: bool = True,n0 = initialiser.n0,
                    B: float = 1.0, Te = initialiser.Te, **kwargs):
        r = eta_grid[0]
        
        if ('n0derivNormalised' in kwargs):
            n0derivNormalised = kwargs.pop('n0derivNormalised',initialiser.n0derivNormalised)
        elif ('n0deriv' in kwargs):
            n0derivNormalised = lambda r:kwargs.pop('n0deriv')(r)/n0(r)
        else:
            n0derivNormalised = initialiser.n0derivNormalised
        
        if (not adiabaticElectrons):
            DiffEqSolver.__init__(self,degree,rspline,eta_grid[1].size,
                        drFactor = lambda r: -(1/r+ n0derivNormalised(r)),
                        ddThetaFactor = lambda r:-1/r**2,
                        rhoFactor = lambda r:B*B/n0(r),
                        lNeumannIdx=[0])
            
            self._stiffness0 = self._stiffnessMatrix
        else:
            assert('chi' in kwargs)
            chi = kwargs.pop('chi')
            
            DiffEqSolver.__init__(self,degree,rspline,eta_grid[1].size,
                        drFactor = lambda r: -(1/r+ n0derivNormalised(r)),
                        ddThetaFactor = lambda r:-1/r**2,
                        rFactor = lambda r: B*B/Te(r),
                        rhoFactor = lambda r:B*B/n0(r),
                        lNeumannIdx=[0])
            
            if (chi==0):
                self._stiffness0 = self._stiffnessMatrix
            elif (chi==1):
                self._stiffness0 = self._dPhidPsi + self._dPhiPsi
            else:
                raise ValueError("The argument chi must be either 0 or 1")
    
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
            if (self._mVals[I]==0):
                stiffnessMatrix = self._stiffness0
            else:
                stiffnessMatrix = (self._stiffnessMatrix - self._mVals[I]*self._k2PhiPsi) \
                                    [self._stiffness_range[I],self._stiffness_range[I]]
            
            # Set Dirichlet boundary conditions
            # In the case of Neumann boundary conditions these values
            # will be overwritten
            self._coeffs[0] = 0
            self._coeffs[-1] = 0
            
            self._solveMode(phi,rho,stiffnessMatrix,i,I)
