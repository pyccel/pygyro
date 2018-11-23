from mpi4py import MPI
import numpy as np
from numpy.linalg                   import solve
from scipy.interpolate              import lagrange
from scipy.integrate                import trapz
from math                           import pi

from ..splines.splines              import BSplines, Spline1D, Spline2D
from ..splines.spline_interpolators import SplineInterpolator1D, SplineInterpolator2D
from ..splines                      import spline_eval_funcs as SEF
from ..initialisation.mod_initialiser_funcs   import fEq
from ..initialisation               import constants
from ..model.layout                 import Layout
from ..model.grid                   import Grid
from .                              import accelerated_advection_steps as AAS

if ('mod_pygyro_advection_accelerated_advection_steps' in dir(AAS)):
    AAS = AAS.mod_pygyro_advection_accelerated_advection_steps
    modFunc = np.transpose
else:
    modFunc = lambda x: x

if ('mod_pygyro_splines_spline_eval_funcs' in dir(SEF)):
    SEF = SEF.mod_pygyro_splines_spline_eval_funcs

def fieldline(theta,z_diff,iota,r):
    return np.mod(theta+iota(r)*z_diff/constants.R0,2*pi)

class ParallelGradient:
    """
    ParallelGradient: Class containing values necessary to derive a function
    along the direction parallel to the flux surface

    Parameters
    ----------
    spline : BSplines
        A spline along the theta direction

    eta_grid : list of array_like
        The coordinates of the grid points in each dimension

    iota : function handle - optional
        Function returning the value of iota at a given radius r.
        Default is constants.iota
    
    order : int - optional
        The order of the finite differences scheme that is used
        Default is 6

    """
    def __init__( self, spline: BSplines, eta_grid: list, layout: Layout, iota = constants.iota, order: int = 6 ):
        # Save z step
        self._dz = eta_grid[2][1]-eta_grid[2][0]
        # Save size in z direction
        self._nz = eta_grid[2].size
        self._nq = eta_grid[1].size
        
        # If there are too few points then the access cannot be optimised
        # at the boundaries in the way that has been used
        assert(self._nz>order)
        
        # Find the coefficients and shifts used to find the first derivative
        # of the correct order
        self.getCoeffsFirstDeriv(order+1)
        
        # Save the inverse as it is used multiple times
        self._inv_dz = 1.0/self._dz
        
        r = eta_grid[0][layout.starts[layout.inv_dims_order[0]] : \
                     layout.ends  [layout.inv_dims_order[0]]    ]
        
        # Determine bz
        self._bz = 1 / np.sqrt(1+(r * iota(r)/constants.R0)[:,None]**2)
        
        # Save the necessary spline and interpolator
        self._interpolator = SplineInterpolator1D(spline)
        self._thetaSpline = Spline1D(spline)
        
        # The positions at which the spline will be evaluated are always the same.
        # They can therefore be calculated in advance
        self._thetaVals = np.empty([eta_grid[0].size, self._nz, order+1, self._nq])
        for i,r in enumerate(eta_grid[0]):
            self._getThetaVals(r,self._thetaVals[i],eta_grid,iota)
    
    def getCoeffsFirstDeriv( self, n: int):
        b=np.zeros(n)
        b[1]=1
        
        start = 1-(n+1)//2
        # Create the shifts
        self._shifts = np.arange(n)+start
        # Save the number of forward and backward steps to avoid
        # unnecessary modulo operations
        self._fwdSteps = -start
        self._bkwdSteps = self._shifts[-1]
        
        # Create the matrix
        A=np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                A[i,j]=(j+start)**i
        
        # Solve the linear system to find the coefficients
        self._coeffs = solve(A,b)
    
    def _getThetaVals( self, r: float, thetaVals: np.ndarray, eta_grid: list, iota ):
        # The positions at which the spline will be evaluated are always the same.
        # They can therefore be calculated in advance
        n = eta_grid[2].size
        
        for k,z in enumerate(eta_grid[2]):
            for i,l in enumerate(self._shifts):
                thetaVals[(k+l)%n,i,:]=fieldline(eta_grid[1],self._dz*l,iota,r)
    
    def parallel_gradient( self, phi_r: np.ndarray, i : int, der: np.ndarray ):
        """
        Get the gradient of a function in the direction parallel to the
        flux surface

        Parameters
        ----------
        phi_r : array_like
            The values of the function at the nodes
        
        i : int
            The current index of r
        
        def : array_like
            Array which will contain the solution
        
        """
        # Get scalar values necessary for this slice
        bz=self._bz[i]
        thetaVals = self._thetaVals[i]
        assert(der.shape==phi_r.shape)
        der[:]=0
        
        # For each value of z interpolate the spline along theta and add
        # the value multiplied by the corresponding coefficient to the
        # derivative at the point at which it is required
        # This is split into three steps to avoid unnecessary modulo operations
        tmp = np.empty(self._nq)
        for i in range(self._fwdSteps):
            self._interpolator.compute_interpolant(phi_r[i,:],self._thetaSpline)
            for j,(s,c) in enumerate(zip(self._shifts,self._coeffs)):
                SEF.eval_spline_1d_vector(thetaVals[i,j,:],self._thetaSpline.basis.knots,
                                    self._thetaSpline.basis.degree,self._thetaSpline.coeffs,tmp,0)
                der[(i-s)%self._nz,:]+=c*tmp
        
        for i in range(self._fwdSteps,self._nz-self._bkwdSteps):
            self._interpolator.compute_interpolant(phi_r[i,:],self._thetaSpline)
            for j,(s,c) in enumerate(zip(self._shifts,self._coeffs)):
                SEF.eval_spline_1d_vector(thetaVals[i,j,:],self._thetaSpline.basis.knots,
                                    self._thetaSpline.basis.degree,self._thetaSpline.coeffs,tmp,0)
                der[(i-s),:]+=c*tmp
        
        for i in range(self._nz-self._bkwdSteps,self._nz):
            self._interpolator.compute_interpolant(phi_r[i,:],self._thetaSpline)
            for j,(s,c) in enumerate(zip(self._shifts,self._coeffs)):
                SEF.eval_spline_1d_vector(thetaVals[i,j,:],self._thetaSpline.basis.knots,
                                    self._thetaSpline.basis.degree,self._thetaSpline.coeffs,tmp,0)
                der[(i-s)%self._nz,:]+=c*tmp
        
        der*= ( bz * self._inv_dz )
        
        return der

class FluxSurfaceAdvection:
    """
    FluxSurfaceAdvection: Class containing information necessary to carry out
    an advection step along the flux surface.

    Parameters
    ----------
    eta_grid: list of array_like
        The coordinates of the grid points in each dimension

    splines: list of BSplines
        The spline approximations along theta and z
        
    layout: Layout
        The current layout
        
    dt: float
        Time-step

    iota: function handle - optional
        Function returning the value of iota at a given radius r.
        Default is constants.iota
    
    zDegree: int - optional
        Order of the lagrange interpolation

    """
    def __init__( self, eta_grid: list, splines: list, layout: Layout,
                  dt: float, iota = constants.iota, zDegree: int = 5 ):
        # Save all pertinent information
        self._zLagrangePts = zDegree+1
        self._points = eta_grid[1:3]
        self._nPoints = (self._points[0].size,self._points[1].size)
        self._interpolator = SplineInterpolator1D(splines[0])
        self._thetaSpline = Spline1D(splines[0])
        
        self._getLagrangePts(eta_grid,layout,dt,iota)
        self._LagrangeVals = np.ndarray([self._nPoints[1],self._nPoints[0], self._zLagrangePts])
    
    def _getLagrangePts( self, eta_grid: list, layout: Layout, dt: float, iota ):
        # Get z step
        dz = eta_grid[2][2]-eta_grid[2][1]
        
        r = eta_grid[0][layout.starts[layout.inv_dims_order[0]] : \
                     layout.ends  [layout.inv_dims_order[0]]    ]
        
        # Get theta step
        dtheta = (dz * iota(r) / constants.R0)[:,None,None]
        
        bz = 1 / np.sqrt(1+(r * iota(r)/constants.R0)[:,None,None]**2)
        
        nR = len(dtheta)
        nV = layout.shape[layout.inv_dims_order[3]]
        
        # Find the distance travelled in the z direction
        zDist = -eta_grid[3][layout.starts[layout.inv_dims_order[3]] : \
                             layout.ends  [layout.inv_dims_order[3]]   ][None,:,None] * \
                bz*dt
        
        # Find the number of steps between the start point and the lines
        # around the end point
        self._shifts = np.ndarray([nR, nV, self._zLagrangePts],dtype=int)
        self._shifts[:] = np.floor( zDist / dz ) + \
                    np.arange(-self._zLagrangePts//2+1,self._zLagrangePts//2+1)[None,None,:]
        
        # Find the corresponding shift in the theta direction
        self._thetaShifts = dtheta*self._shifts
        
        # Find the distance to the points used for the interpolation
        zPts = dz * self._shifts[:,:,:]
        
        # As we have a regular grid and constant advection the endpoint
        # from grid point z_i evaluated on the lagrangian basis spanning z_k:z_{k+6}
        # is the same as the endpoint from grid point z_{i+1} evaluated on the
        # lagrangian basis spanning z_{k+1}:z_{k+7}
        # Thus this evaluation only needs to be done once for each r value and v value
        # not for each z or phi values
        z = eta_grid[2][1]
        zPts = z+zPts
        zPos = z+zDist
        
        # The first barycentric formula is used to find the lagrange coefficients
        zDiff=zPos-zPts
        omega = np.prod(zDiff,axis=2)[:,:,None]
        lambdas = 1/np.prod(zPts[:,:,:,None]-zPts[:,:,None,:]+np.eye(self._zLagrangePts)[None,None,:,:],axis=3)
        
        # If the final position is one of the points then zDiff=0
        # The division by 0 must be avoided and the coefficients should
        # be equal to 0 except at the point where they equal 1
        zComp = (zPts==zPos)
        with np.errstate(invalid='ignore', divide='ignore'):
            self._lagrangeCoeffs=np.where(zComp,1,omega*lambdas/zDiff)
    
    def step( self, f: np.ndarray, cIdx: int, rIdx: int = 0 ):
        """
        Carry out an advection step for the flux parallel advection

        Parameters
        ----------
        f: array_like
            The current value of the function at the nodes.
            The result will be stored here
        
        cIdx: int
            Index of the advection parameter d_tf + c d_xf=0
        
        rIdx: int - optional
            The current index of r. Not necessary if iota does not depend on r
        
        """
        assert(f.shape==self._nPoints)
        
        # find the values of the function at each required point
        for i in range(self._nPoints[1]):
            self._interpolator.compute_interpolant(f[:,i],self._thetaSpline)
            
            AAS.get_lagrange_vals(i,self._nPoints[1],self._shifts[rIdx,cIdx],
                                modFunc(self._LagrangeVals),self._points[0],
                                self._thetaShifts[rIdx,cIdx],self._thetaSpline.basis.knots,
                                self._thetaSpline.basis.degree,
                                self._thetaSpline.coeffs)
            
            #~ for j,s in enumerate(self._shifts[rIdx,cIdx]):
                #~ self._LagrangeVals[(i-s)%self._nPoints[1],:,j] = \
                        #~ self._thetaSpline.eval((self._points[0]+self._thetaShifts[rIdx,cIdx,j])%(2*pi))
        
        AAS.flux_advection(*self._nPoints,modFunc(f),
                            self._lagrangeCoeffs[rIdx,cIdx],
                            modFunc(self._LagrangeVals))
        
        #~ for j in range(self._nPoints[0]):
            #~ for i in range(self._nPoints[1]):
                #~ f[j,i] = np.dot(self._lagrangeCoeffs[rIdx,cIdx],self._LagrangeVals[i,j])
    
    def gridStep( self, grid: Grid ):
        assert(grid.getLayout(grid.currentLayout).dims_order==(0,3,1,2))
        for i,r in grid.getCoords(0):
            for j,v in grid.getCoords(1):
                self.step(grid.get2DSlice([i,j]),j)

class VParallelAdvection:
    """
    VParallelAdvection: Class containing information necessary to carry out
    an advection step along the v-parallel surface.

    Parameters
    ----------
    eta_vals: list of array_like
        The coordinates of the grid points in each dimension

    splines: BSplines
        The spline approximations along v

    edgeFunc: function handle - optional
        Function returning the value at the boundary as a function of r and v
        Default is fEquilibrium

    """
    def __init__( self, eta_vals: list, splines: BSplines, edge : str = 'fEq' ):
        self._points = eta_vals[3]
        self._nPoints = (self._points.size,)
        self._interpolator = SplineInterpolator1D(splines)
        self._spline = Spline1D(splines)
        
        self._evalFunc = np.vectorize(self.evaluate, otypes=[np.float])
        if (edge=='fEq'):
            self._edgeType = 0
            self._edge = lambda r,v: fEq(r,v,constants.CN0,constants.kN0,
                                            constants.deltaRN0,constants.rp,
                                            constants.CTi,constants.kTi,
                                            constants.deltaRTi)
        elif (edge=='null'):
            self._edgeType = 1
            self._edge = lambda r,v: 0.0
        elif (edge=='periodic'):
            self._edgeType = 2
            minPt = self._points[0]
            width = self._points[-1]-self._points[0]
            self._edge = lambda r,v: self._spline.eval((v-minPt)%width + minPt)
        else:
            raise RuntimeError("V parallel boundary condition must be one of 'fEq', 'null', 'periodic'")
    
    def step( self, f: np.ndarray, dt: float, c: float, r: float ):
        """
        Carry out an advection step for the v-parallel advection

        Parameters
        ----------
        f: array_like
            The current value of the function at the nodes.
            The result will be stored here
        
        dt: float
            Time-step
        
        c: float
            Advection parameter d_tf + c d_xf=0
        
        r: float
            The radial coordinate
        
        """
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        AAS.v_parallel_advection_eval_step(f,self._points-c*dt,r,self._points[0],
                                        self._points[-1],self._spline.basis.knots,
                                        self._spline.basis.degree,self._spline.coeffs,
                                        constants.CN0,constants.kN0,
                                        constants.deltaRN0,constants.rp,
                                        constants.CTi,constants.kTi,
                                        constants.deltaRTi,self._edgeType)
        #~ f[:]=self._evalFunc(self._points-c*dt, r)
    
    def evaluate( self, v, r ):
        if (v<self._points[0] or v>self._points[-1]):
            return self._edge(r,v)
        else:
            return self._spline.eval(v)
    
    def gridStep( self, grid: Grid, phi: Grid, parGrad: ParallelGradient, parGradVals: np.array, dt: float):
        for i,r in grid.getCoords(0):
            parGrad.parallel_gradient(np.real(phi.get2DSlice([i])),i,parGradVals[i])
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    self.step(grid.get1DSlice([i,j,k]),dt,parGradVals[i,j,k],r)
    
    def gridStepKeepGradient( self, grid: Grid, parGradVals, dt: float):
        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    self.step(grid.get1DSlice([i,j,k]),dt,parGradVals[i,j,k],r)

class PoloidalAdvection:
    """
    PoloidalAdvection: Class containing information necessary to carry out
    an advection step along the poloidal surface.

    Parameters
    ----------
    eta_vals: list of array_like
        The coordinates of the grid points in each dimension

    splines: list of BSplines
        The spline approximations along theta and r
    
    edgeFunc: function handle - optional
        Function returning the value at the boundary as a function of r and v
        Default is fEquilibrium
    
    explicitTrap: bool - optional
        Indicates whether the explicit trapezoidal method (Heun's method)
        should be used or the implicit trapezoidal method should be used
        instead
    
    tol: float - optional
        The tolerance used for the implicit trapezoidal rule

    """
    def __init__( self, eta_vals: list, splines: list, nulEdge = False,
                    explicitTrap: bool =  True, tol: float = 1e-10 ):
        self._points = eta_vals[1::-1]
        self._shapedQ = np.atleast_2d(self._points[0]).T
        self._nPoints = (self._points[0].size,self._points[1].size)
        self._interpolator = SplineInterpolator2D(splines[0],splines[1])
        self._spline = Spline2D(splines[0],splines[1])
        
        self._explicit = explicitTrap
        self._TOL = tol
        
        self.evalFunc = np.vectorize(self.evaluate, otypes=[np.float])
        self._nulEdge=nulEdge
        if (nulEdge):
            self._edge = lambda r,v: 0.0
        else:
            self._edge = lambda r,v: fEq(x,y,constants.CN0,constants.kN0,
                                            constants.deltaRN0,constants.rp,
                                            constants.CTi,constants.kTi,
                                            constants.deltaRTi)
        
        self._drPhi_0 = np.empty(self._nPoints)
        self._dqPhi_0 = np.empty(self._nPoints)
        self._drPhi_k = np.empty(self._nPoints)
        self._dqPhi_k = np.empty(self._nPoints)
        self._endPts_k1_q = np.empty(self._nPoints)
        self._endPts_k1_r = np.empty(self._nPoints)
        self._endPts_k2_q = np.empty(self._nPoints)
        self._endPts_k2_r = np.empty(self._nPoints)
        
        self._max_loops = 0
        
        self._phiSplines = [Spline2D(splines[0],splines[1]) for i in range(eta_vals[2].size)]
    
    def step( self, f: np.ndarray, dt: float, phi: Spline2D, v: float ):
        """
        Carry out an advection step for the poloidal advection

        Parameters
        ----------
        f: array_like
            The current value of the function at the nodes.
            The result will be stored here
        
        dt: float
            Time-step
        
        phi: Spline2D
            Advection parameter d_tf + {phi,f}=0
        
        v: float
            The parallel velocity coordinate
        
        """
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        phiBases = phi.basis
        polBases = self._spline.basis

        if (self._explicit):
            AAS.poloidal_advection_step_expl( modFunc(f), dt, v, self._points[1],
                            self._points[0], self._nPoints, modFunc(self._drPhi_0),
                            modFunc(self._dqPhi_0), modFunc(self._drPhi_k),
                            modFunc(self._dqPhi_k), modFunc(self._endPts_k1_q),
                            modFunc(self._endPts_k1_r), modFunc(self._endPts_k2_q),
                            modFunc(self._endPts_k2_r), phiBases[0].knots,
                            phiBases[1].knots, modFunc(phi.coeffs),
                            phiBases[0].degree, phiBases[1].degree,
                            polBases[0].knots, polBases[1].knots,
                            modFunc(self._spline.coeffs),
                            polBases[0].degree, polBases[1].degree,constants.CN0,
                            constants.kN0,constants.deltaRN0,constants.rp,
                            constants.CTi,constants.kTi,constants.deltaRTi,
                            constants.B0,self._nulEdge)
        else:
            AAS.poloidal_advection_step_impl( modFunc(f), dt, v, self._points[1],
                            self._points[0], self._nPoints, modFunc(self._drPhi_0),
                            modFunc(self._dqPhi_0), modFunc(self._drPhi_k),
                            modFunc(self._dqPhi_k), modFunc(self._endPts_k1_q),
                            modFunc(self._endPts_k1_r), modFunc(self._endPts_k2_q),
                            modFunc(self._endPts_k2_r), phiBases[0].knots,
                            phiBases[1].knots, modFunc(phi.coeffs),
                            phiBases[0].degree, phiBases[1].degree,
                            polBases[0].knots, polBases[1].knots,
                            modFunc(self._spline.coeffs),
                            polBases[0].degree, polBases[1].degree,constants.CN0,
                            constants.kN0,constants.deltaRN0,constants.rp,
                            constants.CTi,constants.kTi,constants.deltaRTi,
                            constants.B0,self._TOL,self._nulEdge)
        
        """
        multFactor = dt/constants.B0
        
        drPhi_0 = phi.eval(*self._points,0,1)/self._points[1]
        dthetaPhi_0 = phi.eval(*self._points,1,0)/self._points[1]
        
        # Step one of Heun method
        # x' = x^n + f(x^n)
        endPts_k1 = ( self._shapedQ   -     drPhi_0*multFactor,
                     self._points[1] + dthetaPhi_0*multFactor )
        
        drPhi_k = np.empty_like(drPhi_0)
        dthetaPhi_k = np.empty_like(dthetaPhi_0)
        
        multFactor*=0.5
        
        while (True):
            
            for i in range(self._nPoints[0]):
                for j in range(self._nPoints[1]):
                    # Handle theta boundary conditions
                    while (endPts_k1[0][i,j]<0):
                        endPts_k1[0][i,j]+=2*pi
                    while (endPts_k1[0][i,j]>2*pi):
                        endPts_k1[0][i,j]-=2*pi
                    
                    if (not (endPts_k1[1][i,j]<self._points[1][0] or 
                             endPts_k1[1][i,j]>self._points[1][-1])):
                        # Add the new value of phi to the derivatives
                        # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                        #                               ^^^^^^^^^^^^^^^
                        drPhi_k[i,j]     = phi.eval(endPts_k1[0][i,j],endPts_k1[1][i,j],0,1)/endPts_k1[1][i,j]
                        dthetaPhi_k[i,j] = phi.eval(endPts_k1[0][i,j],endPts_k1[1][i,j],1,0)/endPts_k1[1][i,j]
                    else:
                        drPhi_k[i,j]     = 0.0
                        dthetaPhi_k[i,j] = 0.0
            
            if (self._explicit):
                # Step two of Heun method
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
                endPts_k2 = ( np.mod(self._shapedQ   - (drPhi_0     + drPhi_k)*multFactor,2*pi),
                              self._points[1] + (dthetaPhi_0 + dthetaPhi_k)*multFactor )
                break
            else:
                # Step two of Heun method
                # x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )

                # Clipping is one method of avoiding infinite loops due to boundary conditions
                # Using the splines to extrapolate is not sufficient
                endPts_k2 = ( np.mod(self._shapedQ   - (drPhi_0     + drPhi_k)*multFactor,2*pi),
                              np.clip(self._points[1] + (dthetaPhi_0 + dthetaPhi_k)*multFactor,
                                      self._points[1][0], self._points[1][-1]) )
                
                norm = max(np.linalg.norm((endPts_k2[0]-endPts_k1[0]).flatten(),np.inf),
                           np.linalg.norm((endPts_k2[1]-endPts_k1[1]).flatten(),np.inf))
                if (norm<self._TOL):
                    break
                endPts_k1=endPts_k2
        
        # Find value at the determined point
        for i,theta in enumerate(self._points[0]):
            for j,r in enumerate(self._points[1]):
                f[i,j]=self.evalFunc(endPts_k2[0][i,j],endPts_k2[1][i,j],v)
        """
    
    def exact_step( self, f, endPts, v ):
        assert(f.shape==self._nPoints)
        self._interpolator.compute_interpolant(f,self._spline)
        
        for i,theta in enumerate(self._points[0]):
            for j,r in enumerate(self._points[1]):
                f[i,j]=self.evalFunc(endPts[0][i,j],endPts[1][i,j],v)
    
    def evaluate( self, theta, r, v ):
        if (r<self._points[1][0]):
            return self._edge(self._points[1][0],v)
        elif (r>self._points[1][-1]):
            return self._edge(r,v)
        else:
            while (theta>2*pi):
                theta-=2*pi
            while (theta<0):
                theta+=2*pi
            return self._spline.eval(theta,r)
    
    def gridStep ( self, grid: Grid, phi: Grid, dt: float ):
        gridLayout = grid.getLayout(grid.currentLayout)
        phiLayout = phi.getLayout(grid.currentLayout)
        assert(gridLayout.dims_order[1:]==phiLayout.dims_order)
        assert(gridLayout.dims_order==(3,2,1,0))
        # Evaluate splines
        for j,z in grid.getCoords(1):
            self._interpolator.compute_interpolant(np.real(phi.get2DSlice([j])),self._phiSplines[j])
        # Do step
        for i,v in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                self.step(grid.get2DSlice([i,j]),dt,self._phiSplines[j],v)
    
    def gridStep_SplinesUnchanged ( self, grid: Grid, dt: float ):
        gridLayout = grid.getLayout(grid.currentLayout)
        assert(gridLayout.dims_order==(3,2,1,0))
        # Do step
        for i,v in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                self.step(grid.get2DSlice([i,j]),dt,self._phiSplines[j],v)
