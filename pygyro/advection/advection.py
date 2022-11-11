import json
import numpy as np
import os
from numpy.linalg import solve
from math import pi
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags

from ..arakawa.discrete_brackets_polar import assemble_bracket_arakawa, assemble_row_columns_akw_bracket_4th_order_extrapolation, update_bracket_4th_order_dirichlet_extrapolation
from ..splines.splines import BSplines, Spline1D, Spline2D
from ..splines.spline_interpolators import SplineInterpolator1D, SplineInterpolator2D
from ..splines.spline_eval_funcs import eval_spline_1d_vector
from ..model.layout import Layout
from ..model.grid import Grid
from .accelerated_advection_steps import get_lagrange_vals, flux_advection, \
    v_parallel_advection_eval_step, \
    poloidal_advection_step_expl, \
    poloidal_advection_step_impl
from ..initialisation.initialiser_funcs import f_eq


def fieldline(theta, z_diff, iota, r, R0):
    """
    TODO
    """
    return np.mod(theta+iota(r)*z_diff/R0, 2*pi)


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

    constants : Constant class
        Class containing all the constants

    order : int - optional
        The order of the finite differences scheme that is used
        Default is 6

    """

    def __init__(self, spline: BSplines, eta_grid: list, layout: Layout, constants, order: int = 6):
        # Save z step
        self._dz = eta_grid[2][1]-eta_grid[2][0]
        # Save size in z direction
        self._nz = eta_grid[2].size
        self._nq = eta_grid[1].size

        # If there are too few points then the access cannot be optimised
        # at the boundaries in the way that has been used
        assert self._nz > order

        # Find the coefficients and shifts used to find the first derivative
        # of the correct order
        self.getCoeffsFirstDeriv(order+1)

        # Save the inverse as it is used multiple times
        self._inv_dz = 1.0/self._dz

        r = eta_grid[0][layout.starts[layout.inv_dims_order[0]]:
                        layout.ends[layout.inv_dims_order[0]]]

        # Determine bz
        self._bz = 1 / \
            np.sqrt(1+(r * constants.iota(r)/constants.R0)[:, None]**2)

        # Save the necessary spline and interpolator
        self._interpolator = SplineInterpolator1D(spline)
        self._thetaSpline = Spline1D(spline)

        # The positions at which the spline will be evaluated are always the same.
        # They can therefore be calculated in advance
        self._thetaVals = np.empty(
            [eta_grid[0].size, self._nz, order+1, self._nq])
        for i, r in enumerate(eta_grid[0]):
            self._getThetaVals(
                r, self._thetaVals[i], eta_grid, constants.iota, constants.R0)

    def getCoeffsFirstDeriv(self, n: int):
        """
        TODO
        """
        b = np.zeros(n)
        b[1] = 1

        start = 1-(n+1)//2
        # Create the shifts
        self._shifts = np.arange(n)+start
        # Save the number of forward and backward steps to avoid
        # unnecessary modulo operations
        self._fwdSteps = -start
        self._bkwdSteps = self._shifts[-1]

        # Create the matrix
        A = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                A[i, j] = (j+start)**i

        # Solve the linear system to find the coefficients
        self._coeffs = solve(A, b)

    def _getThetaVals(self, r: float, thetaVals: np.ndarray, eta_grid: list, iota, R0):
        """
        TODO
        """
        # The positions at which the spline will be evaluated are always the same.
        # They can therefore be calculated in advance
        n = eta_grid[2].size

        for k in range(eta_grid[2].size):
            for i, l in enumerate(self._shifts):
                thetaVals[(k+l) % n, i, :] = fieldline(eta_grid[1],
                                                       self._dz*l, iota, r, R0)

    def parallel_gradient(self, phi_r: np.ndarray, i: int, der: np.ndarray):
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
        bz = self._bz[i]
        thetaVals = self._thetaVals[i]
        assert der.shape == phi_r.shape
        der[:] = 0

        # For each value of z interpolate the spline along theta and add
        # the value multiplied by the corresponding coefficient to the
        # derivative at the point at which it is required
        # This is split into three steps to avoid unnecessary modulo operations
        tmp = np.empty(self._nq)
        for i in range(self._fwdSteps):
            self._interpolator.compute_interpolant(
                phi_r[i, :], self._thetaSpline)
            for j, (s, c) in enumerate(zip(self._shifts, self._coeffs)):
                eval_spline_1d_vector(thetaVals[i, j, :], self._thetaSpline.basis.knots,
                                      self._thetaSpline.basis.degree, self._thetaSpline.coeffs, tmp, 0)
                der[(i-s) % self._nz, :] += c*tmp

        for i in range(self._fwdSteps, self._nz-self._bkwdSteps):
            self._interpolator.compute_interpolant(
                phi_r[i, :], self._thetaSpline)
            for j, (s, c) in enumerate(zip(self._shifts, self._coeffs)):
                eval_spline_1d_vector(thetaVals[i, j, :], self._thetaSpline.basis.knots,
                                      self._thetaSpline.basis.degree, self._thetaSpline.coeffs, tmp, 0)
                der[(i-s), :] += c*tmp

        for i in range(self._nz-self._bkwdSteps, self._nz):
            self._interpolator.compute_interpolant(
                phi_r[i, :], self._thetaSpline)
            for j, (s, c) in enumerate(zip(self._shifts, self._coeffs)):
                eval_spline_1d_vector(thetaVals[i, j, :], self._thetaSpline.basis.knots,
                                      self._thetaSpline.basis.degree, self._thetaSpline.coeffs, tmp, 0)
                der[(i-s) % self._nz, :] += c*tmp

        der *= (bz * self._inv_dz)

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

    constants : Constant class
        Class containing all the constants

    zDegree: int - optional
        Order of the lagrange interpolation

    """

    def __init__(self, eta_grid: list, splines: list, layout: Layout,
                 dt: float, constants, zDegree: int = 5):
        # Save all pertinent information
        self._zLagrangePts = zDegree+1
        self._points = eta_grid[1:3]
        self._nPoints = (self._points[0].size, self._points[1].size)
        self._interpolator = SplineInterpolator1D(splines[0])
        self._thetaSpline = Spline1D(splines[0])

        self._getLagrangePts(eta_grid, layout, dt,
                             constants.iota, constants.R0)
        self._LagrangeVals = np.ndarray(
            [self._nPoints[1], self._nPoints[0], self._zLagrangePts])

    def _getLagrangePts(self, eta_grid: list, layout: Layout, dt: float, iota, R0):
        """
        TODO
        """
        # Get z step
        dz = eta_grid[2][2]-eta_grid[2][1]

        r = eta_grid[0][layout.starts[layout.inv_dims_order[0]]:
                        layout.ends[layout.inv_dims_order[0]]]

        # Get theta step
        dtheta = (dz * iota(r) / R0)[:, None, None]

        bz = 1 / np.sqrt(1+(r * iota(r)/R0)[:, None, None]**2)

        nR = len(dtheta)
        nV = layout.shape[layout.inv_dims_order[3]]

        # Find the distance travelled in the z direction
        zDist = -eta_grid[3][layout.starts[layout.inv_dims_order[3]]:
                             layout.ends[layout.inv_dims_order[3]]][None, :, None] * \
            bz*dt

        # Find the number of steps between the start point and the lines
        # around the end point
        self._shifts = np.ndarray([nR, nV, self._zLagrangePts], dtype=int)
        self._shifts[:] = np.floor(zDist / dz) + \
            np.arange(-self._zLagrangePts//2+1,
                      self._zLagrangePts//2+1)[None, None, :]

        # Find the corresponding shift in the theta direction
        self._thetaShifts = dtheta*self._shifts

        # Find the distance to the points used for the interpolation
        zPts = dz * self._shifts[:, :, :]

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
        zDiff = zPos-zPts
        omega = np.prod(zDiff, axis=2)[:, :, None]
        lambdas = 1/np.prod(zPts[:, :, :, None]-zPts[:, :, None, :] +
                            np.eye(self._zLagrangePts)[None, None, :, :], axis=3)

        # If the final position is one of the points then zDiff=0
        # The division by 0 must be avoided and the coefficients should
        # be equal to 0 except at the point where they equal 1
        zComp = (zPts == zPos)
        with np.errstate(invalid='ignore', divide='ignore'):
            self._lagrangeCoeffs = np.where(zComp, 1, omega*lambdas/zDiff)

    def step(self, f: np.ndarray, cIdx: int, rIdx: int = 0):
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
        assert f.shape == self._nPoints

        # find the values of the function at each required point
        for i in range(self._nPoints[1]):
            self._interpolator.compute_interpolant(f[:, i], self._thetaSpline)

            get_lagrange_vals(i, self._shifts[rIdx, cIdx],
                              self._LagrangeVals, self._points[0],
                              self._thetaShifts[rIdx,
                                                cIdx], self._thetaSpline.basis.knots,
                              self._thetaSpline.basis.degree,
                              self._thetaSpline.coeffs)

        flux_advection(*self._nPoints, f,
                       self._lagrangeCoeffs[rIdx, cIdx],
                       self._LagrangeVals)

    def gridStep(self, grid: Grid):
        """
        TODO
        """
        assert grid.getLayout(grid.currentLayout).dims_order == (0, 3, 1, 2)
        for i, _ in grid.getCoords(0):  # r
            for j, _ in grid.getCoords(1):  # v
                self.step(grid.get2DSlice(i, j), j)


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

    constants : Constant class
        Class containing all the constants

    edge: str handle - optional
        String defining the boundary conditions. Options are:
        fEq - equilibrium value at boundary
        'null' - 0 at boundary
        'periodic' - periodic boundary conditions

    """

    def __init__(self, eta_vals: list, splines: BSplines, constants, edge: str = 'fEq'):
        self._points = eta_vals[3]
        self._nPoints = (self._points.size,)
        self._interpolator = SplineInterpolator1D(splines)
        self._spline = Spline1D(splines)
        self._constants = constants

        if (edge == 'fEq'):
            self._edgeType = 0
        elif (edge == 'null'):
            self._edgeType = 1
        elif (edge == 'periodic'):
            self._edgeType = 2
            minPt = self._points[0]
            width = self._points[-1]-self._points[0]
        else:
            raise RuntimeError(
                "V parallel boundary condition must be one of 'fEq', 'null', 'periodic'")

    def step(self, f: np.ndarray, dt: float, c: float, r: float):
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
        assert f.shape == self._nPoints
        self._interpolator.compute_interpolant(f, self._spline)

        v_parallel_advection_eval_step(f, self._points-c*dt, r, self._points[0],
                                       self._points[-1], self._spline.basis.knots,
                                       self._spline.basis.degree, self._spline.coeffs,
                                       self._constants.CN0, self._constants.kN0,
                                       self._constants.deltaRN0, self._constants.rp,
                                       self._constants.CTi, self._constants.kTi,
                                       self._constants.deltaRTi, self._edgeType)

    def gridStep(self, grid: Grid, phi: Grid, parGrad: ParallelGradient, parGradVals: np.array, dt: float):
        for i, r in grid.getCoords(0):
            parGrad.parallel_gradient(
                np.real(phi.get2DSlice(i)), i, parGradVals[i])
            for j, _ in grid.getCoords(1):  # z
                for k, _ in grid.getCoords(2):  # q
                    self.step(grid.get1DSlice(
                        i, j, k), dt, parGradVals[i, j, k], r)

    def gridStepKeepGradient(self, grid: Grid, parGradVals, dt: float):
        for i, r in grid.getCoords(0):
            for j, _ in grid.getCoords(1):  # z
                for k, _ in grid.getCoords(2):  # q
                    self.step(grid.get1DSlice(
                        i, j, k), dt, parGradVals[i, j, k], r)


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

    constants : Constant class
        Class containing all the constants

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

    def __init__(self, eta_vals: list, splines: list, constants, nulEdge=False,
                 explicitTrap: bool = False, tol: float = 1e-10):
        self._points = eta_vals[1::-1]
        self._shapedQ = np.atleast_2d(self._points[0]).T
        self._nPoints = (self._points[0].size, self._points[1].size)
        self._interpolator = SplineInterpolator2D(splines[0], splines[1])
        self._spline = Spline2D(splines[0], splines[1])
        self._constants = constants

        self._explicit = explicitTrap
        self._TOL = tol

        self._nulEdge = nulEdge

        self._drPhi_0 = np.empty(self._nPoints)
        self._dqPhi_0 = np.empty(self._nPoints)
        self._drPhi_k = np.empty(self._nPoints)
        self._dqPhi_k = np.empty(self._nPoints)
        self._endPts_k1_q = np.empty(self._nPoints)
        self._endPts_k1_r = np.empty(self._nPoints)
        self._endPts_k2_q = np.empty(self._nPoints)
        self._endPts_k2_r = np.empty(self._nPoints)

        self._max_loops = 0

        self._phiSplines = [Spline2D(splines[0], splines[1])
                            for i in range(eta_vals[2].size)]

    def allow_tests(self):
        """
        TODO
        """
        self._evalFunc = np.vectorize(self.evaluate, otypes=[float])

    def step(self, f: np.ndarray, dt: float, phi: Spline2D, v: float):
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
        assert f.shape == self._nPoints
        self._interpolator.compute_interpolant(f, self._spline)

        phiBases = phi.basis
        polBases = self._spline.basis

        if (self._explicit):
            poloidal_advection_step_expl(f, float(dt), v, self._points[1],
                                         self._points[0], self._drPhi_0,
                                         self._dqPhi_0, self._drPhi_k,
                                         self._dqPhi_k, self._endPts_k1_q,
                                         self._endPts_k1_r, self._endPts_k2_q,
                                         self._endPts_k2_r, phiBases[0].knots,
                                         phiBases[1].knots, phi.coeffs,
                                         phiBases[0].degree, phiBases[1].degree,
                                         polBases[0].knots, polBases[1].knots,
                                         self._spline.coeffs, polBases[0].degree,
                                         polBases[1].degree, self._constants.CN0,
                                         self._constants.kN0, self._constants.deltaRN0,
                                         self._constants.rp, self._constants.CTi,
                                         self._constants.kTi, self._constants.deltaRTi,
                                         self._constants.B0, self._nulEdge)
        else:
            poloidal_advection_step_impl(f, float(dt), v, self._points[1],
                                         self._points[0], self._drPhi_0,
                                         self._dqPhi_0, self._drPhi_k,
                                         self._dqPhi_k, self._endPts_k1_q,
                                         self._endPts_k1_r, self._endPts_k2_q,
                                         self._endPts_k2_r, phiBases[0].knots,
                                         phiBases[1].knots, phi.coeffs,
                                         phiBases[0].degree, phiBases[1].degree,
                                         polBases[0].knots, polBases[1].knots,
                                         self._spline.coeffs, polBases[0].degree,
                                         polBases[1].degree, self._constants.CN0,
                                         self._constants.kN0, self._constants.deltaRN0,
                                         self._constants.rp, self._constants.CTi,
                                         self._constants.kTi, self._constants.deltaRTi,
                                         self._constants.B0, self._TOL, self._nulEdge)

    def exact_step(self, f, endPts, v):
        """
        TODO
        """
        assert f.shape == self._nPoints

        self._interpolator.compute_interpolant(f, self._spline)

        for i in range(self._nPoints[0]):  # theta
            for j in range(self._nPoints[1]):  # r
                f[i, j] = self._evalFunc(endPts[0][i, j], endPts[1][i, j], v)

    def gridStep(self, grid: Grid, phi: Grid, dt: float):
        """
        TODO
        """
        gridLayout = grid.getLayout(grid.currentLayout)
        phiLayout = phi.getLayout(grid.currentLayout)

        assert gridLayout.dims_order[1:] == phiLayout.dims_order
        assert gridLayout.dims_order == (3, 2, 1, 0)

        # Evaluate splines
        for j, _ in grid.getCoords(1):  # z
            self._interpolator.compute_interpolant(
                np.real(phi.get2DSlice(j)), self._phiSplines[j])
        # Do step
        for i, v in grid.getCoords(0):
            for j, _ in grid.getCoords(1):  # z
                self.step(grid.get2DSlice(i, j), dt, self._phiSplines[j], v)

    def gridStep_SplinesUnchanged(self, grid: Grid, dt: float):
        """
        TODO
        """
        gridLayout = grid.getLayout(grid.currentLayout)

        assert gridLayout.dims_order == (3, 2, 1, 0)

        # Do step
        for i, v in grid.getCoords(0):
            for j, _ in grid.getCoords(1):  # z
                self.step(grid.get2DSlice(i, j), dt, self._phiSplines[j], v)

    def evaluate(self, theta, r, v):
        """
        TODO
        """
        if self._nulEdge:
            if (r < self._points[1][0]):
                return 0
            elif (r > self._points[1][-1]):
                return 0
            else:
                theta = theta % (2*pi)
                return self._spline.eval(theta, r)
        else:
            raise NotImplementedError(
                "Can't calculate exactly with fEq as background")


class PoloidalAdvectionArakawa:
    """
    PoloidalAdvection: Class containing information necessary to carry out
    an advection step along the poloidal surface using the Arakawa scheme.

    Parameters
    ----------
        eta_vals: list of array_like
            The coordinates of the grid points in each dimension. Index [0] must be
            the radial coordinate, index [1] must be the angular coordinate.

        constants : Constant class
            Class containing all the constants

        bc : str
            'dirichlet', 'periodic', or 'extrapolation'; which boundary conditions
            should be used in radial direction

        order : int
            which order of the Arakawa scheme should be used

        equilibrium_outside : bool
            if extrapolation is used, the grid step takes the equilibrium function on the outside

        verbose : bool
            if output information should be printed

        explicit: bool
            Indicates whether an explicit time integrator (Runge-Kutta 4)
            should be used or an implicit method (Crank-Nicolson) should be used
            instead

        save_conservation : bool
            If the integral of the distribution function and its square, as well as the energy
            should be saved before and after (and the difference between) the time step.

        foldername : str
            Where the conservation diagnsotics should be saved to if save_conservation == True.
    """

    def __init__(self, eta_vals: list, constants,
                 bc="extrapolation", order: int = 4,
                 equilibrium_outside: bool = True, verbose: bool = False,
                 explicit: bool = True):
        self._points = eta_vals[1::-1]
        self._points_theta = self._points[0]
        self._points_r = self._points[1]

        self._dr = self._points_r[1] - self._points_r[0]
        self._dtheta = self._points_theta[1] - self._points_theta[0]

        self._shapedQ = np.atleast_2d(self._points_theta).T

        self._nPoints_r = self._points_r.size
        self._nPoints_theta = self._points_theta.size

        assert self._nPoints_r == len(self._points_r)
        assert self._nPoints_theta == len(self._points_theta)

        self._nPoints = (self._nPoints_theta, self._nPoints_r)

        self._constants = constants

        self._explicit = explicit

        self.bc = bc

        self._equilibrium_outside = equilibrium_outside

        self.order = order

        self._verbose = verbose

        # left and right boundary indices, do we want to have them seperated?
        self.ind_bd = np.hstack([range(0, np.prod(self._nPoints), self._nPoints_r),
                                 range(self._nPoints_r - 1, np.prod(self._nPoints), self._nPoints_r)])

        self.r_scaling = np.kron(np.ones(self._nPoints_theta), self._points_r)

        # scaling of the boundary measure
        # needed for extrapolation, I guess not
        if self.bc == 'dirichlet':
            self.r_scaling[self.ind_bd] *= 1/2

        elif self.bc == 'extrapolation':
            assert self.order == 4
            # create empty stencils for the bigger J matrix
            f, inds_int, inds_bd = self.calc_ep_stencil()

            self.ind_int_ep = inds_int
            self.ind_bd_ep = inds_bd
            self.f_stencil = f
            self.phi_stencil = np.copy(f)
            self.r_scaling = np.copy(f)

            # increase the size of the scaling
            if self._points_r[0] - (order // 2 + 1) * self._dr > 0:
                self.r_outside = np.hstack([[self._points_r[0] - (j + 1) * self._dr for j in range(self.order//2)],
                                            [self._points_r[-1] + (j + 1) * self._dr for j in range(self.order//2)]])
            else:
                # equidistant extrapolation on the inner side could yield negative r-values hence we extrapolate
                # between 0.05 and r_min. The grid is not uniform anymore!
                print("Warning: the inner r_outside has smaller delta r")
                dr_o = (self._points_r[0] + 0.05) / order
                self.r_outside = np.hstack([[self._points_r[0] - (j + 1) * dr_o for j in range(self.order//2)],
                                            [self._points_r[-1] + (j + 1) * self._dr for j in range(self.order//2)]])

            # set the outside and interior values for the bigger r_scaling
            for k in range(self.order):
                self.r_scaling[self.ind_bd_ep[k]] = self.r_outside[k]
            self.r_scaling[self.ind_int_ep] = np.kron(
                np.ones(self._nPoints_theta), self._points_r)

            # assemble the rows and columns and the right-hand-side-matrix beforehand
            self._data = np.empty(self._nPoints_theta *
                                  (self._nPoints_r * 12 + 10), dtype=float)
            self.rowcolumns, self.J_phi = assemble_row_columns_akw_bracket_4th_order_extrapolation(
                self._points_theta, self._points_r, self._data)

    def calc_ep_stencil(self):
        """
        Calculate the stencil for the increased size of the interpolated stencil.

        Returns
        -------
            vec_ep : np.ndarray
                array of length (N_r+order)*N_theta with 0.0 value

            inds_int : np.darray
                array of length (N_r)*N_theta with the indices of the (interior of the) domain

            inds_bd_vstack: np.darray
                array of length order*N_theta with the outside indices, where the constant lines are placed
        """
        N_r_ep = self._nPoints_r + self.order
        N_ep = N_r_ep * self._nPoints_theta

        vec_ep = np.zeros(N_ep)
        ohalf = self.order // 2

        bd_left = [range(k, N_ep, N_r_ep) for k in range(ohalf)]
        bd_right = [range(N_r_ep-ohalf + k, N_ep, N_r_ep)
                    for k in range(ohalf)]

        inds_bd = np.hstack([bd_left, bd_right])
        inds_int = np.setdiff1d(range(N_ep), inds_bd)

        return vec_ep, inds_int, np.vstack([bd_left, bd_right])

    def allow_tests(self):
        """
        TODO
        """
        raise NotImplementedError(
            "This functionality has not been implemented yet!")

    def RK4(self, z, J, dt):
        """
        Simple Runge-Kutta 4th order for dz/dt = J(z).

        Parameters
        ----------
            z: np.array
                The current values at time t

            J : np.darry
                The rhs of the ode

            dt: float
                time-step size

        Returns
        -------
            z: np.array
                values at the new time t+dt
        """

        k1 = J.dot(z)
        k2 = J.dot(z + dt/2*k1)
        k3 = J.dot(z + dt/2*k2)
        k4 = J.dot(z + dt*k3)

        z += dt/6*k1 + dt/3*k2 + dt/3*k3 + dt/6*k4

    def step(self, f: np.ndarray, dt: float, phi: np.ndarray, values_f=None, values_phi=None):
        """
        TODO

        Parameters
        ----------
            TODO
        """
        if self.bc == "extrapolation":

            # if no values are given, assume 0 value outside
            if values_f == None:
                values_f = np.zeros(self.order)

            if values_phi == None:
                values_phi = np.zeros(self.order)

            self.step_extrapolation(f, dt, phi, values_f, values_phi)

        else:
            self.step_normal(f, dt, phi)

    def step_normal(self, f: np.ndarray, dt: float, phi: np.ndarray):
        """
        Carry out an advection step for the poloidal advection using the Arakawa scheme

        Parameters
        ----------
            f: array_like
                The current value of the function at the nodes.
                The result will be stored here

            dt: float
                Time-step

            phi: array_like
                Advection parameter d_tf + {phi,f} = 0; without scaling
        """

        if len(f.shape) != 1:
            assert f.shape == (self._nPoints_theta, self._nPoints_r), \
                f"{f.shape} != ({self._nPoints_theta}, {self._nPoints_r})"
            f = f.ravel()

        if len(phi.shape) != 1:
            assert phi.shape == (self._nPoints_theta, self._nPoints_r), \
                f"{phi.shape} != ({self._nPoints_theta}, {self._nPoints_r})"
            phi = phi.ravel()

        # np.real allocates new memory and should be replaced
        phi = np.real(phi)

        assert f.shape == np.prod(self._nPoints), \
            f'f shape: {f.shape} != nPoints: {np.prod(self._nPoints)}'

        assert phi.shape == np.prod(self._nPoints), \
            f'phi shape: {phi.shape} != nPoints: {np.prod(self._nPoints)}'

        # enforce bc strongly?
        if self.bc == 'dirichlet':
            # f[self.ind_bd] = -np.ones(len(self.ind_bd))
            phi[self.ind_bd] = np.zeros(len(self.ind_bd))

        # assemble the bracket
        J_phi = assemble_bracket_arakawa(self.bc, self.order, phi,
                                         self._points_theta, self._points_r)

        # algebraically conserving properties
        if self._verbose:
            print('conservation tests global:')
            print(f'Integral of f: {sum(J_phi.dot(f.ravel()))}')
            print(
                f'Energy: {sum(np.multiply(phi.ravel(), J_phi.dot(f.ravel())))}')
            print(
                f'Square integral of f: {sum(np.multiply(f.ravel(), J_phi.dot(f.ravel())))}')

        if self._explicit:
            # add the missing scaling to the rows of J
            I_s = diags(1/self.r_scaling, 0)
            J_s = I_s @ J_phi
        else:
            # scaling is only found in the identity
            I_s = diags(self.r_scaling, 0)
            A = I_s - dt/2 * J_phi
            B = I_s + dt/2 * J_phi

        # execute the time-step
        if self._explicit:
            self.RK4(f[:], J_s, dt)
        else:
            f[:] = spsolve(A, B.dot(f))

    def step_extrapolation(self, f: np.ndarray, dt: float, phi: np.ndarray, values_f: np.ndarray, values_phi: np.ndarray):
        """
        Carry out an advection step for the poloidal advection using the Arakawa scheme

        Parameters
        ----------
            f: array_like
                The current value of the function at the nodes.
                The result will be stored here

            dt: float
                Time-step

            phi: array_like
                Advection parameter d_tf + {phi,f} = 0; without scaling

            values_f : array_like
                Values of f outside the domain

            values_phi : array_like
                Values of phi outside the domain
        """
        if len(f.shape) != 1:
            assert f.shape == (self._nPoints_theta, self._nPoints_r), \
                f"{f.shape} != ({self._nPoints_theta}, {self._nPoints_r})"
            f = f.ravel()

        if len(phi.shape) != 1:
            assert phi.shape == (self._nPoints_theta, self._nPoints_r), \
                f"{phi.shape} != ({self._nPoints_theta}, {self._nPoints_r})"
            phi = phi.ravel()

        # np.real allocates new memory and should be replaced
        phi = np.real(phi)

        # set phi to zero on the boundary (if needed)
        # phi[self.ind_bd] = np.zeros(len(self.ind_bd))

        # fill the working stencils
        self.f_stencil[self.ind_int_ep] = f
        self.phi_stencil[self.ind_int_ep] = phi

        # set extrapolation values
        for k in range(self.order):
            self.f_stencil[self.ind_bd_ep[k]] = values_f[k]
            self.phi_stencil[self.ind_bd_ep[k]] = values_phi[k]

        # assemble the bracket
        # check if precomputation is used (later reduce to one method)
        if self.rowcolumns == None:
            self.J_phi = assemble_bracket_arakawa(self.bc, self.order, self.phi_stencil,
                                                  self._points_theta, self._points_r)
        else:
            update_bracket_4th_order_dirichlet_extrapolation(
                self.J_phi, self.rowcolumns, self.phi_stencil, self._points_theta, self._points_r, self._data)

        # algebraically conserving properties
        if self._verbose:
            print('conservation tests global:')
            print(
                f'Integral of f: {sum(self.J_phi.dot(self.f_stencil.ravel()))}')
            print(
                f'Energy: {sum(np.multiply(self.phi_stencil.ravel(), self.J_phi.dot(self.f_stencil.ravel())))}')
            print(
                f'Square integral of f: {sum(np.multiply(self.f_stencil.ravel(), self.J_phi.dot(self.f_stencil.ravel())))}')

        if self._explicit:
            # add the missing scaling to the rows of J
            I_s = diags(1/self.r_scaling, 0)
            J_s = I_s @ self.J_phi
        else:
            # scaling is only found in the identity
            I_s = diags(self.r_scaling, 0)
            A = I_s - dt/2 * self.J_phi
            B = I_s + dt/2 * self.J_phi

        # execute the time-step
        if self._explicit:

            # Do substepping such that the CFL condition is satisfied
            # Use J_phi / (r*d_theta*dr) = unscaled bracket / dx
            J_max = np.max(np.abs(self.J_phi))

            dx = self._dr * self._dtheta
            CFL = int(J_max * dt / dx + 1)

            if self._verbose:
                print(f'CFL number is {CFL}')

            # Update f_stencil in-place
            for _ in range(1, CFL + 1):
                self.RK4(self.f_stencil, J_s, dt/CFL)

            # Update f
            f[:] = self.f_stencil[self.ind_int_ep]

        else:
            f[:] = spsolve(A, B.dot(self.f_stencil))[self.ind_int_ep]

    def exact_step(self, f, endPts, v):
        """
        TODO
        """
        raise NotImplementedError(
            "This functionality has not been implemented yet!")

    def gridStep(self, f: Grid, phi: Grid, dt: float):
        """
        Does a time stepping on the whole grid by iterating over the velocity and z-direction by calling the
        corresponding step function on each slice. For the extrapolation method, values for the outside arrays
        are found before doing the time step. Diagnostics are run if chosen so at the initialization of the class.

        Parameters
        ----------
            f : pygyro.model.grid.Grid
                Grid object that characterizes the distribution function

            phi : pygyro.model.grid.Grid
                Grid object that characterizes the electric field

            dt : float
                Stepsize of the time stepping
        """
        gridLayout = f.getLayout(f.currentLayout)
        phiLayout = phi.getLayout(f.currentLayout)

        assert (gridLayout.dims_order[1:] == phiLayout.dims_order)
        assert (gridLayout.dims_order == (3, 2, 1, 0))

        if self.bc == "extrapolation":
            # Do step
            for i, v in f.getCoords(0):  # v
                for j, _ in f.getCoords(1):  # z

                    # assume phi equals 0 outside
                    values_phi = np.zeros(self.order)
                    values_f = np.zeros(self.order)
                    if self._equilibrium_outside:
                        values_f = [f_eq(self.r_outside[k], v, self._constants.CN0,
                                         self._constants.kN0, self._constants.deltaRN0,
                                         self._constants.rp, self._constants.CTi, self._constants.kTi,
                                         self._constants.deltaRTi) for k in range(self.order)]

                    self.step_extrapolation(f.get2DSlice(
                        i, j), dt, phi.get2DSlice(j), values_f, values_phi)

        else:
            # Do step
            for i, _ in f.getCoords(0):  # v
                for j, _ in f.getCoords(1):  # z

                    self.step_normal(f.get2DSlice(i, j),
                                     dt, phi.get2DSlice(j))

    def gridStep_SplinesUnchanged(self, grid: Grid, dt: float):
        """
        TODO
        """
        raise NotImplementedError(
            "This functionality has not been implemented yet!")

    def evaluate(self, theta, r, v):
        """
        TODO
        """
        raise NotImplementedError(
            "This functionality has not been implemented yet!")
