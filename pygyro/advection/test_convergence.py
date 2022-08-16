import pytest
import numpy as np
import matplotlib.pyplot as plt

from ..initialisation.constants import get_constants
from ..initialisation.initialiser_funcs import f_eq
from .advection import PoloidalAdvectionArakawa
from ..arakawa.utilities import compute_int_f_squared


def integrate_Arakawa_vortex(N_theta, N_r, theta_grid, r_grid, order,
                             bc='extrapolation', N_t=500, dt=0.001):
    """
    Integrate the advection equation for a vortex in time using Arakawa scheme
    for the discretization of the bracket and (by default) an implicit time integrator.

    Parameters
    ----------
        N_theta : int
            Number of points in angular direction.

        N_r : int
            Number of points in radial direction.

        theta_grid : np.ndarray
            array of length N_theta; grid in angular direction.

        r_grid : np.ndarray
            array of length N_r; grid in radial direction.

        order : int
            order with which the Arakawa scheme is to be constructed

        bc : str
            boundary conditions with which the Arakawa scheme is to be constructed

        N_t : int
            number of time steps

        dt : float
            size of the time steps
    """
    eta_vals = [r_grid, theta_grid, np.linspace(0, 1, 4), np.linspace(0, 1, 4)]

    v = 0.

    f_vals = np.ndarray([N_t + 1, N_theta, N_r])
    phiVals = np.empty([N_theta, N_r])

    phiVals[:] = 10 * eta_vals[0]

    constants = get_constants('testSetups/iota0.json')

    polAdv = PoloidalAdvectionArakawa(eta_vals, constants,
                                      order=order, bc=bc, explicit=False)

    assert polAdv._nPoints_r == N_r, f'{polAdv._nPoints_r} != {N_r}'
    assert polAdv._nPoints_theta == N_theta, f'{polAdv._nPoints_theta} != {N_theta}'

    f_vals[0, :, :] = np.exp(-np.atleast_2d((eta_vals[1] - np.pi)**2).T - (eta_vals[0] - 7)**2) / 4 \
        + f_eq(0.1, v, constants.CN0, constants.kN0,
               constants.deltaRN0, constants.rp,
               constants.CTi, constants.kTi,
               constants.deltaRTi)

    assert phiVals.shape == (
        N_theta, N_r), f'{phiVals.shape} != ({N_theta}, {N_r})'

    for n in range(N_t):
        polAdv.step(f_vals[n, :, :], dt, np.array(phiVals, dtype=float))
        f_vals[n + 1, :, :] = f_vals[n, :, :]

    return f_vals[-1, :, :], polAdv._dtheta, polAdv._dr


def integrate_Arakawa_constAdv(N_theta, N_r, theta_grid, r_grid, order,
                               bc='extrapolation', N_t=500, dt=0.001):
    """
    Integrate the advection equation for a constant advection problem in time using
    Arakawa scheme for the discretization of the bracket and (by default) an implicit
    time integrator.

    Parameters
    ----------
        N_theta : int
            Number of points in angular direction.

        N_r : int
            Number of points in radial direction.

        theta_grid : np.ndarray
            array of length N_theta; grid in angular direction.

        r_grid : np.ndarray
            array of length N_r; grid in radial direction.

        order : int
            order with which the Arakawa scheme is to be constructed

        bc : str
            boundary conditions with which the Arakawa scheme is to be constructed

        N_t : int
            number of time steps

        dt : float
            size of the time steps
    """
    eta_vals = [r_grid, theta_grid, np.linspace(0, 1, 4), np.linspace(0, 1, 4)]

    v = 0.

    f_vals = np.ndarray([N_t + 1, N_theta, N_r])
    phiVals = np.empty([N_theta, N_r])

    phiVals[:] = 3*eta_vals[0]**2

    constants = get_constants('testSetups/iota0.json')

    polAdv = PoloidalAdvectionArakawa(eta_vals, constants,
                                      order=order, bc=bc, explicit=False)

    assert polAdv._nPoints_r == N_r, f'{polAdv._nPoints_r} != {N_r}'
    assert polAdv._nPoints_theta == N_theta, f'{polAdv._nPoints_theta} != {N_theta}'

    f_vals[0, :, :] = np.exp(-np.atleast_2d((eta_vals[1] - np.pi)**2).T - (eta_vals[0] - 7)**2) / 4 \
        + f_eq(0.1, v, constants.CN0, constants.kN0,
               constants.deltaRN0, constants.rp,
               constants.CTi, constants.kTi,
               constants.deltaRTi)

    assert phiVals.shape == (
        N_theta, N_r), f'{phiVals.shape} != ({N_theta}, {N_r})'

    for n in range(N_t):
        polAdv.step(f_vals[n, :, :], dt, np.array(phiVals, dtype=float))
        f_vals[n + 1, :, :] = f_vals[n, :, :]

    return f_vals[-1, :, :], polAdv._dtheta, polAdv._dr


@pytest.mark.long
@pytest.mark.serial
@pytest.mark.parametrize("bc, order", [('dirichlet', 2), ('dirichlet', 4),
                                       ('periodic', 2), ('periodic', 4),
                                       ('extrapolation', 4)])
# @pytest.mark.parametrize("bc, order", [('extrapolation', 4)])
@pytest.mark.parametrize("problem", ['constAdv', 'vortex'])
def test_convergence(bc, order, problem, show_plot=False):
    """
    Test the order of convergence of the Arakawa scheme for the poloidal advection step
    by comparing results on a mesh and a refinement to half grid spacing. The error between
    the two is computed for different mesh sizes and plotted against the grid spacing.
    Tested are the advection problems for constant advection and vortex.

    Parameters
    ----------
        bc : int
            Boundary conditions with which the scheme should be run.

        order : int
            Order of the scheme which is to be expected (2 or 4)

        problem : str
            which problem to consider for the convergence study: constant advection or vortex

        show_plot : bool
            if plots of the convergence curve should be shown
    """
    N_points = 15
    N_iter = 5

    N_t = 50
    dt = 0.05

    f_vals = [None] * N_iter
    d_theta = [None] * N_iter
    d_r = [None] * N_iter
    r_grid = [None] * N_iter
    theta_grid = [None] * N_iter

    norm_diff = [None] * (N_iter - 1)

    r_min = 0.01
    r_max = 14.1

    for n in range(N_iter):
        N_theta = 2**n * N_points
        N_r = 2**n * (N_points - 1) + 1

        theta_grid[n] = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
        r_grid[n] = np.linspace(r_min, r_max, N_r, endpoint=True)

        if problem == 'constAdv':
            f_vals[n], d_theta[n], d_r[n] = integrate_Arakawa_constAdv(N_theta=N_theta, N_r=N_r,
                                                                       theta_grid=theta_grid[n], r_grid=r_grid[n],
                                                                       order=order, bc=bc, N_t=N_t, dt=dt)
        elif problem == 'vortex':
            f_vals[n], d_theta[n], d_r[n] = integrate_Arakawa_vortex(N_theta=N_theta, N_r=N_r,
                                                                     theta_grid=theta_grid[n], r_grid=r_grid[n],
                                                                     order=order, bc=bc, N_t=N_t, dt=dt)
        else:
            raise NotImplementedError(
                f'The problem type {problem} is not implemented')

    for n in range(N_iter - 1):
        assert d_r[n]/2 == d_r[n + 1]
        assert d_theta[n]/2 == d_theta[n + 1]

        norm_diff[n] = np.sqrt(compute_int_f_squared(f_vals[n] - f_vals[n + 1][::2, ::2],
                                                     d_theta[n], d_r[n], r_grid[n]))

    d_r_fit = np.array(d_r)[:-1]

    if show_plot:
        plt.loglog(d_r_fit, norm_diff, '-o', ls='solid')
        plt.xlabel('Grid spacing')
        plt.ylabel('Error between solutions on consequent refined meshes')
        plt.title(
            f'Convergence curve for Arakawa scheme of implemented order {order}')
        plt.show()

    coef, _ = np.polyfit(np.log(d_r_fit), np.log(norm_diff), 1)

    assert coef == order, f'Fit coefficient is {coef} but order is {order}'
