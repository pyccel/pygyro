import os
from mpi4py import MPI
import pytest
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags

from .utilities import compute_int_f, compute_int_f_squared, get_potential_energy
from .discrete_brackets_polar import assemble_bracket_arakawa
from ..initialisation.initialiser_funcs import f_eq
from pygyro.initialisation.setups import setupCylindricalGrid
from pygyro.advection.advection import PoloidalAdvectionArakawa
from pygyro import splines as spl
from pygyro.model.layout import LayoutSwapper
from pygyro.model.process_grid import compute_2d_process_grid
from pygyro.model.grid import Grid


@pytest.mark.parametrize('bc', ['periodic', 'dirichlet', 'extrapolation'])
@pytest.mark.parametrize('order', [2, 4])
def test_skewsymmetry(bc, order, tol=1e-10):
    """
    Test the skew-symmetric property of the discrete bracket.

    Parameters
    ----------
        bc : str
            Boundary conditions for discrete bracket

        order : int
            Order of the Arakawa scheme

        tol : float
            precision with which the quantities should be tested
    """
    N_theta = 60
    N_r = 80
    N_tot = N_theta * N_r

    r_min = 0.01
    r_max = 14.1

    theta_grid = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    r_grid = np.linspace(r_min, r_max, N_r)

    if bc == "extrapolation":
        np.random.seed(1305)
        phi = np.random.rand(N_theta, N_r+order).ravel()
    else:
        np.random.seed(1305)
        phi = np.random.rand(N_theta, N_r).ravel()

    if bc == "dirichlet":
        # Set boundary values to a constant on each boundary
        ind_bd_left = range(0, N_tot, N_r)
        ind_bd_right = range(N_r - 1, N_tot, N_r)

        np.random.seed(123)
        phi[ind_bd_left] = np.random.rand()
        np.random.seed(321)
        phi[ind_bd_right] = np.random.rand()

    # Phi is supposed to be constant (in theta) outside of the domain and on the boundary
    elif bc == "extrapolation":
        N_ep = N_theta*(N_r + order)
        bd_left = [range(k, N_ep, N_r+order) for k in range(order//2+1)]
        bd_right = [range(N_r+order-order//2+k-1, N_ep, N_r+order)
                    for k in range(order//2+1)]
        inds_bd = np.hstack([bd_left, bd_right])
        np.random.seed(815)
        phi[inds_bd] = np.random.rand()

    J_phi = assemble_bracket_arakawa(bc, order,
                                     phi, theta_grid, r_grid).toarray()

    assert (J_phi + J_phi.transpose() < tol).all()


@pytest.mark.parametrize('bc', ['periodic', 'dirichlet', 'extrapolation'])
@pytest.mark.parametrize('order', [2, 4])
def test_bracket_mean(bc, order, tol=1e-10):
    """
    Test mean of the discrete bracket and mean of the bracket times each of its
    arguments to be zero; these are equations (5), (6), and (7) in [1].

    Parameters
    ----------
        bc : str
            Boundary conditions for discrete bracket

        order : int
            Order of the Arakawa scheme

        tol : float
            precision with which the quantities should be tested

    [1] : A. Arakawa, 1966 - Computational Design for Long-Term Numerical Integration of
    the Equations of Fluid Motion: Two-Dimensional Incompressible Flow. Part I
    """
    N_theta = 200
    N_r = 180
    N_tot = N_theta * N_r

    r_min = 0.01
    r_max = 14.1

    theta_grid = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    r_grid = np.linspace(r_min, r_max, N_r)

    if bc == "extrapolation":
        np.random.seed(1206)
        f = np.random.rand(N_theta, N_r+order).ravel()

        np.random.seed(1305)
        phi = np.random.rand(N_theta, N_r+order).ravel()
    else:
        np.random.seed(1206)
        f = np.random.rand(N_theta, N_r).ravel()

        np.random.seed(1305)
        phi = np.random.rand(N_theta, N_r).ravel()

    # Set boundary values to a constant on each boundary
    if bc == "dirichlet":
        ind_bd_left = range(0, N_tot, N_r)
        ind_bd_right = range(N_r - 1, N_tot, N_r)

        np.random.seed(123)
        phi[ind_bd_left] = np.random.rand()
        np.random.seed(321)
        phi[ind_bd_right] = np.random.rand()
    # Phi is supposed to be constant (in theta) outside of the domain and on the boundary
    elif bc == "extrapolation":
        N_ep = N_theta*(N_r + order)
        bd_left = [range(k, N_ep, N_r+order) for k in range(order//2+1)]
        bd_right = [range(N_r+order-order//2+k-1, N_ep, N_r+order)
                    for k in range(order//2+1)]
        inds_bd = np.hstack([bd_left, bd_right])
        np.random.seed(815)
        phi[inds_bd] = np.random.rand()

    J_phi = assemble_bracket_arakawa(bc, order,
                                     phi, theta_grid, r_grid)

    J_phi_f = J_phi.dot(f)

    assert len(J_phi_f.shape) == 1

    if bc == "extrapolation":
        assert J_phi_f.shape[0] == N_theta * (N_r+order)
    else:
        assert J_phi_f.shape[0] == N_theta * (N_r)

    sum_J_phi_f = np.sum(J_phi_f)
    sum_f_J_phi_f = np.sum(np.multiply(J_phi_f, f))
    sum_phi_J_phi_f = np.sum(np.multiply(J_phi_f, phi))

    assert sum_J_phi_f < tol
    assert sum_f_J_phi_f < tol
    assert sum_phi_J_phi_f < tol


# TODO for extrapolation
@pytest.mark.parametrize('bc', ['periodic', 'dirichlet'])
@pytest.mark.parametrize('order', [2, 4])
@pytest.mark.parametrize('int_method', ['sum', 'trapz'])
def test_conservation(bc, order, int_method, tol=1e-10, iter_tol=1e-10):
    """
    Test the conservation of the discrete integral of f, f^2, and phi over
    the whole domain for an evolution in time.

    Parameters
    ----------
        bc : str
            Boundary conditions for discrete bracket

        order : int
            Order of the Arakawa scheme

        int_method : str
            'sum' or 'trapz'; method with which the integral should be computed

        tol : float
            precision with which the initial sum quantities should be tested

        iter_tol : float
            relative precision with which the integral quantities in the time-loop should be tested
    """
    N_theta = 100
    N_r = 80
    N_tot = N_theta * N_r

    r_min = 0.5
    r_max = 14.1

    theta_grid = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    r_grid = np.linspace(r_min, r_max, N_r)

    d_theta = theta_grid[1] - theta_grid[0]
    d_r = r_grid[1] - r_grid[0]

    assert d_r < r_min

    # Set boundary values to a constant on each boundary
    ind_bd_left = range(0, N_tot, N_r)
    ind_bd_right = range(N_r - 1, N_tot, N_r)

    np.random.seed(1206)
    f = np.exp(-np.atleast_2d((theta_grid - np.pi)**2).T - (r_grid - 7)**2) / 4
    f = f.ravel()

    assert f.shape[0] == N_theta * N_r

    np.random.seed(1305)
    phi = np.zeros((N_theta, N_r))
    phi[:] = 3 * r_grid ** 2
    phi = phi.ravel()

    assert phi.shape[0] == N_theta * N_r

    f[ind_bd_left] = f[0]
    f[ind_bd_right] = f[-1]

    phi[ind_bd_left] = phi[0]
    phi[ind_bd_right] = phi[-1]

    J_phi = assemble_bracket_arakawa(bc, order,
                                     phi, theta_grid, r_grid)
    J_phi_f = J_phi.dot(f)

    assert len(J_phi_f.shape) == 1
    assert J_phi_f.shape[0] == N_theta * N_r

    # Ensure that the bracket and products with its arguments are zero at the beginning.
    # Note that since the bracket does not include the scaling we give np.ones as r_grid
    # (so the r scaling in the integral doesn't do anything)
    int_J = compute_int_f(J_phi_f, d_theta, d_r,
                          np.ones(len(r_grid)), method=int_method)
    int_J_f = compute_int_f(np.multiply(J_phi_f, f), d_theta, d_r,
                            np.ones(len(r_grid)), method=int_method)
    int_J_phi = compute_int_f(np.multiply(J_phi_f, phi), d_theta, d_r,
                              np.ones(len(r_grid)), method=int_method)

    assert int_J < tol
    assert int_J_f < tol
    assert int_J_phi < tol

    dt = 0.1
    N = 30

    r_scaling = np.tile(r_grid, N_theta)

    # scaling of the boundary measure
    if bc == 'dirichlet':
        r_scaling[ind_bd_left] *= 1/2
        r_scaling[ind_bd_right] *= 1/2

    int_f_init = compute_int_f(f, d_theta, d_r,
                               r_grid, method=int_method)
    int_f_squared_init = compute_int_f_squared(f, d_theta, d_r,
                                               r_grid, method=int_method)
    total_energy_init = get_potential_energy(f, phi, d_theta, d_r,
                                             r_grid, method=int_method)

    for _ in range(N):
        # scaling is only found in the identity
        I_s = diags(r_scaling, 0)
        A = I_s - dt/2 * J_phi
        B = I_s + dt/2 * J_phi

        f[:] = spsolve(A, B.dot(f))

        int_f = compute_int_f(f, d_theta, d_r,
                              r_grid, method=int_method)
        int_f_squared = compute_int_f_squared(f, d_theta, d_r,
                                              r_grid, method=int_method)
        total_energy = get_potential_energy(f, phi, d_theta, d_r,
                                            r_grid, method=int_method)

        assert np.abs(int_f - int_f_init)/int_f_init < iter_tol
        assert np.abs(int_f_squared - int_f_squared_init) / \
            int_f_squared_init < iter_tol
        assert np.abs(total_energy - total_energy_init) / \
            total_energy_init < iter_tol


def test_extrapolation_bracket(abs_tol=1e-10):
    """
    Test the conservation of the discrete integral of f, f^2, and phi over
    the whole domain for an evolution in time.

    Parameters
    ----------
        int_method : str
            'sum' or 'trapz'; method with which the integral should be computed

        bc : str
            Boundary conditions for discrete bracket

        order : int
            Order of the Arakawa scheme

        tol : float
            precision with which the initial sum quantities should be tested

        iter_tol : float
            relative precision with which the integral quantities in the time-loop should be tested
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # load constants from file
    constantFile = os.path.dirname(os.path.abspath(__file__)) + "/iota0.json"

    # set up constants and distribution function grid object
    distribFunc, constants, _ = setupCylindricalGrid(constantFile=constantFile,
                                                     layout='v_parallel',
                                                     comm=comm,
                                                     allocateSaveMemory=True)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax],
              [0, 2*np.pi],
              [constants.zMin, constants.zMax]]

    nkts = [n + 1 + d * (int(p) - 1)
            for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grids = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel_2d': [0, 2, 1],
                      'mode_solve': [1, 2, 0]}
    layout_vpar = {'v_parallel_1d': [0, 2, 1]}
    layout_poloidal = {'poloidal': [2, 1, 0]}

    nprocs = compute_2d_process_grid(npts, size)

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grids,
                                'v_parallel_2d')

    # Set up phi grid object
    phi = Grid(distribFunc.eta_grid[:3], distribFunc.getSpline(slice(0, 3)),
               remapperPhi, 'mode_solve', comm, dtype=np.complex128)

    # switch layout to poloidal
    """
    Phi Layout has to be in poloidal, i.e.:
        (z, theta, r)
    """
    phi.setLayout('poloidal')
    """
    Layout has to be in poloidal, i.e.:
        (v, z, theta, r)
    """
    distribFunc.setLayout('poloidal')

    layout = distribFunc.getLayout('poloidal')
    eta_grid = distribFunc.eta_grid

    # Make poloidal advection arakawa object
    polAdv = PoloidalAdvectionArakawa(eta_grid, constants, explicit=True)

    # load global and local grid
    r = eta_grid[0]
    q = eta_grid[1]
    z = eta_grid[2]
    v = eta_grid[3]

    idx_r = layout.inv_dims_order[0]
    idx_q = layout.inv_dims_order[1]
    idx_z = layout.inv_dims_order[2]
    idx_v = layout.inv_dims_order[3]

    my_r = r[layout.starts[idx_r]:layout.ends[idx_r]]
    my_q = q[layout.starts[idx_q]:layout.ends[idx_q]]
    my_z = z[layout.starts[idx_z]:layout.ends[idx_z]]
    my_v = v[layout.starts[idx_v]:layout.ends[idx_v]]

    shape = [1, 1, 1, 1]
    shape[idx_r] = my_r.size
    shape[idx_q] = my_q.size
    shape[idx_z] = my_z.size
    shape[idx_v] = my_v.size

    # insert random values into distribution function
    distribFunc._f[:, :, :, :] = 100*np.random.rand(*shape)
    assert distribFunc._f.shape[idx_r] == my_r.shape[0]
    assert distribFunc._f.shape[idx_q] == my_q.shape[0]
    assert distribFunc._f.shape[idx_z] == my_z.shape[0]
    assert distribFunc._f.shape[idx_v] == my_v.shape[0]

    shape_phi = [1, 1, 1]
    shape_phi[idx_r - 1] = my_r.size
    shape_phi[idx_q - 1] = my_q.size
    shape_phi[idx_z - 1] = my_z.size

    # insert random values into phi (do it after running one step, otherwise CFL takes forever)
    phi._f[:, :, :] = 100*np.random.rand(*shape_phi)
    assert phi._f.shape[idx_r - 1] == my_r.shape[0]
    assert phi._f.shape[idx_q - 1] == my_q.shape[0]
    assert phi._f.shape[idx_z - 1] == my_z.shape[0]

    # test zeroness of product of bracket with f and phi on each slice of (z,v)
    for i, v in distribFunc.getCoords(0):
        for j, _ in distribFunc.getCoords(1):

            # assume phi equals 0 outside
            values_phi = np.zeros(polAdv.order)
            values_f = np.zeros(polAdv.order)
            if polAdv._equilibrium_outside:
                values_f = [f_eq(polAdv.r_outside[k], v, polAdv._constants.CN0,
                                 polAdv._constants.kN0, polAdv._constants.deltaRN0,
                                 polAdv._constants.rp, polAdv._constants.CTi, polAdv._constants.kTi,
                                 polAdv._constants.deltaRTi) for k in range(polAdv.order)]

            # fill the working stencils
            polAdv.f_stencil[polAdv.ind_int_ep] = distribFunc.get2DSlice(
                i, j).ravel()
            polAdv.phi_stencil[polAdv.ind_int_ep] = \
                np.real(phi.get2DSlice(j).ravel())

            # set extrapolation values
            for k in range(polAdv.order):
                polAdv.f_stencil[polAdv.ind_bd_ep[k]] = values_f[k]
                polAdv.phi_stencil[polAdv.ind_bd_ep[k]] = values_phi[k]

            # assemble the bracket
            J_phi = assemble_bracket_arakawa(polAdv.bc, polAdv.order, polAdv.phi_stencil,
                                             polAdv._points_theta, polAdv._points_r)

            assert np.sum(J_phi) <= abs_tol, np.sum(J_phi)

            polAdv.f_stencil[polAdv.ind_int_ep] = \
                distribFunc.get2DSlice(i, j).ravel()

            J_phi_f = J_phi.dot(polAdv.f_stencil)

            assert np.sum(J_phi_f) <= abs_tol, np.sum(J_phi_f)
            assert np.sum(np.multiply(np.real(phi.get2DSlice(j)).ravel(),
                                      J_phi_f[polAdv.ind_int_ep])) <= abs_tol
