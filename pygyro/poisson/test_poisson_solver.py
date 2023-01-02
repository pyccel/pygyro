from mpi4py import MPI
import numpy as np
import pytest
from math import pi
from scipy.integrate import trapz

from ..model.process_grid import compute_2d_process_grid
from ..model.layout import LayoutSwapper, getLayoutHandler
from ..model.grid import Grid
from ..initialisation.setups import setupCylindricalGrid
from ..initialisation.constants import Constants
from ..initialisation.initialiser_funcs import Te, f_eq as fEq
from .. import splines as spl
from .poisson_solver import DiffEqSolver, DensityFinder, QuasiNeutralitySolver


@pytest.mark.parallel
def test_DensityFinder_poly_Rho(param_df_poly):
    """
    TODO
    """
    npts_v, coeffs, tol = param_df_poly
    npts = [30, 4, 4, npts_v]
    domain = [[1, 5], [0, 2*pi], [0, 1], [0, 10]]
    degree = [3, 3, 3, 3]
    period = [False, True, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel': [0, 2, 1]}

    grid, constants, _ = setupCylindricalGrid(
        npts=npts, layout='v_parallel', vMax=10, vMin=0)

    nprocs = grid.getLayout(grid.currentLayout).nprocs[:2]

    remapper = getLayoutHandler(comm, layout_poisson, nprocs, eta_grid[:3])

    rho = Grid(eta_grid[:3], bsplines[:3], remapper, 'v_parallel', comm)

    for i, _ in grid.getCoords(0):  # r
        for j, _ in grid.getCoords(1):  # z
            for k, _ in grid.getCoords(2):  # q
                vec = grid.get1DSlice(i, j, k)
                for l, v in grid.getCoords(3):
                    vec[l] = coeffs[0]*v**3 + coeffs[1] * \
                        v**2 + coeffs[2]*v + coeffs[3]

    df = DensityFinder(3, grid.getSpline(3), grid.eta_grid, constants)

    df.getRho(grid, rho)

    vals = coeffs[0]*10**4/4 + coeffs[1]*10**3 / \
        3 + coeffs[2]*10**2/2 + coeffs[3]*10

    err = rho._f - vals
    assert (np.abs(err) < tol).all()


@pytest.mark.parallel
def test_DensityFinder_poly_RhoPerturbed(param_df_poly):
    """
    TODO
    """
    npts_v, coeffs, tol = param_df_poly
    npts = [30, 4, 4, npts_v]
    domain = [[1, 5], [0, 2*pi], [0, 1], [0, 10]]
    degree = [3, 3, 3, 3]
    period = [False, True, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel': [0, 2, 1]}

    grid, constants, _ = setupCylindricalGrid(
        npts=npts, layout='v_parallel', vMax=10, vMin=0)

    nprocs = grid.getLayout(grid.currentLayout).nprocs[:2]

    remapper = getLayoutHandler(comm, layout_poisson, nprocs, eta_grid[:3])

    rho = Grid(eta_grid[:3], bsplines[:3], remapper, 'v_parallel', comm)

    for i, r in grid.getCoords(0):
        for j, z in grid.getCoords(1):
            for k, _ in grid.getCoords(2):  # q
                vec = grid.get1DSlice(i, j, k)
                for l, v in grid.getCoords(3):
                    vec[l] = fEq(r, v, constants.CN0, constants.kN0, constants.deltaRN0,
                                 constants.rp, constants.CTi, constants.kTi, constants.deltaRTi) \
                        + coeffs[0]*v**3 + coeffs[1] * \
                        v**2 + coeffs[2]*v + coeffs[3]

    df = DensityFinder(3, grid.getSpline(3), grid.eta_grid, constants)

    df.getPerturbedRho(grid, rho)

    vals = coeffs[0]*10**4/4 + coeffs[1]*10**3 / \
        3 + coeffs[2]*10**2/2 + coeffs[3]*10

    err = rho._f - vals
    assert (np.abs(err) < tol).all()


@pytest.mark.parallel
@pytest.mark.parametrize("npts_v,tol", [(10, 0.005), (20, 0.0004), (30, 4e-5)])
def test_DensityFinder_cos_Rho(npts_v, tol):
    """
    TODO
    """
    npts = [30, 4, 4, npts_v]
    domain = [[1, 5], [0, 2*pi], [0, 1], [0, 10]]
    degree = [3, 3, 3, 3]
    period = [False, True, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel': [0, 2, 1]}

    grid, constants, _ = setupCylindricalGrid(
        npts=npts, layout='v_parallel', vMax=10, vMin=0)

    nprocs = grid.getLayout(grid.currentLayout).nprocs[:2]

    remapper = getLayoutHandler(comm, layout_poisson, nprocs, eta_grid[:3])

    rho = Grid(eta_grid[:3], bsplines[:3], remapper, 'v_parallel', comm)

    for i, _ in grid.getCoords(0):  # r
        for j, _ in grid.getCoords(1):  # z
            for k, _ in grid.getCoords(2):  # q
                vec = grid.get1DSlice(i, j, k)
                for l, v in grid.getCoords(3):
                    vec[l] = np.cos(v)

    df = DensityFinder(3, grid.getSpline(3), grid.eta_grid, constants)

    df.getRho(grid, rho)

    vals = np.sin(10)

    err = rho._f - vals
    assert (np.abs(err) < tol).all()


@pytest.mark.parallel
@pytest.mark.parametrize("npts_v,tol", [(10, 0.005), (20, 0.0004), (30, 4e-5)])
def test_DensityFinder_cos_RhoPerturbed(npts_v, tol):
    """
    TODO
    """
    npts = [30, 4, 4, npts_v]
    domain = [[1, 5], [0, 2*pi], [0, 1], [0, 10]]
    degree = [3, 3, 3, 3]
    period = [False, True, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel': [0, 2, 1]}

    grid, constants, _ = setupCylindricalGrid(
        npts=npts, layout='v_parallel', vMax=10, vMin=0)

    nprocs = grid.getLayout(grid.currentLayout).nprocs[:2]

    remapper = getLayoutHandler(comm, layout_poisson, nprocs, eta_grid[:3])

    rho = Grid(eta_grid[:3], bsplines[:3], remapper, 'v_parallel', comm)

    for i, r in grid.getCoords(0):
        for j, _ in grid.getCoords(1):  # z
            for k, _ in grid.getCoords(2):  # q
                vec = grid.get1DSlice(i, j, k)
                for l, v in grid.getCoords(3):
                    vec[l] = fEq(r, v, constants.CN0, constants.kN0, constants.deltaRN0,
                                 constants.rp, constants.CTi, constants.kTi, constants.deltaRTi) \
                        + np.cos(v)

    df = DensityFinder(3, grid.getSpline(3), grid.eta_grid, constants)

    df.getPerturbedRho(grid, rho)

    vals = np.sin(10)

    err = rho._f - vals
    assert (np.abs(err) < tol).all()


@pytest.mark.serial
def test_BasicPoissonEquation_Dirichlet_r_function(param_poisson_dirichlet):
    """
    TODO
    """
    deg, npt, eps = param_poisson_dirichlet
    npts = [npt, 8, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0],
                      npts[1], ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in phi_exact.getCoords(0):  # q
        plane = phi_exact.get2DSlice(i)
        plane[:] = -0.5*r**2+0.5 * \
            (domain[0][0]+domain[0][1])*r-domain[0][0]*domain[0][1]*0.5

    ps.solveEquationForFunction(phi, lambda x: np.ones_like(x))

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_BasicPoissonEquation_Dirichlet_r_discrete(param_poisson_dirichlet):
    """
    TODO
    """
    deg, npt, eps = param_poisson_dirichlet
    npts = [npt, 8, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0],
                      npts[1], ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in rho.getCoords(0):  # q
        plane = rho.get2DSlice(i)
        plane[:] = 1
        plane = phi_exact.get2DSlice(i)
        plane[:] = -0.5*r**2+0.5 * \
            (domain[0][0]+domain[0][1])*r-domain[0][0]*domain[0][1]*0.5

    ps.solveEquation(phi, rho)

    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_BasicPoissonEquation_lNeumann_discrete(param_poisson_neumann):
    """
    TODO
    """
    deg, npt, eps = param_poisson_neumann
    npts = [npt, 8, 4]
    domain = [[1, 9], [0, 2*pi], [0, 1]]
    degree = [deg, 2, 2]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    mVals = np.fft.fftfreq(eta_grid[1].size, 1/eta_grid[1].size)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1],
                      lNeumannIdx=mVals, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in rho.getCoords(0):  # q
        plane = rho.get2DSlice(i)
        plane[:] = 1
        plane = phi_exact.get2DSlice(i)
        plane[:] = -0.5*r**2+domain[0][0]*r + \
            domain[0][1]**2*0.5-domain[0][0]*domain[0][1]

    ps.solveEquation(phi, rho)

    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_BasicPoissonEquation_lNeumann_function(param_poisson_neumann):
    """
    TODO
    """
    deg, npt, eps = param_poisson_neumann
    npts = [npt, 8, 4]
    domain = [[1, 9], [0, 2*pi], [0, 1]]
    degree = [deg, 2, 2]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    mVals = np.fft.fftfreq(eta_grid[1].size, 1/eta_grid[1].size)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1],
                      lNeumannIdx=mVals, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in phi_exact.getCoords(0):  # q
        plane = phi_exact.get2DSlice(i)
        plane[:] = -0.5*r**2+domain[0][0]*r + \
            domain[0][1]**2*0.5-domain[0][0]*domain[0][1]

    ps.solveEquationForFunction(phi, np.ones_like)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_BasicPoissonEquation_rNeumann_discrete(param_poisson_neumann):
    """
    TODO
    """
    deg, npt, eps = param_poisson_neumann
    npts = [npt, 8, 4]
    domain = [[1, 9], [0, 2*pi], [0, 1]]
    degree = [deg, 2, 2]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    mVals = np.fft.fftfreq(eta_grid[1].size, 1/eta_grid[1].size)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], uNeumannIdx=mVals,
                      drFactor=lambda r: 0, rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in rho.getCoords(0):  # q
        plane = rho.get2DSlice(i)
        plane[:] = 1
        plane = phi_exact.get2DSlice(i)
        plane[:] = -0.5*r**2+domain[0][1]*r + \
            domain[0][0]**2*0.5-domain[0][0]*domain[0][1]

    ps.solveEquation(phi, rho)

    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_BasicPoissonEquation_rNeumann_function(param_poisson_neumann):
    """
    TODO
    """
    deg, npt, eps = param_poisson_neumann
    npts = [npt, 8, 4]
    domain = [[1, 9], [0, 2*pi], [0, 1]]
    degree = [deg, 2, 2]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    mVals = np.fft.fftfreq(eta_grid[1].size, 1/eta_grid[1].size)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], uNeumannIdx=mVals,
                      drFactor=lambda r: 0, rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in phi_exact.getCoords(0):  # q
        plane = phi_exact.get2DSlice(i)
        plane[:] = -0.5*r**2+domain[0][1]*r + \
            domain[0][0]**2*0.5-domain[0][0]*domain[0][1]

    ps.solveEquationForFunction(phi, lambda x: 1)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_PoissonEquation_Dirichlet(param_poisson_dirichlet):
    """
    TODO
    """
    deg, npt, eps = param_poisson_dirichlet
    npts = [npt, 8, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    grid, constants, _ = setupCylindricalGrid(
        npts=[*npts, 4], layout='v_parallel')

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1])
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'v_parallel', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'v_parallel', comm, dtype=np.complex128)

    q = eta_grid[1]

    for i, r in rho.getCoords(0):
        plane = rho.get2DSlice(i)
        plane[:] = 1
        plane = phi_exact.get2DSlice(i)
        plane[:] = (-0.5*r**2+0.5*(domain[0][0]+domain[0][1])
                    * r-domain[0][0]*domain[0][1]*0.5)

    r = eta_grid[0]

    DensityFinder(3, grid.getSpline(3), grid.eta_grid, constants)

    ps.getModes(rho)

    rho.setLayout('mode_solve')

    ps.solveEquation(phi, rho)

    phi.setLayout('v_parallel')
    ps.findPotential(phi)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_grad_discrete(param_grad):
    """
    TODO
    """
    deg, npt, eps = param_grad
    npts = [npt, 32, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 2*pi/(domain[0][1]-domain[0][0])

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 0,
                      drFactor=lambda r: 1, rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in rho.getCoords(0):  # q
        plane = rho.get2DSlice(i)
        plane[:] = a*np.cos(a*(r-domain[0][0]))
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(a*(r-domain[0][0]))

    ps.solveEquation(phi, rho)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_grad_function(param_grad):
    """
    TODO
    """
    deg, npt, eps = param_grad
    npts = [npt, 32, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 2*pi/(domain[0][1]-domain[0][0])

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 0,
                      drFactor=lambda r: 1, rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)

    r = eta_grid[0]

    for i, _ in phi_exact.getCoords(0):  # q
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(a*(r-domain[0][0]))

    ps.solveEquationForFunction(phi, lambda r: a*np.cos(a*(r-domain[0][0])))

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_grad_r_discrete(param_grad_r):
    """
    TODO
    """
    deg, npt, eps = param_grad_r
    npts = [npt, 32, 4]
    domain = [[1, 8], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 2*pi/(domain[0][1]-domain[0][0])
    r = eta_grid[0]

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 0,
                      drFactor=lambda r: r, rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)

    for i, _ in rho.getCoords(0):  # q
        plane = rho.get2DSlice(i)
        plane[:] = a*np.cos(a*(r-domain[0][0]))*r
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(a*(r-domain[0][0]))

    q = eta_grid[1]
    ps.solveEquation(phi, rho)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_grad_r_function(param_grad_r):
    """
    TODO
    """
    deg, npt, eps = param_grad_r
    npts = [npt, 32, 4]
    domain = [[1, 8], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 2*pi/(domain[0][1]-domain[0][0])
    r = eta_grid[0]

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 0,
                      drFactor=lambda r: r, rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)

    for i, _ in phi_exact.getCoords(0):  # q
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(a*(r-domain[0][0]))

    ps.solveEquationForFunction(phi, lambda r: a*np.cos(a*(r-domain[0][0]))*r)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.parallel
@pytest.mark.parametrize("deg,npt,eps", [(1, 32, 0.07), (2, 32, 0.05), (3, 32, 0.04),
                                         (4, 40, 0.03), (5, 64, 0.02)])
def test_grad_withFFT(deg, npt, eps):
    """
    TODO
    """
    npts = [npt, 32, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 2*pi/(domain[0][1]-domain[0][0])

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 0,
                      drFactor=lambda r: 1, rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'v_parallel', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'v_parallel', comm, dtype=np.complex128)

    for i, r in rho.getCoords(0):
        plane = rho.get2DSlice(i)
        plane[:] = a*np.cos(a*(r-domain[0][0]))
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(a*(r-domain[0][0]))

    ps.getModes(rho)

    rho.setLayout('mode_solve')

    ps.solveEquation(phi, rho)

    phi.setLayout('v_parallel')
    ps.findPotential(phi)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_Sin_r_Sin_theta(param_sin_sin):
    """
    TODO
    """
    deg, npt, eps = param_sin_sin
    npts = [npt, 64, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 2*pi/(domain[0][1]-domain[0][0])

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 1,
                      drFactor=lambda r: 1, rFactor=lambda r: 0, ddThetaFactor=lambda r: -a*a)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'v_parallel', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'v_parallel', comm, dtype=np.complex128)

    q = eta_grid[1]

    for i, r in rho.getCoords(0):
        plane = rho.get2DSlice(i)
        plane[:] = a*np.cos(a*(r-domain[0][0]))*np.sin(q)
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(a*(r-domain[0][0]))*np.sin(q)

    r = eta_grid[0]

    ps.getModes(rho)

    rho.setLayout('mode_solve')

    ps.solveEquation(phi, rho)

    phi.setLayout('v_parallel')
    ps.findPotential(phi)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_ddTheta(param_fft):
    """
    TODO
    """
    deg, npt = param_fft
    eps = 1e-12
    npts = [8, npt, 5]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    mVals = np.fft.fftfreq(eta_grid[1].size, 1/eta_grid[1].size)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 0, drFactor=lambda r: 0,
                      rFactor=lambda r: 1, ddThetaFactor=lambda r: -1, lNeumannIdx=mVals, uNeumannIdx=mVals)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'v_parallel', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'v_parallel', comm, dtype=np.complex128)

    q = eta_grid[1]

    for i, r in rho.getCoords(0):
        plane = rho.get2DSlice(i)
        plane[:] = 2*np.sin(q)
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(q)

    r = eta_grid[0]

    ps.getModes(rho)

    rho.setLayout('mode_solve')

    ps.solveEquation(phi, rho)

    phi.setLayout('v_parallel')

    ps.findPotential(phi)

    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_phi(param_fft):
    """
    TODO
    """
    deg, npt = param_fft
    eps = 1e-12
    npts = [256, npt, 4]
    domain = [[1, 5], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 2*pi/(domain[0][1]-domain[0][0])

    mVals = np.fft.fftfreq(eta_grid[1].size, 1/eta_grid[1].size)

    ps = DiffEqSolver(2*deg, bsplines[0], npts[0], npts[1], ddrFactor=lambda r: 0,
                      drFactor=lambda r: 0, rFactor=lambda r: 1, ddThetaFactor=lambda r: 0,
                      lNeumannIdx=mVals, uNeumannIdx=mVals)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'v_parallel', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'v_parallel', comm, dtype=np.complex128)

    q = eta_grid[1]

    for i, _ in rho.getCoords(0):  # r
        plane = rho.get2DSlice(i)
        plane[:] = np.sin(q)
        plane = phi_exact.get2DSlice(i)
        plane[:] = np.sin(q)

    ps.getModes(rho)

    rho.setLayout('mode_solve')

    ps.solveEquation(phi, rho)

    phi.setLayout('v_parallel')
    ps.findPotential(phi)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < eps).all()


@pytest.mark.serial
def test_quasiNeutrality():
    """
    TODO
    """
    npts = [256, 32, 4]

    constants = Constants()

    domain = [[constants.rMin, constants.rMax], [0, 2*pi], [0, 1]]
    degree = [3, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    a = 1.5*pi/(domain[0][1]-domain[0][0])

    mVals = np.fft.fftfreq(eta_grid[1].size, 1/eta_grid[1].size)

    r = eta_grid[0]

    ps = DiffEqSolver(6, bsplines[0], npts[0], npts[1], lNeumannIdx=mVals,
                      ddrFactor=lambda r: -1,
                      drFactor=lambda r: -(1/r - constants.kN0 * (1 - np.tanh((r - constants.rp) /
                                                                              constants.deltaRN0)**2)),
                      rFactor=lambda r: 1 /
                      Te(r, constants.CTe, constants.kTe,
                         constants.deltaRTe, constants.rp),
                      ddThetaFactor=lambda r: -1/r**2)
    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'v_parallel', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'v_parallel', comm, dtype=np.complex128)

    q = eta_grid[1]

    for i, r in rho.getCoords(0):
        rArg = a*(r-domain[0][0])
        plane = rho.get2DSlice(i)

        plane[:] = -12 * np.cos(rArg)**2 * np.sin(rArg)**2 * a * a * np.sin(q)**3 \
            + 4 * np.cos(rArg)**4 * a * a * np.sin(q)**3 \
            + (1 / r - constants.kN0 * (1 - np.tanh((r - constants.rp) / constants.deltaRN0)**2)) \
            * 4 * np.cos(rArg)**3 * np.sin(rArg) * a * np.sin(q)**3 \
            + np.cos(rArg)**4 * np.sin(q)**3 / Te(r, constants.CTe, constants.kTe, constants.deltaRTe, constants.rp) \
            - 6 * np.cos(rArg)**4 * np.sin(q) * np.cos(q)**2 / r**2 \
            + 3 * np.cos(rArg)**4 * np.sin(q)**3 / r**2

        plane = phi_exact.get2DSlice(i)
        plane[:] = np.cos(rArg)**4 * np.sin(q)**3

    r = eta_grid[0]

    ps.getModes(rho)

    rho.setLayout('mode_solve')

    ps.solveEquation(phi, rho)

    phi.setLayout('v_parallel')
    ps.findPotential(phi)

    # ~ print(np.max(np.abs(phi._f-phi_exact._f)))
    assert (np.abs(phi._f-phi_exact._f) < 0.1).all()


@pytest.mark.parallel
def test_QNSolver():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()

    npts = [32, 64, 32]
    nptsGrid = [*npts, 16]

    n1 = min(npts[0], npts[1])
    n2 = 2

    nprocs = compute_2d_process_grid(nptsGrid, mpi_size)

    # Create dictionary describing layouts
    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    layout_advection = {'dphi': [0, 1, 2],
                        'poloidal': [2, 1, 0],
                        'r_distrib': [0, 2, 1]}

    nproc = nprocs[0]

    grid, constants, _ = setupCylindricalGrid(
        npts=nptsGrid, layout='v_parallel')

    remapper = LayoutSwapper(comm, [layout_poisson, layout_advection], [
                             nprocs, nproc], grid.eta_grid[:3], 'v_parallel')

    rho = Grid(grid.eta_grid[:3], grid.getSpline(
        slice(0, 3)), remapper, 'v_parallel', comm, dtype=np.complex128)
    phi = Grid(grid.eta_grid[:3], grid.getSpline(
        slice(0, 3)), remapper, 'mode_solve', comm, dtype=np.complex128)

    df = DensityFinder(3, grid.getSpline(3), grid.eta_grid, constants)

    df.getPerturbedRho(grid, rho)

    qnSolver = QuasiNeutralitySolver(
        grid.eta_grid, 6, rho.getSpline(0), constants, chi=0)

    qnSolver.getModes(rho)

    rho.setLayout('mode_solve')

    qnSolver.solveEquation(phi, rho)

    phi.setLayout('v_parallel')

    qnSolver.findPotential(phi)


@pytest.mark.parallel
def test_Equilibrium():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()

    npts = [32, 64, 32]
    nptsGrid = [*npts, 32]

    n1 = min(npts[0], npts[1])
    n2 = 2

    nprocs = compute_2d_process_grid(nptsGrid, mpi_size)

    # Create dictionary describing layouts
    layout_poisson = {'mode_solve': [1, 2, 0],
                      'v_parallel': [0, 2, 1]}
    layout_advection = {'dphi': [0, 1, 2],
                        'poloidal': [2, 1, 0],
                        'r_distrib': [0, 2, 1]}

    nproc = nprocs[0]

    grid, constants, _ = setupCylindricalGrid(
        npts=nptsGrid, layout='v_parallel', eps=0.0)

    remapper = LayoutSwapper(comm, [layout_poisson, layout_advection], [
                             nprocs, nproc], grid.eta_grid[:3], 'v_parallel')

    rho = Grid(grid.eta_grid[:3], grid.getSpline(
        slice(0, 3)), remapper, 'v_parallel', comm, dtype=np.complex128)
    phi = Grid(grid.eta_grid[:3], grid.getSpline(
        slice(0, 3)), remapper, 'mode_solve', comm, dtype=np.complex128)

    df = DensityFinder(3, grid.getSpline(3), grid.eta_grid, constants)

    df.getPerturbedRho(grid, rho)

    # print(rho._f)

    assert np.max(rho._f) < 1e-8

    qnSolver = QuasiNeutralitySolver(
        grid.eta_grid, 6, rho.getSpline(0), constants, chi=0)

    qnSolver.getModes(rho)

    rho.setLayout('mode_solve')

    qnSolver.solveEquation(phi, rho)

    phi.setLayout('v_parallel')

    qnSolver.findPotential(phi)

    assert np.max(np.real(phi._f)) < 1e-8


@pytest.mark.serial
@pytest.mark.parametrize("deg", [2, 3, 4, 5])
def test_BasicPoissonEquation_exact(deg):
    """
    TODO
    """
    npt = 16

    npts = [npt, 8, 4]
    # ~ domain = [[1,15],[0,2*pi],[0,1]]
    domain = [[1, 3], [0, 2*pi], [0, 1]]
    degree = [deg, 3, 3]
    period = [False, True, False]
    comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'mode_solve': [1, 2, 0]}
    remapper = getLayoutHandler(comm, layout_poisson, [
                                comm.Get_size()], eta_grid)

    ps = DiffEqSolver(2*degree[0]+1, bsplines[0], npts[0], npts[1], drFactor=lambda r: 0,
                      rFactor=lambda r: 0, ddThetaFactor=lambda r: 0)

    phi = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)
    phi_exact = Grid(eta_grid, bsplines, remapper,
                     'mode_solve', comm, dtype=np.complex128)
    rho = Grid(eta_grid, bsplines, remapper,
               'mode_solve', comm, dtype=np.complex128)

    x = eta_grid[0]

    coeffs = np.array([*np.random.randint(-9, 10, size=deg-2), 1])
    C = np.sum(coeffs*np.power(domain[0][0], np.arange(2, deg+1)))
    D = np.sum(coeffs*np.power(domain[0][1], np.arange(2, deg+1)))
    coeff2 = (D-C)/(domain[0][0]-domain[0][1])
    coeff1 = -D-coeff2*domain[0][1]
    coeffs = np.array([coeff1, coeff2, *coeffs])

    for i, q in rho.getCoords(0):
        plane = rho.get2DSlice(i)
        plane[:] = np.sum(coeffs[2:]*np.power(np.atleast_2d(x).T,
                          np.arange(deg-1))*np.arange(2, deg+1)*np.arange(1, deg), axis=1)
        plane = phi_exact.get2DSlice(i)
        plane[:] = -np.sum(coeffs*np.power(np.atleast_2d(x).T,
                           np.arange(deg+1)), axis=1)

    ps.solveEquation(phi, rho)

    x = eta_grid[0]

    err = (phi._f-phi_exact._f)[0, 0]
    l2 = np.sqrt(trapz(np.real(err*err.conj()), x))
    lInf = np.max(np.abs(phi._f-phi_exact._f))

    assert l2 < 1e-10
    assert lInf < 1e-10


@pytest.mark.serial
def test_WellPosedness():
    """
    TODO
    """
    npt = 16

    npts = [npt, 8, 4]
    # ~ domain = [[1,15],[0,2*pi],[0,1]]
    domain = [[1, 3], [0, 2*pi], [0, 1]]
    degree = [3, 3, 3]
    period = [False, True, False]
    # comm = MPI.COMM_WORLD

    # Compute breakpoints, knots, spline space and grid points
    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, d, p)
             for (b, d, p) in zip(breaks, degree, period)]
    bsplines = [spl.BSplines(k, d, p, True)
                for (k, d, p) in zip(knots, degree, period)]
    # eta_grid = [bspl.greville for bspl in bsplines]

    # layout_poisson = {'mode_solve': [1, 2, 0]}
    # remapper = getLayoutHandler(comm, layout_poisson, [
    #                             comm.Get_size()], eta_grid)

    with pytest.raises(ValueError):
        DiffEqSolver(2*degree[0]+1, bsplines[0], npts[0], npts[1], drFactor=lambda r: 0,
                     rFactor=lambda r: 0, ddThetaFactor=lambda r: 0,
                     lNeumannIdx=[0], uNeumannIdx=[0])

    with pytest.raises(ValueError):
        DiffEqSolver(2*degree[0]+1, bsplines[0], npts[0], npts[1], drFactor=lambda r: 0,
                     rFactor=np.zeros_like, ddThetaFactor=lambda r: 0,
                     lNeumannIdx=[0], uNeumannIdx=[0])
