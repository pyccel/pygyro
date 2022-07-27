from mpi4py import MPI
import numpy as np
import pytest

from .norms import l2, l1, nParticles
from .. import splines as spl
from ..model.process_grid import compute_2d_process_grid_from_max
from ..model.grid import Grid
from ..model.layout import LayoutSwapper
from ..initialisation.setups import setupCylindricalGrid


def args_norm_phi():
    """
    TODO
    """
    for layout in ['v_parallel_2d', 'mode_solve', 'v_parallel_1d', 'poloidal']:
        for _ in range(2):
            R0 = np.random.rand()*1000
            for _ in range(2):
                rMin = np.random.rand()
                for _ in range(2):
                    rMax = np.random.randint(10, 200)/10
                    yield (layout, R0, rMin, rMax)


@pytest.mark.parallel
@pytest.mark.parametrize("layout,R0,rMin,rMax", args_norm_phi())
def test_l2Norm_is_volume(layout, R0, rMin, rMax):
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    layout = comm.bcast(layout)
    R0 = comm.bcast(R0)
    rMin = comm.bcast(rMin)
    rMax = comm.bcast(rMax)

    npts = [10, 11, 12]

    domain = [[rMin, rMax], [0, 2*np.pi], [0, R0*2*np.pi]]
    periodic = [False, True, True]
    nkts = [n+1 for n in npts]
    breaks = [np.linspace(*lims, num=num) for (lims, num) in zip(domain, nkts)]
    knots = [spl.make_knots(b, 3, p) for (b, p) in zip(breaks, periodic)]
    bsplines = [spl.BSplines(k, 3, p) for (k, p) in zip(knots, periodic)]
    eta_grid = [bspl.greville for bspl in bsplines]

    layout_poisson = {'v_parallel_2d': [0, 2, 1],
                      'mode_solve': [1, 2, 0]}
    layout_vpar = {'v_parallel_1d': [0, 2, 1]}
    layout_poloidal = {'poloidal': [2, 1, 0]}

    nprocs = compute_2d_process_grid_from_max(
        min(npts[0], npts[1]), npts[2], size)

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grid,
                                layout)

    theLayout = remapperPhi.getLayout(layout)

    norm = l2(eta_grid, theLayout)

    phi = Grid(eta_grid, bsplines, remapperPhi, layout, comm)
    phi._f[:] = 1

    l2Val = norm.l2NormSquared(phi)
    if (layout in layout_poisson):
        l2Result = comm.reduce(l2Val, op=MPI.SUM, root=0)
    else:
        comm = remapperPhi._managers[remapperPhi._handlers[layout]
                                     ].communicators[0]
        l2Result = comm.reduce(l2Val, op=MPI.SUM, root=0)

    if (rank == 0):
        assert(abs(l2Result-((rMax**2-rMin**2)*np.pi*np.pi*R0*2)) < 5e-7)


def args_norm_grid():
    """
    TODO
    """
    for layout in ['poloidal', 'v_parallel', 'flux_surface', 'poloidal']:
        for _ in range(2):
            R0 = np.random.rand()*1000
            for _ in range(2):
                rMin = np.random.rand()
                for _ in range(2):
                    rMax = np.random.randint(10, 200)/10
                    for _ in range(2):
                        vMax = np.random.randint(1, 100)/10
                        yield (layout, R0, rMin, rMax, vMax)


@pytest.mark.parallel
@pytest.mark.parametrize("layout,R0,rMin,rMax,vMax", args_norm_grid())
def test_l1Norm_grid_is_volume(layout, R0, rMin, rMax, vMax):
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    layout = comm.bcast(layout)
    R0 = comm.bcast(R0)
    rMin = comm.bcast(rMin)
    rMax = comm.bcast(rMax)
    vMax = comm.bcast(vMax)

    npts = [10, 11, 12, 13]
    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout=layout,
                                      rMin=rMin,
                                      rMax=rMax,
                                      vMax=vMax,
                                      vMin=-vMax,
                                      zMax=2*np.pi*R0,
                                      R0=R0)

    theLayout = grid.getLayout(layout)

    norm = l1(grid.eta_grid, theLayout)

    grid._f[:] = 1

    l1Val = norm.l1Norm(grid)
    l1Result = comm.reduce(l1Val, op=MPI.SUM, root=0)

    if (rank == 0):
        # ~ print(l1Result,((rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax))
        assert(abs(l1Result-((rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax)) < 5e-7)


@pytest.mark.parallel
@pytest.mark.parametrize("layout,R0,rMin,rMax,vMax", args_norm_grid())
def test_l2Norm_grid_is_volume(layout, R0, rMin, rMax, vMax):
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    layout = comm.bcast(layout)
    R0 = comm.bcast(R0)
    rMin = comm.bcast(rMin)
    rMax = comm.bcast(rMax)
    vMax = comm.bcast(vMax)

    npts = [10, 11, 12, 13]
    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout=layout,
                                      rMin=rMin,
                                      rMax=rMax,
                                      vMax=vMax,
                                      vMin=-vMax,
                                      zMax=2*np.pi*R0,
                                      R0=R0)

    theLayout = grid.getLayout(layout)

    norm = l2(grid.eta_grid, theLayout)

    grid._f[:] = 1

    l2Val = norm.l2NormSquared(grid)
    l2Result = comm.reduce(l2Val, op=MPI.SUM, root=0)

    if (rank == 0):
        # ~ print(l2Result,((rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax))
        assert(abs(l2Result-((rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax)) < 5e-7)


@pytest.mark.parallel
@pytest.mark.parametrize("layout,R0,rMin,rMax,vMax", args_norm_grid())
def test_l2Norm_grid_is_v(layout, R0, rMin, rMax, vMax):
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    layout = comm.bcast(layout)
    R0 = comm.bcast(R0)
    rMin = comm.bcast(rMin)
    rMax = comm.bcast(rMax)
    vMax = comm.bcast(vMax)

    npts = [10, 11, 12, 13]
    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout=layout,
                                      rMin=rMin,
                                      rMax=rMax,
                                      vMax=vMax,
                                      vMin=-vMax,
                                      zMax=2*np.pi*R0,
                                      R0=R0)

    theLayout = grid.getLayout(layout)

    norm = l2(grid.eta_grid, theLayout)

    idx_v = theLayout.inv_dims_order[3]
    v_range = grid.getGlobalIdxVals(idx_v)

    reshaper = [slice(v_range.start, v_range.stop)]
    reshaper.extend([None]*(3-idx_v))

    grid._f[:] = grid.eta_grid[3][tuple(reshaper)]

    l2Val = norm.l2NormSquared(grid)
    l2Result = comm.reduce(l2Val, op=MPI.SUM, root=0)

    if (rank == 0):
        exactResult = (rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax**3/3
        # ~ print(abs(l2Result-exactResult)/exactResult)
        assert(abs(l2Result-exactResult)/exactResult < 2e-2)


@pytest.mark.parallel
@pytest.mark.parametrize("layout,R0,rMin,rMax,vMax", args_norm_grid())
def test_NParticles_is_volume(layout, R0, rMin, rMax, vMax):
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    layout = comm.bcast(layout)
    R0 = comm.bcast(R0)
    rMin = comm.bcast(rMin)
    rMax = comm.bcast(rMax)
    vMax = comm.bcast(vMax)

    npts = [10, 11, 12, 13]
    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout=layout,
                                      rMin=rMin,
                                      rMax=rMax,
                                      vMax=vMax,
                                      vMin=-vMax,
                                      zMax=2*np.pi*R0,
                                      R0=R0)

    theLayout = grid.getLayout(layout)

    norm = nParticles(grid.eta_grid, theLayout)

    grid._f[:] = 1

    l2Val = norm.getN(grid)
    l2Result = comm.reduce(l2Val, op=MPI.SUM, root=0)

    if (rank == 0):
        # ~ print(l2Result,((rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax))
        assert(abs(l2Result-((rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax)) < 5e-7)


@pytest.mark.parallel
@pytest.mark.parametrize("layout,R0,rMin,rMax,vMax", args_norm_grid())
def test_NParticles_grid_is_v(layout, R0, rMin, rMax, vMax):
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    layout = comm.bcast(layout)
    R0 = comm.bcast(R0)
    rMin = comm.bcast(rMin)
    rMax = comm.bcast(rMax)
    vMax = comm.bcast(vMax)

    npts = [10, 11, 12, 13]
    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout=layout,
                                      rMin=rMin,
                                      rMax=rMax,
                                      vMax=vMax,
                                      vMin=-vMax,
                                      zMax=2*np.pi*R0,
                                      R0=R0)

    theLayout = grid.getLayout(layout)

    norm = nParticles(grid.eta_grid, theLayout)

    idx_v = theLayout.inv_dims_order[3]
    v_range = grid.getGlobalIdxVals(idx_v)

    reshaper = [slice(v_range.start, v_range.stop)]
    reshaper.extend([None]*(3-idx_v))

    grid._f[:] = grid.eta_grid[3][tuple(reshaper)]

    l2Val = norm.getN(grid)
    l2Result = comm.reduce(l2Val, op=MPI.SUM, root=0)

    if (rank == 0):
        # ~ print(l2Result,((rMax**2-rMin**2)*np.pi*np.pi*R0*4*vMax))
        assert(abs(l2Result) < 5e-7)
