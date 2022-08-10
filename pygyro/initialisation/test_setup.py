import os
from ..utilities.savingTools import setupSave
from ..model.process_grid import compute_2d_process_grid, compute_2d_process_grid_from_max
from ..model.layout import getLayoutHandler
from ..model.grid import Grid
from mpi4py import MPI
import numpy as np
from functools import reduce
import pytest

from .constants import Constants
from .setups import setupCylindricalGrid, setupFromFile


def setup_test():
    """
    TODO
    """
    npts = [10, 20, 10, 10]
    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout='flux_surface')

    for (coord, npt) in zip(grid.eta_grid, npts):
        assert (len(coord) == npt)

    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout='poloidal')

    for (coord, npt) in zip(grid.eta_grid, npts):
        assert (len(coord) == npt)

    grid, _, _ = setupCylindricalGrid(npts=npts,
                                      layout='v_parallel')

    for (coord, npt) in zip(grid.eta_grid, npts):
        assert (len(coord) == npt)


@pytest.mark.serial
def test_setup_serial():
    """
    TODO
    """
    setup_test()


@pytest.mark.parallel
def test_setup_parallel():
    """
    TODO
    """
    setup_test()


def define_f(grid):
    """
    TODO
    """
    [_, nEta2, nEta3, nEta4] = grid.nGlobalCoords

    for i, _ in grid.getCoords(0):
        for j, _ in grid.getCoords(1):
            for k, _ in grid.getCoords(2):
                Slice = grid.get1DSlice(i, j, k)
                for l, _ in enumerate(Slice):
                    [I, J, K, L] = grid.getGlobalIndices(i, j, k, l)
                    Slice[l] = I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L


def compare_f(grid, t):
    """
    TODO
    """
    [_, nEta2, nEta3, nEta4] = grid.nGlobalCoords

    for i, _ in grid.getCoords(0):
        for j, _ in grid.getCoords(1):
            for k, _ in grid.getCoords(2):
                Slice = grid.get1DSlice(i, j, k)
                for l, a in enumerate(Slice):
                    [I, J, K, L] = grid.getGlobalIndices(i, j, k, l)
                    assert (a == I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L+t)


@pytest.mark.parallel
def test_setupFromFolder():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    npts = [10, 20, 10, 10]
    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    remapper = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    constants = Constants()
    constants.npts = npts

    grid = Grid(eta_grids, [], remapper, 'flux_surface')

    define_f(grid)

    if (comm.Get_rank() == 0):
        if (not os.path.isdir('testValues')):
            os.mkdir('testValues')

    setupSave(constants, 'testValues')

    grid.writeH5Dataset('testValues', 0)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 20)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 40)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 60)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 80)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 100)

    grid2, _, _ = setupFromFile('testValues')
    compare_f(grid2, 100)


@pytest.mark.parallel
def test_setupFromFolderAtTime():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    npts = [10, 20, 10, 10]
    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    remapper = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    constants = Constants()
    constants.npts = npts

    grid = Grid(eta_grids, [], remapper, 'flux_surface')

    define_f(grid)

    if (comm.Get_rank() == 0):
        if (not os.path.isdir('testValues')):
            os.mkdir('testValues')

    setupSave(constants, 'testValues')

    grid.writeH5Dataset('testValues', 0)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 20)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 40)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 60)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 80)
    grid._f[:] += 20
    grid.writeH5Dataset('testValues', 100)

    grid2, constants, _ = setupFromFile('testValues', timepoint=40)
    compare_f(grid2, 40)
