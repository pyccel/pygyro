from mpi4py import MPI
import numpy as np
import pytest
from math import pi
import os

from .grid import Grid
from .layout import getLayoutHandler, LayoutSwapper
from .process_grid import compute_2d_process_grid, compute_2d_process_grid_from_max


def define_f(grid, t=0):
    """
    TODO
    """
    [nEta1, nEta2, nEta3, nEta4] = grid.nGlobalCoords

    for i, _ in grid.getCoords(0):
        for j, _ in grid.getCoords(1):
            for k, _ in grid.getCoords(2):
                Slice = grid.get1DSlice(i, j, k)
                for l, a in enumerate(Slice):
                    [I, J, K, L] = grid.getGlobalIndices(i, j, k, l)
                    Slice[l] = I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L+t


def define_phi(grid):
    """
    TODO
    """
    [nEta1, nEta2, nEta3] = grid.nGlobalCoords

    for i, _ in grid.getCoords(0):
        for j, _ in grid.getCoords(1):
            Slice = grid.get1DSlice(i, j)
            for k, _ in enumerate(Slice):
                [I, J, K] = grid.getGlobalIndices(i, j, k)
                Slice[k] = I*nEta3*nEta2+J*nEta3+K


def compare_f(grid, t=0):
    """
    TODO
    """
    [nEta1, nEta2, nEta3, nEta4] = grid.nGlobalCoords

    for i, _ in grid.getCoords(0):
        for j, _ in grid.getCoords(1):
            for k, _ in grid.getCoords(2):
                Slice = grid.get1DSlice(i, j, k)
                for l, a in enumerate(Slice):
                    [I, J, K, L] = grid.getGlobalIndices(i, j, k, l)
                    assert(a == I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L+t)


def compare_phi(grid):
    """
    TODO
    """
    [nEta1, nEta2, nEta3] = grid.nGlobalCoords

    for i, _ in grid.getCoords(0):
        for j, _ in grid.getCoords(1):
            Slice = grid.get1DSlice(i, j)
            for k, a in enumerate(Slice):
                [I, J, K] = grid.getGlobalIndices(i, j, k)
                assert(a == I*nEta3*nEta2+J*nEta3+K)


@pytest.mark.serial
def test_Grid_serial():
    """
    TODO
    """
    eta_grids = [np.linspace(0, 1, 10),
                 np.linspace(0, 6.28318531, 10),
                 np.linspace(0, 10, 10),
                 np.linspace(0, 10, 10)]
    comm = MPI.COMM_WORLD

    nprocs = compute_2d_process_grid([10, 10, 10, 10], comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    Grid(eta_grids, [], manager, 'flux_surface')


@pytest.mark.parallel
def test_Grid_parallel():
    """
    TODO
    """
    eta_grids = [np.linspace(0, 1, 10),
                 np.linspace(0, 6.28318531, 10),
                 np.linspace(0, 10, 10),
                 np.linspace(0, 10, 10)]
    comm = MPI.COMM_WORLD

    nprocs = compute_2d_process_grid([10, 10, 10, 10], comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    Grid(eta_grids, [], manager, 'flux_surface')


@pytest.mark.parallel
def test_Grid_max():
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD

    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, 'flux_surface')
    define_f(grid)
    maxVal = grid.getMax()
    assert(maxVal == np.max(grid._f))


@pytest.mark.parallel
def test_Grid_min():
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD

    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, 'flux_surface')
    define_f(grid)
    minVal = grid.getMin()
    assert(minVal == np.min(grid._f))


@pytest.mark.parallel
def test_Grid_save_restore():
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD

    nprocs = compute_2d_process_grid([10, 10, 10, 10], comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, 'flux_surface',
                allocateSaveMemory=True)
    define_f(grid, 0)
    grid.saveGridValues()
    define_f(grid, 100)
    compare_f(grid, 100)
    grid.restoreGridValues()
    compare_f(grid, 0)


@pytest.mark.parallel
def test_Grid_save_restore_once():
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD

    nprocs = compute_2d_process_grid([10, 10, 10, 10], comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, 'flux_surface',
                allocateSaveMemory=True)
    grid.saveGridValues()
    with pytest.raises(AssertionError):
        grid.saveGridValues()
    grid.restoreGridValues()
    grid.saveGridValues()
    grid.freeGridSave()
    grid.saveGridValues()


@pytest.mark.serial
def test_CoordinateSave():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    npts = [30, 20, 15, 10]

    eta_grid = [np.linspace(0.5, 14.5, npts[0]),
                np.linspace(0, 2*pi, npts[1], endpoint=False),
                np.linspace(0, 50, npts[2]),
                np.linspace(-5, 5, npts[3])]

    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grid)

    grid = Grid(eta_grid, [], manager, 'flux_surface')

    dim_order = [0, 3, 1, 2]
    for j in range(4):
        for i, x in grid.getCoords(j):
            assert(x == eta_grid[dim_order[j]][i])


@pytest.mark.parallel
def test_LayoutSwap():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    npts = [40, 20, 10, 30]
    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    remapper = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], remapper, 'flux_surface')

    define_f(grid)

    grid.setLayout('v_parallel')

    compare_f(grid)


@pytest.mark.parallel
def test_Contiguous():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]

    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, 'flux_surface')

    assert(grid.get2DSlice(0, 0).flags['C_CONTIGUOUS'])
    assert(grid.get1DSlice(0, 0, 0).flags['C_CONTIGUOUS'])

    grid.setLayout('v_parallel')

    assert(grid.get2DSlice(0, 0).flags['C_CONTIGUOUS'])
    assert(grid.get1DSlice(0, 0, 0).flags['C_CONTIGUOUS'])

    grid.setLayout('poloidal')

    assert(grid.get2DSlice(0, 0).flags['C_CONTIGUOUS'])
    assert(grid.get1DSlice(0, 0, 0).flags['C_CONTIGUOUS'])

    grid.setLayout('v_parallel')

    assert(grid.get2DSlice(0, 0).flags['C_CONTIGUOUS'])
    assert(grid.get1DSlice(0, 0, 0).flags['C_CONTIGUOUS'])

    grid.setLayout('flux_surface')

    assert(grid.get2DSlice(0, 0).flags['C_CONTIGUOUS'])
    assert(grid.get1DSlice(0, 0, 0).flags['C_CONTIGUOUS'])


@pytest.mark.parallel
def test_PhiLayoutSwap():
    """
    TODO
    """
    comm = MPI.COMM_WORLD
    npts = [40, 20, 40]

    n1 = min(npts[0], npts[2])
    n2 = min(npts[0], npts[1])
    nprocs = compute_2d_process_grid_from_max(n1, n2, comm.Get_size())

    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2])]

    layout_poisson = {'mode_find': [2, 0, 1],
                      'mode_solve': [2, 1, 0]}
    layout_advection = {'dphi': [0, 1, 2],
                        'poloidal': [2, 1, 0]}

    nproc = max(nprocs)
    if (nproc > n1):
        nproc = min(nprocs)

    remapper = LayoutSwapper(comm, [layout_poisson, layout_advection], [
                             nprocs, nproc], eta_grids, 'mode_find')

    phi = Grid(eta_grids, [], remapper, 'mode_find')

    define_phi(phi)

    phi.setLayout('poloidal')

    compare_phi(phi)


@pytest.mark.parallel
def test_h5py():
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

    grid = Grid(eta_grids, [], remapper, 'flux_surface')

    define_f(grid)

    if (comm.Get_rank() == 0):
        if (not os.path.isdir('testValues')):
            os.mkdir('testValues')

    define_f(grid, 20)
    grid.writeH5Dataset('testValues', 20)
    define_f(grid, 40)
    grid.writeH5Dataset('testValues', 40)
    define_f(grid, 80)
    grid.writeH5Dataset('testValues', 80)
    define_f(grid, 100)
    grid.writeH5Dataset('testValues', 100)
    grid.loadFromFile('testValues')
    compare_f(grid, 100)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!       Test plotting functions !!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


@pytest.mark.parametrize("layout", ['flux_surface', 'v_parallel', 'poloidal'])
@pytest.mark.parallel
def test_Grid_max_plotting(layout):
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, layout)
    define_f(grid)
    maxVal = grid.getMax(0)
    if (rank == 0):
        assert(maxVal == (np.prod(npts)-1))

    r = min(1, comm.Get_size()-1)

    maxVal = grid.getMax(r, 0, 0)
    if (rank == r):
        assert(maxVal == (np.prod(npts[1:])-1))

    r = min(2, comm.Get_size()-1)

    maxVal = grid.getMax(r, [0, 1], [0, 0])
    if (rank == r):
        assert(maxVal == (np.prod(npts[2:])-1))


@pytest.mark.parametrize("layout", ['flux_surface', 'v_parallel', 'poloidal'])
@pytest.mark.parallel
def test_Grid_max_plotting_drawRank(layout):
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if (comm.Get_size() == 1):
        return

    layout_comm = comm.Split(rank == 0, comm.Get_rank())

    mpi_size = layout_comm.Get_size()

    try:
        nprocs = compute_2d_process_grid(npts, mpi_size)
    except RuntimeError:
        return

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}

    # Create layout manager
    if (rank == 0):
        manager = getLayoutHandler(
            layout_comm, layouts, nprocs, [[], [], [], []])
    else:
        manager = getLayoutHandler(layout_comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, layout, comm=comm)
    define_f(grid)
    maxVal = grid.getMax(0)

    if (rank == 0):
        assert(maxVal == (np.prod(npts)-1))

    r = min(1, comm.Get_size()-1)

    maxVal = grid.getMax(r, 0, 0)

    if (rank == r):
        assert(maxVal == (np.prod(npts[1:])-1))

    r = min(2, comm.Get_size()-1)

    maxVal = grid.getMax(r, [0, 1], [0, 0])
    if (rank == r):
        assert(maxVal == (np.prod(npts[2:])-1))


@pytest.mark.parametrize("layout", ['flux_surface', 'v_parallel', 'poloidal'])
@pytest.mark.parallel
def test_Grid_min_plotting(layout):
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, layout)
    define_f(grid)
    minVal = grid.getMin(0)
    if (comm.Get_rank() == 0):
        assert(minVal == 0)

    r = min(1, comm.Get_size()-1)

    minVal = grid.getMin(r, 0, 9)
    if (rank == r):
        assert(minVal == 9*np.prod(npts[1:]))

    r = min(2, comm.Get_size()-1)

    minVal = grid.getMin(r, [0, 1], [9, 9])
    if (rank == r):
        assert(minVal == 9*np.prod(npts[1:])+9*np.prod(npts[2:]))


@pytest.mark.parametrize("layout", ['flux_surface', 'v_parallel', 'poloidal'])
@pytest.mark.parallel
def test_Grid_min_plotting_drawRank(layout):
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    layout_comm = comm.Split(rank == 0, comm.Get_rank())

    mpi_size = layout_comm.Get_size()

    try:
        nprocs = compute_2d_process_grid(npts, mpi_size)
    except RuntimeError:
        return

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}

    if (comm.Get_size() == 1):
        return

    # Create layout manager
    if (rank == 0):
        manager = getLayoutHandler(
            layout_comm, layouts, nprocs, [[], [], [], []])
    else:
        manager = getLayoutHandler(layout_comm, layouts, nprocs, eta_grids)

    grid = Grid(eta_grids, [], manager, layout, comm=comm)
    define_f(grid)

    minVal = grid.getMin(0)
    if (comm.Get_rank() == 0):
        assert(minVal == 0)

    r = min(1, comm.Get_size()-1)

    minVal = grid.getMin(r, 0, 9)
    if (rank == r):
        assert(minVal == 9*np.prod(npts[1:]))

    r = min(2, comm.Get_size()-1)

    minVal = grid.getMin(r, [0, 1], [9, 9])
    if (rank == r):
        assert(minVal == 9*np.prod(npts[1:])+9*np.prod(npts[2:]))


def compare_small_f(f, grid, idx, n_idx, val, layout, starts, mpi_data):
    """
    TODO
    """
    [nEta1, nEta2, nEta3, nEta4] = grid.nGlobalCoords
    runs = [range(len(eta)) for eta in grid.eta_grid]

    idx = layout.inv_dims_order[idx]

    split = np.split(f, starts[1:])

    nprocs = [max([mpi_data[i][j] for i in range(len(mpi_data))]) +
              1 for j in range(len(mpi_data[0]))]

    concatReady = np.ndarray(tuple(nprocs), dtype=object)

    runs[idx] = [val]
    layout_shape = [list(layout.shape) for i in range(len(starts))]
    for l, ranks in enumerate(mpi_data):
        visited = False
        for j, d in enumerate(ranks):
            layout_shape[l][j] = layout.mpi_lengths(j)[d]
            if (j == idx):
                visited = True
                if layout.mpi_starts(idx)[d] <= val and \
                        layout.mpi_starts(idx)[d]+layout.mpi_lengths(idx)[d] > val:
                    layout_shape[l][j] = 1
                else:
                    layout_shape[l][j] = 0
        if (not visited):
            layout_shape[l][idx] = 1
        concatReady[tuple(ranks)] = split[l].reshape(layout_shape[l])

    zone = [range(n) for n in nprocs]
    zone.pop()

    for i in range(len(nprocs)-1, 0, -1):
        toConcat = np.ndarray(tuple(nprocs[:i]), dtype=object)

        coords = [0 for n in zone]
        for d in range(len(zone)):
            for j in range(nprocs[d]):
                toConcat[tuple(coords)] = np.concatenate(
                    concatReady[tuple(coords)].tolist(), axis=i)
                coords[d] += 1
        concatReady = toConcat
        zone.pop()

    toConcat = np.ndarray(tuple(nprocs[:i]), dtype=object)

    coords = 0
    f = np.concatenate(concatReady.tolist(), axis=0)

    for i, x in enumerate(runs[0]):
        for j, y in enumerate(runs[1]):
            for k, z in enumerate(runs[2]):
                for l, a in enumerate(runs[3]):
                    [I, J, K, L] = grid.getGlobalIndices(x, y, z, a)
                    assert(f[i, j, k, l] == I*nEta4*nEta3 *
                           nEta2+J*nEta4*nEta3+K*nEta4+L)


@pytest.mark.parametrize("val", [0, 3, 7])
@pytest.mark.parallel
def test_Grid_block_getter(val):
    """
    TODO
    """
    npts = [10, 10, 10, 10]
    # ~ npts = [3,3,3,3]
    eta_grids = [np.linspace(0, 1, npts[0]),
                 np.linspace(0, 6.28318531, npts[1]),
                 np.linspace(0, 10, npts[2]),
                 np.linspace(0, 10, npts[3])]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nprocs = compute_2d_process_grid(npts, comm.Get_size())

    layouts = {'flux_surface': [0, 3, 1, 2],
               'v_parallel': [0, 2, 1, 3],
               'poloidal': [3, 2, 1, 0]}
    manager = getLayoutHandler(comm, layouts, nprocs, eta_grids)

    for my_layout in ['poloidal', 'v_parallel', 'flux_surface']:

        grid = Grid(eta_grids, [], manager, my_layout)
        define_f(grid)

        for i in range(4):
            if (rank == 0):
                layout, starts, mpi_data, new_f = grid.getBlockFromDict(
                    {i: val}, comm, 0)
                compare_small_f(new_f, grid, i, 4, val,
                                layout, starts, mpi_data)
            else:
                grid.getBlockFromDict({i: val}, comm, 0)
            comm.Barrier()
