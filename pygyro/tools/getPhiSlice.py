import h5py
import numpy as np
from mpi4py import MPI
import time

from .. import splines as spl
from ..initialisation import constants
from ..model.process_grid import compute_2d_process_grid
from ..model.grid import Grid
from ..model.layout import LayoutSwapper

setup_time_start = time.clock()

# import cProfile, pstats, io


def get_phi_slice(foldername, tEnd):
    """
    TODO
    """

    assert (len(foldername) > 0)

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()

    filename = "{0}/initParams.h5".format(foldername)
    save_file = h5py.File(filename, 'r', driver='mpio', comm=comm)
    group = save_file['constants']
    for i in group.attrs:
        constants.i = group.attrs[i]
    constants.rp = 0.5*(constants.rMin + constants.rMax)

    npts = save_file.attrs['npts']

    save_file.close()

    degree = [3, 3, 3]
    period = [False, True, True]
    domain = [[constants.rMin, constants.rMax], [
        0, 2*np.pi], [constants.zMin, constants.zMax]]

    nkts = [n+1+d*(int(p)-1) for (n, d, p) in zip(npts, degree, period)]
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

    nprocs = compute_2d_process_grid(npts, mpi_size)

    remapperPhi = LayoutSwapper(comm, [layout_poisson, layout_vpar, layout_poloidal],
                                [nprocs, nprocs[0], nprocs[1]], eta_grids,
                                'v_parallel_2d')

    phi = Grid(eta_grids, bsplines, remapperPhi,
               'v_parallel_2d', comm, dtype=np.complex128)
    phi.loadFromFile(foldername, tEnd, "phi")

    phi.setLayout('poloidal')

    if (0 in phi.getGlobalIdxVals(0)):
        filename = "{0}/PhiSlice_{1}.h5".format(foldername, tEnd)
        file = h5py.File(filename, 'w')
        dset = file.create_dataset(
            "dset", [phi.eta_grid[1].size, phi.eta_grid[0].size])
        dset[:] = np.real(phi.get2DSlice([0]))
        file.close()
