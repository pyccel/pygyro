import h5py
import numpy as np
from mpi4py import MPI

from .. import splines as spl
from ..initialisation.constants import get_constants
from ..model.process_grid import compute_2d_process_grid
from ..model.grid import Grid
from ..model.layout import LayoutSwapper


def get_phi_slice(foldername, tEnd, z_idx = 0):
    """
    Extract a poloidal slice of the electric potential at time tEnd from the specified folder
    and save the slice into a file named PhiSlice_tEnd.h5 . The data is saved with theta
    in the first dimension, and r in the second dimension

    Parameters
    ----------
    foldername : str
                 The folder containing the simulation
    tEnd       : int
                 The time which should be examined to obtain the slice
    z_idx      : int
                 The index of the slice in the z direction
                 Default : 0
    """

    assert(len(foldername) > 0)

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()

    filename = os.path.join(foldername,"initParams.json")
    constants = get_constants(filename)

    npts = constants.npts

    degree = constants.splineDegrees[:-1]
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

    if z_idx is None:
        z_idx = 0
    else:
        assert z_idx < distribFunc.eta_grid[2].size

    if (z_idx in phi.getGlobalIdxVals(0)):
        filename = "{0}/PhiSlice_{1}.h5".format(foldername, tEnd)
        file = h5py.File(filename, 'w')
        dset = file.create_dataset(
            "dset", [phi.eta_grid[1].size, phi.eta_grid[0].size])
        starts = phi.getLayout(phi.currentLayout).starts
        dset[:] = np.real(phi.get2DSlice(z_idx))
        file.close()

def get_flux_surface_phi_slice(foldername, tEnd):
    """
    Extract a poloidal slice of the electric potential at time tEnd from the specified folder
    and save the slice into a file named PhiFluxSlice_tEnd.h5 . The data is saved with z
    in the first dimension, and theta in the second dimension

    Parameters
    ----------
    foldername : str
                 The folder containing the simulation
    tEnd       : int
                 The time which should be examined to obtain the slice
    r_idx      : int
                 The index of the slice in the r direction
                 Default : nr//2
    """

    assert(len(foldername) > 0)

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()

    filename = os.path.join(foldername, "initParams.json")
    constants = get_constants(filename)

    npts = constants.npts

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

    phi.setLayout('v_parallel_2d')

    if r_idx is None:
        nr = distribFunc.eta_grid[0].size
        r_idx = nr//2
    else:
        assert r_idx < distribFunc.eta_grid[0].size

    if (r_idx in phi.getGlobalIdxVals(0)):
        filename = "{0}/PhiFluxSlice_{1}.h5".format(foldername, tEnd)
        file = h5py.File(filename, 'w')
        dset = file.create_dataset(
            "dset", [phi.eta_grid[2].size, phi.eta_grid[1].size])
        i = r_idx - phi.getLayout(phi.currentLayout).starts[0]
        dset[:] = np.real(phi.get2DSlice(i))
        file.close()
